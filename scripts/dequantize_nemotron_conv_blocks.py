#!/usr/bin/env python3
"""Rewrite selected dynamic-quantized Conv blocks back to float Conv."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ORT_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}


@dataclass(frozen=True)
class RewriteSpec:
    first_node_index: int
    nodes_to_remove: set[int]
    replacement_nodes: list[onnx.NodeProto]


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def const_array(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray:
    return numpy_helper.to_array(initializers[name])


def only_consumer(consumers: dict[str, list[int]], tensor_name: str, expected_op: str, nodes: list[onnx.NodeProto]) -> int | None:
    use_sites = consumers.get(tensor_name, [])
    if len(use_sites) != 1:
        return None
    index = use_sites[0]
    return index if nodes[index].op_type == expected_op else None


def matches_name(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(pattern in name for pattern in include):
        return False
    if exclude and any(pattern in name for pattern in exclude):
        return False
    return True


def dequantized_weight(
    initializers: dict[str, onnx.TensorProto],
    quantized_name: str,
    scale_name: str,
    zero_point_name: str,
) -> np.ndarray | None:
    quantized = const_array(initializers, quantized_name).astype(np.float32)
    scale = const_array(initializers, scale_name).astype(np.float32)
    zero_point = const_array(initializers, zero_point_name).astype(np.float32)
    if scale.size == 1 and zero_point.size == 1:
        return (quantized - float(zero_point.reshape(()))) * float(scale.reshape(()))
    if quantized.ndim < 1 or scale.ndim != 1 or zero_point.ndim != 1:
        return None
    if scale.shape != zero_point.shape or scale.shape[0] != quantized.shape[0]:
        return None
    reshape = (scale.shape[0],) + (1,) * (quantized.ndim - 1)
    return (quantized - zero_point.reshape(reshape)) * scale.reshape(reshape)


def build_rewrite_specs(
    model: onnx.ModelProto,
    include: list[str],
    exclude: list[str],
    *,
    preserve_activation_quantization: bool = False,
    integer_float_kernel: bool = False,
) -> list[RewriteSpec]:
    if integer_float_kernel:
        preserve_activation_quantization = True
    initializers = {init.name: init for init in model.graph.initializer}
    nodes = list(model.graph.node)
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            producers[output] = index
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(index)

    specs: list[RewriteSpec] = []
    for conv_index, node in enumerate(nodes):
        if node.op_type != "ConvInteger" or len(node.input) != 4 or len(node.output) != 1:
            continue
        if not matches_name(node.name or node.output[0], include, exclude):
            continue

        lhs_quant, rhs_quant, lhs_zero_point, rhs_zero_point = node.input
        if rhs_quant not in initializers or rhs_zero_point not in initializers:
            continue

        dql_index = producers.get(lhs_quant)
        if dql_index is None:
            continue
        dql = nodes[dql_index]
        if dql.op_type != "DynamicQuantizeLinear" or len(dql.output) != 3:
            continue
        if dql.output[0] != lhs_quant or dql.output[2] != lhs_zero_point:
            continue
        lhs_input = dql.input[0]
        lhs_scale = dql.output[1]

        cast_index = only_consumer(consumers, node.output[0], "Cast", nodes)
        if cast_index is None:
            continue
        cast_node = nodes[cast_index]

        scale_mul_index = None
        rhs_scale_name = None
        for candidate_index in consumers.get(lhs_scale, []):
            candidate = nodes[candidate_index]
            if candidate.op_type != "Mul" or len(candidate.input) != 2:
                continue
            other_input = candidate.input[0] if candidate.input[1] == lhs_scale else candidate.input[1]
            if other_input in initializers:
                scale_mul_index = candidate_index
                rhs_scale_name = other_input
                break
        if scale_mul_index is None or rhs_scale_name is None:
            continue

        scale_mul_output = nodes[scale_mul_index].output[0]
        output_mul_index = None
        for candidate_index in consumers.get(cast_node.output[0], []):
            candidate = nodes[candidate_index]
            if candidate.op_type == "Mul" and scale_mul_output in candidate.input:
                output_mul_index = candidate_index
                break
        if output_mul_index is None:
            continue
        conv_float_output = nodes[output_mul_index].output[0]

        weight = dequantized_weight(initializers, rhs_quant, rhs_scale_name, rhs_zero_point)
        if weight is None:
            continue

        if integer_float_kernel:
            # Only accept depthwise convolutions whose integer accumulations are
            # exactly representable in FP32. Preserve the original scale tail.
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            quantized = const_array(initializers, rhs_quant)
            if (quantized.dtype not in (np.dtype('uint8'), np.dtype('int8'))
                    or quantized.ndim != 3 or quantized.shape[1] != 1
                    or attrs.get('group', 1) != quantized.shape[0]
                    or quantized.shape[2] * 255 * 255 > 2 ** 24):
                continue
            zero = const_array(initializers, rhs_zero_point).astype(np.float32)
            if zero.size == 1:
                weight = quantized.astype(np.float32) - zero.reshape(())
            elif zero.shape == (quantized.shape[0],):
                weight = quantized.astype(np.float32) - zero.reshape(-1, 1, 1)
            else:
                continue

        base = (node.name or node.output[0]).replace("/", "_")
        weight_name = f"{base}_dequant_weight"
        model.graph.initializer.append(numpy_helper.from_array(weight.astype(np.float32), weight_name))

        replacement_nodes = []
        conv_input = lhs_input
        if preserve_activation_quantization:
            # Keep the original dynamic activation rounding. Only the convolution
            # kernel changes; this does not recover unquantized activations.
            conv_input = f"{base}_dequant_activation"
            activation_scale = lhs_scale
            if integer_float_kernel:
                activation_scale = f"{base}_unit_scale"
                model.graph.initializer.append(numpy_helper.from_array(
                    np.array(1, dtype=np.float32), activation_scale))
            replacement_nodes.append(helper.make_node(
                "DequantizeLinear", [lhs_quant, activation_scale, lhs_zero_point],
                [conv_input], name=f"{base}_activation_dequantized",
            ))
        conv_node = helper.make_node(
            "Conv",
            [conv_input, weight_name],
            [cast_node.output[0] if integer_float_kernel else conv_float_output],
            name=f"{base}_conv_dequantized",
        )
        conv_node.attribute.extend(node.attribute)
        replacement_nodes.append(conv_node)

        remove = {conv_index, cast_index}
        if not integer_float_kernel:
            remove.add(output_mul_index)
        dql_outputs = set(dql.output)
        dql_use_sites = {idx for output in dql_outputs for idx in consumers.get(output, [])}
        if not preserve_activation_quantization and dql_use_sites <= {conv_index, scale_mul_index}:
            remove.add(dql_index)
        if not integer_float_kernel and all(idx in {output_mul_index} for idx in consumers.get(scale_mul_output, [])):
            remove.add(scale_mul_index)

        specs.append(RewriteSpec(min(remove), remove, replacement_nodes))
    return specs


def apply_rewrites(model: onnx.ModelProto, specs: list[RewriteSpec]) -> int:
    by_first = {spec.first_node_index: spec for spec in specs}
    removed = {index for spec in specs for index in spec.nodes_to_remove}
    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(model.graph.node):
        spec = by_first.get(index)
        if spec is not None:
            rewritten.extend(spec.replacement_nodes)
        if index in removed:
            continue
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(specs)


def prune_unused_initializers(model: onnx.ModelProto) -> int:
    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    used.update(output.name for output in model.graph.output)
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def ort_optimize_to_file(input_path: Path, output_path: Path, level: str, threads: int) -> None:
    import onnxruntime as ort

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_path)
    options.graph_optimization_level = levels[level]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(output_path))


def update_config(
    source_dir: Path,
    output_dir: Path,
    *,
    include: list[str],
    exclude: list[str],
    rewritten: int,
    pruned_initializers: int,
    ort_optimize_final: str | None,
    preserve_activation_quantization: bool = False,
    integer_float_kernel: bool = False,
) -> None:
    source_config = source_dir / "config.json"
    if not source_config.exists():
        return
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["dequantized_conv_blocks"] = {
        "include": include,
        "exclude": exclude,
        "rewritten_blocks": rewritten,
        "pruned_initializers": pruned_initializers,
        "ort_optimized_final_encoder": ort_optimize_final,
        "preserve_activation_quantization": preserve_activation_quantization or integer_float_kernel,
        "integer_float_kernel": integer_float_kernel,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_model_dir(
    source_dir: Path,
    output_dir: Path,
    include: list[str],
    exclude: list[str],
    ort_optimize_final: str | None,
    ort_optimize_threads: int,
    preserve_activation_quantization: bool = False,
    integer_float_kernel: bool = False,
) -> None:
    source_encoder = source_dir / "encoder.onnx"
    if not source_encoder.exists():
        raise SystemExit(f"Missing source encoder: {source_encoder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_encoder, load_external_data=False)
    specs = build_rewrite_specs(model, include, exclude,
                                preserve_activation_quantization=preserve_activation_quantization,
                                integer_float_kernel=integer_float_kernel)
    rewritten = apply_rewrites(model, specs)
    pruned_initializers = prune_unused_initializers(model)
    output_encoder = output_dir / "encoder.onnx"
    onnx.save(model, output_encoder)
    onnx.checker.check_model(str(output_encoder))
    if ort_optimize_final:
        optimized_encoder = output_dir / "encoder.ort_optimized.onnx"
        ort_optimize_to_file(output_encoder, optimized_encoder, ort_optimize_final, ort_optimize_threads)
        optimized_encoder.replace(output_encoder)

    for sidecar in ("decoder_joint.onnx", "tokenizer.model"):
        src = source_dir / sidecar
        if src.exists():
            hardlink_or_copy(src, output_dir / sidecar)
    update_config(
        source_dir,
        output_dir,
        include=include,
        exclude=exclude,
        rewritten=rewritten,
        pruned_initializers=pruned_initializers,
        ort_optimize_final=ort_optimize_final,
        preserve_activation_quantization=preserve_activation_quantization,
        integer_float_kernel=integer_float_kernel,
    )

    print(
        f"[dequantize-conv] wrote {output_dir} rewritten_blocks={rewritten} "
        f"prunedInitializers={pruned_initializers} ortFinal={ort_optimize_final}"
    )
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


def rewrite_integer_depthwise_file(source: Path, output: Path) -> int:
    """Change only representable depthwise integer kernels, retaining scale tails."""
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    specs = build_rewrite_specs(model, ["depthwise_conv"], [], integer_float_kernel=True)
    count = apply_rewrites(model, specs)
    if not count:
        raise ValueError("no supported integer depthwise convolutions found")
    prune_unused_initializers(model)
    onnx.save(model, output)
    onnx.checker.check_model(str(output))
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[], help="Substring that matched ConvInteger node names must contain.")
    parser.add_argument("--exclude", action="append", default=[], help="Substring that matched ConvInteger node names must not contain.")
    parser.add_argument("--ort-optimize-final", choices=sorted(ORT_OPTIMIZATION_LEVELS))
    parser.add_argument("--ort-optimize-threads", type=int, default=1)
    parser.add_argument("--preserve-activation-quantization", action="store_true",
                        help="Retain original activation quantization before the float convolution kernel.")
    parser.add_argument("--integer-float-kernel", action="store_true",
                        help="Use exact FP32 integer-valued depthwise accumulation, preserving quantization and scales.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_model_dir(
        args.source_dir,
        args.output_dir,
        args.include,
        args.exclude,
        args.ort_optimize_final,
        args.ort_optimize_threads,
        args.preserve_activation_quantization,
        args.integer_float_kernel,
    )


if __name__ == "__main__":
    main()
