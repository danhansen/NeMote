#!/usr/bin/env python3
"""Build a Nemotron model directory with fixed streaming encoder shapes."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ORT_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}
METADATA_PREFIX = "wordpipe.fixed_streaming."
EXTERNAL_DATA_THRESHOLD_BYTES = 1500 * 1024 * 1024


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def set_tensor_shape(value_info: onnx.ValueInfoProto, dims: list[int]) -> None:
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for value in dims:
        dim = shape.dim.add()
        dim.dim_value = int(value)


def evaluate_dim_param(value: str, symbols: dict[str, int]) -> int | None:
    if value in symbols:
        return symbols[value]
    if not value or not re.fullmatch(r"[0-9A-Za-z_+\-*/ ().,]+", value):
        return None
    try:
        result = eval(
            value.replace("Min", "min").replace("Max", "max"),
            {"__builtins__": {}},
            {"floor": math.floor, "ceil": math.ceil, "min": min, "max": max, **symbols},
        )
    except Exception:
        return None
    if isinstance(result, float) and result.is_integer():
        return int(result)
    if isinstance(result, int):
        return result
    return None


def resolve_symbolic_shapes(model: onnx.ModelProto, symbols: dict[str, int]) -> int:
    resolved = 0
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                continue
            value = evaluate_dim_param(dim.dim_param, symbols)
            if value is None:
                continue
            dim.ClearField("dim_param")
            dim.dim_value = value
            resolved += 1
    return resolved


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = tensor_shape(value_info)
        if dims and all(isinstance(dim, int) for dim in dims):
            shapes[value_info.name] = [int(dim) for dim in dims]
    return shapes


def replace_static_shape_nodes(model: onnx.ModelProto) -> int:
    shapes = tensor_shapes(model)
    replacement_initializers: dict[str, onnx.TensorProto] = {}
    rewritten: list[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type == "Shape" and node.input and node.output and node.input[0] in shapes:
            replacement_initializers[node.output[0]] = numpy_helper.from_array(
                np.asarray(shapes[node.input[0]], dtype=np.int64),
                node.output[0],
            )
            replaced += 1
            continue
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    if replacement_initializers:
        kept = [init for init in model.graph.initializer if init.name not in replacement_initializers]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        model.graph.initializer.extend(replacement_initializers.values())
    return replaced


def add_or_replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    kept = [initializer for initializer in model.graph.initializer if initializer.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(value, name))


def remove_graph_input(model: onnx.ModelProto, name: str) -> bool:
    kept = [value_info for value_info in model.graph.input if value_info.name != name]
    if len(kept) == len(model.graph.input):
        return False
    del model.graph.input[:]
    model.graph.input.extend(kept)
    return True


def fix_processed_signal_length(model: onnx.ModelProto, input_frames: int) -> bool:
    removed = remove_graph_input(model, "processed_signal_length")
    if removed:
        add_or_replace_initializer(
            model,
            "processed_signal_length",
            np.asarray([input_frames], dtype=np.int64),
        )
    return removed


def set_fixed_shapes(
    model: onnx.ModelProto,
    *,
    input_frames: int,
    output_frames: int,
    num_layers: int,
    cache_len: int,
    hidden_dim: int,
    conv_context: int,
) -> None:
    for value_info in model.graph.input:
        name = value_info.name
        if name == "processed_signal":
            set_tensor_shape(value_info, [1, 128, input_frames])
        elif name in {"processed_signal_length", "cache_last_channel_len", "prompt_index"}:
            set_tensor_shape(value_info, [1])
        elif name == "cache_last_channel":
            set_tensor_shape(value_info, [num_layers, 1, cache_len, hidden_dim])
        elif name == "cache_last_time":
            set_tensor_shape(value_info, [num_layers, 1, hidden_dim, conv_context])
        elif name.startswith("cache_key_layer_") or name.startswith("cache_value_layer_"):
            set_tensor_shape(value_info, [1, cache_len, hidden_dim])

    for value_info in model.graph.output:
        name = value_info.name
        if name == "encoded":
            set_tensor_shape(value_info, [1, hidden_dim, output_frames])
        elif name in {"encoded_len", "cache_last_channel_len_next"}:
            set_tensor_shape(value_info, [1])
        elif name == "cache_last_channel_next":
            set_tensor_shape(value_info, [num_layers, 1, cache_len, hidden_dim])
        elif name == "cache_last_time_next":
            set_tensor_shape(value_info, [num_layers, 1, hidden_dim, conv_context])
        elif name.startswith("projected_current_key_layer_") or name.startswith(
            "projected_current_value_layer_"
        ):
            set_tensor_shape(value_info, [1, output_frames, hidden_dim])


def update_config(source_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    source_config = source_dir / "config.json"
    if not source_config.exists():
        return
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["fixed_streaming_shapes"] = {
        "input_frames": args.input_frames,
        "output_frames": args.output_frames,
        "num_layers": args.num_layers,
        "cache_len": args.cache_len,
        "hidden_dim": args.hidden_dim,
        "conv_context": args.conv_context,
        "replaced_shape_nodes": args.replaced_shape_nodes,
        "resolved_symbolic_dims": args.resolved_symbolic_dims,
        "constant_processed_signal_length": args.constant_processed_signal_length_applied,
    }
    config["ort_optimized_final_encoder"] = args.ort_optimize_final
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def set_metadata(model: onnx.ModelProto, args: argparse.Namespace) -> None:
    kept = [item for item in model.metadata_props if not item.key.startswith(METADATA_PREFIX)
            and item.key not in ("chunk_size_output_frames", "pre_encode_cache")]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    for key in (
        "input_frames",
        "output_frames",
        "cache_len",
        "hidden_dim",
        "conv_context",
        "constant_processed_signal_length_applied",
    ):
        item = model.metadata_props.add()
        item.key = f"{METADATA_PREFIX}{key}"
        item.value = str(getattr(args, key))
    for key, value in (
        ("chunk_size_output_frames", args.output_frames),
        ("pre_encode_cache", args.input_frames - args.output_frames * 8),
    ):
        item = model.metadata_props.add()
        item.key = key
        item.value = str(value)


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


def initializer_bytes(model: onnx.ModelProto) -> int:
    total = 0
    for initializer in model.graph.initializer:
        if initializer.raw_data:
            total += len(initializer.raw_data)
        elif initializer.HasField("data_location") and initializer.data_location == onnx.TensorProto.EXTERNAL:
            for item in initializer.external_data:
                if item.key == "length":
                    try:
                        total += int(item.value)
                    except ValueError:
                        pass
        else:
            total += initializer.ByteSize()
    return total


def save_model(model: onnx.ModelProto, output_path: Path) -> None:
    if initializer_bytes(model) > EXTERNAL_DATA_THRESHOLD_BYTES:
        (output_path.parent / f"{output_path.name}.data").unlink(missing_ok=True)
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output_path.name + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(model, output_path)


def build_variant(source_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    source_encoder = source_dir / "encoder.onnx"
    if not source_encoder.exists():
        raise SystemExit(f"Missing source encoder: {source_encoder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_encoder, load_external_data=True)
    set_fixed_shapes(
        model,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        num_layers=args.num_layers,
        cache_len=args.cache_len,
        hidden_dim=args.hidden_dim,
        conv_context=args.conv_context,
    )
    if args.constant_processed_signal_length:
        args.constant_processed_signal_length_applied = fix_processed_signal_length(
            model,
            args.input_frames,
        )
    args.resolved_symbolic_dims = resolve_symbolic_shapes(
        model,
        {"batch": 1, "time": args.input_frames, "current_frames": args.output_frames},
    )
    args.replaced_shape_nodes = replace_static_shape_nodes(model)
    set_metadata(model, args)

    fixed_encoder = output_dir / "encoder.onnx"
    save_model(model, fixed_encoder)
    onnx.checker.check_model(str(fixed_encoder))

    if args.ort_optimize_final:
        optimized_encoder = output_dir / "encoder.ort_optimized.onnx"
        ort_optimize_to_file(
            fixed_encoder,
            optimized_encoder,
            args.ort_optimize_final,
            args.ort_optimize_threads,
        )
        optimized_encoder.replace(fixed_encoder)

    for sidecar in ("decoder_joint.onnx", "tokenizer.model"):
        src = source_dir / sidecar
        if src.exists():
            hardlink_or_copy(src, output_dir / sidecar)
    update_config(source_dir, output_dir, args)

    print(
        f"[fixed-shape] wrote {output_dir} inputFrames={args.input_frames} "
        f"outputFrames={args.output_frames} resolvedSymbolicDims={args.resolved_symbolic_dims} "
        f"replacedShapeNodes={args.replaced_shape_nodes} "
        f"constantProcessedSignalLength={args.constant_processed_signal_length_applied} "
        f"ortFinal={args.ort_optimize_final}"
    )
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-frames", type=int, default=65)
    parser.add_argument("--output-frames", type=int, default=7)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--cache-len", type=int, default=56)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--conv-context", type=int, default=8)
    parser.add_argument(
        "--constant-processed-signal-length",
        action="store_true",
        help=(
            "Remove processed_signal_length from graph inputs and replace it "
            "with a fixed int64 initializer equal to --input-frames."
        ),
    )
    parser.add_argument("--ort-optimize-final", choices=sorted(ORT_OPTIMIZATION_LEVELS))
    parser.add_argument("--ort-optimize-threads", type=int, default=1)
    args = parser.parse_args()
    args.resolved_symbolic_dims = 0
    args.replaced_shape_nodes = 0
    args.constant_processed_signal_length_applied = False
    return args


def main() -> None:
    args = parse_args()
    build_variant(args.source_dir, args.output_dir, args)


if __name__ == "__main__":
    main()
