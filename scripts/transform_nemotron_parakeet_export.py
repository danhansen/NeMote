#!/usr/bin/env python3
"""Transform a FP32 Nemotron parakeet export into the runtime int8 package.

Run this after export_nemotron_parakeet_optimized.py --export-only. Keeping
this in a separate process avoids holding the full NeMo/Torch model in memory
while ONNX Runtime quantizes the multi-GB encoder graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from rewrite_nemotron_projected_kv_cache import rewrite_model as rewrite_projected_cache
from rewrite_nemotron_pointwise_convs import rewrite_file as rewrite_pointwise_convs

ORT_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}


KEEP_FILES = {
    "encoder.fp32.consolidated.onnx",
    "encoder.fp32.consolidated.data",
    "decoder_joint.fp32.onnx",
    "tokenizer.model",
    "export_config.json",
    "config.json",
    "encoder.quant.onnx",
    "encoder.onnx",
    "decoder_joint.onnx",
    "encoder.pointwise-fp32.onnx",
    "encoder.pointwise-fp32.data",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--pointwise-linear", action=argparse.BooleanOptionalAction, default=False,
                        help="Lower Conformer unit-kernel pointwise convs to MatMul before quantization.")
    parser.add_argument("--pointwise-parity-output", type=Path,
                        help="Verify lowered FP32 encoder against dynamic native fixtures before quantization.")
    parser.add_argument("--depthwise-integer-float-kernel", action="store_true",
                        help="Use FP32 for exact, integer-valued depthwise kernels after quantization.")
    parser.add_argument("--depthwise-parity-output", type=Path,
                        help="Require bit-exact generic/specialized encoder equivalence before promotion.")
    parser.add_argument(
        "--projected-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply projected K/V cache rewrite to the quantized encoder.",
    )
    parser.add_argument(
        "--projected-cache-current-projection",
        choices=("auto", "dynamic-int8", "fp32"),
        default="auto",
        help=(
            "Projection used for the current chunk inside the projected-cache "
            "rewrite. auto preserves the existing behavior: dynamic-int8 for "
            "quantized graphs, fp32 for unquantized graphs."
        ),
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run dynamic QUInt8 quantization.",
    )
    parser.add_argument(
        "--quantize-per-channel",
        action="store_true",
        help="Use per-channel weight quantization during dynamic QUInt8 quantization.",
    )
    parser.add_argument(
        "--fp32-decoder",
        action="store_true",
        help=(
            "Keep decoder_joint.onnx as the FP32 NeMo export even when the "
            "encoder is dynamically quantized. Benchmark before promoting; "
            "this can trade strict spelling WER for throughput."
        ),
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep FP32 encoder/decoder artifacts after transform.",
    )
    parser.add_argument(
        "--ort-optimize-final",
        choices=sorted(ORT_OPTIMIZATION_LEVELS),
        help="Serialize the final encoder through ONNX Runtime at the selected optimization level.",
    )
    parser.add_argument(
        "--ort-optimize-threads",
        type=int,
        default=1,
        help="Intra-op threads to use while serializing the ORT-optimized final encoder.",
    )
    return parser.parse_args()


def projected_cache_current_projection(args: argparse.Namespace) -> str:
    if args.projected_cache_current_projection != "auto":
        return args.projected_cache_current_projection
    return "dynamic-int8" if args.quantize else "fp32"


def cleanup_export_shards(model_dir: Path) -> None:
    for path in model_dir.iterdir():
        if path.is_file() and path.name not in KEEP_FILES:
            path.unlink()


def quantize_to_single_file(input_path: Path, output_path: Path, *, per_channel: bool) -> None:
    print(
        f"[transform] quantizing {input_path.name} -> {output_path.name} perChannel={per_channel}",
        flush=True,
    )
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        per_channel=per_channel,
        use_external_data_format=False,
    )
    onnx.checker.check_model(str(output_path))


def ort_optimize_to_file(input_path: Path, output_path: Path, level: str, threads: int) -> None:
    import onnxruntime as ort

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    print(f"[transform] ORT optimizing {input_path.name} -> {output_path.name} level={level}", flush=True)
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_path)
    options.graph_optimization_level = levels[level]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(output_path))


def load_config(model_dir: Path) -> dict:
    for name in ("export_config.json", "config.json"):
        path = model_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"Missing model directory: {model_dir}")

    cleanup_export_shards(model_dir)

    encoder_fp32 = model_dir / "encoder.fp32.consolidated.onnx"
    decoder_fp32 = model_dir / "decoder_joint.fp32.onnx"
    if not encoder_fp32.exists() or not decoder_fp32.exists():
        raise SystemExit("Expected encoder.fp32.consolidated.onnx and decoder_joint.fp32.onnx")

    encoder_for_projected = encoder_fp32
    encoder_for_quantization = encoder_fp32
    pointwise_count = 0
    if args.pointwise_linear:
        encoder_for_quantization = model_dir / "encoder.pointwise-fp32.onnx"
        pointwise_count = rewrite_pointwise_convs(encoder_fp32, encoder_for_quantization)
        encoder_for_projected = encoder_for_quantization
        if args.pointwise_parity_output:
            from verify_dynamic_nemotron_export import verify
            report = verify(encoder_for_quantization, model_dir / "dynamic-reference", atol=1e-3, rtol=1e-4)
            args.pointwise_parity_output.parent.mkdir(parents=True, exist_ok=True)
            args.pointwise_parity_output.write_text(json.dumps(report, indent=2) + "\n")
    elif args.pointwise_parity_output:
        raise SystemExit("--pointwise-parity-output requires --pointwise-linear")
    if args.quantize:
        encoder_quant = model_dir / "encoder.quant.onnx"
        decoder_quant = model_dir / "decoder_joint.onnx"
        quantize_to_single_file(encoder_for_quantization, encoder_quant, per_channel=args.quantize_per_channel)
        if args.fp32_decoder:
            decoder_quant.write_bytes(decoder_fp32.read_bytes())
        else:
            quantize_to_single_file(decoder_fp32, decoder_quant, per_channel=args.quantize_per_channel)
        encoder_for_projected = encoder_quant
    else:
        (model_dir / "decoder_joint.onnx").write_bytes(decoder_fp32.read_bytes())

    final_encoder = model_dir / "encoder.onnx"
    current_projection = projected_cache_current_projection(args)
    if args.projected_cache:
        print("[transform] rewriting encoder with projected cache", flush=True)
        rewrite_projected_cache(
            encoder_for_projected,
            final_encoder,
            current_projection,
            external_data=not args.quantize,
        )
    else:
        final_encoder.write_bytes(encoder_for_projected.read_bytes())

    depthwise_count = 0
    if args.depthwise_integer_float_kernel:
        if not args.quantize:
            raise ValueError("integer depthwise lowering requires a quantized encoder")
        if args.depthwise_parity_output is None:
            raise ValueError("integer depthwise lowering requires --depthwise-parity-output")
        from dequantize_nemotron_conv_blocks import rewrite_integer_depthwise_file
        rewritten_encoder = model_dir / "encoder.depthwise-integer.onnx"
        depthwise_count = rewrite_integer_depthwise_file(final_encoder, rewritten_encoder)
        expected = load_config(model_dir)["num_encoder_layers"]
        if depthwise_count != expected:
            raise ValueError(f"incomplete depthwise kernel rewrite: {depthwise_count}/{expected}")
        from verify_nemotron_encoder_rewrite import verify
        report = verify(final_encoder, rewritten_encoder, model_dir / "dynamic-reference")
        args.depthwise_parity_output.parent.mkdir(parents=True, exist_ok=True)
        args.depthwise_parity_output.write_text(json.dumps(report, indent=2) + "\n")
        rewritten_encoder.replace(final_encoder)
    elif args.depthwise_parity_output is not None:
        raise ValueError("--depthwise-parity-output requires integer depthwise lowering")

    if args.ort_optimize_final:
        optimized_encoder = model_dir / "encoder.ort_optimized.onnx"
        ort_optimize_to_file(
            final_encoder,
            optimized_encoder,
            args.ort_optimize_final,
            args.ort_optimize_threads,
        )
        optimized_encoder.replace(final_encoder)

    config = load_config(model_dir)
    config["projected_cache"] = args.projected_cache
    config["pointwise_linear_rewrite_count"] = pointwise_count
    config["depthwise_integer_float_rewrite_count"] = depthwise_count
    config["projected_cache_current_projection"] = current_projection if args.projected_cache else None
    config["dynamic_quint8_quantization"] = args.quantize
    config["dynamic_quint8_per_channel"] = args.quantize_per_channel if args.quantize else False
    config["decoder_joint_fp32"] = bool(args.fp32_decoder or not args.quantize)
    config["ort_optimized_final_encoder"] = args.ort_optimize_final
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.keep_fp32:
        for name in (
            "encoder.fp32.consolidated.onnx",
            "encoder.fp32.consolidated.data",
            "encoder.quant.onnx",
            "decoder_joint.fp32.onnx",
            "export_config.json",
            "encoder.pointwise-fp32.onnx",
            "encoder.pointwise-fp32.data",
        ):
            (model_dir / name).unlink(missing_ok=True)

    print(f"[transform] wrote {model_dir}", flush=True)
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
