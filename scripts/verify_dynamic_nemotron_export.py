#!/usr/bin/env python3
"""Validate a generic FP32 encoder against saved native NeMo fixtures.

Run after the NeMo export process exits to avoid retaining Torch and ORT models
at the same time. This gate is numerical parity, not an ASR/WER benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import onnxruntime as ort


def projection_weights(raw_encoder: Path):
    """Load just the K/V weights, not the entire multi-GB FP32 checkpoint."""
    import onnx
    from rewrite_nemotron_projected_kv_cache import find_projection_node, infer_spec
    model = onnx.load(raw_encoder, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    weights = {}
    for layer in range(infer_spec(model).num_layers):
        for kind, short in (("key", "k"), ("value", "v")):
            node = find_projection_node(model, layer, short)
            if node.op_type != "MatMul":
                raise ValueError("native reference projection requires FP32 MatMul weights")
            weights[f"cache_{kind}_layer_{layer}"] = (layer, onnx.numpy_helper.to_array(
                initializers[node.input[1]], base_dir=str(raw_encoder.parent)))
    return weights


def verify(encoder: Path, fixtures: Path, *, threads: int = 2, atol: float = 1e-4, rtol: float = 1e-4,
           raw_encoder: Path | None = None):
    manifest = json.loads((fixtures / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise ValueError("unsupported fixture format")
    results = []
    weights = projection_weights(raw_encoder) if raw_encoder else {}
    with tempfile.TemporaryDirectory(prefix="wordpipe-dynamic-parity-") as temporary:
        supported_modes = sorted({record["chunk_frames"] for record in manifest["records"]})
        for mode in ("dynamic", *supported_modes):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            cached = Path(temporary) / f"encoder-{mode}.onnx"
            if mode != "dynamic":
                mel_frames = mode * manifest["subsampling_factor"] + manifest["mel_frames_overhead"]
                options.add_free_dimension_override_by_name("batch", 1)
                options.add_free_dimension_override_by_name("mel_frames", mel_frames)
                options.add_free_dimension_override_by_name("encoder_frames", mode)
                options.add_free_dimension_override_by_name("current_frames", mode)
                options.optimized_model_filepath = str(cached)
                options.add_session_config_entry(
                    "session.optimized_model_external_initializers_file_name",
                    f"encoder-{mode}.data",
                )
                options.add_session_config_entry(
                    "session.optimized_model_external_initializers_min_size_in_bytes", "65536",
                )
            session = ort.InferenceSession(str(encoder), options, providers=["CPUExecutionProvider"])
            names = [item.name for item in session.get_inputs()]
            outputs = [item.name for item in session.get_outputs()]
            reload_cases = []
            for record in manifest["records"]:
                if mode != "dynamic" and record["chunk_frames"] != mode:
                    continue
                filename = record["fixture"]
                if Path(filename).name != filename:
                    raise ValueError("unsafe fixture filename")
                with np.load(fixtures / filename, allow_pickle=False) as data:
                    feeds = {name: (data["cache_last_channel"][weights[name][0]] @ weights[name][1]
                                    if name in weights else data[name]) for name in names}
                    started = time.perf_counter()
                    actual = session.run(outputs, feeds)
                    elapsed = time.perf_counter() - started
                    maximum_error = 0.0
                    for name, value in zip(outputs, actual, strict=True):
                        if "expected_" + name not in data:
                            if not name.startswith("projected_current_") or not np.isfinite(value).all():
                                raise ValueError(f"unexpected or nonfinite output: {name}")
                            continue
                        reference = data["expected_" + name]
                        if np.issubdtype(reference.dtype, np.floating):
                            np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol, err_msg=name)
                            maximum_error = max(maximum_error, float(np.max(np.abs(value - reference))))
                        else:
                            np.testing.assert_array_equal(value, reference, err_msg=name)
                    if mode != "dynamic":
                        reload_cases.append((feeds, actual))
                result = {**record, "session": mode, "ort_max_absolute_error": maximum_error,
                          "encoder_seconds": elapsed}
                print(json.dumps(result), flush=True)
                results.append(result)
            del session
            # Never retain two full encoders just to verify a cache reload.
            if reload_cases:
                reload_options = ort.SessionOptions()
                reload_options.intra_op_num_threads = threads
                reload_options.inter_op_num_threads = 1
                reload_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                reloaded = ort.InferenceSession(str(cached), reload_options, providers=["CPUExecutionProvider"])
                for feeds, expected in reload_cases:
                    for value, reference in zip(reloaded.run(outputs, feeds), expected, strict=True):
                        np.testing.assert_array_equal(value, reference)
                del reloaded
                cached.unlink()
                (Path(temporary) / f"encoder-{mode}.data").unlink(missing_ok=True)
    return {"format": 1, "encoder": str(encoder), "ort_version": ort.__version__,
            "raw_reference_encoder": str(raw_encoder) if raw_encoder else None,
            "atol": atol, "rtol": rtol, "records": results, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--raw-encoder", type=Path,
                        help="Native FP32 export used to project fixture caches for a rewritten FP32 encoder.")
    args = parser.parse_args()
    report = verify(args.encoder, args.fixtures, threads=args.threads, raw_encoder=args.raw_encoder,
                    atol=args.atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
