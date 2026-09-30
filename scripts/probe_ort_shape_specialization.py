#!/usr/bin/env python3
"""Demonstrate ORT session-time folding after free-dimension overrides."""
from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="wordpipe-ort-shapes-") as temporary:
        root = Path(temporary)
        graph = helper.make_graph(
            [
                helper.make_node("Shape", ["audio"], ["shape"]),
                helper.make_node("Identity", ["audio"], ["output"]),
            ],
            "shape-specialization",
            [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 128, "frames"])],
            [
                helper.make_tensor_value_info("shape", TensorProto.INT64, [3]),
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128, "frames"]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        source = root / "source.onnx"
        onnx.save(model, source)
        for latency_ms in (560, 1120):
            frames = latency_ms // 10 + 9
            optimized = root / f"{latency_ms}.onnx"
            options = ort.SessionOptions()
            options.add_free_dimension_override_by_name("frames", frames)
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            options.optimized_model_filepath = str(optimized)
            session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
            shape, output = session.run(None, {"audio": np.zeros((1, 128, frames), dtype=np.float32)})
            np.testing.assert_array_equal(shape, [1, 128, frames])
            assert output.shape == (1, 128, frames)
            folded = onnx.load(optimized)
            assert not any(node.op_type == "Shape" for node in folded.graph.node)
            print(f"{latency_ms} ms: session override frames={frames}; Shape folded into initializer")
        if args.fixture_dir is not None:
            args.fixture_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, args.fixture_dir / "source.onnx")
            options = ort.SessionOptions()
            options.add_free_dimension_override_by_name("frames", 65)
            options.optimized_model_filepath = str(args.fixture_dir / "session.ort")
            ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    main()
