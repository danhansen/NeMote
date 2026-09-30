from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _load_rewriter():
    script = Path(__file__).resolve().parents[1] / "scripts" / "rewrite_nemotron_projected_kv_cache.py"
    spec = importlib.util.spec_from_file_location("rewrite_nemotron_projected_kv_cache", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _value(name: str, dims: list[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


def _fp32_projection_model(dynamic: bool = False) -> onnx.ModelProto:
    weight = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    frames = "current_frames" if dynamic else 1
    total = "total_frames" if dynamic else 3
    nodes = [
        helper.make_node(
            "Concat",
            ["raw_key_cache", "current_key"],
            ["key_concat"],
            name="/encoder/layers.0/self_attn/key_concat",
            axis=1,
        ),
        helper.make_node(
            "MatMul",
            ["key_concat", "key_weight"],
            ["key_out"],
            name="/encoder/layers.0/self_attn/linear_k/MatMul",
        ),
        helper.make_node(
            "Concat",
            ["raw_value_cache", "current_value"],
            ["value_concat"],
            name="/encoder/layers.0/self_attn/value_concat",
            axis=1,
        ),
        helper.make_node(
            "MatMul",
            ["value_concat", "value_weight"],
            ["value_out"],
            name="/encoder/layers.0/self_attn/linear_v/MatMul",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "fp32-projected-cache-test",
        [
            _value("cache_last_channel", [1, 1, 2, 4]),
            _value("raw_key_cache", [1, 2, 4]),
            _value("current_key", [1, frames, 4]),
            _value("raw_value_cache", [1, 2, 4]),
            _value("current_value", [1, frames, 4]),
        ],
        [_value("key_out", [1, total, 4]), _value("value_out", [1, total, 4])],
        [
            numpy_helper.from_array(weight, "key_weight"),
            numpy_helper.from_array(weight, "value_weight"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 9
    return model


class ProjectedCacheRewriteTests(unittest.TestCase):
    def test_one_rewritten_graph_accepts_both_chunk_sizes(self) -> None:
        import onnxruntime as ort

        rewriter = _load_rewriter()
        rng = np.random.default_rng(123)
        weight = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.onnx"
            output = Path(tmp) / "output.onnx"
            onnx.save(_fp32_projection_model(dynamic=True), source)
            rewriter.rewrite_model(source, output, "fp32")
            baseline = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
            projected = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            for frames in (7, 14):
                with self.subTest(frames=frames):
                    raw = rng.standard_normal((1, 2, 4)).astype(np.float32)
                    current = rng.standard_normal((1, frames, 4)).astype(np.float32)
                    common = {"cache_last_channel": np.zeros((1, 1, 2, 4), dtype=np.float32),
                              "raw_key_cache": raw, "raw_value_cache": raw,
                              "current_key": current, "current_value": current}
                    expected = baseline.run(["key_out", "value_out"], common)
                    actual = projected.run(["key_out", "value_out"], {
                        **common, "cache_key_layer_0": raw @ weight,
                        "cache_value_layer_0": raw @ weight,
                    })
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_rewrite_supports_native_fp32_matmul_projection(self) -> None:
        rewriter = _load_rewriter()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.onnx"
            output = Path(tmp) / "output.onnx"
            onnx.save(_fp32_projection_model(), source)

            rewriter.rewrite_model(source, output, "fp32")

            rewritten = onnx.load(output)
            onnx.checker.check_model(rewritten)
            counts = Counter(node.op_type for node in rewritten.graph.node)
            input_names = {value.name for value in rewritten.graph.input}
            output_names = {value.name for value in rewritten.graph.output}
            node_names = {node.name for node in rewritten.graph.node}

        self.assertEqual(counts["MatMul"], 2)
        self.assertEqual(counts["Concat"], 2)
        self.assertIn("cache_key_layer_0", input_names)
        self.assertIn("cache_value_layer_0", input_names)
        self.assertIn("projected_current_key_layer_0", output_names)
        self.assertIn("projected_current_value_layer_0", output_names)
        self.assertNotIn("/encoder/layers.0/self_attn/linear_k/MatMul", node_names)
        self.assertNotIn("/encoder/layers.0/self_attn/linear_v/MatMul", node_names)


if __name__ == "__main__":
    unittest.main()
