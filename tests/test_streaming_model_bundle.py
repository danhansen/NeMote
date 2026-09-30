from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import shutil
import types

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def bundle_module():
    path = Path(__file__).resolve().parents[1] / "scripts/bundle_nemotron_streaming_modes.py"
    spec = importlib.util.spec_from_file_location("streaming_bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_dir(path: Path, frames: int, weight: np.ndarray) -> None:
    path.mkdir()
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])], "encoder",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, frames, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, frames, 256])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path / "encoder.onnx")
    (path / "config.json").write_text(json.dumps({
        "right_context": frames - 1, "model_family": "english",
        "dynamic_quint8_quantization": False,
        "projected_cache": True,
        "fixed_streaming_shapes": {
            "input_frames": frames * 8 + 9, "output_frames": frames,
            "num_layers": 24, "cache_len": 70, "hidden_dim": 1024,
            "conv_context": 8,
        },
    }))
    (path / "decoder_joint.onnx").write_bytes(b"same decoder")
    (path / "tokenizer.model").write_bytes(b"same tokenizer")


class StreamingBundleTests(unittest.TestCase):
    def test_downloading_second_mode_reuses_verified_installed_weights(self):
        from wordpipe.models import download_streaming_profile, profile_spec

        module = bundle_module()
        weight = np.ones((256, 256), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub, other = root / "hub", root / "other"
            model_dir(hub, 7, weight)
            model_dir(other, 14, weight)
            module.append_mode(hub, other, 1120)
            installed = profile_spec("fast").output_dir(root / "models", "english")
            installed.mkdir(parents=True)
            for name in ("encoder.onnx", "config.json", "encoder.shared-weights.data",
                         "decoder_joint.onnx", "tokenizer.model"):
                shutil.copy2(hub / name, installed / name)
            downloads = []

            def fetch(*, filename, local_dir, **kwargs):
                downloads.append(filename)
                destination = Path(local_dir) / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hub / filename, destination)
                return str(destination)

            hf = types.ModuleType("huggingface_hub")
            hf.hf_hub_download = fetch
            hf.hf_hub_url = lambda repo, filename: filename
            with mock.patch.dict("sys.modules", {"huggingface_hub": hf}):
                runtime = download_streaming_profile(
                    profile="fast", model_root=root / "models", family="english",
                    latency_ms=1120,
                )
            self.assertEqual(set(downloads), {
                "wordpipe-streaming-bundle.json", "encoder.1120ms.onnx", "config.1120ms.json",
            })
            self.assertEqual((runtime / "encoder.shared-weights.data").stat().st_ino,
                             (installed / "encoder.shared-weights.data").stat().st_ino)
            self.assertEqual(json.loads((runtime / "config.json").read_text())["right_context"], 13)

    def test_both_encoders_use_one_weight_file_with_identical_outputs(self):
        module = bundle_module()
        rng = np.random.default_rng(321)
        weight = rng.standard_normal((256, 256)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, other = root / "base", root / "other"
            model_dir(base, 7, weight)
            model_dir(other, 14, weight)
            manifest = module.append_mode(base, other, 1120)
            self.assertEqual(manifest["shared_files"].count("encoder.shared-weights.data"), 1)
            self.assertEqual((base / "encoder.shared-weights.data").stat().st_size, weight.nbytes)
            for frames, name in ((7, "encoder.onnx"), (14, "encoder.1120ms.onnx")):
                session = ort.InferenceSession(str(base / name), providers=["CPUExecutionProvider"])
                x = rng.standard_normal((1, frames, 256)).astype(np.float32)
                np.testing.assert_allclose(session.run(None, {"x": x})[0], x @ weight,
                                           rtol=1e-5, atol=1e-5)
            self.assertLess((base / "encoder.1120ms.onnx").stat().st_size, 2048)

    def test_different_weights_are_rejected_instead_of_duplicated(self):
        module = bundle_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, other = root / "base", root / "other"
            model_dir(base, 7, np.ones((256, 256), dtype=np.float32))
            model_dir(other, 14, np.zeros((256, 256), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "refusing duplicated"):
                module.append_mode(base, other, 1120)


if __name__ == "__main__":
    unittest.main()
