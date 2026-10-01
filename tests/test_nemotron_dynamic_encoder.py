from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "run with the isolated NeMo/Torch export environment")
class DynamicEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import nemotron_dynamic_encoder as dynamic
        cls.dynamic = dynamic

    def test_masks_match_nemo_formula_at_both_contexts_and_padding(self):
        torch = self.torch
        for frames in (1, 2, 7, 14):
            for cache in (0, 5, 70):
                for valid_current in (0, max(0, frames - 2), frames):
                    n = 70 + frames
                    padding_length = torch.tensor([70 + valid_current])
                    offset = torch.tensor([70 - cache])
                    pad, mask = self.dynamic.chunked_attention_masks(
                        chunk_frames=torch.tensor(frames), left_context=70,
                        padding_length=padding_length, max_audio_length=n,
                        offset=offset, device=torch.device("cpu"),
                    )
                    indices = np.arange(n)
                    chunks = indices // frames
                    differences = chunks[:, None] - chunks[None, :]
                    valid = (indices >= 70 - cache) & (indices < 70 + valid_current)
                    allowed = (differences <= 70 // frames) & (differences >= 0)
                    expected = ~(allowed & valid[:, None] & valid[None, :])
                    np.testing.assert_array_equal(pad.numpy(), (~valid)[None, :])
                    np.testing.assert_array_equal(mask.numpy(), expected[None, :, :])

    def test_native_method_is_restored_on_success_and_failure(self):
        class Encoder:
            att_context_size = [70, 6]
            def _create_masks(self):
                return "native"
        encoder = Encoder()
        for fail in (False, True):
            try:
                with self.dynamic.dynamic_attention_context(encoder, self.torch.tensor(7)):
                    self.assertIn("_create_masks", encoder.__dict__)
                    if fail:
                        raise RuntimeError("test")
            except RuntimeError:
                pass
            self.assertNotIn("_create_masks", encoder.__dict__)
            self.assertEqual(encoder._create_masks(), "native")

    def test_unsupported_encoder_configs_are_rejected(self):
        encoder = types.SimpleNamespace(
            att_context_style="chunked_limited", self_attention_model="rel_pos",
            att_context_size=[70, 6],
            streaming_cfg=types.SimpleNamespace(cache_drop_size=0, last_channel_cache_size=70),
        )
        self.dynamic.validate_dynamic_encoder(encoder)
        encoder.streaming_cfg.cache_drop_size = 1
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            self.dynamic.validate_dynamic_encoder(encoder)

    def test_supported_modes_are_derived_from_checkpoint_metadata(self):
        encoder = types.SimpleNamespace(
            att_context_style="chunked_limited", self_attention_model="rel_pos",
            att_context_size=[70, 6],
            att_context_size_all=[[70, 13], [70, 6], [70, 1], [70, 0], [56, 3]],
            streaming_cfg=types.SimpleNamespace(cache_drop_size=0, last_channel_cache_size=70),
        )
        self.assertEqual(self.dynamic.supported_chunk_frames(encoder), [1, 2, 7, 14])
        encoder.att_context_size_all = [[70, 4], [70, 9]]
        self.assertEqual(self.dynamic.supported_chunk_frames(encoder), [5, 10])
        encoder.att_context_size_all = [[70, 2]]
        with self.assertRaisesRegex(ValueError, "valid finite"):
            self.dynamic.supported_chunk_frames(encoder)

    def test_torch_export_preserves_dynamic_mask_and_ort_specializes_it(self):
        import onnx
        import onnxruntime as ort
        torch = self.torch
        dynamic = self.dynamic

        class Masks(torch.nn.Module):
            def forward(self, features, cache_length):
                chunk = torch.div(torch._shape_as_tensor(features)[2] - 9, 8, rounding_mode="trunc")
                return dynamic.chunked_attention_masks(
                    chunk_frames=chunk, left_context=70,
                    padding_length=70 + chunk.reshape(1),
                    max_audio_length=70 + chunk,
                    offset=70 - cache_length, device=features.device,
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic.onnx"
            torch.onnx.export(
                Masks(), (torch.zeros(1, 128, 65), torch.tensor([5])), str(path),
                input_names=["features", "cache_length"],
                output_names=["pad_mask", "attention_mask"],
                dynamic_axes={"features": {2: "mel_frames"}},
                opset_version=17, dynamo=False,
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            for frames in (1, 2, 7, 14, 7):
                feeds = {"features": np.zeros((1, 128, frames * 8 + 9), np.float32),
                         "cache_length": np.asarray([5], dtype=np.int64)}
                expected = Masks()(torch.from_numpy(feeds["features"]), torch.from_numpy(feeds["cache_length"]))
                actual = session.run(None, feeds)
                for left, right in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(left, right.numpy())
                specialized = Path(directory) / f"specialized-{frames}.onnx"
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.add_free_dimension_override_by_name("mel_frames", frames * 8 + 9)
                options.optimized_model_filepath = str(specialized)
                selected = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                for left, right in zip(selected.run(None, feeds), actual, strict=True):
                    np.testing.assert_array_equal(left, right)
                self.assertFalse(any(node.op_type in ("Shape", "Range", "Div")
                                     for node in onnx.load(specialized).graph.node))


if __name__ == "__main__":
    unittest.main()
