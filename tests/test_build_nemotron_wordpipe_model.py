from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


def _load_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_nemotron_wordpipe_model.py"
    spec = importlib.util.spec_from_file_location("build_nemotron_wordpipe_model", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "profile": "compact-fixed-shape",
        "start_at": "export",
        "stop_after": None,
        "left_context": 56,
        "right_context": 6,
        "sample_rate": 16000,
        "input_frames": 65,
        "output_frames": 7,
        "num_layers": 24,
        "cache_len": 56,
        "hidden_dim": 1024,
        "conv_context": 8,
        "ort_optimize_threads": 1,
        "fp32_decoder": False,
        "quantize_per_channel": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class NemotronWordpipePipelineTests(unittest.TestCase):
    def test_only_dynamic_quantized_builds_lower_and_validate_pointwise_projections(self):
        builder = _load_builder()
        for dynamic, profile in ((True, "compact-fixed-shape"), (True, "fp32-projected"), (False, "compact-fixed-shape")):
            argv = ["build", "source.nemo", "/tmp/wordpipe-dry-run-only/output",
                    "--work-dir", "/tmp/wordpipe-dry-run-only/work", "--profile", profile,
                    "--model-family", "english", "--dry-run", "--stop-after", "transform"]
            if dynamic:
                argv.append("--dynamic-streaming")
            with patch.object(sys, "argv", argv), patch.object(builder, "run") as run:
                builder.main()
            transform = next(call.args[0] for call in run.call_args_list
                             if any("transform_nemotron_parakeet_export.py" in part for part in call.args[0]))
            expected = dynamic and profile != "fp32-projected"
            self.assertEqual("--pointwise-linear" in transform, expected)
            self.assertEqual("--pointwise-parity-output" in transform, expected)

    def test_dynamic_mode_accepts_small_chunks_and_zero_right_context(self) -> None:
        builder = _load_builder()
        for latency, shapes in ((80, (0, 17, 1)), (160, (1, 25, 2))):
            args = _args(dynamic_streaming=True, streaming_latency_ms=latency,
                         right_context=None, input_frames=None, output_frames=None)
            builder.validate_args(args)
            self.assertEqual((args.right_context, args.input_frames, args.output_frames), shapes)
        with self.assertRaisesRegex(SystemExit, "positive multiple"):
            builder.validate_args(_args(dynamic_streaming=True, streaming_latency_ms=81))

    def test_dynamic_mode_rejects_static_ort_conversion(self) -> None:
        with self.assertRaisesRegex(SystemExit, "dynamic ONNX"):
            _load_builder().validate_args(_args(dynamic_streaming=True, emit_ort_format=True))

    def test_1120ms_derives_shapes_and_rejects_560ms_shapes(self) -> None:
        builder = _load_builder()
        args = _args(streaming_latency_ms=1120, right_context=None,
                     input_frames=None, output_frames=None)
        builder.validate_args(args)
        self.assertEqual((args.right_context, args.input_frames, args.output_frames),
                         (13, 121, 14))
        with self.assertRaisesRegex(SystemExit, "must be 13"):
            builder.validate_args(_args(streaming_latency_ms=1120))

    def test_validate_args_rejects_non_positive_shape_values(self) -> None:
        builder = _load_builder()

        with self.assertRaisesRegex(SystemExit, "--input-frames must be positive"):
            builder.validate_args(_args(input_frames=0))

    def test_validate_args_rejects_non_positive_ort_threads(self) -> None:
        builder = _load_builder()

        with self.assertRaisesRegex(SystemExit, "--ort-optimize-threads must be positive"):
            builder.validate_args(_args(ort_optimize_threads=0))

    def test_validate_args_defaults_stop_after_to_profile_final_phase(self) -> None:
        builder = _load_builder()
        args = _args(profile="compact-fixed-shape", stop_after=None)

        builder.validate_args(args)

        self.assertEqual(args.stop_after, "fixed-shape")

    def test_validate_args_rejects_phase_outside_profile(self) -> None:
        builder = _load_builder()

        with self.assertRaisesRegex(SystemExit, "not part of profile"):
            builder.validate_args(_args(profile="compact-fixed-shape", stop_after="ffn-fp32"))

    def test_english_family_defaults_to_c70_cache(self) -> None:
        builder = _load_builder()
        args = _args(
            model_family="english",
            left_context=None,
            cache_len=None,
        )

        builder.validate_args(args)

        self.assertEqual(args.left_context, 70)
        self.assertEqual(args.cache_len, 70)


if __name__ == "__main__":
    unittest.main()
