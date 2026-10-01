from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class DynamicReleaseTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "model"
        source.mkdir()
        config = {
            "model_family": "english", "projected_cache": True,
            "dynamic_quint8_quantization": True, "num_encoder_layers": 24,
            "hidden_dim": 1024, "conv_context": 8,
            "cache_shapes": {"cache_last_channel": [24, 1, 70, 1024],
                             "cache_last_time": [24, 1, 1024, 8]},
            "dynamic_streaming": {
                "format": 1, "shape_derived_attention_context": True,
                "subsampling_factor": 8, "mel_frames_overhead": 9, "cache_len": 70,
                "default_chunk_frames": 7, "supported_chunk_frames": [1, 2, 7, 14],
            },
        }
        (source / "config.json").write_text(json.dumps(config))
        for name in ("encoder.onnx", "decoder_joint.onnx", "tokenizer.model"):
            (source / name).write_bytes(name.encode())
        parity = {"passed": True, "atol": 1e-4, "rtol": 1e-4, "records": [
            {"chunk_frames": frames, "valid_cache": cache}
            for frames in (1, 2, 7, 14) for cache in (0, 5, 70)]}
        modes = {"passed": True, "results": [
            {"label": "compact", "chunk_frames": size} for size in (1, 2, 7, 14)]}
        corpus = {"summary": {}, "results": [
            {"label": "compact-" + label, "clip": str(i), "text": "same transcript"}
            for i in range(5) for label in ("baseline", "generic")]}
        benchmark = {"status": "complete", "settings": {"interleave": True}, "summaries": [
            {"label": "baseline", "runs": 5, "median_decode_seconds": 1, "max_peak_rss_kib": 1000},
            {"label": "generic", "runs": 5, "median_decode_seconds": 1.04, "max_peak_rss_kib": 900}]}
        for name, data in (("parity", parity), ("modes", modes), ("corpus", corpus), ("benchmark", benchmark)):
            (root / (name + ".json")).write_text(json.dumps(data))
        args = argparse.Namespace(source_dir=source, profile="compact", encoder_parity=root / "parity.json",
                                  projected_parity=None, modes_report=root / "modes.json",
                                  corpus_report=root / "corpus.json", benchmark_report=root / "benchmark.json",
                                  startup_report=root / "benchmark.json")
        return args, config, benchmark

    def test_noise_band_and_artifact_binding(self):
        gate = module("validate_dynamic_nemotron_release")
        publisher = module("publish_wordpipe_model_profiles")
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _ = self.fixture(Path(tmp))
            report = gate.validate(args)
            (args.source_dir / "validation.json").write_text(json.dumps(report))
            spec = publisher.profile_spec("compact")
            publisher.validate_publish_source(args.source_dir, spec, "english")
            publisher.validate_release_evidence(args.source_dir, spec, config["dynamic_streaming"])
            (args.source_dir / "encoder.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(SystemExit, "artifact changed"):
                publisher.validate_release_evidence(args.source_dir, spec, config["dynamic_streaming"])

    def test_regression_above_noise_band_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, benchmark = self.fixture(Path(tmp))
            benchmark["summaries"][1]["median_decode_seconds"] = 1.06
            args.benchmark_report.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "exceeds.*5%"):
                module("validate_dynamic_nemotron_release").validate(args)

    def test_missing_runtime_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            args.modes_report.write_text(json.dumps({"passed": True, "results": []}))
            with self.assertRaisesRegex(ValueError, "all supported modes"):
                module("validate_dynamic_nemotron_release").validate(args)

    def test_dynamic_spec_does_not_require_local_export_or_fixed_shapes(self):
        publisher = module("publish_wordpipe_model_profiles")
        with tempfile.TemporaryDirectory() as tmp:
            _, config, _ = self.fixture(Path(tmp))
            dynamic = config["dynamic_streaming"]
            spec = publisher.render_dynamic_model_spec(publisher.profile_spec("compact"), "english", dynamic)
            self.assertIn("80, 160, 560, 1120 ms", spec)
            self.assertIn("no separate chunk-size exports", spec)
            card = publisher.render_model_card("test/repo", ("compact",), "english", dynamic_streaming=dynamic)
            self.assertIn("local model export are not required", card)
            self.assertNotIn("installer converts it to ORT format", card)


if __name__ == "__main__":
    unittest.main()
