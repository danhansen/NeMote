import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wordpipe.cli import build_parser, _cmd_model_install_family
from wordpipe import nemo_models
from wordpipe.models import profile_installed, profile_runtime_dir, profile_supports_streaming_latency


class NemoModelTests(unittest.TestCase):
    def test_verified_atomic_local_install_and_modes(self):
        content = b"GGUF" + b"fixture model"
        model = ("fixture/repo", "revision", "model.gguf", len(content), hashlib.sha256(content).hexdigest())
        with tempfile.TemporaryDirectory() as directory, patch.dict(nemo_models.MODELS, english=model):
            root = Path(directory)
            source = root / "source.gguf"
            source.write_bytes(content)
            runtime = nemo_models.install_nemo_model(root, "english", source=source, progress=lambda _: None)
            self.assertEqual(runtime, profile_runtime_dir(root, "nemo-q8", "english"))
            self.assertTrue(profile_installed(root, "nemo-q8", "english"))
            for latency in (80, 160, 560, 1120):
                self.assertTrue(profile_supports_streaming_latency(runtime, latency))
            self.assertFalse(profile_supports_streaming_latency(runtime, 240))
            self.assertEqual(nemo_models.install_nemo_model(root, "english"), runtime)
            self.assertEqual(list(runtime.glob("*.part")), [])

    def test_failed_replacement_preserves_good_model(self):
        content = b"GGUFfixture"
        model = ("fixture/repo", "revision", "model.gguf", len(content), hashlib.sha256(content).hexdigest())
        with tempfile.TemporaryDirectory() as directory, patch.dict(nemo_models.MODELS, english=model):
            root = Path(directory)
            source = root / "source.gguf"
            source.write_bytes(content)
            runtime = nemo_models.install_nemo_model(root, "english", source=source, progress=lambda _: None)
            source.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                nemo_models.install_nemo_model(root, "english", source=source, force=True, progress=lambda _: None)
            self.assertEqual((runtime / "model.gguf").read_bytes(), content)
            self.assertEqual(list(runtime.glob(".*.part")), [])

    def test_cli_routes_gguf_without_export(self):
        args = build_parser().parse_args(["model-install", "--profile", "nemo-q8", "--model-family", "english",
                                         "--streaming-latency-ms", "80", "--model-root", "/tmp/models"])
        with patch.object(nemo_models, "install_nemo_model", return_value=Path("/tmp/models/nemotron-nemo-en-q8")) as install:
            self.assertEqual(_cmd_model_install_family(args, "english"), 0)
            self.assertEqual(install.call_args.args, (Path("/tmp/models"), "english"))
