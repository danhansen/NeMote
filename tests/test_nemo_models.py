import hashlib
import io
import json
import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from nemote.cli import build_parser, _cmd_model_install_family
from nemote import nemo_models
from nemote.models import profile_installed, profile_runtime_dir, profile_supports_streaming_latency


class NemoModelTests(unittest.TestCase):
    def network_fixture(self, root, content):
        model = ("fixture/repo", "revision", "model.gguf", len(content), hashlib.sha256(content).hexdigest())
        runtime = profile_runtime_dir(root, "nemo-q8", "english")
        runtime.mkdir(parents=True)
        partial = runtime / f".model.{model[4][:16]}.part"
        return model, runtime, partial

    def response(self, content, status=200, content_range=None):
        response = io.BytesIO(content)
        response.status = status
        response.headers = {"Content-Range": content_range} if content_range else {}
        return response

    def test_network_progress_retries_and_resumes(self):
        content = b"GGUFfixture model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, runtime, partial = self.network_fixture(root, content)
            first = self.response(content[:4])
            original_read = first.read
            def interrupted_read(size):
                data = original_read(size)
                if not data:
                    raise URLError("disconnected")
                return data
            first.read = interrupted_read
            second = self.response(content[4:], 206, f"bytes 4-{len(content)-1}/{len(content)}")
            messages = []
            with patch.dict(nemo_models.MODELS, english=model), \
                 patch.object(nemo_models, "urlopen", side_effect=[first, second]) as download, \
                 patch.object(nemo_models.time, "sleep"):
                nemo_models.install_nemo_model(root, "english", progress=messages.append)
            self.assertEqual(download.call_args.args[0].get_header("Range"), "bytes=4-")
            self.assertEqual((runtime / "model.gguf").read_bytes(), content)
            self.assertFalse(partial.exists())
            events = [json.loads(message.removeprefix("nemote-progress ")) for message in messages]
            self.assertTrue(any(event["phase"] == "retrying" for event in events))
            self.assertTrue(any(0 < event["fraction"] < 1 for event in events))
            self.assertEqual(events[-1]["phase"], "complete")
            self.assertEqual(events[-1]["fraction"], 1)

    def test_saved_partial_with_ignored_range_restarts_safely(self):
        content = b"GGUFfixture model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, runtime, partial = self.network_fixture(root, content)
            partial.write_bytes(content[:4])
            with patch.dict(nemo_models.MODELS, english=model), \
                 patch.object(nemo_models, "urlopen", return_value=self.response(content)):
                nemo_models.install_nemo_model(root, "english", progress=lambda _: None)
            self.assertEqual((runtime / "model.gguf").read_bytes(), content)

    def test_failed_transport_preserves_resumable_partial(self):
        content = b"GGUFfixture model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, runtime, partial = self.network_fixture(root, content)
            partial.write_bytes(content[:4])
            with patch.dict(nemo_models.MODELS, english=model), \
                 patch.object(nemo_models, "urlopen", side_effect=URLError("offline")), \
                 patch.object(nemo_models.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "retry to resume"):
                    nemo_models.install_nemo_model(root, "english", progress=lambda _: None)
            self.assertEqual(partial.read_bytes(), content[:4])
            self.assertFalse((runtime / "model.gguf").exists())

    def test_bad_checksum_or_range_never_replaces_installed_model(self):
        content = b"GGUFfixture model"
        for response_kind in ("checksum", "range"):
            with self.subTest(response=response_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, runtime, partial = self.network_fixture(root, content)
                (runtime / "model.gguf").write_bytes(content)
                response = self.response(bytes(len(content))) if response_kind == "checksum" else self.response(content, 206, "bytes 1-2/3")
                with patch.dict(nemo_models.MODELS, english=model), \
                     patch.object(nemo_models, "urlopen", return_value=response):
                    with self.assertRaises(RuntimeError):
                        nemo_models.install_nemo_model(root, "english", force=True, progress=lambda _: None)
                self.assertEqual((runtime / "model.gguf").read_bytes(), content)
                self.assertFalse(partial.exists())

    def test_disk_space_and_concurrent_download_checks(self):
        content = b"GGUFfixture model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, runtime, _ = self.network_fixture(root, content)
            with patch.dict(nemo_models.MODELS, english=model), \
                 patch.object(nemo_models.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()), \
                 patch.object(nemo_models, "urlopen") as download:
                with self.assertRaisesRegex(RuntimeError, "Not enough disk space"):
                    nemo_models.install_nemo_model(root, "english", progress=lambda _: None)
                download.assert_not_called()
            with (runtime / ".download.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    nemo_models.install_nemo_model(root, "english", progress=lambda _: None)

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
