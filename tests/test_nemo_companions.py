import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nemote import nemo_companions as companions


class NemoCompanionTests(unittest.TestCase):
    def test_structured_download_progress_matches_service_protocol(self):
        data = b"GGUFfixture"
        events = []
        with tempfile.TemporaryDirectory() as directory, patch.object(companions, "urlopen", return_value=io.BytesIO(data)):
            companions.download_url("https://fixture.invalid", "fixture.gguf", len(data),
                hashlib.sha256(data).hexdigest(), Path(directory) / "model", lambda _: None,
                download_progress=events.append)
        self.assertEqual([event["fraction"] for event in events], [0.0, 1.0])
        self.assertEqual(events[-1]["completed_bytes"], len(data))
        self.assertEqual(events[-1]["total_bytes"], len(data))

    def test_download_verified_atomic_and_cached_without_network(self):
        data = b"verified tokenizer"
        asset = ("fixture/repo", "pinned", "tokenizer.model", len(data), hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "tokenizer.model"
            with patch.object(companions, "urlopen", return_value=io.BytesIO(data)) as request:
                companions.download(asset, target, lambda _: None)
                self.assertIn("/resolve/pinned/", request.call_args.args[0])
            with patch.object(companions, "urlopen", side_effect=AssertionError("Unexpected network request")):
                companions.download(asset, target, lambda _: None)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(list(target.parent.glob(".*.part")), [])

    def test_failed_download_preserves_existing_file(self):
        asset = ("fixture/repo", "pinned", "model", 3, hashlib.sha256(b"new").hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model"
            target.write_bytes(b"old")
            with patch.object(companions, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    companions.download(asset, target, lambda _: None)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(target.parent.glob(".*.part")), [])

    def test_oversized_download_is_rejected(self):
        asset = ("fixture/repo", "pinned", "model", 3, hashlib.sha256(b"new").hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(companions, "urlopen", return_value=io.BytesIO(b"oversized")):
                with self.assertRaisesRegex(RuntimeError, "exceeds pinned size"):
                    companions.download(asset, Path(directory) / "model", lambda _: None)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_installs_only_requested_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(companions, "download") as download:
                companions.install(root, "english")
                download.assert_not_called()
                companions.install(root, "english", boosting=True)
                self.assertEqual(download.call_args.args[:2], (companions.TOKENIZERS["english"], root / "tokenizer.model"))

    def test_vad_output_is_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(companions, "download", return_value=root / "source"), \
                    patch.object(companions, "silero_gguf", return_value=b"GGUFfixture"):
                companions.install(root, "english", vad=True, progress=lambda _: None)
                target = root / "silero-v6.2.0.gguf"
                before = target.stat().st_mtime_ns
                companions.install(root, "english", vad=True, progress=lambda _: None)
                self.assertEqual(target.stat().st_mtime_ns, before)
                self.assertEqual(target.read_bytes(), b"GGUFfixture")
                self.assertEqual(list(root.glob(".*.part")), [])

    def test_pnc_runs_even_when_vad_is_already_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(companions, "download", return_value=root / "source"), \
                    patch.object(companions, "silero_gguf", return_value=b"GGUFfixture"), \
                    patch("nemote.pnc_install.install") as install_pnc:
                for _ in range(2):
                    companions.install(root, "english", pnc=True, pnc_variant="distilbert", vad=True, progress=lambda _: None)
                self.assertEqual(install_pnc.call_count, 2)
                self.assertEqual(install_pnc.call_args.kwargs["variant"], "distilbert")
