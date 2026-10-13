import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from nemote import pnc_install as pnc
from nemote import nemo_companions as companions


class PncInstallTests(unittest.TestCase):
    def test_explicit_cli_uses_install_progress_not_worker_loading_events(self):
        argv = ["nemote-companion-install", "--runtime-dir", "/unused", "--model-family", "english",
                "--pnc", "--pnc-variant", "distilbert", "--install-progress"]
        for error in (None, RuntimeError("Fixture conversion failure")):
            with self.subTest(error=error), patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as out, \
                    patch("sys.stderr", new_callable=io.StringIO), patch.object(companions, "install", side_effect=error) as install:
                result = companions.main()
                event = json.loads(out.getvalue().removeprefix("nemote-progress "))
                self.assertEqual(event["phase"], "error" if error else "complete")
                self.assertEqual(result, 1 if error else 0)
                self.assertEqual(install.call_args.kwargs["pnc_variant"], "distilbert")
                self.assertNotIn("loading_model", out.getvalue())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = patch.dict(os.environ, XDG_CACHE_HOME=str(self.root / "cache"))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sdk = self.root / "converter"
        for name in ("export_nemo_pnc.py", "convert_model.py", "conversion/pnc.py", pnc.YAML_WHEEL):
            target = self.sdk / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture")
        (self.sdk / "REVISION").write_text(pnc.SDK_REVISION)
        self.model = ("pnc-fixture.gguf", "https://fixture.invalid/model", 3, "source-digest", 16)
        self.data = b"GGUF" + b"\0" * 12

    def convert(self, command, **kwargs):
        output = Path(command[command.index("--outfile") + 1])
        output.write_bytes(self.data)
        self.assertEqual(kwargs["env"]["TMPDIR"], str(output.parent))
        self.assertNotIn("PYTHONPATH", kwargs["env"])

    def test_atomic_conversion_then_offline_cache_without_dependencies(self):
        messages = []
        models = self.root / "models"
        with patch.dict(pnc.MODELS, bert=self.model), patch.object(pnc, "converter_directory", return_value=self.sdk), \
                patch.object(pnc, "check_resources"), patch.object(pnc, "conversion_python", return_value="fixture-python"), \
                patch.object(pnc, "download_url", side_effect=lambda *args, **kwargs: args[-2]) as download, \
                patch.object(pnc.subprocess, "run", side_effect=self.convert) as convert:
            events = []
            result = pnc.install(models, progress=messages.append, install_progress=events.append)
            self.assertEqual([event["phase"] for event in events], ["checking", "preparing", "converting", "verifying"])
            self.assertEqual(download.call_args_list[0].kwargs["download_progress"], events.append)
            self.assertEqual(result.read_bytes(), self.data)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(download.call_args_list[0].args[0], self.model[1])
            self.assertEqual(convert.call_count, 1)
            receipt = json.loads((models / (self.model[0] + ".json")).read_text())
            self.assertEqual(receipt["gguf_sha256"], hashlib.sha256(self.data).hexdigest())
            self.assertEqual(list(models.glob(".pnc-export-*")), [])
            with patch.object(pnc, "conversion_python", side_effect=AssertionError("Unexpected environment setup")), \
                    patch.object(pnc, "download_url", side_effect=AssertionError("Unexpected network")):
                self.assertEqual(pnc.install(models, progress=messages.append), result)
            result.write_bytes(b"bad")
            pnc.install(models, progress=messages.append)
            self.assertEqual(result.read_bytes(), self.data)

    def test_failed_download_preserves_previous_model_and_cleans_scratch(self):
        models = self.root / "models"
        models.mkdir()
        previous = models / self.model[0]
        previous.write_bytes(b"previous")
        with patch.dict(pnc.MODELS, bert=self.model), patch.object(pnc, "converter_directory", return_value=self.sdk), \
                patch.object(pnc, "check_resources"), patch.object(pnc, "conversion_python", return_value="fixture-python"), \
                patch.object(pnc, "download_url", side_effect=RuntimeError("network failed")):
            with self.assertRaisesRegex(RuntimeError, "network failed"):
                pnc.install(models, progress=lambda _: None)
        self.assertEqual(previous.read_bytes(), b"previous")
        self.assertEqual(list(models.glob(".pnc-export-*")), [])

    def test_failed_conversion_preserves_previous_model(self):
        models = self.root / "models"
        models.mkdir()
        (models / self.model[0]).write_bytes(b"previous")
        with patch.dict(pnc.MODELS, bert=self.model), patch.object(pnc, "converter_directory", return_value=self.sdk), \
                patch.object(pnc, "check_resources"), patch.object(pnc, "conversion_python", return_value="fixture-python"), \
                patch.object(pnc, "download_url", side_effect=lambda *args, **kwargs: args[-2]), \
                patch.object(pnc.subprocess, "run", side_effect=RuntimeError("conversion failed")):
            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                pnc.install(models, progress=lambda _: None)
        self.assertEqual((models / self.model[0]).read_bytes(), b"previous")
        self.assertEqual(list(models.glob(".pnc-export-*")), [])

    def test_invalid_variant_and_insufficient_resources_fail_before_download(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            pnc.install(self.root, variant="unknown")
        with patch.object(pnc.Path, "read_text", return_value="MemAvailable: 1024 kB\n"):
            with self.assertRaisesRegex(RuntimeError, "available RAM"):
                pnc.check_resources(self.root, self.root)
        with patch.object(pnc.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "disk space"):
                pnc.check_resources(self.root, self.root)

    def test_both_variants_are_pinned_and_separately_cached(self):
        self.assertNotEqual(pnc.MODELS["bert"][0], pnc.MODELS["distilbert"][0])
        for model in pnc.MODELS.values():
            self.assertIn("/versions/1.0.0rc1/files/", model[1])
            self.assertEqual(len(model[3]), 64)

    def test_cpu_only_dependency_setup_is_isolated(self):
        with patch.dict(os.environ, {"NEMOTE_PNC_PYTHON": ""}), patch.object(pnc.subprocess, "run") as run:
            pnc.conversion_python(self.root, lambda _: None)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(all("-I" in command for command in commands))
        cpu = next(command for command in commands if "https://download.pytorch.org/whl/cpu" in command)
        self.assertIn("--isolated", cpu)
        self.assertIn("--no-deps", cpu)
        self.assertIn("--only-binary=:all:", cpu)
        self.assertNotIn("--extra-index-url", cpu)
        pypi = next(command for command in commands if "https://pypi.org/simple" in command)
        self.assertIn("torch==" + pnc.DEPENDENCIES["torch"], pypi)


if __name__ == "__main__":
    unittest.main()
