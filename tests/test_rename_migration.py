"""Upgrade checks confined to temporary directories, never the desktop session."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from nemote.cli import build_parser


class RenameMigrationTests(unittest.TestCase):
    def test_cli_reuses_legacy_models_but_prefers_new_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "wordpipe/models"
            legacy.mkdir(parents=True)
            with patch.dict(os.environ, {"XDG_DATA_HOME": directory}):
                self.assertEqual(build_parser().parse_args(["model-install"]).model_root, str(legacy))
                current = root / "nemote/models"
                current.mkdir(parents=True)
                self.assertEqual(build_parser().parse_args(["model-install"]).model_root, str(current))

    def test_installer_preserves_legacy_data_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config, prefix = (root / name for name in ("data", "config", "libexec"))
            old_config = config / "wordpipe/service.json"
            old_config.parent.mkdir(parents=True)
            settings = {"itn": True, "endpoint_mode": "preview", "worker_path": "/old/bin/wordpipe-nemo-worker"}
            old_config.write_text(json.dumps(settings))
            models = data / "wordpipe/models"
            models.mkdir(parents=True)
            (models / "important-model").write_bytes(b"preserve me")
            legacy_files = [data / "dbus-1/services/dev.wordpipe.Service.service",
                            config / "systemd/user/wordpipe-service.service"]
            for path in legacy_files:
                path.parent.mkdir(parents=True)
                path.write_text("old activation")
            package = root / "package"
            (package / "bin").mkdir(parents=True)
            (package / "lib").mkdir()
            service = package / "bin/nemote-service"
            service.write_text("#!/bin/sh\nexit 0\n")
            service.chmod(0o755)
            for relative in ("dbus-1/services/dev.nemote.Service.service",
                             "systemd/user/nemote-service.service",
                             "gnome-shell/extensions/nemote@dhansen.dev/schemas/dummy"):
                path = package / "share" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("@NEMOTE_LIBEXEC@")
            tools = root / "tools"
            tools.mkdir()
            # No real session bus, GSettings, or schema compilation is touched.
            for tool in ("systemctl", "gnome-extensions", "dconf", "glib-compile-schemas"):
                path = tools / tool
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
            env = dict(os.environ, XDG_DATA_HOME=str(data), XDG_CONFIG_HOME=str(config),
                       NEMOTE_LIBEXEC_DIR=str(prefix), PATH=f"{tools}:{os.environ['PATH']}")
            installer = Path(__file__).resolve().parents[1] / "scripts/install-nemote-release"
            for _ in range(2):
                subprocess.run([str(installer), "--from-extracted", str(package)], env=env,
                               check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((config / "nemote/service.json").read_text()), settings)
            self.assertEqual(json.loads(old_config.read_text()), settings)
            self.assertTrue((data / "nemote/models").is_symlink())
            self.assertEqual((data / "nemote/models/important-model").read_bytes(), b"preserve me")
            for path in legacy_files:
                self.assertFalse(path.exists())
                backups = list(path.parent.glob(path.name + ".retired.*"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), "old activation")
            self.assertIn(str(prefix), (data / "dbus-1/services/dev.nemote.Service.service").read_text())


if __name__ == "__main__":
    unittest.main()
