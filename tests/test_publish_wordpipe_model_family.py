from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_wordpipe_model_family.py"
    spec = importlib.util.spec_from_file_location("publish_wordpipe_model_family", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublishWordpipeModelFamilyTests(unittest.TestCase):
    def test_english_release_builds_and_publishes_both_profiles(self) -> None:
        module = _load_script()
        args = argparse.Namespace(
            model_family="english",
            source=None,
            model_root=Path("/models"),
            release_root=Path("/release"),
            python=Path("/venv/bin/python"),
            skip_build=False,
            upload=True,
            private=False,
            revision=None,
            force=True,
            force_source=False,
            keep_build_dir=False,
            dry_run=False,
        )

        commands = module.commands(args)

        self.assertEqual(len(commands), 4)
        self.assertEqual(
            [command[command.index("--profile") + 1] for command in commands],
            ["fast", "compact", "fast", "compact"],
        )
        self.assertTrue(all("english" in command for command in commands))
        self.assertTrue(all("--upload" in command for command in commands[2:]))
        self.assertIn("nvidia/nemotron-speech-streaming-en-0.6b", commands[0])


if __name__ == "__main__":
    unittest.main()
