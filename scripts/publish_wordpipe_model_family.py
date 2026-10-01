#!/usr/bin/env python3
"""Build and publish every Wordpipe performance profile for one model family."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_MODEL_ROOT = Path.home() / ".local" / "share" / "wordpipe" / "models"
DEFAULT_RELEASE_ROOT = ROOT / "build" / "model-release"
FAMILY_SOURCES = {
    "english": "nvidia/nemotron-speech-streaming-en-0.6b",
    "multilingual": "nvidia/nemotron-3.5-asr-streaming-0.6b",
}
PROFILES = ("fast", "compact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=tuple(FAMILY_SOURCES), required=True)
    parser.add_argument("--streaming-latency-ms", type=int, choices=(560, 1120), default=560)
    parser.add_argument("--dynamic-streaming", action="store_true",
                        help="Build one generic encoder per profile, without per-mode exports.")
    parser.add_argument("--source", help="Local .nemo path or Hugging Face model id.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-build", action="store_true", help="Publish existing built profiles.")
    parser.add_argument("--upload", action="store_true", help="Upload both packaged profiles to Hugging Face.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--keep-build-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def commands(args: argparse.Namespace) -> list[list[str]]:
    python = str(args.python.expanduser())
    model_root = args.model_root.expanduser()
    source = args.source or FAMILY_SOURCES[args.model_family]
    latency = getattr(args, "streaming_latency_ms", 560)
    dynamic = getattr(args, "dynamic_streaming", False)
    latencies = (latency,) if dynamic else ((560, 1120) if latency == 1120 else (560,))
    result: list[list[str]] = []
    if not args.skip_build:
        for mode in latencies:
            for profile in PROFILES:
                result.append([
                    python, "-m", "wordpipe", "model-install",
                    "--profile", profile, "--model-family", args.model_family,
                    "--model-root", str(model_root), "--build-from-nemo",
                    "--source", source, "--python", python,
                    *(["--dynamic-streaming"] if dynamic else []),
                    *(["--streaming-latency-ms", str(mode)] if mode != 560 else []),
                    *(["--force"] if args.force else []),
                    *(["--force-source"] if args.force_source else []),
                    *(["--keep-build-dir"] if args.keep_build_dir else []),
                    *(["--dry-run"] if args.dry_run else []),
                ])
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from wordpipe.models import profile_spec
    for profile in PROFILES:
        mode_dir = profile_spec(profile).output_dir(model_root / "1120ms", args.model_family)
        result.append([
            python, str(ROOT / "scripts" / "publish_wordpipe_model_profiles.py"),
            "--profile", profile, "--model-family", args.model_family,
            "--model-root", str(model_root),
            *(["--1120ms-dir", str(mode_dir)] if latency == 1120 and not dynamic else []),
            "--output-dir", str(args.release_root.expanduser() / args.model_family / profile),
            "--force-card",
            *(["--force"] if args.force else []),
            *(["--upload"] if args.upload else []),
            *(["--private"] if args.private else []),
            *(["--revision", args.revision] if args.revision else []),
        ])
    return result


def main() -> int:
    args = parse_args()
    env = None
    if SRC.exists():
        import os

        env = os.environ.copy()
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + current if current else "")
    for command in commands(args):
        print("[family-release] " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
