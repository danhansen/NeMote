#!/usr/bin/env python3
"""Offline companion-installer fixture for service installation lifecycle tests."""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--runtime-dir", type=Path, required=True)
parser.add_argument("--model-family", required=True)
parser.add_argument("--pnc", action="store_true")
parser.add_argument("--pnc-variant", choices=("bert", "distilbert"), required=True)
parser.add_argument("--install-progress", action="store_true")
args = parser.parse_args()
assert args.pnc and args.install_progress and args.model_family == "english"

def progress(phase, **fields):
    print("nemote-progress " + json.dumps({"phase": phase, **fields}), flush=True)

progress("downloading", message="Fixture download", fraction=0.5, completed_bytes=10, total_bytes=20)
time.sleep(0.1)
if (args.runtime_dir.parent / ".fail-pnc-install").exists():
    progress("error", message="Fixture download failed; retry installation")
    raise SystemExit(1)
progress("converting", message="Fixture conversion")
args.runtime_dir.mkdir(parents=True, exist_ok=True)
path = args.runtime_dir / f"pnc-{args.pnc_variant}-en.q8_0.gguf"
with path.open("wb") as output:
    output.write(b"GGUF")
    output.truncate(138875840 if args.pnc_variant == "bert" else 93507552)
progress("complete", message="Fixture model installed", fraction=1.0)
