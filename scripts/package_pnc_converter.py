#!/usr/bin/env python3
"""Bundle the unmodified, pinned SDK converter (code only, never model weights)."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "4c101bc7113f49101a3e11d2c994c519f41939f6"
FILES = ("convert_model.py", "conversion/__init__.py", "conversion/registry.py",
         "conversion/source.py", "conversion/quantization.py", "conversion/pnc.py")
YAML_WHEEL = "pyyaml-6.0.3-py3-none-any.whl"


def package(source, destination):
    source, destination = Path(source), Path(destination)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"PnC converter must match SDK revision {REVISION}")
    # Don't inadvertently ship developer modifications in the upstream sources.
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", *FILES], check=True)
    for name in (*FILES, "LICENSE", "NOTICE"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(Path(__file__).with_name("export_nemo_pnc.py"), destination / "export_nemo_pnc.py")
    # PyYAML's optional C extension is unnecessary (safe_load uses Python).
    # Build this tiny universal wheel on the release builder, never on clients.
    cache = Path(__file__).resolve().parents[1] / "build/pnc-wheels"
    cache.mkdir(parents=True, exist_ok=True)
    wheel = cache / YAML_WHEEL
    if not wheel.is_file():
        subprocess.run([sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "wheel",
                        "--no-deps", "--no-cache-dir", "--no-binary=PyYAML", "--index-url", "https://pypi.org/simple",
                        "--wheel-dir", str(cache), "PyYAML==6.0.3"],
                       check=True, env=dict(os.environ, PYYAML_FORCE_LIBYAML="0"))
    if not wheel.is_file():
        raise RuntimeError("Expected a pure-Python PyYAML wheel; client compilation must not be required")
    shutil.copy2(wheel, destination / YAML_WHEEL)
    (destination / "REVISION").write_text(REVISION + "\n")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=root / "build/nemo-speech-cpp")
    parser.add_argument("--output", type=Path, default=root / "build/pnc-converter")
    args = parser.parse_args()
    package(args.sdk, args.output)


if __name__ == "__main__":
    main()
