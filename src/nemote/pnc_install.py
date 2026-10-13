# SPDX-License-Identifier: Apache-2.0
"""One-time, local PnC conversion; never redistribute NVIDIA checkpoint weights."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .nemo_companions import download_url, sha

SDK_REVISION = "4c101bc7113f49101a3e11d2c994c519f41939f6"
MODEL_FILENAME = "pnc-bert-en.q8_0.gguf"
CHECKPOINT_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_bert/versions/1.0.0rc1/files/punctuation_en_bert.nemo"
CHECKPOINT_SIZE = 405906986
CHECKPOINT_SHA = "d69e87388e9162f86840cadc2d4efee45588de4eaf92b542d994f0eccd73e7ae"
ENCODER_URL = "https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f7235943/config.json"
ENCODER_SIZE = 570
# Filled from the pinned BERT configuration, not from a moving branch.
ENCODER_SHA = "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20"
OUTPUT_SIZE = 138875840
MODELS = {
    "bert": (MODEL_FILENAME, CHECKPOINT_URL, CHECKPOINT_SIZE, CHECKPOINT_SHA, OUTPUT_SIZE),
    "distilbert": ("pnc-distilbert-en.q8_0.gguf", CHECKPOINT_URL.replace("punctuation_en_bert", "punctuation_en_distilbert"),
                   245117658, "1d60dff59c927ad7a5c266f915b1e5710332d224bdbd39fb08f1335e39dbe89a", 93507552),
}
DEPENDENCIES = {"torch": "2.14.1+cpu", "numpy": "2.5.4", "gguf": "0.19.0", "PyYAML": "6.0.3"}
YAML_WHEEL = "pyyaml-6.0.3-py3-none-any.whl"


def converter_directory():
    # Installed layout: PREFIX/lib/python/nemote; development: ROOT/src/nemote.
    prefix = Path(__file__).resolve().parents[3]
    packaged = prefix / "share/nemote/pnc-converter"
    if packaged.is_dir():
        return packaged
    root = Path(__file__).resolve().parents[2]
    return root / "build/pnc-converter"


def check_resources(runtime_dir, cache):
    # Measured converter RSS ~783 MiB, plus headroom. ASR has not loaded yet.
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        available = next((int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                          if line.startswith("MemAvailable:")), None)
        if available is not None and available < 1280 * 1024**2:
            raise RuntimeError("PnC conversion needs about 1.25 GiB available RAM; free memory and retry, or supply a converted PnC model override")
    # Check both filesystems: the venv remains cached; conversion scratch is on
    # the model filesystem and needs checkpoint + adapted/extracted copies.
    for directory, minimum in ((runtime_dir, 2 * 1024**3), (cache, 2 * 1024**3)):
        if shutil.disk_usage(directory).free < minimum:
            raise RuntimeError(f"PnC setup needs at least 2 GiB free disk space at {directory}")


def conversion_python(cache, progress):
    override = os.environ.get("NEMOTE_PNC_PYTHON")
    probe = "from importlib.metadata import version; import torch,numpy,gguf,yaml; " + \
            "assert all(version(k)==v for k,v in " + repr(DEPENDENCIES) + ".items())"
    if override:
        subprocess.run([override, "-I", "-c", probe], check=True, stdout=sys.stderr)
        return override
    environment = cache / f"venv-{sys.version_info.major}.{sys.version_info.minor}-v1"
    python = environment / "bin/python"
    if python.is_file():
        if subprocess.run([str(python), "-I", "-c", probe], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
            return str(python)
    progress("Preparing private PnC conversion environment (CPU-only; first setup downloads Python dependencies)")
    try:
        subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True, stdout=sys.stderr)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Could not create the PnC Python environment. Install your distribution's Python venv/ensurepip support (python3-venv on Debian/Ubuntu), then retry") from error
    base = [str(python), "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
            "--no-cache-dir", "--only-binary=:all:", "--timeout", "60", "--retries", "3"]
    # Fetch only the CPU torch wheel from its dedicated index. That index may
    # lack binary wheels for transitive deps on new Python releases. Resolve
    # those from PyPI with the installed +cpu version pinned (no CUDA fallback).
    subprocess.run(base + ["--no-deps", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==" + DEPENDENCIES["torch"]],
                   check=True, stdout=sys.stderr)
    subprocess.run(base + ["--index-url", "https://pypi.org/simple"] +
                   [name + "==" + version for name, version in DEPENDENCIES.items() if name != "PyYAML"] +
                   [str(converter_directory() / YAML_WHEEL)],
                   check=True, stdout=sys.stderr)
    subprocess.run([str(python), "-I", "-c", probe], check=True, stdout=sys.stderr)
    return str(python)


def cached(destination, receipt, model):
    try:
        info = json.loads(receipt.read_text())
        return (isinstance(info, dict) and info.get("sdk_revision") == SDK_REVISION and info.get("checkpoint_sha256") == model[3]
                and info.get("dependencies") == DEPENDENCIES and destination.stat().st_size == model[4]
                and info.get("gguf_sha256") == sha(destination))
    except (OSError, ValueError, TypeError):
        return False


def install(runtime_dir, *, variant="bert", progress=print):
    if variant not in MODELS:
        raise ValueError("PnC variant must be bert or distilbert")
    model = MODELS[variant]
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "nemote/pnc-export"
    cache.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / model[0]
    receipt = runtime_dir / (model[0] + ".json")
    # Serialize dependency setup and publication, even across ASR model families.
    with (cache / "install.lock").open("a") as lock:
        progress("Checking cached English PnC model")
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cached(destination, receipt, model):
            return destination
        converter = converter_directory()
        required = ("export_nemo_pnc.py", "convert_model.py", "REVISION", "conversion/pnc.py", YAML_WHEEL)
        if any(not (converter / name).is_file() for name in required) or (converter / "REVISION").read_text().strip() != SDK_REVISION:
            raise RuntimeError("Pinned PnC converter is missing. Reinstall NeMote; source builds should run scripts/package_pnc_converter.py")
        check_resources(runtime_dir, cache)
        python = conversion_python(cache, progress)
        check_resources(runtime_dir, cache)
        progress("PnC weights download directly from NVIDIA NGC; their model terms apply: https://ngc.nvidia.com/legal/terms")
        # Temporary source and intermediate checkpoints are not retained. Keep
        # all scratch on the checked filesystem, including upstream's tmpdir.
        with tempfile.TemporaryDirectory(dir=runtime_dir, prefix=".pnc-export-") as scratch:
            directory = Path(scratch)
            checkpoint = download_url(model[1], f"punctuation_en_{variant}.nemo", model[2],
                                      model[3], directory / "source.nemo", progress)
            encoder = download_url(ENCODER_URL, "encoder_config.json", ENCODER_SIZE,
                                   ENCODER_SHA, directory / "encoder.json", progress)
            output = directory / model[0]
            progress(f"Converting English {variant} PnC to Q8 GGUF locally (approximately 0.6–0.8 GiB peak RAM)")
            env = dict(os.environ, TMPDIR=scratch, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            # Only the checksum-pinned NVIDIA checkpoint reaches torch.load.
            subprocess.run([python, "-I", str(converter / "export_nemo_pnc.py"), str(checkpoint),
                            "--sdk", str(converter), "--encoder-config", str(encoder), "--outfile", str(output)],
                           check=True, env=env, stdout=sys.stderr)
            with output.open("rb") as handle:
                magic = handle.read(4)
            if output.stat().st_size != model[4] or magic != b"GGUF":
                raise RuntimeError("PnC conversion did not produce the expected GGUF")
            info = {"sdk_revision": SDK_REVISION, "checkpoint_sha256": model[3],
                    "dependencies": DEPENDENCIES, "gguf_sha256": sha(output)}
            temporary_receipt = directory / "receipt.json"
            temporary_receipt.write_text(json.dumps(info) + "\n")
            # Filename is embedded in GGUF metadata, so use a local receipt
            # instead of imposing a developer export's byte-identical hash.
            output.replace(destination)
            temporary_receipt.replace(receipt)
        progress("English PnC ready (converted locally; subsequent starts reuse it)")
        return destination
