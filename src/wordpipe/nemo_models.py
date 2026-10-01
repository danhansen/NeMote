"""Download upstream GGUFs; no NeMo/PyTorch export environment is needed."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

from .models import profile_runtime_dir

# Pinned model revisions and LFS hashes, not mutable resolve/main URLs.
MODELS = {
    "english": (
        "nvidia/nemotron-speech-streaming-en-0.6b",
        "ebe59e5a817142986528bbbee5dba8db7b38ed50",
        "nemotron-speech-streaming-en-0.6b.q8_0.gguf",
        699872960,
        "d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d",
    ),
    "multilingual": (
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "ea30d66debe3740a08b573244286791d423d6b3e",
        "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        742090464,
        "3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1",
    ),
}


def install_nemo_model(model_root: Path, family: str, *, source: Path | None = None,
                       force: bool = False, progress=print) -> Path:
    repo, revision, filename, size, expected_hash = MODELS[family]
    destination = profile_runtime_dir(model_root, "nemo-q8", family) / "model.gguf"
    if destination.exists() and not force:
        if destination.stat().st_size == size:
            with destination.open("rb") as existing:
                if hashlib.file_digest(existing, "sha256").hexdigest() == expected_hash:
                    return destination.parent
        raise RuntimeError(f"Invalid installed GGUF at {destination}; use --force to replace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary files keep concurrent installers from touching each other's data.
    output = tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".model.gguf.", suffix=".part", delete=False)
    temporary = Path(output.name)
    progress(f"Installing NVIDIA Q8 GGUF: {filename}")
    try:
        digest = hashlib.sha256()
        count = 0
        reader = source.open("rb") if source else urlopen(
            f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", timeout=60)
        with reader, output:
            while chunk := reader.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count > size:
                    raise RuntimeError("GGUF exceeds expected download size")
        if count != size or digest.hexdigest() != expected_hash:
            raise RuntimeError("NVIDIA GGUF checksum/size verification failed")
        temporary.replace(destination)
    finally:
        output.close()
        temporary.unlink(missing_ok=True)
    return destination.parent
