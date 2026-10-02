"""Download upstream GGUFs; no NeMo/PyTorch export environment is needed."""
from __future__ import annotations

import hashlib
import fcntl
import http.client
import json
import shutil
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
    directory = profile_runtime_dir(model_root, "nemo-q8", family)
    directory.mkdir(parents=True, exist_ok=True)
    # A persistent lock inode makes retries safe even with multiple installers.
    with (directory / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A NeMo model download is already running") from exc
        try:
            return _install_locked(model_root, family, source, force, progress)
        except Exception as exc:
            _report(progress, "error", str(exc))
            raise


def _report(progress, phase, message, *, count=0, size=0):
    progress("nemote-progress " + json.dumps({
        "phase": phase, "message": message, "completed_bytes": count,
        "total_bytes": size, "fraction": count / size if size else 0.0,
    }))


def _install_locked(model_root, family, source, force, progress):
    repo, revision, filename, size, expected_hash = MODELS[family]
    destination = profile_runtime_dir(model_root, "nemo-q8", family) / "model.gguf"
    if destination.exists() and not force:
        if destination.stat().st_size == size:
            with destination.open("rb") as existing:
                if hashlib.file_digest(existing, "sha256").hexdigest() == expected_hash:
                    _report(progress, "complete", "NeMo model already installed", count=size, size=size)
                    return destination.parent
        raise RuntimeError(f"Invalid installed GGUF at {destination}; use --force to replace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        temporary = destination.parent / f".model.{expected_hash[:16]}.part"
        _download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
                  temporary, filename, size, expected_hash, progress)
        temporary.replace(destination)
        _report(progress, "complete", "NeMo model installed", count=size, size=size)
        return destination.parent
    if shutil.disk_usage(destination.parent).free < size:
        raise RuntimeError(f"Not enough disk space: need {size / 1024**2:.0f} MiB for the NeMo model")
    # Unique temporary files keep concurrent installers from touching each other's data.
    output = tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".model.gguf.", suffix=".part", delete=False)
    temporary = Path(output.name)
    progress(f"Installing NVIDIA Q8 GGUF: {filename}")
    try:
        digest = hashlib.sha256()
        count = 0
        reader = source.open("rb")
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
    _report(progress, "complete", "NeMo model installed", count=size, size=size)
    return destination.parent


def _download(url, temporary, filename, size, expected_hash, progress):
    # Keep only this pinned artifact's partial file on transport interruptions.
    # Never expose it as model.gguf until the full SHA256 has been verified.
    if temporary.exists() and temporary.stat().st_size > size:
        temporary.unlink()
    for attempt in range(1, 4):
        count = temporary.stat().st_size if temporary.exists() else 0
        if shutil.disk_usage(temporary.parent).free < size - count:
            raise RuntimeError(f"Not enough disk space: need another {(size-count) / 1024**2:.0f} MiB for the NeMo model")
        _report(progress, "downloading", f"Downloading {filename} (attempt {attempt}/3)", count=count, size=size)
        try:
            if count < size:
                headers = {"Accept-Encoding": "identity"}
                if count:
                    headers["Range"] = f"bytes={count}-"
                with urlopen(Request(url, headers=headers), timeout=60) as reader:
                    if reader.status == 206:
                        content_range = reader.headers.get("Content-Range", "")
                        if content_range != f"bytes {count}-{size-1}/{size}":
                            raise RuntimeError("NeMo server returned an invalid download range")
                    elif reader.status == 200:
                        # Servers may ignore Range; restart, never append a full response.
                        count = 0
                        if shutil.disk_usage(temporary.parent).free + (temporary.stat().st_size if temporary.exists() else 0) < size:
                            raise RuntimeError("Not enough disk space to restart the NeMo download")
                    else:
                        raise RuntimeError(f"Unexpected NeMo download response: HTTP {reader.status}")
                    last_report = 0.0
                    with temporary.open("ab" if count else "wb") as output:
                        while chunk := reader.read(1024 * 1024):
                            if count + len(chunk) > size:
                                raise RuntimeError("GGUF exceeds expected download size")
                            output.write(chunk)
                            count += len(chunk)
                            now = time.monotonic()
                            if now - last_report >= 0.5 or count == size:
                                _report(progress, "downloading",
                                        f"Downloading {filename}: {count / size:.0%} ({count / 1024**2:.1f} / {size / 1024**2:.1f} MiB)",
                                        count=count, size=size)
                                last_report = now
                if count != size:
                    raise ConnectionError("NeMo download ended before all bytes arrived")
            _report(progress, "verifying", "Verifying NeMo model checksum", count=size, size=size)
            with temporary.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected_hash:
                    raise RuntimeError("NVIDIA GGUF checksum/size verification failed")
            return
        except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise RuntimeError("NeMo download interrupted after 3 attempts; retry to resume the saved download") from exc
            _report(progress, "retrying", f"Download interrupted; retrying ({attempt}/3): {exc}", count=count, size=size)
            time.sleep(attempt)
        except RuntimeError:
            temporary.unlink(missing_ok=True)
            raise
