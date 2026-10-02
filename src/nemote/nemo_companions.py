# SPDX-License-Identifier: Apache-2.0
# Silero layout translation follows NVIDIA/NeMo-Speech.cpp conversion/vad.py,
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Verified, small NeMo companions; no PyTorch or client-side ASR export."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import io
import math
from pathlib import Path
import struct
import sys
import tempfile
from urllib.request import urlopen

# Tokenizers from our existing exports of the matching NVIDIA checkpoints.
TOKENIZERS = {
    # Published tokenizer artifacts retain their legacy repository names and pins.
    "english": ("fractalyzer/wordpipe-nemotron-en-compact-dynamic-quint8",
                "eaddc2b3b98d5ea7b66b58154466d9997be5c488", "tokenizer.model", 251056,
                "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9"),
    "multilingual": ("fractalyzer/wordpipe-nemotron-compact-fixed-shape",
                     "53ec4a3530ab309904564eeeb96d5597bc8ee0a0", "tokenizer.model", 406554,
                     "ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291"),
}
SILERO = ("ggml-org/whisper-vad", "9ffd54a1e1ee413ddf265af9913beaf518d1639b",
          "ggml-silero-v6.2.0.bin", 885098,
          "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(asset, destination, progress):
    repo, revision, filename, size, digest = asset
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size == size and sha(destination) == digest:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading NeMo companion: {filename}")
    output = tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False)
    temporary = Path(output.name)
    try:
        actual = hashlib.sha256()
        count = 0
        with output, urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", timeout=60) as reader:
            while chunk := reader.read(65536):
                count += len(chunk)
                if count > size:
                    raise RuntimeError(f"Companion exceeds pinned size: {filename}")
                actual.update(chunk)
                output.write(chunk)
        if count != size or actual.hexdigest() != digest:
            raise RuntimeError(f"Companion checksum/size verification failed: {filename}")
        temporary.replace(destination)
    finally:
        output.close()
        temporary.unlink(missing_ok=True)
    return destination


def silero_gguf(source):
    """Rewrap whisper.cpp's Silero 6.2 weights, following NeMo conversion/vad.py.

    This only widens stored F16 values to F32 and writes a GGUF header. The
    checkpoint is <1 MiB; no training, model export framework, or numpy needed.
    """
    reader = io.BytesIO(Path(source).read_bytes())
    def read(size):
        data = reader.read(size)
        if len(data) != size:
            raise RuntimeError("Truncated Silero checkpoint")
        return data
    def i32():
        return struct.unpack("<i", read(4))[0]
    if i32() != 0x67676D6C:
        raise RuntimeError("Invalid Silero GGML header")
    length = i32()
    if not 0 < length < 100 or read(length).decode() != "silero-16k":
        raise RuntimeError("Unexpected Silero model type")
    if tuple(i32() for _ in range(3)) != (6, 2, 0):
        raise RuntimeError("Expected Silero 6.2.0")
    if (i32(), i32(), i32()) != (512, 64, 4):
        raise RuntimeError("Unexpected Silero frontend geometry")
    for geometry in ((129, 128, 3), (128, 64, 3), (64, 64, 3), (64, 128, 3)):
        if tuple(i32() for _ in range(3)) != geometry:
            raise RuntimeError("Unexpected Silero encoder geometry")
    if tuple(i32() for _ in range(4)) != (128, 128, 128, 1):
        raise RuntimeError("Unexpected Silero decoder geometry")
    mapping = {}
    for layer in range(4):
        for kind in ("weight", "bias"):
            mapping[f"_model.encoder.{layer}.reparam_conv.{kind}"] = f"encoder.{layer}.conv.{kind}"
    for kind in ("weight", "bias"):
        for bank in ("ih", "hh"):
            mapping[f"_model.decoder.rnn.{kind}_{bank}"] = f"lstm.{bank}.{kind}"
        mapping[f"_model.decoder.decoder.2.{kind}"] = f"final_conv.{kind}"
    mapping["_model.stft.forward_basis_buffer"] = "stft.basis"
    tensors = {}
    while reader.tell() < len(reader.getbuffer()):
        rank, length, dtype = struct.unpack("<iii", read(12))
        if not 0 <= rank <= 4 or not 0 < length < 256 or dtype not in (0, 1):
            raise RuntimeError("Invalid Silero tensor header")
        dims = tuple(i32() for _ in range(rank))
        if any(dim <= 0 for dim in dims) or math.prod(dims) > 1000000:
            raise RuntimeError("Invalid Silero tensor shape")
        name = read(length).decode()
        if name not in mapping or mapping[name] in tensors:
            raise RuntimeError(f"Unexpected Silero tensor: {name}")
        data = read(math.prod(dims) * (2 if dtype == 1 else 4))
        if dtype == 1:
            values = array("f", (value for (value,) in struct.iter_unpack("<e", data)))
            if sys.byteorder != "little":
                values.byteswap()
            data = values.tobytes()
        target = mapping[name]
        if not dims:
            dims = (1,)
        if target == "final_conv.weight":
            dims = (128, 1)
        tensors[target] = (dims, data)
    if len(tensors) != len(mapping):
        raise RuntimeError("Incomplete Silero checkpoint")
    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data
    metadata = {
        "general.architecture": "vad", "general.name": "silero-vad-v6.2.0", "general.file_type": 0,
        "vad.version": "6.2.0", "vad.sample_rate": 16000, "vad.window_size": 512,
        "vad.context_size": 64, "vad.n_encoder_layers": 4,
        "vad.encoder.in_channels": [129, 128, 64, 64], "vad.encoder.out_channels": [128, 64, 64, 128],
        "vad.encoder.strides": [1, 2, 2, 1], "vad.encoder.kernel_size": 3,
        "vad.lstm.input_size": 128, "vad.lstm.hidden_size": 128,
        "vad.final_conv.in_channels": 128, "vad.final_conv.out_channels": 1,
        "vad.stft.n_freqs": 129, "vad.stft.n_basis": 258, "vad.stft.filter_length": 256,
    }
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        header.extend(string(key))
        if isinstance(value, str):
            header.extend(struct.pack("<I", 8) + string(value))
        elif isinstance(value, list):
            header.extend(struct.pack("<IIQ", 9, 5, len(value)) + struct.pack("<" + "i" * len(value), *value))
        else:
            header.extend(struct.pack("<II", 4, value))
    payload = bytearray()
    for name, (dims, data) in tensors.items():
        header.extend(string(name) + struct.pack("<I", len(dims)))
        header.extend(struct.pack("<" + "Q" * len(dims), *dims))
        header.extend(struct.pack("<IQ", 0, len(payload)))
        payload.extend(data)
        payload.extend(b"\0" * (-len(payload) % 32))
    header.extend(b"\0" * (-len(header) % 32))
    return bytes(header + payload)


def install(runtime_dir, family, *, boosting=False, vad=False, progress=print):
    runtime_dir = Path(runtime_dir)
    if boosting:
        download(TOKENIZERS[family], runtime_dir / "tokenizer.model", progress)
    if vad:
        source = download(SILERO, runtime_dir / SILERO[2], progress)
        data = silero_gguf(source)
        destination = runtime_dir / "silero-v6.2.0.gguf"
        expected = hashlib.sha256(data).hexdigest()
        if destination.is_file() and sha(destination) == expected:
            return
        output = tempfile.NamedTemporaryFile(dir=runtime_dir, prefix=".silero.", suffix=".part", delete=False)
        temporary = Path(output.name)
        try:
            with output:
                output.write(data)
            temporary.replace(destination)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=tuple(TOKENIZERS), required=True)
    parser.add_argument("--boosting", action="store_true")
    parser.add_argument("--vad", action="store_true")
    args = parser.parse_args()
    try:
        install(args.runtime_dir, args.model_family, boosting=args.boosting, vad=args.vad,
                progress=lambda message: print(message, file=sys.stderr, flush=True))
    except Exception as error:
        print(f"NeMo companion installation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
