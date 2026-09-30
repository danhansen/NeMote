#!/usr/bin/env python3
"""Package small per-mode encoder graphs against one shared set of weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import onnx
from onnx import numpy_helper
from onnx.external_data_helper import ExternalDataInfo, set_external_data


SHARED_THRESHOLD = 64 * 1024


def payload(tensor: onnx.TensorProto, directory: Path) -> bytes:
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        info = ExternalDataInfo(tensor)
        location = Path(info.location)
        if location.is_absolute() or ".." in location.parts:
            raise ValueError(f"unsafe external tensor path: {location}")
        with (directory / location).open("rb") as handle:
            handle.seek(info.offset or 0)
            return handle.read(info.length) if info.length is not None else handle.read()
    return numpy_helper.to_array(tensor).tobytes()


def tensor_key(tensor: onnx.TensorProto, data: bytes) -> tuple:
    return tensor.data_type, tuple(tensor.dims), hashlib.sha256(data).hexdigest()


def reference(tensor: onnx.TensorProto, location: str, offset: int, length: int) -> None:
    # set_external_data requires raw_data to exist before setting the reference.
    tensor.raw_data = b""
    set_external_data(tensor, location=location, offset=offset, length=length)
    for field in ("raw_data", "float_data", "int32_data", "int64_data",
                  "double_data", "uint64_data", "string_data"):
        tensor.ClearField(field)


def append_mode(base: Path, source: Path, latency_ms: int) -> dict:
    """base is a prepared local publication folder; source is another export."""
    if latency_ms != 1120:
        raise ValueError("additional mode must be 1120 ms")
    config = json.loads((source / "config.json").read_text())
    base_config = json.loads((base / "config.json").read_text())
    if config.get("right_context") != 13:
        raise ValueError("additional encoder must have 1120 ms right context")
    for key in ("model_family", "dynamic_quint8_quantization"):
        if config.get(key) != base_config.get(key):
            raise ValueError(f"streaming modes differ in {key}")
    for name in ("tokenizer.model", "decoder_joint.onnx", "decoder_joint.onnx.data"):
        left, right = base / name, source / name
        if left.exists() != right.exists():
            raise ValueError(f"streaming modes differ in {name}")
        if left.exists() and digest(left) != digest(right):
            raise ValueError(f"streaming modes differ in shared {name}")

    baseline = onnx.load(base / "encoder.onnx", load_external_data=False)
    candidate = onnx.load(source / "encoder.onnx", load_external_data=False)
    shared: dict[tuple, tuple[str, int, int]] = {}
    base_large_shapes: set[tuple] = set()
    extra_path = base / "encoder.shared-weights.data"
    extra = None
    mode_constants = None
    try:
        for tensor in baseline.graph.initializer:
            data = payload(tensor, base)
            key = tensor_key(tensor, data)
            if len(data) >= SHARED_THRESHOLD:
                base_large_shapes.add((tensor.data_type, tuple(tensor.dims)))
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                info = ExternalDataInfo(tensor)
                shared[key] = (info.location, info.offset or 0, len(data))
            elif len(data) >= SHARED_THRESHOLD:
                if extra is None:
                    extra = extra_path.open("wb")
                location = (extra_path.name, extra.tell(), len(data))
                extra.write(data)
                shared[key] = location
                reference(tensor, *location)
        for tensor in candidate.graph.initializer:
            data = payload(tensor, source)
            location = shared.get(tensor_key(tensor, data))
            if location is not None:
                reference(tensor, *location)
            elif len(data) >= SHARED_THRESHOLD:
                if (tensor.data_type, tuple(tensor.dims)) in base_large_shapes:
                    raise ValueError(f"large tensor {tensor.name} differs; refusing duplicated model weights")
                # Shape-dependent folded positional data can legitimately
                # differ. Keep it separate from shared learned weights.
                if mode_constants is None:
                    mode_constants = (base / f"encoder.{latency_ms}ms.constants.data").open("wb")
                location = (Path(mode_constants.name).name, mode_constants.tell(), len(data))
                mode_constants.write(data)
                reference(tensor, *location)
            else:
                # Small mode-specific constants may remain inline.
                tensor.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(tensor, base_dir=str(source)).copy(), tensor.name))
    finally:
        if extra is not None:
            extra.close()
        if mode_constants is not None:
            mode_constants.close()

    onnx.save(baseline, base / "encoder.onnx")
    graph_name = f"encoder.{latency_ms}ms.onnx"
    config_name = f"config.{latency_ms}ms.json"
    onnx.save(candidate, base / graph_name)
    onnx.checker.check_model(str(base / "encoder.onnx"))
    onnx.checker.check_model(str(base / graph_name))
    extra_files = sorted({entry.value for tensor in candidate.graph.initializer
                          for entry in tensor.external_data if entry.key == "location"})
    config["shared_weight_files"] = extra_files
    (base / config_name).write_text(json.dumps(config, indent=2) + "\n")
    base_config["shared_weight_files"] = sorted({
        entry.value for tensor in baseline.graph.initializer
        for entry in tensor.external_data if entry.key == "location"
    })
    (base / "config.json").write_text(json.dumps(base_config, indent=2) + "\n")
    manifest = {
        "format": 1,
        "modes": {
            "560": {"encoder": "encoder.onnx", "config": "config.json"},
            str(latency_ms): {"encoder": graph_name, "config": config_name},
        },
        "shared_files": ["tokenizer.model", "decoder_joint.onnx",
                         *([ "decoder_joint.onnx.data"] if (base / "decoder_joint.onnx.data").exists() else []),
                         *extra_files],
    }
    names = {graph_name, config_name, *manifest["shared_files"]}
    manifest["files"] = {
        name: {"size": (base / name).stat().st_size, "sha256": digest(base / name)}
        for name in sorted(names)
    }
    (base / "wordpipe-streaming-bundle.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--1120ms-dir", dest="mode_dir", type=Path, required=True)
    args = parser.parse_args()
    append_mode(args.publication_dir, args.mode_dir, 1120)


if __name__ == "__main__":
    main()
