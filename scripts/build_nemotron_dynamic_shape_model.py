#!/usr/bin/env python3
"""Package a genuinely dynamic encoder without folding its streaming dimensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import onnx


def safe_relative(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # ONNX rejects multiply-linked external data as a security precaution.
    shutil.copy2(source, destination)


def external_files(model) -> set[str]:
    names = {item.value for tensor in model.graph.initializer
             for item in tensor.external_data if item.key == "location"}
    if any(not safe_relative(name) for name in names):
        raise ValueError("unsafe external weight filename")
    return names


def build(source: Path, output: Path):
    config = json.loads((source / "config.json").read_text())
    dynamic = config.get("dynamic_streaming")
    if not isinstance(dynamic, dict) or dynamic.get("format") != 1:
        raise ValueError("source must be exported with --dynamic-streaming")
    sizes = dynamic.get("supported_chunk_frames")
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("unsupported dynamic streaming modes")
    if dynamic.get("shape_derived_attention_context") is not True:
        raise ValueError("encoder must use shape-derived attention context")
    encoder = onnx.load(source / "encoder.onnx", load_external_data=False)
    signal = next(item for item in encoder.graph.input if item.name == "processed_signal")
    if signal.type.tensor_type.shape.dim[2].dim_param != "mel_frames":
        raise ValueError("source encoder has a fixed mel dimension")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {item.key: item.value for item in encoder.metadata_props}
    metadata.update({
        "wordpipe.dynamic_streaming": "1",
        "chunk_size_output_frames": str(dynamic["default_chunk_frames"]),
        "pre_encode_cache": str(dynamic["mel_frames_overhead"]),
    })
    onnx.helper.set_model_props(encoder, metadata)
    shared = external_files(encoder)
    onnx.save(encoder, output / "encoder.onnx")
    decoder = onnx.load(source / "decoder_joint.onnx", load_external_data=False)
    shared.update(external_files(decoder))
    for name in ("decoder_joint.onnx", "tokenizer.model", *sorted(shared)):
        copy_file(source / name, output / name)
    config.pop("fixed_streaming_shapes", None)
    config["shared_weight_files"] = sorted(shared)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    onnx.checker.check_model(str(output / "encoder.onnx"))
    onnx.checker.check_model(str(output / "decoder_joint.onnx"))
    print(f"[dynamic-shape] wrote {output}; one encoder, chunk frames {sizes}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
