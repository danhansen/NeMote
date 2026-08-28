#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wordpipe.models import (  # noqa: E402
    MODEL_PROFILES,
    ModelProfileSpec,
    model_runtime_dir_valid,
    profile_spec,
)


DEFAULT_OUTPUT_DIR = ROOT / "build" / "model-release"
REQUIRED_ONNX_FILES = ("tokenizer.model", "encoder.onnx", "decoder_joint.onnx")
OPTIONAL_PROFILE_FILES = (
    "encoder.onnx.data",
    "decoder_joint.onnx.data",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
REPRODUCIBILITY_SCRIPTS = (
    "build_nemotron_wordpipe_model.py",
    "export_nemotron_parakeet_optimized.py",
    "transform_nemotron_parakeet_export.py",
    "rewrite_nemotron_projected_kv_cache.py",
    "build_nemotron_fixed_shape_model.py",
    "convert_nemotron_to_ort_format.py",
)


def main() -> int:
    args = parse_args()
    profiles = tuple(dict.fromkeys(args.profile or ("fast",)))
    if len(profiles) != 1:
        raise SystemExit(
            "Publish one profile per Hugging Face model repo. Run this script once for fast and once for compact."
        )
    spec = profile_spec(profiles[0])
    family = args.model_family
    repo_id = args.repo_id or spec.prebuilt_repo_for_family(family)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "repo_id": repo_id,
        "source_model": source_model_for_family(family),
        "model_family": family,
        "model_spec": "MODEL_SPEC.md",
        "reproducibility_scripts": [f"scripts/{name}" for name in REPRODUCIBILITY_SCRIPTS],
        "profiles": {},
    }

    for profile_name in profiles:
        spec = profile_spec(profile_name)
        source = resolve_profile_source(args, spec, family)
        validate_publish_source(source, spec, family)
        copied = copy_profile_files(
            source,
            output_dir,
            force=args.force,
        )
        manifest["profiles"][profile_name] = profile_metadata(copied, source, spec, family)
        print(f"{profile_name}: {output_dir}")

    manifest_path = output_dir / "wordpipe-model-profiles-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")

    readme_path = output_dir / "README.md"
    if not readme_path.exists() or args.force_card:
        readme_path.write_text(render_model_card(repo_id, profiles, family), encoding="utf-8")
        print(f"model card: {readme_path}")

    model_spec_path = output_dir / "MODEL_SPEC.md"
    if not model_spec_path.exists() or args.force_card:
        model_spec_path.write_text(render_model_spec(profiles, family), encoding="utf-8")
        print(f"model spec: {model_spec_path}")

    copy_reproducibility_scripts(output_dir, force=args.force)

    if args.upload:
        upload_release(
            repo_id=repo_id,
            folder_path=output_dir,
            private=args.private,
            revision=args.revision,
            commit_message=args.commit_message,
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package and optionally upload Wordpipe-specialized Nemotron model profiles "
            "to the Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(MODEL_PROFILES),
        help="Profile to publish. Defaults to fast. Publish one profile per Hugging Face model repo.",
    )
    parser.add_argument(
        "--model-family",
        choices=("multilingual", "english"),
        default="multilingual",
        help="Checkpoint family being packaged.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Directory containing profile output directories using Wordpipe's canonical names.",
    )
    parser.add_argument("--fast-dir", type=Path, help="Built ONNX directory for the fast profile.")
    parser.add_argument("--compact-dir", type=Path, help="Built ONNX directory for the compact profile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for release files and README.md. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face model repo id to publish to. Defaults to the selected profile's repo.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing local profile files.")
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0..9",
        help="Ignored compatibility option from the old tarball publisher.",
    )
    parser.add_argument(
        "--force-card",
        action="store_true",
        help="Overwrite an existing generated README.md in the output directory.",
    )
    parser.add_argument("--upload", action="store_true", help="Upload output-dir to Hugging Face.")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face repo as private.")
    parser.add_argument("--revision", help="Branch, tag, or PR ref to upload to.")
    parser.add_argument(
        "--commit-message",
        default="Publish Wordpipe Nemotron model profiles",
        help="Commit message for Hugging Face upload.",
    )
    return parser.parse_args()


def resolve_profile_source(
    args: argparse.Namespace,
    spec: ModelProfileSpec,
    family: str = "multilingual",
) -> Path:
    override = getattr(args, f"{spec.name}_dir")
    if override is not None:
        return override.expanduser()
    if args.model_root is not None:
        return spec.output_dir(args.model_root.expanduser(), family)
    raise SystemExit(f"{spec.name}: pass --{spec.name}-dir or --model-root")


def validate_publish_source(
    source: Path,
    spec: ModelProfileSpec,
    family: str = "multilingual",
) -> None:
    source = source.expanduser()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a directory")
    if not model_runtime_dir_valid(source):
        raise SystemExit(
            f"{source} is not a built Wordpipe profile; expected tokenizer.model plus encoder/decoder files"
        )
    missing = [name for name in REQUIRED_ONNX_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(
            f"{source} is not publishable as a prebuilt profile; missing {', '.join(missing)}. "
            "Publish the ONNX profile directory, not the local ORT runtime cache."
        )
    validate_profile_config(source, spec, family)


def validate_profile_config(
    source: Path,
    spec: ModelProfileSpec,
    family: str = "multilingual",
) -> None:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"{source} is missing config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{config_path} is not valid JSON: {exc}") from exc

    fixed = config.get("fixed_streaming_shapes")
    if not isinstance(fixed, dict):
        raise SystemExit(
            f"{source} is not publishable as {spec.name}: config.json is missing "
            "fixed_streaming_shapes. Publish the fixed-shape profile output, not "
            "the intermediate transform/export directory."
        )

    expected_fixed = {
        "input_frames": 65,
        "output_frames": 7,
        "num_layers": 24,
        "cache_len": 70 if family == "english" else 56,
        "hidden_dim": 1024,
        "conv_context": 8,
    }
    mismatched = [
        f"{key}={fixed.get(key)!r} (expected {value!r})"
        for key, value in expected_fixed.items()
        if fixed.get(key) != value
    ]
    if mismatched:
        raise SystemExit(
            f"{source} is not publishable as {spec.name}: fixed_streaming_shapes mismatch: "
            + ", ".join(mismatched)
        )

    if config.get("projected_cache") is not True:
        raise SystemExit(f"{source} is not publishable as {spec.name}: projected_cache must be true")

    configured_family = config.get("model_family")
    if family == "english" and configured_family != "english":
        raise SystemExit(f"{source} is not publishable as English: model_family must be english")
    if configured_family is not None and configured_family != family:
        raise SystemExit(f"{source} model_family does not match {family}")

    quantized = bool(config.get("dynamic_quint8_quantization"))
    if spec.name == "fast" and quantized:
        raise SystemExit(f"{source} is not publishable as fast: expected FP32, got quantized config")
    if spec.name == "compact" and not quantized:
        raise SystemExit(f"{source} is not publishable as compact: expected dynamic QUInt8 config")


def copy_profile_files(
    source: Path,
    output_dir: Path,
    *,
    force: bool,
) -> list[Path]:
    copied = []
    for path in publish_files(source.expanduser()):
        destination = output_dir / path.name
        if destination.exists() and not force:
            raise SystemExit(f"{destination} already exists; pass --force to overwrite it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def publish_files(source: Path) -> list[Path]:
    names = [*REQUIRED_ONNX_FILES, *OPTIONAL_PROFILE_FILES]
    return [source / name for name in names if (source / name).is_file()]


def profile_metadata(
    files: list[Path],
    source: Path,
    spec: ModelProfileSpec,
    family: str = "multilingual",
) -> dict[str, object]:
    return {
        "title": spec.title,
        "description": spec.description,
        "build_profile": spec.build_profile,
        "repo": spec.prebuilt_repo_for_family(family),
        "model_family": family,
        "source_dir": str(source),
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }


def copy_reproducibility_scripts(output_dir: Path, *, force: bool) -> None:
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name in REPRODUCIBILITY_SCRIPTS:
        source = ROOT / "scripts" / name
        destination = scripts_dir / name
        if destination.exists() and not force:
            continue
        shutil.copy2(source, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_model_for_family(family: str) -> str:
    return (
        "nvidia/nemotron-speech-streaming-en-0.6b"
        if family == "english"
        else "nvidia/nemotron-3.5-asr-streaming-0.6b"
    )


def render_model_card(
    repo_id: str,
    profiles: Iterable[str],
    family: str = "multilingual",
) -> str:
    selected = tuple(profiles)
    if len(selected) != 1:
        raise ValueError("model cards are generated for one profile per repo")
    profile_name = selected[0]
    spec = profile_spec(profile_name)
    source_model = source_model_for_family(family)
    language = "en" if family == "english" else "multilingual"
    license_id = "other" if family == "english" else "openmdw-1.1"
    model_title = "Nemotron Speech Streaming English" if family == "english" else "Nemotron 3.5 ASR Streaming"
    license_text = "the NVIDIA Open Model License" if family == "english" else "the OpenMDW 1.1 license"
    license_terms = "NVIDIA Open Model License" if family == "english" else "OpenMDW"
    return f"""---
language:
- {language}
license: {license_id}
library_name: onnx
pipeline_tag: automatic-speech-recognition
base_model: {source_model}
tags:
- automatic-speech-recognition
- onnx
- wordpipe
- nemotron
- streaming-asr
- desktop-dictation
---

# Wordpipe {model_title} {spec.title} Profile

This repository contains a Wordpipe-specialized ONNX profile derived from
[`{source_model}`](https://huggingface.co/{source_model}).
NVIDIA is the upstream model developer. Wordpipe adds export, graph
specialization, packaging, and local desktop runtime integration; this profile
is not a separately trained checkpoint.

This repository publishes the `{profile_name}` Wordpipe profile:

- Build profile: `{spec.build_profile}`
- Runtime description: {spec.description}

It is consumed by Wordpipe with:

```sh
wordpipe model-install --profile {profile_name} --model-family {family} --prebuilt-repo {repo_id}
```

## Files

The repository root contains the runtime ONNX profile files:

```text
tokenizer.model
encoder.onnx
encoder.onnx.data        # when the encoder uses external ONNX data
decoder_joint.onnx
decoder_joint.onnx.data  # when the decoder uses external ONNX data
config.json              # when produced by the build pipeline
```

`wordpipe-model-profiles-manifest.json` records file sizes, SHA-256 hashes,
source directories used when publishing, and the Wordpipe build profile.
`MODEL_SPEC.md` documents the graph specializations, runtime ABI assumptions,
and bundled reproducibility scripts used to derive these artifacts.

## Intended Use

These artifacts are intended for local Wordpipe desktop dictation on Linux. They
are optimized for the Wordpipe Parakeet/Nemotron runtime layout and are not a
general NeMo checkpoint replacement.

## Evaluation Status

Wordpipe validates candidate profiles with local WER and real-time-factor tests
before promotion. Current Wordpipe release validation is primarily English
LibriSpeech-based unless a release explicitly says otherwise. See the Wordpipe
project documentation and release notes for the exact benchmark set used for a
given upload. Do not read the upstream NVIDIA FLEURS numbers as measurements of
these transformed artifacts.

## Limitations

This profile is packaged for CPU-oriented Wordpipe usage and may not match
NeMo's standard runtime interface. The compact profile intentionally publishes
the ONNX graph; the Wordpipe installer converts it to ORT format locally for
startup-time behavior.

## License and Attribution

The upstream model card states that use of
`{source_model}` is governed by {license_text}.
Review the upstream NVIDIA model card and {license_terms} terms before redistribution or
deployment. This repository preserves that attribution and publishes derived
inference artifacts for Wordpipe.
"""


def render_model_spec(
    profiles: Iterable[str],
    family: str = "multilingual",
) -> str:
    selected = tuple(profiles)
    source_model = source_model_for_family(family)
    cache_len = 70 if family == "english" else 56
    fast_output_name = profile_spec("fast").output_name_for_family(family)
    compact_output_name = profile_spec("compact").output_name_for_family(family)
    profile_rows = []
    for profile_name in selected:
        spec = profile_spec(profile_name)
        if profile_name == "fast":
            recipe = (
                f"FP32 export, projected K/V cache rewrite, fixed c{cache_len} streaming "
                "shapes, ORT extended graph serialization."
            )
            quantization = "None; encoder and decoder_joint remain FP32."
        elif profile_name == "compact":
            recipe = (
                "Dynamic QUInt8 export transform, projected K/V cache rewrite, "
                f"fixed c{cache_len} streaming shapes, ORT extended graph serialization; "
                "Wordpipe converts the installed ONNX profile to ORT format locally."
            )
            quantization = "Dynamic QUInt8 for encoder and decoder_joint before fixed-shape specialization."
        else:
            recipe = spec.description
            quantization = "See the profile config.json generated with the repository."
        profile_rows.append(f"| `{profile_name}` | `{spec.build_profile}` | {recipe} | {quantization} |")
    rows = "\n".join(profile_rows)
    scripts = "\n".join(f"- `scripts/{name}`" for name in REPRODUCIBILITY_SCRIPTS)
    return f"""# Wordpipe Nemotron Model Specification

This repository contains derived inference artifacts for
`{source_model}`. They are not NeMo checkpoints and are
not intended to be drop-in replacements for NVIDIA's standard NeMo runtime.

## Published Profiles

| Profile | Build profile | Graph specialization | Quantization |
| --- | --- | --- | --- |
{rows}

## Common Runtime ABI

Both profiles target Wordpipe's Parakeet/Nemotron streaming runtime ABI:

- 16 kHz mono audio features.
- Batch size 1.
- Encoder input shape `processed_signal=[1, 128, 65]`.
- Encoder output shape `encoded=[1, 1024, 7]`.
- Streaming cache shape assumptions: `num_layers=24`, `hidden_dim=1024`,
  `cache_len={cache_len}`, and `conv_context=8`.
- The graph exposes per-layer projected attention cache inputs
  `cache_key_layer_N` and `cache_value_layer_N` with shape `[1, {cache_len}, 1024]`.
- The graph emits `projected_current_key_layer_N` and
  `projected_current_value_layer_N` with shape `[1, 7, 1024]`.
- The caller, not the graph, rolls the projected K/V cache between streaming
  chunks.
- Fixed-shape specialization resolves symbolic dimensions and replaces static
  `Shape` nodes where possible so ONNX Runtime can fold more graph work.

The projected-cache rewrite changes the encoder from caching raw per-layer
activations and reprojecting old context on every chunk to caching already
projected K/V tensors. This is a Wordpipe runtime contract: a generic NeMo
runner will not know how to feed or roll these extra projected cache tensors.

## Reproducibility Scripts

The Hub repository includes the scripts used by the Wordpipe source tree to
export and specialize these profiles:

{scripts}

The high-level entry point is:

```sh
python scripts/build_nemotron_wordpipe_model.py \\
  {source_model} \\
  build/{fast_output_name} \\
  --profile fp32-projected \\
  --model-family {family}

python scripts/build_nemotron_wordpipe_model.py \\
  {source_model} \\
  build/{compact_output_name} \\
  --profile compact-fixed-shape \\
  --model-family {family}
```

These commands require the Wordpipe source tree and its Python export
dependencies; the scripts are included for auditability and repeatability, not
as standalone installers.

## Repository Contents

The repository root contains only runtime profile files:

```text
tokenizer.model
encoder.onnx
encoder.onnx.data        # if external ONNX data is used
decoder_joint.onnx
decoder_joint.onnx.data  # if external ONNX data is used
config.json              # when produced by the build pipeline
```

Intermediate FP32 exports, quantization scratch files, local ORT runtime caches,
and benchmark outputs are intentionally not included in the repository.
"""


def upload_release(
    *,
    repo_id: str,
    folder_path: Path,
    private: bool,
    revision: str | None,
    commit_message: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --upload. Install it, then run `hf auth login`."
        ) from exc

    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = Exception

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message=commit_message,
            allow_patterns=[
                "README.md",
                "MODEL_SPEC.md",
                "wordpipe-model-profiles-manifest.json",
                "scripts/*.py",
                "tokenizer.model",
                "encoder.onnx",
                "encoder.onnx.data",
                "decoder_joint.onnx",
                "decoder_joint.onnx.data",
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ],
        )
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"Failed to upload to {repo_id}. Make sure `hf auth login` is using a token "
            "with write access to this model repository."
        ) from exc
    print(f"uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    raise SystemExit(main())
