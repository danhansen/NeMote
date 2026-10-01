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
    profile_streaming_latency,
    dynamic_streaming_config,
)


DEFAULT_OUTPUT_DIR = ROOT / "build" / "model-release"
REQUIRED_ONNX_FILES = ("tokenizer.model", "encoder.onnx", "decoder_joint.onnx")
OPTIONAL_PROFILE_FILES = (
    "encoder.onnx.data",
    "encoder.shared-weights.data",
    "decoder_joint.onnx.data",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "validation.json",
)
REPRODUCIBILITY_SCRIPTS = (
    "build_nemotron_wordpipe_model.py",
    "export_nemotron_parakeet_optimized.py",
    "transform_nemotron_parakeet_export.py",
    "rewrite_nemotron_projected_kv_cache.py",
    "rewrite_nemotron_pointwise_convs.py",
    "build_nemotron_fixed_shape_model.py",
    "convert_nemotron_to_ort_format.py",
    "bundle_nemotron_streaming_modes.py",
    "nemotron_dynamic_encoder.py",
    "build_nemotron_dynamic_shape_model.py",
    "verify_dynamic_nemotron_export.py",
    "validate_dynamic_nemotron_release.py",
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
    latency_ms = 560
    repo_id = args.repo_id or spec.prebuilt_repo_for_family(family)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "repo_id": repo_id,
        "source_model": source_model_for_family(family),
        "model_family": family,
        "streaming_latency_ms": latency_ms,
        "model_spec": "MODEL_SPEC.md",
        "reproducibility_scripts": [f"scripts/{name}" for name in REPRODUCIBILITY_SCRIPTS],
        "profiles": {},
    }

    for profile_name in profiles:
        spec = profile_spec(profile_name)
        source = resolve_profile_source(args, spec, family)
        validate_publish_source(source, spec, family)
        config = json.loads((source / "config.json").read_text())
        dynamic = dynamic_streaming_config(config, family)
        if dynamic is not None:
            if args.mode_1120_dir is not None:
                raise SystemExit("a generic encoder must not be bundled with separate per-mode exports")
            validate_release_evidence(source, spec, dynamic)
            manifest["supported_streaming_latencies_ms"] = [size * 80 for size in dynamic["supported_chunk_frames"]]
            manifest["dynamic_streaming"] = dynamic
        elif profile_streaming_latency(source) != latency_ms:
            raise SystemExit("publication mode does not match the model's streaming latency")
        copied = copy_profile_files(
            source,
            output_dir,
            force=args.force,
        )
        manifest["profiles"][profile_name] = profile_metadata(copied, source, spec, family)
        print(f"{profile_name}: {output_dir}")

    if args.mode_1120_dir is not None:
        from bundle_nemotron_streaming_modes import append_mode
        validate_publish_source(args.mode_1120_dir, spec, family)
        append_mode(output_dir, args.mode_1120_dir, 1120)
        manifest["streaming_bundle"] = "wordpipe-streaming-bundle.json"
        manifest["profiles"][spec.name] = profile_metadata(
            publish_files(output_dir), source, spec, family)

    manifest_path = output_dir / "wordpipe-model-profiles-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")

    readme_path = output_dir / "README.md"
    if not readme_path.exists() or args.force_card:
        card = render_model_card(repo_id, profiles, family, dynamic_streaming=dynamic)
        if args.mode_1120_dir is not None:
            card += "\n## Streaming Modes\n\nThis bundle provides 560 ms and 1120 ms encoders sharing one set of weights.\n"
        readme_path.write_text(card, encoding="utf-8")
        print(f"model card: {readme_path}")

    model_spec_path = output_dir / "MODEL_SPEC.md"
    if not model_spec_path.exists() or args.force_card:
        model_spec_path.write_text(
            render_dynamic_model_spec(spec, family, dynamic) if dynamic is not None
            else render_model_spec(profiles, family, latency_ms), encoding="utf-8")
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
    parser.add_argument("--1120ms-dir", dest="mode_1120_dir", type=Path,
                        help="Add a 1120 ms encoder sharing the base 560 ms weights in the same repo.")
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

    if config.get("dynamic_streaming") is not None:
        if dynamic_streaming_config(config, family) is None or config.get("fixed_streaming_shapes") is not None:
            raise SystemExit("invalid generic streaming contract or conflicting fixed shapes")
        validate_profile_kind(config, source, spec, family)
        return
    fixed = config.get("fixed_streaming_shapes")
    if not isinstance(fixed, dict):
        raise SystemExit(
            f"{source} is not publishable as {spec.name}: config.json is missing "
            "fixed_streaming_shapes. Publish the fixed-shape profile output, not "
            "the intermediate transform/export directory."
        )

    right_context = config.get("right_context", 6)
    if right_context not in (6, 13):
        raise SystemExit("unsupported streaming right context")
    output_frames = right_context + 1
    expected_fixed = {
        "input_frames": output_frames * 8 + 9,
        "output_frames": output_frames,
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

    validate_profile_kind(config, source, spec, family)


def validate_profile_kind(config, source, spec, family):
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


def validate_release_evidence(source, spec, dynamic):
    path = source / "validation.json"
    if not path.is_file():
        raise SystemExit("generic publication requires artifact-bound validation.json")
    report = json.loads(path.read_text())
    if report.get("format") != 1 or report.get("passed") is not True or report.get("profile") != spec.name:
        raise SystemExit("invalid dynamic release evidence")
    if report.get("supported_chunk_frames") != dynamic["supported_chunk_frames"]:
        raise SystemExit("release evidence does not cover the advertised chunk modes")
    if report.get("performance_noise_percent") != 5 or report.get("decode_regression_percent", 100) > 5:
        raise SystemExit("release evidence exceeds the 5% performance gate")
    files = report.get("files", {})
    required = set(REQUIRED_ONNX_FILES) | {"config.json"}
    config = json.loads((source / "config.json").read_text())
    required.update(config.get("shared_weight_files", []))
    if set(files) != required:
        raise SystemExit("release evidence does not cover all runtime artifacts")
    for name, record in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise SystemExit("unsafe validation artifact path")
        artifact = source / name
        if artifact.stat().st_size != record.get("bytes") or sha256_file(artifact) != record.get("sha256"):
            raise SystemExit(f"artifact changed since validation: {name}")


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
    *, dynamic_streaming: dict | None = None,
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
    runtime_description = spec.description if dynamic_streaming is None else (
        "One generic projected-cache ONNX encoder; runtime chunk size is selected from checkpoint metadata."
    )
    local_runtime = (
        "The compact profile intentionally publishes the ONNX graph; the Wordpipe installer converts it to ORT format locally for startup-time behavior."
        if dynamic_streaming is None else
        "Users download a ready-to-use model; NeMo/PyTorch and local model export are not required. "
        "ONNX Runtime optimizes the selected chunk size locally and can cache the result. "
        "First-use optimization has a startup cost; validation.json records cold/warm timing and peak RAM."
    )
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
- Runtime description: {runtime_description}

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
NeMo's standard runtime interface. {local_runtime}

## License and Attribution

The upstream model card states that use of
`{source_model}` is governed by {license_text}.
Review the upstream NVIDIA model card and {license_terms} terms before redistribution or
deployment. This repository preserves that attribution and publishes derived
inference artifacts for Wordpipe.
"""


def render_dynamic_model_spec(spec, family, dynamic):
    latencies = ", ".join(str(size * 80) for size in dynamic["supported_chunk_frames"])
    cache = dynamic["cache_len"]
    return f"""# Wordpipe Generic Nemotron Runtime Specification

Source checkpoint: `{source_model_for_family(family)}`. Profile: `{spec.name}`.
One encoder is published for this precision profile; no separate chunk-size exports.
Supported chunk durations, derived from checkpoint metadata: {latencies} ms.
Requires Wordpipe 0.1.19 or newer with its matching parakeet-rs fork.

For selected encoder chunk length C, the input is `[1,128,8*C+9]` and
the encoded output is `[1,1024,C]`. Batch size is 1, audio is mono 16 kHz.
The graph derives attention context from the input shape, not fixed Python state.
Streaming caches have 24 layers, hidden size 1024, left length {cache}, convolution length 8.
`cache_key_layer_N` and `cache_value_layer_N` are `[1,{cache},1024]`;
`projected_current_key_layer_N` and `projected_current_value_layer_N` are `[1,C,1024]`.
The caller rolls the projected K/V cache after each chunk.

The fast profile retains FP32 weights; compact uses one coherent dynamic QUInt8 pass.
Neither profile requires users to install NeMo or export a checkpoint.
ORT free-dimension overrides select C when opening a worker session. A local
optimized cache is optional and specific to the mode, ORT version, and host.
Changing chunk size restarts the idle worker; it is not a mid-utterance context change.

`validation.json` binds numerical, transcript, performance, and peak-memory
checks to SHA-256 hashes of these artifacts. Performance noise band: ±5%.
It includes the measured first-use optimization cost, which is not a NeMo export.
This is a small English LibriSpeech regression set, not a general WER estimate.

Maintainer reproduction (not an end-user installation step):

```sh
python scripts/build_nemotron_wordpipe_model.py SOURCE.nemo OUTPUT \\
  --profile {spec.build_profile} --model-family {family} --dynamic-streaming
```
The separate process phases avoid holding Torch and ORT models concurrently.
"""


def render_model_spec(
    profiles: Iterable[str],
    family: str = "multilingual",
    latency_ms: int = 560,
) -> str:
    selected = tuple(profiles)
    source_model = source_model_for_family(family)
    cache_len = 70 if family == "english" else 56
    output_frames = latency_ms // 80
    input_frames = output_frames * 8 + 9
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
- Streaming chunk duration: {latency_ms} ms.
- Encoder input shape `processed_signal=[1, 128, {input_frames}]`.
- Encoder output shape `encoded=[1, 1024, {output_frames}]`.
- Streaming cache shape assumptions: `num_layers=24`, `hidden_dim=1024`,
  `cache_len={cache_len}`, and `conv_context=8`.
- The graph exposes per-layer projected attention cache inputs
  `cache_key_layer_N` and `cache_value_layer_N` with shape `[1, {cache_len}, 1024]`.
- The graph emits `projected_current_key_layer_N` and
  `projected_current_value_layer_N` with shape `[1, {output_frames}, 1024]`.
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
                "validation.json",
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
