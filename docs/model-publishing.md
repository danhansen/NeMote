# Publishing Wordpipe Model Profiles

Wordpipe installs app-level model profiles from prebuilt Hugging Face model
repositories by default. Each profile is published as its own Hub repo with raw
ONNX files at the repository root:

```text
fractalyzer/wordpipe-nemotron-fast-fp32-projected
fractalyzer/wordpipe-nemotron-compact-fixed-shape
fractalyzer/wordpipe-nemotron-en-fast-fp32-projected
fractalyzer/wordpipe-nemotron-en-compact-fixed-shape
```

Each repo should contain:

```text
tokenizer.model
encoder.onnx
decoder_joint.onnx
```

If an ONNX graph uses external tensor data, the matching `encoder.onnx.data` or
`decoder_joint.onnx.data` file must sit next to the graph. The publishing script
includes those sidecars automatically when present and excludes stale duplicate
export files such as `*.fp32.*`.

Do not publish the local `*-ort-format` runtime cache for `compact`; Wordpipe
downloads the compact ONNX files and converts them to ORT format during install.

## Hub Conventions

The generated Hugging Face `README.md` is the model card. It uses YAML metadata
so the Hub can index the repository correctly:

- `pipeline_tag: automatic-speech-recognition`
- `library_name: onnx`
- `base_model: nvidia/nemotron-3.5-asr-streaming-0.6b`
- `license: openmdw-1.1`
- `language: multilingual`

English-family cards instead identify
`nvidia/nemotron-speech-streaming-en-0.6b`, use `language: en`, and preserve its
NVIDIA Open Model License metadata. The English optimized graphs use a c70
attention cache; multilingual graphs remain c56.

The card should make clear that NVIDIA is the upstream model developer and that
Wordpipe publishes derived inference artifacts: export, graph specialization,
packaging, and desktop runtime integration. Do not claim new training datasets,
new upstream FLEURS results, or a separate checkpoint lineage unless those have
actually been produced and validated.

Before uploading, review the generated model card for:

- Attribution to `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- OpenMDW 1.1 license metadata and prose link.
- A clear intended-use section for local Wordpipe dictation.
- A limitations section explaining that `compact` is converted to ORT locally.
- Evaluation language that points to Wordpipe release documentation instead of
  copying NVIDIA's upstream benchmark numbers.

## Publish Profiles

The full build-and-publish path for both performance profiles is automated by
the family release wrapper:

To include both streaming modes, add `--streaming-latency-ms 1120` to the
family command below. It builds the 560 ms and 1120 ms encoders and packages
each performance profile in its existing Hub repository, with shared weights.
No separate per-latency repository is created.

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_family.py \
  --model-family english \
  --python .venv-nemo-export/bin/python \
  --force \
  --upload
```

It downloads/reuses the family checkpoint, builds `fast` and `compact` with the
correct family-specific cache contract, validates and packages each repository,
then uploads both. Use `--model-family multilingual` for the multilingual pair,
`--dry-run` to inspect the commands without changing anything, or `--skip-build`
to repackage and upload already-built artifacts.

For manual packaging, pass the existing 560 ms output as before and add
`--1120ms-dir /path/to/the/1120ms/onnx-output`. The publisher emits
`wordpipe-streaming-bundle.json`, `encoder.1120ms.onnx`, and
`config.1120ms.json`. Both encoders reference the same learned tensor data.
Shape-dependent positional constants may differ and are stored separately.
Large tensors that should be shared but differ are rejected. The downloader
checks SHA-256 before reusing installed weights and retains the existing
560 ms directory layout.

| Mode | Right context | Mel input frames | Encoder output frames |
| --- | ---: | ---: | ---: |
| 560 ms | 6 | 65 | 7 |
| 1120 ms | 13 | 121 | 14 |

Session-time dimension overrides let ORT fold symbolic shape work and cache a
mode-specific optimized graph. They do not change a baked-in attention mask.
Therefore the current bundle keeps two small encoder graph structures until a
single fully dynamic NeMo export has demonstrated equivalent tokens and speed
at both contexts. Each real export still needs numerical parity and WER/RTF
validation before upload; the packaging and shape probes are not ASR benchmarks.

Publish one profile per Hugging Face model repo. From canonical installed
profile names under a model root:

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --profile fast \
  --model-root ~/.local/share/wordpipe/models \
  --output-dir build/model-release/fast \
  --force

PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --profile compact \
  --model-root ~/.local/share/wordpipe/models \
  --output-dir build/model-release/compact \
  --force
```

Repeat each command with `--model-family english` and a separate output
directory to package the dedicated-English artifacts. The family selects the
canonical source directory, destination Hub repo, model card, and c70
validation contract automatically.

Or from explicit build directories:

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --profile fast \
  --fast-dir build/model-variants/nemotron-fp32-projected-fixed-shape \
  --output-dir build/model-release/fast \
  --force

PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --profile compact \
  --compact-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-release/compact \
  --force
```

The script writes the raw runtime files, a generated `README.md`,
`wordpipe-model-profiles-manifest.json` with sizes and SHA-256 hashes,
`MODEL_SPEC.md`, and a `scripts/` directory with the export and graph rewrite
scripts used by the Wordpipe source tree.

`MODEL_SPEC.md` documents the assumptions baked into the published graphs:

- 65 mel input frames and 7 encoder output frames, with 56 projected-cache
  frames for multilingual or 70 for dedicated English.
- Batch size 1, 24 layers, hidden size 1024, and convolution cache context 8.
- The projected K/V cache ABI: per-layer `cache_key_layer_N` and
  `cache_value_layer_N` inputs plus `projected_current_*` outputs.
- The caller is responsible for rolling projected K/V cache tensors between
  streaming chunks.
- `fast` is FP32 projected-cache ONNX; `compact` is dynamic-QUInt8
  projected-cache ONNX that Wordpipe converts to ORT format locally.

Bundling these scripts in the Hugging Face model repository is intentional:
they are source/reproducibility artifacts for the published inference files, not
extra model weights.

## Upload To Hugging Face

Authenticate first:

```sh
hf auth login
```

Use a token with write access to the destination model repository. A read-only
token is sufficient for `model-install` downloads, but publishing needs write
permission because the script creates or updates the Hub repo.

Then upload:

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --profile fast \
  --fast-dir build/model-variants/nemotron-fp32-projected-fixed-shape \
  --output-dir build/model-release/fast \
  --force \
  --upload

PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --profile compact \
  --model-root ~/.local/share/wordpipe/models \
  --output-dir build/model-release/compact \
  --force \
  --upload
```

The script uses `huggingface_hub.HfApi.create_repo(..., exist_ok=True)` and
`upload_folder()` with model repo type. Recent Hugging Face tooling uses `hf_xet`
for large-file transfer; the script sets `HF_XET_HIGH_PERFORMANCE=1` by default
for uploads.

## Verify Install

After upload, verify both profiles from a clean model root:

```sh
PYTHONPATH=src python3 -m wordpipe model-install \
  --profile fast \
  --model-root /tmp/wordpipe-model-install-check \
  --force-source

PYTHONPATH=src python3 -m wordpipe model-install \
  --profile compact \
  --model-root /tmp/wordpipe-model-install-check \
  --force-source
```

`fast` should install directly as an ONNX runtime directory.
`compact` should install the ONNX profile and then produce the local
`nemotron-wordpipe-compact-fixed-shape-ort-format` runtime directory.
