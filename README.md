# Wordpipe

Release target: **v0.1.25**. Select either Parakeet/ONNX Runtime or the native
NeMo-Speech.cpp backend in preferences. The native backend downloads NVIDIA's
official English or multilingual Q8 GGUF directly, supports all four chunk sizes,
and handles end of stream with its upstream implementation. See
[backend setup and validation](docs/asr-backends.md).

English FP32 and QUInt8 profiles use one generic encoder
each, with checkpoint-supported 80/160/560/1120 ms chunk selection. Existing
fixed-model installations remain compatible; users do not export models locally.
QUInt8 pointwise projections use fused matrix multiplies to preserve the deployed
execution path. See [the measured investigation](docs/dynamic-chunk-performance.md).

Wordpipe is a Wayland-only GNOME dictation app built around true streaming
speech recognition. The primary implementation is a GNOME Shell frontend backed
by a Rust D-Bus service and either a Rust `parakeet-rs` ASR worker or a native
C++ NeMo-Speech.cpp worker.

## Direction

- GNOME Shell-first Linux desktop integration.
- Wayland only; no X11 tooling.
- Streaming ASR with selectable Parakeet and NeMo-Speech.cpp backends.
- Target model family: Parakeet/Nemotron cache-aware streaming ASR.
- No external VAD for the MVP.
- Endpoint detection is disabled by default while raw continuous streaming is
  evaluated.
- Partial recognition appears in Wordpipe UI; committed text is inserted when
  dictation stops.

See [docs/architecture.md](docs/architecture.md) for the current design plan.

## Current MVP

The current implementation provides:

- GNOME Shell extension with a top-bar indicator, Shell shortcut handling,
  preferences UI, D-Bus service client, and Shell-side text insertion adapter.
- Rust `wordpipe-service` D-Bus session service for configuration, profile
  install, dictation control, and worker lifecycle.
- `wordpipe-parakeet-worker` Rust newline-JSON streaming worker.
- `wordpipe-nemo-worker` C++ newline-JSON streaming worker with official GGUF downloads.
- `wordpipe model-install` profile install from the published Wordpipe Nemotron
  `fast` and `compact` Hugging Face model repos.
- `wordpipe probe` capability checks for GNOME, portals, and Python modules.
- `wordpipe asr-worker` legacy sherpa-onnx newline-JSON worker.
- `wordpipe type-text` keyboard insertion through the RemoteDesktop portal.
- `wordpipe daemon` MVP loop that connects the ASR worker to text insertion.
- `wordpipe hotkey-daemon` manual or GlobalShortcuts-controlled dictation.
- `wordpipe voice-keyboard` global-hotkey dictation into the focused text box.

See [docs/gnome-extension-service-experiment.md](docs/gnome-extension-service-experiment.md).

Other frontends remain possible, but the main branch is focused on the GNOME
Shell frontend and Rust service.

## Install

The GitHub release archive contains the GNOME Shell extension, Rust D-Bus
service, ASR worker, user D-Bus activation file, systemd user unit, and the
Python model-install helper.

GNOME Shell caches loaded extension modules. After upgrading Wordpipe, log out
and back in so the Wayland session loads the new JavaScript. For development,
the official GNOME workflow uses a nested Shell session.

```sh
curl -fsSL https://raw.githubusercontent.com/danhansen/wordpipe/main/scripts/install-wordpipe-release | sh
```

Manual equivalent:

```sh
curl -fsSLO https://raw.githubusercontent.com/danhansen/wordpipe/main/scripts/install-wordpipe-release
less install-wordpipe-release
sh install-wordpipe-release
```

After install, open settings with:

```sh
gnome-extensions prefs wordpipe@dhansen.dev
```

Install the compact model profile:

```sh
~/.local/libexec/wordpipe/bin/wordpipe-model-install \
  --profile compact \
  --model-family multilingual
```

Release builds also publish `wordpipe@dhansen.dev.shell-extension.zip`, which
is the source-only GNOME extension archive for review or manual extension
installation. The Shell extension still requires the separately installed Rust
service from the composed release archive.

## Local Development

The checked-in development container is the canonical build environment. It
pins the Rust image and Python model-tool dependencies and includes ALSA, GNOME
schema, Node syntax-check, and release packaging tools. With Podman or Docker:

```sh
scripts/dev-container test
scripts/dev-container shell
scripts/dev-container package dev-local dist
```

The GitHub CI and Release workflows use this same image and
`scripts/run-build-pipeline`, so local container results exercise the same
commands used to publish releases.

To run only the lightweight host-side Python tests:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests
```

Check that the GNOME Shell extension and preferences UI still declare the same
D-Bus methods/signals as the Rust protocol crate:

```sh
python3 scripts/check_gnome_dbus_xml.py
```

Build the Rust Parakeet worker:

```sh
cargo build --release -p wordpipe-parakeet-worker
```

After creating `.venv`, the `scripts/wordpipe-dev` wrapper runs the local source
tree without repeating `PYTHONPATH=src .venv/bin/python -m wordpipe`:

```sh
scripts/wordpipe-dev probe
```

Run the capability probe:

```sh
PYTHONPATH=src python3 -m wordpipe probe
```

Inspect a downloaded sherpa-onnx model directory:

```sh
PYTHONPATH=src python3 -m wordpipe model-info --model-dir /path/to/model
```

Run offline decoding against a WAV file:

```sh
PYTHONPATH=src python3 -m wordpipe transcribe-file \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --wav models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/en.wav
```

Run live partial-only testing with RTF metrics using the Rust Parakeet runtime:

```sh
cargo build --release -p wordpipe-parakeet-worker
scripts/wordpipe-dev listen-test \
  --model-dir /path/to/parakeet-nemotron-streaming-model
```

This opens the microphone and prints `partial` and `commit` events without
inserting text into any app. It also prints periodic `stats` lines with RTF,
audio level, and dropped-chunk counts. When the recognizer has a current
hypothesis, each stats tick also repeats it as a `partial` line, so you can see
stable partial text even when it has not changed. Use Ctrl+C to stop.
The default Parakeet runtime takes raw continuous mic audio into ASR and commits
the accumulated transcript when dictation stops. The legacy sherpa runtime can
still be selected with `--asr-runtime sherpa`.
The Rust worker defaults to ONNX Runtime's `all` graph optimization level; use
`--graph-optimization` only for ablations or debugging.
In interactive daemon mode, the Parakeet worker preloads the model before
emitting `ready`; subsequent hotkey starts reset the resident model instead of
reloading ONNX sessions.
Once a Wordpipe profile is installed under `model_root`, `listen-test` and
`stream-file-test` can use `--model-profile compact|fast` instead of
`--model-dir`.

Optimization work is tracked in
[docs/optimization-experiments.md](docs/optimization-experiments.md). The
Sayboard optimization inventory and harvest results are in
[docs/sayboard-optimization-harvest.md](docs/sayboard-optimization-harvest.md).

For model A/B checks on a concatenated LibriSpeech WAV, build a broader sample
with `scripts/build_librispeech_long_wav.py`, run
`scripts/benchmark_parakeet_variant.py`, then score speed and accuracy together:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/highperf-broad-wer-rtf-001.json \
  --manifest build/librispeech-highperf-validation/manifest.jsonl
```

Inspect ONNX graphs and ORT optimization effects:

```sh
.venv/bin/python scripts/ort_graph_diagnostics.py \
  models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56/encoder.onnx \
  --json-out build/ort-diagnostics/encoder-summary.json
```

For smaller graphs, or when you are comfortable spending the memory to let ORT
load and serialize an optimized model, add `--emit-optimized --opt-level all`.
The resulting summary makes ORT fusions visible, such as
`DynamicQuantizeLinear + MatMulInteger` becoming `DynamicQuantizeMatMul`.

Run a Microsoft Olive ONNX pass experiment against a Wordpipe model directory:

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-olive-peephole \
  --pass-name peephole \
  --force
```

The wrapper keeps Wordpipe's `encoder.onnx`, `decoder_joint.onnx`,
`config.json`, and `tokenizer.model` layout and writes
`olive_pass_summary.json` with before/after node, initializer, size, and op
counts. Olive is not part of the default model-tools extra; see
[docs/optimization-experiments.md](docs/optimization-experiments.md) for the
exact Olive setup and the current pass results.

If you run `target/release/wordpipe-parakeet-worker` directly, set
`ORT_DYLIB_PATH` to the ONNX Runtime library from the local Python wheel. The
`ort` crate's default runtime can hang while loading this encoder on the current
machine:

```sh
ORT_DYLIB_PATH="$PWD/.venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0" \
  target/release/wordpipe-parakeet-worker \
  --model-dir /path/to/parakeet-nemotron-streaming-model \
  --wav /path/to/test.wav
```

List input devices:

```sh
scripts/wordpipe-dev audio-devices
scripts/wordpipe-dev audio-devices --backend parakeet
```

Try a specific input device. Numeric values are sounddevice indices from the
default `audio-devices` listing. For the Parakeet runtime, prefer the
`--backend parakeet` listing because it comes from the same Rust/CPAL worker
that records audio; pass its `cpal:N` selector or a device-name substring.
When a sounddevice index is passed to Parakeet, Wordpipe resolves it to a device
name before handing it to CPAL.

```sh
scripts/wordpipe-dev listen-test \
  --input-device cpal:0 \
  --model-dir /path/to/parakeet-nemotron-streaming-model
```

Record what Wordpipe is hearing:

```sh
scripts/wordpipe-dev record-test --duration 5 --output /tmp/wordpipe-spoken.wav
scripts/wordpipe-dev transcribe-file \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --wav /tmp/wordpipe-spoken.wav
```

If the recorded WAV transcribes but `listen-test` does not produce partials, the
problem is streaming throughput. If the WAV does not transcribe, the issue is
audio capture, device selection, level, or model suitability for the speech.

Test streaming behavior from a known-good WAV:

```sh
scripts/wordpipe-dev stream-file-test \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --wav /tmp/wordpipe-spoken.wav
```

This should print partials if the model emits them in streaming mode.

For the Parakeet/Nemotron app profiles, the smoke wrapper resolves the selected
profile, feeds a WAV through `stream-file-test`, and fails if no commit text is
produced:

```sh
.venv/bin/python scripts/smoke_stream_file.py --model-profile compact
```

Dry-run text insertion:

```sh
PYTHONPATH=src python3 -m wordpipe type-text --dry-run "hello world"
```

Create a config file:

```sh
mkdir -p ~/.config/wordpipe
PYTHONPATH=src python3 -m wordpipe config-example > ~/.config/wordpipe/config.toml
```

Run the MVP daemon:

```sh
PYTHONPATH=src python3 -m wordpipe daemon \
  --model-dir /path/to/parakeet-nemotron-streaming-model
```

Use `--dry-run-insertion` to exercise ASR without opening a portal keyboard
session.

When `~/.config/wordpipe/config.toml` contains `model_dir`, `daemon` and
`hotkey-daemon` can run without `--model-dir`. If `model_dir` is unset, the
commands load the selected `model_profile` from `model_root`. CLI flags override
config values, including `--model-profile fast|compact` and `--model-root`.

Packaging templates live under `packaging/`:

- `packaging/systemd/wordpipe.service`
- `packaging/systemd/wordpipe-service.service`
- `packaging/dbus/dev.wordpipe.Service.service`

They assume `wordpipe` is installed on `PATH` and configuration exists at
`~/.config/wordpipe/config.toml`.

Run the hotkey-controlled daemon:

```sh
PYTHONPATH=src python3 -m wordpipe hotkey-daemon \
  --model-dir /path/to/parakeet-nemotron-streaming-model \
  --mode hold \
  --shortcut 'CTRL+ALT+space' \
  --overlay gtk
```

For development without the GlobalShortcuts portal:

```sh
PYTHONPATH=src python3 -m wordpipe hotkey-daemon \
  --model-dir /path/to/parakeet-nemotron-streaming-model \
  --manual-hotkey \
  --dry-run-insertion
```

Manual commands are `down`, `up`, `toggle`, and `quit`.

The GNOME Shell extension is the normal voice-keyboard frontend. Configure the
shortcut in the extension preferences, focus any text field, trigger dictation,
speak, and trigger it again to stop. In the default mode, Wordpipe inserts
append-only text deltas as ASR produces them; stopping dictation does not
reinsert the final transcript.

For visible logs while debugging, start the resident daemon manually instead:

```sh
PYTHONPATH=src python3 -m wordpipe voice-keyboard \
  --model-profile compact \
  --signal-hotkey \
  --overlay stderr
```

The Nemotron streaming model emits text on a 560 ms cadence, so several words
can arrive in one partial. To make those bursts feel less abrupt, pace the
already-emitted suffix into the target field:

```sh
PYTHONPATH=src python3 -m wordpipe voice-keyboard \
  --model-profile compact \
  --signal-hotkey \
  --overlay stderr \
  --stream-insert-delay-seconds 0.03
```

This does not reduce ASR latency. It inserts the first word-like piece from each
burst immediately and then waits the configured delay between the rest. Omit the
flag, or set `stream_insert_delay_seconds = 0.0`, for fastest raw insertion.

When `voice-keyboard-toggle --start-if-needed` starts the daemon for a GNOME
shortcut, daemon stdout/stderr is written to
`$XDG_CACHE_HOME/wordpipe/voice-keyboard.log`, or
`~/.cache/wordpipe/voice-keyboard.log` when `XDG_CACHE_HOME` is unset. Pass
`--daemon-log-file` to `voice-keyboard-toggle` to override it.

To restore the older behavior where nothing is typed until dictation stops:

```sh
PYTHONPATH=src python3 -m wordpipe voice-keyboard \
  --model-profile compact \
  --signal-hotkey \
  --overlay stderr \
  --final-commit-only
```

The lower-level GlobalShortcuts portal path is still available for diagnostics:

```sh
PYTHONPATH=src python3 -m wordpipe voice-keyboard \
  --model-profile compact \
  --mode toggle \
  --overlay stderr
```

If GNOME rejects that with `An app id is required`, use the Shell extension
frontend.

Use hold mode if you prefer press-and-hold dictation:

```sh
PYTHONPATH=src python3 -m wordpipe voice-keyboard \
  --model-profile compact \
  --mode hold \
  --shortcut 'CTRL+ALT+space'
```

For a non-inserting test of the same flow:

```sh
PYTHONPATH=src python3 -m wordpipe voice-keyboard \
  --model-profile compact \
  --manual-hotkey \
  --dry-run-insertion
```

## Runtime Dependencies

Install Python ASR dependencies only when using the legacy sherpa worker:

```sh
python3 -m pip install '.[asr]'
```

The local development environment has been smoke-tested with
`sherpa-onnx==1.13.3` on Python 3.14 for the legacy worker.

The default Rust runtime uses `parakeet-rs`. Build it with:

```sh
cargo build --release -p wordpipe-parakeet-worker
```

`listen-test`, `daemon`, and `hotkey-daemon` look for
`target/release/wordpipe-parakeet-worker` first, then the debug binary, then
`wordpipe-parakeet-worker` on `PATH`. Use `--asr-worker-path` to point at a
custom binary.

Download the legacy sherpa 560 ms int8 Nemotron model:

```sh
PYTHONPATH=src python3 -m wordpipe download-model
```

This writes to `models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/`
by default. The repository is:

```text
csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
```

The legacy sherpa model directory must contain `tokens.txt` and either a single
`.onnx` model for the Nemotron CTC path or `encoder*.onnx`, `decoder*.onnx`,
and `joiner*.onnx` for a transducer layout.

The default Parakeet/Nemotron runtime expects the model layout used by
`parakeet-rs`: `encoder.onnx`, any associated external data file,
`decoder_joint.onnx`, and `tokenizer.model`.

The tested int8 English model is:

```text
models/nemotron-speech-streaming-en-0.6b-int8/
```

On the current Ivy Bridge CPU, the Rust worker uses dynamic ONNX Runtime
loading. `scripts/wordpipe-dev`, `listen-test`, and daemon launch paths set
`ORT_DYLIB_PATH` automatically when a local `onnxruntime` or `sherpa_onnx`
library is present in `.venv`; the `onnxruntime` wheel library is preferred
because it loads the projected-cache Nemotron encoder reliably here.

### Building A Wordpipe Nemotron Model

Wordpipe exposes four model presets backed by two performance profiles and two
checkpoint families:

- `compact`: multilingual dynamic-int8 projected-cache model.
- `compact (English only, lower WER)`: dedicated-English compact model.
- `fast`: multilingual FP32 projected-cache model. This is the fastest
  validated profile so far and has the largest disk/RAM footprint.
- `fast (English only, lower WER)`: dedicated-English fast model.

List preset status:

```sh
PYTHONPATH=src python3 -m wordpipe model-profiles
```

The app installs only the selected preset. The dedicated
`nvidia/nemotron-speech-streaming-en-0.6b` presets expose English (US/UK), while
the `nvidia/nemotron-3.5-asr-streaming-0.6b` presets expose auto-detect and all
supported languages. Existing model directory names are unchanged.

Install the multilingual compact preset:

```sh
PYTHONPATH=src python3 -m wordpipe model-install \
  --profile compact \
  --model-family multilingual
```

Wordpipe uses `huggingface_hub` and enables `HF_HUB_ENABLE_HF_TRANSFER=1` when
`hf_transfer` is installed. Run `model-install` again with another preset
when you want to try it; its artifact can coexist under `model_root`.
See [docs/model-publishing.md](docs/model-publishing.md) for packaging and
uploading the prebuilt profile archives that this command downloads.

`model-install --source` can also import an already-built Wordpipe profile
directory or archive. Imported profile sources must contain `tokenizer.model`,
`encoder.onnx` or
`encoder.ort`, and `decoder_joint.onnx` or `decoder_joint.ort`.

For release/developer work, keep the reproducible NeMo export pipeline explicit:

```sh
PYTHONPATH=src python3 -m wordpipe model-install \
  --profile compact \
  --build-from-nemo \
  --source models/nemotron-3.5-asr-streaming-0.6b-source/nemotron-3.5-asr-streaming-0.6b.nemo \
  --python .venv-nemo-export/bin/python
```

Maintainers can build the English checkpoint with a generic encoder, projected
cache, and optional quantization. Its checkpoint-defined attention
cache is 70 frames rather than the multilingual model's 56:

```sh
PYTHONPATH=src python3 -m wordpipe model-install \
  --profile compact \
  --model-family english \
  --build-from-nemo \
  --dynamic-streaming \
  --source nvidia/nemotron-speech-streaming-en-0.6b \
  --python .venv-nemo-export/bin/python
```

Since 0.1.19 the English downloads contain one generic encoder per precision
profile, supporting every chunk size advertised by the checkpoint: currently
80, 160, 560, and 1120 ms. Users download ready-to-use FP32 or QUInt8 models;
they do not need NeMo, PyTorch, or a local export. Legacy multilingual/fixed
models continue to use their existing 560/1120 ms paths.

The GNOME service reads supported chunk sizes from installed model metadata.
Settings exposes this as **Streaming Latency**, separate from model preset and
language coverage. A mode change while idle rebuilds the worker and its inference
sessions; model loading, installation, and active dictation must finish first.
The existing model directory names stay unchanged. A generic encoder is reused
for all supported modes; no additional model download is needed when switching.

For example, `model-install --model-family english --streaming-latency-ms 160`
installs a generic profile supporting that mode. Existing fixed installations
can be upgraded with `--force`. Old shared-weight bundles remain supported for
legacy models; their additional runtimes live under `model_root/1120ms/`.

The worker reads chunk size from model metadata and applies session dimension
overrides. Optimized ONNX graphs are cached under `model_root/runtime-cache/`
with mode, model, runtime version, CPU features, and execution options in the
cache identity. Direct ORT-format sessions retain and release their backing
bytes with the session. See [model publishing](docs/model-publishing.md) for
the generic release and legacy shared-weight bundle workflows. First use may
take longer while ORT optimizes the selected mode; this is not a checkpoint export.

After a successful source build, Wordpipe removes the family-specific build
intermediates by default; pass `--keep-build-dir` when you need to inspect or
reuse those files.

Select the default profile in `~/.config/wordpipe/config.toml`:

```toml
model_profile = "compact"
model_root = "/home/you/.local/share/wordpipe/models"
nemo_source = "nvidia/nemotron-3.5-asr-streaming-0.6b"
```

`model_dir` still overrides the selected profile when set explicitly.

Any runtime command that normally loads the default profile can also select one
for a single launch:

```sh
PYTHONPATH=src python3 -m wordpipe daemon --model-profile fast
PYTHONPATH=src python3 -m wordpipe hotkey-daemon --model-profile compact
```

If that profile has not been built yet, run the same `model-install` command
with the missing profile name. The source `.nemo` is reused from
`model_root/sources/` unless `--force-source` is provided.

The current high-performance export path is codified as a thin wrapper around
the individual phase scripts:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-fp32-projected \
  --work-dir build/nemotron-wordpipe-pipeline
```

Use `--force` to overwrite an existing work/output directory. Use `--dry-run`
to print the phase commands without running them. The default profile is
`--profile fp32-projected`, which keeps the encoder and decoder in FP32 and
uses the projected-cache rewrite. This is larger on disk, but it is the fastest
validated local option so far.

The wrapper deliberately keeps the phases separate:

- `export_nemotron_parakeet_optimized.py --export-only` exports FP32 ONNX from
  NeMo and then exits before quantization so Torch/NeMo memory is released.
- `transform_nemotron_parakeet_export.py --no-quantize --projected-cache`
  rewrites the FP32 encoder to use projected K/V cache.
- `build_nemotron_fixed_shape_model.py` specializes the streaming graph to the
  current c56 runtime shape and serializes ORT's optimized encoder graph.

The older compact mixed-int8/FP32 candidate remains available:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-ffn-fp32 \
  --work-dir build/nemotron-wordpipe-pipeline-ffn-fp32 \
  --profile ffn-fp32
```

In that profile, `transform_nemotron_parakeet_export.py` applies dynamic QUInt8
quantization and projected cache, then
`dequantize_nemotron_matmul_blocks.py --include /feed_forward` rewrites FFN
MatMul/Gemm blocks back to FP32. `--fp32-decoder` is also available only in this
profile as a modest-speed experimental option.

The best compact option is the fixed-shape ORT-optimized rebuild of the
sherpa-derived int8/projected-cache package:

```sh
.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

This keeps the model around 600 MB and avoids the selective FP32 rewrites used
by the larger `ffn-fp32` profile.

To build the same compact profile from a NeMo checkpoint instead of an existing
int8/projected-cache package:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-compact-fixed-shape \
  --work-dir build/nemotron-wordpipe-pipeline-compact \
  --profile compact-fixed-shape
```

Important defaults:

```text
left_context = 56
right_context = 6
input_frames = 65
output_frames = 7
cache_len = 56
hidden_dim = 1024
ort_optimize_final = extended
```

The final output directory contains the runtime model files:

```text
encoder.onnx
decoder_joint.onnx
tokenizer.model
config.json
```

For the compact profile, native ORT format is the fastest startup artifact. The
Rust worker automatically prefers `encoder.ort` and `decoder_joint.ort` when
they are present, falling back to ONNX otherwise:

```sh
.venv-nemo-export/bin/python scripts/convert_nemotron_to_ort_format.py \
  models/nemotron-wordpipe-compact-fixed-shape \
  models/nemotron-wordpipe-compact-fixed-shape-ort-format \
  --force \
  --optimization-level all
```

The wrapper can emit that directory after a full build:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-compact-fixed-shape \
  --work-dir build/nemotron-wordpipe-pipeline-compact \
  --profile compact-fixed-shape \
  --emit-ort-format
```

On the local benchmark, the compact ORT-format model loaded in `0.461s` median
versus `1.154s` for the same compact ONNX model. The FP32 projected model is
still the best quality/speed profile, but its ORT-format conversion is
memory-heavy on a 16 GB machine and is not the default build path.

The wrapper supports `--start-at` and `--stop-after` for resuming or debugging
individual phases. For example, after a successful FP32 export:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-ffn-fp32 \
  --work-dir build/nemotron-wordpipe-pipeline \
  --start-at transform
```

## Live Validation

Validated in GNOME 50.2 on Wayland:

- RemoteDesktop portal text insertion into a focused app.
- Manual hotkey end-to-end dictation with the Adwaita/GTK overlay.
- Live microphone capture reaching the ASR `listening` state.
- Offline decoding with the downloaded Nemotron int8 model.

## Performance Notes

The default runtime is:

```text
asr_runtime = "parakeet"
num_threads = 2
queue_seconds = 10.0
```

The Rust worker derives its chunk size from the model's streaming metadata
(560 ms by default, or 1120 ms for the corresponding export). File tests feed three synthetic silence chunks by default so streaming
models can emit trailing tokens before the final commit.

Metrics report both `audio_seconds` for real input and `processed_audio_seconds`
for real input plus padding/flush audio. `real_time_factor` is calculated from
processed audio so synthetic flush work is accounted for fairly;
`real_audio_real_time_factor` keeps the stricter real-input denominator visible.

On the current test machine, the c56 Parakeet int8 export with ORT graph
optimization `all` decodes the known sherpa English test WAV at about 0.94 RTF
over processed audio with the final flush included. The legacy sherpa int8 path
is about 0.99 RTF on the same test and still misses the trailing "gold" token.

The GTK overlay prefers libadwaita (`Adw 1`) and falls back to plain GTK 4 if
libadwaita is not available. Non-UI daemon paths do not require GTK.

Committed text converts common spoken punctuation commands by default:

```text
hello comma world period -> hello, world.
new line -> Enter
new paragraph -> blank line
```

Use `--no-spoken-punctuation` to insert raw ASR output.
