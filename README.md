# Wordpipe

Streaming dictation for GNOME Shell on Wayland, powered by Nemotron and
NVIDIA's NeMo-Speech.cpp runtime. Recognition and the D-Bus service are C++;
the extension is JavaScript. Small standard-library Python helpers download
models. There is no Rust, ONNX Runtime, local model export, or client compilation.

## Install

Download a Linux release from [GitHub Releases](https://github.com/danhansen/wordpipe/releases),
extract it, and run `./install.sh`. Python 3.11 or newer is required; no pip
packages or virtual environment are needed. Log out and back in after an
extension upgrade so GNOME Shell loads the new code.

Open preferences with:

```sh
gnome-extensions prefs wordpipe@dhansen.dev
```

Install either **English (lower WER)** or **Multilingual** in preferences.
Downloads show byte/percentage progress, retry transport errors, resume saved
partials, and verify the pinned model's SHA256 before installation. All four
chunk sizes—80, 160, 560, and 1120 ms—use the same model file.

The default shortcut is Ctrl+Alt+Space. Toggle and push-to-talk are supported.
Choose a microphone, language, chunk size, and CPU thread count in preferences.

## Dictation options

- Inverse text normalization (ITN) formats spoken numbers, dates, and units.
- Partial + committed insertion shows replaceable previews; committed-only
  insertion waits for a finalized endpoint or Stop.
- Pause endpointing can be disabled, run ITN on the current full-utterance
  preview without finalizing, or run ITN and finalize/commit at pauses.
- Vocabulary/phrase boosting favors names and jargon. Its tokenizer downloads
  automatically; boosting can also introduce incorrect matches.
- Optional VAD masks non-speech features. Its Silero model downloads automatically.
- Spoken-punctuation commands, overlay visibility, and insertion delay remain configurable.

ITN grammars are shipped for English, Arabic, Chinese, French, German, Hindi,
and Spanish. ITN is off by default. Pause endpointing and VAD are independent.
Recognition runs one stream at a time, with one stream-state arena slot.
CPU instruction-set plugins are selected at runtime; GPU execution is not enabled.

## Diagnostics

```sh
journalctl --user -u wordpipe-service.service -f
scripts/wordpipe-gnome-status
```

Settings are saved in `~/.config/wordpipe/service.json`; models are under
`~/.local/share/wordpipe/models`. XDG config/data directories are respected.
Old Parakeet selections migrate to Nemotron, while unrelated settings and
downloaded model files are preserved. Old CPAL microphone selectors are reset.

## Development

```sh
scripts/dev-container test
scripts/dev-container all dev build/release
```

The development container builds the pinned NeMo SDK, C++ worker/service, ITN
grammars, and release packages. Native builds require CMake, a C++17 compiler,
GLib/GIO development files, and the dependencies used by `scripts/build-nemo-worker`.
See [architecture](docs/architecture.md) and [runtime/protocol](docs/asr-backends.md).

## Historical performance research

The former Parakeet-RS/ORT implementation and its experiment tooling are preserved
on [archive/parakeet-rs-v0.1.28](https://github.com/danhansen/wordpipe/tree/archive/parakeet-rs-v0.1.28).
The following documents remain as historical evidence for future optimization
work, **not current installation or implementation instructions**. Referenced
retired scripts and source paths can be found on that archival branch.

- [Performance audit](docs/performance-audit.md)
- [Optimization experiments](docs/optimization-experiments.md)
- [Dynamic-chunk performance](docs/dynamic-chunk-performance.md)
- [NVIDIA pipeline audit](docs/nvidia-pipeline-audit.md)
- [Frontend parity and accuracy](docs/frontend-parity.md)
- [Sayboard optimization experiments](docs/sayboard-optimization-harvest.md)
- [GNOME service experiment](docs/gnome-extension-service-experiment.md)
