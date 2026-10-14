# NeMote

Streaming dictation for GNOME Shell 50 and 51 on Wayland, powered by Nemotron and
NVIDIA's NeMo-Speech.cpp runtime. Recognition and the D-Bus service are C++;
the extension is JavaScript. Small standard-library Python helpers download
models. There is no Rust, ONNX Runtime, or client compilation. Optional English
punctuation restoration performs a one-time local model conversion.

## Install

Install the latest release into your user account:

```sh
curl -fsSL https://raw.githubusercontent.com/danhansen/NeMote/main/scripts/install-nemote-release | sh
```

To inspect the installer before running it:

```sh
curl -fsSLO https://raw.githubusercontent.com/danhansen/NeMote/main/scripts/install-nemote-release
less install-nemote-release
sh install-nemote-release
```

Alternatively, download a Linux release from [GitHub Releases](https://github.com/danhansen/NeMote/releases),
extract it, and run `./install.sh`. Python 3.11 or newer is required; no pip
packages or virtual environment are needed for normal dictation. Optional PnC
setup creates a private conversion environment (see below). Log out and back in after an
extension upgrade so GNOME Shell loads the new code.

Open preferences with:

```sh
gnome-extensions prefs nemote@dhansen.dev
```

Install either **English (lower WER)** or **Multilingual** in preferences.
Downloads show byte/percentage progress, retry transport errors, resume saved
partials, and verify the pinned model's SHA256 before installation. All four
chunk sizes—80, 160, 560, and 1120 ms—use the same model file.

The default shortcut is Ctrl+Alt+Space. Toggle and push-to-talk are supported.
Choose a microphone, language, chunk size, and CPU thread count in preferences.

Preferences are organized into three pages:

- **Dictation:** microphone, speech model selection and installation, language,
  chunk size, keyboard shortcut, and vocabulary.
- **Text:** number/date formatting, punctuation models and installation, pauses,
  and live previews. Expand punctuation restoration to manage its models, even
  while the feature is off.
- **Advanced:** runtime paths, CPU threads, model overrides, and diagnostics.

## Dictation options

- **Format Numbers and Dates** enables inverse text normalization (ITN) for spoken numbers, dates, and units.
- Experimental English punctuation/capitalization restoration (PnC) uses a
  selectable BERT or smaller DistilBERT GGUF. It runs after optional ITN at pauses
  or Stop, not on immediate partials. Install explicitly from preferences;
  installation downloads NVIDIA's checkpoint and converts locally, never during
  dictation startup. Setup needs venv support, about 1.25 GiB available RAM,
  free disk space and CPU-only Python conversion dependencies. A supplied GGUF
  override bypasses setup. See [PnC setup and model terms](docs/pnc.md).
- Partial + committed insertion shows replaceable previews; committed-only
  insertion waits for a finalized endpoint or Stop.
- Pause endpointing can be disabled, run enabled ITN/PnC processing on the
  accumulated preview without finalizing, or postprocess and commit at pauses.
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
journalctl --user -u nemote-service.service -f
scripts/nemote-gnome-status
```

Settings are saved in `~/.config/nemote/service.json`; models are under
`~/.local/share/nemote/models`. XDG config/data directories are respected.
Old Parakeet selections migrate to Nemotron, while unrelated settings and
downloaded model files are preserved. Old CPAL microphone selectors are reset.

NeMote was previously named Wordpipe. The installer disables the old extension
and retires its service activation files with backups. Existing service settings
and GNOME preferences are carried over; existing models are reused in place
through a symlink, not copied or downloaded again. Log out and back in after this
upgrade. Custom model directories are unchanged.

## Development

```sh
scripts/dev-container test
scripts/dev-container all dev build/release
```

The development container builds the pinned NeMo SDK, C++ worker/service, ITN
grammars, and release packages. Native builds require CMake, a C++17 compiler,
GLib/GIO development files, and the dependencies used by `scripts/build-nemo-worker`.
Packaging the optional PnC converter also needs Python pip on the build machine.
See [architecture](docs/architecture.md) and [runtime/protocol](docs/asr-backends.md).

Preview the **actual production UI** without installing the extension or starting
ASR, downloading models, or changing your settings:

```sh
scripts/preview-ui --watch
```

UI edits restart the preview automatically. The extra Preview Controls page
simulates missing models, downloads, loading, dictation, errors and an unavailable
service; it also switches theme and font size. Requires GJS, GTK 4 and libadwaita.
On your normal desktop, GNOME's existing global shortcuts still apply; the
headless audit below is fully isolated from those shortcuts.

For a repeatable visual audit without using your desktop:

```sh
scripts/preview-ui --headless --audit
```

This also requires GNOME Shell and `dbus-run-session`. It runs a private,
software-rendered compositor with memory-only settings and no D-Bus service
auto-activation. Open `build/ui-audit/index.html` for the preferences gallery and
`build/ui-audit/shell.html` for the real GNOME panel menu. The audit covers all three
pages (whole windows and individual sections), dropdowns and shortcut capture; normal/light and narrow/dark/large-text
layouts; and representative service states. Layout alerts are recorded in JSON
manifests and make the command fail. Screenshots still require visual review:
geometry checks alone cannot establish that a UI is correct.

## Historical performance research

The former Parakeet-RS/ORT implementation and its experiment tooling are preserved
on [archive/parakeet-rs-v0.1.28](https://github.com/danhansen/NeMote/tree/archive/parakeet-rs-v0.1.28).
The following documents remain as historical evidence for future optimization
work, **not current installation or implementation instructions**. Referenced
retired scripts and source paths can be found on that archival branch.
Historical command names and artifact paths retain their original spelling.

- [Performance audit](docs/performance-audit.md)
- [Optimization experiments](docs/optimization-experiments.md)
- [Dynamic-chunk performance](docs/dynamic-chunk-performance.md)
- [NVIDIA pipeline audit](docs/nvidia-pipeline-audit.md)
- [Frontend parity and accuracy](docs/frontend-parity.md)
- [Sayboard optimization experiments](docs/sayboard-optimization-harvest.md)
- [GNOME service experiment](docs/gnome-extension-service-experiment.md)
