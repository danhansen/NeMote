# Wordpipe Architecture

## Product Shape

Wordpipe is a Wayland-only GNOME dictation app. It should behave like a
consented virtual keyboard driven by streaming ASR, not like a screen scraper or
X11-style automation tool.

The first usable version prioritizes reliable raw streaming diagnostics and
append-only text insertion over live cursor replacement. Partial results are
visible in Wordpipe's own UI; the `voice-keyboard` path can also insert newly
appended partial text as it appears, while the lower-level daemon can still run
in final-commit-only mode.

## Locked Decisions

- GNOME-first.
- Wayland-only.
- No `xdotool` or X11 fallback path.
- No external VAD for the MVP.
- Use a Rust `parakeet-rs` ASR worker as the default runtime.
- Keep the Python sherpa-onnx worker only as a legacy diagnostic path.
- Avoid endpointing as a commit boundary for the default runtime.
- Commit non-empty partial text when dictation is stopped.
- ASR runs out-of-process to avoid making the Python GIL a core design risk.
- Text insertion uses virtual-keyboard semantics through XDG portals/libei.
- Clipboard paste is only a debug or emergency fallback.

## Process Model

```text
GNOME integration
  - global hotkey
  - status indicator
  - optional live transcript surface

wordpipe-daemon
  - owns dictation session state
  - manages portal permissions
  - starts/stops ASR process
  - receives partial/final transcript events
  - inserts committed text through the keyboard injection backend

wordpipe-voice-keyboard
  - app-facing global-hotkey mode
  - keeps focus in the target application
  - uses the GTK overlay for visible partial/status text
  - commits recognized text into the focused text field on stop

wordpipe-parakeet-worker
  - loads Parakeet/Nemotron model through parakeet-rs
  - captures microphone audio while active
  - feeds 560 ms chunks into Nemotron
  - emits partial and committed transcript events
```

The ASR worker is a subprocess speaking newline-delimited JSON. The GNOME
daemon does not depend on the model runtime; it can spawn the default Rust
Parakeet worker or the legacy Python sherpa worker.

## ASR Session Behavior

```text
dictation starts
  open microphone
  create online recognizer stream
  accept audio continuously

speech arrives
  decode fixed 560 ms Nemotron chunks
  emit partial transcript updates

dictation stops
  commit non-empty current partial
  close microphone
  stop or idle ASR worker

legacy sherpa diagnostic mode
  optionally test endpoint detection/reset behavior
```

No external VAD is used. The streaming ASR model receives the audio stream
directly.

Current low-latency defaults:

- 560 ms Nemotron chunks in the Rust worker
- 10 s audio queue before dropping microphone chunks
- endpoint detection disabled
- 2 CPU threads, based on local benchmark results

## Text Insertion

Primary target:

- XDG Desktop Portal `RemoteDesktop` keyboard session.
- Prefer `ConnectToEIS` / libei when available.
- Fall back to portal keyboard notification methods where practical.

The insertion backend should expose a text-oriented interface internally, even
though the implementation sends key events. The first version can assume US
keyboard layout for plain English text and punctuation.

Later work:

- layout-aware key generation
- richer Unicode insertion strategy
- input-method protocol investigation if virtual keyboard events become too
  limiting

## Hotkey And UI

The design supports both hold-to-dictate and toggle-to-dictate. The MVP should
choose one default and keep the state machine compatible with both.

Recommended initial UI:

- GNOME Shell shortcut for focused-app text input
- top-bar status indicator for idle/listening/permission/error states
- preferences UI for profile/status/control
- optional live transcript surface for diagnostics

The default `voice-keyboard` path types appended partial text into the target
app in real time. If a partial hypothesis rewrites already-inserted text,
Wordpipe does not attempt cursor replacement; it waits for append-only growth
or falls back to the final commit if nothing was streamed.

The current `voice-keyboard` command is the primary insertion path because it
does not steal focus from the target text field. The app window and overlay
backend use libadwaita when available, with a GTK 4 fallback. A GNOME Shell
top-bar indicator remains future work.

## Configuration

The app and daemon read `~/.config/wordpipe/config.toml` by default.
Configuration holds model path/profile, ASR runtime, worker path, provider,
thread count, overlay, hotkey mode, shortcut, spoken-punctuation behavior, and
dry-run insertion. CLI flags override file values.

## Model

The default runtime expects the Parakeet/Nemotron model layout used by
`parakeet-rs`:

```text
encoder.onnx
encoder.onnx.data
decoder_joint.onnx
tokenizer.model
```

The earlier sherpa-onnx int8 model remains useful for legacy diagnostics, but it
is not the default runtime target.

Wordpipe keeps two performance profiles, each available with a multilingual or
dedicated-English checkpoint:

- `fast`: FP32 projected-cache export, best validated speed/accuracy, largest
  footprint.
- `compact`: dynamic-int8 projected-cache export with fixed shapes and
  ORT-format startup, smaller footprint and sub-second load target.

`wordpipe model-install --profile fast|compact --model-family
multilingual|english` downloads only the selected variant and installs it under
`model_root`. Profiles that use ORT startup, currently `compact`, then convert
the ONNX profile to a local ORT-format runtime cache. All variants can coexist
under their existing directory names; the service stores `model_profile` and
`model_family` separately. The reproducible NeMo export pipeline remains
available with `model-install --build-from-nemo` for release/developer work.
See [model-publishing.md](model-publishing.md) for packaging and uploading the
prebuilt profile repos that `model-install` downloads.

## Performance

`listen-test` is the primary live tuning mode. It opens the microphone, prints
partial results, and reports realtime factor (RTF) without inserting text.
Endpoint detection is disabled by default across live paths so raw continuous
ASR behavior is visible without phrase-boundary resets.
`audio-devices` and `record-test` are diagnostic commands for validating the
capture device independently from ASR. `audio-devices --backend parakeet`
queries the Rust/CPAL worker and prints `cpal:N` selectors that can be passed
back to Parakeet runtime commands.
`stream-file-test` feeds a known WAV through the streaming recognizer and is the
primary check for whether the model emits partial hypotheses before finalization.

The current test machine exposes only Intel HD Graphics 4000, so CPU tuning and
cache-aware runtime work are the practical performance paths. Its Ivy Bridge
CPU lacks AVX2, so the Rust worker uses dynamic ONNX Runtime loading and reuses
the non-AVX2 `libonnxruntime.so` from the local `sherpa_onnx` installation.

## Packaging

The repository includes early templates for a desktop entry and user systemd
service. These are intended for local GNOME session testing after `wordpipe` is
installed on `PATH`; they are not a complete distro package.

## MVP Milestones

1. Runtime capability probe
   - GNOME version
   - portal availability
   - GlobalShortcuts availability
   - RemoteDesktop keyboard availability
   - EIS/libei availability
   - status: implemented and live-validated on GNOME 50.2 Wayland

2. Streaming ASR spike
   - load Parakeet/Nemotron model
   - stream microphone audio
   - print raw partial text
   - log real-time factor, queue depth, and commit latency
   - status: Rust worker loads the int8 Parakeet/Nemotron model and streams a
     known WAV through partial/stats/commit events; live mic validation is
     blocked in the current tool session because PipeWire/Pulse input access is
     unavailable

3. ASR process protocol
   - newline-delimited JSON over stdio or Unix socket
   - events: `partial`, `commit`, `error`, `ready`
   - commands: `start`, `stop`, `shutdown`
   - status: implemented over stdio

4. Keyboard insertion spike
   - portal permission flow
   - insert simple ASCII phrases into focused GNOME apps
   - test Text Editor, Terminal, Firefox, LibreOffice
   - status: RemoteDesktop `NotifyKeyboardKeysym` backend implemented and
     live-validated

5. GNOME hotkey and status
   - GNOME Shell extension trigger
   - daemon session state
   - visible listening/error state
   - status: Shell extension top-bar indicator and service bridge implemented

6. First integrated dictation
   - hotkey controls dictation
   - top-bar indicator shows active state
   - streaming deltas are inserted as they are produced
   - status: CLI diagnostics, Rust service, worker, and GNOME Shell extension
     implemented

## Validation Matrix

- GNOME Text Editor
- Terminal
- Firefox text fields
- LibreOffice Writer
- password field behavior, which should not receive dictated text unless the
  user explicitly accepts that risk later

## Open Questions

- Default hotkey mode: hold-to-dictate, toggle-to-dictate, or both.
- Model installation: managed `model-install` downloads the selected `fast` or
  `compact` prebuilt profile under the canonical model root.
- App ergonomics: extension preferences should cover the setup and status tasks
  that previously lived in experimental control surfaces.
  `voice-keyboard --signal-hotkey` daemon plus a GNOME shortcut.
- Whether spoken punctuation should remain always-on by default after live
  testing.
