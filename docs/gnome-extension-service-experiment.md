# GNOME Extension Service Experiment

This branch is the primary NeMote runtime: a GNOME Shell extension backed by
a long-lived Rust service. The ASR/model work stays reusable for KDE and other
Wayland desktops by putting the dictation engine behind a session D-Bus API.

## Target Shape

```text
GNOME Shell extension
  - global shortcut
  - panel or quick settings status
  - preferences/setup UI
  - GNOME Shell OSK/internal text insertion
  - D-Bus client

Rust wordpipe service
  - D-Bus server
  - model profile install/export/runtime
  - microphone/device enumeration
  - audio capture
  - ASR streaming
  - session state
  - append-only transcript events
```

The GNOME extension owns desktop integration. The Rust service owns work and
state. D-Bus is the boundary.

## Packaging Model

For local development, one installer can install both pieces:

```text
scripts/install-wordpipe-gnome
  - build/install Rust service under ~/.local/libexec/wordpipe/
  - install D-Bus service activation file
  - optionally install a systemd --user unit
  - install GNOME Shell extension under ~/.local/share/gnome-shell/extensions/
  - compile schemas
  - enable the extension
  - open extension preferences
```

For `extensions.gnome.org`, the extension likely cannot bundle the Rust binary.
GNOME extension review guidelines prohibit shipping binary executables or
libraries inside reviewed extensions. The extension should therefore be able to
detect and explain a missing `wordpipe-service`.

## D-Bus API Draft

Well-known name:

```text
dev.wordpipe.Service
```

Object path:

```text
/dev/wordpipe/Service
```

Interface:

```text
dev.wordpipe.Service1
```

Methods:

```text
Start() -> ()
Stop() -> ()
Toggle() -> ()
Shutdown() -> ()
GetState() -> a{sv}
GetConfig() -> a{sv}
ListBackends() -> aa{sv}
ListModelProfiles() -> aa{sv}
ListInputDevices() -> aa{sv}
SetBackend(s backend) -> ()
SetInputDevice(s selector) -> ()
SetModelProfile(s profile) -> ()
SetShortcut(s accelerator) -> ()
InstallModel(s profile) -> ()
SetInsertionOptions(a{sv} options) -> ()
SetRuntimeOptions(a{sv} options) -> ()
```

Signals:

```text
StateChanged(a{sv} state)
ConfigChanged(a{sv} config)
SessionStarted(t session_id)
TextDelta(t session_id, t seq, s text)
Partial(t session_id, t seq, s full_text)
Commit(t session_id, t seq, s text)
SessionStopped(t session_id)
InstallProgress(s profile, a{sv} progress)
Metrics(a{sv} metrics)
Error(s message)
```

`TextDelta` is append-only and is the primary signal for insertion clients.
`Partial` is for display/debugging. `session_id` and `seq` let clients ignore
stale events after toggles, restarts, or extension reloads.

`GetState` and `StateChanged` use these keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `listening` | `b` | A dictation session is actively streaming audio/ASR output. |
| `stopping` | `b` | The service has asked the worker to stop and is waiting for final flush/exit. |
| `installing` | `b` | A model install/export process is running. |
| `installing_profile` | `s` | Profile id currently being installed, or empty. |
| `loading_model` | `b` | A worker has started and is loading the selected model. |
| `model_loaded` | `b` | The active worker has reported ready/model-loaded. |
| `session_id` | `t` | Monotonic dictation session id. |
| `seq` | `t` | Monotonic transcript event sequence within the current service lifetime. |
| `backend` | `s` | Selected backend id, currently `parakeet`. |
| `model_profile` | `s` | Selected model profile id, currently `fast` or `compact`. |
| `model_family` | `s` | Selected checkpoint family, `multilingual` or `english`. |
| `model_preset` | `s` | Combined UI preset id, such as `compact-english`. |
| `input_device` | `s` | Selected CPAL input-device selector, or empty for system default. |
| `partial_text` | `s` | Most recent full partial transcript for UI display. |
| `last_commit_text` | `s` | Most recent committed/final transcript payload. |
| `selected_runtime_dir` | `s` | Runtime directory expected for the selected model profile. |
| `selected_model_installed` | `b` | Whether the selected runtime directory looks installed. |
| `last_error` | `s` | Most recent service/worker error, or empty. |
| `last_metrics` | `a{sv}` | Most recent worker metrics payload. |
| `last_install_progress` | `a{sv}` | Most recent model-installer progress payload. |

`GetConfig` and `ConfigChanged` use these keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `backend` | `s` | Backend id. Unknown values are rejected. |
| `model_profile` | `s` | Model profile id. Unknown values are rejected. |
| `model_family` | `s` | Checkpoint family selected by the model preset. |
| `model_preset` | `s` | Combined UI preset id. |
| `input_device` | `s` | CPAL input-device selector, or empty for default. |
| `shortcut` | `s` | GNOME accelerator string mirrored from extension settings. |
| `model_root` | `s` | Root directory for installed model profiles. Empty input is normalized to the service default. |
| `worker_path` | `s` | Path to `wordpipe-parakeet-worker`. Empty input is normalized to the service default. |
| `model_installer_path` | `s` | Path to the model install/export wrapper. Empty input is normalized to the service default. |
| `sample_rate` | `u` | Capture/ASR sample rate. Must be positive. |
| `num_threads` | `u` | Worker ORT thread count. Must be positive. |
| `spoken_punctuation` | `b` | Enable spoken punctuation normalization before text events. |
| `insert_partials` | `b` | Insert append-only `TextDelta` events before final stop. |
| `stream_insert_delay_ms` | `u` | Optional delay before inserting streaming deltas. |
| `show_overlay` | `b` | Whether the Shell extension should show the dictation overlay. |

`ListBackends` returns one map per backend:

| Key | Type | Meaning |
| --- | --- | --- |
| `id` | `s` | Backend id used by `SetBackend`. |
| `title` | `s` | User-facing backend label. |
| `description` | `s` | User-facing backend description. |

`ListModelProfiles` returns one map per profile/family preset:

| Key | Type | Meaning |
| --- | --- | --- |
| `id` | `s` | Preset id used by `SetModelProfile` and `InstallModel`. |
| `title` | `s` | User-facing preset label. |
| `description` | `s` | User-facing profile description. |
| `model_profile` | `s` | Performance profile, `fast` or `compact`. |
| `model_family` | `s` | Checkpoint family, `multilingual` or `english`. |
| `build_profile` | `s` | Installer/export recipe name. |
| `output_name` | `s` | Installed model directory base name. |
| `prebuilt_repo` | `s` | Default Hugging Face model repo for raw prebuilt profile files. |
| `ort_format` | `b` | Whether the runtime dir points at ORT-format output. |
| `runtime_dir` | `s` | Full expected installed runtime directory for this profile. |
| `installed` | `b` | Whether `runtime_dir` has the expected installed profile marker/files. |

`ListInputDevices` returns one map per CPAL input device:

| Key | Type | Meaning |
| --- | --- | --- |
| `index` | `u` | Enumeration index from CPAL. |
| `name` | `s` | Device name. |
| `selector` | `s` | Value accepted by `SetInputDevice`. |
| `is_default` | `b` | Whether the device matches the current system default name. |

`GetState` includes `selected_runtime_dir`, `selected_model_installed`, and
`installing_profile`, so clients can disable start/install controls and steer
users to model setup before the service attempts to spawn the worker. It also
includes `last_metrics`, the most recent metrics payload, so preferences opened
after model load or dictation start can show the current runtime snapshot
without waiting for another `Metrics` signal. Similarly,
`last_install_progress` stores the latest installer progress payload, including
the model profile, so setup UI opened mid-install can show the current phase
without waiting for another `InstallProgress` signal.
`partial_text` and `last_commit_text` expose the current transcript snapshot for
preferences and clients that connect after the corresponding transcript signal.
Configuration setters emit `ConfigChanged` followed by `StateChanged`; this is
important when a setting change stops the current worker or changes selected
model readiness.
`InstallModel` emits initial `InstallProgress` and `StateChanged` when the
installer starts, so clients can disable start/install controls immediately.
GNOME clients treat D-Bus proxy/connection failures as service-unavailable, but
ordinary method failures remain in the connected state and are surfaced as
status text.
Start-time failures such as a missing selected model profile or worker spawn
failure are also recorded in `last_error` and emitted through `Error` plus
`StateChanged`, so clients other than the caller see the failure.
The service applies spoken-punctuation normalization before computing
`TextDelta` when `spoken_punctuation` is enabled. Partial normalization holds
ambiguous trailing command prefixes such as `new`, `question`, `full`, and
`exclamation` until they either become a complete command or ordinary text.
The GNOME client inserts `TextDelta` only when `insert_partials` is enabled,
optionally delaying those append-only insertions by `stream_insert_delay_ms`.
The overlay displays `Partial` full-text updates rather than insertion deltas.
The client inserts `Commit` text when it adds text that has not already been
streamed and cancels any delayed deltas before applying a final commit.
`Stop()` moves state to `stopping=true` while the worker flushes. The service
emits final `Commit` before `SessionStopped` when the worker produces one.
`Toggle()` treats both `listening` and `stopping` as stop states, and `Start()`
rejects requests during the stopping window so a repeated hotkey cannot restart
dictation before the previous flush has completed.
If the ASR worker exits while a session is active, the service emits
`SessionStopped`, records `last_error`, emits `Error`, and then publishes
`StateChanged` with all active/loading flags cleared. An exit while already
`stopping` is treated as expected shutdown.

## Migration Steps

1. Add a Rust crate for the service protocol and D-Bus server. Done.
2. Move the current Rust worker loop into long-lived service state. Next.
3. Preserve the current JSON-line worker mode for benchmarks during migration.
   Done.
4. Add a minimal GNOME Shell extension that can connect to D-Bus and show
   service state. Done.
5. Add extension preferences for model profile, mic, and insertion options.
   Done.
6. Replace portal keyboard insertion with GNOME Shell insertion adapter. Done
   for append-only `TextDelta` commits; live cross-app validation remains.
7. Add local installer script for the service plus extension. Done.
8. Keep KDE/other desktop clients as separate adapters using the same D-Bus API.
   Ongoing.

## Current Experiment State

The branch now has three Rust workspace crates:

```text
crates/wordpipe-protocol
  Shared D-Bus constants and model/backend profile metadata.

crates/wordpipe-service
  Session D-Bus service. It owns config/state, lists CPAL input devices,
  exposes model/backend setup methods, supervises the Rust ASR worker, and
  translates worker JSON events into the planned streaming signals.

crates/wordpipe-parakeet-worker
  Existing JSON-line ASR worker retained for benchmarks and as the code source
  for the upcoming service runtime migration.
```

The GNOME Shell extension lives at:

```text
extensions/gnome-shell/wordpipe@dhansen.dev
```

It currently provides:

- Panel indicator and menu for start/stop, service-provided model profile
  selection, and missing-profile install actions.
- GNOME Shell global shortcut using the extension's GSettings key.
- Preferences UI for service-provided backend/model profile lists,
  microphone selection, streaming insertion options, runtime options, overlay,
  shortcut, service status, and model setup progress.
- Overlay/status updates driven by D-Bus state and transcript signals.
- Config synchronization from `ConfigChanged`, so preferences and the shell
  client stay aligned with service-side changes.

The `TextInjector` in `extension.js` is intentionally isolated. The first
GNOME Shell 50 implementation commits append-only text deltas through
`Clutter.get_default_backend().get_input_method().commit(text)`. This is an
internal Shell/input-method path, so it needs live GNOME testing across focused
GTK apps, terminals, and browser text fields.

## Local Install

Before installing, verify that the GJS clients still match the Rust D-Bus
protocol declaration:

```bash
python3 scripts/check_gnome_dbus_xml.py
```

Install the service and extension for the current user:

```bash
scripts/install-wordpipe-gnome
```

The installer:

- Builds `wordpipe-service` in release mode.
- Builds `wordpipe-parakeet-worker` in release mode.
- Installs both Rust binaries to `~/.local/libexec/wordpipe/`.
- Installs a `wordpipe-model-install` wrapper next to the service. The wrapper
  calls the Python model installer from this checkout for local development.
- Installs a session D-Bus activation file for `dev.wordpipe.Service`.
- Installs a systemd user unit at
  `~/.config/systemd/user/wordpipe-service.service`.
- Copies the extension to
  `~/.local/share/gnome-shell/extensions/wordpipe@dhansen.dev`.
- Runs `glib-compile-schemas` for the installed extension.
- Enables the extension with `gnome-extensions enable`.

Open preferences after install:

```bash
gnome-extensions prefs wordpipe@dhansen.dev
```

Check the installed extension files, Shell discovery, service state, config,
and model profile list with:

```bash
scripts/wordpipe-gnome-status
```

In the current GNOME Shell session, newly copied extensions may not be
discovered until logging out and back in. This is expected on Wayland because
GNOME Shell cannot be reliably reloaded in place. If `gnome-extensions info
wordpipe@dhansen.dev` reports that the extension does not exist immediately
after install, log out/in so Shell rescans user extensions, then run:

```bash
gnome-extensions enable wordpipe@dhansen.dev
```

Verify the service side independently:

```bash
gdbus call --session --dest dev.wordpipe.Service \
  --object-path /dev/wordpipe/Service \
  --method dev.wordpipe.Service1.GetState
```

Start the service explicitly for development:

```bash
systemctl --user start wordpipe-service.service
```

Or let D-Bus activate it when the extension first calls the service.

## Service Configuration

The Rust service persists user-facing configuration in:

```text
~/.config/wordpipe/service.json
```

Use `wordpipe-service --config /path/to/service.json` for isolated testing.
When no saved `model_family` exists, the service infers it once from the saved
language so existing installations keep using the same model path. The service
keeps the selected family while looking for an installed performance profile
under `model_root`. An explicitly saved valid preset remains authoritative
even when its model files are not installed yet, so the UI can still drive that
profile's install flow.
The GNOME extension mirrors service config on startup before pushing GSettings
changes back, so an extension reload should not overwrite the service's saved
model profile, microphone, shortcut, model root, sample rate, thread count, or
insertion options with schema defaults.

Runtime options that affect the worker process are updated through:

```text
SetRuntimeOptions(a{sv})
```

Supported keys are `model_root`, `worker_path`, `model_installer_path`,
`sample_rate`, and `num_threads`. Changes that affect the active worker stop the
current worker so the next dictation session starts with the new runtime
settings. Empty or whitespace-only `model_root`, `worker_path`, and
`model_installer_path` values are normalized to the corresponding service
defaults.

## Open Questions

- Which GNOME Shell versions should the first extension target? The current dev
  machine is expected to be GNOME 50-era, but the extension should not claim
  unsupported future versions.
- Which GNOME Shell internal OSK API is stable enough for a first experiment?
- Should release packaging keep the Python model installer as the ORT
  conversion bridge, or should download/extract/ORT conversion move into Rust?
- Should the service be D-Bus activated only, systemd-user managed only, or
  support both?
