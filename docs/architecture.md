# Architecture

NeMote is a GNOME Shell/Wayland dictation app with three small process boundaries:

```text
GNOME Shell extension (JavaScript)
    │ D-Bus: dev.nemote.Service1
Native service (C++, GLib/GIO main loop)
    ├── Native NeMo worker (C++, newline-delimited JSON)
    │     └── miniaudio capture → NeMo features/recognizer → transcript snapshots
    └── Python model installer (standard library, runs only during downloads)
```

The service owns persisted settings, worker lifetime, model-install progress,
session IDs, and signals. Subprocess output is read asynchronously on the GLib
main loop; device enumeration and downloads do not block D-Bus dispatch.
The service has no inference or audio dependencies. Its D-Bus contract is in
`extensions/gnome-shell/nemote@dhansen.dev/service.xml`. CMake embeds it in the
native service; both GNOME clients load it through `serviceClient.js`.
`serviceSettings.js` shares settings synchronization and option groups, using
the GSettings schema for variant types. `serviceValues.js` shares protocol
decoding and display helpers without importing either UI toolkit.

The worker owns capture, its bounded audio queue, model state, and finalization.
The recognizer stays loaded between dictations; each start creates a fresh stream.
The service and worker support one active dictation, and the NeMo state arenas
have one slot. Batching and GPU execution are disabled. Runtime CPU plugins
choose compatible kernels without client compilation.

The native service owns a Global Shortcuts portal session. It registers its
connection as `dev.nemote.Service` through the host Registry API, using the
installed desktop entry for identity. First-time binding is user-initiated;
existing portal assignments are restored after service or portal restarts.
Preferences delegates editing to the portal's `ConfigureShortcuts` UI on v2.
On v1, initial setup still uses `BindShortcuts`; later edits open
`gnome-control-center applications dev.nemote.Service` without unbinding the
shortcut. The desktop entry is visible so GNOME's Apps page can list it. Press
and release signals implement toggle/push-to-talk without Shell key grabs or
raw key-event monitoring. Losing the portal releases any held push-to-talk
session. The shortcut behavior remains a GSettings preference, sent to the
service when the Shell connects or preferences changes it.

The Shell extension owns the panel indicator and text insertion via GNOME's
input-method interface. Partials are replaceable preedit; commits insert only the
uncommitted suffix. No clipboard, X11 automation, or extra keyboard daemon is used.
Final EOF text is forwarded before `SessionStopped`; worker generations are
isolated so a retired worker cannot insert into a subsequent session.

Downloads use pinned upstream files, resumable transfers, an exclusive per-model
lock, disk-space checks, and SHA256 verification before atomic installation.
Python is not part of recognition. There are no ONNX conversion/export code paths.

Old Parakeet configuration is migrated to the sole Nemotron backend. Existing
models are not deleted. The previous implementation and full research tooling
are preserved on `archive/parakeet-rs-v0.1.28`; retained performance documents are
historical and may refer to that branch's source layout.
