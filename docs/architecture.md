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
`native/nemote-service/interface.xml` and is checked against both GNOME clients.

The worker owns capture, its bounded audio queue, model state, and finalization.
The recognizer stays loaded between dictations; each start creates a fresh stream.
The service and worker support one active dictation, and the NeMo state arenas
have one slot. Batching and GPU execution are disabled. Runtime CPU plugins
choose compatible kernels without client compilation.

The Shell extension owns shortcuts, overlays, and text insertion via GNOME's
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
