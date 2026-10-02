# Development container

Run `scripts/dev-container test` for build and regression checks, or
`scripts/dev-container all VERSION build/release` to package a release.
Podman or Docker is required. The image uses Debian Bookworm, C++/CMake, GLib/GIO,
and pinned Python dependencies for building ITN grammars. No Rust or ONNX stack
is required. Runtime model-download helpers use only Python's standard library.

Use `NEMOTE_CONTAINER_SKIP_BUILD=1` to reuse an existing image. Build parallelism
is controlled by `NEMOTE_BUILD_JOBS` (default 2). Existing pinned SDK builds can
be selected with `NEMOTE_NEMO_SOURCE`, `NEMOTE_NEMO_BUILD`, and
`NEMOTE_NEMO_WORKER_BUILD`; `NEMOTE_SERVICE_BUILD` selects the service build.

The image creates a passwd/group entry matching the builder's UID/GID and runs
as that user. This is required for private D-Bus integration tests under Docker;
a bare numeric `--user` without a passwd entry cannot authenticate. Rebuild the
image when moving it to a host with different UID/GID values.
