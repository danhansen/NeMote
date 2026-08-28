# Wordpipe Development Container

`Containerfile.dev` is the canonical environment for tests and release builds.
It provides the pinned Rust toolchain, Python model tooling, ALSA headers,
GNOME schema compiler, Node syntax checker, and archive tools used by CI.

Use the repository wrapper from the project root:

```sh
scripts/dev-container build
scripts/dev-container test
scripts/dev-container shell
scripts/dev-container package dev-local dist
```

Set `CONTAINER_ENGINE=docker` or `CONTAINER_ENGINE=podman` to override automatic
engine selection. Set `WORDPIPE_DEV_IMAGE` to use a different local image tag.
After explicitly building an image, `WORDPIPE_CONTAINER_SKIP_BUILD=1` reuses it.

The source tree is mounted at `/workspace`; model files are not copied into the
image. The CI and release workflows call the same wrapper and
`scripts/run-build-pipeline` used locally.
