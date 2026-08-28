#!/usr/bin/env sh
set -eu

mkdir -p "$HOME" "$CARGO_HOME"
exec "$@"
