#!/usr/bin/env sh
set -eu

mkdir -p /tmp/nemote-config /tmp/nemote-data /tmp/nemote-cache
exec "$@"
