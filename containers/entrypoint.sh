#!/usr/bin/env sh
set -eu

mkdir -p /tmp/wordpipe-config /tmp/wordpipe-data /tmp/wordpipe-cache
exec "$@"
