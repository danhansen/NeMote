#!/usr/bin/env sh
set -eu

if ! getent passwd "$(id -u)" >/dev/null; then
    echo "NeMote development image has no passwd entry for UID $(id -u); rebuild it with scripts/dev-container build." >&2
    exit 1
fi

mkdir -p /tmp/nemote-config /tmp/nemote-data /tmp/nemote-cache
exec "$@"
