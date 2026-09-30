#!/usr/bin/env python3
"""A deterministic, microphone-free worker for private D-Bus lifecycle tests."""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-samples", type=int, required=True)
parser.add_argument("--ort-optimized-model-cache-dir", required=True)
args, _ = parser.parse_known_args()


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


time.sleep(0.02)
emit("model_loaded", data={"worker_pid": os.getpid(), "chunk_samples": args.chunk_samples,
                          "cache_dir": args.ort_optimized_model_cache_dir})
emit("ready")
for line in sys.stdin:
    command = json.loads(line).get("command")
    if command == "start":
        emit("listening")
    elif command == "stop":
        emit("stopped")
    elif command == "shutdown":
        break
