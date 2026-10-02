#!/usr/bin/env python3
"""A deterministic, microphone-free worker for private D-Bus lifecycle tests."""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-samples", type=int, default=8960)
parser.add_argument("--list-input-devices", action="store_true")
args, _ = parser.parse_known_args()
metrics = {"worker_pid": os.getpid(), "chunk_samples": args.chunk_samples}


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


if args.list_input_devices:
    emit("input_device", data={"index": 0, "name": "Fixture microphone", "is_default": True})
    sys.exit(0)
time.sleep(0.02)
emit("model_loaded", data={"worker_pid": os.getpid(), "chunk_samples": args.chunk_samples,
                          "state_arena_slots": 1})
emit("ready")
for line in sys.stdin:
    command = json.loads(line).get("command")
    if command == "start":
        if os.path.basename(sys.argv[0]) == "crash-worker":
            sys.exit(3)
        if os.path.basename(sys.argv[0]) == "error-worker":
            emit("error", message="Fixture capture error")
            continue
        emit("listening")
        emit("partial", text="I have twenty dollars", data=metrics)
        emit("partial", text="I have $20", data=metrics)
    elif command == "stop":
        emit("commit", text="I have $20", data=metrics)
        time.sleep(0.02)
        emit("stopped")
    elif command == "shutdown":
        break
