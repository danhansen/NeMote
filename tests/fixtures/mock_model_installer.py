#!/usr/bin/env python3
"""A local installer process tree for cancellation tests; no network access."""
import json
import os
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print("nemote-progress " + json.dumps({
    "phase": "running", "fraction": 0.0,
    "installer_pid": os.getpid(), "child_pid": child.pid,
    "message": "lifecycle fixture",
}), flush=True)
time.sleep(60)
