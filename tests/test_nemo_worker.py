"""Optional real-SDK checks. Set WORDPIPE_NEMO_TEST_WORKER and WORDPIPE_NEMO_TEST_MODEL."""
import json
import os
import selectors
import subprocess
import time
import unittest


@unittest.skipUnless(os.environ.get("WORDPIPE_NEMO_TEST_WORKER") and os.environ.get("WORDPIPE_NEMO_TEST_MODEL"),
                     "native SDK/model integration fixtures not configured")
class NemoWorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen([
            os.environ["WORDPIPE_NEMO_TEST_WORKER"], "--model-dir", os.environ["WORDPIPE_NEMO_TEST_MODEL"],
            "--input-device", "wordpipe-nonexistent-test-input-device"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.assertEqual(self.event()["event"], "loading_model")
        self.assertEqual(self.event()["event"], "model_loaded")
        self.assertEqual(self.event()["event"], "ready")

    def tearDown(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.selector.close()

    def event(self):
        deadline = time.monotonic() + 15
        while b"\n" not in self.buffer:
            if not self.selector.select(max(0, deadline - time.monotonic())):
                self.fail("native worker event timed out")
            data = os.read(self.process.stdout.fileno(), 4096)
            if not data:
                self.fail("native worker stdout closed unexpectedly")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def send(self, command, **kwargs):
        self.process.stdin.write((json.dumps({"command": command, **kwargs}) + "\n").encode())
        self.process.stdin.flush()

    def test_idle_commands_language_validation_and_shutdown(self):
        self.send("set_language", language="es-US")
        self.assertEqual(self.event()["event"], "error")
        self.send("set_language", language="en-GB")
        self.assertEqual(self.event()["event"], "language_changed")
        self.send("unknown_command")
        self.assertEqual(self.event()["event"], "error")
        self.send("stop")
        self.assertEqual(self.event()["event"], "stopped")
        self.send("shutdown")
        self.assertEqual(self.process.wait(timeout=10), 0)

    def test_capture_failure_allows_another_session_without_reloading_model(self):
        for _ in range(2):
            self.send("start")
            self.assertEqual(self.event()["event"], "error")
            self.assertEqual(self.event()["event"], "stopped")
        self.send("set_language", language="en-US")
        self.assertEqual(self.event()["event"], "language_changed")

    def test_stdin_eof_exits_cleanly(self):
        self.process.stdin.close()
        self.assertEqual(self.process.wait(timeout=10), 0)
