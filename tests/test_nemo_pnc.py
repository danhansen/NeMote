"""Optional WAV-driven adapter checks, not a benchmark of NVIDIA's PnC accuracy."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(all(os.environ.get(key) for key in (
    "NEMOTE_NEMO_TEST_WORKER", "NEMOTE_NEMO_TEST_MODEL", "NEMOTE_NEMO_TEST_PNC", "NEMOTE_NEMO_TEST_WAV")),
    "ASR/PnC/WAV integration fixtures not configured")
class NemoPncTests(unittest.TestCase):
    def test_cached_automatic_model_works_without_conversion_dependencies(self):
        source = Path(os.environ["NEMOTE_NEMO_TEST_PNC"]).resolve()
        receipt = source.with_name(source.name + ".json")
        if not receipt.is_file():
            self.skipTest("Automatic-install receipt fixture unavailable")
        variant = "distilbert" if "distilbert" in source.name else "bert"
        from nemote.pnc_install import MODELS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.gguf").symlink_to(Path(os.environ["NEMOTE_NEMO_TEST_MODEL"]).resolve())
            target = root / MODELS[variant][0]
            target.symlink_to(source)
            shutil.copy2(receipt, target.with_name(target.name + ".json"))
            helper = Path(__file__).resolve().parents[1] / "scripts/nemote-companion-install"
            env = dict(os.environ, XDG_CACHE_HOME=str(root / "cache"), NEMOTE_PNC_PYTHON="/nonexistent-python",
                       https_proxy="http://127.0.0.1:9")
            result = subprocess.run([os.environ["NEMOTE_NEMO_TEST_WORKER"], "--model-dir", str(root),
                "--model-family", "english", "--pnc", "--pnc-variant", variant, "--companion-installer", str(helper),
                "--wav", os.environ["NEMOTE_NEMO_TEST_WAV"]], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            loaded = next(event for event in events if event["event"] == "model_loaded")
            self.assertEqual(loaded["data"]["pnc_variant"], variant)
            self.assertTrue(loaded["data"]["pnc"])
            self.assertTrue(any(event["event"] == "commit" and event["text"] for event in events))

    def test_asr_gguf_is_rejected_as_a_pnc_model(self):
        result = subprocess.run([os.environ["NEMOTE_NEMO_TEST_WORKER"], "--model-dir",
            os.environ["NEMOTE_NEMO_TEST_MODEL"], "--pnc", "--pnc-model",
            os.environ["NEMOTE_NEMO_TEST_MODEL"]], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertIn("architecture must be pnc", json.loads(result.stdout)["message"])

    def run_wav(self, mode, pnc, itn=False):
        command = [os.environ["NEMOTE_NEMO_TEST_WORKER"], "--model-dir", os.environ["NEMOTE_NEMO_TEST_MODEL"],
                   "--wav", os.environ["NEMOTE_NEMO_TEST_WAV"], "--endpoint-mode", mode]
        if pnc:
            command += ["--pnc", "--pnc-model", os.environ["NEMOTE_NEMO_TEST_PNC"]]
        if itn:
            command += ["--itn"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertFalse(any(event["event"] == "error" for event in events), result.stdout)
        loaded = next(event for event in events if event["event"] == "model_loaded")
        self.assertEqual(loaded["data"]["pnc"], pnc)
        return events

    def check_modes(self, itn):
        lexical = lambda text: ''.join(c.casefold() for c in text if c.isalnum())
        for mode in ("disabled", "preview", "commit"):
            with self.subTest(mode=mode, itn=itn):
                native = self.run_wav(mode, False, itn)
                pnc = self.run_wav(mode, True, itn)
                native_partial = next(event["text"] for event in native if event["event"] == "partial")
                pnc_partial = next(event["text"] for event in pnc if event["event"] == "partial")
                self.assertEqual(pnc_partial, native_partial, "immediate partial must remain model-native")
                native_final = [event["text"] for event in native if event["event"] == "commit"][-1]
                pnc_final = [event["text"] for event in pnc if event["event"] == "commit"][-1]
                self.assertEqual(lexical(pnc_final), lexical(native_final), "PnC must not rewrite transcript words")
                self.assertTrue(pnc_final)
                metrics = [event["data"] for event in pnc if event["event"] == "commit"][-1]
                native_metrics = [event["data"] for event in native if event["event"] == "commit"][-1]
                print(f"PnC WAV mode={mode} itn={itn}: {pnc_final}", flush=True)
                print(f"Decode seconds: native={native_metrics['decode_seconds']:.3f} pnc={metrics['decode_seconds']:.3f}", flush=True)

    def test_pnc_preserves_words_and_immediate_partials_in_all_endpoint_modes(self):
        self.check_modes(False)

    @unittest.skipUnless(os.environ.get("NEMOTE_NEMO_TEST_ITN"), "ITN fixtures not configured")
    def test_itn_then_pnc_in_all_endpoint_modes(self):
        self.check_modes(True)
