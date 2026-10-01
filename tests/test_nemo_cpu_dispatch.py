"""Optional relocated CPU plugin checks against a real SDK and model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NemoCpuBuildOptionTests(unittest.TestCase):
    def test_rejects_invalid_options_before_touching_sdk(self):
        script = Path(__file__).resolve().parents[1] / "scripts/build-nemo-worker"
        cases = (
            ({"WORDPIPE_NEMO_NATIVE": "invalid", "WORDPIPE_NEMO_CPU_VARIANTS": "OFF"},
             "WORDPIPE_NEMO_NATIVE must be"),
            ({"WORDPIPE_NEMO_NATIVE": "OFF", "WORDPIPE_NEMO_CPU_VARIANTS": "invalid"},
             "WORDPIPE_NEMO_CPU_VARIANTS must be"),
            ({"WORDPIPE_NEMO_NATIVE": "ON", "WORDPIPE_NEMO_CPU_VARIANTS": "ON"},
             "mutually exclusive"),
        )
        for options, message in cases:
            with self.subTest(options=options):
                result = subprocess.run(["bash", str(script)], env=dict(os.environ, **options),
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


@unittest.skipUnless(all(os.environ.get(key) for key in (
    "WORDPIPE_NEMO_DISPATCH_BUILD", "WORDPIPE_NEMO_DISPATCH_LIBRARIES",
    "WORDPIPE_NEMO_TEST_MODEL")), "dynamic CPU SDK/model fixtures not configured")
class NemoCpuDispatchTests(unittest.TestCase):
    def test_relocated_plugins_and_baseline_fallback(self):
        source = Path(os.environ["WORDPIPE_NEMO_DISPATCH_LIBRARIES"])
        with tempfile.TemporaryDirectory(prefix="wordpipe-cpu-dispatch-") as directory:
            prefix = Path(directory)
            subprocess.run(["cmake", "--install", os.environ["WORDPIPE_NEMO_DISPATCH_BUILD"],
                            "--prefix", str(prefix)], check=True, capture_output=True)
            libraries = prefix / "lib/nemo"
            libraries.mkdir(parents=True)
            for library in source.glob("*.so*"):
                shutil.copy2(library, libraries / library.name, follow_symlinks=True)
            normalizer = source.parent / 'deps/itn/lib'
            if normalizer.is_dir():
                for library in normalizer.glob('*.so*'):
                    shutil.copy2(library, libraries / library.name, follow_symlinks=True)
                dependencies = subprocess.run(['ldd', str(source / 'libnemo_speech_asr.so')],
                                              text=True, capture_output=True, check=True).stdout
                for line in dependencies.splitlines():
                    if 'libprotobuf' in line or 'libre2' in line or 'libz.so' in line:
                        library = Path(line.split()[2])
                        shutil.copy2(library, libraries / library.name)
                for library in libraries.glob('*.so*'):
                    subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(library)], check=True)
            command = [str(prefix / "bin/wordpipe-nemo-worker"), "--model-dir",
                       os.environ["WORDPIPE_NEMO_TEST_MODEL"], "--device", "cpu"]
            wav = os.environ.get("WORDPIPE_NEMO_TEST_WAV")
            if wav:
                command.extend(["--wav", wav])
            environment = dict(os.environ, LD_DEBUG="libs")
            environment.pop("LD_LIBRARY_PATH", None)

            def run():
                result = subprocess.run(command, input='{"command":"shutdown"}\n',
                                        text=True, capture_output=True, timeout=60, env=environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                events = [json.loads(line) for line in result.stdout.splitlines()]
                loaded = next(event["data"] for event in events if event["event"] == "model_loaded")
                # The dynamic loader must initialize the relocated SDK/plugins,
                # not silently use a developer's build directory.
                initialized = [os.path.normpath(line.split("calling init:", 1)[1].strip())
                               for line in result.stderr.splitlines() if "calling init:" in line]
                self.assertTrue(any(str(libraries / "libnemo_speech_asr.so") in line
                                    for line in initialized), result.stderr)
                self.assertFalse(any(str(source.parent.resolve()) in line for line in initialized))
                self.assertTrue(any(str(libraries / "libggml-cpu-") in line
                                    for line in initialized), result.stderr)
                text = next((event["text"] for event in events if event["event"] == "commit"), None)
                if wav:
                    self.assertTrue(text)
                return loaded, text

            selected, selected_text = run()
            self.assertEqual(selected["compute_backend"], "cpu")
            self.assertIsInstance(selected["cpu_features"], dict)
            # Only the universal x86-64 module remains. This exercises fallback
            # without changing the original SDK or requiring an older machine.
            for plugin in libraries.glob("libggml-cpu-*.so"):
                if plugin.name != "libggml-cpu-x64.so":
                    plugin.unlink()
            fallback, fallback_text = run()
            self.assertEqual(fallback_text, selected_text)
            for feature in ("AVX", "AVX2", "AVX_VNNI", "AVX512"):
                self.assertNotEqual(fallback["cpu_features"].get(feature), "1")


if __name__ == "__main__":
    unittest.main()
