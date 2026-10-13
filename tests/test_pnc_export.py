import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("pnc_export", Path(__file__).resolve().parents[1] / "scripts/export_nemo_pnc.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@unittest.skipUnless(importlib.util.find_spec("yaml"), "developer export dependency PyYAML unavailable")
class LegacyPncExportTests(unittest.TestCase):
    def test_repack_preserves_weights_and_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"language_model": {"pretrained_model_name": "bert-base-uncased"},
                      "tokenizer": {"vocab_file": "tokenizer.vocab_file"},
                      "punct_label_ids": {"?": 2, "O": 0, ",": 1}, "capit_label_ids": {"U": 1, "O": 0}}
            with tarfile.open(root / "old.nemo", "w") as archive:
                for name, data in (("model_config.yaml", json.dumps(config).encode()),
                                   ("model_weights.ckpt", b"unchanged weights"), ("tokenizer.vocab_file", b"[UNK]\nhello\n")):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            encoder = {"model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12,
                       "num_attention_heads": 12, "vocab_size": 2}
            (root / "encoder.json").write_text(json.dumps(encoder))
            export.repack(root / "old.nemo", root / "new.nemo", root / "encoder.json")
            with tarfile.open(root / "new.nemo") as archive:
                self.assertEqual(archive.extractfile("model_weights.ckpt").read(), b"unchanged weights")
                self.assertEqual(archive.extractfile("punct_label_ids.csv").read(), b"O\n,\n?\n")
                self.assertEqual(archive.extractfile("capit_label_ids.csv").read(), b"O\nU\n")
                self.assertEqual(archive.extractfile("vocab.txt").read(), b"[UNK]\nhello\n")
            encoder["model_type"] = "distilbert"
            (root / "encoder.json").write_text(json.dumps(encoder))
            with self.assertRaisesRegex(ValueError, "BERT-base"):
                export.repack(root / "old.nemo", root / "bad.nemo", root / "encoder.json")
