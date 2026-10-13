"""Small adapter-correctness smoke test against the unquantized checkpoint.

Not an accuracy benchmark; run explicitly with the conversion Python and local
model fixtures. The reference follows HF DistilBERT's inference equations.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


@unittest.skipUnless(all(os.environ.get(key) for key in (
    "NEMOTE_PNC_PROBE", "NEMOTE_PNC_LIBRARIES", "NEMOTE_NEMO_TEST_MODEL", "NEMOTE_DISTILBERT_CHECKPOINT", "NEMOTE_NEMO_TEST_PNC")),
    "DistilBERT checkpoint parity fixtures not configured")
class DistilBertParityTests(unittest.TestCase):
    def test_labels_against_unquantized_distilbert_and_lossless_layout_mapping(self):
        import torch
        import torch.nn.functional as fn
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory, tarfile.open(os.environ["NEMOTE_DISTILBERT_CHECKPOINT"]) as archive:
            weights = Path(directory) / "weights.ckpt"
            with archive.extractfile("model_weights.ckpt") as source, weights.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            vocab = archive.extractfile("tokenizer.vocab_file").read().decode().splitlines()
        spec = importlib.util.spec_from_file_location("pnc_export", Path(__file__).resolve().parents[1] / "scripts/export_nemo_pnc.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        translated = adapter.distilbert_state_dict(state)
        self.assertEqual(len(translated), len(state) + 1)
        self.assertTrue(torch.all(translated["bert_model.embeddings.token_type_embeddings.weight"] == 0))
        for layer in range(6):
            for source, target in (("attention.q_lin", "attention.self.query"), ("attention.k_lin", "attention.self.key"),
                                   ("attention.v_lin", "attention.self.value"), ("attention.out_lin", "attention.output.dense"),
                                   ("sa_layer_norm", "attention.output.LayerNorm"), ("ffn.lin1", "intermediate.dense"),
                                   ("ffn.lin2", "output.dense"), ("output_layer_norm", "output.LayerNorm")):
                for kind in ("weight", "bias"):
                    self.assertIs(translated[f"bert_model.encoder.layer.{layer}.{target}.{kind}"],
                                  state[f"bert_model.transformer.layer.{layer}.{source}.{kind}"])
        with self.assertRaisesRegex(ValueError, "layout"):
            adapter.distilbert_state_dict({})

        sentences = ("hello how are you", "can you meet me there at three pm", "my name is james",
                     "please send the report tomorrow", "the meeting is on monday at noon")
        sequences = [[vocab.index("[CLS]")] + [vocab.index(word) for word in text.split()] + [vocab.index("[SEP]")]
                     for text in sentences]

        def linear(x, name):
            return fn.linear(x, state[name + ".weight"].float(), state[name + ".bias"].float())

        def norm(x, name):
            return fn.layer_norm(x, (768,), state[name + ".weight"].float(), state[name + ".bias"].float(), 1e-12)

        def reference(ids):
            positions = torch.arange(len(ids))
            x = fn.embedding(torch.tensor(ids), state["bert_model.embeddings.word_embeddings.weight"].float())
            x += fn.embedding(positions, state["bert_model.embeddings.position_embeddings.weight"].float())
            x = norm(x, "bert_model.embeddings.LayerNorm")
            for layer in range(6):
                prefix = f"bert_model.transformer.layer.{layer}."
                q, k, v = [linear(x, prefix + "attention." + kind + "_lin").view(len(ids), 12, 64).transpose(0, 1)
                           for kind in ("q", "k", "v")]
                attention = torch.softmax((q @ k.transpose(-1, -2)) / 8, dim=-1) @ v
                out = linear(attention.transpose(0, 1).reshape(len(ids), 768), prefix + "attention.out_lin")
                x = norm(x + out, prefix + "sa_layer_norm")
                out = linear(fn.gelu(linear(x, prefix + "ffn.lin1")), prefix + "ffn.lin2")
                x = norm(x + out, prefix + "output_layer_norm")
            return {task: linear(x, task + "_classifier.mlp.layer0").argmax(-1).tolist() for task in ("punct", "capit")}

        result = subprocess.run([os.environ["NEMOTE_PNC_PROBE"], os.environ["NEMOTE_PNC_LIBRARIES"],
                                 os.environ["NEMOTE_NEMO_TEST_MODEL"], os.environ["NEMOTE_NEMO_TEST_PNC"]],
                                input="\n".join(json.dumps(ids) for ids in sequences) + "\n",
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(outputs), len(sequences))
        with torch.no_grad():
            for text, ids, actual in zip(sentences, sequences, outputs):
                expected = reference(ids)
                # Only real words are rendered; ignore special-token labels.
                for task in ("punct", "capit"):
                    self.assertEqual(actual[task][1:-1], expected[task][1:-1], (text, task, actual, expected))
