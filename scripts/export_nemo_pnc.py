#!/usr/bin/env python3
"""Adapt legacy NVIDIA BERT/DistilBERT PnC, then use the upstream converter.

Requires the NeMo-Speech.cpp conversion dependencies. Automatic installation
uses only a checksum-pinned NVIDIA checkpoint. Only use trusted checkpoints.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def distilbert_state_dict(state):
    """Translate the supported DistilBERT layout to the SDK's post-LN graph.

    Both graphs use the same attention/FFN operations at inference. DistilBERT
    omits segment embeddings, so the added SDK tensor is identically zero.
    Learned tensor values are preserved, not substituted with BERT weights.
    """
    import torch

    substitutions = {
        "attention.q_lin": "attention.self.query", "attention.k_lin": "attention.self.key",
        "attention.v_lin": "attention.self.value", "attention.out_lin": "attention.output.dense",
        "sa_layer_norm": "attention.output.LayerNorm", "ffn.lin1": "intermediate.dense",
        "ffn.lin2": "output.dense", "output_layer_norm": "output.LayerNorm",
    }
    expected = {f"bert_model.embeddings.{name}" for name in (
        "word_embeddings.weight", "position_embeddings.weight", "LayerNorm.weight", "LayerNorm.bias")}
    expected |= {f"bert_model.transformer.layer.{layer}.{name}.{kind}"
                 for layer in range(6) for name in substitutions for kind in ("weight", "bias")}
    expected |= {f"{task}_classifier.mlp.layer0.{kind}" for task in ("punct", "capit") for kind in ("weight", "bias")}
    if set(state) != expected:
        raise ValueError("Unsupported DistilBERT checkpoint tensor layout")
    result = {}
    for name, tensor in state.items():
        if name.startswith("bert_model.transformer.layer."):
            layer, tail = name.split(".")[3], name.split(".", 4)[4]
            sub, kind = tail.rsplit(".", 1)
            target = f"bert_model.encoder.layer.{layer}.{substitutions[sub]}.{kind}"
        else:
            target = name
        result[target] = tensor
    if tuple(state["bert_model.embeddings.word_embeddings.weight"].shape) != (30522, 768):
        raise ValueError("Unsupported DistilBERT vocabulary/hidden size")
    result["bert_model.embeddings.token_type_embeddings.weight"] = torch.zeros((1, 768), dtype=torch.float32)
    return result


def repack(checkpoint, destination, encoder_config):
    import yaml

    with tarfile.open(checkpoint, "r:*") as source, tarfile.open(destination, "w") as output:
        members = {Path(member.name).name: member for member in source if member.isfile()}
        required = {"model_weights.ckpt", "vocab.txt", "punct_label_ids.csv",
                    "capit_label_ids.csv", "encoder_config.json"}
        if required <= members.keys():
            raise ValueError("Checkpoint already uses the current layout; use upstream convert_model.py directly")
        if "model_config.yaml" not in members or "model_weights.ckpt" not in members:
            raise ValueError("Missing legacy checkpoint configuration or weights")
        config = yaml.safe_load(source.extractfile(members["model_config.yaml"]))
        model_name = config["language_model"]["pretrained_model_name"]
        if model_name not in ("bert-base-uncased", "distilbert-base-uncased"):
            raise ValueError("Legacy adapter supports NVIDIA's BERT-base and DistilBERT-base PnC checkpoints only")
        encoder = json.loads(Path(encoder_config).read_text())
        if (encoder.get("model_type"), encoder.get("hidden_size"), encoder.get("num_hidden_layers"),
                encoder.get("num_attention_heads")) != ("bert", 768, 12, 12):
            raise ValueError("Encoder configuration does not describe BERT-base")

        def add_bytes(name, data):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))

        weight = members["model_weights.ckpt"]
        if model_name == "distilbert-base-uncased":
            import torch
            # Materialize once: torch's random seeks through a compressed tar
            # member otherwise repeatedly decompress the entire checkpoint.
            with tempfile.TemporaryDirectory(prefix="nemote-distilbert-layout-") as directory:
                original = Path(directory) / "original.ckpt"
                translated = Path(directory) / "translated.ckpt"
                import shutil
                with source.extractfile(weight) as reader, original.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
                state = torch.load(original, map_location="cpu", weights_only=True)
                translated_state = distilbert_state_dict(state)
                torch.save(translated_state, translated)
                member = tarfile.TarInfo("model_weights.ckpt")
                member.size = translated.stat().st_size
                with translated.open("rb") as reader:
                    output.addfile(member, reader)
            encoder["num_hidden_layers"] = 6
            encoder["type_vocab_size"] = 1
        else:
            member = tarfile.TarInfo("model_weights.ckpt")
            member.size = weight.size
            output.addfile(member, source.extractfile(weight))
        vocab = source.extractfile(members[Path(config["tokenizer"]["vocab_file"]).name]).read()
        if len(vocab.splitlines()) != encoder["vocab_size"]:
            raise ValueError("Tokenizer vocabulary size does not match the encoder")
        add_bytes("vocab.txt", vocab)
        add_bytes("encoder_config.json", json.dumps(encoder).encode())
        for task in ("punct", "capit"):
            labels = config[f"{task}_label_ids"]
            ordered = sorted(labels, key=labels.get)
            if [labels[label] for label in ordered] != list(range(len(ordered))):
                raise ValueError("Label IDs must be contiguous from zero")
            add_bytes(f"{task}_label_ids.csv", ("\n".join(ordered) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=("q8_0", "f16", "bf16"), default="q8_0")
    args = parser.parse_args()
    if args.outfile.exists():
        parser.error("Output exists; choose a new path")
    with tempfile.TemporaryDirectory(prefix="nemote-pnc-export-") as directory:
        checkpoint = Path(directory) / "pnc.nemo"
        repack(args.checkpoint, checkpoint, args.encoder_config)
        subprocess.run([sys.executable, str(args.sdk / "convert_model.py"), str(checkpoint),
                        "--architecture", "pnc", "--outfile", str(args.outfile),
                        "--outtype", args.outtype], check=True)


if __name__ == "__main__":
    main()
