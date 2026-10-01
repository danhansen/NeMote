#!/usr/bin/env python3
"""Check native worker against the recorded upstream GGUF corpus, plus reset parity."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from score_benchmark_wer import load_eval_helpers


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=2, choices=(1, 2))
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    args = parser.parse_args()
    helpers = load_eval_helpers()
    manifest = json.loads(args.manifest.read_text())
    baseline = json.loads(args.baseline.read_text())
    if baseline["model_sha256"] != sha(args.model) or baseline["manifest_sha256"] != sha(args.manifest):
        raise SystemExit("Baseline model/corpus identity mismatch")
    expected = {(row["chunk_frames"], row["clip"]): row for row in baseline["results"]}
    report = {"status": "running", "worker_sha256": sha(args.worker), "model_sha256": sha(args.model),
              "manifest_sha256": sha(args.manifest), "baseline_sha256": sha(args.baseline),
              "threads": 2, "concurrency": 1, "batching": False, "requested_device": args.device, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    for context in (1, 2, 7, 14):
        for clip in manifest[:args.limit]:
            if sha(args.worker) != report["worker_sha256"]:
                raise SystemExit("Worker binary changed during verification; restart with an immutable build")
            if sha(clip["audio"]) != clip["sha256"]:
                raise SystemExit("Corpus audio identity mismatch")
            # Two independent streams on the SAME recognizer test cache/state isolation.
            command = [str(args.worker), "--model-dir", str(args.model), "--wav", clip["audio"],
                       "--device", args.device,
                       "--num-threads", "2", "--chunk-samples", str(context * 1280), "--wav-repeat", str(args.repeat)]
            process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
            events = [json.loads(line) for line in process.stdout.splitlines()]
            commits = [event for event in events if event["event"] == "commit"]
            if len(commits) != args.repeat or len({event["text"] for event in commits}) != 1:
                raise SystemExit(f"Repeated-session transcript mismatch: {clip['utt_id']}")
            previous = ""
            revisions = []
            for event in events:
                if event["event"] not in ("partial", "commit"):
                    continue
                if not event["text"].startswith(previous):
                    revisions.append({"event": event["event"], "previous": previous, "text": event["text"]})
                previous = "" if event["event"] == "commit" else event["text"]
            text = commits[0]["text"]
            reference = expected[context, clip["utt_id"]]
            edits, words, wer = helpers.wer_stats(clip["reference"], text)
            row = {"chunk_frames": context, "clip": clip["utt_id"], "text": text,
                   "upstream_text": reference["text"], "edits": edits, "words": words, "wer": wer,
                   "upstream_edits": reference["edits"],
                   "normalized_match": helpers.normalize_words(text) == helpers.normalize_words(reference["text"]),
                   "cold_metrics": commits[0]["data"], "warm_metrics": commits[-1]["data"],
                   "text_revisions": revisions}
            report["results"].append(row)
            save()
            print(f"C={context} {clip['utt_id']}: match={row['normalized_match']} edits={edits}/{words}", flush=True)
    report["status"] = "complete"
    report["mismatches"] = sum(not row["normalized_match"] for row in report["results"])
    report["text_revisions"] = sum(len(row["text_revisions"]) for row in report["results"])
    save()
    if report["mismatches"]:
        raise SystemExit(f"{report['mismatches']} upstream transcript mismatches; inspect report")
    if report["text_revisions"]:
        raise SystemExit(f"{report['text_revisions']} emitted text prefix violations; inspect report")


if __name__ == "__main__":
    main()
