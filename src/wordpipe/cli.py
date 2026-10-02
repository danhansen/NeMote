"""Small, standard-library-only model download command."""
import argparse
import os
from pathlib import Path
import sys

from .models import MODEL_FAMILIES, STREAMING_LATENCIES, profile_runtime_dir


def build_parser():
    parser = argparse.ArgumentParser(prog="wordpipe")
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("model-install", help="Download the English or Multilingual Nemotron model")
    # Kept for compatibility with installed clients; there is only one profile.
    install.add_argument("--profile", choices=("nemo-q8",), default="nemo-q8", help=argparse.SUPPRESS)
    install.add_argument("--model-family", choices=(*MODEL_FAMILIES, "all"), default="english")
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    install.add_argument("--model-root", default=str(data_home / "wordpipe/models"))
    install.add_argument("--source", help="Install a local official GGUF instead of downloading")
    install.add_argument("--force", action="store_true")
    install.add_argument("--force-source", action="store_true", help=argparse.SUPPRESS)
    install.add_argument("--streaming-latency-ms", type=int, choices=STREAMING_LATENCIES, default=560, help=argparse.SUPPRESS)
    install.add_argument("--dry-run", action="store_true")
    return parser


def _cmd_model_install_family(args, family):
    from .nemo_models import install_nemo_model
    root = Path(args.model_root).expanduser()
    if args.dry_run:
        print(profile_runtime_dir(root, family=family))
        return 0
    print(install_nemo_model(root, family,
        source=Path(args.source).expanduser() if args.source else None,
        force=args.force or args.force_source,
        progress=lambda message: print(message, file=sys.stderr, flush=True)))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    families = MODEL_FAMILIES if args.model_family == "all" else (args.model_family,)
    if args.source and len(families) != 1:
        raise SystemExit("--source requires one model family")
    try:
        for family in families:
            _cmd_model_install_family(args, family)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"wordpipe: {error}", file=sys.stderr, flush=True)
        return 1
    return 0
