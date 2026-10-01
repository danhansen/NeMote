#!/usr/bin/env python3
"""Build NeMo's English cased ITN FARs once, in CI, not on clients."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REVISION = 'ddadfb2a38d2bc6b8cc6232c4f915eb60f500688'
FAR_FILES = ('tokenize_and_classify.far', 'verbalize.far')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = destination / '.wordpipe-cased-itn.json'
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if (saved.get('revision') == REVISION and saved.get('input_case') == 'cased'
                and set(saved.get('files', {})) == set(FAR_FILES)
                and all((destination / name).is_file() and sha(destination / name) == digest
                        for name, digest in saved['files'].items())):
            return
    if not source.exists():
        subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                        'https://github.com/NVIDIA/NeMo-text-processing.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth', '1', 'origin', REVISION], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError('ITN grammar source does not match pinned revision')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    sys.path.insert(0, str(source.resolve()))
    import pynini
    from nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
    from nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize import VerbalizeFst
    from nemo_text_processing.text_normalization.en.graph_utils import generator_main
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cased-itn-', dir=destination.parent) as work:
        work = Path(work)
        generator_main(str(work / 'tokenize_and_classify.far'), {'TOKENIZE_AND_CLASSIFY': ClassifyFst(input_case='cased').fst})
        generator_main(str(work / 'verbalize.far'), {'ALL': VerbalizeFst().fst, 'REDUP': pynini.accep('REDUP')})
        files = {name: sha(work / name) for name in FAR_FILES}
        for name in files:
            (work / name).replace(destination / name)
        manifest.write_text(json.dumps({'revision': REVISION, 'input_case': 'cased', 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
