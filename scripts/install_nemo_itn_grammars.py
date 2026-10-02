#!/usr/bin/env python3
"""Stage NVIDIA's pinned, precompiled ITN grammars for a binary release."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

URL = 'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/itn_configs.tar.bz2'
SIZE = 2371684
SHA256 = '880c9365d1d52c17450bd930950b0e58eca294421b7be62eb71666fb21b8997f'


def install(destination, source=None):
    destination = Path(destination)
    marker = destination / '.nemote-grammar-sha256'
    legacy_marker = destination / '.wordpipe-grammar-sha256'
    if not marker.exists() and legacy_marker.is_file():
        marker = legacy_marker
    if marker.is_file() and marker.read_text().strip() == SHA256:
        return destination
    if destination.exists():
        raise RuntimeError(f'Unverified ITN directory already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.itn-', dir=destination.parent) as work:
        work = Path(work)
        archive = work / 'grammars.tar.bz2'
        reader = Path(source).open('rb') if source else urlopen(URL, timeout=60)
        digest = hashlib.sha256()
        count = 0
        with reader, archive.open('wb') as output:
            while chunk := reader.read(1024 * 1024):
                count += len(chunk)
                if count > SIZE:
                    raise RuntimeError('ITN download exceeds pinned size')
                digest.update(chunk)
                output.write(chunk)
        if count != SIZE or digest.hexdigest() != SHA256:
            raise RuntimeError('ITN grammar checksum/size verification failed')
        with tarfile.open(archive, 'r:bz2') as bundle:
            members = bundle.getmembers()
            if sum(member.size for member in members) > 64 * 1024 * 1024:
                raise RuntimeError('ITN archive exceeds extraction limit')
            for member in members:
                path = Path(member.name)
                if (path.is_absolute() or '..' in path.parts or not path.parts
                        or path.parts[0] != 'itn_configs'
                        or not (member.isfile() or member.isdir())):
                    raise RuntimeError('Unsafe ITN archive entry')
            # Members are explicitly checked above, including rejecting links;
            # Debian's Python 3.11 does not yet have extraction filters.
            bundle.extractall(work, members=members)
        staged = work / 'itn_configs'
        if not all((staged / 'en' / name).is_file()
                   for name in ('tokenize_and_classify.far', 'verbalize.far')):
            raise RuntimeError('ITN archive is missing English grammars')
        (staged / marker.name).write_text(SHA256 + '\n')
        shutil.move(str(staged), destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    print(install(args.destination, args.source))
