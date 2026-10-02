import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('itn_grammars', Path(__file__).resolve().parents[1] / 'scripts/install_nemo_itn_grammars.py')
grammars = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grammars)


class ItnGrammarTests(unittest.TestCase):
    def test_legacy_cache_marker_is_reused_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / '.wordpipe-grammar-sha256').write_text(grammars.SHA256 + '\n')
            self.assertEqual(grammars.install(destination, destination / 'missing-source'), destination)

    def archive(self, entries):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:bz2') as bundle:
            for name, content in entries:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        return output.getvalue()

    def test_verified_install_is_atomic_and_idempotent(self):
        data = self.archive([('itn_configs/en/tokenize_and_classify.far', b'classifier'),
                             ('itn_configs/en/verbalize.far', b'verbalizer')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.tar.bz2'
            source.write_bytes(data)
            destination = root / 'grammars'
            with patch.object(grammars, 'SIZE', len(data)), patch.object(grammars, 'SHA256', hashlib.sha256(data).hexdigest()):
                self.assertEqual(grammars.install(destination, source), destination)
                self.assertEqual(grammars.install(destination, root / 'missing-source'), destination)
                self.assertEqual((destination / 'en/verbalize.far').read_bytes(), b'verbalizer')
                self.assertEqual(list(root.glob('.itn-*')), [])

    def test_bad_checksum_never_installs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'bad.tar.bz2'
            source.write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                grammars.install(root / 'grammars', source)
            self.assertFalse((root / 'grammars').exists())

    def test_rejects_traversal_even_for_a_checksum_matching_archive(self):
        data = self.archive([('itn_configs/../../outside', b'unsafe')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.tar.bz2'
            source.write_bytes(data)
            with patch.object(grammars, 'SIZE', len(data)), patch.object(grammars, 'SHA256', hashlib.sha256(data).hexdigest()):
                with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                    grammars.install(root / 'grammars', source)
            self.assertFalse((root / 'grammars').exists())
