import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_nemotron_eof_errors import alignment


class EndingAlignmentTests(unittest.TestCase):
    def test_exact(self):
        r=alignment('a b c'.split(),'a b c'.split())
        self.assertEqual(r['tail_error_bounds'],[0,0])
        self.assertEqual(r['final_word_region_error_bounds'],[0,0])

    def test_missing_last_word(self):
        r=alignment('i am hungry too'.split(),'i am hungry'.split())
        self.assertEqual(r['final_word_region_error_bounds'],[1,1])
        self.assertEqual(r['operations'][-1]['operation'],'D')

    def test_extra_word(self):
        r=alignment('i am hungry'.split(),'i am hungry too'.split())
        self.assertEqual(r['final_word_region_error_bounds'],[1,1])
        self.assertTrue(r['operations'][-1]['trailing_insertion'])

    def test_prefix_error_not_tail(self):
        r=alignment('a b c d e'.split(),'x b c d e'.split())
        self.assertEqual(r['edits'],1)
        self.assertEqual(r['tail_error_bounds'],[0,0])

    def test_repeated_word_alignment_is_ambiguous(self):
        r=alignment('a a'.split(),['a'])
        self.assertEqual(r['edits'],1)
        self.assertEqual(r['final_word_region_error_bounds'],[0,1])

    def test_empty_hypothesis(self):
        r=alignment('a b c d'.split(),[])
        self.assertEqual(r['edits'],4)
        self.assertEqual(r['tail_error_bounds'],[3,3])
        self.assertEqual(r['final_word_region_error_bounds'],[1,1])


if __name__=='__main__':
    unittest.main()
