import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('frontend_gate',
    Path(__file__).resolve().parents[1]/'scripts/verify_nemotron_frontend.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FrontendGateTests(unittest.TestCase):
    def test_packetization_covers_each_sample_once(self):
        self.assertEqual(gate.packets(0,[3]),[0])
        self.assertEqual(gate.packets(10,[3,4]),[3,4,3])
        for pattern in ([],[0],[-1]):
            with self.assertRaises(ValueError): gate.packets(10,pattern)

    def test_ordinary_rounding_is_bounded(self):
        reference=np.array([-16.,-12.,0.],np.float32)
        self.assertTrue(gate.numerical_match(reference+0.0005,reference,.001)[0])
        self.assertFalse(gate.numerical_match(reference+0.01,reference,.001)[0])

    def test_quiet_exception_requires_energy_and_log_bounds(self):
        reference=np.array([np.log(2**-24+2.5e-8)],np.float32)
        matched,count=gate.numerical_match(reference+.0028,reference,.001)
        self.assertTrue(matched)
        self.assertEqual(count,1)
        self.assertFalse(gate.numerical_match(reference+.006,reference,.001)[0])
        reference=np.array([np.log(5e-7)],np.float32)
        self.assertFalse(gate.numerical_match(reference+.004,reference,.001)[0])

    def test_quiet_exception_does_not_relax_ordinary_features(self):
        reference=np.array([-5.],np.float32)
        self.assertFalse(gate.numerical_match(reference+.0028,reference,.001)[0])


if __name__=='__main__': unittest.main()
