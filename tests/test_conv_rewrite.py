import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

spec = importlib.util.spec_from_file_location(
    'conv_rewrite', Path(__file__).resolve().parents[1] / 'scripts/dequantize_nemotron_conv_blocks.py')
rewrite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rewrite
spec.loader.exec_module(rewrite)


class ConvRewriteTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(2)
        tensors = [numpy_helper.from_array(rng.integers(0, 256, (4, 1, 9), dtype=np.uint8), 'w'),
                   numpy_helper.from_array(np.array(127, dtype=np.uint8), 'wzp'),
                   numpy_helper.from_array(np.array(.031, dtype=np.float32), 'ws')]
        nodes = [helper.make_node('DynamicQuantizeLinear', ['x'], ['xq', 'xs', 'xzp']),
                 helper.make_node('ConvInteger', ['xq', 'w', 'xzp', 'wzp'], ['yi'],
                                  name='layer/depthwise/Conv', group=4),
                 helper.make_node('Cast', ['yi'], ['yf'], to=TensorProto.FLOAT),
                 helper.make_node('Mul', ['xs', 'ws'], ['scale']),
                 helper.make_node('Mul', ['yf', 'scale'], ['y'])]
        graph = helper.make_graph(nodes, 'depthwise',
                                  [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 'time'])],
                                  [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 'frames'])], tensors)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        return model

    def test_preserved_activation_quantization_matches_integer_reference(self):
        model = self.model()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        original = ort.InferenceSession(model.SerializeToString(), options)
        specs = rewrite.build_rewrite_specs(model, ['depthwise'], [], preserve_activation_quantization=True)
        self.assertEqual(rewrite.apply_rewrites(model, specs), 1)
        rewrite.prune_unused_initializers(model)
        onnx.checker.check_model(model)
        self.assertEqual([n.op_type for n in model.graph.node], ['DynamicQuantizeLinear', 'DequantizeLinear', 'Conv'])
        candidate = ort.InferenceSession(model.SerializeToString(), options)
        for frames in (1, 2, 7, 14):
            for scale in (0, .01, 1, 100):
                x = np.random.default_rng(frames).normal(size=(1, 4, frames + 8)).astype(np.float32) * scale
                np.testing.assert_allclose(candidate.run(None, {'x': x})[0],
                                           original.run(None, {'x': x})[0], atol=1e-3, rtol=1e-5)

    def test_default_still_removes_activation_quantization(self):
        model = self.model()
        specs = rewrite.build_rewrite_specs(model, ['depthwise'], [])
        self.assertEqual(rewrite.apply_rewrites(model, specs), 1)
        rewrite.prune_unused_initializers(model)
        onnx.checker.check_model(model)
        self.assertEqual([n.op_type for n in model.graph.node], ['Conv'])

    def test_integer_float_kernel_is_bit_exact(self):
        model = self.model()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        original = ort.InferenceSession(model.SerializeToString(), options)
        specs = rewrite.build_rewrite_specs(model, ['depthwise'], [], integer_float_kernel=True)
        self.assertEqual(rewrite.apply_rewrites(model, specs), 1)
        rewrite.prune_unused_initializers(model)
        onnx.checker.check_model(model)
        candidate = ort.InferenceSession(model.SerializeToString(), options)
        for frames in (1, 2, 7, 14):
            for scale in (0, .01, 1, 100):
                x = np.random.default_rng(frames).normal(size=(1, 4, frames + 8)).astype(np.float32) * scale
                np.testing.assert_array_equal(candidate.run(None, {'x': x})[0],
                                              original.run(None, {'x': x})[0])

    def test_filters_preserve_unselected_blocks(self):
        model = self.model()
        before = model.SerializeToString()
        self.assertEqual(rewrite.build_rewrite_specs(model, ['pointwise'], []), [])
        self.assertEqual(model.SerializeToString(), before)
        self.assertEqual(rewrite.build_rewrite_specs(model, [], ['depthwise']), [])


if __name__ == '__main__':
    unittest.main()
