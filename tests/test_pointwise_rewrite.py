import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

spec = importlib.util.spec_from_file_location('pointwise', Path(__file__).resolve().parents[1] / 'scripts/rewrite_nemotron_pointwise_convs.py')
rewrite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewrite)


class PointwiseTests(unittest.TestCase):
    def model(self, bias=False, **attrs):
        weights = np.random.default_rng(0).normal(size=(4, 3, 1)).astype(np.float32)
        tensors = [numpy_helper.from_array(weights, 'w')]
        inputs = ['x', 'w']
        if bias:
            inputs.append('b')
            tensors.append(numpy_helper.from_array(np.arange(4, dtype=np.float32), 'b'))
        node = helper.make_node('Conv', inputs, ['y'], name='/encoder/layers.0/conv/pointwise_conv1/Conv', **attrs)
        graph = helper.make_graph([node], 'test',
                                  [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 3, 'frames'])],
                                  [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 4, 'frames'])], tensors)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        return model

    def test_equivalent_for_dynamic_shapes_with_and_without_bias(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        for bias in (False, True):
            model = self.model(bias=bias)
            original = ort.InferenceSession(model.SerializeToString(), options)
            self.assertEqual(rewrite.rewrite(model, base_dir=Path('.')), 1)
            onnx.checker.check_model(model)
            linear = ort.InferenceSession(model.SerializeToString(), options)
            for batch, frames in ((1, 1), (2, 7), (1, 14)):
                x = np.random.default_rng(frames).normal(size=(batch, 3, frames)).astype(np.float32)
                np.testing.assert_allclose(original.run(None, {'x': x})[0], linear.run(None, {'x': x})[0], atol=1e-6, rtol=1e-6)

    def test_rejects_non_equivalent_convolutions(self):
        for attrs in ({'group': 3}, {'strides': [2]}, {'pads': [1, 1]}, {'dilations': [2]}):
            with self.assertRaises(ValueError):
                rewrite.rewrite(self.model(**attrs), base_dir=Path('.'))

    def test_temporal_convolution_and_shared_weights_are_preserved(self):
        model = self.model()
        model.graph.node.append(helper.make_node('Conv', ['x', 'w'], ['other'], name='/encoder/layers.0/conv/depthwise_conv/Conv'))
        self.assertEqual(rewrite.rewrite(model, base_dir=Path('.')), 1)
        self.assertIn('w', [t.name for t in model.graph.initializer])
        self.assertEqual(model.graph.node[-1].op_type, 'Conv')

    def test_file_rewrite_refuses_overwrite_and_cross_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source.onnx', root / 'linear.onnx'
            onnx.save(self.model(), source)
            self.assertEqual(rewrite.rewrite_file(source, output), 1)
            with self.assertRaises(FileExistsError):
                rewrite.rewrite_file(source, output)
            with self.assertRaises(ValueError):
                rewrite.rewrite_file(source, root / 'sub/linear.onnx')
