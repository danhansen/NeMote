#!/usr/bin/env python3
"""Lower Conformer 1x1 channel projections to MatMul before quantization.

NeMo uses Conv1d modules for these projections. Their kernel=1/group=1 ABI
is exactly Transpose(NCT->NTC), MatMul(W.T), bias, Transpose(NTC->NCT).
This enables ORT's fused dynamic-quantized GEMM path instead of ConvInteger.
Depthwise, subsampling, and temporal convolutions are deliberately untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


POINTWISE = re.compile(r"/encoder/layers\.\d+/conv/pointwise_conv[12]/Conv$")


def rewrite(model, *, base_dir: Path) -> int:
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    added = []
    nodes = []
    replaced_weights = set()
    count = 0
    for node in model.graph.node:
        if not POINTWISE.fullmatch(node.name):
            nodes.append(node)
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if node.op_type != "Conv" or len(node.input) not in (2, 3) or len(node.output) != 1:
            raise ValueError(f"unsupported pointwise node: {node.name}")
        if any(attrs.get(key, default) != default for key, default in (
            ("group", 1), ("kernel_shape", [1]), ("strides", [1]),
            ("dilations", [1]), ("pads", [0, 0]), ("auto_pad", b"NOTSET"),
        )):
            raise ValueError(f"pointwise convolution must have unit kernel/stride and no padding: {node.name}")
        weight = initializers.get(node.input[1])
        if weight is None or weight.data_type != TensorProto.FLOAT or len(weight.dims) != 3 or weight.dims[2] != 1:
            raise ValueError(f"pointwise projection requires a static FP32 [out,in,1] weight: {node.name}")
        prefix = node.name + "/linear"
        linear_weight = prefix + "/weight"
        values = numpy_helper.to_array(weight, base_dir=str(base_dir))
        added.append(numpy_helper.from_array(np.ascontiguousarray(values[:, :, 0].T), linear_weight))
        transposed = prefix + "/input"
        product = prefix + "/product"
        nodes.extend([
            helper.make_node("Transpose", [node.input[0]], [transposed], perm=[0, 2, 1], name=prefix + "/TransposeIn"),
            helper.make_node("MatMul", [transposed, linear_weight], [product], name=prefix + "/MatMul"),
        ])
        if len(node.input) == 3 and node.input[2]:
            biased = prefix + "/biased"
            nodes.append(helper.make_node("Add", [product, node.input[2]], [biased], name=prefix + "/Bias"))
            product = biased
        nodes.append(helper.make_node("Transpose", [product], node.output, perm=[0, 2, 1], name=prefix + "/TransposeOut"))
        replaced_weights.add(weight.name)
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | {value.name for value in model.graph.output}
    retained = [tensor for tensor in model.graph.initializer if tensor.name not in replaced_weights or tensor.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend([*retained, *added])
    return count


def rewrite_file(source: Path, output: Path) -> int:
    # External tensors not involved in this rewrite remain unloaded and shared
    # within the same build directory. No multi-GB protobuf copy is needed.
    if source.parent.resolve() != output.parent.resolve():
        raise ValueError("source/output must share a directory to retain external-data references")
    if output.exists() or output.with_suffix(".data").exists():
        raise FileExistsError(output)
    model = onnx.load(source, load_external_data=False)
    count = rewrite(model, base_dir=source.parent)
    if count == 0:
        raise ValueError("no supported Conformer pointwise convolutions found")
    onnx.save_model(model, output, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=output.with_suffix(".data").name, size_threshold=65536)
    onnx.checker.check_model(str(output))
    print(f"[pointwise] replaced {count} Conv1d channel projections with MatMul", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rewrite_file(args.input, args.output)


if __name__ == "__main__":
    main()
