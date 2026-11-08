import io

import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx2torch import convert

from fsop.model.static_shapes import specialize_static_shapes


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("op,shape,constants", [
  ("Reshape", (2, 3), [[0, -1]]),
  ("Slice", (2, 6), [[1], [5], [-1], [2]]),
  ("Squeeze", (1, 3), [[1]]),
  ("Expand", (1, 3), [[1, 2, 3]]),
])
def test_static_shape_values_gradients_and_checkpoint_keys(device, batch, op, shape, constants):
  names = [f"c{i}" for i in range(len(constants))]
  graph = helper.make_graph(
    [helper.make_node(op, ["x", *names], ["y"], name="operation")], "test",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, *shape])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    [numpy_helper.from_array(np.array(c, np.int64), n) for n, c in zip(names, constants, strict=True)])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
  reference = convert(model).to(device)
  optimized = convert(model)
  specialize_static_shapes(optimized)
  optimized.to(device)
  assert reference.state_dict().keys() == optimized.state_dict().keys()
  optimized.load_state_dict(reference.state_dict(), strict=True)
  x = torch.randn(batch, *shape, device=device, requires_grad=True)
  y = x.detach().clone().requires_grad_()
  a, b = reference(x), optimized(y)
  torch.testing.assert_close(a, b, rtol=0, atol=0)
  a.square().sum().backward()
  b.square().sum().backward()
  torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
  if device == "cpu" and batch == 4:
    exported = io.BytesIO()
    torch.onnx.export(optimized, (y.detach(),), exported, input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False, dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
    session = ort.InferenceSession(exported.getvalue(), providers=["CPUExecutionProvider"])
    # Check a different batch size after export.
    sample = torch.randn(2, *shape)
    np.testing.assert_allclose(session.run(None, {"x": sample.numpy()})[0], optimized(sample).numpy())


def test_dynamic_shape_is_not_specialized():
  graph = helper.make_graph(
    [helper.make_node("Reshape", ["x", "shape"], ["y"], name="operation")], "dynamic",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
     helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
  gm = convert(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
  before = type(gm.operation)
  specialize_static_shapes(gm)
  assert type(gm.operation) is before
  assert gm(torch.zeros(2, 3), torch.tensor([3, 2])).shape == (3, 2)
