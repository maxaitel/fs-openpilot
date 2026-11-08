"""Resolve constant ONNX shape metadata once, instead of reading CUDA scalars each forward.

Only integer get_attr inputs are specialized. Dynamic inputs and negative-step slices retain
onnx2torch's implementation. Existing initializer buffers remain, preserving checkpoint keys.
"""

from __future__ import annotations

import torch
from onnx2torch.node_converters.expand import OnnxExpand
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.slice import OnnxSlice
from onnx2torch.node_converters.squeeze import OnnxSqueezeDynamicAxes
from torch import fx, nn


class StaticReshape(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)

  def forward(self, x, *_):
    return x.reshape(tuple(x.shape[i] if d == 0 else d for i, d in enumerate(self.shape)))


class StaticSlice(nn.Module):
  def __init__(self, starts, ends, axes=None, steps=None):
    super().__init__()
    axes = range(len(starts)) if axes is None else axes
    steps = [1] * len(starts) if steps is None else steps
    self.slices = tuple(zip(axes, starts, ends, steps, strict=True))

  def forward(self, x, *_):
    slices = [slice(None)] * x.ndim
    for axis, start, end, step in self.slices:
      slices[axis] = slice(start, end, step)
    return x[tuple(slices)]


class StaticSqueeze(nn.Module):
  def __init__(self, axes):
    super().__init__()
    self.axes = tuple(axes)

  def forward(self, x, *_):
    return x.squeeze(self.axes) if self.axes else x.squeeze()


class StaticExpand(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)

  def forward(self, x, *_):
    # ONNX Expand broadcasts input and target; target [1,512] must retain a dynamic batch.
    return x.expand(torch.broadcast_shapes(x.shape, self.shape))


def specialize_static_shapes(gm: fx.GraphModule) -> None:
  classes = (OnnxReshape, OnnxSlice, OnnxSqueezeDynamicAxes, OnnxExpand)
  for node in gm.graph.nodes:
    if node.op != "call_module":
      continue
    module = gm.get_submodule(node.target)
    if not isinstance(module, classes):
      continue
    constants = []
    for arg in node.args[1:]:
      if not isinstance(arg, fx.Node) or arg.op != "get_attr":
        break
      value = gm
      for part in arg.target.split("."):
        value = getattr(value, part)
      if not isinstance(value, torch.Tensor) or value.dtype != torch.int64:
        break
      constants.append(value.tolist())
    else:
      replacement = None
      if isinstance(module, OnnxReshape) and len(constants) == 1:
        replacement = StaticReshape(*constants)
      elif isinstance(module, OnnxSlice) and 2 <= len(constants) <= 4:
        if len(constants) < 4 or all(s > 0 for s in constants[3]):
          replacement = StaticSlice(*constants)
      elif isinstance(module, OnnxSqueezeDynamicAxes) and len(constants) == 1:
        replacement = StaticSqueeze(*constants)
      elif isinstance(module, OnnxExpand) and len(constants) == 1:
        replacement = StaticExpand(*constants)
      if replacement is not None:
        gm.set_submodule(node.target, replacement)
