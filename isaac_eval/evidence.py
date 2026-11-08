"""Lossless, append-only observations and same-call image-head capture."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def _pack(value, arrays):
  if isinstance(value, np.ndarray):
    if value.dtype.hasobject:
      raise ValueError("object arrays are not supported")
    key = f"a{len(arrays)}"
    arrays[key] = value.copy()
    return {"__array__": key}
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {str(k): _pack(v, arrays) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [_pack(v, arrays) for v in value]
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _unpack(value, arrays):
  if isinstance(value, dict):
    if set(value) == {"__array__"}:
      return arrays[value["__array__"]].copy()
    return {k: _unpack(v, arrays) for k, v in value.items()}
  if isinstance(value, list):
    return [_unpack(v, arrays) for v in value]
  return value


def input_hash(images):
  h = hashlib.sha256()
  for name in ("wide", "road"):
    image = np.asarray(images[name])
    h.update(name.encode())
    h.update(str(image.dtype).encode())
    h.update(json.dumps(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
  return h.hexdigest()


class EvidenceWriter:
  def __init__(self, directory, metadata):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=False)
    self.count = 0
    self.write_seconds = 0.0
    arrays = {}
    packed = _pack(metadata, arrays)
    if arrays:
      raise ValueError("metadata must be JSON-compatible; convert arrays to lists")
    metadata_bytes = (json.dumps(packed, sort_keys=True, indent=2) + "\n").encode()
    (self.directory / "metadata.json").write_bytes(metadata_bytes)
    self.metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    self._manifest = (self.directory / "manifest.jsonl").open("x")

  def append(self, record):
    start = time.perf_counter()
    images = record["images"]
    for cam in ("wide", "road"):
      im = np.asarray(images[cam])
      if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
        raise ValueError(f"{cam} must be uint8 RGB")
    plan = np.asarray(record["prediction"]["plan"])
    if plan.ndim != 2 or plan.shape[0] < 2 or plan.shape[1] < 2:
      raise ValueError("prediction.plan must be [N, >=2]")
    if "action" not in record:
      raise ValueError("action is required")
    arrays = {}
    payload = _pack(record, arrays)
    arrays["__record_json__"] = np.frombuffer(json.dumps(payload, sort_keys=True).encode(), dtype=np.uint8)
    name = f"frame_{self.count:06d}.npz"
    path = self.directory / name
    with path.open("xb") as f:
      np.savez_compressed(f, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    self._manifest.write(json.dumps({"frame": self.count, "file": name, "sha256": digest,
                                     "input_sha256": input_hash(images),
                                     "metadata_sha256": self.metadata_sha256}) + "\n")
    self._manifest.flush()
    self.count += 1
    self.write_seconds += time.perf_counter() - start

  def close(self):
    self._manifest.close()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()


def _completion(directory, metadata, metadata_hash, manifest_bytes, entries):
  receipt_path = directory / "completion.json"
  if not receipt_path.exists():
    if metadata.get("pilot_only") is False:
      raise ValueError("scored evidence requires completion receipt")
    return None
  receipt = json.loads(receipt_path.read_text())
  if (receipt.get("version") != 1 or receipt.get("frame_count") != len(entries)
      or receipt.get("metadata_sha256") != metadata_hash
      or receipt.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()):
    raise ValueError("evidence completion receipt mismatch")
  if {path.name for path in directory.glob("frame_*.npz")} != {entry["file"] for entry in entries}:
    raise ValueError("evidence frame files mismatch")
  summary_bytes = (directory / receipt["summary_path"]).read_bytes()
  if hashlib.sha256(summary_bytes).hexdigest() != receipt.get("summary_sha256"):
    raise ValueError("evidence summary hash mismatch")
  return json.loads(summary_bytes)


def verified_summary(directory):
  """Return the bound terminal summary, or None for unsealed pilot evidence."""
  directory = Path(directory)
  metadata_bytes = (directory / "metadata.json").read_bytes()
  manifest_bytes = (directory / "manifest.jsonl").read_bytes()
  entries = [json.loads(line) for line in manifest_bytes.splitlines()]
  return _completion(directory, json.loads(metadata_bytes), hashlib.sha256(metadata_bytes).hexdigest(),
                     manifest_bytes, entries)


def finalize_evidence(directory, summary_path):
  """Seal a closed evidence stream and its terminal summary against truncation or edits."""
  directory = Path(directory)
  summary_path = Path(summary_path).resolve()
  for _ in _read_evidence(directory, require_completion=False):
    pass
  metadata_bytes = (directory / "metadata.json").read_bytes()
  manifest_bytes = (directory / "manifest.jsonl").read_bytes()
  entries = [json.loads(line) for line in manifest_bytes.splitlines()]
  if {path.name for path in directory.glob("frame_*.npz")} != {entry["file"] for entry in entries}:
    raise ValueError("evidence frame files mismatch")
  summary_bytes = summary_path.read_bytes()
  json.loads(summary_bytes)
  receipt = {"version": 1, "frame_count": len(entries),
             "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
             "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
             "summary_path": os.path.relpath(summary_path, directory.resolve()),
             "summary_sha256": hashlib.sha256(summary_bytes).hexdigest()}
  with (directory / "completion.json").open("x") as f:
    json.dump(receipt, f, sort_keys=True, indent=2)
    f.write("\n")


def read_evidence(directory):
  yield from _read_evidence(directory, require_completion=True)


def _read_evidence(directory, require_completion):
  directory = Path(directory)
  metadata_bytes = (directory / "metadata.json").read_bytes()
  metadata = json.loads(metadata_bytes)
  metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
  manifest_bytes = (directory / "manifest.jsonl").read_bytes()
  entries = [json.loads(line) for line in manifest_bytes.splitlines()]
  if require_completion:
    _completion(directory, metadata, metadata_hash, manifest_bytes, entries)
  for index, entry in enumerate(entries):
    if entry["metadata_sha256"] != metadata_hash:
      raise ValueError("evidence metadata hash mismatch")
    if entry["frame"] != index or entry["file"] != f"frame_{index:06d}.npz":
      raise ValueError("non-sequential evidence manifest")
    path = directory / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
      raise ValueError(f"evidence hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as arrays:
      payload = json.loads(arrays["__record_json__"].tobytes())
      record = _unpack(payload, arrays)
    if input_hash(record["images"]) != entry["input_sha256"]:
      raise ValueError(f"input hash mismatch: {path}")
    yield metadata, record, entry


class ImageHeadCapture:
  """Copy the image head outputs from the policy call without changing its output."""
  def __init__(self, model):
    self.head = getattr(model, "img", None)
    self.output = None
    self.handle = None

  def clear(self):
    self.output = None

  def __enter__(self):
    if self.head is not None:
      def capture(_module, _args, output):
        def copy(value):
          if isinstance(value, dict):
            return {k: copy(v) for k, v in value.items()}
          return value.detach().cpu().numpy().copy()
        self.output = copy(output)
        return None
      self.handle = self.head.register_forward_hook(capture)
    return self

  def __exit__(self, *args):
    if self.handle is not None:
      self.handle.remove()
