import json
from dataclasses import asdict

import numpy as np
import pytest

from fsop.geometry.camera import CameraMount
from fsop.sim.expert import ExpertParams
from fsop.sim.vehicle import VehicleParams
from fsop.track.schema import Track
from fsop.viz.video import compose_frame
from isaac_eval.evidence import EvidenceWriter, ImageHeadCapture, finalize_evidence, read_evidence, verified_summary
from isaac_eval.render_debug import OfflineOracle, render


def sample():
  track = Track(id="test", discipline="acceleration", seed=0, closed=False,
                centerline=np.array([[0., 0.], [10., 0.], [30., 0.], [80., 0.]]),
                width_left=np.full(4, 3.), width_right=np.full(4, 3.), finish_line=((70., -3.), (70., 3.)))
  metadata = {"track": track.to_dict(), "cone_colors": ["blue", "yellow"],
              "initial_cones_xyz": [[5., 3., 0.], [5., -3., 0.]],
              "expert_params": asdict(ExpertParams()), "vehicle_params": asdict(VehicleParams()),
              "label_dt": 0.025, "fps": 20}
  images = {"wide": np.full((256, 512, 3), 31, np.uint8), "road": np.full((256, 512, 3), 73, np.uint8)}
  state = {"x": 0., "y": 0., "yaw": 0., "vx": 2., "vy": 0., "r": 0., "steer": 0., "accel": 0., "t": 0.}
  plan = np.zeros((33, 15), np.float32)
  plan[:, 0] = np.linspace(0, 30, 33)
  record = {"images": images, "prediction": {"plan": plan, "plan_std": np.ones_like(plan),
                                              "cones_xy": np.array([[5., 2.]]), "cones_cls": np.array([0])},
            "state": state, "progress": {"s": 0., "hint": 0}, "action": [0., 1.],
            "T_plan_ego_camera": CameraMount().T_ego_cam(),
            "actual_cones_xyz": np.array(metadata["initial_cones_xyz"]), "image_head": None}
  return track, metadata, record


def test_evidence_roundtrip_hash_and_required_plan(tmp_path):
  _, metadata, record = sample()
  with EvidenceWriter(tmp_path / "episode", metadata) as writer:
    writer.append(record)
    with pytest.raises(KeyError):
      writer.append({**record, "prediction": {}})
  loaded_metadata, loaded, entry = next(read_evidence(tmp_path / "episode"))
  assert loaded_metadata == json.loads(json.dumps(metadata))
  for name in ("wide", "road"):
    np.testing.assert_array_equal(loaded["images"][name], record["images"][name])
  np.testing.assert_array_equal(loaded["prediction"]["plan_std"], record["prediction"]["plan_std"])
  assert loaded["action"] == record["action"]
  assert len(entry["input_sha256"]) == 64
  metadata_path = tmp_path / "episode" / "metadata.json"
  original_metadata = metadata_path.read_bytes()
  metadata_path.write_bytes(original_metadata + b" ")
  with pytest.raises(ValueError, match="metadata hash mismatch"):
    list(read_evidence(tmp_path / "episode"))
  metadata_path.write_bytes(original_metadata)
  path = tmp_path / "episode" / entry["file"]
  path.write_bytes(path.read_bytes() + b"tamper")
  with pytest.raises(ValueError, match="hash mismatch"):
    list(read_evidence(tmp_path / "episode"))


def test_offline_renderer_uses_same_compositor_without_mutation(tmp_path, monkeypatch):
  import isaac_eval.render_debug as debug
  track, metadata, record = sample()
  before = {name: image.copy() for name, image in record["images"].items()}
  plan_before = record["prediction"]["plan"].copy()
  with EvidenceWriter(tmp_path / "episode", metadata) as writer:
    writer.append(record)
  rendered = []
  class Writer:
    def write(self, frame):
      rendered.append(frame.copy())
    def release(self):
      pass
  monkeypatch.setattr(debug, "_writer", lambda *args: Writer())
  import builtins
  original_import = builtins.__import__
  def no_model(name, *args, **kwargs):
    if name.startswith(("fsop.model", "fsop.eval.closed_loop")):
      raise AssertionError(f"offline renderer imported {name}")
    return original_import(name, *args, **kwargs)
  monkeypatch.setattr(builtins, "__import__", no_model)
  (tmp_path / "summary.json").write_text(json.dumps({"result": "finish", "pass": True, "final_state": record["state"]}))
  finalize_evidence(tmp_path / "episode", tmp_path / "summary.json")
  assert render(tmp_path / "episode", tmp_path / "out.mp4") == {"inference_frames": 1, "terminal_frames": 40}
  assert len(rendered) == 41
  assert np.all(rendered[-1][0, 0] == 26)
  expected = compose_frame(track, record["actual_cones_xyz"][:, :2], metadata["cone_colors"], record["images"],
                           record["state"], record["T_plan_ego_camera"], record["prediction"],
                           OfflineOracle(metadata).reference(record), [(0., 0.)],
                           "t 0.0s v 7 km/h laps ?/? DOO ? OC ?")
  np.testing.assert_array_equal(rendered[0], expected)
  for name in before:
    np.testing.assert_array_equal(record["images"][name], before[name])
  np.testing.assert_array_equal(record["prediction"]["plan"], plan_before)


def test_capture_is_read_only_and_can_be_cleared():
  class Head:
    def register_forward_hook(self, hook):
      self.hook = hook
      return self
    def remove(self):
      self.hook = None
  class Tensor:
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def __init__(self): self.value = np.array([2.])
  head = Head()
  model = type("Model", (), {"img": head})()
  tensor = Tensor()
  with ImageHeadCapture(model) as capture:
    output = {"wide": {"heat": tensor}}
    assert head.hook(head, (), output) is None
    tensor.value[0] = 3
    assert capture.output["wide"]["heat"][0] == 2
    capture.clear()
    assert capture.output is None
  assert head.hook is None


def test_offline_rrd_export(tmp_path):
  pytest.importorskip("rerun")
  from isaac_eval.render_debug import export_rrd

  _, metadata, record = sample()
  record["events"] = [{"kind": "doo", "data": {"cone": 1}, "t": 0.0}]
  with EvidenceWriter(tmp_path / "episode", metadata) as writer:
    writer.append(record)
  output = tmp_path / "debug.rrd"
  assert export_rrd(tmp_path / "episode", output) == 1
  assert output.stat().st_size > 1000


def test_scored_completion_binds_manifest_frames_and_summary(tmp_path):
  _, metadata, record = sample()
  metadata["pilot_only"] = False
  episode = tmp_path / "episode"
  with EvidenceWriter(episode, metadata) as writer:
    writer.append(record)
    writer.append(record)
  with pytest.raises(ValueError, match="requires completion"):
    list(read_evidence(episode))
  summary = tmp_path / "summary.json"
  summary.write_text(json.dumps({"final_state": record["state"], "events": [], "pending_events": []}))
  finalize_evidence(episode, summary)
  assert len(list(read_evidence(episode))) == 2
  assert verified_summary(episode)["final_state"] == record["state"]
  manifest = episode / "manifest.jsonl"
  original = manifest.read_bytes()
  manifest.write_bytes(original.splitlines(keepends=True)[0])
  with pytest.raises(ValueError, match="completion receipt mismatch"):
    list(read_evidence(episode))
  manifest.write_bytes(original)
  frame = episode / "frame_000001.npz"
  saved = frame.read_bytes()
  frame.unlink()
  with pytest.raises(ValueError, match="frame files mismatch"):
    list(read_evidence(episode))
  frame.write_bytes(saved)
  extra = episode / "frame_000002.npz"
  extra.write_bytes(saved)
  with pytest.raises(ValueError, match="frame files mismatch"):
    list(read_evidence(episode))
  extra.unlink()
  summary.write_text("{}")
  with pytest.raises(ValueError, match="summary hash mismatch"):
    list(read_evidence(episode))


def test_unsealed_pilot_has_no_verified_summary(tmp_path):
  _, metadata, record = sample()
  metadata["pilot_only"] = True
  with EvidenceWriter(tmp_path / "episode", metadata) as writer:
    writer.append(record)
  (tmp_path / "summary.json").write_text('{}')
  assert len(list(read_evidence(tmp_path / "episode"))) == 1
  assert verified_summary(tmp_path / "episode") is None
