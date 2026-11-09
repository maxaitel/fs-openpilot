"""CPU-only contracts for frozen Lab orchestration; no Isaac or GPU needed."""
import fcntl
import hashlib
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from isaac_eval.evidence import EvidenceWriter, finalize_evidence
from scripts import run_lab_evaluations as coordinator


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def make_plan(self):
        repo = self.root / "repo"
        (repo / "src/fsop").mkdir(parents=True)
        (repo / "src/fsop/sample.py").write_text("original\n")
        for graph in coordinator.GRAPH_FILES:
            path = repo / graph
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(graph.encode())
        case_path = self.root / "case.json"
        case_path.write_text(json.dumps({"id": "case1", "split": "val", "discipline": "skidpad", "seed": 0}))
        checkpoint = self.root / "weights.pt"
        checkpoint.write_bytes(b"frozen")
        plan = {"version": 1, "timeout_seconds": 30, "cases": ["case.json"],
                "models": [{"name": "baseline", "checkpoint": "weights.pt", "sha256": coordinator.digest(checkpoint)}]}
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return repo, plan_path, case_path, checkpoint

    def test_plan_freezes_inputs_and_detects_source_or_checkpoint_changes(self):
        repo, plan_path, _, checkpoint = self.make_plan()
        lock = coordinator.load_plan(plan_path, repo)
        out, _ = coordinator.prepare(self.root / "runs", lock)
        coordinator.check_inputs(lock, repo)
        (repo / "src/fsop/sample.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "source"):
            coordinator.check_inputs(lock, repo)
        with self.assertRaisesRegex(ValueError, "Frozen plan"):
            coordinator.prepare(out, coordinator.load_plan(plan_path, repo))
        checkpoint.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            coordinator.load_plan(plan_path, repo)

    def make_episode(self, pilot=False):
        d = self.root / "episode"
        d.mkdir()
        case = {"name": "case1", "case": coordinator.normalize_case(
            {"id": "case1", "split": "val", "discipline": "skidpad", "seed": 0})}
        model = {"name": "baseline", "sha256": "a" * 64}
        lock = {"source_sha256": {"sample.py": hashlib.sha256(b"source").hexdigest()},
                "graph_sha256": {"graph": "c" * 64}}
        def write(name, obj):
            (d / name).write_text(json.dumps(obj))
        write("case.json", case["case"])
        write("track.json", {"name": "track"})
        for name in ("realization.json", "appearance.json", "scene.usda"):
            (d / name).write_bytes(name.encode())
        with ZipFile(d / "source.zip", "w") as archive:
            archive.writestr("sample.py", b"source")
        write("export.json", {"dependencies": [], "sha256": {}})
        write("frames.json", [{"frame": 0}])
        write("telemetry.json", [])
        metadata = {"case": case["case"], "backend": "Isaac Lab PhysX", "fps": 20,
                    "pilot_only": pilot, "checkpoint_sha256": model["sha256"], "state_sha256": "d" * 64,
                    "source_sha256": lock["source_sha256"], "graph_sha256": lock["graph_sha256"],
                    "track": {"name": "track"}}
        for key, filename in (("source_archive_sha256", "source.zip"), ("realization_sha256", "realization.json"),
                              ("appearance_sha256", "appearance.json"), ("scene_sha256", "scene.usda"),
                              ("export_sha256", "export.json")):
            metadata[key] = coordinator.digest(d / filename)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        record = {"frame": 0, "images": {"wide": image, "road": image},
                  "prediction": {"plan": np.zeros((2, 2))}, "action": [0, 0]}
        with EvidenceWriter(d / "evidence", metadata) as writer:
            writer.append(record)
        summary = {"case": case["case"], "pilot_only": pilot, "checkpoint_sha256": model["sha256"],
                   "checkpoint_sha256_after": model["sha256"], "state_sha256": "d" * 64,
                   "state_sha256_after": "d" * 64, "frozen_unchanged": True, "recorded_frames": 1,
                   "result": "timeout", "final_state": {"t": .05}}
        write("summary.json", summary)
        finalize_evidence(d / "evidence", d / "summary.json")
        return d, case, model, lock

    def test_sealed_scored_failure_is_valid_but_frame_tampering_fails(self):
        d, case, model, lock = self.make_episode()
        receipt = coordinator.validate_episode(d, case, model, lock)
        self.assertEqual(receipt["result"], "timeout")
        (d / "evidence/frame_000000.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            coordinator.validate_episode(d, case, model, lock)

    def test_pilot_cannot_be_resumed_as_scored(self):
        d, case, model, lock = self.make_episode(pilot=True)
        with self.assertRaisesRegex(ValueError, "not a scored"):
            coordinator.validate_episode(d, case, model, lock)

    def test_interrupted_run_is_not_skipped_or_relaunched(self):
        repo, plan_path, _, _ = self.make_plan()
        lock = coordinator.load_plan(plan_path, repo)
        out, manifest = coordinator.prepare(self.root / "runs", lock)
        manifest["episodes"]["baseline/case1"] = {"status": "running"}
        coordinator.atomic_json(out / "manifest.json", manifest)
        with patch.object(coordinator, "load_plan", return_value=lock), patch.object(coordinator, "run_process") as run:
            with self.assertRaisesRegex(ValueError, "incomplete or failed"):
                coordinator.run(plan_path, out)
            run.assert_not_called()

    def test_batch_limit_and_resume_preserve_order(self):
        repo, plan_path, _, _ = self.make_plan()
        lock = coordinator.load_plan(plan_path, repo)
        lock["models"].append({**lock["models"][0], "name": "candidate"})
        out = self.root / "batches"
        commands = []
        def fake_run(command, logfile, timeout, disk_path):
            commands.append(command)
            Path(command[command.index("--out") + 1]).mkdir()
            return 0
        receipt = {"frame_count": 1, "result": "timeout", "completion_sha256": "x",
                   "summary_sha256": "y", "frames_sha256": "z"}
        with (patch.object(coordinator, "load_plan", return_value=lock),
              patch.object(coordinator, "check_inputs"),
              patch.object(coordinator, "run_process", side_effect=fake_run),
              patch.object(coordinator, "validate_episode", return_value=receipt),
              patch.object(coordinator.shutil, "disk_usage", return_value=types.SimpleNamespace(free=100 * 2**30))):
            first = coordinator.run(plan_path, out, 1)
            self.assertFalse(first["complete"])
            self.assertEqual(list(first["episodes"]), ["baseline/case1"])
            second = coordinator.run(plan_path, out, 1)
            self.assertTrue(second["complete"])
            self.assertEqual(list(second["episodes"]), ["baseline/case1", "candidate/case1"])
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(command[-1] == "--async-evidence" for command in commands))

    def test_second_coordinator_cannot_take_active_lock(self):
        out = self.root / "locked"
        out.mkdir()
        with (out / ".coordinator.lock").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                coordinator.run(self.root / "unused.json", out)

    def test_reserve_breach_terminates_process(self):
        marker = self.root / "reserve-survivor"
        child = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('alive')"
        with patch.object(coordinator.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "reserve"):
                coordinator.run_process([sys.executable, "-c", child], self.root / "reserve.log", 5, self.root)
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_timeout_terminates_child_process_group(self):
        marker = self.root / "descendant-survived"
        child = f"import time,pathlib; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('alive')"
        parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(5)"
        with self.assertRaises(TimeoutError):
            coordinator.run_process([sys.executable, "-c", parent], self.root / "process.log", .3,
                                    self.root, reserve_bytes=0)
        time.sleep(1.7)
        self.assertFalse(marker.exists())

    def test_v2_controller_contract_is_frozen_and_required_in_episode(self):
        from fsop.eval.controller import controller_contract
        repo, plan_path, _, checkpoint = self.make_plan()
        import torch
        torch.save({"model": {}}, checkpoint)
        plan = json.loads(plan_path.read_text())
        plan["models"][0]["sha256"] = coordinator.digest(checkpoint)
        plan["version"] = 2
        plan["models"][0]["controller_contract"] = controller_contract("op_tangent_v1")
        plan_path.write_text(json.dumps(plan))
        lock = coordinator.load_plan(plan_path, repo)
        self.assertEqual(lock["models"][0]["controller_contract"], controller_contract("op_tangent_v1"))
        plan["models"][0]["controller_contract"]["speed_gain"] = 9
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "controller contract"):
            coordinator.load_plan(plan_path, repo)
        d, case, model, episode_lock = self.make_episode()
        episode_lock["version"] = 2
        model["controller_contract"] = controller_contract("op_tangent_v1")
        with self.assertRaisesRegex(ValueError, "schema 2|controller contract"):
            coordinator.validate_episode(d, case, model, episode_lock)
        # Re-seal matching metadata and summary, then verify substitution fails.
        metadata_path, summary_path = d / "evidence/metadata.json", d / "summary.json"
        metadata, summary = json.loads(metadata_path.read_text()), json.loads(summary_path.read_text())
        metadata["schema_version"] = 2
        metadata["controller_contract"] = summary["controller_contract"] = model["controller_contract"]
        metadata_path.write_text(json.dumps(metadata))
        summary_path.write_text(json.dumps(summary))
        completion_path = d / "evidence/completion.json"
        completion = json.loads(completion_path.read_text())
        manifest_path = d / "evidence/manifest.jsonl"
        entries = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        for entry in entries:
            entry["metadata_sha256"] = coordinator.digest(metadata_path)
        manifest_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        completion["manifest_sha256"] = coordinator.digest(manifest_path)
        completion["metadata_sha256"] = coordinator.digest(metadata_path)
        completion["summary_sha256"] = coordinator.digest(summary_path)
        completion_path.write_text(json.dumps(completion))
        self.assertEqual(coordinator.validate_episode(d, case, model, episode_lock)["result"], "timeout")
        model["controller_contract"] = controller_contract("pp")
        with self.assertRaisesRegex(ValueError, "controller contract"):
            coordinator.validate_episode(d, case, model, episode_lock)

    def test_invalid_names_are_rejected(self):
        for name in ("../escape", "/absolute", "", "x/y"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                coordinator._name(name, "case")


if __name__ == "__main__":
    unittest.main()
