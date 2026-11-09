"""Offline checkpoint screening and provenance tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from scripts import select_lab_action as selector


def write(path, data):
    path.write_text(json.dumps(data) + "\n")


def metrics(steers, plan=1.0, longitudinal=1.0):
    episodes = [
        {
            "path": p,
            "count": 1,
            "steer": s,
            "plan_l1": plan,
            "longitudinal": longitudinal,
            "plan_nll": -1.0,
            "loss": 1.0,
        }
        for p, s in zip(("a", "b"), steers, strict=True)
    ]
    return {
        "precision": "float32",
        "aggregation": "equal_episode",
        "count": 2,
        "aggregate": {
            "steer": sum(steers) / 2,
            "plan_l1": plan,
            "longitudinal": longitudinal,
            "plan_nll": -1.0,
            "loss": 1.0,
        },
        "episodes": episodes,
    }


class SelectorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.action = self.root / "action"
        self.base = self.root / "base"
        self.action.mkdir()
        self.base.mkdir()
        self.manifest = self.root / "manifest.json"
        write(self.manifest, {"schema_version": 1, "complete": True, "membership_sha256": "b" * 64})
        self.receipt = self.root / "verification.json"
        write(
            self.receipt,
            {"passed": True, "manifest_sha256": selector.sha(self.manifest), "membership_sha256": "b" * 64},
        )
        self.init = self.root / "init.pt"
        self.init.write_bytes(b"init")
        self.reference = {
            "base.vision.weight": {"sha256": "v"},
            "bev.weight": {"sha256": "b"},
            "img.weight": {"sha256": "i"},
        }
        self.args = {
            "manifest": str(self.manifest),
            "init": str(self.init),
            "freeze_vision": True,
            "w_plan": 0.1,
            "steps": 2,
            "val_every": 1,
            "history_context": 8,
        }
        self.anchors = [{"path": "a", "frames": [1]}, {"path": "b", "frames": [2]}]
        self.verification = {
            "status": "verified",
            "path": str(self.receipt),
            "sha256": selector.sha(self.receipt),
            "manifest_sha256": selector.sha(self.manifest),
            "membership_sha256": "b" * 64,
        }
        for folder in (self.action, self.base):
            source = folder / "source.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("a.py", b"abc")
            info = {
                "args": dict(self.args),
                "manifest_sha256": selector.sha(self.manifest),
                "dataset_verification": self.verification,
                "validation_selection": self.anchors,
                "episodes": [{"path": "a", "meta_sha256": "1" * 64}],
                "source": {
                    "archive_sha256": selector.sha(source),
                    "files_sha256": {"a.py": hashlib.sha256(b"abc").hexdigest()},
                },
                "init_sha256": selector.sha(self.init),
                "baseline_state_sha256": self.reference,
            }
            if folder == self.base:
                info["args"]["freeze_vision"] = False
            write(folder / "run.json", info)
        baseline = metrics([0.001, 0.003])
        baseline["step"] = 0
        for row in [baseline["aggregate"], *baseline["episodes"]]:
            del row["plan_l1"]
        write(self.base / "baseline_validation.json", baseline)
        zero = metrics([0.001, 0.003])
        zero.update(step=0, saved_state_sha256=self.reference)
        write(self.action / "baseline_validation.json", zero)
        (self.base / "history.jsonl").write_text(
            json.dumps({"step": 9, "selected_sha256": selector.sha(self.init)}) + "\n"
        )
        self.rows = []
        for step in (1, 2):
            path = self.action / f"step_{step:06d}.pt"
            path.write_bytes(f"checkpoint {step}".encode())
            self.rows.append(
                {
                    "step": step,
                    "checkpoint": str(path),
                    "checkpoint_sha256": selector.sha(path),
                    "state_sha256": self.reference,
                    "validation": metrics([0.001, 0.003]),
                }
            )
        self.save_history()
        self.patch = patch.object(selector, "frozen_state", return_value=self.reference)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def save_history(self):
        (self.action / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def run_selection(self):
        return selector.select(self.action, self.base, 2)

    def test_boundaries_and_tie_choose_earlier(self):
        for row in self.rows:
            row["validation"] = metrics([0.002, 0.003], 1.1, 1.25)
        self.save_history()
        result = self.run_selection()
        self.assertEqual(result["selected"]["step"], 1)
        self.assertTrue(all(row["eligible"] for row in result["candidates"]))

    def test_per_episode_regression_rejected_and_receipt_written(self):
        for row in self.rows:
            row["validation"] = metrics([0.0021, 0.001])
        self.save_history()
        output = self.root / "selection.json"
        with self.assertRaises(SystemExit):
            selector.main(
                [
                    "--training",
                    str(self.action),
                    "--baseline-training",
                    str(self.base),
                    "--out",
                    str(output),
                    "--expected-steps",
                    "2",
                ]
            )
        result = selector.read(output)
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["selected"])
        self.assertIn("episode steer", result["candidates"][0]["rejection_reasons"][0])
        with self.assertRaises(FileExistsError):
            selector.main(["--training", str(self.action), "--baseline-training", str(self.base), "--out", str(output)])

    def test_each_aggregate_limit_rejects_above_boundary(self):
        cases = [
            ([0.002, 0.00301], 1.0, 1.0, "steer exceeds"),
            ([0.001, 0.003], 1.1001, 1.0, "plan_l1 exceeds"),
            ([0.001, 0.003], 1.0, 1.2501, "longitudinal exceeds"),
        ]
        for steers, plan, longitudinal, reason in cases:
            with self.subTest(reason=reason):
                for row in self.rows:
                    row["validation"] = metrics(steers, plan, longitudinal)
                self.save_history()
                result = self.run_selection()
                self.assertIsNone(result["selected"])
                self.assertTrue(any(reason in r for r in result["candidates"][0]["rejection_reasons"]))

    def test_bad_provenance_and_anchor_mismatch(self):
        run_path = self.action / "run.json"
        run = selector.read(run_path)
        run["validation_selection"][0]["frames"] = [3]
        write(run_path, run)
        with self.assertRaisesRegex(ValueError, "anchors differ"):
            self.run_selection()
        run["validation_selection"] = self.anchors
        run["dataset_verification"]["sha256"] = "0" * 64
        write(run_path, run)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_selection()

    def test_checkpoint_hash_and_history_mismatch(self):
        (self.action / "step_000001.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            self.run_selection()
        self.rows = self.rows[1:]
        self.save_history()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.run_selection()

    def test_frozen_state_mismatch_and_config_mismatch(self):
        self.patch.stop()
        with patch.object(selector, "frozen_state", return_value={}):
            with self.assertRaisesRegex(ValueError, "frozen state"):
                self.run_selection()
        run_path = self.action / "run.json"
        run = selector.read(run_path)
        run["args"]["w_plan"] = 1.0
        write(run_path, run)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_selection()


if __name__ == "__main__":
    unittest.main()
