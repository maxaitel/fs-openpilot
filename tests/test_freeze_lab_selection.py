"""CPU-only tests for checkpoint selection and frozen coverage gates."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from scripts import freeze_lab_selection as selection
from scripts.lab_comparison import FIELDS
from scripts.run_lab_evaluations import GRAPH_FILES, digest, load_plan, normalize_case

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "docs/isaac-lab-training/evaluation_coverage.json"
MEMBERSHIP = COVERAGE.parent / "membership.json"


class FakeTensor:
    def __init__(self, values):
        self.value = np.asarray(values, dtype=np.float32)
        self.shape, self.dtype = self.value.shape, self.value.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.value


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_checkpoint_reads_cpu_full_state_and_retained_heads(self):
        path = self.root / "checkpoint.pt"
        path.write_bytes(b"checkpoint")
        state = {
            f"{prefix}.weight": FakeTensor([i, i + 1])
            for i, prefix in enumerate(("base.vision", "base.policy", "bev", "img"))
        }
        fake = types.ModuleType("torch")
        fake.is_tensor = lambda value: isinstance(value, FakeTensor)
        calls = []

        def load(value, **kwargs):
            calls.append(kwargs)
            return {"model": state, "step": 9}

        fake.load = load
        with patch.dict(sys.modules, {"torch": fake}):
            result = selection.checkpoint_info(path)
        self.assertEqual(calls, [{"map_location": "cpu", "weights_only": False}])
        self.assertEqual(result["step"], 9)
        self.assertEqual(result["tensor_count"], 4)
        self.assertEqual(result["state_sha256"], selection.state_digest(state))
        self.assertEqual(set(result["groups"]), {"vision", "policy", "bev_head", "image_head"})
        del state["img.weight"]
        with patch.dict(sys.modules, {"torch": fake}), self.assertRaisesRegex(ValueError, "image_head"):
            selection.checkpoint_info(path)

    def make_training(self):
        dataset = self.root / "manifest.json"
        dataset.write_text(json.dumps({"schema_version": 1, "complete": True, "membership_sha256": "d" * 64}))
        verification = self.root / "verification.json"
        verification.write_text(
            json.dumps({"passed": True, "manifest_sha256": digest(dataset), "membership_sha256": "d" * 64})
        )
        training = self.root / "training"
        training.mkdir()
        with ZipFile(training / "source.zip", "w") as archive:
            archive.writestr("fsop/source.py", b"source")
        run = {
            "init_sha256": "a" * 64,
            "manifest_sha256": digest(dataset),
            "dataset_verification": {
                "status": "verified",
                "sha256": digest(verification),
                "manifest_sha256": digest(dataset),
                "membership_sha256": "d" * 64,
            },
            "source": {
                "archive_sha256": digest(training / "source.zip"),
                "files_sha256": {"fsop/source.py": hashlib.sha256(b"source").hexdigest()},
            },
        }
        (training / "run.json").write_text(json.dumps(run))
        (training / "baseline_validation.json").write_text(json.dumps({"step": 0, "selected_sha256": "b" * 64}))
        (training / "history.jsonl").write_text(json.dumps({"step": 7, "checkpoint_sha256": "c" * 64}) + "\n")
        return training, dataset, verification

    def test_training_history_and_step_binding(self):
        training, dataset, verification = self.make_training()
        proof = selection.verify_training(training, dataset, verification, "a" * 64, "c" * 64, 7)
        self.assertEqual(proof["candidate_records"][0]["step"], 7)
        with self.assertRaisesRegex(ValueError, "step differs"):
            selection.verify_training(training, dataset, verification, "a" * 64, "c" * 64, 8)
        with self.assertRaisesRegex(ValueError, "absent"):
            selection.verify_training(training, dataset, verification, "a" * 64, "e" * 64, 7)
        zero = selection.verify_training(training, dataset, verification, "a" * 64, "b" * 64, 0)
        self.assertEqual(zero["candidate_records"][0]["step"], 0)
        data = json.loads(verification.read_text())
        data["passed"] = False
        verification.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "verification"):
            selection.verify_training(training, dataset, verification, "a" * 64, "c" * 64, 7)

    def test_real_coverage_produces_full_and_remaining_plans(self):
        baseline = {"name": "baseline", "checkpoint": "/base.pt", "sha256": "a" * 64}
        candidate = {"name": "candidate", "checkpoint": "/candidate.pt", "sha256": "b" * 64}
        coverage, inventory, paired, legacy, fresh = selection.coverage_plans(
            COVERAGE, digest(MEMBERSHIP), baseline, candidate, 7200
        )
        self.assertEqual((len(paired["cases"]), len(legacy["cases"]), len(fresh["cases"])), (24, 37, 16))
        self.assertEqual(len(inventory), 61)
        self.assertTrue(
            all(
                inventory[i]["case"]["split"] == "test"
                for i in coverage["paired_case_ids"]
                if inventory[i]["path"] in fresh["cases"]
            )
        )
        with self.assertRaisesRegex(ValueError, "membership"):
            selection.coverage_plans(COVERAGE, "e" * 64, baseline, candidate, 7200)

    def development_fixture(self, include_baseline=True, split="val", pass_audit=True):
        root = self.root / "development"
        root.mkdir()
        case = {"name": "one", "case": {"id": "one", "split": split, "frozen_case": None}}
        models = [{"name": "candidate", "sha256": "c" * 64}]
        if include_baseline:
            models.append({"name": "baseline", "sha256": "b" * 64})
        lock = {
            "cases": [case],
            "models": models,
            "source_sha256": {},
            "graph_sha256": {},
            "coordinator_sha256": "coord",
            "timeout_seconds": 7200,
        }
        lock_path = root / "plan.lock.json"
        lock_path.write_text(json.dumps(lock))
        receipt = {
            "completion_sha256": "1" * 64,
            "frames_sha256": "2" * 64,
            "frame_count": 3,
            "summary_sha256": "3" * 64,
            "result": "finished",
        }
        manifest = {"complete": True, "lock_sha256": digest(lock_path), "episodes": {}}
        for model in models:
            d = root / model["name"] / "one"
            d.mkdir(parents=True)
            (d / "summary.json").write_text("{}")
            (d / "telemetry.json").write_text("[]")
            audit = {
                "passed": pass_audit,
                "completion_sha256": receipt["completion_sha256"],
                "frames_sha256": receipt["frames_sha256"],
                "frame_count": 3,
                "telemetry_sha256": digest(d / "telemetry.json"),
            }
            (d / "lab_artifact_audit.json").write_text(json.dumps(audit))
            manifest["episodes"][f"{model['name']}/one"] = {"status": "complete", "receipt": receipt}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return lock_path, receipt

    def test_development_rejects_fresh_cases_and_unbound_audits(self):
        lock, receipt = self.development_fixture(split="test")
        with patch.object(selection, "validate_episode", return_value=receipt):
            with self.assertRaisesRegex(ValueError, "fresh test"):
                selection.verify_development([lock], "c" * 64, "b" * 64, "e" * 64, "f" * 64)
        data = json.loads(lock.read_text())
        data["cases"][0]["case"]["split"] = "val"
        lock.write_text(json.dumps(data))
        manifest = json.loads((lock.parent / "manifest.json").read_text())
        manifest["lock_sha256"] = digest(lock)
        (lock.parent / "manifest.json").write_text(json.dumps(manifest))
        audit_path = lock.parent / "candidate/one/lab_artifact_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["completion_sha256"] = "0" * 64
        audit_path.write_text(json.dumps(audit))
        with patch.object(selection, "validate_episode", return_value=receipt):
            with self.assertRaisesRegex(ValueError, "unbound"):
                selection.verify_development([lock], "c" * 64, "b" * 64, "e" * 64, "f" * 64)

    def test_development_accepts_paired_audited_states_and_rejects_mismatch(self):
        root = self.root / "paired"
        root.mkdir()
        cases = [
            {"name": "val", "case": {"id": "val", "split": "val", "frozen_case": None}},
            {"name": "reg", "case": {"id": "reg", "split": "regression", "frozen_case": "old"}},
        ]
        models = [{"name": "candidate", "sha256": "c" * 64}, {"name": "baseline", "sha256": "b" * 64}]
        lock_path = root / "plan.lock.json"
        lock_path.write_text(
            json.dumps(
                {
                    "cases": cases,
                    "models": models,
                    "source_sha256": {},
                    "graph_sha256": {},
                    "coordinator_sha256": "coord",
                    "timeout_seconds": 7200,
                }
            )
        )
        receipt = {
            "completion_sha256": "1" * 64,
            "frames_sha256": "2" * 64,
            "frame_count": 3,
            "summary_sha256": "3" * 64,
            "result": "finished",
        }
        manifest = {"complete": True, "lock_sha256": digest(lock_path), "episodes": {}}
        for case in cases:
            for model in models:
                episode = root / model["name"] / case["name"]
                (episode / "evidence").mkdir(parents=True)
                (episode / "summary.json").write_text('{"result": "finished"}')
                (episode / "telemetry.json").write_text("[]")
                state_sha = "e" * 64 if model["name"] == "candidate" else "f" * 64
                (episode / "export.json").write_text(json.dumps({"dependencies": [], "sha256": {}}))
                for name in ("realization.json", "appearance.json"):
                    (episode / name).write_text("{}")
                with ZipFile(episode / "source.zip", "w"):
                    pass
                metadata = {key: "same" for key in FIELDS}
                metadata.update(
                    source_sha256={},
                    source_archive_sha256=digest(episode / "source.zip"),
                    realization_sha256=digest(episode / "realization.json"),
                    appearance_sha256=digest(episode / "appearance.json"),
                )
                metadata.update(
                    state_sha256=state_sha, case=case["case"], export_sha256=digest(episode / "export.json")
                )
                (episode / "evidence/metadata.json").write_text(json.dumps(metadata))
                (episode / "lab_artifact_audit.json").write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "completion_sha256": receipt["completion_sha256"],
                            "frames_sha256": receipt["frames_sha256"],
                            "frame_count": 3,
                            "telemetry_sha256": digest(episode / "telemetry.json"),
                        }
                    )
                )
                manifest["episodes"][f"{model['name']}/{case['name']}"] = {"status": "complete", "receipt": receipt}
        (root / "manifest.json").write_text(json.dumps(manifest))
        with patch.object(selection, "validate_episode", return_value=receipt):
            result = selection.verify_development([lock_path], "c" * 64, "b" * 64, "e" * 64, "f" * 64)
            self.assertEqual(result["paired_case_ids"], ["reg", "val"])
            self.assertEqual(len(result["episodes"]), 4)
            with self.assertRaisesRegex(ValueError, "model state differs"):
                selection.verify_development([lock_path], "c" * 64, "b" * 64, "0" * 64, "f" * 64)

    def test_freeze_requires_all_eight_validation_and_eight_regression_pairs(self):
        members = json.loads(MEMBERSHIP.read_text())["cases"]
        dev_cases = [normalize_case(c) for c in members if c["split"] in ("val", "regression")]
        dev = {
            "plans": [],
            "paired_case_ids": [c["id"] for c in dev_cases],
            "episodes": [{"case": case, "role": role} for case in dev_cases for role in ("baseline", "candidate")],
        }
        signature = {"tensor": {"shape": [1], "dtype": "float32"}}
        candidate = {"checkpoint_sha256": "c" * 64, "step": 7, "state_sha256": "e" * 64, "signature": signature}
        baseline = {"checkpoint_sha256": "b" * 64, "step": 4, "state_sha256": "f" * 64, "signature": signature}
        args = argparse.Namespace(
            candidate=self.root / "candidate.pt",
            baseline=self.root / "baseline.pt",
            training=self.root,
            dataset=self.root / "manifest.json",
            dataset_verification=None,
            coverage=COVERAGE,
            development_plan=[self.root / "plan.lock.json"],
            reason="Selected using development evidence",
            timeout_seconds=7200,
            out=self.root / "selection",
        )
        training = {"membership_sha256": digest(MEMBERSHIP)}
        with (
            patch.object(selection, "checkpoint_info", side_effect=[copy.deepcopy(candidate), copy.deepcopy(baseline)]),
            patch.object(selection, "verify_training", return_value=training),
            patch.object(selection, "verify_development", return_value=copy.deepcopy(dev)),
            patch.object(selection, "recheck_inputs"),
        ):
            result = selection.freeze(args)
        self.assertEqual(len(result["development"]["reusable_paired_regression_episodes"]), 16)
        self.assertEqual(len(json.loads((args.out / "fresh_plan.json").read_text())["cases"]), 16)
        self.assertEqual(len(json.loads((args.out / "paired_plan.json").read_text())["cases"]), 24)
        self.assertEqual(result["plans"]["fresh_plan.json"], digest(args.out / "fresh_plan.json"))
        args.out = self.root / "missing"
        dev["episodes"].pop()
        with (
            patch.object(selection, "checkpoint_info", side_effect=[copy.deepcopy(candidate), copy.deepcopy(baseline)]),
            patch.object(selection, "verify_training", return_value=training),
            patch.object(selection, "verify_development", return_value=dev),
            patch.object(selection, "recheck_inputs"),
        ):
            with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
                selection.freeze(args)
        self.assertFalse(args.out.exists())

    def test_portable_plans_survive_copying_the_mirrored_tree(self):
        source = self.root / "source"
        coverage_dir = source / "docs/isaac-lab-training"
        shutil.copytree(COVERAGE.parent, coverage_dir)
        candidate_path = source / "models/candidate.pt"
        baseline_path = source / "models/baseline.pt"
        candidate_path.parent.mkdir(parents=True)
        candidate_path.write_bytes(b"candidate checkpoint")
        baseline_path.write_bytes(b"baseline checkpoint")
        members = json.loads((coverage_dir / "membership.json").read_text())["cases"]
        cases = [normalize_case(c) for c in members if c["split"] in ("val", "regression")]
        development = {
            "plans": [],
            "paired_case_ids": [case["id"] for case in cases],
            "episodes": [{"case": case, "role": role} for case in cases for role in ("baseline", "candidate")],
        }
        signature = {"tensor": {"shape": [1], "dtype": "float32"}}
        candidate = {
            "checkpoint_sha256": digest(candidate_path),
            "step": 7,
            "state_sha256": "e" * 64,
            "signature": signature,
        }
        baseline = {
            "checkpoint_sha256": digest(baseline_path),
            "step": 4,
            "state_sha256": "f" * 64,
            "signature": signature,
        }
        args = argparse.Namespace(
            candidate=candidate_path,
            baseline=baseline_path,
            training=source / "training",
            dataset=source / "data/manifest.json",
            dataset_verification=None,
            coverage=coverage_dir / COVERAGE.name,
            development_plan=[source / "dev/plan.lock.json"],
            reason="Selected using development evidence",
            timeout_seconds=7200,
            out=source / "runs/selection",
            portable_plans=True,
        )
        training = {"membership_sha256": digest(coverage_dir / "membership.json")}
        with (
            patch.object(selection, "checkpoint_info", side_effect=[copy.deepcopy(candidate), copy.deepcopy(baseline)]),
            patch.object(selection, "verify_training", return_value=training),
            patch.object(selection, "verify_development", return_value=development),
            patch.object(selection, "recheck_inputs"),
        ):
            selection.freeze(args)
        repo = self.root / "runtime"
        (repo / "src/fsop").mkdir(parents=True)
        (repo / "src/fsop/runtime.py").write_text("pass\n")
        for graph in GRAPH_FILES:
            (repo / graph).parent.mkdir(parents=True, exist_ok=True)
            (repo / graph).write_bytes(b"graph")
        copied = (self.root / "copied").resolve()
        shutil.copytree(source, copied)
        for name in ("fresh_plan.json", "paired_plan.json", "legacy_plan.json"):
            original = args.out / name
            transferred = copied / "runs/selection" / name
            self.assertEqual(digest(original), digest(transferred))
            plan = json.loads(transferred.read_text())
            self.assertTrue(all(not Path(path).is_absolute() for path in plan["cases"]))
            self.assertTrue(all(not Path(model["checkpoint"]).is_absolute() for model in plan["models"]))
            lock = load_plan(transferred, repo=repo)
            for model in lock["models"]:
                self.assertTrue(Path(model["checkpoint"]).is_relative_to(copied))
                self.assertEqual(digest(model["checkpoint"]), model["sha256"])
            self.assertTrue(all(Path(case["path"]).is_relative_to(copied) for case in lock["cases"]))


if __name__ == "__main__":
    unittest.main()


def test_training_chain_checks_each_checkpoint_and_run(tmp_path):
    import torch

    fixture = SelectionTests()
    fixture.root = tmp_path
    first, dataset, verification = fixture.make_training()
    second = tmp_path / "second"
    shutil.copytree(first, second)

    def checkpoint(path, step, value):
        torch.save(
            {
                "step": step,
                "model": {
                    f"{prefix}.weight": torch.tensor([value]) for prefix in ("base.vision", "base.policy", "bev", "img")
                },
            },
            path,
        )
        return selection.checkpoint_info(path)

    baseline = checkpoint(tmp_path / "base.pt", 0, 1)
    intermediate = checkpoint(tmp_path / "mid.pt", 7, 2)
    candidate = checkpoint(tmp_path / "candidate.pt", 5, 3)
    for run, init, result in ((first, baseline, intermediate), (second, intermediate, candidate)):
        data = json.loads((run / "run.json").read_text())
        data["init_sha256"] = init["checkpoint_sha256"]
        (run / "run.json").write_text(json.dumps(data))
        (run / "history.jsonl").write_text(
            json.dumps({"step": result["step"], "checkpoint_sha256": result["checkpoint_sha256"]}) + "\n"
        )
    chain = [(first, tmp_path / "mid.pt")]
    proof = selection.verify_training_chain(second, dataset, verification, baseline, candidate, chain)
    assert proof["ancestors"][0]["checkpoint_sha256"] == intermediate["checkpoint_sha256"]
    with unittest.TestCase().assertRaisesRegex(ValueError, "initialization"):
        selection.verify_training_chain(
            first, dataset, verification, baseline, candidate, [(second, tmp_path / "mid.pt")]
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate or cyclic"):
        selection.verify_training_chain(second, dataset, verification, baseline, candidate, chain + chain)
    (first / "history.jsonl").write_text(
        json.dumps({"step": 8, "checkpoint_sha256": intermediate["checkpoint_sha256"]}) + "\n"
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "step differs"):
        selection.verify_training_chain(second, dataset, verification, baseline, candidate, chain)
    (first / "history.jsonl").write_text(
        json.dumps({"step": 7, "checkpoint_sha256": intermediate["checkpoint_sha256"]}) + "\n"
    )
    (first / "source.zip").write_bytes(b"tampered")
    with unittest.TestCase().assertRaisesRegex(ValueError, "source archive"):
        selection.verify_training_chain(second, dataset, verification, baseline, candidate, chain)


def test_reused_baseline_preserves_old_candidate_and_rejects_mismatches(tmp_path):
    from scripts.lab_comparison import FIELDS

    cases = [
        {"name": "val", "case": {"id": "val", "split": "val", "frozen_case": None}},
        {"name": "reg", "case": {"id": "reg", "split": "regression", "frozen_case": "old"}},
    ]
    receipt = {"completion_sha256": "1" * 64, "frames_sha256": "2" * 64, "frame_count": 1}

    def make_lock(name, models):
        root = tmp_path / name
        root.mkdir()
        lock = {
            "cases": cases,
            "models": models,
            "source_sha256": {"source": "hash"},
            "graph_sha256": {"graph": "hash"},
            "coordinator_sha256": "coordinator",
            "timeout_seconds": 7200,
        }
        path = root / "plan.lock.json"
        path.write_text(json.dumps(lock))
        manifest = {"complete": True, "lock_sha256": digest(path), "episodes": {}}
        for case in cases:
            for model in models:
                episode = root / model["name"] / case["name"]
                (episode / "evidence").mkdir(parents=True)
                (episode / "summary.json").write_text('{"result": "timeout"}')
                (episode / "telemetry.json").write_text("[]")
                (episode / "export.json").write_text(json.dumps({"dependencies": [], "sha256": {}}))
                for name in ("realization.json", "appearance.json"):
                    (episode / name).write_text("{}")
                with ZipFile(episode / "source.zip", "w"):
                    pass
                metadata = {key: "same" for key in FIELDS}
                metadata.update(
                    source_sha256={},
                    source_archive_sha256=digest(episode / "source.zip"),
                    realization_sha256=digest(episode / "realization.json"),
                    appearance_sha256=digest(episode / "appearance.json"),
                )
                metadata.update(
                    case=case["case"],
                    state_sha256="f" * 64 if model["name"] == "baseline" else "e" * 64,
                    export_sha256=digest(episode / "export.json"),
                )
                (episode / "evidence/metadata.json").write_text(json.dumps(metadata))
                (episode / "lab_artifact_audit.json").write_text(
                    json.dumps({"passed": True, **receipt, "telemetry_sha256": digest(episode / "telemetry.json")})
                )
                manifest["episodes"][f"{model['name']}/{case['name']}"] = {"status": "complete", "receipt": receipt}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return path

    old = make_lock("old", [{"name": "baseline", "sha256": "b" * 64}, {"name": "candidate", "sha256": "a" * 64}])
    new = make_lock("new", [{"name": "candidate", "sha256": "c" * 64}])
    args = ([old, new], "c" * 64, "b" * 64, "e" * 64, "f" * 64)
    with patch.object(selection, "validate_episode", return_value=receipt):
        result = selection.verify_development(*args)
        assert result["paired_case_ids"] == ["reg", "val"]
        assert sum(row["role"] is None for row in result["episodes"]) == 2
        with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate development lock"):
            selection.verify_development([old, new, new], *args[1:])
        metadata_path = new.parent / "candidate/val/evidence/metadata.json"
        metadata = json.loads(metadata_path.read_text())
        for field in (
            "source_sha256",
            "graph_sha256",
            "realization_sha256",
            "initial_physical_state",
            "intrinsics",
            "sim_render_settings",
            "command_mode",
            "asset_sha256",
        ):
            changed = dict(metadata, **{field: "different"})
            metadata_path.write_text(json.dumps(changed))
            with unittest.TestCase().assertRaises(ValueError):
                selection.verify_development(*args)
        missing = dict(metadata)
        del missing["physics_settings"]
        metadata_path.write_text(json.dumps(missing))
        with unittest.TestCase().assertRaisesRegex(ValueError, "Missing comparison"):
            selection.verify_development(*args)
        metadata_path.write_text(json.dumps(metadata))
        base_only = make_lock("base", [{"name": "baseline", "sha256": "b" * 64}])
        assert selection.verify_development([base_only, new], *args[1:])["paired_case_ids"] == ["reg", "val"]
        same = make_lock("same", [{"name": "baseline", "sha256": "b" * 64}, {"name": "candidate", "sha256": "b" * 64}])
        for case in cases:
            path = same.parent / "candidate" / case["name"] / "evidence/metadata.json"
            item = json.loads(path.read_text())
            item["state_sha256"] = "f" * 64
            path.write_text(json.dumps(item))
        assert selection.verify_development([same], "b" * 64, "b" * 64, "f" * 64, "f" * 64)["paired_case_ids"] == [
            "reg",
            "val",
        ]
        with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate selected"):
            selection.verify_development([old, base_only, new], *args[1:])
        lock = json.loads(new.read_text())
        lock["graph_sha256"] = {"graph": "different"}
        new.write_text(json.dumps(lock))
        manifest_path = new.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["lock_sha256"] = digest(new)
        manifest_path.write_text(json.dumps(manifest))
        with unittest.TestCase().assertRaisesRegex(ValueError, "lock comparison conditions"):
            selection.verify_development(*args)
