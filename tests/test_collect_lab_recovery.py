import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from fsop.sim.expert import ExpertParams
from scripts import collect_lab_recovery as coordinator


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    source = tmp_path / "source"
    source.mkdir()
    cases = [{"id": f"case-{i}", "split": "train" if i < 39 else "val"} for i in range(40)]
    ids = [c["id"] for c in cases[:8]]
    membership = docs / "membership.json"
    write(membership, {"cases": cases})
    for case in cases[:8]:
        write(docs / "cases" / (case["id"] + ".json"), case)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    recipe = {
        "schema_version": 1,
        "membership": "membership.json",
        "membership_sha256": coordinator.digest(membership),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": coordinator.digest(checkpoint),
        "recovery_case_ids": ids,
        "pilot_case_ids": ids[:4],
        "collection": {"takeover_seconds": 120, "lateral_threshold_m": 0.4, "capture_hz": 20, "image_hz": 5},
        "gate": {
            "min_lateral_m": 0.3,
            "min_yaw_rad": 0.1,
            "min_anchors": 8,
            "window_seconds": 2,
            "return_within_seconds": 4,
            "settled_lateral_m": 0.2,
            "settled_yaw_rad": 0.1,
            "settled_seconds": 0.5,
        },
        "case_sha256": {key: coordinator.digest(docs / "cases" / (key + ".json")) for key in ids},
        "teachers": {key: asdict(ExpertParams()) for key in ids},
    }
    recipe_path = docs / "recipe.json"
    write(recipe_path, recipe)
    for case in cases:
        episode = source / case["id"]
        episode.mkdir()
        (episode / "original").write_text(case["id"])
    manifest = source / "manifest.json"
    write(manifest, {"episodes": [{"path": case["id"]} for case in cases]})
    monkeypatch.setattr(coordinator, "SOURCE_MANIFEST", coordinator.digest(manifest))
    report = {"passed": True, "manifest_sha256": coordinator.digest(manifest)}
    write(source / "verification.json", report)
    monkeypatch.setattr(coordinator, "verify_dataset", lambda *a: report)
    monkeypatch.setattr(coordinator, "verify_recovery", lambda episode, recipe: {"passed": True, "id": episode.name})
    monkeypatch.setattr(coordinator.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 2**30))
    monkeypatch.setattr(coordinator, "source_inventory", lambda: {"collector.py": "digest"})
    monkeypatch.setattr(coordinator, "require_idle_gpu", lambda: None)
    runs = []

    def child(command, log, timeout):
        key = Path(command[command.index("--out") + 1]).name
        runs.append(key)
        log.write_text("attempt")
        (tmp_path / "out" / key).mkdir()
        write(tmp_path / "out" / key / "meta.json", {"source_sha256": {"collector.py": "digest"}})

    monkeypatch.setattr(coordinator, "run_child", child)
    args = SimpleNamespace(
        recipe=recipe_path,
        source=source,
        source_receipt=None,
        out=tmp_path / "out",
        checkpoint=None,
        timeout=1200,
        phase="pilot",
    )
    return args, runs, recipe


def test_pilot_stop_and_no_silent_retry(setup, monkeypatch):
    args, runs, recipe = setup
    monkeypatch.setattr(
        coordinator, "verify_recovery", lambda episode, recipe: {"passed": episode.name != "case-1", "id": episode.name}
    )
    with pytest.raises(ValueError, match="gate failed"):
        coordinator.collect(args)
    assert runs == ["case-0", "case-1"]
    assert json.loads((args.out / "recovery_gates/case-1.json").read_text())["passed"] is False
    with pytest.raises(ValueError, match="Missing or failed prior attempt"):
        coordinator.collect(args)
    assert runs == ["case-0", "case-1"]
    assert os.path.samefile(args.source / "case-8/original", args.out / "case-8/original")


@pytest.mark.parametrize("drift", ["recipe", "checkpoint", "receipt"])
def test_resume_rejects_drift(setup, drift):
    args, runs, recipe = setup
    coordinator.collect(args)
    assert runs == recipe["pilot_case_ids"]
    if drift == "recipe":
        with args.recipe.open("a") as stream:
            stream.write(" ")
    elif drift == "checkpoint":
        Path(recipe["checkpoint"]).write_bytes(b"changed")
    else:
        with (args.out / "recovery_gates/case-0.json").open("a") as stream:
            stream.write(" ")
    args.phase = "remaining"
    with pytest.raises(ValueError):
        coordinator.collect(args)
    assert runs == recipe["pilot_case_ids"]


def test_remaining_requires_pilots(setup):
    args, runs, _ = setup
    args.phase = "remaining"
    with pytest.raises(ValueError, match="Missing or failed prior attempt"):
        coordinator.collect(args)
    assert runs == []


def test_timeout_kills_child_process_group(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
        'open(sys.argv[1], "w").write(str(p.pid))\n'
        "time.sleep(60)\n"
    )
    pidfile = tmp_path / "pid"
    with pytest.raises(subprocess.TimeoutExpired):
        coordinator.run_child([sys.executable, str(script), str(pidfile)], tmp_path / "log", 1)
    pid = int(pidfile.read_text())
    # A killed child may briefly remain as an unreaped zombie; it cannot execute.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith("Z")


def test_successful_phases_write_complete_verified_manifest(setup, monkeypatch):
    args, runs, recipe = setup

    class Episode:
        def __init__(self, path):
            self.anchors = [1, 2]
            self.rejections = {"contact": 0}
            self.meta = {"wall_seconds": 3, "summary": {"result": "finished"}}

    monkeypatch.setattr(coordinator, "NativeEpisode", Episode)
    coordinator.collect(args)
    args.phase = "remaining"
    coordinator.collect(args)
    assert runs == recipe["recovery_case_ids"]
    result = json.loads((args.out / "manifest.json").read_text())
    assert result["complete"] and len(result["episodes"]) == 40
    assert json.loads((args.out / "verification.json").read_text())["passed"]
    coordinator.collect(args)
    assert runs == recipe["recovery_case_ids"]


def test_gpu_guard_allows_only_sunshine(monkeypatch):
    monkeypatch.setattr(
        coordinator.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="123, /usr/bin/sunshine\n")
    )
    coordinator.require_idle_gpu()
    monkeypatch.setattr(
        coordinator.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="123, /usr/bin/sunshine\n456, python\n"),
    )
    with pytest.raises(ValueError, match="another compute process"):
        coordinator.require_idle_gpu()
