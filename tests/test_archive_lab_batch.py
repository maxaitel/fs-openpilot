import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import archive_lab_batch as archive


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "remote" / "batch"
    case = {"id": "one"}
    lock = {
        "version": 1,
        "cases": [{"name": "one", "case": case}],
        "models": [{"name": "candidate", "sha256": "checkpoint"}],
    }
    write(root / "plan.lock.json", lock)
    episode = root / "candidate/one"
    evidence = episode / "evidence"
    evidence.mkdir(parents=True)
    (episode / "assets").mkdir()
    (episode / "assets/mesh.bin").write_bytes(b"asset")
    (episode / "source.zip").write_bytes(b"source")
    (episode / "scene.usda").write_bytes(b"scene")
    write(episode / "realization.json", {})
    write(episode / "appearance.json", {})
    write(
        episode / "export.json",
        {"dependencies": ["mesh.bin"], "sha256": {"mesh.bin": sha(episode / "assets/mesh.bin")}},
    )
    metadata = {"case": case, "checkpoint_sha256": "checkpoint", "pilot_only": False}
    for field, name in (
        ("source_archive_sha256", "source.zip"),
        ("realization_sha256", "realization.json"),
        ("appearance_sha256", "appearance.json"),
        ("scene_sha256", "scene.usda"),
        ("export_sha256", "export.json"),
    ):
        metadata[field] = sha(episode / name)
    write(evidence / "metadata.json", metadata)
    (evidence / "frame_000000.npz").write_bytes(b"raw data")
    frame = {
        "frame": 0,
        "file": "frame_000000.npz",
        "sha256": sha(evidence / "frame_000000.npz"),
        "metadata_sha256": sha(evidence / "metadata.json"),
    }
    write(evidence / "manifest.jsonl", frame)
    write(episode / "case.json", case)
    write(
        episode / "summary.json",
        {
            "case": case,
            "checkpoint_sha256": "checkpoint",
            "pilot_only": False,
            "recorded_frames": 1,
            "result": "timeout",
        },
    )
    write(episode / "frames.json", [])
    write(episode / "telemetry.json", [])
    write(
        evidence / "completion.json",
        {
            "version": 1,
            "frame_count": 1,
            "summary_path": "../summary.json",
            "summary_sha256": sha(episode / "summary.json"),
            "metadata_sha256": sha(evidence / "metadata.json"),
            "manifest_sha256": sha(evidence / "manifest.jsonl"),
        },
    )
    audit = {
        "passed": True,
        "frame_count": 1,
        "completion_sha256": sha(evidence / "completion.json"),
        "frames_sha256": sha(episode / "frames.json"),
        "telemetry_sha256": sha(episode / "telemetry.json"),
    }
    write(episode / "lab_artifact_audit.json", audit)
    videos = {}
    for name in ("debug", "research"):
        (episode / f"{name}.mp4").write_bytes(name.encode())
        videos[name] = {"sha256": sha(episode / f"{name}.mp4"), "fps": 20, "decoded_frames": 41}
    write(
        episode / "lab_video_audit.json",
        {
            "inference_frames": 1,
            "terminal_frames": 40,
            "completion_sha256": sha(evidence / "completion.json"),
            "artifact_audit_sha256": sha(episode / "lab_artifact_audit.json"),
            "videos": videos,
        },
    )
    row = {
        "frame_count": 1,
        "result": "timeout",
        "completion_sha256": sha(evidence / "completion.json"),
        "summary_sha256": sha(episode / "summary.json"),
        "frames_sha256": sha(episode / "frames.json"),
    }
    write(
        root / "manifest.json",
        {
            "version": 1,
            "complete": True,
            "lock_sha256": sha(root / "plan.lock.json"),
            "episodes": {"candidate/one": {"status": "complete", "receipt": row}},
        },
    )
    return root


def transport(monkeypatch, batch, mutate=None):
    calls = []

    def remote(host, root, name):
        calls.append(1)
        if len(calls) == 2 and mutate == "remote":
            (batch / "candidate/one/assets/mesh.bin").write_bytes(b"changed")
        return {
            "hostname": "spark",
            "root": str(batch.parent),
            "batch": str(batch),
            "entries": archive.inventory(batch),
        }

    def run(args, **kwargs):
        assert args[0] == "rsync"
        shutil.copytree(batch, Path(args[-1]), dirs_exist_ok=True)
        if mutate == "local":
            (Path(args[-1]) / "candidate/one/assets/mesh.bin").write_bytes(b"changed")

    monkeypatch.setattr(archive, "remote_inventory", remote)
    monkeypatch.setattr(archive.subprocess, "run", run)


def test_archive_copies_and_binds_all_files(batch, tmp_path, monkeypatch):
    transport(monkeypatch, batch)
    dest = tmp_path / "archived"
    receipt = json.loads(archive.archive("spark", "/work/task", "batch", dest).read_text())
    assert receipt["inventory"] == archive.inventory(batch) == archive.inventory(dest)
    assert receipt["remote_hostname"] == "spark"
    assert receipt["episodes"]["candidate/one"]["completion_sha256"] == sha(
        dest / "candidate/one/evidence/completion.json"
    )
    assert not (dest / "archived.receipt.json").exists()


@pytest.mark.parametrize("mutation", ["remote", "local"])
def test_no_receipt_if_source_or_copy_changes(batch, tmp_path, monkeypatch, mutation):
    transport(monkeypatch, batch, mutation)
    with pytest.raises(ValueError):
        archive.archive("spark", "/work/task", "batch", tmp_path / "archived")
    assert not (tmp_path / "archived.receipt.json").exists()


@pytest.mark.parametrize("bad", ["manifest", "audit", "video", "symlink", "frame", "asset"])
def test_incomplete_or_unsafe_batch_rejected(batch, tmp_path, monkeypatch, bad):
    episode = batch / "candidate/one"
    if bad == "manifest":
        manifest = json.loads((batch / "manifest.json").read_text())
        manifest["complete"] = False
        write(batch / "manifest.json", manifest)
    elif bad == "audit":
        (episode / "lab_artifact_audit.json").unlink()
    elif bad == "video":
        (episode / "research.mp4").write_bytes(b"wrong")
    elif bad == "symlink":
        (episode / "assets/escape").symlink_to(tmp_path)
    elif bad == "frame":
        (episode / "evidence/frame_000000.npz").write_bytes(b"wrong")
    else:
        (episode / "assets/mesh.bin").write_bytes(b"wrong")
    transport(monkeypatch, batch)
    with pytest.raises((ValueError, OSError)):
        archive.archive("spark", "/work/task", "batch", tmp_path / "archived")
    assert not (tmp_path / "archived.receipt.json").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "batch/../else", "batch//else"])
def test_rejects_unconfined_batch_name(tmp_path, name):
    with pytest.raises(ValueError):
        archive.archive("spark", "/work/task", name, tmp_path / "archived")


def test_remote_inventory_executes_safely_with_quoted_paths(batch, tmp_path, monkeypatch):
    import shlex
    import subprocess

    awkward = tmp_path / "task 'quoted'; literal"
    batch.parent.rename(awkward)
    original_run = subprocess.run

    def ssh(args, **kwargs):
        assert args[:4] == ["ssh", "-o", "BatchMode=yes", "spark"]
        return original_run(shlex.split(args[4]), **kwargs)

    monkeypatch.setattr(archive.subprocess, "run", ssh)
    result = archive.remote_inventory("spark", str(awkward), "batch")
    assert result["batch"] == str(awkward / "batch")
    assert result["entries"] == archive.inventory(awkward / "batch")
    (awkward / "link").symlink_to(awkward / "batch")
    with pytest.raises(subprocess.CalledProcessError):
        archive.remote_inventory("spark", str(awkward), "link")
