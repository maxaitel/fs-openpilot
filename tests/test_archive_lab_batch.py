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


def version2(batch):
    from fsop.eval.controller import controller_contract

    contract = controller_contract("pp")
    lock = json.loads((batch / "plan.lock.json").read_text())
    lock["version"] = 2
    lock["models"][0]["controller_contract"] = contract
    write(batch / "plan.lock.json", lock)
    episode = batch / "candidate/one"
    for name in ("evidence/metadata.json", "summary.json", "lab_artifact_audit.json", "lab_video_audit.json"):
        path = episode / name
        value = json.loads(path.read_text())
        value["controller_contract"] = contract
        write(path, value)
    reseal(batch)


def reseal(batch):
    episode = batch / "candidate/one"
    evidence = episode / "evidence"
    frame = json.loads((evidence / "manifest.jsonl").read_text())
    frame["metadata_sha256"] = sha(evidence / "metadata.json")
    write(evidence / "manifest.jsonl", frame)
    completion = json.loads((evidence / "completion.json").read_text())
    for field, path in (("metadata_sha256", evidence / "metadata.json"),
                        ("manifest_sha256", evidence / "manifest.jsonl"),
                        ("summary_sha256", episode / "summary.json")):
        completion[field] = sha(path)
    write(evidence / "completion.json", completion)
    audit = json.loads((episode / "lab_artifact_audit.json").read_text())
    audit["completion_sha256"] = sha(evidence / "completion.json")
    write(episode / "lab_artifact_audit.json", audit)
    video = json.loads((episode / "lab_video_audit.json").read_text())
    video["completion_sha256"] = sha(evidence / "completion.json")
    video["artifact_audit_sha256"] = sha(episode / "lab_artifact_audit.json")
    write(episode / "lab_video_audit.json", video)
    manifest = json.loads((batch / "manifest.json").read_text())
    manifest["lock_sha256"] = sha(batch / "plan.lock.json")
    receipt = manifest["episodes"]["candidate/one"]["receipt"]
    receipt["completion_sha256"] = sha(evidence / "completion.json")
    receipt["summary_sha256"] = sha(episode / "summary.json")
    write(batch / "manifest.json", manifest)


def test_version2_contract_is_accepted(batch):
    version2(batch)
    assert "candidate/one" in archive.validate_batch(batch, archive.inventory(batch))


@pytest.mark.parametrize("name", ["plan.lock.json", "candidate/one/evidence/metadata.json",
                                 "candidate/one/summary.json", "candidate/one/lab_artifact_audit.json",
                                 "candidate/one/lab_video_audit.json"])
def test_version2_rejects_resealed_controller_tampering(batch, name):
    version2(batch)
    path = batch / name
    data = json.loads(path.read_text())
    contract = data["models"][0]["controller_contract"] if name == "plan.lock.json" else data["controller_contract"]
    contract["mode"] = "op"
    write(path, data)
    reseal(batch)
    with pytest.raises(ValueError, match="Controller contract mismatch"):
        archive.validate_batch(batch, archive.inventory(batch))


def test_version2_producer_receipt_is_bound(batch):
    version2(batch)
    episode = batch / "candidate/one"
    video_path = episode / "lab_video_audit.json"
    video = json.loads(video_path.read_text())
    code = {key: "abc" for key in ("render_code_sha256", "process_code_sha256", "wrapper_code_sha256")}
    video.update(code)
    raw = {}
    for name in ("debug", "research"):
        path = episode / f"{name}-raw.mp4"
        path.write_bytes(name.encode())
        raw[name] = {"path": path.name, "sha256": sha(path)}
    inputs = {field: sha(episode / name) for field, name in (
        ("completion_sha256", "evidence/completion.json"), ("artifact_audit_sha256", "lab_artifact_audit.json"),
        ("frames_sha256", "frames.json"), ("telemetry_sha256", "telemetry.json"))}
    producer = {"version": 1, "numpy_version": "1.26.0", "inputs": {**inputs, **code},
                "controller_contract": dict(video["controller_contract"]), "inference_frames": 1,
                "terminal_frames": 40, "raw_videos": raw}
    receipt_path = episode / "producer_render_receipt.json"
    write(receipt_path, producer)
    video.update(producer_render_receipt_sha256=sha(receipt_path), producer_numpy_version="1.26.0")
    write(video_path, video)
    archive.validate_batch(batch, archive.inventory(batch))
    producer["controller_contract"]["mode"] = "op"
    write(receipt_path, producer)
    video["producer_render_receipt_sha256"] = sha(receipt_path)
    write(video_path, video)
    with pytest.raises(ValueError, match="Producer receipt mismatch"):
        archive.validate_batch(batch, archive.inventory(batch))


def test_verify_existing_requires_matching_stable_inventories(batch, tmp_path, monkeypatch):
    import importlib.util

    helper_path = Path(__file__).resolve().parents[1] / "scripts/verify_lab_archive.py"
    spec = importlib.util.spec_from_file_location("verify_existing", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    destination = tmp_path / "archived"
    shutil.copytree(batch, destination)
    calls = []

    def remote(host, root, name):
        calls.append(1)
        return {"hostname": "spark", "root": root, "batch": root + "/" + name,
                "entries": archive.inventory(batch)}

    monkeypatch.setattr(helper.archive, "remote_inventory", remote)
    receipt = helper.verify("spark", "/work/task", "batch", destination)
    assert len(calls) == 2
    assert json.loads(receipt.read_text())["inventory"] == archive.inventory(destination)
    with pytest.raises(ValueError, match="Receipt already exists"):
        helper.verify("spark", "/work/task", "batch", destination)

    other = tmp_path / "different"
    shutil.copytree(batch, other)
    (other / "candidate/one/summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Local inventory differs"):
        helper.verify("spark", "/work/task", "batch", other)
    assert not (tmp_path / "different.receipt.json").exists()
