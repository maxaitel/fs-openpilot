import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import rerun as rr

from scripts.finalize_isaac_rrd import finalize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recording(tmp_path, monkeypatch):
    original_is_dir, original_glob = Path.is_dir, Path.glob
    monkeypatch.setattr(Path, "is_dir", lambda p: True if str(p) == "/proc" else original_is_dir(p))
    monkeypatch.setattr(Path, "glob", lambda p, pattern: iter(()) if str(p) == "/proc"
                        else original_glob(p, pattern))
    run = tmp_path / "run"
    case = run / "case"
    (case / "evidence").mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "case"}]}))
    (run / "status.json").write_text(json.dumps({"complete": True, "expected_cases": 1,
        "cases": [{"case_id": "case", "technical_success": True}]}))
    (run / "postprocess_status.json").write_text(json.dumps({"complete": True,
        "cases": [{"case_id": "case"}]}))
    (case / "evidence/completion.json").write_text("{}")
    (case / "artifact_audit.json").write_text(json.dumps({"passed": True}))
    rr.init("finalization-test", spawn=False)
    rr.save(str(case / "debug.rrd"))
    rr.log("point", rr.Points3D([[1, 2, 3]]))
    rr.get_global_data_recording().flush()
    audit = {"case_id": "case", "completion_sha256": sha(case / "evidence/completion.json"),
             "artifact_audit_sha256": sha(case / "artifact_audit.json"),
             "rrd_sha256": sha(case / "debug.rrd"), "videos": {}}
    for name in ("debug", "research"):
        (case / f"{name}.mp4").write_bytes(b"final-video-hash-fixture")
        (case / f"{name}-source.mp4").write_bytes(b"source-video-hash-fixture")
        audit["videos"][name] = {"sha256": sha(case / f"{name}.mp4"),
                                 "source_sha256": sha(case / f"{name}-source.mp4")}
    (case / "video_audit.json").write_text(json.dumps(audit, indent=2))
    original = (case / "video_audit.json").read_bytes()
    yield run, manifest, case, original
    rr.disconnect()


def test_finalization_preserves_receipt_and_survives_repeat(recording):
    run, manifest, case, original = recording
    rr.disconnect()
    closed_hash = sha(case / "debug.rrd")
    assert closed_hash != json.loads(original)["rrd_sha256"]
    for _ in range(2):
        result = finalize(run, manifest)
        updated = json.loads((case / "video_audit.json").read_text())
        assert (case / "video_audit.before_rrd_close.json").read_bytes() == original
        assert updated["rrd_sha256"] == closed_hash
        assert updated["videos"] == json.loads(original)["videos"]
        assert updated["rrd_finalization"]["original_audit_sha256"] == hashlib.sha256(original).hexdigest()
        assert result["cases"][0]["video_audit_sha256"] == sha(case / "video_audit.json")
        assert result["complete"] and result["cases"][0]["footer_verified"]


def test_open_recording_is_rejected_without_rewriting_audit(recording):
    run, manifest, case, original = recording
    with pytest.raises(subprocess.CalledProcessError):
        finalize(run, manifest)
    assert (case / "video_audit.json").read_bytes() == original
    assert not (case / "video_audit.before_rrd_close.json").exists()


def test_changed_video_is_rejected_without_rewriting_audit(recording):
    run, manifest, case, original = recording
    rr.disconnect()
    (case / "debug.mp4").write_bytes(b"different-video")
    with pytest.raises(ValueError, match="Video changed"):
        finalize(run, manifest)
    assert (case / "video_audit.json").read_bytes() == original
    assert not (case / "video_audit.before_rrd_close.json").exists()
