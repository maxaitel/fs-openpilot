import json

import numpy as np
import pytest

from isaac_eval.evidence import EvidenceWriter, finalize_evidence
from isaac_eval.parity_cases import sha256_file
from isaac_eval.stress_experiment import CASES, VARIANTS
from scripts import report_isaac_stress as report


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "validate_plan", lambda *_: None)
    monkeypatch.setattr(report, "read_receipt", lambda *_: {})
    plan = {
        "manifest": {},
        "plan_sha256": "plan",
        "source_sha256": {"runtime": "source"},
        "tool_sha256": {"scripts/audit_isaac_episode.py": "auditor", "scripts/process_isaac_parity.py": "processor"},
    }
    plan_path = tmp_path / "experiment.json"
    write(plan_path, plan)
    for case in CASES:
        for variant in VARIANTS:
            path = tmp_path / variant / case
            path.mkdir(parents=True)
            metadata = {key: "same" for key in report.PAIR_KEYS}
            metadata["case"] = {"case_id": case}
            metadata["pilot_only"] = False
            evidence = path / "evidence"
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            with EvidenceWriter(evidence, metadata) as writer:
                writer.append(
                    {
                        "images": {"wide": image, "road": image},
                        "prediction": {"plan": [[0, 0], [1, 1]]},
                        "action": [0, 0],
                    }
                )
            summary = {
                "case": {"case_id": case},
                "variant": variant,
                "experiment_plan_sha256": "plan",
                "pass": False,
                "result": "timeout",
                "physical_cone_hits": 0,
                "doo": 0,
                "oc": 3,
                "recorded_frames": 1,
                "physics_sha256": "physics",
                "wall_seconds": 2.0,
                "seconds_simulated": 1.0,
            }
            write(path / "summary.json", summary)
            finalize_evidence(evidence, path / "summary.json")
            write(path / "frames.json", [{key: 0.01 for key in report.TIMING_KEYS}])
            write(path / "realization.json", {"same": True})
            write(path / "telemetry.json", [{"gpu_pct": 20, "cpu_pct": 10}, {"gpu_pct": 80}])
            for name in ("debug.mp4", "research.mp4", "debug.rrd"):
                (path / name).write_bytes(name.encode())
            completion = sha256_file(evidence / "completion.json")
            audit = {
                "passed": True,
                "case_id": case,
                "completion_sha256": completion,
                "frame_count": 1,
                "source_sha256": {"runtime": "source", "scripts/audit_isaac_episode.py": "auditor"},
                "summary_sha256": sha256_file(path / "summary.json"),
                "metadata_sha256": sha256_file(evidence / "metadata.json"),
                "manifest_sha256": sha256_file(evidence / "manifest.jsonl"),
                "frames_sha256": sha256_file(path / "frames.json"),
                "telemetry_sha256": sha256_file(path / "telemetry.json"),
                "realization_sha256": sha256_file(path / "realization.json"),
            }
            write(path / "artifact_audit.json", audit)
            videos = {
                "case_id": case,
                "completion_sha256": completion,
                "inference_frames": 1,
                "terminal_frames": 40,
                "artifact_audit_sha256": sha256_file(path / "artifact_audit.json"),
                "producer_sha256": "processor",
                "rrd_sha256": sha256_file(path / "debug.rrd"),
                "videos": {
                    name: {"sha256": sha256_file(path / f"{name}.mp4"), "fps": 20, "decoded_frames": 41}
                    for name in ("debug", "research")
                },
            }
            write(path / "video_audit.json", videos)
    return plan_path


def test_valid_report_and_failed_episodes_are_not_clean(experiment):
    value = report.build(experiment)
    assert value["totals"]["matched"] == {
        "passes": 0,
        "failures": 4,
        "terminal_results": {"timeout": 4},
        "physical_clean": 0,
        "contacts": 0,
    }
    page = (experiment.parent / "index.html").read_text()
    assert '<video id="matched"' in page and '<video id="stress"' in page
    assert len(value["pairs"]) == 4 and len(value["episodes"]) == 8
    episode = value["episodes"][0]
    assert episode["hardware"]["gpu_pct"] == {"count": 2, "median": 50.0, "p95": 77.0}
    assert episode["hardware"]["temp_c"] is None
    assert episode["real_time_factor"] == 0.5
    assert "Hardware by episode" in page and "50.0 / 77.0 (n=2)" in page


@pytest.mark.parametrize(
    "target",
    [
        "summary",
        "frames",
        "telemetry",
        "metadata",
        "manifest",
        "realization",
        "rrd",
        "audit_source",
        "audit_telemetry",
        "producer",
        "fps",
        "decoded",
        "inference",
        "terminal",
    ],
)
def test_report_rejects_tampered_evidence(experiment, target):
    path = experiment.parent / VARIANTS[0] / CASES[0]
    if target in {"summary", "frames", "telemetry", "metadata", "manifest", "realization", "rrd"}:
        file = {
            "summary": "summary.json",
            "frames": "frames.json",
            "telemetry": "telemetry.json",
            "metadata": "evidence/metadata.json",
            "manifest": "evidence/manifest.jsonl",
            "realization": "realization.json",
            "rrd": "debug.rrd",
        }[target]
        with (path / file).open("a") as stream:
            stream.write(" ")
    else:
        file = path / ("artifact_audit.json" if target in {"audit_source", "audit_telemetry"} else "video_audit.json")
        data = json.loads(file.read_text())
        if target == "audit_source":
            data["source_sha256"] = {}
        elif target == "audit_telemetry":
            data.pop("telemetry_sha256")
        elif target == "producer":
            data["producer_sha256"] = "wrong"
        elif target == "fps":
            data["videos"]["debug"]["fps"] = 1
        elif target == "decoded":
            data["videos"]["debug"]["decoded_frames"] = 0
        elif target == "inference":
            data["inference_frames"] = 2
        else:
            data["terminal_frames"] = 0
        write(file, data)
    with pytest.raises((ValueError, KeyError)):
        report.build(experiment)


def test_pair_mismatch_rejected_even_with_valid_seal(experiment):
    path = experiment.parent / VARIANTS[1] / CASES[0]
    evidence = path / "evidence"
    # Rebuild one legitimate seal with a different model state, then update the synthetic audit chain.
    metadata = json.loads((evidence / "metadata.json").read_text())
    metadata["state_sha256"] = "different"
    import shutil

    shutil.rmtree(evidence)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with EvidenceWriter(evidence, metadata) as writer:
        writer.append(
            {"images": {"wide": image, "road": image}, "prediction": {"plan": [[0, 0], [1, 1]]}, "action": [0, 0]}
        )
    finalize_evidence(evidence, path / "summary.json")
    audit = json.loads((path / "artifact_audit.json").read_text())
    audit.update(
        completion_sha256=sha256_file(evidence / "completion.json"),
        metadata_sha256=sha256_file(evidence / "metadata.json"),
        manifest_sha256=sha256_file(evidence / "manifest.jsonl"),
    )
    write(path / "artifact_audit.json", audit)
    video = json.loads((path / "video_audit.json").read_text())
    video.update(
        completion_sha256=audit["completion_sha256"], artifact_audit_sha256=sha256_file(path / "artifact_audit.json")
    )
    write(path / "video_audit.json", video)
    with pytest.raises(ValueError, match="Pair invariant"):
        report.build(experiment)


def test_telemetry_rejects_invalid_samples():
    with pytest.raises(ValueError, match="Invalid telemetry"):
        report.hardware_stats([{"gpu_pct": float("nan")}])
    assert all(value is None for value in report.hardware_stats([]).values())
