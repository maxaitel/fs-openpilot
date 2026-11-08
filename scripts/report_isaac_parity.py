#!/usr/bin/env python3
"""Build an offline report from a frozen Isaac parity suite and its manifest."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("skidpad", "retention", "ablation", "prior_failure")
TELEMETRY = ("gpu_pct", "power_w", "temp_c", "cpu_pct", "host_ram_gib", "process_rss_gib")
TIMINGS = ("render_s", "policy_with_capture_s", "evidence_s", "physics_s", "step_wall_s")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def clean(row):
    return bool(row.get("pass") is True and row.get("doo") == 0 and row.get("oc") == 0)


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return None
    pos = 0.95 * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": values[lo] + (values[hi] - values[lo]) * (pos - lo),
    }


def val(row, key):
    return str(row.get(key)) if row.get(key) is not None else "-"


def main():
    sys.path.insert(0, str(ROOT))
    from isaac_eval.evidence import verified_summary

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "isaac_eval/runs/parity_cases/manifest.json")
    p.add_argument("--protocol", type=Path, default=ROOT / "isaac_eval/runs/parity_protocol.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partial", action="store_true", help="allow incomplete monitoring output")
    p.add_argument(
        "--remote-evidence-root", help="SSH host:path retaining full NPZ/Rerun evidence audited on that host"
    )
    a = p.parse_args()
    manifest, protocol = read(a.manifest), read(a.protocol)
    cases = manifest["cases"]
    canonical = json.dumps(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest["manifest_sha256"]:
        raise ValueError("Manifest content hash mismatch")
    if (
        len(cases) != 41
        or Counter(c["group"] for c in cases) != {"skidpad": 20, "retention": 16, "ablation": 4, "prior_failure": 1}
        or len({c["case_id"] for c in cases}) != 41
    ):
        raise ValueError("Manifest does not contain the required 41 unique cases")
    if protocol["manifest_sha256"] != manifest["manifest_sha256"] or protocol["case_ids"] != [
        c["case_id"] for c in cases
    ]:
        raise ValueError("Protocol and manifest differ")
    provenance_path = ROOT / "isaac_eval/parity_reference/provenance.json"
    if digest(provenance_path) != manifest["reference_sha256"]:
        raise ValueError("Reference provenance hash mismatch")
    provenance = read(provenance_path)
    references = {r["source"]: r for r in provenance["results"]}
    for case in cases:
        ref = references[case["reference_source"]]
        path = ROOT / "isaac_eval/parity_reference" / ref["local_path"]
        if digest(path) != ref["sha256"] or ref["sha256"] != case["reference_sha256"]:
            raise ValueError(f"Reference hash mismatch for {case['case_id']}")
        document = read(path)
        if document.get("runs", [document])[case["reference_index"]] != case["reference_result"]:
            raise ValueError(f"Reference row mismatch for {case['case_id']}")
    status_path = a.run / "status.json"
    status = read(status_path) if status_path.exists() else {}
    statuses = {r["case_id"]: r for r in status.get("cases", [])}
    rows, all_frames, all_telemetry = [], [], []
    protocol_sha = digest(a.protocol)
    for case in cases:
        cid = case["case_id"]
        folder = a.run / cid
        record = statuses.get(cid)
        ref = case["reference_result"]
        row = {
            "case_id": cid,
            "group": case["group"],
            "dr": case["dr"],
            "command_mode": case["command_mode"],
            "reference_pass": ref.get("pass"),
            "reference_complete": ref.get("complete"),
            "reference_clean": clean(ref),
            "reference_doo": ref.get("doo"),
            "reference_oc": ref.get("oc"),
            "reference_result": ref.get("result"),
            "reference_source": case["reference_source"],
            "reference_index": case["reference_index"],
            "status": "missing",
            "error": None,
            "returncode": record.get("returncode") if record else None,
        }
        try:
            if record and not record.get("technical_success"):
                row["status"] = "technical_failure"
            elif record and record.get("technical_success"):
                summary = read(folder / "summary.json")
                frames = read(folder / "frames.json")
                telemetry = read(folder / "telemetry.json")
                receipt = read(folder / "evidence/completion.json")
                artifact_audit = read(folder / "artifact_audit.json")
                if (
                    not artifact_audit["passed"]
                    or artifact_audit["frame_count"] != len(frames)
                    or artifact_audit["completion_sha256"] != digest(folder / "evidence/completion.json")
                    or artifact_audit["summary_sha256"] != digest(folder / "summary.json")
                    or artifact_audit["metadata_sha256"] != digest(folder / "evidence/metadata.json")
                    or artifact_audit["manifest_sha256"] != digest(folder / "evidence/manifest.jsonl")
                ):
                    raise ValueError("Artifact audit does not bind the completed evidence")
                if not a.remote_evidence_root and verified_summary(folder / "evidence") != summary:
                    raise ValueError("Evidence completion does not bind this summary")
                video_audit = read(folder / "video_audit.json")
                if (
                    video_audit["completion_sha256"] != digest(folder / "evidence/completion.json")
                    or video_audit.get("artifact_audit_sha256") != digest(folder / "artifact_audit.json")
                    or video_audit["inference_frames"] != len(frames)
                    or video_audit["terminal_frames"] != 40
                ):
                    raise ValueError("Video audit does not bind this completed recording")
                for video in ("debug", "research"):
                    audit = video_audit["videos"][video]
                    if (
                        audit["sha256"] != digest(folder / f"{video}.mp4")
                        or audit["fps"] != 20
                        or audit["decoded_frames"] != len(frames) + 40
                    ):
                        raise ValueError(f"Incomplete or changed {video} video")
                if (
                    summary["case"] != case
                    or summary["manifest_sha256"] != manifest["manifest_sha256"]
                    or summary["protocol_sha256"] != protocol_sha
                    or summary["checkpoint_sha256"] != manifest["checkpoint_sha256"]
                    or not summary["frozen_unchanged"]
                    or summary.get("pilot_only")
                    or len(frames) != summary["recorded_frames"]
                ):
                    raise ValueError("case, provenance, frozen state, or frame count mismatch")
                if (
                    receipt.get("version") != 1
                    or receipt.get("frame_count") != len(frames)
                    or receipt.get("summary_sha256") != digest(folder / "summary.json")
                    or not isinstance(telemetry, list)
                    or not frames
                ):
                    raise ValueError("completion receipt, telemetry, or frames mismatch")
                row.update(
                    status="complete",
                    isaac_pass=summary["pass"],
                    isaac_complete=summary["complete"],
                    isaac_clean=clean(summary),
                    isaac_result=summary["result"],
                    isaac_doo=summary["doo"],
                    isaac_oc=summary["oc"],
                    physical_cone_hits=summary["physical_cone_hits"],
                    recorded_frames=len(frames),
                    wall_seconds=summary["wall_seconds"],
                    startup_seconds=summary["startup_seconds"],
                    seconds_simulated=summary["seconds_simulated"],
                    progress=summary.get("progress"),
                    runtime=summary.get("runtime"),
                    state_sha256=summary.get("state_sha256"),
                    physics_sha256=summary.get("physics_sha256"),
                )
                for key in TIMINGS:
                    vals = [f[key] for f in frames]
                    row[key + "_sum"] = sum(vals)
                all_frames.extend(frames)
                all_telemetry.append((cid, telemetry))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError) as exc:
            row["status"], row["error"] = "invalid_or_incomplete", str(exc)
        for video in ("debug.mp4", "research.mp4"):
            path = folder / video
            row[video.replace(".", "_")] = str(path.resolve()) if path.is_file() else None
        rows.append(row)
    complete = len(statuses) == 41 and status.get("complete") is True and all(r["status"] == "complete" for r in rows)
    if not complete and not a.partial:
        raise SystemExit("Suite is incomplete or invalid; use --partial for monitoring output")
    a.out.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r["status"] == "complete"]
    groups = {}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        observed = [r for r in subset if r["status"] == "complete"]
        groups[group] = {
            "expected": len(subset),
            "observed": len(observed),
            "reference_pass": sum(r["reference_pass"] is True for r in subset),
            "reference_clean": sum(r["reference_clean"] for r in subset),
            "isaac_pass": sum(r["isaac_pass"] is True for r in observed),
            "isaac_clean": sum(r["isaac_clean"] for r in observed),
        }
    timing = (
        {
            k: {
                "seconds": sum(f[k] for f in all_frames),
                "mean_ms_per_frame": 1000 * sum(f[k] for f in all_frames) / len(all_frames),
                "distribution_ms": stats([1000 * f[k] for f in all_frames]),
            }
            for k in TIMINGS
        }
        if all_frames
        else {}
    )
    telemetry_stats = {
        key: stats([sample.get(key) for _, samples in all_telemetry for sample in samples]) for key in TELEMETRY
    }
    run_wall = sum(r["wall_seconds"] for r in valid)
    sim_seconds = sum(r["seconds_simulated"] for r in valid)
    payload = {
        "remote_evidence_root": a.remote_evidence_root,
        "complete": complete,
        "counts": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "frame_count": len(all_frames),
        "frame_timing": timing,
        "telemetry": telemetry_stats,
        "total_simulated_seconds": sim_seconds,
        "real_time_factor": sim_seconds / run_wall if run_wall > 0 else None,
        "total_run_wall_seconds": sum(r["wall_seconds"] for r in valid),
        "total_startup_seconds": sum(r["startup_seconds"] for r in valid),
        "suite_status_elapsed_wall_seconds": status.get("elapsed_wall_seconds"),
        "provenance": {
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": digest(a.manifest),
            "protocol_sha256": protocol_sha,
            "reference_sha256": manifest["reference_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_hashes": manifest["source_hashes"],
            "backend_differences": manifest["backend_differences"],
            "protocol": protocol,
        },
        "cases": rows,
    }
    (a.out / "parity.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (a.out / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    if valid:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs = list(range(4))
        width = 0.2
        for j, (key, label) in enumerate(
            (
                ("reference_pass", "Custom pass"),
                ("reference_clean", "Custom clean"),
                ("isaac_pass", "Isaac pass"),
                ("isaac_clean", "Isaac clean"),
            )
        ):
            ax.bar([x + (j - 1.5) * width for x in xs], [groups[g][key] for g in GROUPS], width, label=label)
        ax.set_xticks(xs, [f"{g}\nIsaac {groups[g]['observed']}/{groups[g]['expected']}" for g in GROUPS])
        ax.set_ylabel("Cases")
        ax.legend(ncol=2)
        ax.set_title("Pass and clean counts; incomplete groups have fewer Isaac observations")
        fig.tight_layout()
        fig.savefig(a.out / "outcomes.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.32), 5))
        x = range(len(valid))
        ax.bar([i - 0.25 for i in x], [r["reference_doo"] for r in valid], 0.25, label="Custom legacy DOO")
        ax.bar(x, [r["isaac_doo"] for r in valid], 0.25, label="Isaac legacy DOO")
        ax.bar([i + 0.25 for i in x], [r["physical_cone_hits"] for r in valid], 0.25, label="Isaac PhysX cone hits")
        ax.set_xticks(list(x), [r["case_id"] for r in valid], rotation=90, fontsize=6)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.out / "paired_contacts.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = list(timing)
        ax.bar(labels, [timing[k]["mean_ms_per_frame"] for k in labels])
        ax.set_ylabel("ms per recorded frame")
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("Frame-weighted timing; step wall includes components")
        fig.tight_layout()
        fig.savefig(a.out / "frame_timing.png", dpi=150)
        plt.close(fig)
        if any(t for _, t in all_telemetry):
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for _, samples in all_telemetry:
                for ax, key in zip(axes, ("gpu_pct", "power_w"), strict=True):
                    points = [
                        (s["wall_s"], s[key]) for s in samples if s.get(key) is not None and s.get("wall_s") is not None
                    ]
                    if points:
                        ax.plot(*zip(*points, strict=True), alpha=0.35, linewidth=0.7)
            axes[0].set_ylabel("GPU utilization (%)")
            axes[1].set_ylabel("Power (W)")
            axes[1].set_xlabel("Seconds since case run start")
            axes[0].set_title("Per-case GPU telemetry overlaid; startup excluded")
            fig.tight_layout()
            fig.savefig(a.out / "gpu_telemetry.png", dpi=150)
            plt.close(fig)
    lines = [
        "# Frozen Isaac parity report",
        "",
        "**" + ("COMPLETE" if complete else "INCOMPLETE MONITORING OUTPUT") + "**",
        "",
        "Historical custom results are compared with new Isaac runs using the frozen case manifest and checkpoint. "
        "Rendering, physics, scene response, platform and trajectories differ, and historical initialization was "
        "reconstructed. This is not a controlled physics-only comparison.",
        "",
        "Clean means pass and zero legacy DOO and OC. Pass and complete do not imply clean. "
        "PhysX cone hits are a separate contact measure, not legacy DOO.",
        "",
        "| Group | Cases observed / expected | Custom pass | Custom clean | Isaac pass | Isaac clean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g, v in groups.items():
        lines.append(
            f"| {g} | {v['observed']}/{v['expected']} | {v['reference_pass']}/{v['expected']} | "
            f"{v['reference_clean']}/{v['expected']} | {v['isaac_pass']}/{v['observed']} | "
            f"{v['isaac_clean']}/{v['observed']} |"
        )
    if valid:
        lines.extend(["", "| Metric | Samples | Median | P95 |", "|---|---:|---:|---:|"])
        for key, value in telemetry_stats.items():
            if value:
                lines.append(f"| {key} | {value['count']} | {value['median']:.2f} | {value['p95']:.2f} |")
        lines.extend(["", "| Frame timing | Frames | Median ms | P95 ms |", "|---|---:|---:|---:|"])
        for key, value in timing.items():
            distribution = value["distribution_ms"]
            lines.append(
                f"| {key} | {distribution['count']} | {distribution['median']:.2f} | {distribution['p95']:.2f} |"
            )
        lines.append(
            f"\nReal-time factor: {payload['real_time_factor']:.3f} simulated seconds per run wall second."
            if payload["real_time_factor"] is not None
            else "\nReal-time factor unavailable."
        )
    lines += [
        "",
        f"Recorded frames: {len(all_frames)}. Run wall: {run_wall:.1f} s. "
        f"Startup: {payload['total_startup_seconds']:.1f} s. "
        "Suite elapsed includes process and orchestration overhead.",
        "Frame timing is weighted by recorded frames. Step wall includes the measured components, "
        "so their bars should not be added to it. Telemetry starts at run start and excludes startup. "
        "Video export is offline. P95 uses linear interpolation between sorted samples.",
        "",
        f"Checkpoint SHA-256: `{manifest['checkpoint_sha256']}`. "
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`. Protocol file SHA-256: `{protocol_sha}`. "
        "Full provenance, protocol, and source hashes are in parity.json.",
        "",
        "| Case | Status | Custom pass / clean / DOO / OC | Isaac pass / clean / DOO / OC | PhysX hits | Videos |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        videos = (
            " ".join(
                f"[{label}]({Path(r[key])})"
                for key, label in [("debug_mp4", "debug"), ("research_mp4", "research")]
                if r.get(key)
            )
            or "unavailable"
        )
        lines.append(
            f"| {r['case_id']} | {r['status']} | {val(r, 'reference_pass')} / {val(r, 'reference_clean')} / "
            f"{val(r, 'reference_doo')} / {val(r, 'reference_oc')} | {val(r, 'isaac_pass')} / "
            f"{val(r, 'isaac_clean')} / {val(r, 'isaac_doo')} / {val(r, 'isaac_oc')} | "
            f"{val(r, 'physical_cone_hits')} | {videos} |"
        )
    lines += [
        "",
        "Graphs: outcomes.png, paired_contacts.png, frame_timing.png and, when telemetry exists, "
        "gpu_telemetry.png. Technical errors and return codes are retained in cases.csv and parity.json.",
    ]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{'Complete' if complete else 'Incomplete'} report: {a.out / 'report.md'} ({len(valid)}/41 cases)")


if __name__ == "__main__":
    main()
