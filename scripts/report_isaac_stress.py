"""Build a paired exploratory report from audited episode and video receipts."""

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from isaac_eval.evidence import verified_summary
from isaac_eval.parity_cases import sha256_file
from isaac_eval.stress_experiment import CASES, VARIANTS, read_receipt, validate_plan

PAIR_KEYS = (
    "case",
    "initial_physical_state",
    "checkpoint_sha256",
    "state_sha256",
    "manifest_sha256",
    "graph_sha256",
    "source_sha256",
    "track",
    "initial_cones_xyz",
    "cone_colors",
    "expert_params",
    "vehicle_params",
    "label_dt",
    "nominal_detection_calibration",
    "intrinsics",
)
TELEMETRY_KEYS = ("gpu_pct", "cpu_pct", "host_ram_gib", "process_rss_gib", "temp_c", "power_w")
TIMING_KEYS = ("render_s", "policy_with_capture_s", "evidence_s", "physics_s", "step_wall_s")


def hardware_stats(samples):
    if not isinstance(samples, list) or any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("Telemetry must be a list of samples")
    result = {}
    for key in TELEMETRY_KEYS:
        values = [sample[key] for sample in samples if key in sample and sample[key] is not None]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in values):
            raise ValueError(f"Invalid telemetry value: {key}")
        result[key] = ({"count": len(values), "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95))} if values else None)
    return result


def build(plan_path):
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, read_receipt(plan["manifest"]))
    rows, pairs = [], []
    for case in CASES:
        pair = []
        for variant in VARIANTS:
            path = root / variant / case
            evidence = path / "evidence"
            summary = verified_summary(evidence)
            receipt = json.loads((evidence / "completion.json").read_text())
            metadata = json.loads((evidence / "metadata.json").read_text())
            audit = json.loads((path / "artifact_audit.json").read_text())
            videos = json.loads((path / "video_audit.json").read_text())
            completion = sha256_file(evidence / "completion.json")
            expected_sources = {
                **plan["source_sha256"],
                "scripts/audit_isaac_episode.py": plan["tool_sha256"]["scripts/audit_isaac_episode.py"],
            }
            if (
                summary is None
                or (evidence / receipt["summary_path"]).resolve() != path / "summary.json"
                or not audit["passed"]
                or audit["case_id"] != case
                or audit["completion_sha256"] != completion
                or audit["source_sha256"] != expected_sources
                or any(
                    audit[key] != sha256_file(file)
                    for key, file in (
                        ("summary_sha256", path / "summary.json"),
                        ("metadata_sha256", evidence / "metadata.json"),
                        ("manifest_sha256", evidence / "manifest.jsonl"),
                        ("frames_sha256", path / "frames.json"),
                        ("telemetry_sha256", path / "telemetry.json"),
                        ("realization_sha256", path / "realization.json"),
                    )
                )
                or videos["case_id"] != case
                or videos["completion_sha256"] != completion
                or videos["artifact_audit_sha256"] != sha256_file(path / "artifact_audit.json")
                or videos["producer_sha256"] != plan["tool_sha256"]["scripts/process_isaac_parity.py"]
                or videos["rrd_sha256"] != sha256_file(path / "debug.rrd")
                or summary["case"]["case_id"] != case
                or summary["variant"] != variant
                or summary["experiment_plan_sha256"] != plan["plan_sha256"]
            ):
                raise ValueError(f"Unverified episode: {path}")
            frames = json.loads((path / "frames.json").read_text())
            count = len(frames)
            if (
                count == 0
                or count != summary["recorded_frames"]
                or count != audit["frame_count"]
                or count != receipt["frame_count"]
                or count != videos["inference_frames"]
                or videos["terminal_frames"] != 40
            ):
                raise ValueError(f"Frame count mismatch: {path}")
            for name in ("debug", "research"):
                video = videos["videos"][name]
                if (
                    sha256_file(path / f"{name}.mp4") != video["sha256"]
                    or video["fps"] != 20
                    or video["decoded_frames"] != count + 40
                ):
                    raise ValueError(f"Video receipt mismatch: {path}/{name}")
            hardware = hardware_stats(json.loads((path / "telemetry.json").read_text()))
            wall, simulated = summary["wall_seconds"], summary["seconds_simulated"]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   for v in (wall, simulated)):
                raise ValueError(f"Invalid episode durations: {path}")
            timing = {key: float(np.mean([f[key] for f in frames])) * 1000 for key in TIMING_KEYS}
            row = {
                "case_id": case,
                "variant": variant,
                "pass": summary["pass"],
                "result": summary["result"],
                "physical_clean": bool(summary["pass"]) and summary["physical_cone_hits"] == 0 and summary["oc"] == 0,
                "physical_contacts": summary["physical_cone_hits"],
                "doo": summary["doo"],
                "oc": summary["oc"],
                "frames": count,
                "mean_ms": timing,
                "hardware": hardware,
                "wall_seconds": wall,
                "seconds_simulated": simulated,
                "real_time_factor": simulated / wall if wall > 0 else None,
                "telemetry_sha256": audit["telemetry_sha256"],
                "summary_sha256": audit["summary_sha256"],
                "completion_sha256": completion,
                "audit_sha256": sha256_file(path / "artifact_audit.json"),
                "video_audit_sha256": sha256_file(path / "video_audit.json"),
                "videos": {
                    name: {"url": f"{variant}/{case}/{name}.mp4", **videos["videos"][name]}
                    for name in ("debug", "research")
                },
            }
            rows.append(row)
            pair.append((summary, metadata, audit))
        left, right = pair
        if (
            any(left[1][key] != right[1][key] for key in PAIR_KEYS)
            or left[0]["physics_sha256"] != right[0]["physics_sha256"]
            or left[2]["realization_sha256"] != right[2]["realization_sha256"]
        ):
            raise ValueError(f"Pair invariant mismatch: {case}")
        pairs.append(
            {
                "case_id": case,
                "physics_sha256": left[0]["physics_sha256"],
                "realization_sha256": left[2]["realization_sha256"],
                "invariants": {key: left[1][key] for key in PAIR_KEYS},
                "metadata_sha256": [item[2]["metadata_sha256"] for item in pair],
            }
        )
    totals = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        totals[variant] = {
            "passes": sum(bool(r["pass"]) for r in selected),
            "failures": sum(not r["pass"] for r in selected),
            "terminal_results": dict(Counter(str(r["result"]) for r in selected)),
            "physical_clean": sum(r["physical_clean"] for r in selected),
            "contacts": sum(r["physical_contacts"] for r in selected),
        }
    report = {
        "scope": "Four preselected cases, eight exploratory episodes. No full-suite inference.",
        "plan_sha256": plan["plan_sha256"],
        "totals": totals,
        "pairs": pairs,
        "episodes": rows,
    }
    (root / "paired_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_gallery(root, report)
    return report


def chart(rows, key, label):
    values = [row["physical_contacts"] if key == "contacts" else row["mean_ms"]["step_wall_s"] for row in rows]
    maximum = max(values, default=0) or 1
    marks = []
    for i in range(len(rows)):
        x, y = 120 + (i // 2) * 150, 35 + (i % 2) * 20
        width = values[i] / maximum * 100
        color = "#28659b" if i % 2 == 0 else "#d16c38"
        marks.append(
            f'<rect x="{x}" y="{y}" width="{width:.2f}" height="12" fill="{color}"/>'
            f'<text x="{x}" y="{y + 11}">{values[i]:.1f}</text>'
        )
    labels = "".join(f'<text x="{120 + i * 150}" y="20">Case {i + 1}</text>' for i in range(4))
    return (f'<figure><figcaption>{html.escape(label)}. Blue matched; orange stress.</figcaption>'
            f'<svg viewBox="0 0 760 80" role="img" aria-label="{html.escape(label)}">'
            f'{labels}{"".join(marks)}</svg></figure>')


def write_gallery(root, report):
    rows = report["episodes"]
    cards = []
    for row in rows:
        cards.append(
            f"<tr><td>{html.escape(row['case_id'])}</td><td>{html.escape(row['variant'])}</td>"
            f"<td>{row['pass']}</td><td>{html.escape(str(row['result']))}</td><td>{row['physical_contacts']}</td>"
            f"<td>{row['doo']}</td><td>{row['oc']}</td><td>{row['frames']}</td><td>{row['mean_ms']['step_wall_s']:.1f}</td></tr>"
        )
    def metric(row, key):
        value = row["hardware"][key]
        return f"{value['median']:.1f} / {value['p95']:.1f} (n={value['count']})" if value else "n/a"

    hardware_rows = []
    for row in rows:
        factor = row["real_time_factor"]
        factor_text = f"{factor:.2f}" if factor is not None else "n/a"
        hardware_rows.append(
            f"<tr><td>{html.escape(row['case_id'])}</td><td>{html.escape(row['variant'])}</td>"
            + "".join(f"<td>{metric(row, key)}</td>" for key in TELEMETRY_KEYS)
            + f"<td>{row['wall_seconds']:.1f}</td><td>{row['seconds_simulated']:.1f}</td>"
            + f"<td>{factor_text}</td></tr>"
        )
    options = "".join(f'<option value="{i}">{html.escape(case)}</option>' for i, case in enumerate(CASES))
    receipts = "".join(
        f"<li>{html.escape(pair['case_id'])}: physics <code>{pair['physics_sha256']}</code>, "
        f"realization <code>{pair['realization_sha256']}</code>, "
        f"metadata <code>{', '.join(pair['metadata_sha256'])}</code></li>"
        for pair in report["pairs"]
    )
    totals = html.escape(json.dumps(report["totals"]))
    video_urls = [[r["videos"][name]["url"] for name in ("debug", "research")] for r in rows]
    page = (
        """<!doctype html><meta charset="utf-8"><title>Paired visual stress</title>
<style>body{font:16px system-ui;max-width:1500px;margin:3em auto;padding:0 1em}table{border-collapse:collapse}
td,th{border:1px solid #bbb;padding:.5em;text-align:left}code{overflow-wrap:anywhere}
svg{width:100%;max-width:950px}svg text{font:12px sans-serif}.videos{display:flex;gap:1em}
.videos section{width:50%}video{width:100%}</style>
<h1>Paired visual stress</h1><p>Four preselected cases and eight exploratory episodes.
These are not a full-suite estimate.
The visual treatment may change subsequent trajectories, so later frames are not matched inputs. People and clutter are
appearance-only. This is not a pedestrian-safety test. Rollover terminal evidence and physical contacts
have audit limitations.</p>
<p>Frozen plan <code>"""
        + report["plan_sha256"]
        + """</code>. Physical clean requires a referee pass, zero contacts, and zero off-course events.</p>
"""
        + "<p>Totals: <code>" + totals + "</code></p>"
        + chart(rows, "contacts", "Physical contacts per episode")
        + chart(rows, "timing", "Mean step wall time in milliseconds")
        + """
<h2>Compare videos</h2><label>Case <select id="case">"""
        + options
        + """</select></label>
<label>View <select id="view"><option value="0">Debug</option><option value="1">Research</option></select></label>
<div class="videos"><section><h3>Matched</h3><video id="matched" controls preload="metadata"></video></section>
<section><h3>Visual stress</h3><video id="stress" controls preload="metadata"></video></section></div>
<p>Videos include 40 terminal-card frames. Playback is independent; trajectories can diverge.</p>
<h2>All episodes</h2><table><thead><tr><th>Case</th><th>Variant</th><th>Pass</th><th>Terminal result</th>
<th>Contacts</th><th>DOO</th><th>Off course</th>
<th>Frames</th><th>Mean step ms</th></tr></thead><tbody>"""
        + "".join(cards)
        + """</tbody></table>
<h2>Hardware by episode</h2><p>One-second samples during the run, excluding startup. Values are median / p95
with sample counts. CPU and host RAM are system-wide; RSS is the evaluator process. Real-time factor is simulated
seconds per run wall second.</p><table><thead><tr><th>Case</th><th>Variant</th><th>GPU %</th><th>CPU %</th>
<th>Host RAM GiB</th><th>Process RSS GiB</th><th>GPU °C</th><th>Power W</th><th>Run s</th><th>Sim s</th>
<th>Real-time factor</th></tr></thead><tbody>"""
        + "".join(hardware_rows)
        + """</tbody></table><h2>Pair invariant receipts</h2><ul>"""
        + receipts
        + """</ul><p><a href="paired_report.json">Full results, invariant values, timing, totals
and artifact hashes</a></p>
<script>const urls = """
        + json.dumps(video_urls).replace("<", "\\u003c")
        + """; const c=document.getElementById('case'), v=document.getElementById('view');
function select(){
document.getElementById('matched').src=urls[2*Number(c.value)][Number(v.value)];
document.getElementById('stress').src=urls[2*Number(c.value)+1][Number(v.value)];}
c.onchange=select;v.onchange=select;select();</script>"""
    )
    (root / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan)["totals"]))


if __name__ == "__main__":
    main()
