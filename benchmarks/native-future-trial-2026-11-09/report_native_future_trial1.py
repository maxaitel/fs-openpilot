#!/usr/bin/env python3
"""Generate the completed native-future trial report from bound local receipts."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib

from fsop.train.action_target import NATIVE_FUTURE_CONTRACT

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

STATS = ("mean", "p95", "max")
ACTION = ("plan_l1", "steer", "longitudinal")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def series(baseline, rows):
    return [(0, baseline)] + [(r["step"], r["validation"]) for r in rows if "validation" in r]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.training, args.out
    out.mkdir(parents=True, exist_ok=True)
    paths = [root / x for x in ("run.json", "history.jsonl", "baseline_validation.json", "selection_protocol.json")]
    run, baseline, protocol = load(paths[0]), load(paths[2]), load(paths[3])
    rows = [json.loads(line) for line in paths[1].read_text().splitlines() if line.strip()]
    protocol_sha = sha(paths[3])
    if [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("history is incomplete or not consecutive")
    if any(r.get("selection_protocol_sha256") != protocol_sha for r in rows):
        raise ValueError("history does not bind the frozen protocol")
    if protocol.get("schema_version") != 2 or protocol.get("action_target_contract") != NATIVE_FUTURE_CONTRACT:
        raise ValueError("requires native_future_v1 schema-2 protocol")
    records = [run, baseline, *rows, *(r["validation"] for r in rows if "validation" in r)]
    if any(record.get("action_target_contract") != NATIVE_FUTURE_CONTRACT for record in records):
        raise ValueError("action target contract changed within the run")
    bounds = protocol["bounds"]
    data = series(baseline, rows)
    names = list(bounds["coherence"])
    cases = list(bounds["episode"])
    steps = [step for step, _ in data]
    final_step, final = data[-1]
    if len(names) != 16:
        raise ValueError("requires all sixteen frozen coherence metrics")
    for _, validation in data:
        for group in [validation["aggregate"], *validation["episodes"]]:
            if any(not np.isfinite(group[key]) for key in ("teacher_steer", "teacher_longitudinal")):
                raise ValueError("missing or nonfinite teacher diagnostics")

    def episode_map(val):
        return {Path(ep["path"]).name: ep for ep in val["episodes"]}

    if any(set(episode_map(val)) != set(cases) for _, val in data):
        raise ValueError("validation episode inventory differs from protocol")
    complete = len(rows) == protocol["expected_steps"] and set(steps) == {0, 1, 100, 200, 300, 400, 500}
    if not complete:
        raise ValueError("requires completed 500-update schedule")
    receipt = load(args.selection)
    if not complete or receipt.get("protocol") != protocol:
        raise ValueError("selection receipt does not bind the complete run protocol")
    for path in paths:
        suffix = "/" + root.name + "/" + path.name
        matches = [value for name, value in receipt["input_sha256"].items() if name.endswith(suffix)]
        if matches != [sha(path)]:
            raise ValueError(f"selection receipt does not bind {path.name}")

    if receipt.get("action_target_contract") != NATIVE_FUTURE_CONTRACT:
        raise ValueError("selection has a different action target contract")

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.22})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for key in ("loss", "plan_nll", "plan_l1", "steer", "longitudinal"):
        ax[0, 0].plot([r["step"] for r in rows], [r[key] for r in rows], label=key, lw=0.8)
    ax[0, 0].set(title="Training batch metrics (native-future action targets)", xlabel="Update")
    ax[0, 0].legend(ncol=2)
    for key in ACTION:
        ax[0, 1].plot(steps, [v["aggregate"][key] / bounds["aggregate"][key] for _, v in data], marker="o", label=key)
    ax[0, 1].axhline(1, color="red", ls="--")
    ax[0, 1].set(title="Native-future validation / frozen bound", xlabel="Update", ylabel="Ratio (≤ 1 passes)")
    ax[0, 1].legend()
    for key in ("vision", "policy"):
        ax[1, 0].plot([r["step"] for r in rows], [r["grad_norms"][key] for r in rows], label=key, lw=0.8)
    ax[1, 0].set(title="Gradient norm", xlabel="Update")
    ax[1, 0].legend()
    ax[1, 1].plot([r["step"] for r in rows], [r["step_seconds"] for r in rows], lw=0.8)
    ax[1, 1].set(title="Update time", xlabel="Update", ylabel="seconds")
    fig.tight_layout()
    save(fig, out / "training.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for axis, key in zip(axes, ("steer", "longitudinal"), strict=False):
        axis.plot(steps, [v["aggregate"][key] for _, v in data], marker="o", label="Native-future target (optimized)")
        axis.plot(
            steps,
            [v["aggregate"]["teacher_" + key] for _, v in data],
            marker="o",
            label="Recorded teacher command (diagnostic only)",
        )
        axis.set(title=key + " SmoothL1 against two distinct targets", xlabel="Update", ylabel="SmoothL1")
        axis.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "action-targets.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for axis, key in zip(axes, ACTION, strict=False):
        for case in cases:
            axis.plot(
                steps,
                [episode_map(v)[case][key] / bounds["episode"][case][key] for _, v in data],
                marker=".",
                label=case.replace("val-", ""),
            )
        axis.axhline(1, color="red", ls="--")
        axis.set(ylabel="ratio", title=key + " by episode / frozen bound")
    axes[-1].set_xlabel("Update")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, out / "episode-actions.png")

    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    for axis, name in zip(axes.flat, names, strict=False):
        for stat in STATS:
            axis.plot(
                steps,
                [v["coherence"][name][stat] / bounds["coherence"][name][stat] for _, v in data],
                marker=".",
                label=stat,
            )
        axis.axhline(1, color="red", ls="--", lw=0.8)
        axis.set_title(name, fontsize=8)
        axis.set_xlabel("update")
        axis.set_ylabel("overall / bound")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("All 16 coherence metrics: aggregate values against frozen bounds")
    fig.tight_layout()
    save(fig, out / "coherence-aggregate.png")

    # Each cell shows the largest ratio among mean, p95 and max for an episode.
    fig, axis = plt.subplots(figsize=(14, 11))
    matrix = np.array(
        [
            [
                max(
                    episode_map(final)[case]["coherence"][name][stat] / bounds["episode_coherence"][case][name][stat]
                    for stat in STATS
                )
                for case in cases
            ]
            for name in names
        ]
    )
    image = axis.imshow(
        np.minimum(matrix, 3), aspect="auto", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=3), cmap="RdYlGn_r"
    )
    axis.set_xticks(range(len(cases)), [x.replace("val-", "") for x in cases], rotation=35, ha="right", fontsize=7)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(cases)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    axis.grid(False)
    axis.set_title(f"Step {final_step}: worst of mean / p95 / max, episode value / bound")
    fig.colorbar(image, ax=axis, label="ratio, colors capped at 3; ≤ 1 passes")
    fig.tight_layout()
    save(fig, out / "coherence-episodes.png")

    telemetry = load(args.telemetry)
    if telemetry["returncode"] != 0 or not telemetry.get("samples"):
        raise ValueError("completed successful training telemetry required")
    samples = telemetry["samples"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, key, title in zip(
        axes.flat,
        ("gpu_pct", "power_w", "host_ram_gib", "temp_c"),
        ("GPU utilization (%)", "GPU power (W)", "Host RAM in use (GiB)", "GPU temperature (°C)"),
        strict=False,
    ):
        axis.plot(
            [s["wall_s"] for s in samples if s.get(key) is not None],
            [s[key] for s in samples if s.get(key) is not None],
            lw=0.8,
        )
        axis.set(title=title, xlabel="Job wall seconds")
    fig.suptitle("Sampled hardware telemetry, including startup and validation")
    fig.tight_layout()
    save(fig, out / "hardware.png")
    esc = html.escape

    def fmt(x):
        return f"{x:.6g}"

    def table(headers, body):
        return (
            "<table><thead><tr>"
            + "".join("<th>" + esc(str(x)) + "</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join("<td>" + esc(str(x)) + "</td>" for x in row) + "</tr>" for row in body)
            + "</tbody></table>"
        )

    report = [
        (
            '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width"><title>Native-future trial training report</title><style>'
            "body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 20px;"
            "color:#17232c}img{max-width:100%;height:auto}table{border-collapse:collapse;"
            "font-size:13px}td,th{border:1px solid #bbb;padding:5px;"
            "text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e7eef2}section{margin:30px "
            "0}.warning{padding:1em;background:#fff1d6}</style></head><body>"
        )
    ]
    report += [
        "<h1>Isaac Lab native-future training diagnostics</h1>",
        (
            "<p>The goal is native Isaac training that reduces cone contacts and timeouts, "
            "with the same auditable debug videos.</p>"
        ),
        (
            '<p class="warning">This report measures supervised training and validation. It does '
            "not establish a physical driving improvement. Step 1 is recorded for diagnostics and "
            "excluded by the frozen selector. Validation is 256 samples, equal-weighted across 8 "
            "episodes. Ratios above 1 exceed a frozen bound.</p>"
        ),
    ]
    report += [
        (
            f"<p>Updates recorded: {len(rows)} / {protocol['expected_steps']}. Latest "
            f"validation step: {final_step}. Complete schedule: {complete}. Controller: "
            "op_tangent_v1. Action targets: native_future_v1. FP32, batch 128, context 8; "
            "vision frozen; 16 recovery draws per batch, 112 ordinary draws. Initialization "
            "and recovery collection use different checkpoints. The full model contains "
            "31,154,802 parameters; 6,949,412 policy parameters are trainable. The 8,000 "
            "recovery draws resample 80 distinct anchors.</p>"
        )
    ]
    candidates = receipt["candidates"]
    selected = receipt["selected"]
    report += [
        (
            f'<p class="warning">Selector status: {esc(receipt["status"])}. Selected step: '
            f"{selected['step']}. Selected eligible: {receipt['selected_eligible']}. Trained "
            f"candidate selected: {receipt['trained_candidate']}. A retained baseline is not "
            "a trained candidate.</p>"
        )
    ]
    report += [
        table(
            ["Step", "Eligible", "Improves step zero steering", "Failed gates"],
            [(c["step"], c["eligible"], c.get("improves_step_zero"), len(c["rejection_reasons"])) for c in candidates],
        ),
        (
            '<p><a href="../lab_v4/native_future_trial1/selection_receipt.json">Full selector '
            "receipt and rejection reasons</a>.</p>"
        ),
    ]
    report += [
        (
            '<section><h2>Training and aggregate validation</h2><img src="training.png" '
            'alt="Training and aggregate validation graphs">'
        ),
        table(
            ["Step", "Loss", "Plan L1", "Native-future steering", "Native-future longitudinal"],
            [
                (step, *(fmt(val["aggregate"][key]) for key in ("loss", "plan_l1", "steer", "longitudinal")))
                for step, val in data
            ],
        ),
        (
            "<h2>Distinct action targets</h2><p>The optimized action targets are the deployed "
            "follower applied to detached native future states, measured velocity and vehicle "
            "limits before mission braking. Raw teacher commands are separate diagnostics and do "
            "not enter the new objective or selection gates. Their loss values cannot be "
            'interchanged.</p><img src="action-targets.png" alt="Native-future and raw-teacher '
            'action loss curves">'
        ),
        table(
            ["Step", "Teacher steering diagnostic", "Teacher longitudinal diagnostic"],
            [
                (step, fmt(val["aggregate"]["teacher_steer"]), fmt(val["aggregate"]["teacher_longitudinal"]))
                for step, val in data
            ],
        ),
        (
            '<h2>Per-episode action and plan bounds</h2><img src="episode-actions.png" '
            'alt="Per-episode validation metric to bound ratios">'
        ),
    ]
    action_rows = []
    for scope, actuals, limits in [("aggregate", final["aggregate"], bounds["aggregate"])] + [
        (case, episode_map(final)[case], bounds["episode"][case]) for case in cases
    ]:
        for key in ACTION:
            value, limit = actuals[key], limits[key]
            action_rows.append((scope, key, fmt(value), fmt(limit), "pass" if value <= limit else "FAIL"))
    report += [
        (
            f"<p>Actuals and limits at final scheduled step {final_step}. This is not "
            "necessarily the selected checkpoint.</p>"
        ),
        table(["Scope", "Metric", "Actual", "Bound", "Gate"], action_rows),
        "</section>",
        (
            "<section><h2>Sixteen trajectory coherence measures</h2><img "
            'src="coherence-aggregate.png" alt="Aggregate coherence ratios"><img '
            'src="coherence-episodes.png" alt="Per-episode maximum coherence ratios"><p>The table '
            "gives actual values and frozen limits at the latest validation step. Units are "
            "encoded in metric names.</p>"
        ),
    ]
    detail = []
    for scope, actuals, limits in [("aggregate", final["coherence"], bounds["coherence"])] + [
        (case, episode_map(final)[case]["coherence"], bounds["episode_coherence"][case]) for case in cases
    ]:
        for name in names:
            for stat in STATS:
                value, limit = actuals[name][stat], limits[name][stat]
                detail.append((scope, name, stat, fmt(value), fmt(limit), "pass" if value <= limit else "FAIL"))
    report += [table(["Scope", "Metric", "Statistic", "Actual", "Bound", "Gate"], detail), "</section>"]
    report += ["<section><h2>Hardware</h2>"]
    gpu = [s["gpu_pct"] for s in samples if s.get("gpu_pct") is not None]
    power = [s["power_w"] for s in samples if s.get("power_w") is not None]
    ram = [s["host_ram_gib"] for s in samples if s.get("host_ram_gib") is not None]
    report += [
        (
            f"<p>Job wall time {telemetry['wall_seconds']:.2f} s. Process exit code "
            f"{telemetry['returncode']}. {len(samples)} samples. Whole-device sampled median "
            f"GPU utilization {np.median(gpu):.1f}% and power {np.median(power):.2f} W. Peak "
            f"sampled host RAM in use {max(ram):.2f} GiB. This is one measured job with no "
            "repeatability estimate. Job time includes setup and validation, and excludes "
            "data collection, physical evaluation and video export. GPU utilization is a "
            "busy-time counter, not achieved peak FLOPS. Host RAM is unified system memory, "
            "not dedicated VRAM. These values do not establish a speedup over a different "
            'run.</p><img src="hardware.png" alt="Hardware telemetry graphs">'
        )
    ]
    report += [
        "</section><section><h2>Input receipts</h2>",
        table(
            ["File", "SHA-256"],
            [(str(p), sha(p)) for p in paths + [args.telemetry] + [args.selection] + [Path(__file__)]],
        ),
        "</section></body></html>",
    ]
    (out / "index.html").write_text("\n".join(report))
    receipt_paths = paths + [args.telemetry, args.selection, Path(__file__)]
    binding = {
        "version": 1,
        "kind": "native_future_training_report",
        "updates": len(rows),
        "action_target_contract": NATIVE_FUTURE_CONTRACT,
        "selector_status": receipt["status"],
        "input_sha256": {str(p): sha(p) for p in receipt_paths},
        "output_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.suffix in (".html", ".png")},
    }
    (out / "report_receipt.json").write_text(json.dumps(binding, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "updates": len(rows),
                "validation_steps": steps,
                "complete": complete,
                "selector_status": receipt["status"],
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
