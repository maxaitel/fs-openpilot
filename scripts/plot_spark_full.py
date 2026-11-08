"""Export measured training curves, hardware plots and machine-readable summaries."""

import json
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path("runs/spark_full")
FIG = OUT / "graphs"
STAGES = [
    ("spark_full_stage1", "Frozen trunk", 6000),
    ("spark_full_bootstrap", "Last trunk stage", 3000),
    ("spark_full_v2", "Full trunk", 7000),
    ("spark_full_refresh", "History refresh", 4000),
]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(training_only=False):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    metrics = [
        ("train/loss", "Training loss (logged batches)"),
        ("val/loss", "Validation loss"),
        ("val/lat_err_1s", "Validation lateral error at 1 s (m)"),
        ("val/lat_err_2s", "Validation lateral error at 2 s (m)"),
    ]
    offset = 0
    summaries = []
    all_scalars = {}
    fig2, ax2 = plt.subplots(2, 1, figsize=(13, 7), layout="constrained")
    for (name, label, steps), color in zip(STAGES, COLORS, strict=True):
        ea = EventAccumulator(str(Path("runs") / name / "tb"), size_guidance={"scalars": 0})
        ea.Reload()
        scalars = {
            tag: [{"step": p.step, "wall_time": p.wall_time, "value": p.value} for p in ea.Scalars(tag)]
            for tag in ea.Tags()["scalars"]
        }
        all_scalars[name] = scalars
        for ax, (tag, title) in zip(axes.flat, metrics, strict=True):
            pts = scalars.get(tag, [])
            ax.plot([p["step"] + offset for p in pts], [p["value"] for p in pts], color=color, lw=1.4, label=label)
            ax.set_title(title)
            ax.set_xlabel("Cumulative optimizer steps")
            ax.axvspan(offset, offset + steps, color=color, alpha=0.04)
        data = json.loads((OUT / f"{name}.json").read_text())
        vals = [json.loads(s) for s in (Path("runs") / name / "val.jsonl").read_text().splitlines()]
        log = (OUT / f"{name}.log").read_text()
        points = [(int(s), float(v)) for s, v in re.findall(r"step\s+(\d+).*?\|\s+([\d.]+) samples/s", log)]
        # Trainer's interval immediately after validation includes its time; exclude it for steady throughput.
        every = data["config"]["val_every"]
        log_every = data["config"]["log_every"]
        steady = [(s, v) for s, v in points if s > log_every and (s - log_every) % every != 0]
        ax2[0].plot([s + offset for s, v in steady], [v for s, v in steady], color=color, label=label)
        f1 = scalars.get("val/bev_f1", [])
        ax2[1].plot(
            [p["step"] + offset for p in f1], [p["value"] for p in f1], color=color, marker="o", ms=3, label=label
        )
        best = min([v for v in vals if v["step"] > 0], key=lambda v: v["loss"])
        summaries.append(
            {
                "name": name,
                "label": label,
                "steps": data["steps"],
                "batch_size": data["config"]["batch_size"],
                "fit_seconds": data["fit_seconds"],
                "effective_samples_per_second": data["steps"] * data["config"]["batch_size"] / data["fit_seconds"],
                "peak_allocated_gib": data["peak_allocated_gib"],
                "hardware_mean": data["hardware_mean"],
                "steady_samples_per_second_median": float(np.median([v for s, v in steady])),
                "best_validation": best,
            }
        )
        offset += steps
    axes[0, 0].legend(fontsize=9, ncol=2)
    fig.suptitle(
        "DGX Spark • full venue training\nLoss changes across phases include different heads / refreshed history",
        fontsize=15,
    )
    save(fig, "training-curves")
    ax2[0].set_title("Logged training throughput • excludes validation intervals")
    ax2[0].set_ylabel("Samples / second")
    ax2[1].set_title("Validation BEV cone F1")
    ax2[1].set_ylabel("F1")
    ax2[1].set_ylim(0, 1)
    for ax in ax2:
        ax.set_xlabel("Cumulative optimizer steps")
        ax.legend(fontsize=9, ncol=4)
    save(fig2, "throughput-and-cones")
    rows = [json.loads(s) for s in (OUT / "hardware.jsonl").read_text().splitlines() if s.endswith("}")]
    start = rows[0]["time"]
    x = np.array([(r["time"] - start) / 60 for r in rows])
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), layout="constrained", sharex=True)
    # Five-second means keep the graph readable while raw 1 Hz data remains available.
    for ax, keys, title, ylabel in [
        (axes[0], ["gpu_percent", "cpu_percent"], "Hardware utilization", "Busy (%)"),
        (axes[1], ["system_used_gib"], "Unified system memory in use", "GiB"),
        (axes[2], ["power_w"], "GPU power", "W"),
        (axes[3], ["temperature_c"], "GPU temperature", "°C"),
    ]:
        for key in keys:
            y = np.array([r[key] for r in rows])
            n = len(y) // 5 * 5
            ax.plot(x[:n].reshape(-1, 5).mean(1), y[:n].reshape(-1, 5).mean(1), lw=1, label=key.replace("_", " "))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left")
        for (name, _label, _), color in zip(STAGES, COLORS, strict=True):
            t = [(r["time"] - start) / 60 for r in rows if r["phase"] == name]
            if t:
                ax.axvspan(min(t), max(t), color=color, alpha=0.12)
                if ax is axes[0]:
                    ax.text((min(t) + max(t)) / 2, 104, _label, ha="center", fontsize=8)
    axes[0].set_ylim(0, 112)
    axes[-1].set_xlabel("Minutes since hardware monitoring started")
    fig.suptitle("DGX Spark • measured hardware use (whole system memory; GPU telemetry)", fontsize=14)
    save(fig, "hardware-utilization")
    (OUT / "training-scalars.json").write_text(json.dumps(all_scalars, indent=2))
    (OUT / "training-summary.json").write_text(json.dumps(summaries, indent=2))
    if training_only:
        return
    evaluation = OUT / "evaluation"
    models = ["untouched", "short", "full"]
    results = {
        name: json.loads((evaluation / f"{name}_open_loop.json").read_text())["summary"]["all"] for name in models
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["lat_1.0s", "lat_2.0s", "ade_4s"],
        ["Lateral error at 1 s", "Lateral error at 2 s", "Displacement error through 4 s"],
        strict=True,
    ):
        values = [results[m][key] for m in models]
        bars = ax.bar(["Pretrained", "Short test", "Full recipe"], values, color=["#9EA7AD", "#F58518", "#54A24B"])
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_title(title)
        ax.set_ylabel("Meters • lower is better")
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle("Same held-out dataset • freshly recomputed model history", fontsize=14)
    save(fig, "held-out-comparison")
    closed = {
        name: json.loads((evaluation / file).read_text())["aggregate"]
        for name, file in [("Short test", "short_closed_loop.json"), ("Full recipe", "closed_loop.json")]
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, key, title, factor in [
        (axes[0], "pass_rate", "Laps finished without disqualification", 100),
        (axes[1], "doo_per_run", "Mean cone hits per run", 1),
    ]:
        for i, (name, results) in enumerate(closed.items()):
            values = [results[f"trackdrive/{dr}"][key] * factor for dr in ["light", "heavy"]]
            bars = ax.bar(np.arange(2) + (i - 0.5) * 0.36, values, 0.36, label=name, color=["#F58518", "#54A24B"][i])
            ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_xticks([0, 1], ["Light randomization", "Heavy randomization"])
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("Percent • higher is better")
    axes[0].set_ylim(0, 125)
    axes[1].set_ylabel("Cones • lower is better; read alongside completion")
    axes[1].set_ylim(0, max(axes[1].get_ylim()[1] * 1.2, 1))
    fig.suptitle("Same eight held-out tracks • four regular + four tight • all venue scenes", fontsize=14)
    save(fig, "closed-loop-comparison")
    benches = [
        json.loads((OUT / f"full_bench_{name}.json").read_text())
        for name in ["baseline", "workers4", "workers8", "workers12"]
    ]
    labels = ["Baseline / 12 workers", "Optimized / 4 workers", "Optimized / 8 workers", "Optimized / 12 workers"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    bars = axes[0].barh(
        labels, [r["samples_per_second"] for r in benches], color=["#9EA7AD", "#F58518", "#54A24B", "#4C78A8"]
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=4)
    axes[0].set_xlim(0, max(r["samples_per_second"] for r in benches) * 1.18)
    axes[0].set_xlabel("Samples per second • higher is better")
    axes[0].set_title("Measured training throughput")
    bars = axes[1].barh(
        labels,
        [100 * r["data_wait_seconds"] / r["seconds"] for r in benches],
        color=["#9EA7AD", "#F58518", "#54A24B", "#4C78A8"],
    )
    axes[1].bar_label(bars, fmt="%.2f%%", padding=4)
    axes[1].set_xlim(0, 25)
    axes[1].set_xlabel("Percent of measured time • lower is better")
    axes[1].set_title("Time waiting for the next batch")
    fig.suptitle("Same starting weights and dataset • full network • batch 64 • 200 measured steps", fontsize=13)
    save(fig, "performance-comparison")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-only", action="store_true")
    main(parser.parse_args().training_only)
