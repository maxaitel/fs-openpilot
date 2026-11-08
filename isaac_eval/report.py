# ruff: noqa: E501
# Long lines below are HTML/CSS templates.
"""Build graphs, an all-case video overview, and a local report from completed runs."""

import argparse
import html
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("isaac_eval/runs/suite_v2"))
p.add_argument("--skip-video", action="store_true")
args = p.parse_args()
root = args.root.resolve()
protocol = json.loads((root / "protocol.json").read_text())
cases = []
for case in protocol["cases"]:
    name = f"{case['discipline']}_{case['seed']}"
    directory = root / name
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["frozen_unchanged"] and summary["checkpoint_sha256"] == protocol["checkpoint_sha256"]
    cases.append(
        {
            "name": name,
            "dir": directory,
            "summary": summary,
            "frames": json.loads((directory / "frames.json").read_text()),
            "telemetry": json.loads((directory / "telemetry.json").read_text()),
            "track": json.loads((directory / "track.json").read_text()),
        }
    )

plt.rcParams.update(
    {
        "figure.facecolor": "#111827",
        "axes.facecolor": "#172033",
        "savefig.facecolor": "#111827",
        "text.color": "#e5e7eb",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "axes.edgecolor": "#64748b",
        "font.size": 10,
        "axes.titleweight": "bold",
        "grid.alpha": 0.15,
    }
)
labels = [
    f"{'Track' if c['summary']['discipline'] == 'trackdrive' else 'Skid'} {c['summary']['seed'] - 91000 + 1}"
    for c in cases
]
colors = ["#38bdf8" if c["summary"]["discipline"] == "trackdrive" else "#c084fc" for c in cases]
fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
for i, d in enumerate(["trackdrive", "skidpad"]):
    subset = [c for c in cases if c["summary"]["discipline"] == d]
    passed = sum(c["summary"]["pass"] for c in subset)
    axs[0, 0].bar(i, passed / len(subset) * 100, color=["#38bdf8", "#c084fc"][i], width=0.5)
    axs[0, 0].text(i, passed / len(subset) * 100 + 3, f"{passed}/{len(subset)}", ha="center")
axs[0, 0].set(
    xticks=[0, 1],
    xticklabels=["Cone track", "Skidpad"],
    ylim=(0, 115),
    ylabel="Completion + safe-stop pass rate (%)",
    title="Frozen model — five cases per discipline",
)
hits = [c["summary"]["physical_cone_hits"] for c in cases]
axs[0, 1].bar(labels, hits, color=colors)
axs[0, 1].set(
    ylabel="Unique PhysX cone contacts", ylim=(0, max(max(hits) + 1, 2)), title="Every scored run, including failures"
)
axs[0, 1].tick_params(axis="x", rotation=40)
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
for ax, d in zip(axs[1], ["trackdrive", "skidpad"], strict=True):
    subset = [c for c in cases if c["summary"]["discipline"] == d]
    track = subset[0]["track"]
    xy = np.array([[v["x"], v["y"]] for v in track["cones"]])
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color="#94a3b8", alpha=0.65, label="Cones")
    for c in subset:
        frames = c["frames"]
        x = [r["x"] for r in frames]
        y = [r["y"] for r in frames]
        ax.plot(x, y, lw=1.1, label=f"Seed {c['summary']['seed']}")
        ax.scatter(x[-1], y[-1], s=28, marker="o" if c["summary"]["pass"] else "x")
    hit_ids = sorted(
        {int(event["cone"].rsplit("/c", 1)[1]) for c in subset for event in c["summary"]["contact_events"]}
    )
    if hit_ids:
        ax.scatter(
            xy[hit_ids, 0], xy[hit_ids, 1], s=42, marker="x", color="#fb7185", label="Contacted cones", zorder=10
        )
    ax.set_aspect("equal")
    ax.set(
        title="Cone-track trajectories" if d == "trackdrive" else "Skidpad trajectories",
        xlabel="World x (m)",
        ylabel="World y (m)",
    )
    ax.legend(fontsize=7, loc="best")
fig.savefig(root / "outcomes.png", dpi=160)
plt.close(fig)

keys = ["render_s", "inference_s", "physics_s", "video_s"]
names = ["RTX capture", "Model + controller", "PhysX", "Video overlay/encode"]
palette = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24"]
fig, axs = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
bottom = np.zeros(len(cases))
for key, name, color in zip(keys, names, palette, strict=True):
    values = np.array([c["summary"][key]["mean"] * 1000 for c in cases])
    axs[0, 0].bar(labels, values, bottom=bottom, label=name, color=color)
    bottom += values
axs[0, 0].axhline(50, color="#fb7185", ls="--", label="20 Hz wall-clock budget")
axs[0, 0].set(title="Measured wall time per 20 Hz control step", ylabel="Milliseconds")
axs[0, 0].tick_params(axis="x", rotation=40)
axs[0, 0].legend(fontsize=8)
axs[0, 1].bar(labels, [c["summary"]["real_time_factor"] for c in cases], color=colors)
axs[0, 1].axhline(1, color="#fb7185", ls="--")
axs[0, 1].set(title="Simulation speed with all three cameras", ylabel="Simulated seconds / wall second")
axs[0, 1].tick_params(axis="x", rotation=40)
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
offset = 0.0
telemetry = []
for c in cases:
    t = c["telemetry"]
    x = np.array([r["wall_s"] for r in t]) + offset
    axs[1, 0].plot(x, [r["gpu_pct"] for r in t], color="#38bdf8", lw=1)
    axs[1, 1].plot(x, [r["power_w"] for r in t], color="#fbbf24", lw=1)
    offset += c["summary"]["wall_seconds"]
    telemetry.extend(t)
axs[1, 0].set(
    title="Spark GPU utilization during driving",
    xlabel="Cumulative timed wall seconds (startup excluded)",
    ylabel="GPU busy (%)",
    ylim=(0, 105),
)
axs[1, 1].set(
    title="NVML-reported GPU power during driving",
    xlabel="Cumulative timed wall seconds (startup excluded)",
    ylabel="Power (W)",
)
for ax in axs.flat:
    ax.grid(axis="y")
fig.savefig(root / "performance.png", dpi=160)
plt.close(fig)

all_frames = [r for c in cases for r in c["frames"]]
aggregate = {
    "checkpoint_sha256": protocol["checkpoint_sha256"],
    "cases": len(cases),
    "wall_seconds": sum(c["summary"]["wall_seconds"] for c in cases),
    "startup_seconds": sum(c["summary"]["startup_seconds"] for c in cases),
    "sim_seconds": sum(c["summary"]["seconds_simulated"] for c in cases),
    "physical_cone_hits": sum(hits),
    "frozen_all": True,
    "disciplines": {},
}
aggregate["real_time_factor"] = aggregate["sim_seconds"] / aggregate["wall_seconds"]
aggregate["wall_hz"] = len(all_frames) / aggregate["wall_seconds"]
for d in ["trackdrive", "skidpad"]:
    rows = [c["summary"] for c in cases if c["summary"]["discipline"] == d]
    aggregate["disciplines"][d] = {
        "n": len(rows),
        "passed": sum(r["pass"] for r in rows),
        "route_complete": sum(r["complete"] for r in rows),
        "cone_hits": sum(r["physical_cone_hits"] for r in rows),
        "off_course_events": sum(r["oc"] for r in rows),
        "clean_passes": sum(r["pass"] and r["physical_cone_hits"] == 0 for r in rows),
        "mean_sim_seconds": float(np.mean([r["seconds_simulated"] for r in rows])),
        "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in rows])),
        "mean_cone_hits": float(np.mean([r["physical_cone_hits"] for r in rows])),
    }
for key in keys + ["step_wall_s"]:
    values = np.array([r[key] for r in all_frames])
    aggregate[key] = {"mean_ms": float(values.mean() * 1000), "p95_ms": float(np.percentile(values, 95) * 1000)}
for key in ["gpu_pct", "power_w", "temp_c", "host_ram_gib", "process_rss_gib", "cpu_pct"]:
    values = np.array([r[key] for r in telemetry])
    aggregate[key] = {"mean": float(values.mean()), "max": float(values.max())}
aggregate["torch_peak_allocated_gib"] = max(c["summary"]["torch_peak_allocated_gib"] for c in cases)
aggregate["torch_peak_reserved_gib"] = max(c["summary"]["torch_peak_reserved_gib"] for c in cases)
benchmark_path = root.parent / "model_only_benchmark.json"
benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
if benchmark:
    aggregate["model_only_benchmark"] = {k: v for k, v in benchmark.items() if k != "times_seconds"}
benchmark_text = (
    f"Model-only fixed-image benchmark without rendering: {benchmark['mean_ms']:.2f} ms mean, "
    f"{benchmark['p95_ms']:.2f} ms p95 across 500 calls after 20 warmups."
    if benchmark
    else ""
)
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

crossover_contacts = sum(
    -0.2 <= event["t"] - c["summary"]["route"]["laps"][1]["finish_time"] <= 1.6
    for c in cases
    if c["summary"]["discipline"] == "skidpad"
    for event in c["summary"]["contact_events"]
)
crossover_total = sum(c["summary"]["physical_cone_hits"] for c in cases if c["summary"]["discipline"] == "skidpad")
crossover_note = f"{crossover_contacts}/{crossover_total} skidpad contacts cluster around the right-to-left crossover (−0.2 to +1.6 seconds from the second right lap's end)."
setup_video = root.parent / "excluded_setup" / "video.mp4"
if not args.skip_video:
    overview = []
    for c in cases:
        video = c["dir"] / "video.mp4"
        if not video.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(c["dir"] / "raw.mp4"),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video),
                ],
                check=True,
            )
        duration = float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]
            )
        )
        assert abs(duration - (c["summary"]["seconds_simulated"] + 3.0)) < 0.15, (c["name"], duration)
        if duration <= 18:
            overview.append(video)
        else:
            intervals = [(0.0, 3.0), (max(duration - 12.0, 0.0), duration)]
            intervals += [
                (max(event["t"] - 1.5, 0.0), min(event["t"] + 2.5, duration))
                for event in c["summary"]["contact_events"]
            ]
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1] + 0.3:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            (c["dir"] / "overview_intervals.json").write_text(json.dumps(merged, indent=2))
            for suffix, (start, end) in enumerate(merged):
                length = end - start
                clip = c["dir"] / f"overview_{suffix}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        str(start),
                        "-i",
                        str(video),
                        "-t",
                        str(length),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        str(clip),
                    ],
                    check=True,
                )
                overview.append(clip)
    if setup_video.exists():
        overview.append(setup_video)
    listing = root / "overview_concat.txt"
    listing.write_text("".join(f"file '{path}'\n" for path in overview))
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(root / "overview.mp4"),
        ],
        check=True,
    )

rows = []
md = []
for c in cases:
    s = c["summary"]
    status = "PASS" if s["pass"] else s["result"]
    rows.append(
        f"<tr><td>{html.escape(c['name'])}</td><td>{status}</td><td>{s['physical_cone_hits']}</td><td>{s['oc']}</td><td>{s['seconds_simulated']:.2f}</td><td>{s['wall_seconds']:.1f}</td><td><a href='{c['name']}/video.mp4'>Full video</a></td></tr>"
    )
    md.append(
        f"| {c['name']} | {status} | {s['physical_cone_hits']} | {s['oc']} | {s['seconds_simulated']:.2f} | {s['wall_seconds']:.1f} |"
    )
summary = "; ".join(
    f"{d}: {r['passed']}/{r['n']} passes, {r['cone_hits']} cone hits" for d, r in aggregate["disciplines"].items()
)
page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Isaac zero-shot evaluation</title>
<style>body{{background:#0b1220;color:#e5e7eb;font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.55}}a{{color:#7dd3fc}}video,img{{width:100%;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #334155;text-align:left}}.card{{background:#172033;padding:24px;border-radius:12px;margin:24px 0}}small{{color:#94a3b8}}code{{overflow-wrap:anywhere}}</style>
<h1>Frozen model in Isaac Sim</h1><p>{summary}.</p><p>Full checkpoint, unchanged driving controller, generic FS-sized PhysX Vehicle car. Five lighting/start-pose variants on each fixed layout. No training.</p>
<div class="card"><h2>Every case in the overview</h2><video controls preload="metadata" src="overview.mp4"></video><p>Edited overview: each run's start, every recorded cone-contact moment, and its final 12 seconds/status. The clearly marked rejected setup failure is appended. Full uncut recordings are linked below.</p></div>
<table><thead><tr><th>Case</th><th>Result</th><th>Cone hits</th><th>Off-course</th><th>Sim seconds</th><th>Wall seconds</th><th>Recording</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p><strong>Remaining weakness:</strong> {crossover_note} {sum(v['clean_passes'] for v in aggregate['disciplines'].values())}/{len(cases)} runs were completely contact-free.</p><h2>Outcomes and trajectories</h2><img src="outcomes.png" alt="Completion, contacts and all trajectories">
<h2>Hardware performance</h2><p>{aggregate["wall_hz"]:.2f} control steps/s in wall time; {aggregate["real_time_factor"]:.3f}× real time. Mean GPU busy {aggregate["gpu_pct"]["mean"]:.1f}%, mean power {aggregate["power_w"]["mean"]:.1f} W. Mean inference/controller time {aggregate["inference_s"]["mean_ms"]:.1f} ms, synchronized rendering {aggregate["render_s"]["mean_ms"]:.1f} ms. {benchmark_text}</p><p>Peak unified host RAM usage: {aggregate["host_ram_gib"]["max"]:.2f} GiB; peak process RSS: {aggregate["process_rss_gib"]["max"]:.2f} GiB. Torch peak allocated memory: {aggregate["torch_peak_allocated_gib"]:.3f} GiB (excludes renderer/context allocations). Peak GPU temperature: {aggregate["temp_c"]["max"]:.0f}°C.</p><img src="performance.png" alt="Timing breakdown, simulation speed, GPU utilization and power">
<div class="card"><h2>What this result covers</h2><p>Completion requires the existing physical-route referee and safe stop. Cone hits are unique PhysX contacts. Ground-truth route commands and mission launch/finish braking are retained; this does not test autonomous lap-counting. This is a small pilot on two fixed layouts, not a broad reliability estimate or a validated real-car dynamics model.</p><p>Physics: 120 Hz. Model/control: 20 Hz of simulated time. Videos: 10 Hz. Rendering and inference share the Spark GPU. Startup and kernel warmup are excluded from per-step timings and reported separately. System RAM and GPU utilization are whole-host/device measurements; Torch allocation excludes Isaac rendering allocations.</p><p>The <a href="../excluded_setup/video.mp4">rejected rigid-wheel setup recording</a> is retained separately and excluded from these scores. Its high-speed braking and renderer configuration failed verification. The final PhysX Vehicle setup passed camera, motion, braking and contact tests before scoring.</p><p>Checkpoint SHA-256: <code>{protocol["checkpoint_sha256"]}</code>. Every episode verified unchanged weights and in-memory state.</p><p><a href="protocol.json">Frozen protocol</a> · <a href="aggregate.json">Aggregate measurements</a> · <a href="../../README.md">Integration details and reproduction</a></p></div></html>"""
(root / "report.html").write_text(page)
(root / "RESULTS.md").write_text(
    f"""# Isaac Sim zero-shot results\n\n{summary}.\n\n[Overview video](overview.mp4) · [Full report and all videos](report.html) · [Outcome graph](outcomes.png) · [Performance graph](performance.png)\n\n| Case | Result | PhysX cone hits | Off-course events | Sim seconds | Wall seconds |\n|---|---|---:|---:|---:|---:|\n"""
    + "\n".join(md)
    + f"\n\nAggregate speed: {aggregate['real_time_factor']:.3f}× real time, {aggregate['wall_hz']:.2f} steps/s. Mean GPU busy: {aggregate['gpu_pct']['mean']:.1f}%; mean power: {aggregate['power_w']['mean']:.1f} W. All checkpoint and state hashes remained unchanged.\n"
)
print(json.dumps(aggregate, indent=2))
