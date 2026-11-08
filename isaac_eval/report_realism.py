# ruff: noqa: E501
"""Build an appearance-only paired evaluation report from completed run artifacts."""

import argparse
import base64
import hashlib
import html
import json
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def stats(values, scale=1):
    values = [float(v) * scale for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def format_stats(value, digits=1):
    if value is None:
        return "unavailable"
    return f"{value['median']:.{digits}f} ({value['min']:.{digits}f}–{value['max']:.{digits}f})"


def load_cases(root, protocol):
    cases = []
    protocol_bytes = (root / "protocol.json").read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    for spec in protocol["cases"]:
        for variant in protocol["variants"]:
            name = f"{variant}_{spec['discipline']}_{spec['seed']}"
            directory = root / name
            summary = read(directory / "summary.json")
            if (directory / "protocol.json").read_bytes() != protocol_bytes or summary[
                "protocol_sha256"
            ] != protocol_sha256:
                raise ValueError(f"Protocol integrity failed for {name}")
            if "render_profile" in protocol and summary.get("render_profile") != protocol["render_profile"]:
                raise ValueError(f"Render profile mismatch for {name}")
            if (summary["scene"], summary["discipline"], summary["seed"]) != (
                variant,
                spec["discipline"],
                spec["seed"],
            ):
                raise ValueError(f"Run does not match protocol: {name}")
            cases.append(
                {
                    "name": name,
                    "directory": directory,
                    "summary": summary,
                    "frames": read(directory / "frames.json"),
                    "telemetry": read(directory / "telemetry.json"),
                    "track": read(directory / "track.json"),
                }
            )
    return cases


def aggregate_cases(cases, protocol):
    groups = {}
    for variant in protocol["variants"]:
        groups[variant] = {}
        for discipline in sorted({c["summary"]["discipline"] for c in cases}):
            rows = [
                c["summary"]
                for c in cases
                if c["summary"]["scene"] == variant and c["summary"]["discipline"] == discipline
            ]
            contacts = sum(r["physical_cone_hits"] for r in rows)
            groups[variant][discipline] = {
                "n": len(rows),
                "completed": sum(bool(r["complete"]) for r in rows),
                "passed": sum(bool(r["pass"]) for r in rows),
                "clean_completions": sum(bool(r["complete"]) and r["physical_cone_hits"] == 0 for r in rows),
                "contact_total": contacts,
                "contact_mean": contacts / len(rows),
                "off_course_events": sum(r["oc"] for r in rows),
                "hardware": {
                    k: stats(
                        [
                            t.get(k)
                            for c in cases
                            if c["summary"]["scene"] == variant and c["summary"]["discipline"] == discipline
                            for t in c["telemetry"]
                        ]
                    )
                    for k in ("gpu_pct", "cpu_pct", "temp_c", "host_ram_gib", "power_w")
                },
                "performance": {
                    k: stats([r[k] for r in rows]) for k in ("real_time_factor", "control_hz_wall", "wall_seconds")
                },
                "step_median_ms": {
                    k: stats(
                        [
                            statistics.median(f[k] for f in c["frames"]) * 1000
                            for c in cases
                            if c["summary"]["scene"] == variant and c["summary"]["discipline"] == discipline
                        ]
                    )
                    for k in ("render_s", "inference_s", "physics_s")
                },
            }
    pairs = []
    for spec in protocol["cases"]:
        pair = [
            c
            for c in cases
            if c["summary"]["discipline"] == spec["discipline"] and c["summary"]["seed"] == spec["seed"]
        ]
        if len(pair) != 2 or {c["summary"]["scene"] for c in pair} != {"plain", "paddock"}:
            raise ValueError(f"Expected one plain and one paddock run for {spec}")
        summaries = [c["summary"] for c in pair]
        physics = [s["physics_sha256"] for s in summaries]
        checkpoint = [s["checkpoint_sha256"] for s in summaries]
        state = [s["state_sha256"] for s in summaries]
        checks = {
            "track": pair[0]["track"] == pair[1]["track"],
            "start offsets": all(
                summaries[0][key] == summaries[1][key] for key in ("start_lateral_offset_m", "start_yaw_offset_rad")
            ),
            "actual initial state": all(
                abs(pair[0]["frames"][0][k] - pair[1]["frames"][0][k]) < 1e-6 for k in ("x", "y", "yaw", "speed", "s")
            ),
            "physics": len(set(physics)) == 1 and bool(physics[0]),
            "checkpoint": len(set(checkpoint)) == 1 and checkpoint[0] == protocol["checkpoint_sha256"],
            "state": len(set(state)) == 1 and bool(state[0]),
            "frozen": all(s["frozen_unchanged"] for s in summaries),
        }
        failures = [name for name, ok in checks.items() if not ok]
        if failures:
            raise ValueError(f"Pair integrity failed for {spec}: {', '.join(failures)}")
        by_scene = {c["summary"]["scene"]: c for c in pair}
        plain, paddock = by_scene["plain"], by_scene["paddock"]

        def off_course_seconds(c):
            return sum(bool(f["off_course"]) for f in c["frames"]) * 0.05

        pairs.append(
            {
                "contact_delta": paddock["summary"]["physical_cone_hits"] - plain["summary"]["physical_cone_hits"],
                "completion_delta": int(paddock["summary"]["complete"]) - int(plain["summary"]["complete"]),
                "off_course_seconds_delta": off_course_seconds(paddock) - off_course_seconds(plain),
                **spec,
                "track_equal": checks["track"],
                "start_offsets_equal": checks["start offsets"],
                "actual_initial_state_equal": checks["actual initial state"],
                "physics_sha256": physics,
                "physics_equal": len(set(physics)) == 1,
                "checkpoint_sha256": checkpoint,
                "state_sha256": state,
                "checkpoint_equal_and_frozen": len(set(checkpoint)) == 1
                and checkpoint[0] == protocol["checkpoint_sha256"],
                "state_equal": len(set(state)) == 1,
                "unchanged": all(s["frozen_unchanged"] for s in summaries),
            }
        )
    details = []
    for c in cases:
        s = c["summary"]
        details.append(
            {
                "name": c["name"],
                "result": s["result"],
                "complete": s["complete"],
                "pass": s["pass"],
                "contacts": s["physical_cone_hits"],
                "off_course": s["oc"],
                "timing_ms": {
                    k: stats([r[k] for r in c["frames"]], 1000) for k in ("render_s", "inference_s", "physics_s")
                },
                "telemetry": {
                    k: stats([r.get(k) for r in c["telemetry"]])
                    for k in ("gpu_pct", "power_w", "host_ram_gib", "process_rss_gib", "cpu_pct", "temp_c")
                },
                "torch_peak_allocated_gib": s["torch_peak_allocated_gib"],
                "torch_peak_reserved_gib": s["torch_peak_reserved_gib"],
            }
        )
    return {
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "case_count": len(cases),
        "groups": groups,
        "pairs": pairs,
        "cases": details,
    }


def graphs(root, cases, aggregate):
    plt.rcParams.update(
        {
            "figure.facecolor": "#101827",
            "axes.facecolor": "#172337",
            "savefig.facecolor": "#101827",
            "text.color": "#e5edf7",
            "axes.labelcolor": "#e5edf7",
            "xtick.color": "#c3cedc",
            "ytick.color": "#c3cedc",
            "axes.edgecolor": "#64748b",
        }
    )
    preview_paths = [
        root.parent / f"preview_{variant}_track" / f"start_{camera}.png"
        for variant in ("plain", "paddock")
        for camera in ("chase", "road", "wide")
    ]
    if all(path.exists() for path in preview_paths):
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), layout="constrained")
        for index, (ax, path) in enumerate(zip(axes.flat, preview_paths, strict=True)):
            ax.imshow(plt.imread(path))
            ax.set_title(f"{('Plain', 'Realistic')[index // 3]} · {('Chase', 'Road input', 'Wide input')[index % 3]}")
            ax.axis("off")
        fig.savefig(root / "before_after.png", dpi=140)
        plt.close(fig)
    labels = [c["name"].replace("trackdrive", "track").replace("9100", "") for c in cases]
    colors = ["#60a5fa" if c["summary"]["scene"] == "plain" else "#fbbf24" for c in cases]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
    axes[0].bar(labels, [int(c["summary"]["complete"]) for c in cases], color=colors)
    axes[0].set(title="Route completion by case", ylabel="Completed", ylim=(0, 1.2), yticks=[0, 1])
    axes[1].bar(labels, [c["summary"]["physical_cone_hits"] for c in cases], color=colors)
    axes[1].set(title="Unique PhysX cone contacts by case", ylabel="Contacts")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.savefig(root / "outcomes.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), layout="constrained")
    for ax, (key, title, unit) in zip(
        axes.flat,
        [
            ("render_s", "Render", "ms"),
            ("inference_s", "Inference and controller", "ms"),
            ("physics_s", "Physics", "ms"),
            ("gpu_pct", "GPU utilization", "%"),
            ("power_w", "GPU power", "W"),
            ("host_ram_gib", "Host RAM", "GiB"),
        ],
        strict=True,
    ):
        vals = [d["timing_ms"].get(key) if key.endswith("_s") else d["telemetry"][key] for d in aggregate["cases"]]
        for i, v in enumerate(vals):
            if v:
                ax.vlines(i, v["min"], v["max"], color=colors[i], alpha=0.65)
                ax.scatter(i, v["median"], color=colors[i], s=16)
        ax.set(title=f"{title}: median and range", ylabel=unit, xticks=range(len(labels)), xticklabels=labels)
        ax.tick_params(axis="x", rotation=60, labelsize=6)
    fig.savefig(root / "performance.png", dpi=140)
    plt.close(fig)
    disciplines = sorted({c["summary"]["discipline"] for c in cases})
    fig, axes = plt.subplots(
        len(disciplines), 2, figsize=(13, 6 * len(disciplines)), squeeze=False, layout="constrained"
    )
    for i, discipline in enumerate(disciplines):
        for j, variant in enumerate(["plain", "paddock"]):
            ax = axes[i][j]
            subset = [c for c in cases if c["summary"]["discipline"] == discipline and c["summary"]["scene"] == variant]
            cones = subset[0]["track"]["cones"]
            ax.scatter([p["x"] for p in cones], [p["y"] for p in cones], s=5, c="#94a3b8", label="Cones")
            for c in subset:
                frames = c["frames"]
                ax.plot([f["x"] for f in frames], [f["y"] for f in frames], lw=1, label=str(c["summary"]["seed"]))
                if frames:
                    ax.scatter(frames[-1]["x"], frames[-1]["y"], marker="o" if c["summary"]["pass"] else "x", s=30)
            hit_ids = {
                int(event["cone"].rsplit("/c", 1)[1]) for c in subset for event in c["summary"]["contact_events"]
            }
            if hit_ids:
                ax.scatter(
                    [cones[k]["x"] for k in sorted(hit_ids)],
                    [cones[k]["y"] for k in sorted(hit_ids)],
                    marker="x",
                    s=45,
                    c="#fb7185",
                    label="Contacted cones (original positions)",
                    zorder=10,
                )
            ax.set(title=f"{variant} · {discipline}", xlabel="World x (m)", ylabel="World y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=7)
    fig.savefig(root / "trajectories.png", dpi=140)
    plt.close(fig)


def videos(root, cases):
    all_clips = []
    for c in cases:
        directory, s = c["directory"], c["summary"]
        full = directory / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(directory / "raw.mp4"),
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
                str(full),
            ],
            check=True,
        )
        duration = float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(full)]
            )
        )
        if not s["pass"]:
            intervals = [[0.0, duration]]
        else:
            windows = [(0.0, min(3.0, duration)), (max(duration - 12.0, 0.0), duration)]
            windows += [(max(0.0, event["t"] - 1.5), min(duration, event["t"] + 2.5)) for event in s["contact_events"]]
            intervals = []
            for start, end in sorted(windows):
                if start >= end:
                    continue
                if intervals and start <= intervals[-1][1] + 0.3:
                    intervals[-1][1] = max(intervals[-1][1], end)
                else:
                    intervals.append([start, end])
        (directory / "overview_intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
        c["clips"] = []
        for i, (start, end) in enumerate(intervals):
            clip = directory / f"overview_{i}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    str(full),
                    "-t",
                    str(end - start),
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
                    str(clip),
                ],
                check=True,
            )
            c["clips"].append((clip.name, start, end))
            all_clips.append(clip)
    listing = root / "overview_concat.txt"
    listing.write_text("".join("file '" + str(path.relative_to(root)).replace("'", "'\\''") + "'\n" for path in all_clips))
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


def embedded(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def report(root, protocol, cases, aggregate, skip_video):
    esc = html.escape
    group_rows = []
    for variant, groups in aggregate["groups"].items():
        for discipline, g in groups.items():
            group_rows.append(
                f"<tr><td>{esc(variant)}</td><td>{esc(discipline)}</td><td>{g['completed']}/{g['n']}</td>"
                f"<td>{g['passed']}/{g['n']}</td><td>{g['clean_completions']}/{g['n']}</td>"
                f"<td>{g['contact_total']}</td><td>{g['contact_mean']:.2f}</td></tr>"
            )
    pair_rows = []
    for p in aggregate["pairs"]:
        flags = [
            p["track_equal"],
            p["start_offsets_equal"],
            p["actual_initial_state_equal"],
            p["physics_equal"],
            p["checkpoint_equal_and_frozen"],
            p["state_equal"],
            p["unchanged"],
        ]
        pair_rows.append(
            f"<tr><td>{esc(p['discipline'])} {p['seed']}</td>"
            + "".join(f"<td>{'OK' if x else 'MISMATCH'}</td>" for x in flags)
            + "</tr>"
        )
    delta_rows = [
        f"<tr><td>{esc(p['discipline'])} {p['seed']}</td><td>{p['contact_delta']:+d}</td>"
        f"<td>{p['completion_delta']:+d}</td><td>{p['off_course_seconds_delta']:+.2f}</td></tr>"
        for p in aggregate["pairs"]
    ]
    performance_rows = []
    for variant, groups in aggregate["groups"].items():
        for discipline, g in groups.items():
            metrics = [g["performance"][k] for k in ("real_time_factor", "control_hz_wall", "wall_seconds")]
            metrics += [g["step_median_ms"][k] for k in ("render_s", "inference_s", "physics_s")]
            performance_rows.append(
                f"<tr><td>{esc(variant)}</td><td>{esc(discipline)}</td>"
                + "".join(f"<td>{format_stats(v, 2)}</td>" for v in metrics)
                + "</tr>"
            )
    rows, overview = [], []
    for c, detail in zip(cases, aggregate["cases"], strict=True):
        s = c["summary"]
        status = "PASS" if s["pass"] else str(s["result"])
        timing = detail["timing_ms"]
        telem = detail["telemetry"]
        link = f"<a href='{esc(c['name'])}/video.mp4'>Full video</a>" if not skip_video else "Video skipped"
        rows.append(
            f"<tr><td>{esc(c['name'])}</td><td>{esc(status)}</td><td>{s['physical_cone_hits']}</td>"
            f"<td>{s['oc']}</td><td>{format_stats(timing['render_s'])}</td>"
            f"<td>{format_stats(timing['inference_s'])}</td><td>{format_stats(timing['physics_s'])}</td>"
            f"<td>{format_stats(telem['gpu_pct'])}</td><td>{format_stats(telem['power_w'])}</td>"
            f"<td>{format_stats(telem['host_ram_gib'], 2)}</td><td>{link}</td></tr>"
        )
        clips = c.get("clips", [])
        elements = "".join(
            f"<figure><video controls preload='none' src='{esc(c['name'])}/{esc(name)}'></video>"
            f"<figcaption>{start:.1f}–{end:.1f} s</figcaption></figure>"
            for name, start, end in clips
        )
        overview.append(
            f"<details><summary>{esc(c['name'])} · {esc(status)} · {s['physical_cone_hits']} contacts</summary>"
            f"{elements or '<p>Video generation skipped.</p>'}</details>"
        )
    previews = []
    for discipline, dirname in [("trackdrive", "track"), ("skidpad", "skidpad")]:
        for pose in ("start", "turn"):
            for camera in ("road", "wide", "chase"):
                figures = []
                for variant in ("plain", "paddock"):
                    path = root.parent / f"preview_{variant}_{dirname}" / f"{pose}_{camera}.png"
                    if path.exists():
                        figures.append(
                            f"<figure><img src='{embedded(path)}' alt='{esc(variant)} {esc(discipline)} {pose} {camera}'>"
                            f"<figcaption>{esc(variant)} · {esc(discipline)} · {pose} {camera}</figcaption></figure>"
                        )
                if figures:
                    previews.append(
                        f"<h3>{esc(discipline)} · {pose} · {camera}</h3><div class='grid'>{''.join(figures)}</div>"
                    )
    profile = protocol.get("render_profile")
    profile_text = (
        f"Both variants use {esc(str(profile['subframes']))} render subframes. {esc(profile.get('reason', ''))}"
        if profile
        else "The protocol does not specify a render profile."
    )
    benchmark_path = root.parent / "model_only_benchmark.json"
    benchmark_html = ""
    if benchmark_path.exists():
        benchmark = read(benchmark_path)
        if benchmark["checkpoint_sha256"] != protocol["checkpoint_sha256"] or not benchmark["frozen_unchanged"]:
            raise ValueError("Model-only benchmark used a different or changed checkpoint")
        benchmark_html = (
            '<div class="card"><h2>Model-only reference</h2>'
            f'<p>{benchmark["samples"]} calls after {benchmark["warmup"]} warmup calls, with fixed lossless '
            'calibration images and no renderer running. This measures the policy and controller call, '
            'including preprocessing and synchronized GPU execution. '
            f'Mean {benchmark["mean_ms"]:.2f} ms; median {benchmark["median_ms"]:.2f} ms; '
            f'95th percentile {benchmark["p95_ms"]:.2f} ms. '
            'This fixture is a latency reference, not an autonomous driving evaluation. '
            '<a href="../model_only_benchmark.json">Raw benchmark</a>.</p></div>'
        )
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Paired appearance evaluation</title><style>
body{{background:#101827;color:#e5edf7;font:15px system-ui;max-width:1500px;margin:36px auto;padding:0 24px;line-height:1.5}}
a{{color:#93c5fd}}h1,h2{{line-height:1.2}}.card{{background:#172337;padding:20px;border-radius:12px;margin:20px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}figure{{margin:8px 0}}img,video{{width:100%;border-radius:6px}}figcaption{{color:#c3cedc}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #42516a;text-align:left;white-space:nowrap}}
details{{border-top:1px solid #42516a;padding:8px}}summary{{cursor:pointer}}code{{overflow-wrap:anywhere}}
</style><h1>Paired appearance evaluation</h1><p>Dry, appearance-only simulation using a frozen model on two fixed geometries. This is not real-world validation. Route commands and launch/finish braking remain privileged. The variants share the intended physics.</p>
<div class="card"><h2>Completion and contacts</h2><div class="scroll"><table><tr><th>Scene</th><th>Event</th><th>Route complete</th><th>Pass</th><th>Contact-free completions</th><th>Total contacts</th><th>Mean contacts/run</th></tr>{"".join(group_rows)}</table></div>
<p>Pass uses the existing completion and safe-stop referee. Contacts count unique cones with PhysX contact. Clean completion means route complete with zero cone contacts; it does not add an off-course criterion.</p></div>
<div class="card"><h2>Paired differences</h2><p>Paddock minus plain for each seed. Completion is 1 for route complete and 0 otherwise. Off-course duration counts sampled off-course frames at 0.05 seconds per frame.</p><div class="scroll"><table><tr><th>Case</th><th>Contact difference</th><th>Completion difference</th><th>Off-course seconds difference</th></tr>{"".join(delta_rows)}</table></div></div>
<div class="card"><h2>Pair integrity</h2><div class="scroll"><table><tr><th>Case</th><th>Track equal</th><th>Start offsets equal</th><th>Initial state equal</th><th>Physics equal</th><th>Checkpoint matches protocol</th><th>State equal</th><th>Weights and state unchanged</th></tr>{"".join(pair_rows)}</table></div><p>Checkpoint SHA-256: <code>{esc(protocol["checkpoint_sha256"])}</code>. Full per-pair hashes are in <a href="aggregate.json">aggregate.json</a>.</p></div>
<div class="card"><h2>Fixed-pose previews</h2>{"".join(previews) or "<p>Preview images were not available.</p>"}</div>
<div class="card"><h2>Every case on video</h2>{("<p>Video generation skipped.</p>" if skip_video else '<video controls preload="metadata" src="overview.mp4"></video><p><a href="overview.mp4">Open overview video</a></p>')}<p>Each passing case includes the start, every recorded contact window and the final 12 seconds. Failed cases are shown in full. Expand a case to play its clips; full recordings are linked below. Times refer to the recording.</p>{"".join(overview)}</div>
<div class="card"><h2>Outcomes</h2><img src="{embedded(root / "outcomes.png")}" alt="Completion and contacts for every run"><img src="{embedded(root / "trajectories.png")}" alt="Trajectories for every run"></div>
<div class="card"><h2>Performance by scene and event</h2><p>Median (minimum–maximum) across runs. Timing columns summarize each run's median frame time. Failed runs may cover different route portions, so group wall time does not isolate the causal overhead of the environment.</p><div class="scroll"><table><tr><th>Scene</th><th>Event</th><th>Real-time factor</th><th>Control Hz wall</th><th>Wall seconds</th><th>Render ms</th><th>Inference ms</th><th>Physics ms</th></tr>{"".join(performance_rows)}</table></div></div>
<div class="card"><h2>Performance per case</h2><p>Values are median (minimum–maximum) over sampled frames or telemetry. Render, inference/controller and physics times are milliseconds. GPU utilization and power are device-wide; host RAM is whole-host usage. Startup is excluded from step timings. Torch peak allocation and process RSS are in the aggregate data.</p><div class="scroll"><table><tr><th>Case</th><th>Result</th><th>Contacts</th><th>Off-course events</th><th>Render ms</th><th>Inference ms</th><th>Physics ms</th><th>GPU %</th><th>Power W</th><th>Host RAM GiB</th><th>Recording</th></tr>{"".join(rows)}</table></div><img src="{embedded(root / "performance.png")}" alt="Per-case timing and GPU median and range"></div>
{benchmark_html}
<div class="card"><h2>Limits of this experiment</h2><p>Five seeded trials per fixed layout vary starting pose and scene-specific details, including plain lighting and paddock surface repairs. They test dry daylight only. Cameras have no measured ISP, noise, rolling shutter or weather effects. Distant trees and sky come from an HDR environment, so they do not provide close-range parallax. The car uses a generic Ackermann physics model with a box chassis collider beneath the tapered visual body. A physical contact can occur outside the visible body. Venue props outside the cone envelope are visual only. Successful route completion can include cone hits and does not establish real-world driving ability.</p></div>
<p>{profile_text} Powered by <a href="https://polyhaven.com/">Poly Haven</a>; assets are <a href="https://polyhaven.com/license">CC0</a>.</p><p><a href="protocol.json">Protocol</a> · <a href="aggregate.json">Aggregate data</a> · <a href="artifact_audit.json">Scene and video audit</a> · <a href="../frozen_source_audit.json">Source audit</a> · <a href="../integration_realistic/verification.json">Integration verification</a>. Full composite recordings, model plans and measurements are preserved in each case directory. The labeled camera panels show the model views, with lossy video compression; the policy received the original unannotated RGB arrays.</p></html>'''
    (root / "report.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("isaac_eval/runs/realism_v1"))
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read(root / "protocol.json")
    cases = load_cases(root, protocol)
    aggregate = aggregate_cases(cases, protocol)
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    graphs(root, cases, aggregate)
    if not args.skip_video:
        videos(root, cases)
    report(root, protocol, cases, aggregate, args.skip_video)
    print(f"Wrote {root / 'report.html'} for {len(cases)} runs")


if __name__ == "__main__":
    main()
