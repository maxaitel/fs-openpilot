import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/pi-correction-2026-11-10"
OUT = Path(__file__).resolve().parent
track_path = BENCH / "development/baseline/heavy_skid/track.json"
track = json.loads(track_path.read_text())
track_hash = hashlib.sha256(json.dumps(track, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
assert track["timekeeping_line"] == [[0.0, 1.5], [0.0, -1.5]]
assert track["start_pose"][2] == 0.0
half_gate = 1.5
center_y = track["generator"]["center_distance"] / 2
source_paths = [ROOT / p for p in (
    "src/fsop/sim/skidpad.py", "src/fsop/sim/expert.py",
    "src/fsop/sim/referee.py", "src/fsop/eval/closed_loop.py", "isaac_eval/scene.py",
)]
archive = BENCH / "head_training/source.zip"
with zipfile.ZipFile(archive) as z:
    for path in source_paths:
        member = str(path.relative_to(ROOT))
        member = member.removeprefix("src/")
        assert z.read(member) == path.read_bytes(), path
inputs = [track_path, archive, *source_paths]
result = {
    "schema_version": 1,
    "method": "Linear interpolation of consecutive saved 20 Hz car positions at x=0; finite gate test matches SkidpadRoute.advance.",
    "geometry": {"track_canonical_sha256": track_hash, "gate": track["timekeeping_line"],
                 "circle_centers": [[0, -center_y], [0, center_y]], "stop_zone": track["stop_zone"]},
    "source_matches_training_archive": True,
    "episodes": {},
}
for arm in ("baseline", "candidate"):
    directory = BENCH / "head_development" / arm / "heavy_skid"
    paths = {name: directory / f"{name}.json" for name in ("summary", "frames", "realization")}
    inputs.extend(paths.values())
    summary, frames, realization = [json.loads(paths[n].read_text()) for n in ("summary", "frames", "realization")]
    assert realization["track_sha256"] == track_hash
    assert len(frames) == summary["recorded_frames"]
    crossings = []
    increments = []
    for previous, current in zip(frames, frames[1:]):
        assert math.isclose(current["t"] - previous["t"], 0.05, abs_tol=1e-12)
        increments.append({"t": current["t"], "delta_s_m": current["s"] - previous["s"],
                           "travel_m": math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])})
        if previous["x"] * current["x"] < 0:
            alpha = -previous["x"] / (current["x"] - previous["x"])
            y = previous["y"] + alpha * (current["y"] - previous["y"])
            crossings.append({"t": previous["t"] + alpha * (current["t"] - previous["t"]),
                              "x": 0.0, "y": y, "forward": current["x"] > previous["x"],
                              "inside_gate": abs(y) <= half_gate,
                              "gate_miss_m": max(abs(y) - half_gate, 0.0),
                              "before_frame": previous["frame"], "after_frame": current["frame"]})
    accepted_crossings = [c["t"] for c in crossings if c["forward"] and c["inside_gate"]]
    assert len(accepted_crossings) == len(summary["route"]["crossings"])
    assert all(abs(a - b) < 1e-10 for a, b in zip(accepted_crossings, summary["route"]["crossings"]))
    rr_time = summary["route"]["laps"][1]["finish_time"]
    contacts = []
    for contact in summary["contact_events"]:
        frame = frames[contact["control_frame"]]
        index = int(contact["cone"].rsplit("c", 1)[-1])
        xy = realization["cones"]["xy"][index]
        contacts.append({**contact, "car_before_contact_xy": [frame["x"], frame["y"]],
                         "car_before_contact_left_radius_m": math.hypot(frame["x"], frame["y"] - center_y),
                         "initial_cone_xy": xy,
                         "initial_cone_left_radius_m": math.hypot(xy[0], xy[1] - center_y),
                         "after_second_right_lap": contact["t"] >= rr_time})
    assert len(contacts) == summary["physical_cone_hits"]
    s_end = realization["mission"]["s_end"]
    terminal_progress = frames[-1]["s"]
    first_end = next((f for f in frames if f["s"] >= s_end), None)
    stationary_tail = []
    for frame in reversed(frames):
        if frame["speed"] >= 0.05:
            break
        stationary_tail.append(frame)
    post_rr = [f for f in frames if f["t"] >= rr_time]
    episode = {
        "result": summary["result"], "seconds_simulated": summary["seconds_simulated"],
        "physical_contacts": summary["physical_cone_hits"], "legacy_doo": summary["doo"],
        "off_course_events": summary["oc"], "accepted_route": summary["route"]["accepted_route"],
        "route_complete": summary["route"]["complete"], "route_violation": summary["route"]["violation"],
        "all_x_zero_crossings": crossings, "physical_contact_timeline": contacts,
        "contacts_before_second_right_finish": sum(c["t"] < rr_time for c in contacts),
        "contacts_after_second_right_finish": sum(c["t"] >= rr_time for c in contacts),
        "progress": {"s_end_m": s_end, "last_saved_s_m": terminal_progress,
                     "last_saved_s_fraction": terminal_progress / s_end,
                     "summary_fraction": summary["progress"], "first_s_at_least_end": first_end,
                     "largest_positive_increment": max(increments, key=lambda d: d["delta_s_m"]),
                     "minimum_increment": min(increments, key=lambda d: d["delta_s_m"])},
        "continuous_stationary_tail_first_frame": stationary_tail[-1] if stationary_tail else None,
        "post_rr_max_y": max(f["y"] for f in post_rr),
        "post_rr_min_y": min(f["y"] for f in post_rr),
        "final_state": summary["final_state"],
    }
    result["episodes"][arm] = episode
result["limitations"] = [
    "One exposed episode per model. Different durations prevent treating fewer total contacts as improved avoidance.",
    "Contact timestamps identify the start of the control interval containing first physical contact. Listed car positions precede that interval.",
    "Cone coordinates are initial positions. They do not reconstruct cone motion after collision.",
    "This trace identifies missed gates, contacts, and end-state mismatch. It does not isolate a perception, plan, controller, or contact-dynamics root cause.",
    "The terminal brake rule is source-derived. Per-frame controller diagnostics are not included in these compact inputs.",
]
result["input_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT / "trace.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), sharex=True, sharey=True)
palette = {"blue": "#3a7bd5", "yellow": "#d3ac00", "orange": "#e77a2b", "big_orange": "#e77a2b"}
for arm, ax, color in zip(("baseline", "candidate"), axes, ("#153b66", "#7d3ca3")):
    directory = BENCH / "head_development" / arm / "heavy_skid"
    frames = json.loads((directory / "frames.json").read_text())
    realization = json.loads((directory / "realization.json").read_text())
    episode = result["episodes"][arm]
    for xy, cone_color in zip(realization["cones"]["xy"], realization["cones"]["colors"]):
        ax.scatter(*xy, marker="^", color=palette.get(cone_color, "gray"), s=17, alpha=0.75, zorder=1)
    ax.add_patch(Polygon(track["stop_zone"], facecolor="#e7efe6", edgecolor="#667766", alpha=0.6))
    ax.plot([f["x"] for f in frames], [f["y"] for f in frames], color=color, linewidth=1.8, zorder=2)
    ax.plot([0, 0], [-half_gate, half_gate], color="#218c45", linewidth=4, zorder=3)
    for crossing in episode["all_x_zero_crossings"]:
        if crossing["forward"]:
            ax.scatter(crossing["x"], crossing["y"], marker="*" if crossing["inside_gate"] else "x",
                       color="#14833a" if crossing["inside_gate"] else "#d01d8b", s=95, zorder=5)
    contacts = episode["physical_contact_timeline"]
    ax.scatter([c["car_before_contact_xy"][0] for c in contacts], [c["car_before_contact_xy"][1] for c in contacts],
               facecolors="none", edgecolors="#bd2828", s=62, linewidths=1.4, zorder=4)
    final = episode["final_state"]
    ax.scatter(final["x"], final["y"], marker="s", color="#161616", s=65, zorder=6)
    ax.scatter(frames[0]["x"], frames[0]["y"], marker=">", color=color, s=65, zorder=5)
    ax.annotate("Start", (frames[0]["x"], frames[0]["y"]), xytext=(-19, -4), fontsize=9)
    ax.annotate("Terminal state", (final["x"], final["y"]), xytext=(final["x"] + 2, final["y"] - 4),
                arrowprops={"arrowstyle": "-", "color": "#444444"}, fontsize=9)
    ax.text(22, 0, "Stop zone", ha="center", fontsize=9, color="#445544")
    ax.set_title(f"{arm.title()} | {episode['result']}\n{episode['seconds_simulated']:.2f} sim s | {episode['physical_contacts']} contacts | accepted R,R", fontsize=12)
    ax.set(xlim=(-21, 39), ylim=(-23, 27), xlabel="World x (m)", aspect="equal")
    ax.grid(alpha=0.15)
axes[0].set_ylabel("World y (m)")
axes[0].annotate("Missed gate twice\ny = 2.60 m and 2.68 m\ngate ends at 1.50 m", (0, 2.64), xytext=(-20, 9),
                 arrowprops={"arrowstyle": "->", "color": "#d01d8b"}, fontsize=9, color="#9b1567")
legend = [Line2D([], [], color="#555555", label="Recorded car path"),
          Line2D([], [], color="#777777", marker="^", linestyle="None", label="Initial cone positions"),
          Line2D([], [], color="#14833a", marker="*", linestyle="None", label="Accepted timing crossing"),
          Line2D([], [], color="#d01d8b", marker="x", linestyle="None", label="Missed forward crossing"),
          Line2D([], [], color="#bd2828", marker="o", markerfacecolor="none", linestyle="None", label="Car before contact interval"),
          Line2D([], [], color="#161616", marker="s", linestyle="None", label="Terminal state")]
fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055), fontsize=9)
fig.suptitle("Head retraining did not complete skidpad", fontsize=17, y=0.94)
fig.text(0.5, 0.025, "One exposed run per model. Fewer contacts in a shorter failed run do not establish better avoidance.\nCones show their initial saved positions. Contact markers show the car at the start of the first-contact interval.",
         ha="center", fontsize=9)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.83, wspace=0.13)
fig.savefig(OUT / "trajectory.png", dpi=180, facecolor="white")
plt.close(fig)
(OUT / "output_sha256.json").write_text(json.dumps({name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                                    for name in ("trace.json", "trajectory.png")}, indent=2) + "\n")
print(json.dumps({arm: {k: v for k, v in record.items() if k in (
    "result", "seconds_simulated", "physical_contacts", "accepted_route",
    "contacts_before_second_right_finish", "contacts_after_second_right_finish")}
                  for arm, record in result["episodes"].items()}, indent=2))
