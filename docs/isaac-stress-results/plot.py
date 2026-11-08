"""Plot the audited paired_report.json copied into this directory."""

import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
with zipfile.ZipFile(root / "receipts.zip") as archive:
    report = json.loads(archive.read("paired_report.json"))
rows = report["episodes"]
labels = ["Skidpad\nlight", "Skidpad\nheavy", "Trackdrive\nlight", "Trackdrive\nheavy"]
colors = ["#28659b", "#d16c38"]
x = np.arange(4)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
for variant, color, offset in zip(("matched", "visual_stress_v1"), colors, (-0.18, 0.18), strict=True):
    selected = [r for r in rows if r["variant"] == variant]
    name = "Matched" if variant == "matched" else "Visual stress"
    bars = axes[0].bar(x + offset, [r["physical_contacts"] for r in selected], 0.36, label=name, color=color)
    axes[0].bar_label(bars, labels=[f"{r['physical_contacts']}\n{r['result']}" for r in selected], fontsize=8)
    axes[1].bar(x + offset, [r["real_time_factor"] for r in selected], 0.36, color=color)
axes[0].set_ylabel("Unique cone contacts")
axes[0].set_ylim(0, max(r["physical_contacts"] for r in rows) * 1.45 + 1)
axes[0].legend()
axes[1].set_ylabel("Simulated seconds / wall second")
for ax in axes:
    ax.set_xticks(x, labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
fig.suptitle(
    "Frozen-policy visual stress • one episode per condition\nFour paired cases, exploratory subset", fontsize=14
)
fig.savefig(root / "paired-results.png", dpi=180)
fig.savefig(root / "paired-results.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
keys = ("render_s", "policy_with_capture_s", "evidence_s", "physics_s")
names = ("Render and image processing", "Policy with capture", "Evidence write", "Physics")
bottom = np.zeros(8)
for key, name, color in zip(keys, names, ("#28659b", "#d16c38", "#789c6c", "#9b80b8"), strict=True):
    values = np.array([r["mean_ms"][key] for r in rows])
    ax.bar(np.arange(8), values, bottom=bottom, color=color, label=name)
    bottom += values
ax.set_xticks(np.arange(8), [f"{labels[i // 2]}\n{'matched' if i % 2 == 0 else 'stress'}" for i in range(8)])
ax.set_ylim(0, max(bottom) * 1.3)
ax.set_ylabel("Mean milliseconds per control step")
ax.set_title("Instrumented episode timings • CPU video export excluded")
ax.legend(loc="upper left", ncol=2, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)
fig.savefig(root / "step-timings.png", dpi=180)
fig.savefig(root / "step-timings.svg")

for name in ("paired-results.svg", "step-timings.svg"):
    path = root / name
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
