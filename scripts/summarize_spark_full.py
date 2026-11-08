"""Build a compact result record from the completed Spark run's measured artifacts."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path("runs/spark_full")
    stages = [read(root / f"spark_full_{n}.json") for n in ["stage1", "bootstrap", "v2", "refresh"]]
    fit_seconds = sum(r["fit_seconds"] for r in stages)
    checkpoint = Path("runs/spark_full_refresh/ckpt/best.pt")
    verification = read(root / "checkpoint-verification.json")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == verification["sha256"]
    benches = {n: read(root / f"full_bench_{n}.json") for n in ["baseline", "workers4", "workers8", "workers12"]}
    evaluation = root / "evaluation"
    loops = {n: read(evaluation / f"{n}_open_loop.json") for n in ["untouched", "short", "full"]}
    closed = {"short": read(evaluation / "short_closed_loop.json"), "full": read(evaluation / "closed_loop.json")}
    driving = {}
    for name, result in closed.items():
        rows = [r for r in result["runs"] if r["discipline"] == "trackdrive"]
        driving[name] = {
            "runs": len(rows),
            "passed": sum(r["pass"] for r in rows),
            "clean_passes": sum(r["pass"] and r["doo"] == 0 and r["oc"] == 0 for r in rows),
            "total_cones": sum(r["doo"] for r in rows),
            "cones_per_run": sum(r["doo"] for r in rows) / len(rows),
            "off_course_events": sum(r["oc"] for r in rows),
        }
    origin = read(root / "timing-origin.json")
    finished = read(root / "finished.json")
    phases = [json.loads(s) for s in (root / "phases.jsonl").read_text().splitlines()]
    result = {
        "name": "spark_full_venue",
        "protocol": "V2-scale venue training with local initialization; independent held-out suite",
        "dataset": read(root / "data-audit.json"),
        "checkpoint": verification,
        "training": {
            "steps": sum(r["steps"] for r in stages),
            "fit_seconds": fit_seconds,
            "gpu_busy_time_weighted_mean_percent": sum(
                r["fit_seconds"] * r["hardware_mean"]["gpu_percent"] for r in stages
            )
            / fit_seconds,
            "peak_cuda_allocated_gib": max(r["peak_allocated_gib"] for r in stages),
            "peak_system_used_gib": max(t["system_used_gib"] for r in stages for t in r["telemetry"]),
            "stages": [
                {
                    k: r[k]
                    for k in [
                        "label",
                        "config",
                        "steps",
                        "fit_seconds",
                        "peak_allocated_gib",
                        "hardware_mean",
                        "torch",
                        "cuda_version",
                        "gpu",
                    ]
                }
                for r in stages
            ],
        },
        "timing": {
            "dataset_elapsed_seconds_approx": origin["dataset_elapsed_seconds_approx"],
            "cache_seconds": sum(
                read(root / f"{n}.done.json")["seconds"] for n in ["cache_base", "cache_bootstrap", "cache_v2"]
            ),
            "benchmark_process_seconds": sum(read(root / f"full_bench_{n}.done.json")["seconds"] for n in benches),
            "remote_end_to_end_seconds_approx": finished["finished_at"] - origin["first_track_written"],
            "basis": "End-to-end starts at the first generated track file; includes tuning and recovery pauses.",
        },
        "performance": {
            "selected": read(root / "selected_settings.json"),
            "speedup_percent": (
                benches["workers8"]["samples_per_second"] / benches["baseline"]["samples_per_second"] - 1
            )
            * 100,
            "measurements": {
                n: {
                    k: r[k]
                    for k in [
                        "steps",
                        "warmup_steps",
                        "seconds",
                        "samples_per_second",
                        "seconds_per_step",
                        "data_wait_seconds",
                        "peak_allocated_gib",
                        "hardware_mean",
                    ]
                }
                for n, r in benches.items()
            },
        },
        "open_loop": {
            n: {"episodes": len(r["episodes"]), "wall_seconds": r["wall_seconds"], "summary": r["summary"]}
            for n, r in loops.items()
        },
        "closed_loop": closed,
        "trackdrive_comparison": driving,
        "video": read(evaluation / "video.json"),
        "phase_records": phases,
        "final_training_source_sha256": stages[-1]["source_sha256"],
    }
    out = Path("benchmarks/performance/spark-full-2026-11-08.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "training_minutes": fit_seconds / 60,
                "performance": result["performance"]["speedup_percent"],
                "trackdrive": driving,
                "video": result["video"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
