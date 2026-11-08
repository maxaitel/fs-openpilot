"""Durable, measured full training pipeline; run in the Spark container."""

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("runs/spark_full")
DATA = "data/sim/spark_full_venue"


def run(label, command):
    marker = OUT / f"{label}.done.json"
    if marker.exists():
        return json.loads(marker.read_text())
    start = time.time()
    status = {"phase": label, "started_at": start, "command": command}
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
    print(f"START {label}", flush=True)
    with (OUT / f"{label}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    status.update(finished_at=time.time(), seconds=time.time() - start, returncode=result.returncode)
    with (OUT / "phases.jsonl").open("a") as log:
        log.write(json.dumps(status) + "\n")
    if result.returncode:
        (OUT / "status.json").write_text(json.dumps(status, indent=2))
        raise RuntimeError(f"{label} failed: see {OUT}/{label}.log")
    marker.write_text(json.dumps(status, indent=2))
    print(f"DONE {label} in {status['seconds']:.1f}s", flush=True)
    return status


def cache(label, checkpoint=None, hidden=False):
    code = (
        "from fsop.data.feature_cache import build_cache; "
        f"build_cache({DATA!r}, ckpt={checkpoint!r}, batch=64, hidden_only={hidden!r}, prefetch=4)"
    )
    run(label, [sys.executable, "-c", code])


def train(name, overrides=()):
    command = [
        sys.executable,
        "scripts/benchmark_training.py",
        "--exit-after-save",
        "--cfg",
        f"configs/train/{name}.yaml",
        "--fit",
        "--label",
        name,
        "--out",
        str(OUT),
    ]
    for item in overrides:
        command += ["-o", item]
    run(name, command)


def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(32768, hard)), hard))
    OUT.mkdir(parents=True, exist_ok=True)
    while not (OUT / "data.exit").exists():
        time.sleep(10)
    if (OUT / "data.exit").read_text().strip() != "0":
        raise RuntimeError("Dataset generation failed")
    cache("cache_base")
    train("spark_full_stage1")
    train("spark_full_bootstrap")
    cache("cache_bootstrap", "runs/spark_full_bootstrap/ckpt/best.pt", True)
    comparisons = []
    for label, workers, optimized in [
        ("baseline", 12, False),
        ("workers4", 4, True),
        ("workers8", 8, True),
        ("workers12", 12, True),
    ]:
        label = f"full_bench_{label}"
        command = [
            sys.executable,
            "scripts/benchmark_training.py",
        "--exit-after-save",
            "--cfg",
            "configs/train/spark_full_v2.yaml",
            "--label",
            label,
            "--out",
            str(OUT),
            "--warmup",
            "20",
            "--steps",
            "200",
            "-o",
            f"num_workers={workers}",
        ]
        if not optimized:
            command += ["-o", "channels_last=false", "-o", "fused_adamw=false", "-o", "cudnn_benchmark=false"]
        run(label, command)
        measurement = json.loads((OUT / f"{label}.json").read_text())
        assert measurement["finite_loss"]
        if optimized:
            comparisons.append((measurement["samples_per_second"], workers))
    speed, workers = max(comparisons)
    (OUT / "selected_settings.json").write_text(json.dumps({"num_workers": workers, "samples_per_second": speed}))
    train("spark_full_v2", [f"num_workers={workers}"])
    cache("cache_v2", "runs/spark_full_v2/ckpt/best.pt", True)
    train("spark_full_refresh", [f"num_workers={workers}"])
    (OUT / "status.json").write_text(json.dumps({"phase": "training_complete", "finished_at": time.time()}))


if __name__ == "__main__":
    main()
