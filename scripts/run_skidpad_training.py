import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    output = Path("runs/spark_skidpad")
    output.mkdir(parents=True, exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(32768, hard)), hard))
    python = sys.executable
    commands = [
        ("record", [python, "scripts/prepare_skidpad_training.py"]),
        (
            "code_only",
            [python, "scripts/probe_skidpad.py", "drive", "--output",
             str(output / "code_only_validation"), "--seeds", "61000", "61001", "61002"],
        ),
        (
            "cache",
            [
                python,
                "-c",
                "from fsop.data.feature_cache import build_cache; "
                "build_cache('data/sim/spark_skidpad_intent', "
                "ckpt='runs/spark_full_refresh/ckpt/best.pt', batch=64, prefetch=4)",
            ],
        ),
        (
            "policy",
            [
                python,
                "scripts/benchmark_training.py",
                "--cfg",
                "configs/train/spark_skidpad_policy.yaml",
                "--label",
                "spark_skidpad_policy",
                "--fit",
                "--exit-after-save",
                "--out",
                str(output),
            ],
        ),
        (
            "validation",
            [
                python,
                "scripts/probe_skidpad.py",
                "drive",
                "--checkpoint",
                "runs/spark_skidpad_policy/ckpt/best.pt",
                "--output",
                str(output / "policy_validation"),
                "--seeds",
                "61000",
                "61001",
                "61002",
            ],
        ),
    ]
    for name, command in commands:
        marker = output / f"{name}.done.json"
        if marker.exists():
            continue
        if name in ("code_only", "cache"):
            deadline = time.monotonic() + 1200
            while not (output / "baseline_test" / "driving.json").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("Baseline evaluation has not completed; GPU training remains deferred")
                (output / "status.json").write_text(json.dumps({"phase": "waiting_for_baseline"}))
                time.sleep(5)
        started = time.time()
        status = {"phase": name, "started_at": started, "command": command}
        (output / "status.json").write_text(json.dumps(status, indent=2))
        print(f"START {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status.update(seconds=time.time() - started, returncode=result.returncode)
        (output / "status.json").write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output / f'{name}.log'}")
        marker.write_text(json.dumps(status, indent=2))
        print(f"DONE {name} in {status['seconds']:.1f}s", flush=True)
    (output / "status.json").write_text(json.dumps({"phase": "policy_validation_complete", "time": time.time()}))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        Path("runs/spark_skidpad/pipeline.exit").write_text("1")
        raise
    Path("runs/spark_skidpad/pipeline.exit").write_text("0")
