"""Record whole-device telemetry and command wall time without changing the child workload."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pynvml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    started = time.monotonic()
    process = subprocess.Popen(command)
    rows = []
    try:
        while process.poll() is None:
            ram = psutil.virtual_memory()
            rows.append(
                {
                    "wall_s": time.monotonic() - started,
                    "gpu_pct": pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu,
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000,
                    "temp_c": pynvml.nvmlDeviceGetTemperature(gpu, 0),
                    "cpu_pct": psutil.cpu_percent(),
                    "host_ram_gib": ram.used / 2**30,
                    "host_available_gib": ram.available / 2**30,
                }
            )
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        pynvml.nvmlShutdown()
        receipt = {
            "command": command,
            "wall_seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "samples": rows,
            "scope": "Whole GPU and host; host RAM is unified system memory, not dedicated VRAM",
        }
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
