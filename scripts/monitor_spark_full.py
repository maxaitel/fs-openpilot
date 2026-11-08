"""Record one-second host/GPU telemetry independently of the training processes."""

import json
import time
from pathlib import Path

import psutil
import pynvml as nv

out = Path("runs/spark_full")
nv.nvmlInit()
h = nv.nvmlDeviceGetHandleByIndex(0)
with (out / "hardware.jsonl").open("a", buffering=1) as f:
    while not (out / "finished.json").exists():
        try:
            phase = json.loads((out / "status.json").read_text())["phase"]
        except (OSError, ValueError, KeyError):
            phase = "data"
        row = {
            "time": time.time(),
            "phase": phase,
            "cpu_percent": psutil.cpu_percent(),
            "system_used_gib": psutil.virtual_memory().used / 2**30,
            "disk_free_gib": psutil.disk_usage(".").free / 2**30,
            "gpu_percent": nv.nvmlDeviceGetUtilizationRates(h).gpu,
            "memory_busy_percent": nv.nvmlDeviceGetUtilizationRates(h).memory,
            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
            "temperature_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
        }
        f.write(json.dumps(row) + "\n")
        time.sleep(1)
nv.nvmlShutdown()
