"""Measure real training batches, with CUDA synchronization and hardware telemetry.

Run from the repository root; results are separate from the driving-quality leaderboard.
Example: python scripts/benchmark_training.py --cfg configs/train/cuda_stage2.yaml \
  -o sim_root=data/sim/spark_test -o init_ckpt= --label baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

import fsop
from fsop.config import load_config
from fsop.train.device import autocast_ctx
from fsop.train.trainer import TrainCfg, Trainer


class Telemetry:
  def __init__(self):
    self.rows = []
    self.stop = threading.Event()
    self.thread = threading.Thread(target=self._sample, daemon=True)

  def _sample(self):
    import psutil

    # NVML avoids spawning subprocesses while DataLoader forks workers. A subprocess's
    # exec-error pipe can otherwise be inherited by a worker, hanging Popen/shutdown.
    try:
      import pynvml as nvml

      nvml.nvmlInit()
      handle = nvml.nvmlDeviceGetHandleByIndex(0)
    except (ImportError, RuntimeError):
      nvml = None
    except Exception as exc:
      self.rows.append({"time": time.time(), "telemetry_error": str(exc)})
      nvml = None
    while not self.stop.is_set():
      row = {"time": time.time(), "cpu_percent": psutil.cpu_percent(),
             "system_used_gib": psutil.virtual_memory().used / 2**30}
      if nvml is not None:
        queries = {
          "gpu_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
          "memory_busy_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).memory,
          "power_w": lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
          "temperature_c": lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
          "sm_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
        }
        for key, query in queries.items():
          try:
            row[key] = float(query())
          except nvml.NVMLError:
            pass
      self.rows.append(row)
      self.stop.wait(1)
    if nvml is not None:
      nvml.nvmlShutdown()

  def __enter__(self):
    self.thread.start()
    return self

  def __exit__(self, *_):
    self.stop.set()
    self.thread.join()


def sync(device):
  if device.type == "cuda":
    torch.cuda.synchronize(device)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--cfg", type=Path, required=True)
  p.add_argument("-o", "--override", action="append", default=[])
  p.add_argument("--label", required=True)
  p.add_argument("--warmup", type=int, default=10)
  p.add_argument("--steps", type=int, default=40)
  p.add_argument("--fit", action="store_true", help="Run the normal trainer including validation/checkpoints")
  p.add_argument("--exit-after-save", action="store_true",
                 help="Isolated NGC worker: close DataLoader workers and exit after all outputs are saved, "
                      "avoiding native-library interpreter-shutdown crashes")
  p.add_argument("--inference", action="store_true", help="Also time batch-one model inference")
  p.add_argument("--out", type=Path, default=Path("runs/spark_benchmark"))
  args = p.parse_args()
  cfg = load_config(TrainCfg, args.cfg, args.override + [f"name={args.label}"])
  t0 = time.perf_counter()
  trainer = Trainer(cfg)
  sync(trainer.device)
  result = {"label": args.label, "config": vars(cfg), "torch": torch.__version__,
            "device": str(trainer.device), "setup_seconds": time.perf_counter() - t0}
  source_root = Path(fsop.__file__).parent
  result["source_sha256"] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(source_root.rglob("*.py"))}
  result["cpu_threads"] = torch.get_num_threads()
  result["cuda_version"] = torch.version.cuda
  result["cudnn_version"] = torch.backends.cudnn.version()
  if trainer.device.type == "cuda":
    result["gpu"] = torch.cuda.get_device_name()
    torch.cuda.reset_peak_memory_stats()
  if args.fit:
    with Telemetry() as telemetry:
      t0 = time.perf_counter()
      trainer.fit()
      sync(trainer.device)
      result["fit_seconds"] = time.perf_counter() - t0
    result["steps"] = trainer.step
  else:
    loader = trainer._loader(trainer.train_ds, True)
    if not len(loader):
      raise ValueError("Training dataset is smaller than one full batch")
    it = iter(loader)
    params = [p for g in trainer.opt.param_groups for p in g["params"]]
    trainer.model.train()
    waits, events = [], []
    last_batch = None

    def step(measure=False):
      nonlocal it, last_batch
      start = time.perf_counter()
      try:
        b = next(it)
      except StopIteration:
        it = iter(loader)
        b = next(it)
      if measure:
        waits.append(time.perf_counter() - start)
      b = trainer._to(b)
      last_batch = b
      if measure and trainer.device.type == "cuda":
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
      with autocast_ctx(trainer.device):
        out = trainer.forward(b)
        loss = sum(trainer.losses(out, b).values()) + trainer.l2sp()
      trainer.opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
      trainer.opt.step()
      if measure and trainer.device.type == "cuda":
        end.record()
        events.append((begin, end))
      return loss.detach()

    t0 = time.perf_counter()
    for _ in range(args.warmup):
      step()
    sync(trainer.device)
    result["warmup_seconds"] = time.perf_counter() - t0
    with Telemetry() as telemetry:
      t0 = time.perf_counter()
      for _ in range(args.steps):
        loss = step(True)
      sync(trainer.device)
      elapsed = time.perf_counter() - t0
    result.update(steps=args.steps, warmup_steps=args.warmup, seconds=elapsed,
                  samples_per_second=args.steps * cfg.batch_size / elapsed,
                  seconds_per_step=elapsed / args.steps, data_wait_seconds=sum(waits),
                  last_loss=float(loss), finite_loss=bool(torch.isfinite(loss)))
    if events:
      result["cuda_step_ms_median"] = float(np.median([a.elapsed_time(b) for a, b in events]))
    if args.inference:
      trainer.model.eval()
      b = {k: v[:1] for k, v in last_batch.items()}
      result["inference"] = {}
      with torch.inference_mode():
        for amp in (False, True):
          with autocast_ctx(trainer.device, enabled=amp):
            for _ in range(10):
              trainer.forward(b)
            times = []
            for _ in range(100):
              sync(trainer.device)
              t0 = time.perf_counter()
              trainer.forward(b)
              sync(trainer.device)
              times.append((time.perf_counter() - t0) * 1000)
          result["inference"]["bf16" if amp else "fp32"] = {
            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
  if trainer.device.type == "cuda":
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
  result["telemetry"] = telemetry.rows
  keys = {k for r in telemetry.rows for k, v in r.items() if k != "time" and isinstance(v, int | float)}
  result["hardware_mean"] = {k: float(np.mean([r[k] for r in telemetry.rows if k in r])) for k in keys}
  trainer.tb.close()
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / f"{args.label}.json").write_text(json.dumps(result, indent=2))
  print(json.dumps({k: v for k, v in result.items() if k not in ("telemetry", "config", "source_sha256")}, indent=2))
  if args.exit_after_save:
    # All checkpoints, TensorBoard events, synchronized GPU work and JSON results are complete.
    # This opt-in mode is for isolated worker processes on the pinned NGC image, whose native
    # finalizers can crash at interpreter teardown. Stop persistent loaders before bypassing them.
    for active_loader in (getattr(trainer, "_val_loader", None), locals().get("loader")):
      iterator = getattr(active_loader, "_iterator", None)
      if iterator is not None:
        iterator._shutdown_workers()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
  main()
