# DGX Spark performance report — 8 November 2026

Measured on `max-spark`, using the actual simulator dataset, CUDA training, checkpoint saving and model inference. The main result is **22.2% higher stage-two training throughput**, or **18.2% less training time**, at the same batch size after the fixes.

## Hardware and workload

- NVIDIA GB10, 48 SMs; 128 GB unified memory (Linux reports about 121 GiB).
- Driver 580.178.04; NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.12-py3`.
- PyTorch `2.10.0a0+b4e4ee81d3.nv25.12`, CUDA 13.1 through the container's forward-compatibility driver, cuDNN 9.17.
- Four PyTorch CPU threads, one OpenBLAS thread. No other CUDA compute jobs were present at the initial check; existing desktop/services were left running.
- Upstream revision: `d75c7d902ffc082076345ed843df9d6ea982cea8`.
- Generated 40 trackdrive tracks (seeds 1000–1039), plus two skidpad and two acceleration episodes. Classic procedural rendering, seed 42, default domain-randomization mix, no optional MetaDrive textures or FSOCO.
- **44 episodes / 4,486 frames / 1.1 GB including feature caches**. Dataset split: 37 training episodes, five validation episodes, two test episodes; 3,749 usable training samples and 495 validation samples.
- Recording took **23.1 s**, with the tracks already generated. FP32 feature caching took **7.6 s** including process/model startup. These exclude initial downloads and environment installation.

This is a small hardware/training test, not a reproduction of the published A100 model. The upstream fine-tuned checkpoints and recorded v1/v2 datasets are not supplied in the repository. Its driving-quality results must not be added to the upstream leaderboard as a comparable run.

## Main comparison: full-trunk stage two

Both runs used batch 64, 12 data workers, BF16 autocast, seed 0, the same examples and loss, and 31.2 million trainable parameters. The unchanged baseline was imported from an archived copy of the upstream source. Each run performed 20 warm-up steps, then **200 measured optimizer steps** spanning multiple dataset passes. Timing synchronizes CUDA at the measurement boundaries. Hardware telemetry was sampled approximately once per second.

| Measurement | Upstream | Optimized |
|---|---:|---:|
| Training samples/s | 233.7 | **285.6** |
| Wall time per optimizer step | 273.8 ms | **224.1 ms** |
| 200 measured steps | 54.77 s | **44.82 s** |
| Mean GPU busy | 83.4% | **89.0%** |
| Time waiting for data, total | 4.28 s | **2.59 s** |
| Peak allocated CUDA tensors | 3.44 GiB | 5.27 GiB |
| Peak CUDA allocator reservation | 4.33 GiB | 6.06 GiB |
| Mean whole-system used memory | 18.96 GiB | 22.10 GiB |
| Mean CPU use, across all cores | 20.1% | 21.3% |
| Mean GPU-reported power | 37.36 W | 37.30 W |
| Maximum sampled temperature | 69°C | 70°C |
| Batch-one FP32 forward, median / p95 | 4.36 / 4.42 ms | 4.10 / 4.40 ms |
| Batch-one BF16 forward, median / p95 | 6.65 / 6.70 ms | **4.12 / 4.14 ms** |

Inference numbers cover the PyTorch model forward with inputs already on the GPU, including both camera inputs and prediction heads. They exclude rendering, image preparation, transfer, decoding, control and scheduling. They are not an end-to-end control-loop latency claim. FP32 is already fast at batch one; BF16 is useful for training and is not automatically faster for this small inference workload.

**Startup matters:** the 20 warm-up training steps took 7.56 s upstream versus 17.29 s optimized, primarily because convolution autotuning searches kernels. Warm-up plus 200 measured steps therefore took about 62 s in both cases. The steady-state improvement pays off over longer training runs.

At these measured steady-state rates, **20,000 stage-two steps are approximately 91.3 minutes upstream versus 74.7 minutes optimized**, saving 16.6 minutes. These are extrapolations, excluding initialization, validation, checkpoints and changes in dataset/storage behavior; a 20,000-step run was not performed.

## Is the hardware being used efficiently?

The optimized run keeps the GPU busy about 89% of the time. It is not memory-capacity limited: about 5.3 GiB is allocated to CUDA tensors and about 22 GiB is used by the whole system. The remaining capacity is useful headroom, not a reason to make batches arbitrarily larger. GPU utilization measures busy time, not achieved FLOPs, tensor-core occupancy, memory bandwidth or percentage of the advertised peak. The measurements do not establish maximum theoretical hardware efficiency.

The main remaining costs are actual GPU kernels, many small eager operations, and brief data-loading pauses at dataset boundaries. SM clocks stayed around 2.41 GHz; queried thermal-slowdown status was inactive. Power above is GPU-reported, not wall-socket power. NVIDIA-SMI's memory-busy metric returned zero on this unified-memory system and is not used to infer bandwidth utilization.

Batch sweeps confirmed that filling more memory did not improve this workload:

| Exploratory setting | Samples/s | Peak allocated CUDA memory |
|---|---:|---:|
| Tuned batch 64, 40 measured steps | 307.8 | 5.28 GiB |
| Tuned batch 128, 12 workers, 40 steps | 288.4 | 6.98 GiB |
| Tuned batch 256, 12 workers, 20 steps | 219.6 | 13.60 GiB |
| Channels-last + fused optimizer, **autotuning disabled**, batch 64, 80 steps | 201.2 | 3.55 GiB |

Short exploratory measurements benefit more from prefetched batches, so use the longer **285.6 samples/s** result for planning. Batch 256 spent 8.03 of 23.32 measured seconds waiting for data. Batch 64 is the selected stage-two profile. Keep channels-last and autotuning together on this setup.

Cached-feature stage one achieved 1,288 samples/s upstream and 1,362 samples/s in a short tuned sweep (batch 256; the latter used eight workers). These were 40-step exploratory measurements, not a sustained matched-worker comparison.

## Changes retained

1. Specialize constant ONNX `Reshape`, `Slice`, `Squeeze` and `Expand` metadata once when loading the model. The original conversion repeatedly read shape/index tensors back from CUDA: a five-forward profile recorded 205 stream synchronizations. Dynamic metadata and negative-step slices retain the original converter. Existing initializer/checkpoint keys are preserved.
2. Cache Zarr camera-array handles per episode/worker, avoiding repeated metadata opens for every sample.
3. Pin CUDA loader batches so existing nonblocking transfers can work as intended, including validation.
4. Add explicit settings for channels-last tensors, cuDNN autotuning and fused CUDA AdamW. These are enabled in the Spark profiles; generic configurations retain their prior layout, optimizer and autotuning defaults.
5. Add reproducible throughput/telemetry, dataset-generation and train/evaluate scripts, plus a container recipe.

These follow the relevant [PyTorch performance guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) and account for Spark's [unified-memory architecture](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

## Reproduce

The measured machine has the checkout at `/home/maxaitel/fs-openpilot` and the container `fsop-spark-bench`. Results, data and checkpoints live in the mounted checkout. The local Mac checkout contains the source changes and copied reports.

For a fresh container, from the repository root on the Spark:

```bash
docker build -f scripts/Dockerfile.spark -t fsop-spark .
docker run --rm --gpus all --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace/fs-openpilot" -w /workspace/fs-openpilot \
  fsop-spark bash scripts/run_spark_smoke.sh
```

The smoke script records/reuses the dataset, fetches/prepares the pinned openpilot model, caches features, trains 500 steps in each stage, saves checkpoints and evaluates the final stage-two best checkpoint. Data, caches and run directories are intentionally gitignored. To retain an earlier run, change the labels/output paths before rerunning.

To measure steady-state stage-two throughput independently:

```bash
python scripts/benchmark_training.py \
  --cfg configs/train/spark_stage2.yaml \
  -o sim_root=data/sim/spark_test -o 'init_ckpt=""' \
  --label spark_stage2_measure --warmup 20 --steps 200 --inference
```

The committed machine-readable measurements are in [`benchmarks/performance/spark-2026-11-08.json`](../benchmarks/performance/spark-2026-11-08.json). Full per-second telemetry, logs, environment versions and source hashes are under `runs/spark_benchmark/` on both machines.

## Completed short training run

| Phase | Actual steps | Batch | Time inside `Trainer.fit()` |
|---|---:|---:|---:|
| Stage one, frozen vision trunk | 500 | 256 | **103.85 s** |
| Stage two, full vision trunk | 500 | 64 | **137.00 s** |
| Total | 1,000 | — | **240.85 s (4 min 1 s)** |

These times include the initial validation, validation at steps 250 and 500, and checkpoint writes; they exclude model construction (about 0.85 s per stage), initial environment setup, data generation and evaluation. Stage two initialized from stage one's best validation checkpoint. Both best and last checkpoints were saved. Peak CUDA allocation across stage-two training **and FP32 validation** was **8.97 GiB**, higher than the training-only throughput test; mean whole-system used memory was 23.35 GiB. Average GPU busy across that complete fit, including initialization/validation pauses, was 79.5%.

A monitoring-only issue occurred after stage one's fit had finished: concurrent telemetry subprocess creation and DataLoader worker creation left the telemetry thread blocked during cleanup. The completed checkpoints were intact, and `fit_seconds` had already been measured before the cleanup stall. That run's hardware telemetry was discarded. The benchmark harness now uses NVML directly, and stage two completed and exited normally. The debugging pause is not included in the training times above.

Open-loop evaluation used **190 frames from two held-out tracks**, excluding the first five frames per episode. The evaluated model recomputed its own temporal history.

| Metric | Untouched openpilot | Short trained checkpoint |
|---|---:|---:|
| Lateral error at 1 s | 1.093 m | **0.474 m** |
| Lateral error at 2 s | 4.034 m | **1.706 m** |
| Average displacement error through 4 s | 4.406 m | **2.204 m** |
| Heading error at 1 s | 12.96° | **5.47°** |
| Image-head projected cone F1 | Not applicable | **0.484** |

These are useful smoke-test improvements, not evidence of convergence or a statistically representative driving benchmark. The checkpoint is still from a deliberately small training run.

Closed-loop evaluation used those same two held-out trackdrive tracks, one lap each, plus the fixed acceleration and skidpad layouts; light DR, standing starts, no steering noise, and a 90-second simulated-time limit. Training-track IDs are explicitly excluded from the trackdrive test.

| Mission | Passed | Cones down/out | Result |
|---|---:|---:|---|
| Trackdrive seed 1006 | 1/1 | 11 | Finished |
| Trackdrive seed 1013 | 1/1 | 7 | Finished |
| Acceleration | 1/1 | 0 | Finished |
| Skidpad | 0/1 | 23 | Disqualified / incomplete |

The four episodes took **58.82 s** of real execution time. Median policy-call times were **6.49–6.70 ms**, with per-run p95 values **6.84–6.95 ms**. This includes CPU image preparation, GPU model execution, cone decoding and control, with CUDA synchronization, but excludes rendering/dynamics. The complete simulated loops achieved **62.6–66.3 steps/s**. These timings support the model's 20 Hz runtime target on this Spark; they do not establish real-car driving performance. The cone hits and skidpad failure remain quality limitations.

## Validation and saved artifacts

- Full-model CUDA outputs before/after the constant-shape specialization were **bit-identical** for batches 1 and 3, including the plan, hidden state, BEV and camera heads.
- Constant-shape tests check outputs and gradients on CPU/CUDA, checkpoint keys, dynamic-shape fallback, and ONNX export with a different batch size. Frame-pair tests check temporal order and the first-frame boundary.
- Final suite: **47 passed, one failed**. The failing pre-existing test is `test_conversion_parity_cpu`: the original FP16 ONNX reference emits NaNs on this ARM CPU / ONNX Runtime 1.31.1 environment. The same test fails using unchanged upstream source. The FP32 reference comparison remains finite (about 4.7e-7 relative error before changes).
- `ruff check src tests scripts` passes.
- The trained checkpoint was exported successfully. ONNX Runtime versus PyTorch maximum absolute error over the original model outputs was **2.52e-4**, within the exporter's 1e-2 threshold.
- The stage-two checkpoint and exported model were copied to the local checkout:
  - `runs/spark_smoke_stage2/ckpt/best.pt`
  - `runs/spark_benchmark/spark_smoke.onnx`
- On the Spark, both stage-one and stage-two best/last checkpoints, dataset, caches and all logs remain in `/home/maxaitel/fs-openpilot/`.

The benchmark container can be restarted with `docker start fsop-spark-bench`; the existing environment then runs commands via `docker exec -w /workspace/fs-openpilot fsop-spark-bench ...`.

## Realistic venue video

The packaged venue renderer was enabled after downloading all 20 optional Poly Haven sky panoramas with `python -m fsop.cli sim fetch-assets`. Venue mode adds photographic skies, textured asphalt, shadows and event scenery; it does not change the checkpoint.

The same held-out trackdrive track (seed 1013), one lap, light domain randomization, seed 0, and `venue_prob=1.0` finished in **52.80 s**, with **4 cones down/out**, **0 off-course events**, and an **8 s penalty** (60.80 s scored time). Rendering and simulation took 53.67 s wall time and produced 1,161 frames at 20 FPS. This is a single scene result, not evidence of improved model quality or a controlled graphics-only comparison.

The video still uses our short-trained checkpoint, SHA256 `5ed2e9d3fff8e86f09237af8202a5342f2852648acdebf952d0085bf76bbdbbb`. It is the full network, but not the author's `a100_v2_refresh` weights. No downloadable checkpoint was found in the repository tree, GitHub releases, or matching public Hugging Face model listings during this check. The author's published demo and benchmark are distinct from this smoke test.

Local playback: `artifacts/videos/spark-venue-test.mp4`. Raw result: `runs/spark_benchmark/video/spark_venue.json`. Reproduction helper: `runs/spark_benchmark/render_venue.py`.
