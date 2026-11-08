# Full venue training on DGX Spark

This run trains the full SupercomboFS architecture using the published V2 approach, starting from the pretrained openpilot model. It is a new training run, not the author's unavailable `a100_v2_refresh` checkpoint or an exact reproduction of its initialization/data.

## Measured results

All 20,000 optimizer updates completed in **78 min 22 sec**, including validation and checkpoint writes. This is training time; dataset generation, feature caching, tuning and evaluation are additional. The model has **31,154,802 parameters**. Validation selected update 3,500 of the final 4,000-step phase.

| Phase | Updates | Training wall time | Mean GPU busy | Peak CUDA allocation |
|---|---:|---:|---:|---:|
| Initial head training | 6,000 | 20 min 41 sec | 93.7% | 2.96 GiB |
| Initial vision adaptation | 3,000 | 10 min 16 sec | 89.5% | 8.94 GiB |
| V2 full fine-tuning | 7,000 | 29 min 04 sec | 90.1% | 8.97 GiB |
| History refresh | 4,000 | 18 min 22 sec | 86.9% | 8.97 GiB |

Time-weighted GPU busy during training was **90.3%**. The matched full-network benchmark improved from **255.2 to 294.6 samples/sec (+15.4%)**, using eight loader workers. Four workers starved the GPU; twelve were slightly slower. GPU clocks stayed near 2.4 GHz and the sampled NVIDIA throttle counters showed no thermal or power-cap throttling. Peak sampled whole-system memory during training was **29.34 GiB**, while peak CUDA tensor allocation was **8.97 GiB**. Filling the Spark's 128 GB memory would not itself make this model train faster. GPU busy time does not establish Tensor Core/FLOP utilization.

Held-out predictions use 27,442 frames from all 143 test episodes, with temporal history recomputed separately for each model:

| Model | Lateral error at 1 sec | Lateral error at 2 sec | Trajectory ADE over 4 sec | Image-projected cone F1 |
|---|---:|---:|---:|---:|
| Untouched openpilot | 1.063 m | 3.604 m | 4.811 m | — |
| Earlier short training | 0.690 m | 2.403 m | 3.616 m | 0.418 |
| Full recipe | **0.221 m** | **0.715 m** | **1.381 m** | **0.657** |

The full model reduces 1-second and 2-second lateral errors by 68.0% and 70.3% relative to the earlier short-trained model. These are prediction metrics; driving outcomes are evaluated separately below.

The complete remote workflow took approximately **2 hr 32 min**, measured from the first generated track through evaluation/video/graph completion. This includes **16 min 25 sec** of dataset preparation (including tuning/restarts), **15 min 41 sec** of feature caching, **4 min 27 sec** of benchmark processes, the 78 min 22 sec of training, and evaluation/recovery gaps. Local video conversion and file transfer are additional.

## Driving results and video

The comparison uses eight held-out trackdrive tracks, each under light and heavy randomization: 16 runs per model.

| Metric | Earlier short training | Full recipe |
|---|---:|---:|
| Finished without disqualification | 3 / 16 (18.8%) | **15 / 16 (93.8%)** |
| Clean finishes (zero cones and off-course events) | 1 / 16 | **7 / 16** |
| Light-scene finishes | 3 / 8 | **8 / 8** |
| Heavy-scene finishes | 0 / 8 | **7 / 8** |
| Total cone hits | 55 | **42** |
| Off-course events | 25 | **6** |
| Acceleration passes | 2 / 2 | 2 / 2 |
| Skidpad passes | 0 / 2 | 0 / 2 |

The model is substantially better at completing trackdrive, but clean driving remains inconsistent. The failing heavy track had 12 cone hits and six off-course events. Skidpad is still unsuccessful and cone hits increased from 9/7 (light/heavy) to 18/32. Low cone counts in a failed baseline run can reflect its leaving the course early, so read cone counts alongside completion.

Full-model control decisions took approximately **7.05 ms** (median across the 20 per-run medians); the worst per-run p95 was **7.35 ms**. This includes preprocessing, synchronized GPU inference, decoding and control, excluding rendering. It fits comfortably within the simulator's 50 ms control period.

The predetermined video track produced a **59.7-second clean lap: zero cone hits and zero off-course events**. The earlier short-trained model's same-scene result was 52.8 seconds with four cone hits. The new lap is slower but clean. The 64.75-second video also includes the standing-start period.

These are a small, fixed simulated test suite, not an upstream leaderboard reproduction or a claim of real-world driving capability. Signed longitudinal position can become negative after driving backwards; reported completion fractions are bounded to [0, 1], with original `raw_progress` retained. Pass/fail, cone and off-course outcomes are unchanged.

## Recipe

- 480 regular trackdrive tracks and 160 tight tracks, using the upstream seeds and presets.
- Two random starts per track, additional standing starts, 40 skidpad episodes and 20 acceleration episodes: 1,438 planned episodes.
- Recovery excursions up to ±1 m (skidpad capped at ±0.6 m).
- The upstream light/heavy/off randomization mix, with venue probability 0.85 and all 20 optional sky panoramas installed. Venue scenes include textured asphalt, photographic skies, shadows and event scenery.
- Standard camera profile. This is not the separately published adaptation for a particular pitched real-world camera.
- Deterministic splits by track; fixed layouts split by episode.

| Phase | Steps | Batch | Trainable vision | Initialization / history |
|---|---:|---:|---|---|
| Initial head training | 6,000 | 256 | Frozen | Pretrained openpilot; base feature cache |
| Initial vision adaptation | 3,000 | 64 | Last stage | Best head-training checkpoint; base history |
| V2 full fine-tuning | 7,000 | 64 | All | Best initial-adaptation checkpoint; refreshed history |
| History-refresh fine-tuning | 4,000 | 64 | All | Best V2 checkpoint; refreshed history; lower LR |

The 7k + 4k phases follow the actual run described in the upstream roadmap. The initial 6k + 3k phases replace the unavailable M1 starting checkpoint, use larger Spark batches, and use the new venue dataset throughout. Config files record the exact learning rates and other settings. Optional FSOCO mixing is disabled, as in the checked-in defaults.

Each phase selects its checkpoint using validation loss, not test-track driving outcomes. Every validation uses the same deterministic set of up to 3,840 frames. The final open-loop evaluation recomputes temporal history from each evaluated checkpoint.

## Efficiency and instrumentation

The earlier verified static-shape and Zarr-handle fixes remain active. Training uses BF16 autocast on CUDA, channels-last images for stage two, fused AdamW, pinned host memory and cuDNN autotuning. Before V2 training, four matched 200-step measurements compare the unoptimized settings and optimized loaders with 4, 8 and 12 workers. The fastest measured optimized worker count is used for V2 and history refresh.

Dataset recording uses 16 single-threaded workers. Renderer memory grew across scenes in the initial long run; workers now recycle after 20 episodes, bounding retained scene allocations. Completed episodes survive restarts. The run also raises its inherited open-file limit to 32,768 because the full feature cache exceeds the container default of 1,024; stage-two loading now maps only the history features it actually uses.

One-second hardware telemetry records CPU/GPU busy, memory traffic busy, whole-system used memory, GPU power, temperature, clock and free disk space. Per-stage timing includes validation and checkpoint writes. Graphs distinguish that wall time from steady training throughput. GPU utilization is a busy-time measurement, not a percentage of theoretical peak FLOPs, and filling all unified memory is not an efficiency objective.

The completed bootstrap training encountered a native-library SIGSEGV during process shutdown, after all 3,000 steps, validation, checkpoint writes and timing output had completed. Both checkpoints were reloaded and verified; all newly trained tensors were finite, and the sole infinite scalar was the unchanged pretrained ONNX sentinel. The completed result was retained, with the shutdown failure recorded in the phase log. Subsequent isolated training workers explicitly stop their DataLoaders and exit after saving/flushing outputs to avoid native interpreter finalizers. A fresh short training/validation run verified this shutdown path. Empty validation visualization requests now return immediately instead of creating unused Rerun streams.

## Evaluation protocol

Open-loop comparison: untouched openpilot, the earlier 500+500-step checkpoint, and the final full-recipe checkpoint on the same complete held-out test split.

Closed-loop comparison: the earlier short-trained checkpoint and the full-recipe checkpoint on the same first four regular and first four tight tracks in the dataset's test split, plus acceleration and skidpad. Each runs in venue mode under light and heavy randomization, seed 0, no injected steering noise, one trackdrive lap and a 120-second simulated-time limit. All regular/tight track IDs are checked against the training split. Fixed layouts were seen during training, with different episodes/scenes.

The final video uses trackdrive seed 1013, light randomization, venue probability 1, seed 0: the same track and scene as the earlier short-trained venue clip, selected before examining the new model's outcomes. This track is absent from the new training set. Finishing without disqualification is distinct from a clean, zero-cone lap.

## Graphs and saved model

- [Training and validation curves](../runs/spark_full/graphs/training-curves.png) · [PDF](../runs/spark_full/graphs/training-curves.pdf)
- [Hardware utilization](../runs/spark_full/graphs/hardware-utilization.png) · [PDF](../runs/spark_full/graphs/hardware-utilization.pdf)
- [Throughput and cone F1](../runs/spark_full/graphs/throughput-and-cones.png)
- [Held-out prediction comparison](../runs/spark_full/graphs/held-out-comparison.png)
- [Driving comparison](../runs/spark_full/graphs/closed-loop-comparison.png)
- [Matched performance benchmarks](../runs/spark_full/graphs/performance-comparison.png)
- [Venue driving video](../artifacts/videos/spark-full-venue.mp4)
- [Final checkpoint](../runs/spark_full_refresh/ckpt/best.pt)
- [Machine-readable result record](../benchmarks/performance/spark-full-2026-11-08.json)

The final checkpoint SHA-256 is `3e5c848ec9e6fe2308255836e0209ff472390dcd54f92de0895e4e9708609dc0`. The downloaded copy matches the Spark copy. All model parameters are finite; the original ONNX model's infinite sentinel buffer remains unchanged.

These links refer to local, gitignored artifacts. PNG and PDF versions of all six graphs are available in the graph directory.

## Artifacts

- `runs/spark_full/`: dataset manifest, phase logs/timings, hardware telemetry, matched performance measurements, scalar exports, summaries and graphs.
- `runs/spark_full_{stage1,bootstrap,v2,refresh}/`: checkpoints, configurations, TensorBoard events and validation JSONL.
- `runs/spark_full/evaluation/`: open-loop comparison, closed-loop protocol/results and raw video.
- `configs/train/spark_full_*.yaml` and `scripts/*spark_full.py`: reproducible configuration and orchestration.

The completed dataset audit read every camera frame and verified finite trajectory labels and disjoint track splits:

| Split | Episodes | Frames | Distinct trackdrive tracks |
|---|---:|---:|---:|
| Train | 1,149 | 227,173 | 512 |
| Validation | 146 | 29,064 | 63 |
| Test | 143 | 28,157 | 65 |
| Total | 1,438 | 284,394 | 640 |

Exactly 1,100 episodes / 217,678 frames use venue scenes (about 76.5% of all frames). The configured 0.85 venue probability applies to randomized scenes; the 10% `off` randomization arm stays classic.

Raw data, weights and runtime output are intentionally gitignored. The complete dataset stays on the Spark.

## Validation and cleanup

The full dataset passed its frame/label/split audit; the completed training and evaluation processes produced all expected outputs; the final checkpoint loaded with finite parameters; repository-wide Ruff checks passed. The frame-storage/frame-pair regression checks and the short training/shutdown integration check passed. Graphs and the encoded H.264 video were visually inspected. The dedicated training container was stopped after artifacts were collected.
