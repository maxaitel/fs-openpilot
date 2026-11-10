# TRAIN image detector recording protocol

This recorder supplies predicted cones for later detector quality and planner feasibility analysis. It does not qualify a driving candidate. It reads existing TRAIN recordings and runs no simulation or training. The frozen final suite stays unopened.

Run from the project root in the Spark container. This CPU-only recorder disables CUDA. The parent may schedule it while the existing native diagnostic uses PhysX; any resulting CPU contention disqualifies isolated timing comparisons, but does not change the saved input data. No timing or speedup result is an output of this probe. Use a new output path for each attempt. The output directory must not exist.

```sh
PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/vision_corridor_probe_v1/record_detector.py \
  --manifest data/native/lab_v7/manifest.json \
  --checkpoint probes/pi_models/latest_pi_train.pt \
  --out .work/lab/vision_corridor_probe_v1/results_clean_v1
```

To include both verified recovery continuations in a separate attempt, add `--recovery-root runs/direct_recovery_v1`. The recorder requires the passed `skidpad-receipt-v2.json` and `trackdrive-receipt-v2.json` beside those two episode directories. It checks the fixed verifier hash, metadata hashes, and receipt array inventory before using either continuation. Those continuations used an earlier controller before expert takeover, so report them separately from the clean episodes.

The manifest SHA-256 is `db7150b60fdf50e861c6ca003ae844b90ff398fd7b244c9201b7de9d3af5b309`. The original checkpoint SHA-256 is `48db741c8a686b2e635ae7a81b4e4c08d79fbfbd9f4dc4c163260da6f93f0cfb`. The default selection is exactly `train-skidpad-82000-matched` and `train-trackdrive-82000-matched`. The recorder validates the manifest, metadata identity, split, seed, native array hashes, and completed zero-contact summaries. It requires both prepared FP32 graphs to exist and refuses the graph-fetch path.

For each stored 5 Hz frame, including the last stored frame and those without a future training horizon, the recorder concatenates the preceding and current packed camera frames. Frame zero repeats itself. It uses the road and wide pairs from the same index and casts the packed uint8 values to FP32, as `NativeEpisode.frame_pair` and `ModelPolicy` do. It forwards the full vision encoder and image head on CPU with four torch threads. It runs no policy, BEV head, future targets, or feature cache. The decoder uses the episode's runtime `T_ego_cam`, the same calibration passed to `bev_from_image_head` in `ModelPolicy`. The calibration is an existing camera parameter, not a cone position or vehicle pose. The decoder retains its default score threshold 0.25, per-class merge distance 0.6 m, and camera range limits.

`frames.jsonl` contains one row per stored frame. Each row has the episode ID, frame index, native 20 Hz tick, timestamp, measured vx, vy and yaw rate, saved 25 by 8 desire queue, model-input pair hashes, six raw image-head hashes, and decoded ego-frame cone positions, classes, and scores. A hash covers the array dtype, shape, and contiguous bytes. The first eight frames, every 50th frame, and the last frame are inferred twice and their raw heads and decoded outputs must match exactly within the same process.

`references.json` records array paths, shapes, dtypes, and camera calibration. Ground-truth cone, contact, and state paths are references for later scoring. They are not detector inputs. The full native state array also preserves the other 20 Hz phases for later alignment. The predicted cones and desire queues exist only at saved phase-zero 5 Hz ticks. The last saved image is not necessarily at the final 20 Hz state tick, so do not infer an unsaved terminal image.

`receipt.json` reports success or failure, selected episode IDs, frame counts, output hashes, runtime versions, and SHA-256 hashes taken before and after processing for the recorder, protocol, repository Python source, prepared graph files, checkpoint, manifest, metadata, receipt, and metadata-listed input arrays. A mismatch fails the run. Third-party package versions are recorded, but their installed library bytes and the operating system are not pinned. A failed or interrupted attempt keeps its output directory and receipt. An abruptly killed process may leave a `starting` or `running` receipt, which is not a passing result.

Check `receipt.json` for `status: passed` before using the data. Check that each frame count equals its referenced ticks array length, and keep the receipt and references with the output. The repeat check establishes exact repeatability only on the sampled frames in that process. It does not establish cross-host reproducibility or detector accuracy.

The local alignment check uses a small fixture and runs no model.

```sh
python3 .work/lab/vision_corridor_probe_v1/record_detector.py --self-test
```
