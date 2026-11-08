# Frozen Isaac parity report

Generate the final report after downloading all suite artifacts and videos:

```sh
python3 scripts/report_isaac_parity.py --run isaac_eval/runs/parity_v1 --out isaac_eval/runs/parity_v1_report
```

Add `--partial` only for monitoring. The command otherwise refuses to report a complete suite unless all 41 cases have successful suite status, matching summary provenance, frame counts, telemetry and completion receipts. It retains every expected case, including missing, technical failure and invalid cases. A case is complete only when the evidence receipt verifies the summary and the video audit binds that receipt and both video hashes, frame counts, and frame rates. Video links appear when the files are present.

When bulk evidence stays on the Spark, add `--remote-evidence-root max-spark:/home/maxaitel/fsop-isaac-debug-parity/isaac_eval/runs/parity_v1`. This mode requires the full-evidence audit produced on the Spark, bound to the copied metadata, manifest, completion receipt and summary. The video audit also binds that artifact audit. The local report verifies copied videos but does not claim that raw NPZ frames or Rerun files were downloaded. Without this flag, the completion verifier additionally requires the local frame-file inventory.

Run `scripts/process_isaac_parity.py` on the Spark after the measured suite finishes. It independently replays every stored state through the original mission, route commands and referee, verifies frame hashes, and produces both videos and a Rerun recording. `scripts/collect_isaac_parity.py` copies completed videos, diagnostic tables and audit receipts without the bulk NPZ, Rerun or scene-asset files. `scripts/build_parity_gallery.py` builds a local interactive index from the report.

The script verifies the manifest content hash, protocol and ordered case list, archived reference provenance, result file hashes and indexed rows. The report records the checkpoint, protocol, manifest, source hashes and backend differences. It uses the frozen manifest's historical reference row for each case. The skidpad and retention sets are reported separately from command ablations and the previous failure.

Pass is the referee's pass field. Clean requires pass, zero legacy DOO and zero OC. Physical cone hits use PhysX contact data and are shown separately from legacy DOO. Per-frame timings are pooled over all recorded frames, not averaged across cases. Their median and p95 are in milliseconds. Telemetry median and p95 pool available one-second samples from verified cases, so longer cases contribute more samples. P95 is linearly interpolated at rank 0.95 times (n - 1). GPU utilization, power, and temperature are device-wide. CPU and host RAM describe the host; process RSS describes the evaluator process. Step wall time includes the component timings. Run wall excludes startup; their sum excludes orchestration and offline video export. Real-time factor is total simulated seconds divided by total run wall seconds across verified cases. GPU traces are per-case samples overlaid relative to each case's run start, not a continuous suite timeline. Missing results are never counted as passes or silently removed from the expected count.

The historical custom simulator and new Isaac evaluation share the frozen checkpoint, cases and matching contract, but differ in renderer, physics, scenery responses, platform and resulting trajectories. Historical initialization is reconstructed. The comparison cannot isolate a physics-only effect or establish a training curve.

The raw vision view decodes image-head boxes at the original 0.25 threshold. Class IDs are 0 for blue, 1 for yellow, 2 for orange and 3 for large orange. Scores are uncalibrated model confidences. This auxiliary head does not directly control the car; the predicted path feeds the existing path follower.
