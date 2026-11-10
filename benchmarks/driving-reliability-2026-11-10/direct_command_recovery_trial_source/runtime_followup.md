# Runtime and next-hypothesis notes

The frozen development pilot requires full saved-input CPU replay at the original 1e-5 thresholds. Original trial3 CUDA production failed that CPU gate only on unchanged plan/image diagnostics. Learned command checks matched. Full same-device CUDA replay passed. Do not present CUDA replay as CPU qualification.

The existing immutable evaluate_v2.py already supports --device cpu and records policy_device. A separately declared CPU deployment would permit same-runtime CPU evidence verification without widening a tolerance or changing weights. If a future candidate passes offline gates, freeze its deployment device before its first qualification pilot and keep that device fixed through all development and final runs. A runtime choice must be disclosed in comparisons, and every actual physical gate must still pass. Never spoof a replay device or discard failed CUDA/CPU receipts.

If the data-only recovery trial fails, inspect its first error before choosing the next change. A targeted measured-speed intervention is another available hypothesis because the original pilot shows weak braking before the missed turn. A fixed measurable speed controller would be a new execution contract, not a successful result for the rejected direct-command checkpoint. No speed intervention is frozen or launched by this note.
