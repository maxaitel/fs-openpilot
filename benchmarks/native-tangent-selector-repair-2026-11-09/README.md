# Selector runtime identity repairs

The first tangent run completed all 500 updates before selection. The original selector failed before writing a result because the configured recovery recipe used a relative path while the sampler recorded the absolute path of the same file. After resolving both paths, it exposed a second incorrect assumption: there were 50 trainable policy tensors, while 42 described tensors changed in earlier runs.

The repaired selector compares canonical paths and derives the exact ordered trainable policy inventory from the loaded initializer graph. It retains complete-state signatures, all frozen/non-trainable tensor comparisons, dataset and calibration identities, numerical limits and ranking rules. Ten focused tests passed, including a different-file rejection and a graph-inventory mismatch. Both repairs received independent code review.

The actual repaired selector completed successfully and rejected every saved checkpoint. Its fallback to step zero is explicitly ineligible. No numerical threshold or checkpoint-selection rule was relaxed. The pre-update source and both failure logs are preserved here; the amendment records their hashes. Full selection receipts remain in `runs/lab_v4/tangent_trial1/selection_receipt.json` locally and on the Spark.
