# Diagnose the failed cone-set head without parameter updates

This frozen diagnostic addresses the failed fixed32 TRAIN wiring gate for the zero-contact vision FormulaStudent task. It does not qualify a model for driving. The final suite remains unopened.

The data shape is one ordered batch of 32 TRAIN anchors. Each anchor contains 200 masked eight-field cone tokens, the unchanged 25 by 8 route history, three body-motion values, and the same-tick two-value teacher target. `bindings.json` fixes every anchor, target, and failed-audit input hash before execution.

## Frozen comparison

Recreate the original 47,618-parameter head with seed zero and verify all eight initial tensor hashes against the failed audit. Verify all 417 frozen base tensor hashes. Generate the same ordered 32 inputs afresh with the frozen data owner, model, and decoder. Check the original collision hashes and targets exactly. Repeat the fresh batch and require exact inputs and outputs. Write the verified input binding before paired measurements.

Compare exactly two forward definitions using the same parameters. The baseline is the original concatenation. The candidate applies `layer_norm(mean_and_max, (128,), weight=None, bias=None, eps=1e-5)` before appending count, all route fields, and motion. No parameter, loss, width, learning rate, or input changes are permitted. The baseline instrumentation must match the original head bit for bit.

There is no optimizer, backward accumulation, parameter update, analytic fit, selected checkpoint, or exported weight file. `autograd.grad` computes diagnostics without assigning parameter gradients. The existing failed weights were never saved and this probe does not reconstruct them.

## Paired measurements

Record pooled feature common offset and across-anchor variation. Record per-feature means and standard deviations, per-anchor means and standard deviations, and complete singular spectra of raw pooling, normalized pooling, combined inputs, and readout features. The last-layer design is the 32 by 129 matrix formed by the 128 hidden features and a bias column.

All spectra use float64 SVD on the observed float32 matrices. Numerical rank uses `max(rows, columns) * float32_epsilon * largest_singular_value`. Report the untruncated condition number and the condition number of resolved singular directions. Centering necessarily limits across-anchor rank to 31. No ridge, truncation sweep, or analytic regression is performed.

The physical geometry axes are an ego lateral shift in metres and an ego yaw rotation in radians. Shift every cone y by minus lateral displacement, then rotate every cone xy by minus yaw. Differentiate the range channel consistently with the lateral shift. Rotation leaves the range channel unchanged. At zero perturbation the original token values are preserved. These transformations leave class, score, count, route, and body motion unchanged.

Two batched output-sum gradient calls produce both output rows of the per-anchor physical geometry and body-motion Jacobians. Batch independence makes these exact per-anchor derivatives. Report the full two by two geometry rows and two by three motion rows. Also report their row norms, geometry sensitivity relative to the constant output-bias derivatives, and local hidden-feature Jacobian spectra for both physical geometry and motion.

The Smooth L1 loss gradients to token coordinates, each original parameter tensor, and body motion are auxiliary measurements. Independent token-coordinate gradients are not the physical decision metric. Report them separately to avoid mistaking an independently perturbed range channel for physical geometry.

Record all 496 pairwise distances in pooled, combined, and hidden spaces. Include route, scaled body-motion, count, and target differences. The pooled spaces are invariant to cone ordering. These measurements help distinguish compression from small raw feature scale. They do not prove inconsistent supervision.

## Decision fixed before inference

A candidate is eligible only for a separately frozen wiring trial when either of these material-effect conditions holds.

- Both outputs have at least a threefold median gain across anchors in physical geometry Jacobian row norm relative to their output-bias derivatives.
- The untruncated readout-design condition number improves at least tenfold, with no measured rank reduction.

Every eligibility decision also requires all of the following.

- Each output retains at least 90 percent of its original geometry row norm at every anchor.
- Each output retains at least 90 percent of its original motion row norm at every anchor and in the median.
- Every anchor retains its measured output and hidden-feature geometry and motion ranks.
- The readout design retains its numerical rank.
- No previously separated pair newly collapses in pooled, combined, or hidden space. Collapse means distance at most one millionth of that space's across-anchor RMS, with a floor of `1e-12` on the RMS denominator.

Threefold sensitivity and tenfold condition changes are deliberately large structural effects. They are much larger than the required zero observed repeat noise. The 90 percent limits reject material loss of the body-motion or geometry paths while allowing small changes from the nonlinear hidden activation. These thresholds are diagnostic design choices, not statistical claims or calibrated predictors of successful fitting.

Layer normalization removes each row's common offset and nearly removes its scale. Even unchanged local ranks and all 496 pair separations cannot establish global information preservation. A passing result permits only consideration of one versioned LN wiring trial. It does not establish learnability, generalization, visual dependence, or zero-contact driving. A failing or ambiguous result does not justify LN. If raw input distinctions disappear in pooled features, investigate the representation separately. If conditioning is adequate but teacher differences remain unexplained, inspect observation sufficiency and label semantics separately. Neither path authorizes a new fit here.

## Integrity and execution

`freeze.json` binds the script, this protocol, the threshold and anchor bindings, and byte-for-byte copies of the five failed receipts. The caller supplies its SHA256 on the command line. Verify the failed run's live receipts, successful audit and conditioning completion markers, failed overfit marker, original source inventory, source archive, recipe, and inherited audit independently. Do not invoke `checked_audit`, which correctly rejects failed attempts for training.

The prelaunch source archive SHA256 is `ed81aab2154a032d5a9c8b857b5d6b54709367e39626f174c9740d23e8444975`. The failed audit's own source archive SHA256 is `74b0a79176e90f3cbf6beedf3a9452077423ac5cac2c7068237a37c5ea85eff7`. These are distinct archives of the same source inventory. The probe binds the prelaunch freeze record and verifies equality of that inventory. It verifies all members of the live audit archive. It does not require the prelaunch ZIP at a new remote path.

Use CUDA as in the failed audit, FP32, four CPU threads, and TF32 disabled. The parent owns deployment and launch. The output directory must be new. Example command from `/scratch/fsop/head_trial_work` follows.

```sh
/isaac-sim/python.sh .work/lab/cone_set_conditioning_probe_v1/probe.py --out runs/cone_set_conditioning_probe_v1 --freeze-sha256 FROZEN_SHA256
```

After the original episode pool closes, reopen the frozen data owner for independent current verification. Recheck source and receipt hashes before writing the final report and completion marker. Any failure leaves an exclusive failure receipt and prevents a completion marker. No failed run is resumed. No simulator, main fit, or final case is opened.

## Existing failure evidence

The copied overfit receipt reports MAE `0.005447729` steer and `0.314533442` acceleration after 200 updates. Original-anchor MAE is `0.004556329` and `0.342165085`. New skidpad recovery MAE is `0.001469441` and `0.070771261`. New trackdrive recovery MAE is `0.011208817` and `0.503032360`. Every cone set is nonempty. Exact input collisions impose zero MAE lower bound on this batch.

The largest acceleration errors are `2.651730776` on original trackdrive frame 141, `1.653119326` on recovery trackdrive frame 45, and `1.602125645` on original trackdrive frame 174. The old nearest-input metric often equals one because it combines one-hot classes, masks, and unscaled motion with geometry. It does not establish near-identical geometry or contradictory labels. `existing_failure_summary.json` retains the receipt-derived details.

The throughput checkpoint is to freeze this one executable and protocol, run it once under parent ownership, and inspect the paired report before any versioned architecture change. No further candidate search is part of this probe.
