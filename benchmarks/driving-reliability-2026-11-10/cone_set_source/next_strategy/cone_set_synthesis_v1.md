# Cone-set representation experiment

The goal remains full trackdrive and R,R,L,L skidpad completion with zero physical contacts. No model qualifies, and the frozen final suite remains unopened.

The independent deterministic corridor and learned local-corridor designs exposed a choice between explicit geometric control and learned route shape. The deterministic first prototype never reached a path on 706 recorded frames. Its nearfield and two-sided bootstrap rules failed. That result does not condemn geometry or show that learned association alone would repair the same rules. The independent direct cone-set design offers a smaller representation experiment while retaining the existing command target and gates.

Select the direct cone-set actuator design first. Its new 47618-parameter head reads individual predicted cone positions, range, class and score, full supplied route history, and measured vx, vy and yaw rate. It does not read the original policy's compressed activation. Keep all 417 original base states exactly unchanged. Preserve original plan and cone diagnostics. The learned eight-point corridor remains an alternative if the direct contract fails a concrete information or physical test. No training has run for either candidate.

The data shape is FP32 tokens `[B,200,8]`, Boolean mask `[B,200]`, full desire `[B,25,8]`, and motion `[B,3]`. The head has four affine layers and eight state tensors. Its shared 8-to-64-to-64 token encoder pools masked mean and maximum, appends normalized count, desire and motion, then uses 332-to-128-to-2 output layers. Empty sets have zero pooled features. Token validity, no truncation, normalization and raw-to-applied actuation belong to one core module.

Keep the exact original 7430 TRAIN and 2067 VAL anchors, same-tick actual teacher controls, sampler, AdamW recipe, 500 updates, checkpoint schedule and raw-command gates. No feature cache, AMP, TF32 or channels-last path. All sampled detector outputs are freshly recomputed. The 573 recorded detector frames are design evidence, never a training cache. Freeze fixed-32 collision analysis, one disposable update, disposable wiring, full-source audit and all training stages before optimization. Discard conditioning and wiring weights. Require independent postflight in a fresh child of the live trainer, with one owned native episode pool and explicit closure.

Parent clarification before implementation changes one proposed cross-device boundary. Training-device versus CPU detector and command differences are measured diagnostics, because GPU and CPU can produce different FP32 values near discrete detector thresholds. They are not saved-input replay. Do not claim cross-device identity or relax CPU replay tolerances. Qualification requires every original offline group to pass, followed by a fresh complete 2067-anchor CPU validation of the single selected package at the identical thresholds. A CPU validation failure rejects that selected package without substituting another checkpoint. Exact CPU producer/replay remains required at rtol=atol=1e-5, including discrete identities. CPU repeated inference must remain exact within the same source contract. The existing GPU validation gate is preserved; CPU validation adds an actual deployment check.

Batched decoder implementation must match the stock singleton decoder on identical raw image-head tensors. This separates decoder semantics from encoder floating-point differences. Use the pinned stock threshold, class set, projection, range and merge rules. A new tie rule cannot silently change the pinned decoder. Keep its deterministic order and measure actual discrepancies.

Only a package passing offline, complete CPU validation, full-image visual dependence and runtime gates may enter physical qualification. If no package passes offline, stop this fit with no physical run. A separately proposed causal diagnostic needs its own frozen question and exposure budget. This is stricter than spending another rejected-model pair. Existing six exposed qualification runs, eight validation-driving cases and all final acceptance predicates remain required for promotion.

## Execution checklist

- [x] Ground the current representation, targets, perception and actual sampling.
- [x] Sketch independent geometric, learned-corridor and direct cone-set designs.
- [x] Agree on the bounded representation change and cross-device distinction.
- [ ] Implement core and training in one isolated directory.
- [ ] Implement producer, replay, physical auditor and video adaptations against the same core contract.
- [ ] Independent source and comment review, focused tests and complete source freeze.
- [ ] Real Spark input/decoder audit, conditioning and discarded fixed-32 wiring.
- [ ] Fixed main fit and independent postflight.
- [ ] CPU validation, visual dependence, runtime and native pilot if eligible.
- [ ] Physical development and validation gates before any final candidate binding.

Parallel implementation partitions core/training and native evidence pipeline. The core public API is `ConeSetHead`, `ConeSetPolicy.load(path,device)`, `reset`, `__call__`, `pack_cones`, `decode_batch`, `KIND`, `CONTRACT`, `digest_state` and `model_digest`. It owns all numerical and package semantics. The pipeline imports that core and owns recording and independent audit only. Native schema 6 distinguishes this representation from joint schema 5. No worker launches a GPU job, training or scene. Parent owns all launches, timing, data transport and experiment decisions.

Throughput is measured end to end, including fresh encoder/image head, device transfers, decoder, projection, merge, head backward and all stage bookkeeping. A faster image-only training forward would perform different work from the historical full-history policy fit. Do not call that an equivalent-work speedup or FLOPS efficiency. Keep native and training GPU jobs serialized. The configured design/review models share one model family, so there is no independent model-family judge.
