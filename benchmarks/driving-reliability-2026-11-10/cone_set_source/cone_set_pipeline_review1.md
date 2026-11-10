# Cone-set native pipeline review 1

The goal is full trackdrive and R,R,L,L skidpad completion within the frozen mission limits, with zero physical contacts. This independent read-only review covers the eight pipeline files listed below. It does not qualify a model or authorize an episode.

## Findings

One P2 finding. No P1 finding. No concrete numerical replay, physical-audit, or portable-transport defect was found in the reviewed source.

### P2. Render the saved cone count and raw-to-applied command trace

Location is `.work/lab/cone_set_actuator_trial/export_audited_command.py`, lines 87 to 89, with the only new annotation at lines 79 to 81.

The wrapper changes only `_writer` and adds a fixed banner before invoking the unchanged renderer. The inherited `research_frame` in `isaac_eval/render_debug.py`, lines 86 to 104, shows `record['action']` and the current route command. It never reads the cone-set controller's `raw`, `bounded`, `mission_override`, or `mask`. The inherited debug composition also does not show those controller fields. The selected design's `cone_set_actuator_design_v1.md`, line 70, explicitly requires fresh cone counts, raw and applied commands, and mission state alongside the diagnostic original plan.

Trigger this on an episode frame where clipping or the AS layer changes the request, or where the valid cone set becomes empty. Both exported views omit the distinction between the head's request and the final action, and omit the actual set count. A reviewer cannot tell from the video whether the apparent acceleration came from the learned head or the launch override. The NPZ evidence remains correct, so this is an evidence-display defect rather than an incorrect control or false numerical pass.

A CPU probe called the real `research_frame` with a saved controller containing `raw=[9.0,-5.0]`, `bounded=[0.4,-3.0]`, `applied=[0.4,2.0]`, a launch override, and two true mask entries. Capturing `cv2.putText` arguments showed only `action steer / accel: [0.4, 2.0]`. The raw request, bounds, override, and count were absent. The wrapper's fixed banner cannot supply these per-frame fields. Add the annotations through the isolated export adapter so the original renderer stays frozen, and test the resulting text for an overridden frame and an empty set.

## Verified behavior

- Ran `PYTHONPATH=src:. .venv/bin/python .work/lab/cone_set_actuator_trial/test_pipeline.py` from the correction worktree. All eight tests passed. The actual ffmpeg fixture encoded both views and fully decoded both raw and H.264 outputs, with 41 frames at 1280 by 768 and 20 Hz for each view. Those 41 frames include the declared 40 terminal frames. A second encode rejected existing outputs.
- Compared the complete AST of every `checks.compare` call in old and new `process.py`. Both contain 72 calls. Seventy-one calls are identical. The only changed call is `controller.proprio`, at old line 237 and new line 238. It explicitly casts the saved list to FP32, as the shared contract requires. The physical, referee, route, terminal, contact, source, and receipt checks retain their existing arguments.
- Exercised the replay comparator directly. A class-ID change and a Boolean mask change each produce a mismatch. A float32 perturbation of 0.000005 at 1.0 passes the unchanged absolute and relative 0.00001 tolerance.
- Exercised the current `decode_batch` adapter on three identical-input batch slices against separate stock singleton `bev_from_image_head` calls. Coordinates, classes, and scores matched exactly for all three examples. The examples produced 15, 14, and 14 merged cones. This is a synthetic decoder-boundary check, not a measured detector or training-device parity result.
- Reconstructed 12 sequential desire updates over all four phases. Pooling the post-call queue matched `TemporalState.push_desire` exactly and retained both an earlier route pulse and a later pulse. The source replay constructs its own temporal object and checks the full 25 by 8 queue.
- Instantiated the current head on CPU and counted 47,618 parameters in eight state tensors. The producer and replay require 417 base states, eight head states, and the specified parameter totals. The loader calls the core package validator, which binds actual base bytes to the trusted original digest. The core was read for API compatibility only and remains outside this review's implementation scope.
- Called the producer source inventory. All 176 entries resolved locally and were relative paths. Replay requires the exact inventory and exact archive membership, rejects duplicate archive entries, and rechecks executable source and graph hashes. Portable transport maps all replay-bound episode files to the local episode, binds the exact package hash and size, checks archive contents against the recorded source inventory, and explicitly labels graph identities as producer replay bindings without local graph reexecution.
- Source inspection confirms fresh CPU model calls on every saved frame, full raw image-head and plan comparison, saved-head decoding with saved nominal calibration, repacking and comparison of saved cone inputs, and head reconstruction from declared FP32 and Boolean saved inputs. It also confirms raw bounds, the AS override, final bounds, and recorded action comparison. The inherited physical auditor does not mistake its consistency pass for zero-contact qualification. The notes explicitly preserve full mission completion, stopping, zero contacts, zero off-course events, and the 120-second limit.

## Validation limits

No GPU, simulator, training, SSH, final case, or new physical episode was run. No source file was changed. This report is the only review artifact added.

There is no local cone-set package or sealed cone-set native episode. Therefore the complete fresh CPU saved-input replay and the real saved-TRAIN runtime preflight were not executed. The local eight-test suite uses synthetic receipts for transport and does not exercise complete model replay, physical scoring, or model qualification. The source review establishes that those paths are wired, not that a future selected package passes them.

The full original 417-state weight digest was not recomputed from a real package during this review. Package rejection and state invariants were inspected through the current core API. Core training implementation, its in-progress tests, and missing or pending training artifacts were not reviewed as defects. The inherited stock renderer's terminal `pass` field remains the legacy referee result, not a zero-contact eligibility result.

Prove It Works shaped the review by requiring actual CPU tests, actual encoder and decoder execution, AST argument comparison, and a real renderer text probe. Unslop and Technical Writing shaped the report. The requested P1 and P2 correctness scope takes precedence over style-only comment changes.

## Source SHA-256 values

The following hashes bind the reviewed pipeline, baseline comparisons, and read-only supporting sources. The core entry records the API snapshot read for compatibility, not approval of its training implementation.

Recorded at 2026-11-10T09:36:52.338129+00:00.

```json
{
  ".work/lab/cone_set_actuator_trial/evaluate.py": "82f9b90d52da591713d44ed30976ccdbf32a5fe3a81b6ee86d9064ecb63070a4",
  ".work/lab/cone_set_actuator_trial/verify_command_episode.py": "74c02f32d9c5c794e39291f968cac7355114e8235d24431dbb43d492c44df494",
  ".work/lab/cone_set_actuator_trial/process.py": "51dd7ac4d1d0d122cf77650043b35718cfe45c21cda99fdfefe47f28c5977f75",
  ".work/lab/cone_set_actuator_trial/export_audited_command.py": "9c14f5dd4be8b086edb675c944ca99b28f12fcb7c692a441cfef3c108d57c18e",
  ".work/lab/cone_set_actuator_trial/encode_portable.py": "bbffef7f10e701c75f9df03b14f9886492ce0bcd5b00f142480b500a15c0fdbc",
  ".work/lab/cone_set_actuator_trial/runtime_preflight.py": "ec99d0bfa9aceb88ab4e3b50849133409233eac8f1c226dc5fef53c7a54612f1",
  ".work/lab/cone_set_actuator_trial/test_pipeline.py": "c10eeb5ecd9022f39b5c142e3d0e78ef4aa625d09f070b15734545b508c58561",
  ".work/lab/cone_set_actuator_trial/pipeline_notes.md": "8cf512c5dfad7059ff2ebcb041a9649d5a71c5371185e755d02b00b386e92d70",
  ".work/lab/joint_command_trial/evaluate.py": "e5158eb985d895f024035abb77eea96530370c13bdb8c4a48d1204fadc7497e7",
  ".work/lab/joint_command_trial/verify_command_episode.py": "96a3afa8c236a884dbbd613a604fed769015f809328a940733a3158077e9a91e",
  ".work/lab/joint_command_trial/process.py": "09ba5aa3a89cf10b42eeda29496224bb251a3c15fb3603e4266fea3215fc8ce9",
  ".work/lab/joint_command_trial/export_audited_command.py": "95816dfd6cd03b94a47522c4b31114322ce220d9380c4ed8b8822d4957421502",
  ".work/lab/joint_command_trial/encode_portable.py": "573fb2a3f8dab69ab9fa1fe51726f11b1bc98e8f399f4c558f3ff983adaa5f42",
  ".work/lab/joint_command_trial/test_pipeline.py": "05e6983925884b1622bf4fc7ed5d4c0db7ef749ec1ea4d58d8c258f995eb3f89",
  ".work/lab/next_strategy/cone_set_synthesis_v1.md": "4812acd4f64dcc08c96ebfd9a28afbca396116914e7e8060837fa83df5a3d74e",
  ".work/lab/next_strategy/cone_set_interface_v1.md": "bccb01f1f520d59ad9ff23d055944debfeaeb81d7bae2e99d6f1e7713d7494cd",
  ".work/lab/next_strategy/cone_set_actuator_design_v1.md": "a86c9df7ebd9bbe433779e193500ac05ac5a51fbea7ef87eaafe37789931bb3d",
  "isaac_eval/render_debug.py": "176fcbe153412eca84bfcff85d4c1d104809aba978b240ac8bd9ae47314a7ebb",
  "src/fsop/viz/video.py": "94013abd1f1e78a2869101f12aa3f53ab3c6accd86819069b78234ea9b696cd9",
  "isaac_lab_fsop/verify_inference.py": "98db4a9642eb60b82b1289560ab0cd61f236f4009418a59d108a0496d28a0ae8",
  "scripts/audit_isaac_episode.py": "5610cf1be7fd5a0f7e96516184fcc698b7298a0c87286b61a5762cf1f8eb3ec2",
  "src/fsop/model/temporal.py": "3b7ef159ae6a0f3f39dd950ae3943fb5f59f164af0d8e9ab329f6f5de876b8c7",
  "src/fsop/eval/img_to_bev.py": "bcf10ece06c575512c45ea773335a1024fbc7ab95d20426a91ff18bd71d2d2d1",
  "isaac_eval/evidence.py": "00a38b56c9f813821674dfcf65b1d9f5b1db679c67e851d191f06a55540ac23f",
  ".work/lab/cone_set_actuator_trial/cone_core.py": "7e73e45ba42443fc1fb028d886677e2345393ea4b8b8c5f3f79bd8ba7b1251fb"
}
```
