# Independent Isaac parity artifact review

The goal is to show the custom simulator's diagnostics and matched tests in Isaac with frozen-model inference and honest outcomes. The final artifact audit passes after a documented repair of four Rerun footers. All 41 completed case audits and final media receipts bind to their actual files. The observed driving failures remain in the results.

This review independently checked the remote files after export and Rerun finalization. The visual content itself is outside this artifact review. The coordinating agent owns the gallery and video inspection.

## Measured outcomes

These counts come from all 41 saved summaries after checking each against the frozen case, suite status, and completion receipt. Pass and completion are separately counted. Legacy clean requires pass, zero geometric DOO, and zero off-course events. Physical clean requires pass, zero unique PhysX cone contacts, and zero off-course events. The main comparison contains the 20 skidpad and 16 retention cases. Ablations and the prior-failure diagnostic are separate.

| Group | Cases | Pass | Complete | Legacy clean | Physical clean | DOO | PhysX contacts | Off-course |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 36 | 31 | 31 | 12 | 9 | 144 | 159 | 37 |
| Skidpad | 20 | 16 | 16 | 3 | 3 | 97 | 106 | 17 |
| Retention | 16 | 15 | 15 | 9 | 6 | 47 | 53 | 20 |
| Command ablation | 4 | 0 | 0 | 0 | 0 | 6 | 8 | 3 |
| Prior failure | 1 | 0 | 0 | 0 | 0 | 9 | 9 | 0 |

Skidpad results contain 16 finished, one DNF, two invalid routes, and one timeout. Retention contains 15 finished and one timeout. All four ablations and the prior-failure case ended as invalid routes. All 41 cases report technical success and return code zero. Technical success does not mean the car passed.

The frozen historical main reference contains 36 passes, 36 completions, 11 legacy-clean cases, 86 DOO, and three off-course events. The historical and Isaac environments differ in rendering, physics, platform, and trajectories. This comparison does not isolate physics or establish a training curve.

## Frozen inputs and receipts

The 41 unique ordered case IDs agree across the manifest, protocol, and suite status. Manifest and protocol canonical hashes match their contents. Every summary and evidence metadata document matches its case and binds the manifest, protocol, checkpoint, runtime source inventory, and graph hashes. All 41 completion receipts bind the actual summary, metadata, frame manifest, and recorded frame count. All summaries report `frozen_unchanged` and exclude pilot-only runs. The contact-event lists have the recorded number of distinct cone IDs. All 41 summary and metadata model-state hashes match the pilot state hash `1921a60567422c86df551cd2a89a69170af38cc5c390bdd3d4cd049be8a9c50a`.

The current 108-file runtime source inventory, 76 original baseline source hashes, and both executable graph hashes match the protocol. The actual container checkpoint hashes to `cf3a8d253dfd5a417e1967d2d6dc8cdd0263e3d16d3f1f02f15495f490a83b7e`. The vision graph hashes to `bac14fd2d539b10eabd966b462d0dcba4fcaab6b96bc49cfc4a8703f7450531c`. The policy graph hashes to `a27088e94f581de472f9610183e3979987cf59a1b272816a339bb9757d00a17a`.

The vehicle, camera, and inference gate receipts match the protocol and record a pass. All 122 pilot evidence hashes match. The 120-frame pilot reports exact saved-output replay and unchanged outputs with the diagnostic hook disabled. A black-image replay changed the predicted plan on all 120 pilot frames. GPU inference replay covered the pilot only, not the 41 complete runs.

The archived historical provenance and each referenced row match their hashes and case records. The original local training checkout is clean at `e59014d6060de30b40eeba5f810cc797b27fe59c`.

The CPU export runs after scoring. Its present container image ID is `sha256:7092a76a990f364ea684b64a48681e9a7c1992a8d4af2edfc758e640b57668bb`. The export environment has Python 3.11.13, NumPy 2.4.6, torch 2.9.0+cu130, OpenCV 4.11.0.86, rerun-sdk 0.42.0, and FFmpeg 6.1.1-3ubuntu5. These versions describe the post-scoring environment and do not establish the package state during the driving runs.

## Export and recording verification

All 41 postprocess records are complete, and all 41 artifact audits pass with no mismatches. Each artifact audit binds the actual summary, metadata, frame manifest, completion receipt, and auditor source hash. Each video audit binds its artifact audit, completion receipt, and the executed producer hash. The executed producer is preserved at `docs/isaac-parity-results/provenance/export_producer.py`, with SHA-256 `ff960a135ef5e611424fa65c8f6cfcf49c5ec902ec7b7e4562ee67c62105ff60`.

Independent `ffprobe -count_frames` checks decoded all 82 final MP4s. Every video is 1280 by 768 pixels at 20 frames per second and contains the recorded inference frames plus 40 terminal frames. After finalization, all 82 final MP4 hashes and all 82 source MP4 hashes match their video audits. All 41 final Rerun recording hashes match the video audits and the finalization receipt. An independent `rerun rrd verify --check-footers=true` passed for every final recording.

The original producer hashed Rerun files before closing them. The first independent check found stale recording hashes, and the initial finalizer discovered four recordings with missing footers. The team preserved every original video audit and repaired the four incomplete footers with the Rerun migration command. Each original incomplete file remains as `debug.backup.rrd`. For all four pairs, an independent `rerun rrd compare --unordered` passed. This comparison ignores log time and row order. The repair and finalization receipts bind the original audit, preserved backup, repaired file, source, and verification results. Final video audits differ from the original audits only in the updated Rerun hash and explicit finalization lineage. In each repaired recording, independent `rerun rrd print -vvv --entity` checks found exactly one road and one wide image row for every inference-frame index and a terminal row at the next index. The frame counts were 1,271 and 1,705 for retention 030061 light and heavy, 269 for the heavy-swapped ablation, and 288 for the prior failure. This checks row coverage, not pixel equality with the NPZ inputs.

The finalization receipt hashes to `3219ec461023a59abf3f8e2b75b41f7e62afedaceea81b1c49a65ce6ab511f1a`. The footer-repair receipt hashes to `b51e7a954e55401a7f657e38fda3d71f31c5c7d5127338bd0f1ee392bfaa33ad`. The finalizer source hashes to `ba61e4390e212a314ce408a3a4e12068581105d3c9772c06d42bb64eb026c06f`, and the preserved repair source hashes to `63c8e036233b98556a12277db25125719da65185d3176ee8900ed81133e6e74f`.

## Limits

CPU artifact audits replay stored states through the original mission, route commands, and referee. They do not independently rerun inference on the full suite, resimulate PhysX contacts, or reconstruct the final physics transition. A rollover outcome would lack an independent final-body-pose check. None of the 41 recorded outcomes is a rollover. The unique contact count is consistent with the saved contact-event list but is not a second physical simulation. Bulk NPZ and Rerun files remain on the Spark, with selected cases also copied locally.

Prove It Works directed this review to the saved files, actual checkpoint, and completion receipts. Explain the Number directed the separate case-group denominators and the distinction between technical execution and driving outcomes.
