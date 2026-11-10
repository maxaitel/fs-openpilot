# Launch supervision audit

The goal is zero-contact native full-track and RRLL driving. This audit tests whether missing standing-start supervision explains the direct command heads failing early.

The dataset-wide absence hypothesis is false. The joint trial actually sampled all 72 eligible early low-speed targets, 480 times, and sampled standing tick-zero targets 143 times. Launch coverage is reduced for seven replaced TRAIN episodes and absent in both new recoveries. These facts do not establish dilution as the cause of physical failure.

The exact eligibility code requires a qualified episode, 200 future teacher controls, and zero contacts at the anchor and horizon. The direct head target is the same-tick executed control. The audit executed only the source AST suffix that computes anchors. Every manifest accepted count and rejection dictionary matched.

throughput checkpoint: n/a, read-only investigation

| Dataset and split | Episodes | Qualified | Tick-zero eligible | Eligible targets | t < 3 s | t < 3 s and vx < 2 |
|---|---:|---:|---:|---:|---:|---:|
| lab_v7/train/all | 32 | 31 | 24 | 7006 | 360 | 72 |
| lab_v7/val/all | 8 | 8 | 8 | 2067 | 120 | 24 |
| lab_v2/train/all | 32 | 31 | 31 | 7222 | 465 | 93 |
| lab_v2/val/all | 8 | 8 | 8 | 2067 | 120 | 24 |
| new_recovery/train/all | 2 | 2 | 0 | 424 | 0 | 0 |

V7 TRAIN collection modes are 5 learner_prefix, 3 phase_recovery_v1, and 24 absent. Absent means no collection_policy metadata. Teacher arrays confirm those 24 are teacher controlled throughout. One is unqualified. The later-phase recovery episode retains launch targets, producing 24 launch-covered TRAIN episodes total, 12 per discipline. All eight VAL episodes are pure teacher and retain their starts.

The ordinary joint pool has 7,330 targets, including all 360 launch targets. Its 96 of 128 batch draws imply a 3.684% launch draw probability and 0.737% early low-speed probability. Recorded training exposure was 2,349 of 64,000 early draws, 480 early low-speed draws, and 359 of 360 unique early targets seen. These are recorded sampling facts, not a diagnosis of inadequate learning.

The future filter excludes zero additional same-tick teacher targets at t < 3 s. In v7 TRAIN it excludes 1,596 otherwise qualified same-tick targets. These are 1,546 tail targets beyond the future horizon and 50 targets before the later learner interval of train-skidpad-82002-visual_stress_v1, ticks 212 through 408. Therefore relaxing future eligibility cannot restore the seven missing starts. Their actual controls were learner controls. No unexecuted teacher labels were inferred.

The new skidpad recovery first becomes eligible at tick 64, 3.2 s, vx 7.0624. Its learner interval is ticks 0 through 62. The new trackdrive recovery first becomes eligible at tick 164, 8.2 s, vx 1.9831. Its learner interval is ticks 0 through 163. Their oversampled recovery groups contain no t < 3 s target.

V2 has 31 qualified all-teacher TRAIN episodes. Both v2 and v7 exclude train-skidpad-82007-visual_stress_v1 because its summary records two physical cone hits despite finished/pass/complete. Seven qualified v2 originals can supply 105 early targets and 21 early low-speed targets missing from their current v7 counterparts. The candidate inventory records every qualified v2 TRAIN path, metadata hash, verified small-array hash, and recorded camera hash. No images were copied.

| V7 episode missing launch | First eligible tick | Teacher count in first 72 ticks | Qualified v2 anchors | V2 states |
|---|---:|---:|---:|---:|
| train-trackdrive-82000-matched | 68 | 6 | 235 | 1138 |
| train-trackdrive-82001-visual_stress_v1 | 76 | 0 | 204 | 1015 |
| train-trackdrive-82002-visual_stress_v1 | 124 | 0 | 234 | 1136 |
| train-trackdrive-82003-visual_stress_v1 | 164 | 0 | 301 | 1402 |
| train-skidpad-82000-matched | 72 | 0 | 232 | 1128 |
| train-skidpad-82001-visual_stress_v1 | 72 | 3 | 232 | 1128 |
| train-skidpad-82003-visual_stress_v1 | 92 | 0 | 232 | 1126 |

Boundaries and limitations. Both 40-episode manifests were hashed. All 80 episode metadata hashes matched their manifests. The two additional metadata hashes matched their stored qualification receipts. All 574 required small arrays matched metadata SHA-256. Camera files were opened with read-only mmap for shape and dtype headers only. Camera payload hashes and the historical collection source tree were not recomputed. This is a coverage audit, not fresh native physics qualification or image-integrity qualification. No model, GPU, simulation, training, source change, final case, or remote write was used.

Artifacts. results.json contains all 82 per-episode state/control observations up to five seconds, teacher run boundaries, masks, eligibility counts, modes, hashes, and summaries. episodes.csv contains one compact row per episode. v2_qualified_inventory.json contains all 31 qualified old TRAIN candidates. joint_draw_counts.json binds recorded training draw counts to the source receipt hash. draw_coverage.json contains per-episode exposure counts.

Principles applied. Attack the Premise changed the conclusion after checking every episode instead of extrapolating the two matched cases. Build the Lever produced the rerunnable audit and summarizer. Model the Domain kept one episode record with explicit eligibility groups. Prove It Works tied the coverage to verified arrays, the exact anchor code, manifest counts, and actual recorded training draws.

Reproduce from this worktree. Execute audit.py over the authorized SSH connection with docker exec -i -w /scratch/fsop/head_trial_work fsop-pi-scratch /isaac-sim/python.sh -, capture stdout as results.json, then run python3 summarize.py locally. The remote audit only reads and prints.
