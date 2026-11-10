# Driving reliability continuation

The goal is a vision model that finishes cone tracks and full R,R,L,L skidpads without physical cone contacts in realistic Isaac scenes. No new checkpoint is promoted. The inherited failed trials and videos remain in [the correction report](../pi-correction-2026-11-10/README.md).

[The final acceptance suite](acceptance_suite.json) was frozen before new training or final checkpoint selection. It contains 16 held-out conditions and four candidate runs per condition. Every candidate run must finish with zero physical contacts and zero off-course events. Repeated realizations are reported separately from independent scene coverage. No final condition has been launched by this continuation.

## Execution diagnosis

[The reproducible divergence analysis](native_divergence_analysis/analyze.py) verifies eight original input hashes and the exact recorded teacher replay. The first follower command differs at tick 255 despite identical pose. At contact interval 449, OP plus PI is 0.29468 m laterally from the source path. Nearby spatial projection leaves 0.29479 m error, so this is more than fixed-clock phase lag. The velocity-displacement accounting is descriptive, not a causal decomposition.

The following are single-source privileged native diagnostics. They do not use a vision model to choose the executed path. The suffix arms retain teacher controls through tick254.

| Execution | Contact count | Off-course events | Simulated seconds | Full route and stop gate |
|---|---:|---:|---:|---|
| Inherited recorded teacher | 0 | 0 | 46.35 | Pass |
| Inherited OP plus PI suffix | 1 | 0 | 46.20 | Fail contact gate |
| OP plus position feedback, unchanged PI suffix | 0 | 0 | 46.15 | Pass |
| Unchanged OP, recorded longitudinal suffix | 0 | 0 | 46.35 | Pass |
| Position feedback and recorded longitudinal suffix | 0 | 0 | 46.35 | Pass |
| Unchanged OP plus PI from standing | 6 | 0 | 46.10 | Fail contact gate |

[The frozen suffix protocol and traces](native_transition_correction/) separate position restoration from longitudinal intervention. Both isolated interventions clear the contact on this source. They do not uniquely identify an underlying cause. The [standing-start run](native_standing_reference/) shows that a legitimate zero-integral start also contacts cones. The cold handoff is not a sufficient explanation of the execution gap.

The successful position correction reads source-minus-actual displacement at the initial plan knot. Neural targets start that knot at zero. Applying that correction directly to the neural output would not supply the privileged cross-track error.

[The independent command audit](audit_native_controls.py) reconstructs world-coordinate plans, steering, complete PI state, mission intervention and executed actions across all four new arms. It passes at absolute tolerance 1e-10, with maximum residual below 2.2e-12. Its first implementation exposed rounded-geometry reconstruction error. Native execution constructs the skidpad directly, while serialized track JSON rounds centerline coordinates to four decimal places. The audit now reconstructs the same native factory geometry and binds the relevant canonical geometry fields.

## Frozen-input model interventions

[The command probe](reliability_command_probe/PROTOCOL.md) replays all 601 saved frames through frame 600 with no saved-output mismatches. It preserves actual visual features, complete route-command history and pre-step PI memory. Swapping active route histories changes steering by 0.22439 rad at frame 120 and changes the predicted path substantially at frame 430. Frames 170 and 600 have no active nominal pulses, so swapping them is a null intervention. Frames 430 and 600 follow earlier physical contacts. These fixed-state results reject route blindness on the tested inputs without establishing correct route behavior.

[The compact summary](reliability_command_probe/summary.json) binds a lossless compressed full result and its raw archive. The result retains every plan and controller proposal.

[The appearance protocol](reliability_appearance_probe/PROTOCOL.md) changes surfaces, people, barriers and lighting independently across the corresponding 26 physical poses and 25 encoded image pairs at precontact frame 170. All four baseline repeats pass the existing noise limits. The largest repeat differences are 0.00658 m in near-term XY,0.000209 rad in steering and0.00691 m/s² in acceleration. Removing surface overlays changes near-term XY by 0.03973 m, exceeding the preset five-times-noise threshold. Its steering effect is below the preset gate. Other interventions have no effects above their preset thresholds on this one state. This does not establish general perception robustness or a closed-loop cause.

## Targeted command experiment

[The design synthesis](architecture/synthesis/README.md) chooses a direct executed-command head on the existing frozen temporal-policy representation. The head adds 33,538 parameters to the complete 31,154,802-parameter model. Raw teacher controls supervise only its actuator output. No privileged displacement, future state, ground-truth cones or teacher identifier enters that head. Original physical plans and cone heads remain recorded diagnostics.

Real Spark preflight, the fixed 32-anchor wiring test, a fixed 500-update recipe, all-validation command qualification, visual dependence and native driving are separate gates. The first two setup attempts stopped before optimization on canonical dependency-path and source-archive issues. Their failures are retained. Numerical gates were unchanged.

The third setup passed. It verified 7,006 TRAIN anchors, 2,067 validation anchors and all 417 unchanged original model tensors. The discarded wiring head passed at update 62. The main run completed 500 updates, but all six saved checkpoints failed validation. No checkpoint was selected. At update 500 the aggregate command MAE was 0.00612 rad and 0.17908 m/s². Acceleration p95 was 0.78333 m/s², above its 0.75 limit. Three track episodes and low-speed steering also failed their unchanged limits. The complete per-group records are in [the training receipts](direct_command_training/train.json).

The measured training and validation job took 519.58 seconds. The external sampling wrapper took 526.99 seconds. Median whole-device GPU utilization was 74%, median power was 46.37 W and peak PyTorch allocation was 1,359,339,008 bytes. Median whole-host CPU utilization was 5.2%. No other GPU compute process was present immediately before or after the job, and the earlier CFD processes had ended independently. These are achieved job and whole-device measurements, not FLOPS efficiency or a speedup. [The hardware samples](direct_command_training/hardware_measurement.json) retain their time series.

![Recorded optimization loss](plots/training_loss.png)

![Validation command errors and unchanged ceilings](plots/validation.png)

![Whole-device and host hardware samples](plots/hardware.png)

The frozen development protocol allows only step 500 of an unqualified run to undergo the bounded TRAIN pilot and one full exposed run per discipline. It does not permit qualification repeats. The six-second pilot made three physical contacts and one off-course event. [The first-error diagnosis](direct_command_diagnosis/result.json) finds understeering and insufficient braking before contact. At three seconds, the raw steering error against a privileged teacher query at the actual learner state was 0.11578 rad, and acceleration error was 1.5638 m/s². Those labels were not supplied to the learned controller.

![Actual rejected pilot with privileged diagnostic teacher queries](plots/first_error.png)

Full CUDA replay of all 120 pilot frames passed at the original 1e-5 numerical tolerances. CPU replay failed on unchanged image and plan diagnostics, while the learned command checks matched. The CPU failure remains retained and does not qualify the model. The producer artifact audit reconstructs the declared float32 proprio inputs with no tolerance change. Both actual debug and research videos passed full 160-frame decoding. Their banner distinguishes frozen plan diagnostics from learned executed commands. [The video audit](direct_command_pilot/lab_video_audit.json) binds producer rendering and portable encoding.

The full exposed skidpad diagnostic timed out after 120 simulated seconds with 14 physical contacts, seven off-course events and no accepted laps. The exposed track diagnostic also timed out at 120 seconds with three contacts, two off-course events and no completed lap. Its legacy `raw_time=0` and penalty-derived `time=26` do not represent a completion time. These are rejected-model failure diagnostics, not final acceptance samples. The next targeted change is independently verified on-policy TRAIN recovery data from the missed entry state. Collection, fitting and native evaluation remain separate gates.

## Reproduction and retained artifacts

[The source copy receipt](direct_command_source/published_copy_receipt.json) binds exact experiment-source copies and training receipts. Restore the files in `direct_command_source` to `.work/lab/direct_command_trial` from the repository root to preserve their bound paths. The training source archive also retains the exact original runtime sources and explicitly mapped external dependencies. Full checkpoint packages remain under the ignored local `runs/direct_command_trial3_archive` directory, with all six hashes verified in its published copy receipt. No failed checkpoint was removed.

The current local report is `runs/driving_reliability_report/index.html`. It includes native traces, command and appearance probes, actual training and validation curves, hardware samples, first-error plots, and links to all eight inherited videos and both new failed-pilot views. Its evidence inventory hashes every consumed artifact. Full exposed episodes will be added only after replay, producer audit and video verification complete.

The [work plan](../../docs/isaac-lab-training/reliability_work_plan.md) and append-only [decision log](../../docs/isaac-lab-training/decisions.tsv) record continuation state. New diagnostics use the existing RAM-backed workspace. The running container, unrelated CFD processes and inherited evidence are preserved.

## Corrective data collection

The first new episode switches the frozen rejected learner to the teacher at tick 63, before contact. The teacher completes full R,R,L,L, exit and stop with zero physical contacts and zero off-course events. The independent verifier replays all 64 prefix frames on CUDA, checks every saved training desire, confirms 200 actual terminal braking ticks and qualifies 219 anchors. Ten anchors lie in the first two seconds after handoff. The settling window starts at tick 95. This is qualified training data, not learned-model performance. [The data receipt](direct_command_recovery_data/skidpad/receipt_v2.json) and [metadata](direct_command_recovery_data/skidpad/meta.json) bind all arrays and source bytes.

The [first verifier receipt](direct_command_recovery_data/skidpad/failed_receipt_v1.json) failed because JSON writes a vehicle parameter tuple as a list. The scalar values, mount, camera and scene matched. A separately versioned [derived verifier](direct_command_recovery_source/verify_v2.py) applies the producer's JSON serialization to the expected vehicle dataclass. The original collector, verifier, protocol, failed receipt and source archive remain intact. No numerical or physical gate changes. The track recovery also passes independent verification. It switches at tick 164, settles from tick 178, completes with zero contacts and zero off-course events, and supplies 205 qualified anchors. All 165 prefix frames replay on CUDA. [Its receipt](direct_command_recovery_data/trackdrive/receipt_v2.json) binds the cached track geometry and actual camera history.

The [prepared data-only training trial](direct_command_recovery_trial_source/protocol_v2.md) draws 16 old recovery, 16 new recovery and 96 ordinary anchors per batch. A fresh head uses the original 500-update optimizer and validation limits. Both new data receipts must pass before its audit or optimization can start. The independent source review found no blocker, and three CPU sampler and receipt-rejection tests pass. The real Spark audit is running. Optimization has not started.
