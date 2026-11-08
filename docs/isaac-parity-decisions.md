# Isaac diagnostic parity decisions

The user wants the custom simulator's diagnostics and tests in Isaac, with evidence that the displayed predictions drove the car.

## Record the inference call before rendering a video

The prior Isaac videos showed driving footage without the custom diagnostic panels. Those recordings did not retain the original image-head outputs or lossless inputs. Adding new inference to compressed historical footage would not recover the original calls.

The new runner stores every 20 Hz input pair, model plan, model scale, raw image-head output, projected detections, control action, command pulse, vehicle state and camera transform. A forward hook copies the image head during the existing policy call. The hook does not replace its output. Video generation runs offline and cannot load a checkpoint.

Both backends use the same diagnostic compositor. A separate research view shows raw image-head boxes and confidence scores. The expert reference is computed offline using the original bicycle model and is labeled as privileged reference data. Ground-truth map data is labeled separately.

The predicted path drives the existing controller. The cone image head is an auxiliary output sharing the visual encoder; its detections are not fed into the path follower. Displayed boxes use the original 0.25 confidence threshold before ground projection and cross-camera merging. A visible detection therefore does not establish that the planner will avoid that cone.

## Freeze the actual custom case matrix

The matrix contains 20 skidpad cases, 16 trackdrive retention cases, four live command ablations and one previous-failure diagnostic. The exact eight retention track files and historical results are archived. Environment initialization reproduces the original panorama inventory, reset rendering random draws, camera order and expert randomization.

The original controller, model, temporal queues, image conversion and referee remain unchanged. A baseline guard covers 76 original Python files. A runtime protocol covers 108 source files, both executable ONNX graphs and the checkpoint. The training checkout and its artifacts remain separate and read-only to the evaluation container.

## Verify before scoring

The vehicle gate checks steering direction, acceleration, braking, contact detection and capture timing. The camera gate tests 36 marker projections under sampled mounts and body pitch, roll and height changes. Its maximum error was 1.305 pixels, below the fixed two-pixel threshold.

A 120-frame pilot replay reproduced the saved outputs exactly. Disabling diagnostic capture also produced identical outputs. Black-image replay changed the plan on all 120 frames. This is an input-sensitivity check, not a driving score. The code and protocol were frozen before the full suite started.

An independent review found that a truncated manifest could look complete and that the final interval's events were missing. Completion receipts now bind frame count, metadata, manifest and terminal summary. Terminal events are saved. Scored videos reject incomplete recordings. The terminal card contains no current model prediction.

The review also found a mismatch between full-body velocity and planar state. The adapter now uses yaw-rotated world XY velocity for the original planar state and saves full-body velocity separately. This convention was fixed before scoring.

## Preserve failures and distinguish measurements

The suite runs every frozen case once and stops on technical execution failure. A driving failure is retained as an outcome. No outcome-based reruns or controller tuning are part of this study.

Original geometric cone penalties and unique PhysX cone contacts are separate measures. Pass requires the original referee's acceptance. Clean also requires zero geometric cone penalties and zero off-course events.

RTX rendering and PhysX dynamics intentionally differ from the custom simulator. Several Panda appearance parameters are not reproduced. The experiment therefore compares the combined backend change, not an isolated physics effect or real-world driving performance.

## Keep bulk evidence on the machine that produced it

The lossless inputs and raw head arrays are about a gigabyte per typical skidpad episode. Bulk transfer became a bottleneck. Full evidence and Rerun recordings remain on the Spark. CPU postprocessing starts only after the measured GPU suite completes. Videos, plots and hash-bound audit receipts are copied locally. The first scored episode is also available locally with all raw frames.

Each completed episode receives an independent CPU replay of the original mission, route-command generator and referee. The audit checks every stored counter and the terminal summary. It does not claim to resimulate physical contacts or the final PhysX transition. A rollover outcome would be explicitly marked as lacking an independent final-body-pose check.

## Keep system package installation outside Isaac's Python paths

After all 41 driving cases finished, FFmpeg installation failed because Ubuntu's Python 3.12 package hook imported Isaac's Python 3.11 standard library. The export helper now removes `PYTHONHOME` and `PYTHONPATH` from system-package subprocesses. The isolated container's package state was repaired before restarting export. No driving case was rerun, and the measured runtime remained unchanged.

## Hash Rerun recordings after their files close

The independent artifact review found that Rerun hashes changed after export. A reproduction in the actual Rerun 0.42.0 container showed why. `flush()` wrote the data, but `disconnect()` added the footer. The open copy failed Rerun's footer verifier; the closed recording passed.

The executed export producer is preserved with its original hash. Once its worker pool exits, the finalizer verifies the completed case registry, unchanged videos and evidence receipts, and each RRD footer. It preserves every original video audit and records the corrected closed-file hash with explicit finalization provenance. No simulation, model inference or video is rerun for this repair. Future exports close the recording before hashing it.

Four recordings still lacked footers after the worker processes exited. Their data passed Rerun verification with footer checking disabled. Rerun's migration command preserved each original as a backup and wrote a valid indexed recording. Its semantic comparison confirmed equal logged data, ignoring the wall-clock `log_time` timeline and row ordering. The repair receipts preserve commands and before/after hashes. The full finalization then verifies the repaired files normally.

A separate 120-frame export regression in Rerun 0.42.0 confirmed the corrected producer's hash stays unchanged after return, after another recording starts, and after the process exits. This regression reused pilot evidence and is excluded from all 41 driving outcomes.
