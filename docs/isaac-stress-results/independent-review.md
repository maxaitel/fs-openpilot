# Independent artifact review

The four paired visual-stress cases have complete, internally consistent artifacts. The two stressed trackdrive episodes timed out. The matched episodes finished. These eight runs are an exploratory subset, so they do not establish a full-suite failure rate or a performance comparison.

| Case | Matched result | Matched contacts | Stress result | Stress contacts | Stress off course |
| --- | --- | ---: | --- | ---: | ---: |
| Skidpad light | Finished | 3 | Finished | 12 | 0 |
| Skidpad heavy | Finished | 15 | Finished | 4 | 0 |
| Trackdrive light | Finished | 0 | Timeout | 11 | 13 |
| Trackdrive heavy | Finished | 0 | Timeout | 9 | 22 |

The matched runs had zero off-course events. Their total was four finishes, two physically clean runs, and 18 contacts. The stress runs had two finishes, no physically clean runs, and 36 contacts. Physical clean means a referee pass with zero contacts and zero off-course events.

The independent scan ran on Spark after the producer completed. It checked the following actual files and receipts:

- The frozen plan identified exactly the expected eight episode directories. Its digest was `e35dce2437e04f6e67debbdc768e02b07e81c3e6732b60db1cb194ead9c2a8de`. Plan validation checked current runtime and tool sources, checkpoint, graph, assets, gates, previews, and their frozen identities.
- All eight completion seals matched their summary, metadata, and manifest files. All 11,101 sequential raw NPZ files matched their manifest SHA-256 hashes. Audit receipts matched the sealed files and declared frozen source identities.
- Exactly eight `debug.mp4` and eight `research.mp4` files occupied the expected episode paths. Each final file matched its video receipt SHA-256. An independent `ffprobe -count_frames` decode found 20 frames per second, 1280 by 768 pixels, and exactly the recorded inference frame count plus 40 terminal frames for each video.
- Exactly eight `debug.rrd` files occupied the expected paths. Each matched its recorded SHA-256 and passed `rerun rrd verify --check-footers=true`.
- Each of the four pairs had identical declared initial state, case, model and source identities, track, cone layout, vehicle and expert parameters, camera calibration, physics hash, and realization hash. Trajectories can diverge after the visual treatment changes the model input.

[Machine-readable results](independent-verification.json) contain per-episode hashes and frame counts. The scan script is [`provenance/independent_verify.py`](provenance/independent_verify.py). It reads `/work` inside the `fsop-isaac-stress-export` container. The verification did not overwrite episode artifacts or receipts.

These checks establish file integrity, producer identity, and the stated paired invariants. They do not rerun the model or independently simulate physical contacts. The existing episode audits state that model outputs are not rerun, contact events are not independently simulated, and the final physics transition is not reconstructed. People and clutter are appearance-only, so this experiment is not a pedestrian-safety test. Visual inspection of the gallery is separate from this artifact review.
