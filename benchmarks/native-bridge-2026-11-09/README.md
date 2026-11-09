# Native trajectory-to-controller checks

These are short physics diagnostics, not learned-model evaluations. They use actual recorded futures from two training episodes and three fixed two-second windows: straight, steady turn and recovery. Fresh test cases were not used.

Both experiments replayed canonical reset and the complete recorded prefix. Every recorded-control and native-teacher repeat matched all nine recorded state columns and contacts exactly. Every intervention separately passed its prefix gate. Cameras initialized normally; rendering stopped after reset because these arms consume no images. This establishes measured state parity, not camera or hidden wheel-state equivalence.

| Window | Original full PP maximum position error | PP + tangent | OP + tangent |
|---|---:|---:|---:|
| Straight | 0.07502 m | 0.06715 m | 0.08371 m |
| Steady turn | 1.09016 m | 0.41773 m | 0.21013 m |
| Recovery | 1.10097 m | 0.41366 m | 0.19934 m |

Errors compare actual native position to the recorded teacher at the same clock. Each window contains 40 controls and 41 states. All arms had zero contacts and off-course events, but their referees were still running at window end. These results do not establish completed missions or collision-free learned driving.

The original PP longitudinal controller treats velocity and acceleration along the initial ego x axis as forward speed and acceleration. Native targets instead express future planar vectors in that fixed frame. Rotation in a constant-speed turn therefore produces a false braking request. The tangent experiment computes planar speed and tangential acceleration at every knot before interpolating to 0.3 seconds. Steering remains the existing PP or Openpilot follower. The current measured planar speed is used for longitudinal feedback; steering still receives measured body-forward velocity.

A residual remains: net trajectory acceleration differs from the wheel-torque request needed to overcome resistance. OP steering with recorded teacher acceleration had smaller turn/recovery errors, 0.04718/0.07525 m. No fitted resistance compensation was added.

## Artifacts and reproduction

`original/` contains 15 arms and `tangent/` contains 12. The selected receipt subset includes frozen protocols, selections, results, traces, prefix checks and progress diagnostics. Original process logs remain on Spark and in the complete local receipt archives. `sha256.json` hashes this published subset; it is not a claim that omitted logs are included.

The files in `source/` preserve the exact executed script and protocol bytes. Python scripts have `.txt` suffixes to distinguish archived experimental source from maintained code. Copy them into the original locations before execution:

```sh
mkdir -p .work/lab
cp benchmarks/native-bridge-2026-11-09/source/native_bridge_probe.py.txt .work/lab/native_bridge_probe.py
cp benchmarks/native-bridge-2026-11-09/source/native_bridge_protocol.json .work/lab/native_bridge_protocol.json
cp benchmarks/native-bridge-2026-11-09/source/native_bridge_tangent_probe.py.txt .work/lab/native_bridge_tangent_probe.py
cp benchmarks/native-bridge-2026-11-09/source/native_bridge_tangent_protocol.json .work/lab/native_bridge_tangent_protocol.json
```

Use the runtime source at commit `8d02a32ba7c75e53de61c1b1766a5c87088a18b6`, the pinned renderer container described in the [run record](../../docs/isaac-lab-training/run.md), and the original verified `data/native/lab_v4` dataset and assets on Spark. The source, dataset, physics, scene and protocol checks intentionally reject changed inputs. Do not recreate nominally similar data and call it the same experiment.

Run each script with `/isaac-sim/python.sh`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and a new `--out` directory. The original script SHA-256 is `eff34f9a45d13e009f945c1be00eeb4229ec1ad6f6148f8d59581da36e2565e5`; the tangent script is `5cb354f56a70dd2623a902fb2b1b9d34087494816015a9144e7f28ca0505aaaf`. Their protocols retain all gates and limitations.
