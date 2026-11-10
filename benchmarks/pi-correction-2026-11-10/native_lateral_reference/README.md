# Native lateral reference probe

The user goal is to fix inside skidpad cuts and the training regression, with truthful numbers and evidence. This throwaway calibration tests whether the existing OP steering map can execute an exact native trajectory before another training loss assumes that it can.

The data shape is one frozen source episode, a protocol, three independent arm traces, and one coordinator receipt per closed child process. A trace stores native state `[N+1,9]`, controls `[N,2]`, cumulative cone contacts `[N+1]`, route progress/lateral error `[N+1,2]`, and exact actual-ego plans `[K,33,15]` indexed by source control tick. No network, image model, or checkpoint participates.

The original `lab_v2/train-skidpad-82000-matched` source is teacher-only. All 1127 control flags are true. It has a qualified R,R,L,L route with zero physical cone hits and zero OC. The branch is tick 255, immediately after the first right lap. Both followers then receive the same fixed-clock source future through tick 926. The source has real native terminal-braking continuation through state tick 1127, so the final required state at tick 1126 exists. The recorded arm replays all 1127 controls, including that continuation.

The three frozen alternatives are recorded controls, OP steering plus current tangent PI longitudinal control, and pure-pursuit steering plus the same tangent PI longitudinal control. Each follower starts its integral at zero. All pre-branch controls come directly from the source recording. Native route failure terminates an arm and remains a failure. A fixed source-time horizon distinguishes valid completion of both left laps from the stricter full mission finish and stop gate.

The control period is 0.05 seconds. Future positions are expressed relative to the actual car position and yaw. Velocity and acceleration vectors rotate into the actual ego frame. Future yaw includes the source-to-actual heading error. The source clock never rematches to the actual pose. Each transformed plan is saved and independently recomputed after the child exits.

The full recorded arm must reproduce all native states with the inherited 1e-4 state limits and 1e-8 second clock limit. Every source contact count must equal zero. Its native mission must finish at tick 927 with a valid R,R,L,L route. No follower starts unless this gate passes. Cameras and reset match the collection setup. Later render-free steps are provisional until the full recorded baseline proves native physical parity.

The local probe directory and remote `/work/probes/native_lateral_reference1` are the only mutation locations. The protocol binds source arrays, geometry, metadata, producer archive, current runtime inventory, controller snapshots, branch, horizons, and acceptance before execution. A 25 GiB free-space reserve and a 100 MB artifact cap are mandatory. Preparation does not launch Isaac. Parent authorization is required before the GPU run.

Preparation checks have verified the source and the actual-ego transformation without launching Isaac. Independent direct world-coordinate calculations at ticks 255, 440, 650, and 926 agree within 2.1538326677728037e-14 after a synthetic actual-pose translation of (1.7,-2.3) meters and yaw offset of 0.43 radians. This verifies coordinate arithmetic, not native tracking. The initial freeze attempt stopped before mutation at the reserve gate because another running job had reduced free disk below 25 GiB. The frozen protocol and native results remain pending.

Attack the Premise changed the next step from another training loss to a perfect-reference executability test. Model the Domain changed the instrument into explicit source, protocol, arm trace, and closure receipt records. Explain the Number changed acceptance to predeclared native route/contact gates and kept one-source calibration separate from learned-policy performance.

The throughput checkpoint is one source, one recorded arm, and two follower arms. Full native parity is the first stopping gate. No videos or RGB arrays are created. Expected evidence is several MB. No production integration follows automatically from this probe.

Prototype checklist.

1. Scope the decision the prototype exists to make. Complete. Test steering executability of a perfect native future.
2. Gather references when the design space is open. Skip. The three arms are specified by the parent.
3. Build throwaway in an isolated scratch dir, separate from production source. Complete.
4. When comparing alternatives, build them behind one switcher, each variant labeled. Complete. `--worker-arm` selects the three frozen arms.
5. Verify on the matching surface. Pending parent launch authorization and the disk reserve gate. Read-only source and rebasing checks pass.
6. Present alternatives, tradeoffs, and a recommendation. Pending native results. No controller conclusion is currently supported.
