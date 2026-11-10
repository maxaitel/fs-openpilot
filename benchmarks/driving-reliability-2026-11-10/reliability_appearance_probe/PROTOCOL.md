# Fixed-history appearance probe

The decision is whether independent changes to surface overlays, people, barriers or scene lighting cause a response larger than render repeat noise at the pre-contact skidpad frame 170. This is an open-loop diagnostic, not a driving trial or proof of contact avoidance.

Two minimal designs were considered before implementation. Reusing saved RGB and masking objects in image space would be cheap and repeatable, but would invent occluded pixels and cannot test lighting or the full visual history. Rendering the same 26 recorded physical poses under controlled scene changes recreates all 25 encoded pairs and tests the actual renderer. It is selected, subject to an interleaved baseline repeat gate because earlier renderer probes were unstable. Every intervention is a scene-only counterfactual, never an input to the driving controller.

Use `runs/development_head_trial1/baseline/heavy_skid`, frame 170. The first contact is frame 181. Replay frames 70 through 170 in steps of four with recorded body and cone poses and camera transforms. Recreate all 25 image pairs, raw commands from frame zero, the 25 pooled desire slots, common 20 Hz postprocess random stream and traffic convention. The graph reads the last nine feature slots, but the full 25 pairs preserve the actual caller's temporal shape. Derive the controller mode from evidence metadata. Use the saved pre-step PI integral and the previous frame's timestamp, checked against the independently replayed command-probe anchor. The renderer does not reproduce the episode pixels, so saved prediction differences are diagnostics, not a baseline parity gate.

Interleave baseline stress rerenders with each independent intervention. The conditions are baseline, remove Ground and Surface while restoring the 14 original repair patches, baseline, hide only kinds beginning `person_`, baseline, hide only kind `barrier`, baseline, multiply only `/World/Sky` and `/World/Sun` intensity by 0.5, baseline. `wall` is explicitly not a barrier. Ground is a surface overlay, not lighting. Reapply the full baseline state before each condition. Use 16 render subframes per pose and no simulation clock advance. Verify physical contracts, exact camera receipts and RNG end states across all conditions. Record all individual plans, proposals, PNGs and source and model provenance even when the repeat gate fails.

Compare each baseline rerender to the first. Interpretation fails if the maximum near-horizon XY norm exceeds 0.02 m, steering exceeds 0.002 rad, or acceleration exceeds 0.03 m/s², or if sources, model state or checkpoint change. For stable repeats, record each intervention's difference against the first baseline and flag a dimension only when it exceeds both five times the observed maximum repeat noise and its repeat limit. Repeat noise is not a confidence interval. A single frame cannot establish reliability, lap completion, or causal closed-loop driving behavior.

Run on Spark from the matching repository root with the pinned checkpoint and evidence present:

```sh
PYTHONPATH=src:.:isaac_eval /isaac-sim/python.sh .work/lab/reliability_appearance_probe/probe.py --episode runs/development_head_trial1/baseline/heavy_skid --frame 170 --checkpoint /scratch/fsop/head_trial_work/probes/pi_models/latest_pi_train.pt --command-result .work/lab/reliability_command_probe/result.json --out .work/lab/reliability_appearance_probe/result
```

This is a throwaway prototype. Local preparation does not run Isaac or establish a result. The Spark operator should review the script and protocol before execution.
