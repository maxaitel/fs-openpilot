# First-step learning-rate probe

The skidpad goal is fewer cone contacts and valid full finishes. This probe measures one isolated TRAIN update and cannot establish a closed-loop result.

All three tested learning rates reduced the plan NLL plus consistency fit objective in each of four independent batches. The retained 3e-5 control increased the same objective in all four batches in the prior probe. Smaller first steps therefore remove the observed fit overshoot in these batches. None qualifies under the frozen monitor rule. In particular, aggregate monitor steering MAE increased in every cell, while aggregate raw acceleration and target-speed MAEs decreased. Learning rate alone does not resolve the measured steering tradeoff. The smallest changes are near the arithmetic discrepancies described below.

Each cell starts from the exact latest model and creates a fresh AdamW optimizer. The four batches are the same 128 draws used by the v2 probe, including 16 recovery draws each. The 16 monitor anchors are disjoint by index from the fit batches, but may share episodes and history or future windows. Each phase contains four monitor anchors. These are diagnostic TRAIN measurements, not independent validation.

The tables use NumPy float64 reconstructions from the saved FP32 predictions. A positive objective margin means the fit objective decreased. A positive action delta means MAE worsened. Each range covers the four independently reset batch updates.

| Learning rate | Float64 fit objective decrease by batch 0, 1, 2, 3 |
|---|---|
| 3e-06 | 0.036016224, 0.022410881, 0.016939381, 0.026096435 |
| 3e-07 | 0.005112456, 0.003263193, 0.003437266, 0.004492576 |
| 3e-08 | 0.000522083, 0.000333694, 0.000363046, 0.000464942 |

The aggregate monitor baseline MAEs are steer_rad 0.02015075, raw_accel_mps2 0.08074343, target_speed_mps 0.04808729. Steering is in radians, raw acceleration is in m/s², and target speed is in m/s.

| Learning rate | Steering delta, mean [range] | Raw acceleration delta, mean [range] | Target speed delta, mean [range] |
|---|---:|---:|---:|
| 3e-06 | +0.00013474 [+0.00006959, +0.00018000] | -0.00162951 [-0.00245888, -0.00019041] | -0.00172889 [-0.00212479, -0.00105864] |
| 3e-07 | +0.00001341 [+0.00000707, +0.00001793] | -0.00016414 [-0.00024688, -0.00002176] | -0.00021409 [-0.00025659, -0.00016625] |
| 3e-08 | +0.00000135 [+0.00000073, +0.00000181] | -0.00001666 [-0.00002480, -0.00000244] | -0.00002147 [-0.00002581, -0.00001667] |

The phase table gives the same mean and range over the four fit-batch updates. Each phase baseline is fixed across the learning rates. Individual batch and phase values appear in `result/float64_receipt.json`.

| Learning rate | Phase | Steering delta, mean [range] | Raw acceleration delta, mean [range] | Target speed delta, mean [range] |
|---|---|---:|---:|---:|
| 3e-06 | steady_right | -0.00001179 [-0.00002188, +0.00000696] | +0.00019869 [-0.00010395, +0.00041532] | -0.00262815 [-0.00283435, -0.00250921] |
| 3e-06 | right_to_left | +0.00017660 [-0.00001165, +0.00031047] | -0.00421310 [-0.00722435, +0.00076743] | -0.00347723 [-0.00551269, +0.00004531] |
| 3e-06 | steady_left | +0.00001290 [+0.00000646, +0.00001783] | +0.00011249 [+0.00003725, +0.00021477] | +0.00014080 [+0.00009864, +0.00019060] |
| 3e-06 | recovery | +0.00036125 [+0.00029688, +0.00042497] | -0.00261613 [-0.00344341, -0.00202014] | -0.00095098 [-0.00192023, -0.00034609] |
| 3e-07 | steady_right | -0.00000149 [-0.00000237, -0.00000041] | +0.00002005 [-0.00001040, +0.00004160] | -0.00042617 [-0.00049367, -0.00038593] |
| 3e-07 | right_to_left | +0.00001764 [-0.00000116, +0.00003098] | -0.00042116 [-0.00072201, +0.00007715] | -0.00034763 [-0.00055097, +0.00000476] |
| 3e-07 | steady_left | +0.00000130 [+0.00000066, +0.00000177] | +0.00000818 [-0.00000018, +0.00002076] | +0.00001184 [+0.00000698, +0.00001857] |
| 3e-07 | recovery | +0.00003620 [+0.00003006, +0.00004246] | -0.00026362 [-0.00034489, -0.00020970] | -0.00009439 [-0.00018872, -0.00003523] |
| 3e-08 | steady_right | -0.00000013 [-0.00000021, -0.00000002] | +0.00000216 [-0.00000087, +0.00000419] | -0.00004217 [-0.00004901, -0.00003802] |
| 3e-08 | right_to_left | +0.00000177 [-0.00000009, +0.00000311] | -0.00004242 [-0.00007271, +0.00000726] | -0.00003499 [-0.00005545, +0.00000012] |
| 3e-08 | steady_left | +0.00000015 [+0.00000008, +0.00000019] | +0.00000036 [-0.00000021, +0.00000168] | +0.00000090 [+0.00000056, +0.00000159] |
| 3e-08 | recovery | +0.00000362 [+0.00000302, +0.00000425] | -0.00002673 [-0.00003516, -0.00002066] | -0.00000961 [-0.00001857, -0.00000396] |

The strongest tested rate increases aggregate steering MAE for all four batches and increases recovery steering MAE in all four. It reduces aggregate raw acceleration and target-speed MAEs for all four. Its phase results vary. For example, raw acceleration worsens in steady left across all four batches, and target speed worsens in steady left across all four. Phase changes at the smallest rate can be comparable to arithmetic differences, so their signs alone do not demonstrate a reliable effect.

The retained 3e-5 control comes from the prior 48-cell run and was not rerun. Before any new update, the new run matched every saved before prediction array and the fit objective and component losses against that control with absolute tolerance 1e-7 and relative tolerance 1e-6. The independent verifier then checked motion and phase MAEs from the saved arrays, checked source, model, data-manifest and artifact hashes, and confirmed the update process had exited with code 0.

The float64 audit reconstructed plan NLL, consistency and motion from saved outputs without a model run. The largest absolute discrepancy between a reconstructed objective and its FP32 recorded value was 3.7e-07. The largest discrepancy between reconstructed and recorded fit margins was 4.51e-07. The largest monitor MAE discrepancy was 4.35e-07 and the largest monitor delta discrepancy across aggregate and phase metrics was 6.76e-07. These are arithmetic comparisons, not estimates of run-to-run variability. They limit interpretation of small action deltas.

The run configured 50 policy tensors and found gradients and optimizer state for 42. It checked the learning rate in the optimizer parameter group, one optimizer step per cell, exact initial model reset, and unchanged frozen tensor hashes. The saved arrays are 10,836,443 bytes and match the run and float64 receipt hashes. The original v2 CPU FP32 parameter direction diagnostics remain disqualified. This probe does not report them.

Attack the Premise shaped the learning-rate sweep after the production-rate step increased its own fit objective. Model the Domain represented the experiment as four frozen batches crossed with three rates. Explain the Number kept the fit and monitor results separate and required the float64 discrepancy check. Prove It Works required the independent receipt and a hash check of the copied arrays.
