The user goal is fewer skidpad contacts and valid full finishes. This throwaway probe measures one local TRAIN update. It does not measure closed-loop improvement.

The frozen matrix contains 2 initializers, 4 fit batches, and 6 objective arms. Each cell resets exact initial weights and creates a fresh production AdamW optimizer. Each fit batch has 128 draws, including 16 recovery draws. Sixteen monitor anchors are disjoint by index, with four anchors per phase. History, future windows, and episodes can overlap fit data.

The policy learning rate is 0.00003. Weight decay is 0.0001 and gradient clipping is 1.0. Vision and auxiliary tensors remain frozen. The configuration marks 50 policy tensors containing 6,949,412 parameters trainable. Each cell creates optimizer state for 42 tensors that receive gradients from these plan losses. The other eight eligible tensors do not receive a gradient.

All 48 cells completed. The run process exited 0. The separate verifier confirmed finite output arrays, unchanged model and source files, and independently recomputed motion and phase action errors. The original parameter-space cosine and norm diagnostics are unqualified because 72 cosine entries exceed their mathematical range. The supplemental numerical-limit receipt narrows the verification claim. No optimizer rerun occurred.

All seven copied artifact hashes match the remote inventory. The files total 40,508,975 bytes. A local NumPy float64 check reconstructed all five losses for 192 before-and-after measurement groups. Its largest absolute discrepancy was 0.000000242 for plan NLL. The receipt is in analysis_v2/local_loss_receipt2.json.

The following values are monitor MAEs. After values are the mean across four independent fit-batch updates. Steering is in radians, raw acceleration and tangent acceleration are in m/s², and target speed is in m/s.

| Initializer | Objective | Steering | Raw acceleration | Target speed | Tangent acceleration |
|---|---|---:|---:|---:|---:|
| latest | Initial | 0.020151 | 0.080743 | 0.048087 | 0.080715 |
| latest | plan_consistency | 0.022642 | 0.093660 | 0.056103 | 0.084586 |
| latest | plan | 0.022646 | 0.097192 | 0.059142 | 0.084575 |
| latest | consistency | 0.020239 | 0.077095 | 0.048518 | 0.081333 |
| latest | action | 0.020873 | 0.092323 | 0.066315 | 0.079311 |
| latest | steer | 0.024224 | 0.079388 | 0.046792 | 0.087557 |
| latest | longitudinal | 0.020478 | 0.092249 | 0.066283 | 0.079285 |
| vision | Initial | 0.020349 | 0.638044 | 0.357625 | 0.310649 |
| vision | plan_consistency | 0.020859 | 0.569978 | 0.315239 | 0.305549 |
| vision | plan | 0.020929 | 0.570500 | 0.315747 | 0.305487 |
| vision | consistency | 0.019208 | 0.565078 | 0.319195 | 0.292379 |
| vision | action | 0.019951 | 0.420433 | 0.210614 | 0.310989 |
| vision | steer | 0.023829 | 0.647724 | 0.361427 | 0.317569 |
| vision | longitudinal | 0.019649 | 0.420435 | 0.210634 | 0.310981 |

Plan NLL alone and plan NLL plus consistency increased their own fit objective in all four batches for both initializers. Latest also increased its fit objective under consistency alone and longitudinal loss alone. Steering-only reduced fit steering loss in every batch for both initializers, but worsened monitor steering in every batch. These observations support investigating the size of the first AdamW step and the fit-to-monitor distribution. They do not identify a safe learning rate or establish that an objective is intrinsically unsuitable.

The next table gives phase MAEs under the current plan-plus-consistency objective. Each phase has four anchors. Each after value averages the four independent updates.

| Initializer | Phase | Steering before | Steering after | Raw acceleration before | Raw acceleration after | Target speed before | Target speed after | Tangent acceleration before | Tangent acceleration after |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| latest | steady_right | 0.003802 | 0.005792 | 0.037408 | 0.103760 | 0.028272 | 0.068856 | 0.016245 | 0.006600 |
| latest | right_to_left | 0.005307 | 0.007065 | 0.137501 | 0.094185 | 0.061803 | 0.029744 | 0.089591 | 0.110565 |
| latest | steady_left | 0.002269 | 0.004869 | 0.030970 | 0.053845 | 0.018300 | 0.036985 | 0.007042 | 0.008538 |
| latest | recovery | 0.069225 | 0.072844 | 0.117095 | 0.122852 | 0.083975 | 0.088825 | 0.209982 | 0.212641 |
| vision | steady_right | 0.003807 | 0.003647 | 0.806821 | 0.692552 | 0.424749 | 0.351025 | 0.339395 | 0.332030 |
| vision | right_to_left | 0.004067 | 0.006325 | 0.229498 | 0.172609 | 0.193894 | 0.158650 | 0.122686 | 0.130731 |
| vision | steady_left | 0.003179 | 0.003391 | 0.784177 | 0.801901 | 0.431979 | 0.448693 | 0.272416 | 0.257721 |
| vision | recovery | 0.070346 | 0.070072 | 0.731679 | 0.612848 | 0.379878 | 0.302590 | 0.508098 | 0.501714 |

The latest initializer worsens steering in every phase under the current objective. Its raw-acceleration error improves at the right-to-left transition and worsens in steady right, steady left, and recovery. Vision improves its aggregate raw-acceleration error, but steady-left raw acceleration worsens. Aggregate error therefore hides a phase tradeoff.

The consistency nullspace check passed for both initializers. Adding 0.25 to position, yaw, and yaw rate leaves the loss unchanged. Adding the same planar velocity offset to every knot also leaves it unchanged. The residual uses only planar acceleration and adjacent velocity differences. Consistency alone cannot directly constrain steering or the absolute velocity level.

The effective NLL mean weights below equal time weight × column weight × exp(-clamped logstd), divided by the total time-column weight. The batch-average factor is omitted equally for every column.

| Initializer | Position z | Velocity z | Yaw | Yaw rate | Velocity x | Velocity y | Acceleration x | Acceleration y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| latest | 0.791222 | 0.791222 | 0.891378 | 0.073356 | 0.077383 | 0.141512 | 0.011925 | 0.009265 |
| vision | 0.791222 | 0.791222 | 1.034134 | 0.127172 | 0.066372 | 0.172697 | 0.010323 | 0.010234 |

The native z and vz targets are exactly zero. Their predicted log scales are below the -7 clamp at every monitor knot. Their combined direct contribution is only 0.012% of latest monitor NLL deterioration and 0.034% of vision deterioration under the current objective. XY position supplies most of the measured NLL increase. The first time knot contributes less than 1%. These output contributions do not measure each column’s shared-parameter gradient.

The combined action arm closely matches longitudinal-only output changes. Direct steering-only supervision improves fit loss but worsens these monitors. No coefficient or production training recipe is selected from this probe.

The complete four-batch means, ranges, and improvement counts are in summary_v2.json. Full predictions, log scales, native targets, observed motion, and desired motion are in result_v2/arrays.npz. Exact case IDs, frame indices, ticks, source hashes, model hashes, and draw inventory are in protocol_v2.json. The numerical limitation is recorded in analysis_v2/pi_update_numerical_limit2.json.

Model the Domain shaped the implementation. A six-entry objective table defines the experiment cells. The same plan and motion output shape supplies every before-and-after measurement. The benchmark checklist kept timing descriptive because CPU export work could overlap the run.
