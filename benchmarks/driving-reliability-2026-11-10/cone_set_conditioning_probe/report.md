# Pooled normalization failed the frozen diagnostic criterion

The zero-contact vision FormulaStudent investigation does not advance the tested pooled LayerNorm candidate. Its geometry sensitivity improved, but it failed the prespecified requirement to retain at least 90 percent of motion sensitivity at every anchor. The protocol and thresholds remain unchanged.

The [completed report](results/report.json) compares the original seed-zero head with one affine-free LayerNorm over the 128 pooled features. Both variants use the same 47,618 parameters and exact ordered 32 TRAIN inputs. Geometry derivatives use coherent ego lateral shifts and yaw rotations with a consistent range channel.

| Measured quantity | Steering | Acceleration |
| --- | ---: | ---: |
| Median geometry sensitivity gain | 17.9369 | 11.0052 |
| Median retained motion sensitivity | 1.1867 | 0.9014 |
| Minimum retained motion sensitivity | 0.7200 | 0.8091 |
| Anchors below 0.90 motion retention | 2 of 32 | 14 of 32 |

The readout condition number changed from 9911.7691 to 1519.0567, a 6.5250-fold improvement. This did not reach the alternative tenfold condition criterion. The threefold geometry criterion passed, but motion retention was mandatory under either route. All measured local ranks and the rank-32 readout were retained. No new pair collapse was detected.

These measurements show that normalization changes how the initialized head responds to geometry and motion. They do not establish learnability, generalization, or successful driving. The result gives no basis to recommend this LayerNorm candidate under the frozen decision rule.

The [input receipt](results/input_binding.json) binds all 32 fresh input hashes, same-tick targets, and original parameter hashes. The probe verified all 417 base states and eight head states unchanged. It performed no optimizer update or analytic fit and exported no weights. The [completion receipt](results/complete.json) binds the report, input receipt, attempt, and frozen source hash.

The earlier cone-set wiring v1 remains failed after 200 updates, with MAE 0.005447729 steering and 0.314533442 acceleration. Its [failed overfit receipt](source/receipts/overfit.json) and [failure marker](source/receipts/overfit_failure.json) are retained. No main fit ran and no model was selected. The final suite remains sealed.

The [frozen protocol](source/protocol.md), [executable](source/probe.py), and [source freeze](source/freeze.json) are exact archival copies. The executable retains its original repository paths and imports. The [source index](source_index.json) distinguishes those sources from execution results and transfer receipts. The [publication receipt](published_copy_receipt.json) records independent local verification of the source freeze, failed receipts, completion chain, ordered input bindings, and remote-copy inventory. Publication did not rerun the probe.

Explain the Number preserved the decision threshold despite the geometry gain. Prove It Works required byte verification of the actual source and result copies.
