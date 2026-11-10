# Final-layer update experiment

Eight independent TRAIN updates change only the existing final 990 by 256 affine weight and its 990-element bias. Original forward gradient flags remain unchanged. The strict initial-prediction gate passes. The other 415 state tensors remain byte-identical. The larger learning rate increases fitting loss in every batch. The smaller rate reduces fitting loss, but monitored steering worsens in every batch at both rates. Restricting the updated parameters reduces regression versus full-policy updates, but does not fix it.

| Policy rate | Fit reduction range | Steering MAE increase range, rad |
|---|---:|---:|
| 3e-05 | -0.095086452 to -0.074787340 | 0.000064528 to 0.000082557 |
| 3e-06 | 0.003156915 to 0.005644959 | 0.000006471 to 0.000008269 |

The first verifier failed because the older control stores phase action metrics under `action_mae`. Its error log is retained. A repaired verifier reads that field and independently recomputes all losses and motion metrics in NumPy float64 without changing bounds. A first repair supplied a placeholder PID and its receipt is explicitly invalid process-closure evidence. The authoritative `independent_receipt_v2.json` uses the shell wait completion and an exact `/proc` argument scan. The parent observed the wrapper finish after the training child exited zero.

Maximum loss discrepancy is 3.41102401e-07; maximum MAE discrepancy is 6.24850704e-07. These are arithmetic checks, not a repeatability noise floor. Monitoring indices are excluded from fitting, but episodes and temporal windows can overlap. No checkpoint was saved or promoted.

Attack the Premise prompted this test after reduced learning rates still worsened steering. Explain the Number kept fitting and monitoring outcomes separate. Prove It Works required exact frozen-state checks and reconstruction from saved predictions.
