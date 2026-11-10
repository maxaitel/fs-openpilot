# Forward arithmetic check

This zero-update experiment isolates the initial-output difference that aborted the first head-scope attempt. On the same exact model state, alternating full-policy and head-only requires_grad flags reproduces two prediction arrays. Restoring full-policy flags exactly restores the retained control. Repeating each setting is exact. Two fresh processes with opposite data/model loading orders produce identical arrays. The maximum plan-mean difference is 0.0000128746. Vision features remain identical. The specific backend kernel mechanism has not been established.

The first head experiment remains failed with zero updates. The follow-up preserves the original forward flags, computes gradients only for optimizer-selected leaf parameters, and keeps the original strict starting comparison unchanged. No numerical tolerance was relaxed.

Both result files and their compressed prediction arrays were copied and SHA-256 checked. Full arrays remain in `.work/lab/head_forward_parity/`. No weights changed and no checkpoints were produced.
