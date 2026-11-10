# Native transition intervention

Throwaway privileged TRAIN diagnostic. The inherited OP plus PI reference completed R,R,L,L with one physical contact. At that contact its lateral error was about 0.295 m, while nearby spatial matching left essentially the same residual. The first post-branch command differs even with identical pose. This experiment separates missing position feedback from longitudinal drift.

The three fixed arms add path-normal feedback to OP with unchanged PI, retain OP with recorded longitudinal commands, or combine both. The position gain is the existing expert default 2.0, fixed without a tuning sweep. All arms reproduce the recorded prefix exactly. Frozen protocol, source bytes, native runtime, initial scene, physics, control rate and horizon are verified. The original full replay is reused with hash-bound receipts because it already proved all 1,128 states exactly.

Keep an arm only as an execution hypothesis if it completes the full mission with accepted R,R,L,L, no physical contact, no off-course event and a valid stop under the original horizon. All other results remain rejected. One TRAIN source is not robustness or vision-model performance. A correction based on plan time-zero offset may not transfer to a learned plan trained to start at zero. Test that separately before any integration.

GPU work is serialized. Three render-free native suffixes produce compact traces, not videos or neural inference. This is not a timing comparison. No production code changes.
