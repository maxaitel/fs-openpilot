# Skidpad design decision

The model must choose the next skidpad branch from camera features and route commands. The scoring must prove the route independently of simulator progress.

Two candidates considered a refreshed whole-phase intent and a refreshed next-junction request. The comparison used four criteria, scored from one to five.

| Criterion | Whole-phase intent | Next-junction request |
|---|---:|---:|
| Unambiguous meaning at repeated crossings | 3 | 5 |
| Existing checkpoint and pulse compatibility | 5 | 5 |
| Independent physical validation | 5 | 5 |
| Small, traceable implementation | 5 | 4 |

Choose the next-junction request. Derive five decisions from the existing route metadata. Emit a one-tick pulse when the car approaches a decision, repeat about once a second while waiting, and release after passing it. Read the cached current-frame command from observations. This preserves the model input and temporal queue. The command means the next outgoing branch, including a repeated right or left branch. It is not an instruction to steer immediately.

Adopt the first candidate's emphasis on exact recorder/live queue comparison and old-checkpoint measurements before training. Keep the second candidate's physical route classifier. It must use observed positions and signed circle traversal, directional finite gate crossings, and a stop in the exit zone. It must reject wrong circles, reversed crossings, and repeated gate oscillations. Keep cone penalties visible.

The old model already benefits from commands on expert-driven images. Its first live deviation occurs before pulse expiry. Command delivery is therefore one repair, not a complete diagnosis. New data must include recovery excursions and increased skidpad representation. Start with frozen-vision policy adaptation plus original trackdrive replay. Change the vision trunk only if validation justifies it.

No available configured judge is from another model family. The parent scored both designs and selected the base. A later independent review will check the code and evidence without claiming cross-family review.

The implementation keeps command scheduling and physical scoring in separate owners. The scoring receives no mission progress or command state. Other disciplines retain zero route commands and their existing lap scoring. Standstill detection now uses total speed, so reverse motion does not count as stationary. The final model still follows its predicted plan with the existing controller.
