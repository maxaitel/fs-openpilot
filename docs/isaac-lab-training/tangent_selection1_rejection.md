# Draft 1 rejected before training

Independent review found an exact nullspace in the first draft's trapezoidal plan-consistency gates. A constant-velocity plan with alternating acceleration of +20 and -20 m/s² at consecutive knots has zero integrated acceleration residual. The tangent controller interpolates the instantaneous acceleration, so its command can still change. The draft did not guard against this form of unphysical compensation.

No optimizer updates used draft 1. Its JSON, calibration, configuration and rationale remain as originally written. They are not authorization to run training. The first calibration and the completed camera/inference/video pilot remain valid records of their original source.

Before another training run, the validation code now measures endpoint-to-secant disagreement for velocity versus position, acceleration versus velocity, and yaw rate versus yaw. For each interval it compares each endpoint derivative separately with the finite difference over the interval. It takes the worst endpoint and interval for each anchor, then reports mean, p95 and maximum globally and per case. These checks detect opposite-sign derivatives that cancel in a trapezoidal sum.

Real motion also has endpoint/secant disagreement because the derivative changes across a finite interval. Therefore the revised source requires a new zero-update calibration and a new numerical protocol. The first-second window ends at the last model knot at or below one second, 0.9765625 seconds. It contains both controller interpolation horizons, 0.2 seconds for steering and 0.3 seconds for longitudinal control. These are finite-horizon consistency checks, not a complete proof that every future plan is physically realizable.
