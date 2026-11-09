def plan_coherence(plans, examples):
  """Near-horizon consistency in OP coordinates, per anchor; no acceptance threshold."""
  times = np.asarray(T_IDXS)
  n = int(np.searchsorted(times, 1.0, side='right'))
  dt = np.diff(times[:n])
  rows = []
  for plan, sample in zip(plans, examples, strict=True):
    pos = np.diff(plan[:n, :2], axis=0) - .5 * (plan[:n-1, 3:5] + plan[1:n, 3:5]) * dt[:, None]
    vel = np.diff(plan[:n, 3:5], axis=0) - .5 * (plan[:n-1, 6:8] + plan[1:n, 6:8]) * dt[:, None]
    # The follower consumes raw relative yaw; modulo wrapping would hide a 2π shortcut.
    angle_delta = np.diff(plan[:n, 11])
    yaw = angle_delta - .5 * (plan[:n-1, 14] + plan[1:n, 14]) * dt
    velocity_secant = np.diff(plan[:n, :2], axis=0) / dt[:, None]
    acceleration_secant = np.diff(plan[:n, 3:5], axis=0) / dt[:, None]
    yaw_rate_secant = angle_delta / dt
    velocity_endpoint = np.stack((plan[:n-1, 3:5], plan[1:n, 3:5])) - velocity_secant
    acceleration_endpoint = np.stack((plan[:n-1, 6:8], plan[1:n, 6:8])) - acceleration_secant
    yaw_rate_endpoint = np.stack((plan[:n-1, 14], plan[1:n, 14])) - yaw_rate_secant
    rows.append({'velocity_secant_max_mps': float(np.linalg.norm(velocity_endpoint, axis=-1).max()),
                 'acceleration_secant_max_mps2': float(np.linalg.norm(acceleration_endpoint, axis=-1).max()),
                 'yaw_rate_secant_max_radps': float(np.abs(yaw_rate_endpoint).max()),
                 'position_interval_m': float(np.linalg.norm(pos, axis=1).mean()),
                 'position_interval_max_m': float(np.linalg.norm(pos, axis=1).max()),
                 'velocity_interval_mps': float(np.linalg.norm(vel, axis=1).mean()),
                 'velocity_interval_max_mps': float(np.linalg.norm(vel, axis=1).max()),
                 'yaw_interval_rad': float(np.abs(yaw).mean()),
                 'yaw_interval_max_rad': float(np.abs(yaw).max()),
                 'position_cumulative_max_m': float(np.linalg.norm(np.cumsum(pos, axis=0), axis=1).max()),
                 'velocity_cumulative_max_mps': float(np.linalg.norm(np.cumsum(vel, axis=0), axis=1).max()),
                 'yaw_cumulative_max_rad': float(np.abs(np.cumsum(yaw)).max()),
                 'initial_position_m': float(np.linalg.norm(plan[0, :2])),
                 'initial_yaw_rad': float(abs(plan[0, 11])),
                 'initial_velocity_mps': float(np.linalg.norm(
                     plan[0, 3:5] - [sample['speed'], -sample['lateral_speed']])),
                 'initial_yaw_rate_radps': float(abs(plan[0, 14] + sample['yaw_rate']))})
  return rows
