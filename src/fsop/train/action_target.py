LEGACY_ACTION_TARGET = 'teacher_command_v1'
NATIVE_FUTURE_ACTION_TARGET = 'native_future_v1'
DESIRED_MOTION_TARGET = 'desired_motion_v1'
DESIRED_MOTION_CONTRACT = {
  'version': 1,
  'mode': DESIRED_MOTION_TARGET,
  'source': 'detached native future through the stateless tangent motion request before PI, mission override, and actuator clipping',
}
NATIVE_FUTURE_CONTRACT = {
  'version': 1,
  'mode': NATIVE_FUTURE_ACTION_TARGET,
  'source': ('detached native future plan through op_tangent_v1 before mission braking, '
             'with measured state and vehicle limits'),
}


def action_target_contract(mode, controller_mode):
  if mode == DESIRED_MOTION_TARGET and controller_mode == 'op_tangent_pi_v1':
    return DESIRED_MOTION_CONTRACT.copy()
  if mode == LEGACY_ACTION_TARGET and controller_mode != 'op_tangent_pi_v1':
    return None
  if mode == NATIVE_FUTURE_ACTION_TARGET and controller_mode == 'op_tangent_v1':
    return NATIVE_FUTURE_CONTRACT.copy()
  raise ValueError('unsupported action target mode for controller')
