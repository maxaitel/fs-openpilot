LEGACY_ACTION_TARGET = 'teacher_command_v1'
NATIVE_FUTURE_ACTION_TARGET = 'native_future_v1'
NATIVE_FUTURE_CONTRACT = {
  'version': 1,
  'mode': NATIVE_FUTURE_ACTION_TARGET,
  'source': ('detached native future plan through op_tangent_v1 before mission braking, '
             'with measured state and vehicle limits'),
}


def action_target_contract(mode, controller_mode):
  if mode == LEGACY_ACTION_TARGET:
    return None
  if mode == NATIVE_FUTURE_ACTION_TARGET and controller_mode == 'op_tangent_v1':
    return NATIVE_FUTURE_CONTRACT.copy()
  raise ValueError('unsupported action target mode for controller')
