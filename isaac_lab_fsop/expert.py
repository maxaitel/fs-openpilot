"""Feedback teacher driven by measured native vehicle state."""

from dataclasses import replace

import numpy as np

from fsop.sim.expert import Expert, ExpertParams, Mission


class NativeExpert:
    def __init__(self, env, params=None, controller=None):
        self.env = getattr(env, "env", env)
        self.params = replace(params) if params is not None else ExpertParams(v_cap=8.0, a_lat_max=4.0)
        if controller is not None:
            self.params.controller = controller
        if self.params.controller not in ("pure_pursuit", "stanley_ff"):
            raise ValueError("controller must be pure_pursuit or stanley_ff")
        self.mission = Mission(self.env.track, self.params)
        self.expert = Expert(self.mission, self.env.vp)

    def act(self, drive=None):
        env = self.env if drive is None else getattr(drive, "env", drive)
        steer, accel = self.expert.act(env.state, env.s)
        if not np.isfinite([steer, accel]).all():
            raise ValueError("Native teacher produced a nonfinite action")
        return float(steer), float(accel)
