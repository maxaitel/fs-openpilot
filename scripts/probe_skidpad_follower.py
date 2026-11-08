import json
from pathlib import Path

from fsop.eval.closed_loop import mission_brake
from fsop.eval.controller import make_controller
from fsop.sim.env import FSEnv, SimCfg
from fsop.track.fixed_layouts import skidpad


def main():
    rows = []
    for seed in (84000, 84001):
        for level in ('light', 'heavy'):
            for controller in ('expert', 'pp', 'op'):
                cfg = SimCfg(render=False, dr=level, max_time=120, venue_prob=1.0)
                cfg.noise.enabled = False
                env = FSEnv(skidpad(), cfg, seed=seed)
                follower = None if controller == 'expert' else make_controller(controller, env.vp)
                obs = env.observe(render=False)
                while not env.done:
                    steer, accel = env.expert_action() if follower is None else follower(env.label(), obs.state.vx)
                    obs = env.step(steer, mission_brake(accel, obs, env), add_noise=False, render=False)
                row = env.referee.summary()
                row.update(seed=seed, dr=level, controller=controller, plan_source='expert')
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path('runs/spark_skidpad/follower-diagnostic.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
