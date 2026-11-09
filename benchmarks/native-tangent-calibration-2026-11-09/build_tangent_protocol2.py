from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.select_lab_tangent import TRAINING_ARGS, protocol_check

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / '.work/lab/tangent_calibration2/calibration.json'
RUN = ROOT / '.work/lab/tangent_calibration2/run.json'
PROTOCOL = ROOT / 'docs/isaac-lab-training/tangent_selection2.json'
CONFIG = ROOT / '.work/lab/native_tangent_trial2.yaml'
CAPS = {
    'position_interval_m': 0.04,
    'position_interval_max_m': 0.10,
    'position_cumulative_max_m': 0.20,
    'velocity_interval_mps': 0.30,
    'velocity_interval_max_mps': 0.80,
    'velocity_cumulative_max_mps': 2.25,
    'yaw_interval_rad': 0.012,
    'yaw_interval_max_rad': 0.025,
    'yaw_cumulative_max_rad': 0.12,
    'initial_position_m': 1e-6,
    'initial_yaw_rad': 1e-6,
    'initial_velocity_mps': 1.5,
    'initial_yaw_rate_radps': 0.30,
    'velocity_secant_max_mps': 1.0,
    'acceleration_secant_max_mps2': {'mean': 1.0, 'p95': 2.0, 'max': 4.0},
    'yaw_rate_secant_max_radps': 0.25,

}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coherence_bounds(row):
    return {
        metric: {
            stat: min(CAPS[metric][stat] if isinstance(CAPS[metric], dict) else CAPS[metric],
                      max(1e-6, 1.10 * value, 1.25 * row['target_coherence'][metric][stat]))
            for stat, value in stats.items()
        }
        for metric, stats in row['coherence'].items()
    }


cal = json.loads(CALIBRATION.read_text())
run = json.loads(RUN.read_text())
validation = cal['validation']
p = {
    'schema_version': 1,
    'kind': 'op_tangent_v1_selection',
    'calibration_sha256': sha(CALIBRATION),
    'initializer_sha256': cal['identity']['init_sha256'],
    'manifest_sha256': cal['identity']['manifest_sha256'],
    'controller_contract': cal['identity']['controller_contract'],
    'expected_steps': 500,
    'checkpoint_steps': [0, 100, 200, 300, 400, 500],
    'training_args': {key: run['args'][key] for key in TRAINING_ARGS},
    'bounds': {
        'aggregate': {
            'plan_l1': validation['aggregate']['plan_l1'] * 1.05,
            'steer': validation['aggregate']['steer'],
            'longitudinal': validation['aggregate']['longitudinal'],
        },
        'episode': {
            Path(row['path']).name: {key: row[key] * 1.10 for key in ('plan_l1', 'steer', 'longitudinal')}
            for row in validation['episodes']
        },
        'coherence': coherence_bounds(validation),
        'episode_coherence': {Path(row['path']).name: coherence_bounds(row) for row in validation['episodes']},
    },
}
protocol_check(p)
with PROTOCOL.open('x') as f:
    f.write(json.dumps(p, indent=2, allow_nan=False) + '\n')
config = (ROOT / '.work/lab/native_tangent_calibration.yaml').read_text()
config = config.replace('out: runs/lab_v4/tangent_calibration1', 'out: runs/lab_v4/tangent_trial1')
config = config.replace('calibration_only: true', 'calibration_only: false')
config += ('expected_calibration: runs/lab_v4/tangent_calibration2/calibration.json\n'
           'selection_protocol: docs/isaac-lab-training/tangent_selection2.json\n'
           f'selection_protocol_sha: {sha(PROTOCOL)}\n')
with CONFIG.open('x') as f:
    f.write(config)
print(json.dumps({'protocol_sha256': sha(PROTOCOL), 'config_sha256': sha(CONFIG),
                  'calibration_sha256': sha(CALIBRATION), 'aggregate': p['bounds']['aggregate']}, indent=2))
