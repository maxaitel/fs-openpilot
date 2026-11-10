import json
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.work/lab/direct_command_recovery')
import verify
from fsop.geometry.yuv import rgb_to_model
from fsop.model.temporal import TemporalState

class Head(torch.nn.Module):
    def forward(self):
        return {c: {leaf: torch.full((1, 2), float(n + 1)) for n, leaf in enumerate(('heat', 'wh', 'off'))} for c in ('road', 'wide')}
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.img = Head()
    def policy_raw(self):
        return (torch.tensor([1.0]),)
class Policy:
    def __init__(self):
        self.model = Model(); self.head = object()
    @staticmethod
    def load(*args, **kwargs):
        return Policy()
    def reset(self): pass
    def __call__(self, obs, env):
        self.model.policy_raw(); self.model.img()
        return 0.1, 0.2, {'controller': 'test'}
sys.modules['actuator_trial'] = types.SimpleNamespace(NativeActuatorPolicy=Policy, model_digest=lambda model: 'digest')
class Env:
    track = types.SimpleNamespace(discipline='trackdrive')
    mission = object()
    cfg = types.SimpleNamespace(fps=20)
    def T_world_ego(self): return np.eye(4)
env = Env()
n = 5
images = {name: np.zeros((n, 256, 512, 3), np.uint8) for name in ('road', 'wide')}
rows = [{'tick': i, 'state': [0.] * 9, 'progress_m': 0., 'lateral_m': 0., 'hint': 0, 'desire': [0.] * 8,
         'proposal': [.1,.2], 'prediction': {'controller':'test'}, 'policy_raw': [[1.]], 'image_head': {'index': i}} for i in range(n)]
t = TemporalState(); ds=[]
for i in range(9):
    d=t.push_desire(np.zeros(8, np.float32))
    if i % 4 == 0: ds.append(d)
ep=types.SimpleNamespace(states=np.zeros((10, 9)), desires=np.array(ds), road=np.stack([rgb_to_model(images['road'][i]) for i in (0,4)]), wide=np.stack([rgb_to_model(images['wide'][i]) for i in (0,4)]))
route=[(0.,0.,0.,0)] * 10

def rejects(label, fn):
    try: fn()
    except ValueError as e: print(label, 'REJECTED', str(e)); return
    raise AssertionError(label + ' accepted')

with tempfile.TemporaryDirectory() as d:
    path=Path(d); manifest={}; hashes={}
    for camera, leaf in verify.HEAD_KEYS:
        value=np.full((n, 1, 2), float(('heat','wh','off').index(leaf)+1), np.float32)
        name=f'head_{camera}_{leaf}.npy'; np.save(path/name,value)
        manifest[f'{camera}.{leaf}']={'file':name, 'shape':list(value.shape)}; hashes[name]=verify.digest(path/name)
    meta={'image_head_arrays':manifest, 'arrays_sha256':hashes}
    heads=verify.load_head_arrays(path,meta,n)
    verify.replay_prefix(rows, images, heads, env, None, 'cpu', ['digest','digest'])
    verify.verify_prefix_inputs(rows,images,ep,env,route)
    print('valid synthetic head and packed prefix PASS')
    tampered=np.load(path/'head_road_heat.npy'); tampered[2,0,0] += 1; np.save(path/'head_road_heat.npy',tampered)
    meta['arrays_sha256']['head_road_heat.npy']=verify.digest(path/'head_road_heat.npy')
    heads=verify.load_head_arrays(path,meta,n)
    rejects('head numeric tamper',lambda: verify.replay_prefix(rows,images,heads,env,None,'cpu',['digest','digest']))
    ep.road[1,0,0,0] ^= 1
    rejects('packed image tamper',lambda: verify.verify_prefix_inputs(rows,images,ep,env,route))
    ep.road[1,0,0,0] ^= 1
    rows[1]['desire'][0]=1.
    rejects('route desire tamper',lambda: verify.verify_prefix_inputs(rows,images,ep,env,route))
    rows[1]['desire'][0]=0.
    ep.desires[1,0,0]=1.
    rejects('packed desire tamper',lambda: verify.verify_prefix_inputs(rows,images,ep,env,route))

    ep.desires[1,0,0]=0.
    ep.desires[2,0,0]=1.
    rejects('suffix pooled desire tamper',lambda: verify.verify_prefix_inputs(rows,images,ep,env,route))
