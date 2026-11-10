import json
from pathlib import Path
import numpy as np
root=Path('/work/runs/development_pi_trial1_retry1/candidate/heavy_skid'); frames=json.load(open(root/'frames.json')); T=10*(np.arange(33)/32)**2

def wrap(x): return float((x+np.pi)%(2*np.pi)-np.pi)
def plan_ang(p,t):
 a=max(t-.1,0); b=t+.1
 return np.arctan2(np.interp(b,T,p[:,1])-np.interp(a,T,p[:,1]),np.interp(b,T,p[:,0])-np.interp(a,T,p[:,0]))
def real_ang(f,t):
 a=f+round(max(t-.1,0)/.05); b=f+round((t+.1)/.05)
 return np.arctan2(frames[b]['y']-frames[a]['y'],frames[b]['x']-frames[a]['x'])
rows=[]
for f in (160,170,176,420,430,438,440,450,480,500):
 z=np.load(root/'evidence'/f'frame_{f:06d}.npz'); p=z['a4']; r=json.loads(bytes(z['__record_json__']).decode());
 pred_turn=wrap(plan_ang(p,.6)-plan_ang(p,.05)); real_turn=wrap(real_ang(f,.6)-real_ang(f,.05)); pred_yaw=float(np.interp(.6,T,p[:,11])); real_yaw=wrap(frames[f+12]['yaw']-frames[f]['yaw'])
 rows.append({'frame':f,'time':r['state']['t'],'plan_path_turn_0_to_0_6':pred_turn,'actual_path_turn_0_to_0_6':real_turn,'plan_yaw_0_6':pred_yaw,'actual_yaw_0_6':real_yaw,'plan_path_minus_yaw':wrap(pred_turn-pred_yaw),'actual_path_minus_yaw':wrap(real_turn-real_yaw)})
print(json.dumps(rows,indent=2))
