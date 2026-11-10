import json
from pathlib import Path
import numpy as np
from fsop.sim.expert import plan_from_rollout, T_IDXS
from fsop.sim.vehicle import VehicleParams
from fsop.eval.controller import OpenpilotFollower
from fsop.geometry.op_frames import ego_to_op_plan
out=[]
for name,ticks in [('train-skidpad-82000-matched',[268,456,464,664,668]),('train-skidpad-82002-visual_stress_v1',[140,452,464,652,656])]:
 p=Path('/work/data/native/lab_v7')/name
 m=json.loads((p/'meta.json').read_text()); s=np.load(p/'states.npy'); u=np.load(p/'controls.npy'); vp=VehicleParams(**m['vehicle'])
 for t in ticks:
  future=s[t:t+201].copy();future[:,8]-=future[0,8];plan=plan_from_rollout(future,.05); yaw=float(np.interp(.2,T_IDXS,plan[:,11])); rate=float(plan[0,14]);pred=OpenpilotFollower(vp)(plan,float(s[t,3]))[0]
  out.append({'case':name,'tick':t,'time_s':float(s[t,8]),'wheelbase':vp.wheelbase,'body_speed':float(s[t,3]),'plan_yaw_0_2':yaw,'mean_heading_rate_0_2':yaw/.2,'instantaneous_yaw_rate':rate,'raw_heading_change_rate':float(np.unwrap(s[t:t+5,2])[-1]-s[t,2])/.2,'teacher_steer':float(u[t,0]),'actual_steer':float(s[t,6]),'op_steer':pred})
print(json.dumps(out,indent=2))
