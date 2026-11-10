import hashlib, json
from pathlib import Path
import numpy as np
import torch
from fsop.sim.expert import plan_from_rollout
from fsop.geometry.op_frames import ego_to_op_plan, op_to_ego_plan
from fsop.eval.controller import OpenpilotFollower
from fsop.sim.vehicle import VehicleParams
from fsop.train.losses import op_tangent_actions_from_op_plan

root=Path('/work/data/native/lab_v7')
manifest=json.loads((root/'manifest.json').read_text())
result={'description':'TRAIN source anchors only. OP native future versus recorded pre-mission teacher steering, rad. No inference or simulation.', 'cases':[]}
for name in ['train-skidpad-82000-matched','train-skidpad-82002-visual_stress_v1']:
 p=root/name
 raw=(p/'meta.json').read_bytes(); m=json.loads(raw)
 entry=next(e for e in manifest['episodes'] if e['path']==name)
 assert hashlib.sha256(raw).hexdigest()==entry['meta_sha256'] and m['split']=='train'
 arrays={}
 for k in ('states','controls','teacher','contacts','ticks'):
  f=p/(k+'.npy'); assert hashlib.sha256(f.read_bytes()).hexdigest()==m['arrays_sha256'][f.name]
  arrays[k]=np.load(f,allow_pickle=False)
 states,controls,teacher,contacts,ticks=(arrays[k] for k in ('states','controls','teacher','contacts','ticks'))
 assert m['summary']['result']=='finished' and m['summary']['pass'] and m['summary']['complete'] and m['summary']['physical_cone_hits']==0 and m['summary']['oc']==0
 bad=np.r_[0,np.cumsum(~teacher,dtype=np.int64)]
 eligible=[int(t) for t in ticks if t+200<len(states) and bad[t+200]==bad[t] and contacts[t]==contacts[t+200]==0]
 plans=[]
 for t in eligible:
  future=states[t:t+201].astype(np.float64,copy=True); future[:,8]-=future[0,8]
  plans.append(ego_to_op_plan(plan_from_rollout(future,dt=.05)).astype(np.float32))
 plans=np.stack(plans); ss=states[eligible]; vp=VehicleParams(**m['vehicle']); n=len(eligible)
 ft=lambda a:torch.as_tensor(a,dtype=torch.float32)
 with torch.no_grad():
  st,_=op_tangent_actions_from_op_plan(ft(plans),ft(ss[:,3]),ft(ss[:,4]),ft(np.full(n,vp.wheelbase)),ft(np.full(n,vp.max_brake)),ft(np.full(n,vp.max_accel)))
 op=st.numpy(); follower=OpenpilotFollower(vp)
 check=np.array([follower(op_to_ego_plan(plan),float(s[3]))[0] for plan,s in zip(plans,ss)])
 diff=op-controls[eligible,0]; actual=ss[:,6]; yaw=ss[:,5]
 rows=[]
 for i,t in enumerate(eligible):
  rows.append({'tick':t,'time_s':round(float(ss[i,8]),3),'yaw_rate':round(float(yaw[i]),6),'speed':round(float(ss[i,3]),5),'teacher_request':round(float(controls[t,0]),6),'measured_steer':round(float(actual[i]),6),'op_target':round(float(op[i]),6),'error':round(float(diff[i]),6)})
 # Identify turns by state yaw rate, transition by sign change in a surrounding 1 s window.
 masks={'steady_right':[], 'steady_left':[], 'right_to_left_transition':[]}
 for i,t in enumerate(eligible):
  if t>=10 and t+10<len(states):
   window=states[t-10:t+11,5]
   if np.all(window < -0.3): masks['steady_right'].append(i)
   if np.all(window > 0.3): masks['steady_left'].append(i)
   if window[0]<-0.15 and window[-1]>0.15: masks['right_to_left_transition'].append(i)
 def stats(ii):
  d=diff[ii]
  if len(ii)==0:return {'count':0}
  return {'count':len(ii),'time_range_s':[float(ss[ii[0],8]),float(ss[ii[-1],8])], 'mean_signed_rad':float(d.mean()), 'mae_rad':float(np.abs(d).mean()), 'max_abs_rad':float(np.abs(d).max()), 'op_mean_rad':float(op[ii].mean()), 'teacher_mean_rad':float(controls[np.array(eligible)[ii],0].mean()), 'measured_mean_rad':float(actual[ii].mean()), 'sample_rows':[rows[j] for j in (ii[0],ii[len(ii)//2],ii[-1])]}
 result['cases'].append({'case':name,'meta_sha256':entry['meta_sha256'],'input_hashes':{k:m['arrays_sha256'][k+'.npy'] for k in arrays},'eligible_anchor_count':n,'max_follower_disagreement_rad':float(np.max(np.abs(op-check))),'groups':{k:stats(v) for k,v in masks.items()},'all_rows':rows})
print(json.dumps(result,indent=2))
