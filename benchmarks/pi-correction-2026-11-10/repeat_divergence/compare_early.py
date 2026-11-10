import hashlib,json,zipfile
from pathlib import Path
import numpy as np
root=Path('/work/runs')
names=['pi_latest_heavy1','development_pi_trial1/baseline/heavy_skid']
paths=[root/n/'evidence' for n in names]
manifest=[[json.loads(l) for l in (p/'manifest.jsonl').read_text().splitlines() if l.strip()] for p in paths]
out={'names':names,'manifest_entries':list(map(len,manifest)),'same_input_hash_early100':sum(a['input_sha256']==b['input_sha256'] for a,b in zip(manifest[0][:100],manifest[1][:100])),'frames':[]}
def unpack(v,z):
 if isinstance(v,dict):
  if set(v)=={'__array__'}:return z[v['__array__']]
  return {k:unpack(x,z) for k,x in v.items()}
 if isinstance(v,list):return [unpack(x,z) for x in v]
 return v
def stats(a,b):
 a,b=np.asarray(a),np.asarray(b)
 if a.shape!=b.shape:return {'shapes':[a.shape,b.shape]}
 d=a.astype(float)-b.astype(float)
 return {'equal':bool(np.array_equal(a,b)),'mae':float(np.abs(d).mean()),'max_abs':float(np.abs(d).max()),'different_fraction':float((d!=0).mean())}
for i in list(range(11))+[20,50,99]:
 rec=[]
 for p,m in zip(paths,manifest):
  f=p/m[i]['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==m[i]['sha256']
  with np.load(f,allow_pickle=False) as z:rec.append(unpack(json.loads(z['__record_json__'].tobytes()),z))
 a,b=rec
 row={'frame':i,'states':[a['state'],b['state']],'actions':[a['action'],b['action']],'state_equal':a['state']==b['state'],'progress':[a['progress'],b['progress']],'timings':[a['diagnostics'],b['diagnostics']], 'arrays':{}}
 for field in ['raw_images','images','body']:
  for k in a[field]:row['arrays'][field+'.'+k]=stats(a[field][k],b[field][k])
 for k in ['actual_cones_xyz','actual_cone_quaternions','T_plan_ego_camera','T_world_camera_body','nominal_detection_calibration']:
  row['arrays'][k]=stats(a[k],b[k])
 row['capture_state_body_position_max_abs']=[float(np.max(np.abs(np.asarray([x['state']['x'],x['state']['y']])-x['body']['position'][:2]))) for x in rec]
 row['capture_state_body_yaw_abs']=[abs(float(np.arctan2(x['body']['rotation'][1,0],x['body']['rotation'][0,0]))-x['state']['yaw']) for x in rec]
 row['arrays']['prediction.plan']=stats(a['prediction']['plan'],b['prediction']['plan'])
 row['arrays']['policy_raw.0']=stats(a['policy_raw'][0],b['policy_raw'][0])
 out['frames'].append(row)
print(json.dumps(out,indent=2))
