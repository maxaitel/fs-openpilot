import json
from pathlib import Path
import numpy as np
root=Path('/work/runs/development_pi_trial1_retry1/candidate/heavy_skid/evidence'); out=[]
for f,ci in [(160,37),(170,37),(176,37),(180,37),(420,30),(430,30),(438,30),(440,30),(450,31),(480,21),(500,21)]:
 z=np.load(root/f'frame_{f:06d}.npz'); r=json.loads(bytes(z['__record_json__']).decode()); st=r['state']; p=z['a25'][ci,:2]; dx,dy=p[0]-st['x'],p[1]-st['y']; c,s=np.cos(st['yaw']),np.sin(st['yaw']); e=np.array([c*dx+s*dy,-s*dx+c*dy]); pred=z['a6']; ds=np.linalg.norm(pred-e,axis=1); j=int(np.argmin(ds)) if len(ds) else None
 out.append({'frame':f,'time':st['t'],'cone':ci,'cone_xy_ego':e.tolist(),'n_detections':len(pred),'nearest_detection':None if j is None else {'xy':pred[j].tolist(),'distance_m':float(ds[j]),'class':int(z['a7'][j]),'score':float(z['a8'][j])}})
print(json.dumps(out,indent=2))
