import json
from pathlib import Path
import numpy as np
root=Path('/work/runs/development_pi_trial1_retry1/candidate/heavy_skid/evidence')

def distance(point, cx, cy, ang, hl, hw):
 x,y=point[0]-cx,point[1]-cy
 c,s=np.cos(ang),np.sin(ang)
 qx,qy=abs(c*x+s*y)-hl,abs(-s*x+c*y)-hw
 return np.hypot(max(qx,0),max(qy,0))+min(max(qx,qy),0)

for frame in (160,170,176,180,182,183,184,185,186,430,438,440,441,442,443,444,450,453,454):
 z=np.load(root/f'frame_{frame:06d}.npz'); r=json.loads(bytes(z['__record_json__']).decode()); st=r['state']; cone_i=37 if frame<400 else (30 if frame<450 else 31); p=z['a25'][cone_i,:2]; dx,dy=p[0]-st['x'],p[1]-st['y']; c,s=np.cos(st['yaw']),np.sin(st['yaw']); e=np.array([c*dx+s*dy,-s*dx+c*dy]); steer=st['steer']; k=np.tan(steer)/1.55
 vals={}
 for label,x,y in [('fl',.8,.6),('fr',.8,-.6),('rl',-.75,.6),('rr',-.75,-.6)]:
  ang=np.arctan2(1.55*k,1-k*y) if x>0 else 0
  vals[label]=round(float(distance(e,x,y,ang,.23,.1)-.095),3)
 body=distance(e,.15,0,0,1.45,.48)-.114
 print(json.dumps({'frame':frame,'t':st['t'],'cone':cone_i,'cone_xy_ego':e.tolist(),'wheel_planar_clearances_m':vals,'chassis_projected_disk_clearance_m':float(body),'cone_world':p.tolist()}))
