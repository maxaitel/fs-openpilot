import json
from pathlib import Path
import numpy as np
root=Path('/work/runs/development_pi_trial1_retry1/candidate/heavy_skid/evidence')
T=10*(np.arange(33)/32)**2

def distance(point,cx,cy,ang):
 x,y=point[0]-cx,point[1]-cy; c,s=np.cos(ang),np.sin(ang)
 qx,qy=abs(c*x+s*y)-.23,abs(-s*x+c*y)-.1
 return np.hypot(max(qx,0),max(qy,0))+min(max(qx,qy),0)-.095

out=[]
for frame,ci in [(160,37),(170,37),(176,37),(180,37),(420,30),(430,30),(438,30),(440,30),(450,31),(480,21),(500,21)]:
 z=np.load(root/f'frame_{frame:06d}.npz'); r=json.loads(bytes(z['__record_json__']).decode()); st=r['state']; p=z['a25'][ci,:2]; plan=z['a4']; dx,dy=p[0]-st['x'],p[1]-st['y']; c,s=np.cos(st['yaw']),np.sin(st['yaw']); pc=np.array([c*dx+s*dy,-s*dx+c*dy]); steer=r['action'][0]; k=np.tan(steer)/1.55
 vals=[]
 for t in np.arange(0,2.001,.01):
  x,y,yaw=[np.interp(t,T,plan[:,j]) for j in (0,1,11)]; dx,dy=pc[0]-x,pc[1]-y; c,s=np.cos(yaw),np.sin(yaw); ep=np.array([c*dx+s*dy,-s*dx+c*dy]);
  for name,wx,wy in [('fl',.8,.6),('fr',.8,-.6),('rl',-.75,.6),('rr',-.75,-.6)]:
   angle=np.arctan2(1.55*k,1-k*wy) if wx>0 else 0
   vals.append((float(distance(ep,wx,wy,angle)),float(t),name))
 out.append({'frame':frame,'cone':ci,'constant_steer':steer,'min_wheel_projected_disk_clearance_m':min(vals)})
print(json.dumps(out,indent=2))
