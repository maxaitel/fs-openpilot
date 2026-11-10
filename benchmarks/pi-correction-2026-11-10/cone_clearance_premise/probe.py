import json
from pathlib import Path
import numpy as np

ROOT = Path('/work/runs/development_pi_trial1_retry1/candidate/heavy_skid')
DATA = Path('/work/data/native/lab_v7')
T = 10 * (np.arange(33) / 32)**2
RADIUS = 0.114

def angular(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def clearance(x, y, yaw, cones):
    dx, dy = cones[:, 0] - x, cones[:, 1] - y
    c, s = np.cos(yaw), np.sin(yaw)
    a, b = c*dx+s*dy-0.15, -s*dx+c*dy
    qx, qy = np.abs(a)-1.45, np.abs(b)-0.48
    signed = np.hypot(np.maximum(qx,0), np.maximum(qy,0)) + np.minimum(np.maximum(qx,qy),0) - RADIUS
    i = int(np.argmin(signed))
    return float(signed[i]), i

def get_frame(n):
    z = np.load(ROOT / 'evidence' / f'frame_{n:06d}.npz')
    rec = json.loads(bytes(z['__record_json__']).decode())
    return rec, z['a4'], z['a25']

def interp(plan, t, col):
    return float(np.interp(t, T, plan[:,col]))

def run_case(frame):
    rec, plan, cones = get_frame(frame)
    st=rec['state']; yaw=st['yaw']; c,s=np.cos(yaw),np.sin(yaw)
    intervals=np.arange(0,2.001,0.05)
    results=[]
    for t in intervals:
        px, py, pyaw = [interp(plan,t,i) for i in (0,1,11)]
        x,y=st['x']+c*px-s*py,st['y']+s*px+c*py
        val, idx=clearance(x,y,yaw+pyaw,cones)
        results.append((val,float(t),idx))
    comparisons=[]
    frames=json.load(open(ROOT/'frames.json'))
    for t in (.2,.6,1.):
        actual=frames[min(frame+round(t/.05),len(frames)-1)]
        dx,dy=actual['x']-st['x'],actual['y']-st['y']
        ax,ay=c*dx+s*dy,-s*dx+c*dy
        px,py,ph=[interp(plan,t,i) for i in (0,1,11)]
        comparisons.append({'t':t,'plan_xy':[px,py],'actual_xy':[ax,ay], 'plan_yaw':ph,'actual_yaw':float(angular(actual['yaw']-yaw)), 'position_error':float(np.hypot(px-ax,py-ay))})
    # Mean velocity-direction change of the predicted path between 0 and 0.6 s.
    def path_angle(t):
        h=.1
        return np.arctan2(interp(plan,t+h,1)-interp(plan,max(t-h,0),1),interp(plan,t+h,0)-interp(plan,max(t-h,0),0))
    path_turn=float(angular(path_angle(.6)-path_angle(.05)))
    psi=interp(plan,.2,11); r0=float(plan[0,14]); v=max(st['vx'],1.)
    curv=(2*psi/.2-r0)/v
    return {'frame':frame,'t':st['t'],'speed':st['vx'],'controller_steer':rec['action'][0], 'yaw_0_2':psi,'yaw_rate_0':r0,'controller_curvature':curv,'path_turn_0_to_0_6':path_turn,'plan_yaw_0_6':interp(plan,.6,11),'min_projected_chassis_disk_clearance_m':min(results), 'comparison':comparisons}

def train_episode(d):
    m=json.load(open(d/'meta.json')); sm=m['summary']
    assert m['split']=='train' and sm['result']=='finished' and sm['pass'] and sm['complete'] and sm['physical_cone_hits']==0 and sm['oc']==0
    a={k:np.load(d/(k+'.npy'), mmap_mode='r') for k in ('states','cones','teacher','contacts','ticks')}
    states, cones, teacher, contacts, ticks=[a[k] for k in ('states','cones','teacher','contacts','ticks')]
    valid=[int(t) for t in ticks if t+200<len(states) and np.all(teacher[t:t+200]) and contacts[t]==contacts[t+200]==0]
    sampled=np.array(valid)[np.linspace(0,len(valid)-1,30,dtype=int)]
    mins=[]; sens=[]
    for tick in sampled:
        points=[]
        for st in states[tick:tick+41]:
            val,idx=clearance(*st[:3],cones)
            points.append((val,st,idx))
        val,st,idx=min(points,key=lambda x:x[0]); mins.append(val)
        eps=.001
        deriv=(clearance(st[0],st[1],st[2]+eps,cones)[0]-clearance(st[0],st[1],st[2]-eps,cones)[0])/(2*eps)
        sens.append(abs(deriv))
    return {'episode':d.name,'qualified_anchor_count':len(valid),'sample_count':len(sampled),'first_tick':int(sampled[0]),'last_tick':int(sampled[-1]),'clearance_min_m':float(min(mins)), 'clearance_median_m':float(np.median(mins)), 'count_clearance_below_0':int(np.sum(np.array(mins)<0)), 'count_clearance_below_0_1':int(np.sum(np.array(mins)<.1)), 'count_clearance_below_0_25':int(np.sum(np.array(mins)<.25)), 'median_absolute_yaw_derivative_m_per_rad':float(np.median(sens)), 'min_absolute_yaw_derivative_m_per_rad':float(min(sens)), 'max_absolute_yaw_derivative_m_per_rad':float(max(sens))}

out={'method':'2D signed distance from cone center to chassis rectangle x=[-1.30,1.60], y=[-0.48,0.48], minus 0.114 m. This is a conservative projected disk envelope and omits wheel and vertical geometry, cone tipping, and continuous collision; not a PhysX contact prediction. Frames use current actual cone poses held fixed over horizon. Future actual frames are closed loop. Plan in initial left-positive ego coordinates.', 'candidate':[run_case(i) for i in (160,170,176,180,184,420,430,438,440,450,480,500)],'train':[train_episode(d) for d in sorted(DATA.glob('train-skidpad-8200*'))[:3]]}
print(json.dumps(out,indent=2))
