#!/usr/bin/env python3
"""Analyze saved tangent outputs without model inference or source changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path('.work/lab/tangent_residual1'))
args = parser.parse_args()
root = args.input
report = json.loads((root/'report.json').read_text())
anchors = json.loads((root/'anchors.json').read_text())
for name, expected in report['output_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
z = np.load(root/'arrays.npz', allow_pickle=False)
times = z['times_s']
dt = z['interval_dt_s']
target = z['target']
cold = np.array([row['usable_history_frames'] < 8 for row in anchors])
full = ~cold
assert len(anchors) == 256 and cold.sum() == 13
stats = lambda x: dict(mean=float(np.mean(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
summary = {'scope': 'Descriptive saved-output analysis. No training, new cases, physical evaluation or promotion.',
           'inputs_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                            for name in ('report.json','anchors.json','arrays.npz')},
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'models': {}, 'output_ablation': {}, 'action_target_comparison': {}}
for name in ('step0','step500','native'):
    summary['action_target_comparison'][name] = {}
    for group,mask in (('all',np.ones(256,bool)),('history_lt8',cold),('history_8',full)):
        summary['action_target_comparison'][name][group] = {}
        for label,key in (('teacher','teacher_actions'),('native_controller','native_actions')):
            error = z[name+'_actions'][mask]-z[key][mask]
            loss = np.where(abs(error)<.05,.5*error**2/.05,abs(error)-.025)
            summary['action_target_comparison'][name][group][label] = {
                'mae_steer_longitudinal':abs(error).mean(0).tolist(),
                'smooth_l1_steer_longitudinal':loss.mean(0).tolist()}
rows = np.arange(256)
all_maximum = {}
for name in ('step0','step500'):
    mu = z[name+'_mu']
    residual = z[name+'_acceleration_endpoint_residual_mps2']
    native_residual = z['native_acceleration_endpoint_residual_mps2']
    aerror = z[name+'_acceleration_endpoint_error_to_native_mps2']
    verror = z[name+'_velocity_secant_error_to_native_mps2']
    np.testing.assert_allclose(residual-native_residual, aerror-verror[:,:,None,:], atol=1e-5, rtol=1e-6)
    norms = np.linalg.norm(residual, axis=-1)
    maximum = norms.max(axis=(1,2))
    all_maximum[name] = maximum
    np.testing.assert_allclose(list(stats(maximum).values()), list(report['models'][name]['groups']['all']['acceleration_secant'].values()))
    locations = np.array([np.unravel_index(np.argmax(row), row.shape) for row in norms])
    wr = residual[rows,locations[:,0],locations[:,1]]
    wy = native_residual[rows,locations[:,0],locations[:,1]]
    wa = aerror[rows,locations[:,0],locations[:,1]]
    wv = verror[rows,locations[:,0]]
    excess = wr-wy
    direction = excess / np.maximum(np.linalg.norm(excess,axis=-1)[:,None],1e-12)
    masks = {'all':np.ones(256,bool),'history_lt8':cold,'history_8':full,
             'above4':maximum>4,'history_8_above4':full&(maximum>4)}
    result = {'worst_interval_counts':dict(Counter(locations[:,0].tolist())), 'groups':{},
              'above4_anchors':[anchors[int(i)] for i in np.flatnonzero(maximum>4)],
              'near_knot_values':{}, 'nll_scales':{}}
    for group,mask in masks.items():
        result['groups'][group] = {'count':int(mask.sum()),'acceleration_secant':stats(maximum[mask]),
            'above4':int(np.sum(maximum[mask]>4)),
            'worst_interval_counts':dict(Counter(locations[mask,0].tolist())),
            'mean_acceleration_error_norm':float(np.linalg.norm(wa[mask],axis=-1).mean()),
            'mean_velocity_secant_error_norm':float(np.linalg.norm(wv[mask],axis=-1).mean()),
            'mean_acceleration_error_projected_on_excess':float((wa[mask]*direction[mask]).sum(-1).mean()),
            'mean_negative_velocity_secant_error_projected_on_excess':float((-wv[mask]*direction[mask]).sum(-1).mean()),
            'x_fraction_of_sum_absolute_residual_components':float(abs(wr[mask,0]).sum()/abs(wr[mask]).sum())}
    for group,mask in (('history_lt8',cold),('history_8',full)):
        result['near_knot_values'][group] = {'mean_vx':mu[mask,:11,3].mean(0).tolist(),
            'mean_vx_error':(mu-target)[mask,:11,3].mean(0).tolist(),
            'mean_ax_error':(mu-target)[mask,:11,6].mean(0).tolist(),
            'mean_abs_vx_error':abs(mu-target)[mask,:11,3].mean(0).tolist(),
            'mean_abs_ax_error':abs(mu-target)[mask,:11,6].mean(0).tolist()}
        result['nll_scales'][group] = {str(k):{str(c):{
            'mean_logstd':float(z[name+'_logstd'][mask,k,c].mean()),
            'mean_effective_mu_scale':float(z[name+'_nll_mu_gradient_scale'][mask,k,c].mean())}
            for c in (3,6)} for k in (0,4,5,6,7,10)}
    summary['models'][name] = result

mu = z['step500_mu']
changed = mu.copy()
for k in (5,6):
    weight = (times[k]-times[4])/(times[7]-times[4])
    changed[:,k,3:5] = (1-weight)*mu[:,4,3:5]+weight*mu[:,7,3:5]
logb = np.clip(z['step500_logstd'],-7,7)
weights = np.where(times<=4,1.,np.maximum(.2,1-.8*(times-4)/6))[:,None] * np.array([1,1,.1,.5,.5,.1,.1,.1,0,0,0,.5,0,0,.3])[None,:]
for name,plan in (('actual',mu),('interpolate_own_velocity_knots5_6',changed)):
    secant = np.diff(plan[:,:11,3:5],axis=1)/dt[None,:,None]
    endpoints = np.stack((plan[:,:10,6:8],plan[:,1:11,6:8]),axis=2)
    maxima = np.linalg.norm(endpoints-secant[:,:,None,:],axis=-1).max(axis=(1,2))
    speed = np.linalg.norm(plan[:,:,3:5],axis=-1)
    tangent = (plan[:,:,3:5]*plan[:,:,6:8]).sum(-1)/np.maximum(speed,.001)
    raw = 1.5*(.48*speed[:,5]+.52*speed[:,6]-np.linalg.norm(z['observed_vx_vy_yaw_rate'][:,:2],axis=-1)) + .5*(.48*tangent[:,5]+.52*tangent[:,6])
    error = abs(raw-z['teacher_actions'][:,1])
    long_loss = np.where(error<.05,.5*error**2/.05,error-.025)
    new_error = abs(raw-z['native_actions'][:,1])
    new_long_loss = np.where(new_error<.05,.5*new_error**2/.05,new_error-.025)
    nll = ((abs(plan-target)*np.exp(-logb)+logb)*weights[None]).sum((1,2))/weights.sum()
    if name=='actual':
        np.testing.assert_allclose(raw,z['step500_actions'][:,1],atol=4e-6,rtol=1e-6)
    else:
        assert (raw[cold]>=0).all() and (raw[cold] <= z['step500_actions'][cold,1]).all()
    summary['output_ablation'][name] = {group:{'count':int(mask.sum()),'acceleration_secant':stats(maxima[mask]),
        'above4':int((maxima[mask]>4).sum()),'plan_nll_with_frozen_logstd':float(nll[mask].mean()),
        'unclipped_longitudinal_smooth_l1':float(long_loss[mask].mean()),
        'controller_target_longitudinal_smooth_l1':float(new_long_loss[mask].mean()),
        'nll_plus_unclipped_longitudinal':float((nll+long_loss)[mask].mean()),
        'nll_plus_controller_target_longitudinal':float((nll+new_long_loss)[mask].mean())}
        for group,mask in (('all',np.ones(256,bool)),('history_lt8',cold),('history_8',full))}
summary['output_ablation']['limits'] = [
    'Post-hoc output intervention only. It is not a trained model, controller correction, or eligibility result.',
    'Only own velocity knots5/6 change. Yaw, acceleration, all other knots and uncertainty remain fixed.',
    'No native target is used for interpolation. Native labels are used only to score the output.',
    'Cold-anchor interpolated commands are positive and below their actual unclipped commands. Thus clipping cannot alter the cold comparison.',
    'Losses outside the cold group use raw commands. No vehicle-limit assumption is made for altered full-history commands.',
]
representatives = [int(np.flatnonzero(cold)[np.argmax(all_maximum['step500'][cold])]),
                   int(np.flatnonzero(full)[np.argmax(all_maximum['step500'][full])])]
summary['representatives'] = []
for index in representatives:
    row = {'anchor':anchors[index], 'times_s':times[:11].tolist(),
           'predictions':{name:{'vx':z[name+'_mu'][index,:11,3].tolist(),
                                'ax':z[name+'_mu'][index,:11,6].tolist()}
                          for name in ('step0','step500','native')}}
    summary['representatives'].append(row)

plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
colors = {'step0':'#777777','step500':'#2166ac','native':'#238b45'}
fig,axes = plt.subplots(2,2,figsize=(13,9))
for name in ('step0','step500','native'):
    values = np.linalg.norm(z[name+'_acceleration_endpoint_residual_mps2'],axis=-1).max(axis=-1)
    axes[0,0].plot(np.arange(10),np.percentile(values,95,axis=0),marker='o',label=name,color=colors[name])
axes[0,0].set(title='Per-interval 95th percentile across 256 anchors',xlabel='Velocity interval index',ylabel='Endpoint residual (m/s²)',xticks=np.arange(10))
axes[0,0].legend()
for ax,group,mask in ((axes[0,1],'13 anchors with fewer than 8 history frames',cold),
                       (axes[1,0],'243 anchors with 8 history frames',full)):
    for name in ('step0','step500'):
        ax.plot(times[:11],(z[name+'_mu']-target)[mask,:11,3].mean(0),marker='o',label=name,color=colors[name])
    ax.axhline(0,color=colors['native'],label='native target')
    ax.axvspan(times[5],times[6],color='#dfb23f',alpha=.15,label='longitudinal interpolation')
    ax.set(title=group,xlabel='Plan time (s)',ylabel='Mean forward-velocity error (m/s)')
axes[0,1].legend(fontsize=8)
axes[1,1].scatter(all_maximum['step0'][full],all_maximum['step500'][full],s=18,alpha=.65,label='Full history',color=colors['step500'])
axes[1,1].scatter(all_maximum['step0'][cold],all_maximum['step500'][cold],s=35,marker='x',label='Fewer than 8 frames',color='#d95f02')
axes[1,1].plot([0,9.3],[0,9.3],color='grey',lw=.7)
axes[1,1].axhline(4,color='#b2182b',ls='--',lw=.8,label='Frozen maximum cap')
axes[1,1].set(title='Paired anchor maxima',xlabel='Step 0 (m/s²)',ylabel='Step 500 (m/s²)')
axes[1,1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(root/'residual_localization.png',dpi=150)
plt.close(fig)

fig,axes = plt.subplots(2,3,figsize=(16,8))
for row,index in enumerate(representatives):
    anchor = anchors[index]
    for name in ('step0','step500','native'):
        plan = z[name+'_mu'][index]
        axes[row,0].plot(times[:11],plan[:11,3],marker='o',label=name,color=colors[name])
        axes[row,1].plot(times[:11],plan[:11,6],marker='o',label=name,color=colors[name])
        res = z[name+'_acceleration_endpoint_residual_mps2'][index]
        axes[row,2].plot(np.arange(10),np.linalg.norm(res,axis=-1).max(axis=-1),marker='o',label=name,color=colors[name])
    for col in (0,1):
        axes[row,col].axvspan(times[5],times[6],color='#dfb23f',alpha=.15)
        axes[row,col].set_xlabel('Plan time (s)')
    axes[row,0].set_ylabel('Forward velocity (m/s)')
    axes[row,1].set_ylabel('Forward acceleration (m/s²)')
    axes[row,2].set(xlabel='Velocity interval index',ylabel='Endpoint residual (m/s²)',xticks=np.arange(10))
    axes[row,0].set_title(anchor['case'].replace('val-','')+' / frame '+str(anchor['frame']),fontsize=9)
    axes[row,1].set_title('Acceleration head')
    axes[row,2].set_title('Maximum of the two endpoint residuals')
axes[0,0].legend()
fig.suptitle('Largest cold-history and full-history residuals. Amber marks controller velocity knots 5 and 6.')
fig.tight_layout()
fig.savefig(root/'representative_plans.png',dpi=150)
plt.close(fig)
summary['plot_sha256'] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ('residual_localization.png','representative_plans.png')}
(root/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'representative_anchors':representatives,'above4_step500':summary['models']['step500']['groups']['above4'],
                  'output':str(root/'analysis.json')},indent=2))
