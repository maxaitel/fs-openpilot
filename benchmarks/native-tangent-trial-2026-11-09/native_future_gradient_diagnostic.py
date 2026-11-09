#!/usr/bin/env python3
"""Compare old and native-future action targets on four fixed TRAIN sampler batches."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

import fsop
from fsop.data.native_replay import NativeEpisode, NativeReplay, episode_paths
from fsop.model.build import load_fs
from fsop.sim.expert import T_IDXS
from fsop.train.lab_train import RecoverySampler, batch_loss, configure, dataset_verification, prove_policy_context, sha256
from fsop.train.losses import op_tangent_actions_from_op_plan


def all_state_hashes(model):
    result = {}
    for name,tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        result[name] = {'shape':list(value.shape),'dtype':str(value.dtype),
                       'sha256':hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def cosine(left,right):
    denominator = float(left.norm()*right.norm())
    return None if denominator == 0 else float(torch.dot(left,right)/denominator)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--training',type=Path,required=True,help='Completed original tangent_trial1, not a proposed new run')
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
args = parser.parse_args()
signal.alarm(600)
start = time.monotonic()
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device(args.device)
run_path = args.training/'run.json'
run = json.loads(run_path.read_text())
config = run['args']
assert config['controller_mode']=='op_tangent_v1' and config.get('action_target_mode','teacher_command_v1')=='teacher_command_v1'
assert config['freeze_vision'] and config['history_context']==8 and config['batch_size']==128
assert not config['amp'] and not config.get('cache_frozen_features',False)
assert (config['w_plan'],config['w_steer'],config['w_long'])==(1.,3.281158240898283,1.)
assert config['seed']==0 and config['recovery_draws']==16
history = [json.loads(line) for line in (args.training/'history.jsonl').read_text().splitlines() if line]
assert [row['step'] for row in history]==list(range(1,501))
source_root = Path(fsop.__file__).resolve().parent.parent
for name,digest in run['source']['files_sha256'].items():
    assert sha256(source_root/name)==digest, f'Requires original frozen runtime source: {name}'
assert sha256(args.training/'source.zip')==run['source']['archive_sha256']
assert sha256(config['manifest'])==run['manifest_sha256']
assert sha256(config['init'])==run['init_sha256']=='e59efc8ace2d05a5ecfc29023712e7b5ce1d34a64133dd869a7bc17f849bc35f'
verification = dataset_verification(config['manifest'],config.get('dataset_verification'))
assert verification==run['dataset_verification']
paths = episode_paths(manifest=config['manifest'])
train_paths = [path for path in paths if json.loads((path/'meta.json').read_text())['split']=='train']
expected = [row for row in run['episodes'] if row['split']=='train']
assert [str(path) for path in train_paths]==[row['path'] for row in expected]
for path,row in zip(train_paths,expected,strict=True):
    assert sha256(path/'meta.json')==row['meta_sha256']
episodes = [NativeEpisode(path) for path in train_paths]
dataset = NativeReplay(episodes,'train')
assert len(dataset)==run['train_anchors']
sampler = RecoverySampler(dataset,config['recovery_cases'],config['recovery_gates'],128,16,run['init_sha256'],
    provenance_mode=config['recovery_provenance_mode'],collection_policy_checkpoint_sha=config['collection_policy_checkpoint_sha'])
sampler_protocol = sampler.protocol()
canonical = lambda document: {**document,'recovery_episodes':[
    {key:value for key,value in row.items() if key!='episode'} for row in document['recovery_episodes']]}
assert canonical(sampler_protocol)==canonical(run['recovery_sampling'])
rng = np.random.default_rng(0)
batches = [sampler.draw(rng) for _ in range(4)]
inventory = []
for batch_index,indices in enumerate(batches):
    draws = []
    for draw_index,index in enumerate(indices):
        ei,frame = dataset.index[int(index)]
        ep = episodes[ei]
        tick = int(ep.ticks[frame])
        draws.append({'batch':batch_index,'draw':draw_index,'dataset_index':int(index),'case':ep.meta['id'],
            'episode':ei,'frame':int(frame),'tick':tick,'history_frames':min(frame,8),
            'is_recovery_draw':int(index) in sampler.recovery_indices,'state':ep.states[tick].tolist(),
            'teacher_action':ep.controls[tick].tolist()})
    assert sum(row['is_recovery_draw'] for row in draws)==16
    inventory.append(draws)
args.out.mkdir(parents=True,exist_ok=False)
result = {'schema_version':1,'scope':'TRAIN-only zero-update paired gradient diagnostic at e59. No validation-label inference, optimizer step, weight fitting, selection or physical trial.',
    'decision':'Compare the existing weights with two target definitions on identical four production-sampler batches. This report cannot change weights or promote a checkpoint.',
    'started_utc':datetime.now(timezone.utc).isoformat(),
    'source_files_sha256':run['source']['files_sha256'],
    'input_sha256':{str(path):sha256(path) for path in (run_path,args.training/'source.zip',Path(config['init']),Path(__file__))},
    'manifest_sha256':run['manifest_sha256'],'dataset_verification':verification,
    'episodes':[{'path':str(ep.path),'meta_sha256':sha256(ep.path/'meta.json'),'arrays_sha256':ep.meta['arrays_sha256']} for ep in episodes],
    'sampler':sampler_protocol,'sampling_counts':sampler.draw_counts(),
    'sampler_seed':0,'batches':[],
    'episode_index_note':'Training-only manifest order renumbers episode indices. Exact case/frame inventory and flattened training dataset indices match the original sampler protocol.',
    'weights':{'plan':1.,'steer':3.281158240898283,'longitudinal':1.},
    'target_definitions':{'old':'Recorded teacher commands, teacher_command_v1',
                          'new':'Detached op_tangent_actions_from_op_plan(native future, measured body vx/vy, same wheelbase and limits), before mission braking. Independent semantic reference for native_future_v1.'},
    'execution':{'device':str(device),'precision':'float32','tf32':False,'cache':False,'history_context':8,
                 'batch_size':128,'torch_threads':4,'interop_threads':2,'deadline_seconds':600},
}
(args.out/'inventory.json').write_text(json.dumps(inventory,indent=2,allow_nan=False)+'\n')
result['inventory_sha256']=sha256(args.out/'inventory.json')
torch.manual_seed(0)
model = load_fs(config['init']).to(device).eval()
if config['channels_last']:
    model.to(memory_format=torch.channels_last)
configure(model,freeze_vision=True)
result['history_context_proof']=prove_policy_context(model)
assert result['history_context_proof']==run['history_context_proof']
before=all_state_hashes(model)
parameters=[(name,value) for name,value in model.named_parameters() if value.requires_grad]
assert [name for name,_ in parameters]==run['trainable_tensors']
assert sum(value.numel() for _,value in parameters)==run['trainable_parameters']==6949412
result['parameters']=[{'name':name,'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in parameters]
result['trainable_parameters']=sum(value.numel() for _,value in parameters)
print('Verified TRAIN inventory and four fixed 16/112 batches; no optimizer is created',flush=True)
arrays={}
for batch_index,indices in enumerate(batches):
    total,parts,out=batch_loss(model,dataset,indices,device,config['w_steer'],config['w_long'],config['history_chunk_size'],
        amp=False,channels_last=config['channels_last'],history_context=8,w_plan=1.,return_outputs=True,controller_mode='op_tangent_v1')
    observed=out['coherence_observed']
    vehicle=torch.tensor([[episodes[dataset.index[int(index)][0]].wheelbase,
                           episodes[dataset.index[int(index)][0]].max_brake,
                           episodes[dataset.index[int(index)][0]].max_accel] for index in indices],dtype=torch.float32,device=device)
    with torch.no_grad():
        native_steer,native_long=op_tangent_actions_from_op_plan(out['coherence_targets'],observed[:,0],observed[:,1],vehicle[:,0],vehicle[:,1],vehicle[:,2])
    assert not native_steer.requires_grad and not native_long.requires_grad
    predicted_steer,predicted_long=op_tangent_actions_from_op_plan(out['plan_mu'],observed[:,0],observed[:,1],vehicle[:,0],vehicle[:,1],vehicle[:,2])
    new_steer=F.smooth_l1_loss(predicted_steer,native_steer,beta=.05)
    new_long=F.smooth_l1_loss(predicted_long,native_long,beta=.05)
    losses={'plan':parts['plan_nll'],'old_steer':config['w_steer']*parts['steer'],
            'old_longitudinal':parts['longitudinal'],'new_steer':config['w_steer']*new_steer,'new_longitudinal':new_long}
    old_total=losses['plan']+losses['old_steer']+losses['old_longitudinal']
    new_total=losses['plan']+losses['new_steer']+losses['new_longitudinal']
    torch.testing.assert_close(old_total,total,rtol=1e-6,atol=1e-6)
    vectors={}
    unused={}
    for key,loss in losses.items():
        gradients=torch.autograd.grad(loss,[value for _,value in parameters],retain_graph=True,allow_unused=True)
        unused[key]=[name for (name,_),gradient in zip(parameters,gradients,strict=True) if gradient is None]
        vectors[key]=torch.cat([(torch.zeros_like(value) if gradient is None else gradient).detach().reshape(-1)
                                for (_,value),gradient in zip(parameters,gradients,strict=True)])
        assert torch.isfinite(vectors[key]).all()
    vectors['old_action']=vectors['old_steer']+vectors['old_longitudinal']
    vectors['new_action']=vectors['new_steer']+vectors['new_longitudinal']
    vectors['old_total']=vectors['plan']+vectors['old_action']
    vectors['new_total']=vectors['plan']+vectors['new_action']
    pairs=[('plan',name) for name in ('old_steer','new_steer','old_longitudinal','new_longitudinal','old_action','new_action','old_total','new_total')]
    pairs.extend((('old_steer','new_steer'),('old_longitudinal','new_longitudinal'),('old_action','new_action'),('old_total','new_total')))
    direction=torch.zeros_like(out['plan_mu'])
    for knot in (5,6):
        weight=float((T_IDXS[knot]-T_IDXS[4])/(T_IDXS[7]-T_IDXS[4]))
        direction[:,knot,3:5]=(1-weight)*out['plan_mu'][:,4,3:5].detach()+weight*out['plan_mu'][:,7,3:5].detach()-out['plan_mu'][:,knot,3:5].detach()
    directional={}
    for key,loss in (('old_total',old_total),('new_total',new_total)):
        gradient=torch.autograd.grad(loss,out['plan_mu'],retain_graph=True)[0]
        per_draw=(gradient*direction).sum((1,2)).detach().cpu().numpy()
        arrays[f'batch{batch_index}_{key}_output_directional_derivative']=per_draw
        directional[key]={'batch_derivative':float(per_draw.sum()),'draws_negative':int((per_draw<0).sum()),
                          'draws_positive':int((per_draw>0).sum())}
    native_actions=torch.stack((native_steer,native_long),dim=-1).detach().cpu().numpy()
    teacher=np.array([row['teacher_action'] for row in inventory[batch_index]],dtype=np.float32)
    predicted=torch.stack((predicted_steer,predicted_long),dim=-1).detach().cpu().numpy()
    prefix=f'batch{batch_index}_'
    arrays.update({prefix+'plan_mu':out['plan_mu'].detach().cpu().numpy(),prefix+'plan_logstd':out['plan_logstd'].detach().cpu().numpy(),
        prefix+'native_plan':out['coherence_targets'].detach().cpu().numpy(),prefix+'observed_state':observed.detach().cpu().numpy(),
        prefix+'native_controller_actions':native_actions,prefix+'teacher_actions':teacher,prefix+'predicted_actions':predicted})
    result['batches'].append({'batch':batch_index,'draws':128,'recovery_draws':16,'unique_anchors':len(set(indices.tolist())),
        'history_lt8_draws':sum(row['history_frames']<8 for row in inventory[batch_index]),
        'unweighted_losses':{**{key:float(value.detach()) for key,value in parts.items()},'new_steer':float(new_steer.detach()),'new_longitudinal':float(new_long.detach())},
        'weighted_losses':{key:float(value.detach()) for key,value in losses.items()},
        'total_losses':{'old':float(old_total.detach()),'new':float(new_total.detach())},
        'weighted_gradient_l2':{key:float(value.norm()) for key,value in vectors.items()},
        'weighted_gradient_cosine':{left+'__'+right:cosine(vectors[left],vectors[right]) for left,right in pairs},
        'unused_parameter_names':unused,
        'teacher_minus_native_target':{'mean_signed_steer_longitudinal':(teacher-native_actions).mean(0).tolist(),
                                       'mae_steer_longitudinal':abs(teacher-native_actions).mean(0).tolist()},
        'output_directional_derivative':directional})
    assert all(value.grad is None for _,value in parameters)
    print(json.dumps({'batch':batch_index,'total_losses':result['batches'][-1]['total_losses'],
                     'plan_cos_old':result['batches'][-1]['weighted_gradient_cosine']['plan__old_total'],
                     'plan_cos_new':result['batches'][-1]['weighted_gradient_cosine']['plan__new_total']}),flush=True)
    del vectors,losses,total,parts,out,old_total,new_total,new_steer,new_long,predicted_steer,predicted_long,gradients,gradient
after=all_state_hashes(model)
assert after==before, 'Model state changed during zero-update diagnostic'
for name,digest in run['source']['files_sha256'].items():
    assert sha256(source_root/name)==digest
assert sha256(config['init'])==run['init_sha256']
result['model_unchanged']=True
result['model_state_sha256']=before
result['ended_utc']=datetime.now(timezone.utc).isoformat()
result['elapsed_seconds']=time.monotonic()-start
result['limits']=[
    'Four sampler batches are 512 draws with replacement, not 512 independent anchors or physical trials.',
    'No gradient clipping, Adam preconditioning or optimizer step is applied. Raw gradient agreement is not a guarantee of later learning.',
    'Output directional derivatives hold logstd and all other output coordinates fixed. Parameter gradients may move different coordinates together.',
    'The output direction interpolates only own velocity knots5/6 between own4/7. It is a diagnostic direction, not a runtime correction.',
    'No consistency penalty is added or used for choosing a coefficient. All existing weights stay fixed.',
    'This frozen-source script is an independent mathematical reference for new targets. The proposed implementation requires its own parity tests.',
    'Validation episode metadata are read to verify the manifest. No validation episode arrays are loaded and no validation labels enter this diagnostic.',
]
np.savez_compressed(args.out/'arrays.npz',**arrays)
result['arrays_sha256']=sha256(args.out/'arrays.npz')
(args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'out':str(args.out),'elapsed_seconds':result['elapsed_seconds'],'model_unchanged':True}),flush=True)
