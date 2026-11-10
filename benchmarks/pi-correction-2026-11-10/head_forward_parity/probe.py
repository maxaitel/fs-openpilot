import argparse, importlib.util, sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
from fsop.data.native_replay import NativeEpisode, NativeReplay, episode_paths
from fsop.model.build import load_fs
from fsop.train.lab_train import configure, batch_loss, sha256
from isaac_lab_fsop.evaluate import model_digest, source_inventory
p=argparse.ArgumentParser();p.add_argument('--order',choices=['data_first','model_first'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(exist_ok=False)
control=Path('/scratch/fsop/pi_update_probe2'); hp=json.loads(Path('/scratch/fsop/pi_head_scope_probe1/protocol.json').read_text()); cp=json.loads((control/'protocol.json').read_text())
assert source_inventory()==hp['source_sha256']==cp['source_sha256']
for name,h in hp['graph_sha256'].items(): assert sha256(name)==h
path=cp['models']['latest']['path'];assert sha256(path)==cp['models']['latest']['sha256']
def data():
 return NativeReplay([NativeEpisode(q) for q in episode_paths(manifest='/work/data/native/lab_v7/manifest.json') if json.loads((q/'meta.json').read_text())['split']=='train'],'train')
def model():
 torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.manual_seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=load_fs(path).to('cuda').eval();configure(m,freeze_vision=True);return m
if a.order=='data_first': ds=data();m=model()
else: m=model();ds=data()
flags={n:q.requires_grad for n,q in m.named_parameters()};initial=model_digest(m)
assert initial==json.loads((control/'report.json').read_text())['models']['latest']['initial_model_digest']
ids=[r['index']for r in cp['monitor']];arrays={};rows=[]
old=np.load(control/'arrays.npz')
for label in ('full1','head1','full2','head2'):
 if label.startswith('full'):
  for n,q in m.named_parameters():q.requires_grad_(flags[n])
 else:m.requires_grad_(False);m.base.policy.p_node_linear_20.requires_grad_(True)
 with torch.no_grad():
  loss,parts,out=batch_loss(m,ds,ids,torch.device('cuda'),0.,0.,64,history_context=8,w_plan=1.,w_consistency=1.,return_outputs=True,controller_mode='op_tangent_pi_v1',action_target_mode='desired_motion_v1')
 mu=out['plan_mu'].cpu().numpy();arrays[label+'_mu']=mu;arrays[label+'_logstd']=out['plan_logstd'].cpu().numpy();arrays[label+'_hidden']=out['hidden'].cpu().numpy()
 expected=old['latest_monitor_before_mu'];err=np.abs(mu-expected);tol=1e-7+1e-6*np.abs(expected)
 rows.append({'label':label,'control_max_abs':float(err.max()),'control_strict_mismatches':int((err>tol).sum()),'within_existing_inference_tolerance':bool(np.allclose(mu,expected,atol=1e-5,rtol=1e-5)),'losses':{k:float(v)for k,v in parts.items()}})
 assert model_digest(m)==initial
pairs={}
for left,right in [('full1','full2'),('head1','head2'),('full1','head1'),('full2','head2')]:
 pairs[left+'_'+right]={k:float(np.max(np.abs(arrays[left+'_'+k]-arrays[right+'_'+k])))for k in ('mu','logstd','hidden')}
np.savez_compressed(a.out/'arrays.npz',**arrays)
r={'scope':'Zero optimizer updates. Repeated same-model forwards with alternating requires_grad scopes. Data/model load order is the only process-level alternative. Original failed head experiment is not reclassified.','order':a.order,'model_digest':initial,'weights_unchanged':True,'rows':rows,'pairs':pairs,'script_sha256':sha256(__file__),'head_protocol_sha256':sha256('/scratch/fsop/pi_head_scope_probe1/protocol.json'),'control_protocol_sha256':sha256(control/'protocol.json'),'arrays_sha256':sha256(a.out/'arrays.npz')}
(a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
