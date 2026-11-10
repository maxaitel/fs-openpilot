from pathlib import Path
import json,hashlib,copy,yaml,torch
from scripts.select_lab_tangent import protocol_check,TRAINING_ARGS,PI_RANKING
from scripts.select_lab_action import source_check
from fsop.train.action_target import DESIRED_MOTION_CONTRACT
from fsop.eval.controller import controller_contract
from scripts.freeze_lab_selection import checkpoint_info

root=Path('runs/lab_v7/head_calibration2');run=json.loads((root/'run.json').read_text());cal=json.loads((root/'calibration.json').read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert run['calibration_only'] is True and not (root/'history.jsonl').exists()
assert run['init_sha256']=='48db741c8a686b2e635ae7a81b4e4c08d79fbfbd9f4dc4c163260da6f93f0cfb'
assert cal['validation']['count']==256
prior_cal=json.loads(Path('/work/runs/lab_v7/pi_calibration2/calibration.json').read_text())
assert cal['validation']['aggregate']['plan_l1']==prior_cal['validation']['aggregate']['plan_l1']
assert cal['validation']['aggregate']['steer']==prior_cal['validation']['aggregate']['steer']
assert cal['validation']['aggregate']['longitudinal']==prior_cal['validation']['aggregate']['longitudinal']
source_check(root,run)
assert checkpoint_info(root/'step_000000.pt')['state_sha256']==checkpoint_info(run['args']['init'])['state_sha256']
prior=json.loads(Path('docs/isaac-lab-training/native_future_selection1.json').read_text())
fields=(*TRAINING_ARGS,'action_target_mode','w_consistency','optimizer_scope')
args=run['args'];args=dict(args,cache_frozen_features=args.get('cache_frozen_features',False))
assert args['channels_last'] is False and args['cache_frozen_features'] is False and args['amp'] is False
assert args['seed']==0 and args['optimizer_scope']=='final_plan'
assert args['lr']==1e-4 and args['lr_policy']==.03 and args['w_steer']==3.281158240898283
assert args['w_plan']==args['w_consistency']==1 and args['w_long']==0
protocol={'schema_version':4,'kind':'op_tangent_pi_v1_selection','calibration_sha256':sha(root/'calibration.json'),
          'initializer_sha256':run['init_sha256'],'manifest_sha256':run['manifest_sha256'],
          'controller_contract':controller_contract('op_tangent_pi_v1'),'expected_steps':500,
          'checkpoint_steps':[0,100,200,300,400,500], 'training_args':{key:args[key] for key in fields},
          'action_target_contract':DESIRED_MOTION_CONTRACT,'ranking':PI_RANKING,'bounds':copy.deepcopy(prior['bounds'])}
protocol_check(protocol)
assert protocol['bounds']==prior['bounds']
folder=Path('benchmarks/pi-correction-2026-11-10');destination=folder/'head_selection1.json';assert not destination.exists()
destination.write_text(json.dumps(protocol,indent=2)+'\n')
config=yaml.safe_load((folder/'head_calibration2.yaml').read_text());config.update(calibration_only=False,out='runs/lab_v7/head_trial1',expected_calibration=str(root/'calibration.json'),selection_protocol=str(destination),selection_protocol_sha=sha(destination))
target=folder/'head_trial1.yaml';assert not target.exists();target.write_text(yaml.safe_dump(config,sort_keys=False))
receipt={'protocol_sha256':sha(destination),'config_sha256':sha(target),'calibration_sha256':sha(root/'calibration.json'),
         'prior_protocol_sha256':sha('docs/isaac-lab-training/native_future_selection1.json'),
         'all_prior_numeric_bounds_unchanged':True,'builder_sha256':sha(__file__),
         'initializer_sha256':run['init_sha256'],'manifest_sha256':run['manifest_sha256'],
         'source_archive_sha256':sha(root/'source.zip')}
(folder/'head_freeze_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
