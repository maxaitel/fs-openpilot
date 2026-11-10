import argparse, hashlib, inspect, json, shlex, subprocess, sys
from pathlib import Path
from scripts.archive_lab_batch import inventory

p=argparse.ArgumentParser();p.add_argument('--remote',required=True);p.add_argument('--local',type=Path,required=True);a=p.parse_args()
assert a.remote.startswith('/scratch/fsop/head_trial_work/runs/') and '..' not in a.remote
assert not a.local.exists()
ssh=['ssh','-o','HostName=192.168.50.183','-o','HostKeyAlias=100.116.25.114','-o','StrictHostKeyChecking=yes','max-spark']
code='import hashlib,json,os,stat\nfrom pathlib import Path\n'+inspect.getsource(inventory)+'\nprint(json.dumps(inventory('+repr(a.remote)+'),sort_keys=True))'
remote=shlex.join(['docker','exec','fsop-pi-scratch','/isaac-sim/python.sh','-c',code])
before=subprocess.check_output([*ssh,remote])
a.local.mkdir(parents=True)
a.local.with_suffix('.remote_inventory.json').write_bytes(before)
command=shlex.join(['docker','exec','fsop-pi-scratch','tar','-C',a.remote,'-cf','-','.'])
with subprocess.Popen([*ssh,command],stdout=subprocess.PIPE) as producer:
 try:
  subprocess.run(['tar','-C',str(a.local),'-xf','-'],stdin=producer.stdout,check=True)
 finally:
  producer.stdout.close()
 status=producer.wait()
assert status==0,status
after=subprocess.check_output([*ssh,remote])
assert before==after
expected=json.loads(before)
actual=inventory(a.local)
assert expected==actual, {'missing':sorted(expected.keys()-actual.keys()),'extra':sorted(actual.keys()-expected.keys()),'changed':[k for k in expected.keys()&actual.keys() if expected[k]!=actual[k]]}
receipt={'remote':a.remote,'local':str(a.local.resolve()),'verified_files':sum(v['type']=='file' for v in expected.values()),'verified_bytes':sum(v.get('size',0) for v in expected.values()),'remote_inventory_sha256':hashlib.sha256(before).hexdigest(),'remote_before_after_identical':True,'local_inventory_exact':True,'source_deleted':False}
a.local.with_suffix('.copy_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
