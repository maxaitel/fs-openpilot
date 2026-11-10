import argparse,json,shlex,subprocess
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('label');a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text());assert c.pop('backend')=='isaac_lab'
args=[]
for k,v in c.items():
 if v is None or v is False:continue
 args.append('--'+k.replace('_','-'))
 if v is not True:args.append(str(v))
command=['/isaac-sim/python.sh','scripts/measure_native.py','--out',f'runs/{a.label}.measure.json','--','/isaac-sim/python.sh','-m','fsop.train.lab_train',*args]
root='/scratch/fsop/head_trial_work'
remote=['docker','exec','-w',root,'-e',f'PYTHONPATH={root}:{root}/src:{root}/isaac_eval','fsop-pi-scratch','sh','-c',shlex.join(command)+' > '+shlex.quote(f'.work/lab/head_training/{a.label}.log')+' 2>&1']
Path(f'.work/lab/head_training/{a.label}.command.json').write_text(json.dumps({'command':command,'configuration':a.config,'reason':'The image lacks the optional Typer CLI package; invoke the same lab_train.main parser directly.'},indent=2)+'\n')
result=subprocess.run(['ssh','-o','HostName=192.168.50.183','-o','HostKeyAlias=100.116.25.114','-o','StrictHostKeyChecking=yes','max-spark',shlex.join(remote)])
raise SystemExit(result.returncode)
