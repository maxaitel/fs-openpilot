import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/vision_corridor_probe_v1'
OUT = Path(__file__).resolve().parent
BANDS = ((0, 5), (5, 10), (10, 20))
NAMES = ('blue', 'yellow', 'orange', 'big_orange')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
    copy_path = SOURCE / 'copy_receipt.json'
    assert digest(copy_path) == '4b20da5aa35dcfc2b65eb06f643f9d451271255b257c5642be77dc54bada1c8b'
    copy = json.loads(copy_path.read_text())
    assert copy['all_passed']
    hashes = {str(copy_path.relative_to(ROOT)): digest(copy_path)}
    for item in copy['files']:
        p = ROOT / item['local']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], p
        hashes[item['local']] = item['sha256']
    receipt = json.loads((SOURCE / 'receipt.json').read_text())
    assert receipt['status'] == 'passed' and receipt['input_sha256_before'] == receipt['input_sha256_after']
    for n, h in receipt['output_sha256'].items():
        assert digest(SOURCE / n) == h
    for ep in receipt['episodes']:
        d = SOURCE / 'references_data' / ep
        meta = json.loads((d / 'meta.json').read_text())
        assert meta['id'] == ep and meta['split'] == 'train'
        for n in ('states.npy', 'cones.npy', 'ticks.npy', 'controls.npy', 'desires.npy', 'contacts.npy'):
            h = digest(d / n)
            assert h == meta['arrays_sha256'][n]
            assert h == receipt['input_sha256_before'][f'/work/data/native/lab_v7/{ep}/{n}']
        assert digest(d / 'meta.json') == receipt['input_sha256_before'][f'/work/data/native/lab_v7/{ep}/meta.json']
    assert digest(OUT / 'protocol.md') == 'af391fe9e13653e29600e1c2e88d425cda825ebb254848fcdd9711f097b4d1f3'
    hashes['protocol.md'] = digest(OUT / 'protocol.md')
    hashes['score.py'] = digest(Path(__file__))
    return hashes

def run():
    before = verify()
    receipt = json.loads((SOURCE / 'receipt.json').read_text())
    refs = json.loads((SOURCE / 'references.json').read_text())
    assert [r['id'] for r in refs] == receipt['episodes']
    episodes = {}
    for ref in refs:
        ep = ref['id']; d = SOURCE / 'references_data' / ep
        episodes[ep] = dict(meta=json.loads((d/'meta.json').read_text()), states=np.load(d/'states.npy'), cones=np.load(d/'cones.npy'), ticks=np.load(d/'ticks.npy'))
    records = defaultdict(list)
    for line in (SOURCE/'frames.jsonl').read_text().splitlines():
        row = json.loads(line); records[row['episode']].append(row)
    assert set(records) == set(episodes)
    per_frame = []
    accum = defaultdict(lambda: {'pred': 0, 'correct': 0, 'wrong_class': 0, 'errors': []})
    for ep, data in episodes.items():
        rows = records[ep]; meta = data['meta']; ticks = data['ticks']; states = data['states']; cones = data['cones']
        assert len(rows) == receipt['frame_counts'][ep] == len(ticks)
        assert np.array_equal(ticks, np.arange(len(ticks))*4)
        for i, r in enumerate(rows):
            tick = int(ticks[i]); assert r['frame'] == i and r['tick'] == tick and abs(r['time_s'] - states[tick, 8]) < 1e-6
            state = states[tick]; delta = cones[:,:2] - state[:2]; c, s = np.cos(state[2]), np.sin(state[2]); gt = np.column_stack((c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1])); gc = cones[:,2].astype(int)
            p = np.asarray(r['cones_xy'], dtype=float).reshape(-1,2); pc = np.asarray(r['cones_cls'], dtype=int); ps = np.asarray(r['cones_score'], dtype=float)
            assert len(p) == len(pc) == len(ps)
            valid = np.isfinite(p).all(axis=1) & np.isfinite(ps) & (pc >= 0) & (pc < 4)
            p, pc, ps = p[valid], pc[valid], ps[valid]
            pd = np.linalg.norm(p,axis=1)
            stage = 'terminal' if tick >= meta['terminal_continuation']['start_tick'] else 'driving'
            phase = 'trackdrive'
            if meta['discipline'] == 'skidpad':
                crossings = meta['summary']['route']['crossings']; phase = ('entry','right_1','right_2','left_1','left_2','exit')[int(np.searchsorted(crossings, r['time_s'], side='right'))]
            fr = {'episode':ep,'frame':i,'tick':tick,'time_s':r['time_s'],'stage':stage,'phase':phase,'ranges':{}}
            for low,high in BANDS:
                key=f'{low}-{high}'; selected=(pd>=low)&(pd<high)
                counts = {NAMES[j]:int((selected & (pc==j)).sum()) for j in range(4)}
                counts['confidence'] = {'lt_0.5':int((selected & (ps<.5)).sum()),'0.5_to_0.75':int((selected & (ps>=.5)&(ps<.75)).sum()),'gte_0.75':int((selected & (ps>=.75)).sum())}
                counts['forward_blue'] = int((selected & (p[:,0]>0) & (pc==0)).sum()); counts['forward_yellow'] = int((selected & (p[:,0]>0) & (pc==1)).sum())
                counts['both_forward_colors'] = bool(counts['forward_blue'] and counts['forward_yellow']); fr['ranges'][key]=counts
            per_frame.append(fr)
            for group, cls in (('color',(0,1)),('orange',(2,3))):
                pp=np.flatnonzero(np.isin(pc,cls)); gg=np.flatnonzero(np.isin(gc,cls)); pi=np.array([],int); gi=np.array([],int); dist=np.empty((len(pp),len(gg)))
                if len(pp) and len(gg):
                    dist=np.linalg.norm(p[pp,None,:]-gt[None,gg,:],axis=2); a,b=linear_sum_assignment(np.where(dist<.75,dist,1e6)); ok=dist[a,b]<.75; pi,gi=a[ok],b[ok]
                for low,high in BANDS:
                    for cl in cls:
                        mask=(pc[pp[pi]]==cl)&(pd[pp[pi]]>=low)&(pd[pp[pi]]<high); same=pc[pp[pi]]==gc[gg[gi]]
                        for ph in (phase, 'all_phases'):
                            key=(ep,stage,ph,group,NAMES[cl],f'{low}-{high}'); ac=accum[key]; ac['pred']+=int(((pc==cl)&(pd>=low)&(pd<high)).sum())
                            ac['correct']+=int((mask&same).sum()); ac['wrong_class']+=int((mask&~same).sum()); ac['errors'].extend(dist[pi[mask&same],gi[mask&same]].tolist())
    assert verify() == before
    associations=[]
    for key,a in accum.items():
        errors=a.pop('errors'); associations.append(dict(zip(('episode','stage','phase','group','class','range_m'),key),**a, precision_to_any_gt=a['correct']/a['pred'] if a['pred'] else None, error_p50_m=float(np.median(errors)) if errors else None,error_p95_m=float(np.quantile(errors,.95)) if errors else None))
    support=[]
    for ep in episodes:
        for stage in ('driving','terminal'):
            all_rows=[r for r in per_frame if r['episode']==ep and r['stage']==stage]
            phases=['all_phases'] + sorted({r['phase'] for r in all_rows})
            for phase in phases:
                rr=[r for r in all_rows if phase=='all_phases' or r['phase']==phase]
                for low,high in BANDS:
                    key=f'{low}-{high}'; gaps=[not r['ranges'][key]['both_forward_colors'] for r in rr]; longest=cur=0
                    for g in gaps: cur=cur+1 if g else 0; longest=max(longest,cur)
                    fields=('blue','yellow','orange','big_orange','forward_blue','forward_yellow')
                    distributions={f: {'min':min(r['ranges'][key][f] for r in rr), 'median':float(np.median([r['ranges'][key][f] for r in rr])), 'max':max(r['ranges'][key][f] for r in rr)} for f in fields}
                    confidence={f:sum(r['ranges'][key]['confidence'][f] for r in rr) for f in ('lt_0.5','0.5_to_0.75','gte_0.75')}
                    support.append({'episode':ep,'stage':stage,'phase':phase,'range_m':key,'frames':len(rr),'missing_both_colors_frames':sum(gaps),'longest_missing_streak_frames':longest,'longest_streak_s_sample_span':max(0,longest-1)*.2,'counts_per_frame':distributions,'confidence_counts':confidence})
    (OUT/'per_frame.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in per_frame))
    (OUT/'results.json').write_text(json.dumps({'protocol_sha256':before['protocol.md'],'input_hashes_before_after':before,'association':associations,'support':support},indent=2)+'\n')
    print('frames',len(per_frame),'association rows',len(associations),'support rows',len(support))

if __name__ == '__main__': run()
