"""Apply the documented, outcome-independent gate-order correction.

Some benchmark files are ordered by class. A predictable betting strategy
must not receive that label-related ordering. This correction was specified
before inspection of new-suite gate outcomes or assessment results. Models,
scores, budgets, and role membership remain frozen. Preserve original gates.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from study import NEW,SEEDS,ordered_gate,betting,dump
from seb import logloss

ROOT=Path(__file__).resolve().parents[1]
def main(datasets=None):
    datasets=NEW if datasets is None else datasets
    for name in datasets:
        for b in ['MiniRocket','Hydra']:
            for seed in SEEDS:
                path=ROOT/'results'/name/f'{b}_{seed}'
                if not (path/'records.npz').exists():raise RuntimeError('Study incomplete: '+str(path))
    for name in datasets:
        for b in ['MiniRocket','Hydra']:
            for seed in SEEDS:
                path=ROOT/'results'/name/f'{b}_{seed}'
                frozen=json.loads((path/'frozen.json').read_text())
                if frozen.get('gate_order')=='hash-v1':continue
                data=dict(np.load(path/'records.npz'));names=data['score_names'].tolist();s=data['scores'];c=data['correct'];gi=ordered_gate(name,data['roles'])
                ir=names.index('reference');iq=names.index('candidate');ip=names.index('policy')
                d=logloss(c[gi],s[ir,gi])-logloss(c[gi],s[iq,gi]);g=betting(d,frozen['bound'])
                (path/'gate_before_order_correction.json').write_bytes((path/'gate.json').read_bytes())
                s[ip]=s[iq] if g['authorized'] else s[ir]
                data['scores']=s;data['gate_order_ids']=gi
                np.savez_compressed(path/'records.npz',**data)
                frozen['gate_order']='hash-v1';frozen['gate_order_correction_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                dump(path/'frozen.json',frozen);dump(path/'gate.json',g)
    print('All gates replayed using label-independent hash ordering.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=NEW);args=parser.parse_args();main(args.datasets)
