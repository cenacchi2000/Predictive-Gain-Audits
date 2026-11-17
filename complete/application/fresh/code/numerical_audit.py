"""Independent dot-product checks for local BLAS warning diagnostics."""
from pathlib import Path
import json,joblib
import numpy as np
from study import load,logits_matrix,NEW,SEEDS,dump
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[]
    for name in NEW:
        x,y,xt,yt,meta=load(name)
        for seed in SEEDS:
            for kind in ['MiniRocket','Hydra']:
                path=ROOT/'results'/name/f'backbones_{seed}'/(kind+'.joblib')
                if not path.exists():continue
                m=joblib.load(path);xx=xt[:256]
                if xx.shape[-1]<9:xx=np.pad(xx,((0,0),(0,0),(0,9-xx.shape[-1])),mode='reflect')
                f=m['scaler'].transform(m['transform'].transform(xx))
                if hasattr(f,'numpy'):f=f.numpy()
                f=np.asarray(f,dtype=np.float32)[:16];model=m['model']
                independent=np.einsum('ij,kj->ik',f.astype(float),np.atleast_2d(model.coef_).astype(float),optimize=False)+model.intercept_
                if independent.shape[1]==1:independent=independent[:,0]
                predicted=model.decision_function(f)
                delta=float(np.max(np.abs(predicted-independent)))
                assert np.isfinite(f).all() and np.isfinite(predicted).all()
                assert np.allclose(predicted,independent,atol=2e-4,rtol=2e-4)
                rows.append({'dataset':name,'seed':seed,'backbone':kind,'max_dot_discrepancy':delta,'finite':True})
                print(name,seed,kind,delta,flush=True)
    dump(ROOT/'results'/'numerical_audit.json',rows)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
