"""Targeted stress on assessment rows; clean fitted objects and gates stay fixed.

These interventions retain archive labels for diagnostic scoring. They are
not assumed semantically label-preserving and carry no clean-gate guarantee.
"""
from pathlib import Path
import json,sys,joblib,argparse
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from study import (NEW,load,roles,normalized,logits_matrix,output_features,
                   time_features,probability,bounded_update,dump)
from seb import bundle,metrics,logloss

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def corrupt(x,kind):
    if kind=='time_block':
        x=x.copy();start=x.shape[-1]//3;x[:,:,start:start+max(1,x.shape[-1]//5)]=0
        return normalized(x)
    f=np.fft.rfft(x,axis=-1);m=f.shape[-1]-1;k=max(1,int(.2*m))
    if kind=='dominant_band':
        ix=np.argsort((abs(f[:,:,1:])**2).sum(1),axis=1)[:,-k:]+1
        for i,j in enumerate(ix):f[i,:,j]=0
    else:
        rng=np.random.default_rng(2027);start=rng.integers(1,m-k+2,size=len(x))
        for i,s in enumerate(start):f[i,:,s:s+k]=0
    return normalized(np.fft.irfft(f,n=x.shape[-1],axis=-1))

def backbone_predict(m,x):
    xx=x if x.shape[-1]>=9 else np.pad(x,((0,0),(0,0),(0,9-x.shape[-1])),mode='reflect')
    z=[];rep=[];distance=[]
    for start in range(0,len(xx),256):
        f=m['scaler'].transform(m['transform'].transform(xx[start:start+256]))
        if hasattr(f,'numpy'):f=f.numpy()
        f=np.asarray(f,dtype=np.float32)
        z.append(logits_matrix(m['model'].decision_function(f)))
        rep.append(f[:,m['indices']]);distance.append(np.log1p(m['nn'].kneighbors(rep[-1])[0]))
    return {'logits':np.concatenate(z),'representation':np.concatenate(rep),'distance':np.concatenate(distance)}

def main(datasets=None):
    results=[]
    for name in NEW if datasets is None else datasets:
        x,y,xt,yt,meta=load(name);rr=roles(name,len(xt),meta['eligible_audit_ids']);ix=np.flatnonzero(rr=='assessment')
        base={b:joblib.load(OUT/name/'backbones_17'/(b+'.joblib')) for b in ['MiniRocket','Hydra']}
        for kind in ['dominant_band','random_band','time_block']:
            target=OUT/name/('stress_'+kind+'.npz')
            if target.exists():
                a=np.load(target);rows=[]
                for b in ['MiniRocket','Hydra']:
                    for method in ['reference','candidate','policy']:
                        rows.append({'dataset':name,'backbone':b,'seed':17,'stress':kind,'method':method,**metrics(a[b+'_correct'],a[b+'_'+method])})
                dump(target.with_suffix('.json'),rows);results.extend(rows);continue
            cx=corrupt(xt[ix],kind);g=bundle(cx);tf=time_features(cx)
            rec={b:backbone_predict(m,cx) for b,m in base.items()};arrays={'ids':ix,'labels':yt[ix]};rows=[]
            for b in base:
                folder=OUT/name/f'{b}_17';frozen=json.loads((folder/'frozen.json').read_text());gate=json.loads((folder/'gate.json').read_text())
                if frozen.get('gate_order')!='hash-v1':raise RuntimeError('Clean gate not finalized')
                models=joblib.load(folder/'reliability.joblib');z=rec[b]['logits'];h=output_features(z)
                other=rec['Hydra' if b=='MiniRocket' else 'MiniRocket']['logits'];p=softmax(z,1);po=softmax(other,1)
                ag=np.c_[(z.argmax(1)==other.argmax(1)),po.max(1),(p*po).sum(1)]
                feats={'output':h,'time':np.c_[h,tf],'representation':np.c_[h,rec[b]['representation']],
                       'agreement':np.c_[h,ag],'atypicality':np.c_[h,rec[b]['distance']],
                       'all_controls':np.c_[h,tf,rec[b]['representation'],ag,rec[b]['distance']]}
                refname=frozen['reference'].split('/')[0]
                if refname=='raw':r=p.max(1)
                elif refname in ['constant','intercept']:r=np.full(len(cx),models['reference'])
                else:r=probability(models['reference'],feats[refname])
                candname=frozen['candidate'].split('/')[0]
                if candname=='offset_spectral':s=models['candidate'].predict(r,g)
                elif candname=='constant':s=np.full(len(cx),models['candidate'])
                else:s=probability(models['candidate'],np.c_[h if candname=='spectral' else feats['all_controls'],g])
                q=bounded_update(r,s,frozen['bound']);policy=q if gate['authorized'] else r;c=(z.argmax(1)==yt[ix]).astype(int)
                for method,score in [('reference',r),('candidate',q),('policy',policy)]:
                    row={'dataset':name,'backbone':b,'seed':17,'stress':kind,'method':method,**metrics(c,score)};rows.append(row)
                    arrays[b+'_'+method]=score
                arrays[b+'_correct']=c;arrays[b+'_predicted']=z.argmax(1)
            np.savez_compressed(target,**arrays);dump(target.with_suffix('.json'),rows);results.extend(rows)
            print(name,kind,'complete',flush=True)
    dump(OUT/'stress_metrics.json',results)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=NEW);args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.datasets)
