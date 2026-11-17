"""Release assessment results only after all declared configurations are frozen."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from seb import metrics,logloss
from study import NEW,SEEDS,betting,ordered_gate

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def main():
    paths=[OUT/d/f'{b}_{s}' for d in NEW for b in ['MiniRocket','Hydra'] for s in SEEDS]
    missing=[str(p) for p in paths if not (p/'records.npz').exists()]
    if missing:raise RuntimeError('Assessment release blocked: incomplete declared configurations: '+str(missing))
    rows=[];comparisons=[];checks=[]
    for folder in paths:
        frozen=json.loads((folder/'frozen.json').read_text());gate=json.loads((folder/'gate.json').read_text());a=np.load(folder/'records.npz')
        names=a['score_names'].tolist();scores=dict(zip(names,a['scores']));r=a['roles'];c=a['correct'];ix=r=='assessment';gi=ordered_gate(frozen['dataset'],r)
        assert frozen.get('gate_order')=='hash-v1'
        gain=logloss(c[gi],scores['reference'][gi])-logloss(c[gi],scores['candidate'][gi]);replay=betting(gain,frozen['bound'])
        assert replay['authorized']==gate['authorized'] and abs(replay['log_e']-gate['log_e'])<1e-10
        assert np.isfinite(a['scores']).all()
        assert np.array_equal(scores['policy'],scores['candidate'] if gate['authorized'] else scores['reference'])
        assert np.max(abs(gain))<=frozen['bound']+1e-10
        for key,value in scores.items():
            m=metrics(c[ix],value[ix]);row={'dataset':frozen['dataset'],'backbone':frozen['backbone'],'seed':frozen['seed'],'method':key,**m}
            rows.append(row)
        ref=metrics(c[ix],scores['reference'][ix]);cand=metrics(c[ix],scores['candidate'][ix]);policy=metrics(c[ix],scores['policy'][ix])
        d=logloss(c[ix],scores['reference'][ix])-logloss(c[ix],scores['candidate'][ix])
        comparisons.append({**frozen,'authorized':gate['authorized'],'gate_log_e':gate['log_e'],
                            'assessment_nll_gain':float(d.mean()),'assessment_auc_gain':cand['auc']-ref['auc'],
                            'assessment_policy_nll_gain':ref['nll']-policy['nll'],
                            'assessment_policy_auc_gain':policy['auc']-ref['auc'],
                            'assessment_harm_fraction':float(np.mean(d < -1e-12)),
                            'assessment_gain_fraction':float(np.mean(d > 1e-12)),
                            'assessment_error_upcross_09':float(np.mean((scores['reference'][ix][c[ix]==0]<.9)&(scores['candidate'][ix][c[ix]==0]>=.9))) if (c[ix]==0).any() else None,
                            'assessment_error_downcross_09':float(np.mean((scores['reference'][ix][c[ix]==0]>=.9)&(scores['candidate'][ix][c[ix]==0]<.9))) if (c[ix]==0).any() else None,
                            'assessment_gain_quantiles':np.quantile(d,[0,.05,.25,.5,.75,.95,1]).tolist(),
                            'assessment_gain_on_errors':float(d[c[ix]==0].mean()) if (c[ix]==0).any() else None,
                            'assessment_gain_on_correct':float(d[c[ix]==1].mean()) if (c[ix]==1).any() else None,
                            'assessment_accuracy':float(c[ix].mean())})
        checks.append({'configuration':folder.name,'dataset':frozen['dataset'],'records_sha256':hashlib.sha256((folder/'records.npz').read_bytes()).hexdigest(),'gate_replay':True,'fallback_exact':True,'all_finite':True})
    def clean(obj):
        if isinstance(obj,np.generic):return clean(obj.item())
        if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
        if isinstance(obj,list):return [clean(v) for v in obj]
        if isinstance(obj,float) and not np.isfinite(obj):return None
        return obj
    for name,data in [('assessment_metrics',rows),('assessment_comparisons',comparisons),('replay_checks',checks)]:
        (OUT/(name+'.json')).write_text(json.dumps(clean(data),indent=2,allow_nan=False))
    rng=np.random.default_rng(2027);summary={'configurations':len(paths),'authorizations':sum(x['authorized'] for x in comparisons),'datasets':len(NEW)}
    for metric in ['assessment_nll_gain','assessment_auc_gain','assessment_policy_nll_gain','assessment_policy_auc_gain','assessment_harm_fraction']:
        bydataset=[]
        for d in NEW:
            values=[a[metric] for a in comparisons if a['dataset']==d and np.isfinite(a[metric])]
            bydataset.append(float(np.mean(values)) if values else float('nan'))
        bydataset=np.array(bydataset);valid=bydataset[np.isfinite(bydataset)]
        boot=valid[rng.integers(0,len(valid),size=(10000,len(valid)))].mean(1) if len(valid) else np.array([float('nan')])
        summary[metric]={'macro_mean':float(valid.mean()) if len(valid) else None,'dataset_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'by_dataset':dict(zip(NEW,bydataset.tolist())),'valid_datasets':len(valid),'valid_configurations':sum(np.isfinite(a[metric]) for a in comparisons)}
    (OUT/'assessment_summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False))
    print(json.dumps(clean(summary),indent=2))

if __name__=='__main__':main()
