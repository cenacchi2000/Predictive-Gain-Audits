"""Unblind only once every declared fresh policy is frozen. No fitting here."""
from pathlib import Path
import hashlib,json
import numpy as np
from study import NEW,OUT,betting,old_betting,ordered_gate,dump
from seb import metrics,logloss

def main():
 paths=[]
 for name in NEW:
  selected=OUT/name/'selected_policy.json'
  if not selected.exists():raise RuntimeError('Release blocked: missing policy '+name)
  choice=json.loads(selected.read_text());path=OUT/name/f"{choice['backbone']}_{choice['seed']}"
  if not (path/'records.npz').exists():raise RuntimeError('Release blocked: missing records '+name)
  paths.append(path)
 rows=[];comp=[];checks=[]
 for path in paths:
  x=json.loads((path/'frozen.json').read_text());g=json.loads((path/'gate.json').read_text());a=np.load(path/'records.npz');sc=dict(zip(a['score_names'].tolist(),a['scores']));c=a['correct'];r=a['roles'];gi=ordered_gate(x['dataset'],r);ai=r=='assessment'
  gain=logloss(c[gi],sc['reference'][gi])-logloss(c[gi],sc['candidate'][gi]);re=betting(gain,x['bound'],x['fraction']);old=old_betting(gain,x['bound'])
  assert abs(re['log_e']-g['log_e'])<1e-10 and re['authorized']==g['authorized']
  assert np.isfinite(a['scores']).all()
  assert np.array_equal(sc['policy'],sc['candidate'] if re['authorized'] else sc['reference'])
  for key,q in sc.items():rows.append({'dataset':x['dataset'],'backbone':x['backbone'],'seed':x['seed'],'method':key,**metrics(c[ai],q[ai])})
  ref=metrics(c[ai],sc['reference'][ai]);candidate=metrics(c[ai],sc['candidate'][ai]);policy=metrics(c[ai],sc['policy'][ai]);d=logloss(c[ai],sc['reference'][ai])-logloss(c[ai],sc['candidate'][ai]);err=c[ai]==0
  comp.append({**x,'authorized':g['authorized'],'old_rule_authorized':old['authorized'],'gate_log_e':g['log_e'],'old_gate_log_e':old['log_e'],'gate_gain':g['mean_gain'],'assessment_nll_gain':float(d.mean()),'assessment_policy_nll_gain':ref['nll']-policy['nll'],'assessment_auc_gain':candidate['auc']-ref['auc'],'assessment_policy_auc_gain':policy['auc']-ref['auc'],'assessment_accuracy':float(c[ai].mean()),'assessment_harm_fraction':float(np.mean(d< -1e-12)),'assessment_gain_quantiles':np.quantile(d,[0,.05,.25,.5,.75,.95,1]).tolist(),'assessment_gain_on_correct':float(d[~err].mean()) if (~err).any() else None,'assessment_gain_on_errors':float(d[err].mean()) if err.any() else None,'assessment_error_upcross_09':float(np.mean((sc['reference'][ai][err]<.9)&(sc['candidate'][ai][err]>=.9))) if err.any() else None,'assessment_error_downcross_09':float(np.mean((sc['reference'][ai][err]>=.9)&(sc['candidate'][ai][err]<.9))) if err.any() else None})
  checks.append({'dataset':x['dataset'],'records_sha256':hashlib.sha256((path/'records.npz').read_bytes()).hexdigest(),'gate_replay':True,'exact_fallback':True,'finite':True})
 rng=np.random.default_rng(2027);summary={'datasets':len(NEW),'authorizations':sum(x['authorized'] for x in comp),'old_rule_authorizations':sum(x['old_rule_authorized'] for x in comp),'positive_assessment_gains':sum(x['assessment_nll_gain']>0 for x in comp)}
 for key in ['assessment_nll_gain','assessment_policy_nll_gain','assessment_auc_gain','assessment_policy_auc_gain','assessment_harm_fraction']:
  v=np.array([x[key] for x in comp]);v=v[np.isfinite(v)];boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
  summary[key]={'mean':float(v.mean()),'descriptive_dataset_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'valid_datasets':len(v)}
 for name,data in [('assessment_metrics',rows),('assessment_comparisons',comp),('assessment_summary',summary),('replay_checks',checks)]:dump(OUT/(name+'.json'),data)
 print(json.dumps(summary,indent=2));print('All declared fresh policies released together.')
if __name__=='__main__':main()
