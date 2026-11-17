from pathlib import Path
import json,math,sys,hashlib
import numpy as np
from seb import metrics,logloss,gate,shapley
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
rows=json.loads((OUT/'metrics.json').read_text());configs=json.loads((OUT/'configurations.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
assert len(configs)==protocol['family_size']
D=protocol['datasets'];methods=['raw','intercept','output','output_quadratic','time','random_projection','agreement','atypicality','all_features','spectral','reference','policy']
fields=['auc','nll','brier','ece','fce','aurc','excess_aurc','prevalence']
def usable(x):return x is not None and np.isfinite(x)
def avg(xs):
 xs=[x for x in xs if usable(x)];return float(np.mean(xs)) if xs else None
def summarize(condition,method,dataset=None):
 rr=[r for r in rows if r['condition']==condition and r['method']==method and (dataset is None or r['dataset']==dataset)]
 dd=D if dataset is None else [dataset]
 return {f:avg([avg([r[f] for r in rr if r['dataset']==d]) for d in dd]) for f in fields}|{'valid_auc':sum(usable(r['auc']) for r in rr),'valid_fce':sum(usable(r['fce']) for r in rr),'nconfigs':len(rr)}
summary={condition:{m:summarize(condition,m) for m in methods} for condition in ['clean','block','dominant','band']}
checks=0;harm=[];utilities=[];ledger=[];examples=[]
for cf in configs:
 folder=OUT/f"{cf['dataset']}_{cf['backbone']}_{cf['seed']}";a=np.load(folder/'predictions.npz');c=a['correct'];test=a['roles']=='test';cal=a['roles']=='calibration';gg=a['roles']=='gate'
 assert len(set(a['ids']))==len(a['ids']);assert set(a['roles']) in ({'fit','calibration','gate','test'},{'gate','test'})
 for key in a.files:
  if a[key].dtype.kind in 'fc':assert np.isfinite(a[key]).all(),(folder,key)
 assert np.array_equal(a['correct'],(a['predicted']==a['label']).astype(int))
 if not cf['gate']['authorized']:assert np.array_equal(a['policy'],a['reference'])
 g=gate(c[gg],a['output'][gg],a['spectral'][gg],a['reference'][gg],cf['threshold_spectral'],cf['threshold_reference'],protocol['family_size'])
 for key,val in g.items():
  if isinstance(val,float):assert abs(val-cf['gate'][key])<1e-12,(folder,key)
  else:assert val==cf['gate'][key],(folder,key)
 vv=logloss(c[None,:],a['coalitions'][0][None,:])-logloss(c[None,:],a['coalitions'])
 ph=shapley(vv)
 if 'shapley' in a.files:np.testing.assert_allclose(ph,a['shapley'],atol=1e-12)
 delta=vv[15,test];ct=c[test];spec=a['spectral'][test];out=a['output'][test];ref=a['reference'][test]
 h={k:cf[k] for k in ['dataset','backbone','seed']};h.update(mean=float(delta.mean()),negative_fraction=float((delta<0).mean()),correct_mean=avg(delta[ct==1]),error_mean=avg(delta[ct==0]),q10=float(np.quantile(delta,.1)),q50=float(np.quantile(delta,.5)),q90=float(np.quantile(delta,.9)),high_error_up=avg(((spec[ct==0]>=.9)&(ref[ct==0]<.9)).astype(float)),high_error_down=avg(((spec[ct==0]<.9)&(ref[ct==0]>=.9)).astype(float)))
 harm.append(h);ledger.append({**{k:cf[k] for k in ['dataset','backbone','seed']},'allocations':ph[:,test].mean(1).tolist()})
 for i in np.argsort(delta)[:1]:examples.append({**{k:cf[k] for k in ['dataset','backbone','seed']},'test_index':int(i),'correct':int(ct[i]),'output':float(out[i]),'spectral':float(spec[i]),'gain':float(delta[i])})
 for method in methods:
  mm=metrics(ct,a[method][test]);rr=next(r for r in rows if r['dataset']==cf['dataset'] and r['backbone']==cf['backbone'] and r['seed']==cf['seed'] and r['condition']=='clean' and r['method']==method)
  for f in fields:
   if usable(mm[f]):assert abs(mm[f]-rr[f])<1e-12
  for cov in (.7,.8,.9,.95):
   u=mm['utility_'+str(cov)];q=u['coverage'];assert abs(q*u['risk']+(1-q)*u['precision']-mm['prevalence'])<1e-12
   utilities.append({**{k:cf[k] for k in ['dataset','backbone','seed']},'method':method,'nominal_coverage':cov,**u})
 for condition in ['block','dominant','band']:
  zz=np.load(folder/(condition+'.npz'))
  for method in methods:
   mm=metrics(zz['correct'],zz[method]);rr=next(r for r in rows if r['dataset']==cf['dataset'] and r['backbone']==cf['backbone'] and r['seed']==cf['seed'] and r['condition']==condition and r['method']==method)
   for f in fields:
    if usable(mm[f]):assert abs(mm[f]-rr[f])<1e-12
 checks+=1

def hmacro(field,dataset=None):
 return avg([avg([h[field] for h in harm if h['dataset']==d]) for d in (D if dataset is None else [dataset])])
def fmt(x):return '--' if not usable(x) else f'{x:.3f}'
labels={'raw':'Raw class confidence','intercept':'Calibration intercept','output':'Output only','output_quadratic':'Quadratic outputs','time':'Time summaries','random_projection':'Random projection','agreement':'Inter-model agreement','atypicality':'Atypicality proxy','all_features':'All-feature control','spectral':'Spectral candidate','reference':'Selected reference','policy':'Authorized policy'}
main=r'''\begin{table}[t]
\centering\small
\caption{Fresh clean-test results. Equal dataset weights; each dataset averages six configurations. Candidate, reference, and deployed policy are distinct. These are descriptive fixed-suite means, not confidence bounds.}
\label{tab:main}
\begin{tabular}{lrrr}\toprule
Score & AUROC & NLL & AURC\\\midrule
'''
for m in methods:
 s=summary['clean'][m];main+=f"{labels[m]} & {fmt(s['auc'])} & {fmt(s['nll'])} & {fmt(s['aurc'])} \\\\\n"
main+=r'\bottomrule\end{tabular}\end{table}'+'\n'
main+=r'''\begin{table}[t]
\centering\small
\caption{Candidate value and deployment by dataset. $\widehat\Delta$ compares the spectral candidate with output only in nats; ranking compares candidate (S) with the selected reference (R). Auth. counts gate authorizations out of six frozen configurations.}
\label{tab:dataset}
\begin{tabular}{@{}lrrrr@{}}\toprule
Dataset & R AUC & S AUC & $\widehat\Delta$ & Auth.\\\midrule
'''
for d in D:
 main+=f"{d} & {fmt(summarize('clean','reference',d)['auc'])} & {fmt(summarize('clean','spectral',d)['auc'])} & {fmt(hmacro('mean',d))} & {sum(c['gate']['authorized'] for c in configs if c['dataset']==d)}/6 \\\\\n"
main+=r'\bottomrule\end{tabular}\end{table}'+'\n'
(ROOT/'main_tables.tex').write_text(main)
# Full metric table.
supp=r'''\begin{table}[H]\centering\small
\caption{All clean-test metrics, with the same dataset-first weighting. FCE is conditional on errors; excess AURC subtracts the exact per-configuration oracle.}\begin{tabular}{lrrrrrrr}\toprule
Method & AUROC & NLL & Brier & ECE & FCE & AURC & Excess\\\midrule
'''
for m in methods:
 s=summary['clean'][m];supp+=labels[m]+' & '+' & '.join(fmt(s[f]) for f in ['auc','nll','brier','ece','fce','aurc','excess_aurc'])+r' \\'+'\n'
supp+=r'\bottomrule\end{tabular}\end{table}'+'\n'
supp+=r'''\begin{table}[H]\centering\small
\caption{Signed spectral log-score gain relative to output only. Means are shown separately for correct and incorrect predictions. Negative fraction counts harmful realized adjustments, irrespective of authorization.}\begin{tabular}{lrrrrr}\toprule
Dataset & Mean & Correct & Errors & Negative fraction & Upward FCE\\\midrule
'''
for d in D:supp+=d+' & '+' & '.join(fmt(hmacro(f,d)) for f in ['mean','correct_mean','error_mean','negative_fraction','high_error_up'])+r' \\'+'\n'
supp+=r'\bottomrule\end{tabular}\end{table}'+'\n'
supp+=r'''\begin{table}[H]\centering\small
\caption{Targeted test corruptions with a frozen clean-data policy. Band removal zeros a random contiguous 20\% band; dominant removal zeros the 20\% largest-magnitude bins; block masking zeros a contiguous 20\% time segment.}\begin{tabular}{lrrrrrr}\toprule
Condition & R AUC & S AUC & R FCE & S FCE & R AURC & S AURC\\\midrule
'''
for cond in ['clean','block','dominant','band']:
 s=summary[cond]['spectral'];r=summary[cond]['reference'];supp+=cond+' & '+' & '.join(fmt(x) for x in [r['auc'],s['auc'],r['fce'],s['fce'],r['aurc'],s['aurc']])+r' \\'+'\n'
supp+=r'\bottomrule\end{tabular}\end{table}'+'\n'
supp+=r'''\begin{table}[H]\centering\small
\caption{Finite-budget utility. Mean actual coverage differs slightly from nominal coverage through integer rounding. R denotes selected reference; S denotes spectral candidate.}\begin{tabular}{llrrrr}\toprule
Nominal & Score & Actual & Risk & Recall & Precision\\\midrule
'''
for cov in (.7,.8,.9,.95):
 for m in ['reference','spectral']:
  xx=[u for u in utilities if u['method']==m and u['nominal_coverage']==cov]
  vals=[avg([avg([u[f] for u in xx if u['dataset']==d]) for d in D]) for f in ['coverage','risk','recall','precision']]
  supp+=f'{cov:.2f} & '+('R' if m=='reference' else 'S')+' & '+' & '.join(fmt(v) for v in vals)+r' \\'+'\n'
supp+=r'\bottomrule\end{tabular}\end{table}'+'\n'
supp+=r'''\begin{longtable}{llrlrrr}\caption{Every frozen reference and gate outcome. All counts refer to distinct gate examples within configuration. $n_-$ is the gate error count; $n_+$ is the gate correct count.}\\
\toprule Dataset & Backbone & Seed & Reference & $n_-$ & $n_+$ & Auth.\\\midrule\endfirsthead
\toprule Dataset & Backbone & Seed & Reference & $n_-$ & $n_+$ & Auth.\\\midrule\endhead
'''
for c in configs:
 supp+=f"{c['dataset']} & {c['backbone']} & {c['seed']} & {c['reference'].replace('_',' ')} & {c['gate']['errors']} & {c['gate']['correct']} & {int(c['gate']['authorized'])} \\\\\n"
supp+=r'\bottomrule\end{longtable}'+'\n'
supp+=r'''\begin{table}[H]\centering\small
\caption{Mean exact evidence allocations in nats. The four columns sum to the full fitted log-score gain before rounding.}\begin{tabular}{@{}lrrrr@{}}\toprule
Dataset & Energy & Entropy & Peaks & Phase\\\midrule
'''
for d in D:
 a=np.mean([z['allocations'] for z in ledger if z['dataset']==d],axis=0);supp+=d+' & '+' & '.join(fmt(v) for v in a)+r' \\'+'\n'
supp+=r'\bottomrule\end{tabular}\end{table}'+'\n'
worst=min(examples,key=lambda x:x['gain'])
supp+=r'\paragraph{A harmful retrospective example.} '+f"The most negative stored test gain occurs for {worst['dataset']}, {worst['backbone']}, seed {worst['seed']}, zero-based test row {worst['test_index']}. This incorrectly classified case has output-only correctness score {worst['output']:.8f} and spectral score {worst['spectral']:.8f}, giving signed log-score value {worst['gain']:.5f} nats. The example is selected retrospectively to expose a failure, not used to fit or authorize a score. Its coalition record is included in the artifact.\n"
(ROOT/'supp_tables.tex').write_text(supp)
# Counterexample and numeric validation report.
eta=1/(1+np.exp(-np.array([2.,-2.,2.])));hb=lambda p: -p*np.log(p)-(1-p)*np.log1p(-p)
report={'configurations_verified':checks,'authorized':sum(c['gate']['authorized'] for c in configs),'insufficient_classes':sum(c['gate']['reason'].startswith('fewer') for c in configs),'conditional_null_information':0,'conditional_null_fitted_gain':float(hb(eta.mean())-hb(eta).mean()),'mean_test_gain':hmacro('mean'),'mean_negative_fraction':hmacro('negative_fraction'),'mean_correct_gain':hmacro('correct_mean'),'mean_error_gain':hmacro('error_mean'),'undefined_auc_per_method':{m:36-summary['clean'][m]['valid_auc'] for m in methods},'undefined_fce_per_method':{m:36-summary['clean'][m]['valid_fce'] for m in methods},'missing_fit_classes_configs':sum(c['fit_missing_classes']>0 for c in configs),'maximum_closure_error':max(c['coalition_closure'] for c in configs),'log_radius_min':min(math.log((1-1e-6)/1e-6)*math.sqrt(2*math.log(7*36/.05)/c['gate_n']) for c in configs)}
for name,obj in [('summary',summary),('harm',harm),('utility',utilities),('ledger',ledger),('harm_examples',examples),('validation',report)]:
 (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=True,default=lambda x:x.item()))
print(json.dumps(report,indent=2,default=lambda x:x.item()));print(json.dumps(summary['clean'],indent=2,default=lambda x:x.item()))
