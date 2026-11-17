from pathlib import Path
import json,numpy as np
from scipy.special import logsumexp,xlogy
from scipy.stats import beta
from scipy.optimize import linprog
from categorical_floor_witness import witness,intervals
R=Path(__file__).resolve().parents[1];rows=json.loads((R/'results/natural_label_audit.json').read_text())['rows'];recs=json.loads((R/'results/natural_label_records.json').read_text());cfg=json.loads((R/'results/natural_label_audit.json').read_text())['config'];maxscore=0
for row,r in zip(rows,recs):
 parts=[set(r[k]) for k in ['train_indices','witness_indices','audit_indices']];assert all(not parts[i]&parts[j] for i in range(3) for j in range(i));assert len(set.union(*parts))==row['N']
 y=np.array(r['y']);p=np.array(r['q0']);q=np.array(r['q1']);b=np.array(r['b']);D=np.where(y,np.log(q/p),np.log((1-q)/(1-p)));maxscore=max(maxscore,float(np.max(abs(D-r['D']))));T=D-b;a=row['support_lower'];c=row['support_upper'];assert np.all(T>=a-1e-10) and np.all(T<=c+1e-10)
 assert abs(D.mean()-row['mean_gain'])<1e-12;fr=np.array([.01,.025,.05,.1,.25,.5,.9]);m=row['betting_lower'];le=logsumexp(np.log1p(fr[:,None]*(T-m)/(m-a)).sum(1))-np.log(7);assert abs(le-np.log(1/cfg['alpha_per_audit']))<1e-7
 for w in row['witness_counts']:
  k=w['successes'];n=w['n'];tail=cfg['gamma_per_witness']/4
  lo=0 if k==0 else beta.ppf(tail,k,n-k+1);hi=1 if k==n else beta.ppf(1-tail,k+1,n-k);assert abs(lo-w['lower'])+abs(hi-w['upper'])<1e-12
assert abs(9*(cfg['alpha_per_audit']+cfg['gamma_per_witness'])-.05)<1e-12
rng=np.random.default_rng(94327);err=0
for rep in range(200):
 K=int(rng.integers(2,16));S=int(rng.integers(20,2000));u=rng.dirichlet(np.ones(K)*2);counts=rng.multinomial(S,u);bb,parts=witness(counts);lo,hi,a,c=np.array([intervals(int(k),S,K,.02,.01) for k in counts]).T;r=(counts+.5)/(S+.5*K);fl=xlogy(a,a/r);fc=xlogy(c,c/r);sl=(fc-fl)/(c-a);lp=linprog(-sl,A_eq=np.ones((1,K)),b_eq=[1],bounds=list(zip(a,c)),method='highs');assert lp.success;value=-lp.fun+np.sum(fl-sl*a);err=max(err,abs(value-parts['chord']));assert abs(value-parts['chord'])<1e-7
old=json.loads((R/'results/natural_label_pre_numerical_cleanup/natural_label_audit.json').read_text())['rows'];diff={k:max(abs(x[k]-y[k]) for x,y in zip(old,rows)) for k in ['mean_gain','betting_lower','eb_lower']}
assert all((x['betting_lower']>0)==(y['betting_lower']>0) for x,y in zip(old,rows))
out={'natural_cases':len(rows),'disjoint_splits':True,'score_replay_max_error':maxscore,'betting_inversion_checked':True,'CP_intervals_checked':True,'family_error_allocation':.05,'independent_LP_comparisons':200,'greedy_vs_LP_max_error':err,'numerical_cleanup_max_change':diff,'numerical_cleanup_conclusions_unchanged':True}
(R/'results/review_additions_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
