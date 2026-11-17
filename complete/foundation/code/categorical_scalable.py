"""Check the least-mass exact categorical reduction and large-K scaling."""
from pathlib import Path
import json,time
import numpy as np
from experiments import kl,reverse_interval
from scipy.special import expit,logit,xlogy
from scipy.optimize import brentq
R=Path(__file__).resolve().parents[1]
def exact(q,delta):
 m=float(np.min(q))
 if delta==0:return 0.,m
 a=float(logit(m))
 def f(t):return float(xlogy(m,m)+xlogy(1-m,1-m)+np.logaddexp(0,t)-m*t-delta)
 right=a+1
 while f(right)<0:right=a+2*(right-a)
 v=float(expit(brentq(f,a,right,xtol=1e-12)))
 return float(kl(v,m)),v
old=json.loads((R/'results/multiclass_witness.json').read_text());err=[]
for x in old['bound_comparisons']:
 b,v=exact(np.array(x['baseline']),x['budget']);err.append(abs(b-x['exact_witness']))
assert max(err)<1e-9
rng=np.random.default_rng(29092026);rows=[]
for K in [10,100,1000,10000,100000]:
 for delta in [.001,.01,.1]:
  for rep in range(10):
   q=rng.dirichlet(np.ones(K)*2);t=time.perf_counter();b,v=exact(q,delta);runtime=time.perf_counter()-t;j=q.argmin();p=q*(1-v)/(1-q[j]);p[j]=v
   rev=float(np.sum(q*np.log(q/p)));fwd=float(np.sum(p*np.log(p/q)));assert abs(rev-delta)<1e-8 and abs(fwd-b)<1e-8
   rows.append({'K':K,'delta':delta,'rep':rep,'minimum_q':float(q[j]),'witness':b,'runtime_seconds':runtime,'reverse_kl':rev,'forward_kl':fwd})
# Analytic derivative identity and monotonic branches over a wide range.
checks=0
for delta in [.0001,.001,.01,.1,1]:
 ws=np.geomspace(1e-7,.5,300);fs=np.array([exact(np.array([w,1-w]),delta)[0] for w in ws]);assert np.max(np.diff(fs))<1e-8;checks+=len(ws)-1
out={'enumeration_cases':len(err),'max_enumeration_error':max(err),'monotonicity_checks':checks,'large_class_cases':len(rows),'rows':rows,'claim':'Exact fixed known baseline; find its least-probable class and solve one upper binary root. Numerical checks use floating point.'};(R/'results/categorical_scalable.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='rows'})
