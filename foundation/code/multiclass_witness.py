"""Exact finite-alphabet witness by enumerating binary partitions.
The mathematical reduction is proved in the manuscript. This implementation
uses floating-point roots and is practical only for small alphabets.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.special import xlogy
from scipy.optimize import minimize
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1]
def divergence(p,q):return float(np.sum(xlogy(p,p/q)))
def categorical_witness(q,delta):
 q=np.asarray(q,dtype=float)
 if q.ndim!=1 or len(q)<2 or np.any(q<=0) or not np.isclose(q.sum(),1):raise ValueError('q must be a strictly positive probability vector')
 if delta<0:raise ValueError('negative budget')
 if delta==0:return 0.,q.copy()
 best=-1.;bestp=None;K=len(q)
 # Complements give the same two endpoints, so enumerate subsets containing 0.
 for mask in range(2**(K-1)-1):
  ix=np.array([True]+[bool(mask&(1<<j)) for j in range(K-1)]);w=float(q[ix].sum())
  for v in reverse_interval(w,delta):
   value=float(kl(v,w))
   if value>best:
    best=value;bestp=np.where(ix,q*v/w,q*(1-v)/(1-w))
 return best,bestp

def main():
 rng=np.random.default_rng(24092027);rows=[];checks=0;details=[]
 for K in [3,5,8,10]:
  times=[];bounds=[];maxgap=0.;maxviolation=0.;optimizer_best=[]
  for t in range(12):
   q=rng.dirichlet(np.full(K,2.));delta=[.01,.05,.2][t%3]
   start=time.perf_counter();bound,pstar=categorical_witness(q,delta);times.append(time.perf_counter()-start);bounds.append(bound)
   assert abs(pstar.sum()-1)<1e-10 and divergence(q,pstar)<=delta+1e-8
   assert abs(divergence(pstar,q)-bound)<1e-9
   # Independent feasible random points and constrained numerical maximization.
   random=rng.dirichlet(np.ones(K),3000);reverse=np.sum(q*np.log(q/random),axis=1);valid=random[reverse<=delta+1e-12]
   for p in valid:assert divergence(p,q)<=bound+1e-8;checks+=1
   bestnum=0.
   for j in range(6):
    startp=q if j==0 else .9*q+.1*rng.dirichlet(np.ones(K))
    fit=minimize(lambda p:-divergence(p,q),startp,method='SLSQP',bounds=[(1e-12,1)]*K,constraints=[{'type':'eq','fun':lambda p:p.sum()-1},{'type':'ineq','fun':lambda p:delta-divergence(q,p)}],options={'maxiter':500,'ftol':1e-11})
    if fit.success and divergence(q,fit.x)<=delta+1e-7:
     bestnum=max(bestnum,-fit.fun);assert -fit.fun<=bound+1e-6
   maxgap=max(maxgap,bound-bestnum);maxviolation=max(maxviolation,bestnum-bound)
   eps=float(q.min());c=(np.log(1/eps)-1+eps)/(1-eps+eps*np.log(eps));generic=min(np.log(1/eps),c*delta);pinsker=min(np.log(1/eps),2*delta/eps)
   assert bound<=generic+1e-8 and bound<=pinsker+1e-8
   details.append({'classes':K,'budget':delta,'exact_witness':bound,'generic_divergence_bound':float(generic),'pinsker_bound':float(pinsker),'runtime_seconds':times[-1],'best_slsqp':bestnum,'maximizer':pstar.tolist(),'baseline':q.tolist()})
  rows.append({'classes':K,'partitions':2**(K-1)-1,'problems':12,'mean_runtime_seconds':float(np.mean(times)),'max_witness_minus_best_slsqp':float(maxgap),'max_slsqp_above_witness':float(maxviolation),'min_witness':min(bounds),'max_witness':max(bounds)})
 out={'seed':24092027,'budgets':[.01,.05,.2],'q_distribution':'Dirichlet(2,...,2)','rows':rows,'bound_comparisons':details,'feasible_random_points_checked':checks,'passed':True,'note':'Optimizer is a numerical comparison, not a proof of global optimality. Wall-clock timings are machine-specific.'}
 (R/'results/multiclass_witness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
