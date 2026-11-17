"""Fresh paired comparison with exact-conditional-average TV Floodgate.
Original asymptotic bound and a separately labelled finite-sample betting variant.
"""
from pathlib import Path
import numpy as np,json,hashlib
from scipy.special import expit,logit,logsumexp
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.datasets import load_digits
from experiments import kl
R=Path(__file__).resolve().parents[1];SEED=20260929;REPS=20;ALPHA=.05;EPS=.02;B=3
cfg={'seed':SEED,'reps_per_fit_condition_size':REPS,'sizes':[2000,8000],'alpha':ALPHA,'teacher_and_probes':'Previously frozen representation study, all three fitted pairs in both conditions','comparator':'TV Floodgate with exact averaging over the known binary conditional feature law; asymptotic normal bound and separate bounded-betting variant','status':'Fresh comparison; complete design written before execution. Not external preregistration.'}
p=R/'results/tv_floodgate_protocol.json'
if not p.exists():p.write_text(json.dumps(cfg|{'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
def lower(T,a,c):
 fs=np.array([.01,.025,.05,.1,.25,.5,.9])
 def fn(m):return float(logsumexp(np.log1p(fs[:,None]*(T-m)/(m-a)).sum(1))-np.log(7)-np.log(1/ALPHA))
 return brentq(fn,a+1e-10,c) if fn(a+1e-10)>0 else a

def main():
 w=np.load(R/'results/representation_teacher.npz');fits=json.loads((R/'results/representation_fitted_probes.json').read_text());old=json.loads((R/'results/representation_audit.json').read_text());source=load_digits().data[old['pool_indices']]/16;rng=np.random.default_rng(SEED);raw=[]
 for f in fits:
  gen=f['genuine'];q0c=np.array(f['q0_coef']).ravel();q1c=np.array(f['q1_coef']).ravel()
  for n in cfg['sizes']:
   for rep in range(REPS):
    h=source[rng.integers(len(source),size=n)]+rng.normal(0,.03,(n,64));repres=np.maximum(0,np.einsum('ij,jk->ik',h,w['W0'])+w['b0']);repres=np.maximum(0,np.einsum('ij,jk->ik',repres,w['W1'])+w['b1']);t=np.clip((np.einsum('ij,jk->ik',repres,w['W2'])+w['b2']).ravel(),-4,4);z=2*rng.integers(0,2,n)-1;pp=EPS+(1-2*EPS)*expit(t+2*z) if gen else EPS+(1-2*EPS)*expit(t);p0=.5*(EPS+(1-2*EPS)*expit(t+2)+EPS+(1-2*EPS)*expit(t-2)) if gen else pp;y=rng.random(n)<pp
    r=np.clip(expit(np.einsum('ij,j->i',h,q0c)+f['q0_intercept'][0]),EPS,1-EPS);z0=logit(r);base=np.einsum('ij,j->i',repres,q1c[:-1] if gen else q1c)+f['q1_intercept'][0];target=base+q1c[-1]*z if gen else base;q=expit(z0+np.clip(target-z0,-B,B));D=np.where(y,np.log(q/r),np.log((1-q)/(1-r)));T=D-kl(p0,r);ours=lower(T,-B-np.log(1/EPS),B)
    fobs=np.where(y,q>r,q<r).astype(float)
    if gen:
     alternatives=[expit(z0+np.clip(base+q1c[-1]*zz-z0,-B,B)) for zz in [-1,1]];fbar=sum(np.where(y,qq>r,qq<r).astype(float) for qq in alternatives)/2;V=fobs-fbar
    else:V=np.zeros(n)
    normal=max(0,float(V.mean()-norm.ppf(1-ALPHA)*V.std(ddof=1)/np.sqrt(n)));bet=max(0,lower(V,-.5,.5));raw.append({'genuine':gen,'fit_seed':f['seed'],'n':n,'rep':rep,'ours_lower':ours,'TV_normal_information_lower':2*normal**2,'TV_betting_information_lower':2*bet**2,'V_mean':float(V.mean()),'information_integrand':float(kl(pp,p0).mean())})
   print(f['seed'],gen,n,flush=True)
 rows=[]
 for gen in [False,True]:
  for n in cfg['sizes']:
   rr=[x for x in raw if x['genuine']==gen and x['n']==n];d={'genuine':gen,'n':n,'trials':len(rr)}
   for key in ['ours_lower','TV_normal_information_lower','TV_betting_information_lower']:
    d['mean_'+key]=float(np.mean([max(0,x[key]) for x in rr]));d['positive_'+key]=float(np.mean([x[key]>0 for x in rr]))
   rows.append(d)
 (R/'results/tv_floodgate_comparison.json').write_text(json.dumps({'config':cfg,'rows':rows,'trials':raw},indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
