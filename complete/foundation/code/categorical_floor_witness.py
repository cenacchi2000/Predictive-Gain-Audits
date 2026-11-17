"""Finite-sample multiclass witness that remains defined with unobserved classes."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,time
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import xlogy
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1];BETA=.01;DELTA=.02;REPS=1000;SEED=20260928
cfg={'K':[3,10,50,100],'S':[100,1000,10000],'beta':BETA,'delta':DELTA,'reps':REPS,'seed':SEED,'u':'Dirichlet(5,...,5), one fixed vector per K','p':'least-mass extremizer at delta=.02','report':'Jeffreys-smoothed empirical vector','status':'Design and full grid recorded before first execution. Comparison uses paired samples.'}
proto=R/'results/categorical_floor_protocol.json'
if not proto.exists():proto.write_text(json.dumps(cfg|{'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
@lru_cache(None)
def intervals(k,S,K,delta,beta):
 tail=beta/(2*K);lo=0. if k==0 else float(beta_dist.ppf(tail,k,S-k+1));hi=1. if k==S else float(beta_dist.ppf(1-tail,k+1,S-k));pl=reverse_interval(lo,delta)[0];pu=reverse_interval(hi,delta)[1]
 return lo,hi,pl,pu

def witness(counts,delta=DELTA,beta=BETA):
 S=int(np.sum(counts));K=len(counts);r=(counts+.5)/(S+.5*K);lo,hi,pl,pu=np.array([intervals(int(k),S,K,delta,beta) for k in counts]).T;L=float(-np.log(r.min()))
 if pl.sum()>1+1e-10 or pu.sum()<1-1e-10:return L,{'chord':L,'ratio':L,'envelope':L}
 fl=xlogy(pl,pl/r);fu=xlogy(pu,pu/r);slope=(fu-fl)/(pu-pl);rem=max(0.,1-pl.sum());chord=float(fl.sum())
 for i in np.argsort(-slope):
  extra=min(rem,pu[i]-pl[i]);chord+=slope[i]*extra;rem-=extra
  if rem<=1e-14:break
 ratio=float(np.log(np.max(pu/r)));m=float(lo.min());env=L if m==0 else float(kl(reverse_interval(m,delta)[1],m)+np.max(np.log(hi/r)))
 return max(0.,min(L,chord,ratio,env)),{'chord':chord,'ratio':ratio,'envelope':env}

def main():
 rng=np.random.default_rng(SEED);rows=[];start=time.monotonic();vectors=[]
 for K in cfg['K']:
  u=rng.dirichlet(np.ones(K)*5);m=u.min();v=reverse_interval(m,DELTA)[1];p=u*(1-v)/(1-m);p[u.argmin()]=v;vectors.append({'K':K,'u':u.tolist(),'p':p.tolist()})
  for S in cfg['S']:
   vals=[]
   for rep in range(REPS):
    counts=rng.multinomial(S,u);r=(counts+.5)/(S+.5*K);L=-np.log(r.min());b,parts=witness(counts);rad=np.sqrt(np.log(2*K/BETA)/(2*S));low=np.maximum(0,counts/S-rad);high=np.minimum(1,counts/S+rad);mm=low.min();old=L if mm==0 else min(L,float(kl(reverse_interval(mm,DELTA)[1],mm))+np.max(np.log(high/r)))
    actual=float(np.sum(xlogy(p,p/r)));vals.append([old,b,L,actual,old>=L-1e-10,b>=L-1e-10,b+1e-9<actual,b<.5,b<1.,parts['envelope']])
   a=np.array(vals);rows.append({'K':K,'S':S,'mean_old':float(a[:,0].mean()),'mean_new':float(a[:,1].mean()),'mean_L':float(a[:,2].mean()),'mean_actual':float(a[:,3].mean()),'old_vacuous':float(a[:,4].mean()),'new_vacuous':float(a[:,5].mean()),'violations':int(a[:,6].sum()),'fraction_below_half_nat':float(a[:,7].mean()),'fraction_below_one_nat':float(a[:,8].mean()),'mean_CP_envelope':float(a[:,9].mean())})
   print(rows[-1],flush=True)
 (R/'results/categorical_floor_witness.json').write_text(json.dumps({'config':cfg,'rows':rows,'vectors':vectors,'elapsed_seconds':time.monotonic()-start},indent=2))
if __name__=='__main__':main()
