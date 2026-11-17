from pathlib import Path
import json,numpy as np
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1];rng=np.random.default_rng(30092027);out=[];beta=.01;delta=.02
for K in [3,10,50]:
 u=rng.dirichlet(np.ones(K)*5);m=u.min();v=reverse_interval(m,delta)[1];p=u*(1-v)/(1-m);p[u.argmin()]=v
 for S in [100,10000]:
  vals=[]
  for _ in range(1000):
   k=rng.multinomial(S,u);hat=k/S;rad=np.sqrt(np.log(2*K/beta)/(2*S));low=np.maximum(0,hat-rad);high=np.minimum(1,hat+rad);r=(k+.5)/(S+.5*K);L=np.log((S+.5*K)/.5);ml=low.min()
   if ml==0:b=L
   else:b=min(L,float(kl(reverse_interval(ml,delta)[1],ml))+np.max(np.log(high/r)))
   actual=float(np.sum(p*np.log(p/r)));vals.append((b,actual,ml==0,b+1e-10<actual))
  out.append({'K':K,'S':S,'trials':1000,'mean_bound':float(np.mean([x[0] for x in vals])),'mean_actual':float(np.mean([x[1] for x in vals])),'vacuous_fraction':float(np.mean([x[2] for x in vals])),'failure_fraction':float(np.mean([x[3] for x in vals]))})
(R/'results/categorical_mc.json').write_text(json.dumps({'beta':beta,'delta':delta,'rows':out},indent=2));print(out)
