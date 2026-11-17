"""Review B: paired downstream decisions, interval and component ablations."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,time
import numpy as np
from scipy.stats import beta
from scipy.special import xlogy
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1];SEED=20261107;REPS=100;DELTA=.02;GAMMA=.01;ALPHA=.04
CFG={'seed':SEED,'repetitions':REPS,'K':[3,10,50,100],'S':[1000,10000,100000],'n':[2000,8000],'signal':[.7,.9],'delta':DELTA,'gamma':GAMMA,'alpha':ALPHA,'u':'fixed Dirichlet(5) per K','p':'least-mass extremizer','candidate':'fixed true conditional in controlled experiment, identical across every witness comparator','audit':'same outcomes and predictors for all witnesses; global witness event gamma; mean-score EB alpha','status':'full grid saved before first execution; reviewer-requested controlled experiment, not real-world validation'}
@lru_cache(None)
def coord(k,S,K,kind):
 if kind=='CP':
  l=0 if k==0 else beta.ppf(GAMMA/(2*K),k,S-k+1);h=1 if k==S else beta.ppf(1-GAMMA/(2*K),k+1,S-k)
 else:
  t=np.sqrt(np.log(2*K/GAMMA)/(2*S));l=max(0,k/S-t);h=min(1,k/S+t)
 return float(l),float(h),reverse_interval(float(l),DELTA)[0],reverse_interval(float(h),DELTA)[1]
def generic(m):return (np.log(1/m)-1+m)/(1-m+m*np.log(m))*DELTA

def bounds(cnt,kind):
 K=len(cnt);S=int(sum(cnt));r=(cnt+.5)/(S+.5*K);l,h,a,c=np.array([coord(int(k),S,K,kind) for k in cnt]).T;L=-np.log(r.min());m=l.min();offset=np.log(h/r).max();E=L if m==0 else min(L,float(kl(reverse_interval(m,DELTA)[1],m))+offset);EG=L if m==0 else min(L,generic(m)+offset)
 fa=xlogy(a,a/r);fc=xlogy(c,c/r);sl=(fc-fa)/(c-a);remaining=1-a.sum();C=fa.sum()
 for j in np.argsort(-sl):
  v=min(remaining,c[j]-a[j]);C+=v*sl[j];remaining-=v
  if remaining<1e-14:break
 ratio=np.log((c/r).max());parts={'universal':float(L),'chord':float(C),'ratio':float(ratio),'exact_envelope':float(E)};full=max(0,min(parts.values()));active=min(parts,key=parts.get)
 return {'generic_envelope':float(EG),'exact_envelope':float(E),'chord_only':float(min(L,C)),'ratio_only':float(min(L,ratio)),'full':float(full)},active

def main():
 protocol=R/'results/matched_protocol.json'
 if protocol.exists():
  old=json.loads(protocol.read_text());assert all(old[k]==v for k,v in CFG.items())
 if not protocol.exists():protocol.write_text(json.dumps(CFG|{'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
 rng=np.random.default_rng(SEED);trials=[];vectors=[];start=time.monotonic()
 for K in CFG['K']:
  u=rng.dirichlet(np.ones(K)*5);m=u.min();v=reverse_interval(m,DELTA)[1];p=u*(1-v)/(1-m);p[u.argmin()]=v;partition=np.arange(K)>=K//2;vectors.append({'K':K,'u':u.tolist(),'p':p.tolist()})
  for S in CFG['S']:
   for rep in range(REPS):
    cnt=rng.multinomial(S,u);r=(cnt+.5)/(S+.5*K);bd={};active={}
    for kind in ['CP','Hoeffding']:
     vals,ac=bounds(cnt,kind);bd.update({kind+'_'+k:v for k,v in vals.items()});active[kind]=ac
    assert all(b>=0 for b in bd.values());actual=float(np.sum(xlogy(p,p/r)))
    for signal in CFG['signal']:
     likelihood=np.stack([np.where(partition,1-signal,signal),np.where(partition,signal,1-signal)]);pg=likelihood@p;candidate=likelihood*p[None,:]/pg[:,None];I=float(np.sum(pg[:,None]*candidate*np.log(candidate/p)));support=np.log(candidate/r);Dpop=float(np.sum(pg[:,None]*candidate*support))
     for n in CFG['n']:
      y=rng.choice(K,n,p=p);g=np.where(rng.random(n)<signal,partition[y],~partition[y]).astype(int);D=support[g,y];radius=np.sqrt(2*D.var(ddof=1)*np.log(2/ALPHA)/n)+7*np.ptp(support)*np.log(2/ALPHA)/(3*(n-1));lower=float(D.mean()-radius)
      exactsupport=np.log(candidate/u);De=exactsupport[g,y];el=float(De.mean()-np.sqrt(2*De.var(ddof=1)*np.log(2/ALPHA)/n)-7*np.ptp(exactsupport)*np.log(2/ALPHA)/(3*(n-1)))
      exactw=float(kl(v,m));genw=min(float(-np.log(m)),generic(m));lows={k:lower-b for k,b in bd.items()};lows.update({'known_exact':el-exactw,'known_generic':el-genw})
      trials.append({'K':K,'S':S,'rep':rep,'n':n,'signal':signal,'information':I,'gain':float(D.mean()),'gain_population':Dpop,'gain_lower':lower,'actual_error':actual,'bounds':bd,'lower':lows,'active':active,'counts':cnt.tolist() if signal==.7 and n==2000 else None})
   print('completed',K,S,flush=True)
 rows=[]
 for K in CFG['K']:
  for S in CFG['S']:
   for n in CFG['n']:
    for signal in CFG['signal']:
     rr=[t for t in trials if (t['K'],t['S'],t['n'],t['signal'])==(K,S,n,signal)];methods=list(rr[0]['lower']);rows.append({'K':K,'S':S,'n':n,'signal':signal,'N':len(rr),'information':rr[0]['information'],'mean_actual_error':float(np.mean([t['actual_error'] for t in rr])),'methods':{m:{'mean_lower':float(np.mean([t['lower'][m] for t in rr])),'positive':sum(t['lower'][m]>0 for t in rr),'violations':sum(t['lower'][m]>t['information']+1e-9 for t in rr)} for m in methods},'CP_full_vs_envelope_gained':sum(t['lower']['CP_full']>0>=t['lower']['CP_exact_envelope'] for t in rr),'known_exact_vs_generic_gained':sum(t['lower']['known_exact']>0>=t['lower']['known_generic'] for t in rr),'CP_components':{k:sum(t['active']['CP']==k for t in rr) for k in ['universal','chord','ratio','exact_envelope']}})
 (R/'results/matched_categorical.json').write_text(json.dumps({'config':CFG,'rows':rows,'vectors':vectors,'trials':trials,'seconds':time.monotonic()-start},indent=2,default=lambda x:x.item()))
if __name__=='__main__':main()
