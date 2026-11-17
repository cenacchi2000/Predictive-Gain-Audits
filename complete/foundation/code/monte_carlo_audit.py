"""Independent check of the per-case failure-aware information certificate.
Synthetic latent likelihood is Bernoulli(1/2), so posterior predictive is 1/2.
All settings are constructed diagnostics, not a post-selection benchmark.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit,logit
from experiments import reverse_interval,kl
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(17092027)
n=2000; trials=1000; alpha=.05; beta=.01; eps=.05; B=2.; L=np.log(1/eps); delta=.005
rows=[]
for S in [16,64,256,1024]:
    radius=np.sqrt(np.log(2/beta)/(2*S))
    simultaneous_radius=np.sqrt(np.log(2*n/beta)/(2*S))
    lookup=[]; lookup_union=[]; reports=[]
    for k in range(S+1):
        u=k/S;r=np.clip(u,eps,1-eps);reports.append(r)
        for rad,target in [(radius,lookup),(simultaneous_radius,lookup_union)]:
            lo=reverse_interval(max(0,u-rad),delta)[0]
            hi=reverse_interval(min(1,u+rad),delta)[1]
            target.append(max(0.,float(kl(lo,r)),float(kl(hi,r))))
    lookup=np.array(lookup);lookup_union=np.array(lookup_union);reports=np.array(reports)
    rows_for_S=[]
    for genuine in [False,True]:
        lowers=[]; unions=[]; means=[]; bs=[]; misses=[]
        for t in range(trials):
            k=rng.binomial(S,.5,n);r=reports[k]
            g=rng.integers(0,2,n);p=np.where(g==1,.9,.1) if genuine else np.full(n,.5)
            y=rng.random(n)<p
            target=np.where(g==1,.85,.15) if genuine else np.full(n,.5)
            q=expit(logit(r)+np.clip(logit(target)-logit(r),-B,B))
            D=np.where(y,np.log(q/r),np.log((1-q)/(1-r)))
            T=D-lookup[k]
            def lower(z):
                return z.mean()-np.sqrt(2*z.var(ddof=1)*np.log(2/alpha)/n)-7*(2*B+L)*np.log(2/alpha)/(3*(n-1))
            lowers.append(lower(T)-beta*L)
            unions.append(lower(D-lookup_union[k])-beta*L/n)
            means.append(D.mean());bs.append(lookup[k].mean());misses.append(np.mean(kl(.5,r)>lookup[k]+1e-12))
        rows.append({'draws_per_case':S,'genuine_information':genuine,'audit_cases':n,'trials':trials,'mean_gain':float(np.mean(means)),'mean_witness':float(np.mean(bs)),'observed_witness_failure_fraction':float(np.mean(misses)),'failure_aware_positive_fraction':float(np.mean(np.array(lowers)>0)),'simultaneous_interval_positive_fraction':float(np.mean(np.array(unions)>0)),'mean_failure_aware_lower':float(np.mean(lowers)),'mean_simultaneous_lower':float(np.mean(unions))})
# Endpoint monotonicity makes the union over unknown predictive means exact.
for _ in range(1000):
    a,b=sorted(rng.uniform(0,1,2));d=10**rng.uniform(-5,-.3)
    left=reverse_interval(a,d)[0];right=reverse_interval(b,d)[1]
    for u in np.linspace(a,b,11):
        lo,hi=reverse_interval(u,d)
        assert left<=lo+1e-10 and hi<=right+1e-10
out={'seed':17092027,'alpha':alpha,'beta':beta,'epsilon':eps,'B':B,'reverse_budget':delta,'true_information_positive':float(np.log(2)+.9*np.log(.9)+.1*np.log(.1)),'rows':rows,'monotonicity_tests':1000,'note':'Both use the failure-aware theorem at level alpha; comparator uses conservative per-case beta/n and subtracts beta*L/n. Neither is optimized.'}
(ROOT/'results/monte_carlo_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out,indent=2))
