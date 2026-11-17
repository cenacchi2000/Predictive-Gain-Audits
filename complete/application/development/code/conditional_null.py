"""Separate a conditional-information null from a fitted-score null."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit,logsumexp

def logwealth(d,b):
    fixed=np.stack([np.log1p(d*f/b).sum(1) for f in 2.**np.arange(-8,1)],1)
    past=np.c_[np.zeros(len(d)),np.cumsum(d[:,:-1],1)]
    square=np.c_[np.zeros(len(d)),np.cumsum(d[:,:-1]**2,1)]
    lam=np.clip(past/(2*square+b*b),0,.5/b)
    return logsumexp(np.c_[fixed,np.log1p(lam*d).sum(1)],axis=1)-np.log(10)

def main():
    rng=np.random.default_rng(819);rows=[];repeats=5000
    eta=expit(np.array([2.,-2.,2.]));prior=eta.mean()
    entropy=lambda p:-p*np.log(p)-(1-p)*np.log1p(-p)
    gain=float(entropy(prior)-entropy(eta).mean())
    bound=float(np.max(np.abs(logit(eta)-logit(prior))))
    for n in [100,250,1000]:
        h=rng.integers(0,3,size=(repeats,n));p=eta[h];c=rng.binomial(1,p)
        d=c*np.log(p/prior)+(1-c)*np.log((1-p)/(1-prior))
        rows.append({'scenario':'conditional_null_misspecified_reference','n':n,'repeats':repeats,'conditional_information':0,'true_fitted_gain':gain,'bound':bound,'authorization_rate':float(np.mean(logwealth(d,bound)>=np.log(48/.05)))})
        # A reference that includes the nonlinear output term equals the candidate.
        rows.append({'scenario':'conditional_null_correct_reference','n':n,'repeats':repeats,'conditional_information':0,'true_fitted_gain':0,'bound':bound,'authorization_rate':0.})
    (Path(__file__).resolve().parents[1]/'results'/'conditional_null.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
