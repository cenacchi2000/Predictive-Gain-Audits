"""Constructed spectral side-channel with known conditional information.

This is a mechanism check, not an additional real-world benchmark. The
frozen classifier observes only the waveform mean. Frequency determines
reliability conditional on that mean. The output reference is an oracle.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from seb import bundle,logloss
from study import bounded_update,betting

def main():
    rng=np.random.default_rng(2519);h=np.array([-1.,-1.,1.,1.]);e=np.array([-1.,1.,-1.,1.])
    t=np.arange(128)/128;freq=np.where(e==1,3,19)
    x=h[:,None]+np.cos(2*np.pi*freq[:,None]*t[None,:])
    g=bundle(x);f=np.c_[h,g];eta=expit(2+h+2*e)
    ref=np.repeat(eta.reshape(2,2).mean(1),2)
    entropy=lambda a:-a*np.log(a)-(1-a)*np.log1p(-a)
    information=float(np.mean(entropy(ref)-entropy(eta)))
    rows=[]
    for trial in range(200):
        ic=rng.integers(0,4,1000);c=rng.binomial(1,eta[ic])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500)).fit(f[ic],c)
        s=np.clip(model.predict_proba(f)[:,1],1e-6,1-1e-6)
        # This constructed check fixes the direct learner and selects only B.
        b=min([.25,.5,1.,2.],key=lambda b:logloss(c,bounded_update(ref[ic],s[ic],b)).mean())
        q=bounded_update(ref,s,b)
        exact=float(np.mean(eta*np.log(q/ref)+(1-eta)*np.log((1-q)/(1-ref))))
        for n in [100,250,500,1000]:
            ii=rng.integers(0,4,n);cc=rng.binomial(1,eta[ii]);d=logloss(cc,ref[ii])-logloss(cc,q[ii]);gate=betting(d,b)
            rows.append({'trial':trial,'n':n,'exact_fitted_gain':exact,'bound':b,'authorized':gate['authorized'],'log_e':gate['log_e']})
    result={'conditional_information':information,'state_H':h.tolist(),'state_environment':e.tolist(),'state_frequency':freq.tolist(),
            'true_correctness':eta.tolist(),'oracle_output_reference':ref.tolist(),'calibration_n':1000,'trials':200,'family_threshold':48,
            'summary':[{'gate_n':n,'authorization_rate':float(np.mean([a['authorized'] for a in rows if a['n']==n])),
                        'mean_exact_fitted_gain':float(np.mean([a['exact_fitted_gain'] for a in rows if a['n']==n]))} for n in [100,250,500,1000]],'records':rows}
    p=Path(__file__).resolve().parents[1]/'results'/'positive_control.json';p.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
