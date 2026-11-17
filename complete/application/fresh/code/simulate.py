"""Calibration-learned betting comparison in a fully specified Bernoulli model."""
from pathlib import Path
import numpy as np
from scipy.special import logit,logsumexp
from study import dump,OUT

def main():
 rng=np.random.default_rng(92027);B=float(logit(.8)-logit(.7));d1=np.log(.8/.7);d0=np.log(.2/.3);boundary=float(-d0/(d1-d0));rows=[];trials=5000;fractions=np.r_[0,2.**np.arange(-8,0)]
 for name,p in [('negative',.7),('boundary_null',boundary),('small_positive',.77),('positive',.8)]:
  cal=rng.binomial(500,p,trials);objectives=(cal[:,None]*np.log1p(fractions*d1/B)+(500-cal[:,None])*np.log1p(fractions*d0/B))/500
  rho=fractions[objectives.argmax(1)];c=rng.random((trials,2000))<p;d=np.where(c,d1,d0)
  past_sum=np.c_[np.zeros(trials),np.cumsum(d[:,:-1],axis=1)];past_sq=np.c_[np.zeros(trials),np.cumsum(d[:,:-1]**2,axis=1)]
  lam=np.clip(past_sum/(2*past_sq+B*B),0,.5/B);ad=np.cumsum(np.log1p(lam*d),axis=1);fixed=np.cumsum(np.log1p(rho[:,None]*d/B),axis=1)
  for n in [100,250,500,1000,2000]:
   learned=logsumexp(np.c_[np.log(.95)+fixed[:,n-1],np.log(.05)+ad[:,n-1]],axis=1)
   constants=np.stack([np.log1p(frac*d[:,:n]/B).sum(1) for frac in 2.**np.arange(-8,1)],axis=1)
   old=logsumexp(np.c_[constants,ad[:,n-1]],axis=1)-np.log(10)
   rows.append({'scenario':name,'p':p,'expected_gain':float(p*d1+(1-p)*d0),'calibration_n':500,'gate_n':n,'trials':trials,'family':10,'new_rate':float(np.mean(learned>=np.log(200))),'old_rate':float(np.mean(old>=np.log(200)))})
 dump(OUT/'calibration_betting_simulation.json',rows);print('Constructed comparison complete.')
if __name__=='__main__':main()
