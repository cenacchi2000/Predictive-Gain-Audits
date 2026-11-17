"""Data-conditioned finite Bayesian model; model-relative predictive audits.
All settings below are fixed before this script's first run. Real held-out labels
are descriptive checks only; certificate repetitions use independent draws from
the fitted posterior predictive, not a claim that the real data obey the model.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.special import expit,logsumexp,logit,xlogy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1];SEED=26092027;EPS=.01;ALPHA=.05;B=3.;BETA=.001;REPS=200
MASSES=[.9,.99,.999];DRAWS=[256,1024,4096];AUDIT=[2000,10000]
def ent(p):return -xlogy(p,p)-xlogy(1-p,1-p)
def radius(v,n):return np.sqrt(2*v.var(ddof=1)*np.log(2/ALPHA)/n)+7*(2*B+np.log(1/EPS))*np.log(2/ALPHA)/(3*(n-1))
def main():
 start=time.monotonic();ds=load_breast_cancer();y=(ds.target==0).astype(int);idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.3,random_state=SEED,stratify=y)
 # Existing measurement H=mean texture; extra measurement G=mean radius.
 thresholds=np.median(ds.data[tr][:,[1,0]],axis=0);binary=(ds.data[:,[1,0]]>thresholds).astype(int);cell=2*binary[:,0]+binary[:,1]
 counts=np.bincount(cell[tr],minlength=4);success=np.bincount(cell[tr],weights=y[tr],minlength=4);pcell=(counts+1)/(len(tr)+4);ph=pcell.reshape(2,2).sum(1);pg=pcell.reshape(2,2)/ph[:,None]
 grid=np.arange(-8,9,dtype=float);theta=np.array(np.meshgrid(grid,grid,grid,indexing='ij')).reshape(3,-1).T
 X=np.array([[1,2*h-1,2*g-1] for h in range(2) for g in range(2)])
 eta=np.einsum('ij,kj->ik',theta,X);pred=expit(eta)
 lp=-np.sum(theta**2,axis=1)/8;lp-=logsumexp(lp);ll=np.sum(success*np.logaddexp(0,-eta)*-1+(counts-success)*np.logaddexp(0,eta)*-1,axis=1);unnorm=lp+ll;post=np.exp(unnorm-logsumexp(unnorm))
 assert np.isclose(post.sum(),1,atol=1e-12)
 p1=np.einsum('i,ij->j',post,pred);p0=np.sum(p1.reshape(2,2)*pg,axis=1);I=float(np.sum(ph*ent(p0))-np.sum(pcell*ent(p1)))
 # Independent extended-precision finite-sum verification.
 ld=unnorm.astype(np.longdouble);ww=np.exp(ld-ld.max());ww/=ww.sum();ref=np.sum(ww[:,None]*pred.astype(np.longdouble),axis=0);err=float(np.max(np.abs(ref-p1)));assert err<1e-12
 model={'dataset':'Wisconsin Diagnostic Breast Cancer','features':['mean texture','mean radius'],'target':'malignant','seed':SEED,'train_n':len(tr),'heldout_n':len(te),'training_indices':tr.tolist(),'heldout_indices':te.tolist(),'thresholds':thresholds.tolist(),'posterior_states':len(theta),'grid':grid.tolist(),'prior':'independent N(0, 2^2) weights restricted and normalized on the stated grid; this grid defines the model','cell_probability':pcell.tolist(),'p0':p0.tolist(),'p1':p1.tolist(),'conditional_information':I,'float_longdouble_predictive_discrepancy':err,'data_sha256':hashlib.sha256(ds.data.tobytes()+ds.target.tobytes()).hexdigest()}
 rng=np.random.default_rng(SEED+1);rows=[];raw=[];approximations=[]
 for mass in MASSES:
  order=np.argsort(-post);num=np.searchsorted(np.cumsum(post[order]),mass)+1;keep=order[:num];retained=float(post[keep].sum());q=np.zeros_like(post);q[keep]=post[keep]/retained;delta=-np.log(retained)
  q1=np.einsum('i,ij->j',q,pred);u=np.sum(q1.reshape(2,2)*pg,axis=1)
  actual=float(np.sum(q[keep]*(np.log(q[keep])-np.log(post[keep]))));assert abs(actual-delta)<1e-12;assert np.all(kl(u,p0)<=delta+1e-12)
  exact_w=np.array([max(float(kl(t,r)) for t in reverse_interval(r,delta)) for r in u]);assert np.all(kl(p0,np.clip(u,EPS,1-EPS))<=exact_w+1e-10) # no clipping active here
  held_gain=float(np.mean(np.where(y[te],np.log(q1[cell[te]]/u[binary[te,0]]),np.log((1-q1[cell[te]])/(1-u[binary[te,0]])))))
  approximations.append({'retained_target':mass,'retained_mass':retained,'states':int(num),'reverse_budget':float(delta),'actual_forward_baseline_kl':float(np.sum(ph*kl(p0,u))),'exact_mean_witness':float(np.sum(ph*exact_w)),'q1':q1.tolist(),'u':u.tolist(),'heldout_predictive_gain':held_gain})
  for S in DRAWS:
   reps=np.clip(np.arange(S+1)/S,EPS,1-EPS);look=np.zeros(S+1);rad=np.sqrt(np.log(2/BETA)/(2*S))
   for k in range(S+1):
    a=reverse_interval(max(0,k/S-rad),delta)[0];c=reverse_interval(min(1,k/S+rad),delta)[1];look[k]=max(float(kl(a,reps[k])),float(kl(c,reps[k])))
   for n in AUDIT:
    for genuine in [False,True]:
     values=[]
     for rep in range(REPS):
      ix=rng.choice(4,n,p=pcell);h=ix//2;truth=p1[ix] if genuine else p0[h];yy=rng.random(n)<truth;k=rng.binomial(S,u[h]);r=reps[k];target=q1[ix] if genuine else u[h]
      qq=expit(logit(r)+np.clip(logit(target)-logit(r),-B,B));D=np.where(yy,np.log(qq/r),np.log((1-qq)/(1-r)));assert np.all(np.abs(D)<=B+1e-12)
      t=D-look[k];low=float(t.mean()-BETA*np.log(1/EPS)-radius(t,n));gain=float(D.mean());gain_lower=float(D.mean()-np.sqrt(2*D.var(ddof=1)*np.log(2/ALPHA)/n)-7*2*B*np.log(2/ALPHA)/(3*(n-1)))
      v={'mass':mass,'S':S,'n':n,'genuine':genuine,'rep':rep,'gain':gain,'predictive_lower':gain_lower,'information_lower':low,'witness':float(look[k].mean())};values.append(v);raw.append(v)
     l=np.array([v['information_lower'] for v in values]);rows.append({'mass':mass,'S':S,'n':n,'genuine':genuine,'mean_gain':float(np.mean([v['gain'] for v in values])),'mean_information_lower':float(l.mean()),'positive_information_rate':float(np.mean(l>0)),'coverage_failure_rate':float(np.mean(l>(I if genuine else 0))),'positive_predictive_rate':float(np.mean([v['predictive_lower']>0 for v in values])),'mean_witness':float(np.mean([v['witness'] for v in values]))})
   print('Completed mass',mass,'draws',S,flush=True)
 out={'model':model,'approximations':approximations,'rows':rows,'settings':{'epsilon':EPS,'alpha':ALPHA,'B':B,'beta':BETA,'repetitions':REPS,'audit_sizes':AUDIT,'draws':DRAWS,'retained_masses':MASSES},'elapsed_seconds':time.monotonic()-start,'scope':'Finite discrete Bayesian logistic model fitted to real training data. Audit outcomes are simulated from its exact posterior predictive. This is a model-relative validation and not certification of real-world conditional information. Posterior-predictive sampling uses Bernoulli(u), equivalent in law to integrating the approximate posterior before drawing the binary outcome. No audit-based selection.'}
 (R/'results/bayesian_application.json').write_text(json.dumps(out,indent=2));(R/'results/bayesian_application_trials.json').write_text(json.dumps(raw));np.savez_compressed(R/'results/bayesian_posterior.npz',theta=theta,posterior=post,likelihood_probability=pred)
 print('Model information',I,'approximations',approximations,flush=True)
 print('Final budget:',[r for r in rows if r['n']==10000 and r['S']==4096],flush=True)
if __name__=='__main__':main()
