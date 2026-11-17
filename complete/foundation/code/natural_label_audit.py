"""Natural-label audit: independent binomial witness, approximate Bayesian candidate.
All nine dataset/split audits are reported; no teacher outcomes or oracle conditionals.
"""
from pathlib import Path
import numpy as np,json,hashlib,time
from scipy.special import expit,logit,logsumexp
from scipy.optimize import minimize,brentq
from scipy.stats import beta as beta_dist
from sklearn.datasets import load_breast_cancer,load_digits,load_wine
from experiments import kl
R=Path(__file__).resolve().parents[1];SEEDS=[20260927,20261001,20261007];B=2.;EPS=.02;ALPHA=.025/9;GAMMA=.025/9;DRAWS=512
cfg={'seeds':SEEDS,'train_fraction':.4,'witness_fraction':.3,'audit_fraction':.3,'B':B,'epsilon':EPS,'posterior_draws':DRAWS,'prior_precision':1.,'alpha_per_audit':ALPHA,'gamma_per_witness':GAMMA,'family_total_error':.05,'family_size':9,'permutations':999,'H':'two groups from training median of mean texture (WDBC), mean pixel intensity (digits), alcohol (wine)','labels':'natural labels: malignant; digit>=5; wine class0','candidate':'Laplace approximation to continuous Bayesian logistic posterior with standard-normal prior','scope':'finite-context natural-label information, iid-superpopulation assumption; not conditional on the original unbinned covariate','status':'Fixed before first execution; exploratory new experiment, not external preregistration'}
proto=R/'results/natural_label_protocol.json'
if not proto.exists():proto.write_text(json.dumps(cfg|{'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
def main():
 out=[];records=[];fits=[];start=time.monotonic()
 for name,loader in [('WDBC',load_breast_cancer),('Digits',load_digits),('Wine',load_wine)]:
  ds=loader();X=ds.data.astype(float);Y=(ds.target==0).astype(int) if name!='Digits' else (ds.target>=5).astype(int);hraw=X[:,1] if name=='WDBC' else X.mean(1) if name=='Digits' else X[:,0]
  for seed in SEEDS:
   rng=np.random.default_rng(seed);idx=rng.permutation(len(Y));nt=int(.4*len(Y));nw=int(.3*len(Y));train=idx[:nt];wit=idx[nt:nt+nw];audit=idx[nt+nw:];n=len(audit);threshold=float(np.median(hraw[train]));H=(hraw>threshold).astype(int)
   mu=X[train].mean(0);sd=X[train].std(0);sd[sd==0]=1;Z=np.c_[np.ones(len(X)),(X-mu)/sd];Zt=Z[train];yt=Y[train];d=Z.shape[1]
   def objective(w):
    z=np.einsum("ij,j->i",Zt,w);return float(np.logaddexp(0,z).sum()-np.einsum("i,i->",yt,z)+.5*np.einsum("i,i->",w,w)),np.einsum("ij,i->j",Zt,expit(z)-yt)+w
   opt=minimize(objective,np.zeros(d),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-8,'ftol':1e-12});assert opt.success
   w=opt.x;p=expit(np.einsum("ij,j->i",Zt,w));A=np.eye(d)+np.einsum("ij,i,ik->jk",Zt,p*(1-p),Zt);cov=np.linalg.inv(A);draws=rng.multivariate_normal(w,cov,DRAWS);cand=expit(np.einsum("ij,sj->is",Z,draws)).mean(1)
   q0=[];bounds=[];counts=[]
   for h in [0,1]:
    tr=train[H[train]==h];wi=wit[H[wit]==h];r=np.clip((Y[tr].sum()+.5)/(len(tr)+1),EPS,1-EPS);k=int(Y[wi].sum());m=len(wi);tail=GAMMA/4
    lo=0. if k==0 else float(beta_dist.ppf(tail,k,m-k+1));hi=1. if k==m else float(beta_dist.ppf(1-tail,k+1,m-k));b=max(float(kl(lo,r)),float(kl(hi,r)))
    q0.append(float(r));bounds.append(b);counts.append({'h':h,'n':m,'successes':k,'lower':lo,'upper':hi,'witness':b})
   r=np.array(q0)[H[audit]];q=expit(logit(r)+np.clip(logit(np.clip(cand[audit],1e-10,1-1e-10))-logit(r),-B,B));y=Y[audit];D=np.where(y,np.log(q/r),np.log((1-q)/(1-r)));bw=np.array(bounds)[H[audit]];T=D-bw
   # Deterministic support from the frozen baseline, logit cap, and witness, not observed extrema.
   supports=[]
   for r0,b0 in zip(q0,bounds):
    for shift in [-B,B]:
     qv=expit(logit(r0)+shift);supports += [np.log(qv/r0)-b0,np.log((1-qv)/(1-r0))-b0]
   a=float(min(supports));upper=float(max(supports));fractions=np.array([.01,.025,.05,.1,.25,.5,.9])
   def loge(m):return float(logsumexp(np.log1p(fractions[:,None]*(T[None,:]-m)/(m-a)).sum(1))-np.log(len(fractions)))
   # The mixture is decreasing in m; inversion yields a numerical lower mean bound.
   lower=brentq(lambda m:loge(m)-np.log(1/ALPHA),a+1e-10,upper) if loge(a+1e-10)>np.log(1/ALPHA) else a
   rad=np.sqrt(2*T.var(ddof=1)*np.log(2/ALPHA)/n)+7*(upper-a)*np.log(2/ALPHA)/(3*(n-1));eb=float(T.mean()-rad)
   # Exact stratified permutation test under Y independent of G given the two-valued H.
   observed=float(np.where(y,np.log(q),np.log1p(-q)).sum());exceed=0
   for rep in range(999):
    qp=q.copy()
    for h in [0,1]:
     group=np.flatnonzero(H[audit]==h);qp[group]=q[rng.permutation(group)]
    exceed+=np.where(y,np.log(qp),np.log1p(-qp)).sum()>=observed-1e-12
   pv=(1+exceed)/1000
   row={'dataset':name,'seed':seed,'N':len(Y),'train':nt,'witness_n':nw,'audit_n':n,'dimensions':d,'mean_gain':float(D.mean()),'mean_witness':float(bw.mean()),'betting_lower':float(lower),'eb_lower':eb,'log_e_zero':loge(0.),'permutation_p':pv,'support_lower':a,'support_upper':upper,'accuracy':float(np.mean((q>.5)==y)),'witness_counts':counts}
   out.append(row);records.append({'dataset':name,'seed':seed,'train_indices':train.tolist(),'witness_indices':wit.tolist(),'audit_indices':audit.tolist(),'H':H[audit].tolist(),'y':y.tolist(),'q0':r.tolist(),'q1':q.tolist(),'b':bw.tolist(),'D':D.tolist()});fits.append({'dataset':name,'seed':seed,'MAP':w.tolist(),'covariance':cov.tolist(),'threshold':threshold,'q0':q0,'bounds':bounds,'data_sha256':hashlib.sha256(np.c_[X,Y].tobytes()).hexdigest()})
   print(name,seed,'gain',round(D.mean(),4),'b',round(bw.mean(),4),'bettingLCB',round(lower,4),'EB',round(eb,4),'permutation',pv,flush=True)
 (R/'results/natural_label_audit.json').write_text(json.dumps({'config':cfg,'rows':out,'elapsed_seconds':time.monotonic()-start},indent=2));(R/'results/natural_label_records.json').write_text(json.dumps(records));(R/'results/natural_label_fits.json').write_text(json.dumps(fits))
if __name__=='__main__':main()
