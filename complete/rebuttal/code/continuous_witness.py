"""Review A: computable witness for a nonconjugate continuous posterior.
No exact normalizer or true posterior predictive is used by the witness.
Certificates are model-relative, not certificates for naturally occurring labels.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.special import expit,logit,ndtr,ndtri
from scipy.stats import beta,norm
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1];SEED=20261108;ALPHA=.045;GAMMA=.005;REPS=20
CFG={'seed':SEED,'normalizer_proposals':100000,'repetitions':REPS,'audit_sizes':[2000,8000],'quadrature_nodes':[32,128,512],'alpha':ALPHA,'global_normalizer_failure':GAMMA,'prior_sd':2.,'cap':2.,'target':'continuous-context posterior-predictive information under a fitted one-coefficient Bayesian logistic model; G is fair +/-1 independent of H','witness':'rejection-envelope normalizer UCB + deterministic convex-chord ELBO bound + uniform Gaussian quantile quadrature error','status':'full grid saved before first execution; small nonconjugate example, not arbitrary high-dimensional validated inference'}
def main():
 proto=R/'results/continuous_protocol.json'
 if proto.exists():raise RuntimeError('Protocol exists')
 proto.write_text(json.dumps(CFG|{'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2));start=time.monotonic();rng=np.random.default_rng(SEED);ds=load_breast_cancer();idx=rng.permutation(len(ds.target));tr=idx[:398];pool=idx[398:];X=ds.data[:,[1,0]];mu=X[tr].mean(0);sd=X[tr].std(0);X=(X-mu)/sd;y=(ds.target==0).astype(int);fit=LogisticRegression(C=1.,max_iter=1000).fit(X[tr],y[tr]);off=fit.intercept_[0]+X[tr,0]*fit.coef_[0,0];x=X[tr,1];yy=y[tr];lam=.25
 def V(t):
  t=np.atleast_1d(t);ans=[]
  for j in range(0,len(t),1024):
   tt=t[j:j+1024];eta=off[None,:]+tt[:,None]*x[None,:];ans.extend((.5*lam*tt**2+np.sum(np.logaddexp(0,eta)-yy*eta,axis=1)).tolist())
  return np.array(ans)
 def grad(t):return lam*t+np.sum(x*(expit(off+x*t)-yy))
 opt=minimize_scalar(lambda t:V(t)[0],bounds=(-20,20),method='bounded',options={'xatol':1e-12});m=float(opt.x);g=grad(m);pr=expit(off+x*m);var=1/(lam+np.sum(x*x*pr*(1-pr)));sdq=np.sqrt(var);M=lam+.25*np.sum(x*x);vm=V(m)[0]
 # Rigorous (in real arithmetic) upper expectation by convex chords and quadratic tails.
 edges=np.linspace(m-10*sdq,m+10*sdq,8193);ve=V(edges);z=(edges-m)/sdq;pdf=norm.pdf(z);prob=np.diff(ndtr(z));first=m*prob+sdq*(pdf[:-1]-pdf[1:]);sl=np.diff(ve)/np.diff(edges);evupper=float(np.sum(ve[:-1]*prob+sl*(first-edges[:-1]*prob)))
 for e,left in [(edges[0],True),(edges[-1],False)]:
  zz=(e-m)/sdq;P=ndtr(zz) if left else ndtr(-zz);z1=-norm.pdf(zz) if left else norm.pdf(zz);z2=P-zz*norm.pdf(zz) if left else P+zz*norm.pdf(zz);d1=(m-e)*P+sdq*z1;d2=(m-e)**2*P+2*(m-e)*sdq*z1+var*z2;evupper+=V(e)[0]*P+grad(e)*d1+.5*M*d2
 rm=m-g/lam;rs=1/np.sqrt(lam);logC=-vm+g*g/(2*lam)+.5*np.log(2*np.pi/lam)
 def accepted(batch):
  theta=rng.normal(rm,rs,batch);lr=norm.logpdf(theta,rm,rs);lw=-V(theta)-lr-logC;assert lw.max()<1e-8;return theta[np.log(rng.random(batch))<=np.minimum(0,lw)]
 proposals=CFG['normalizer_proposals'];ac=accepted(proposals);k=len(ac);upper=float(beta.ppf(1-GAMMA,k+1,proposals-k));logZupper=logC+np.log(upper);entropy=.5*np.log(2*np.pi*np.e*var);delta=max(0,float(evupper-entropy+logZupper));print('mode',m,'sd',sdq,'accept',k/proposals,'KL upper',delta,flush=True)
 # Independent posterior draws for audit outcomes; each accepted proposal is an exact draw.
 cache=np.array([]);proposal_counter=0
 def posterior(n):
  nonlocal cache,proposal_counter
  while len(cache)<n:
   batch=max(2048,int((n-len(cache))/(k/proposals)*1.1));cache=np.r_[cache,accepted(batch)];proposal_counter+=batch
  out=cache[:n];cache=cache[n:];return out
 nodes={S:m+sdq*ndtri((np.arange(S)+.5)/S) for S in CFG['quadrature_nodes']};table={};grid=10000
 for S in nodes:
  err=1/(2*S);a=np.array([reverse_interval(max(0,j/grid-err),delta)[0] for j in range(grid+1)]);c=np.array([reverse_interval(min(1,j/grid+err),delta)[1] for j in range(grid+1)]);table[S]=(a,c)
 def predict(o,g,S):
  ans=[]
  for j in range(0,len(o),256):ans.extend(expit(o[j:j+256,None]+g[j:j+256,None]*nodes[S][None,:]).mean(1).tolist())
  return np.array(ans)
 trials=[];records=[]
 for n in CFG['audit_sizes']:
  for rep in range(REPS):
   h=X[pool[rng.integers(len(pool),size=n)],0]+rng.normal(0,.03,n);gg=rng.choice([-1.,1.],n);o=fit.intercept_[0]+fit.coef_[0,0]*h;theta=posterior(n);truth=expit(o+theta*gg);ya=rng.random(n)<truth
   for S in nodes:
    u=.5*(predict(o,np.ones(n),S)+predict(o,-np.ones(n),S));r=np.clip(u,.01,.99);candidate=predict(o,gg,S);q=expit(logit(r)+np.clip(logit(np.clip(candidate,1e-12,1-1e-12))-logit(r),-2,2));a,c=table[S];lo=a[np.floor(u*grid).astype(int)];hi=c[np.ceil(u*grid).astype(int)];b=np.maximum(kl(lo,r),kl(hi,r));D=np.where(ya,np.log(q/r),np.log((1-q)/(1-r)));T=D-b;L=np.log(100);eb=float(T.mean()-np.sqrt(2*T.var(ddof=1)*np.log(2/ALPHA)/n)-7*(4+L)*np.log(2/ALPHA)/(3*(n-1)));row={'n':n,'S':S,'rep':rep,'gain':float(D.mean()),'witness':float(b.mean()),'lower':eb};trials.append(row);records.append({'n':n,'S':S,'rep':rep,'h':h.tolist() if S==32 else None,'g':gg.tolist() if S==32 else None,'y':ya.tolist() if S==32 else None,'q0':r.tolist(),'q1':q.tolist(),'b':b.tolist(),'D':D.tolist()})
  print('audit complete',n,flush=True)
 rows=[]
 for n in CFG['audit_sizes']:
  for S in nodes:
   rr=[t for t in trials if t['n']==n and t['S']==S];rows.append({'n':n,'S':S,'mean_gain':float(np.mean([t['gain'] for t in rr])),'mean_witness':float(np.mean([t['witness'] for t in rr])),'mean_lower':float(np.mean([t['lower'] for t in rr])),'positive':sum(t['lower']>0 for t in rr),'reps':REPS})
 out={'config':CFG,'rows':rows,'trials':trials,'normalizer_acceptances':k,'normalizer_upper_acceptance':upper,'logC':logC,'logZ_upper':logZupper,'Q_mode':m,'Q_sd':sdq,'EQ_V_upper':evupper,'entropyQ':entropy,'delta_upper':delta,'gradient_at_mode':float(g),'audit_proposals':proposal_counter,'training_indices':tr.tolist(),'source_indices':pool.tolist(),'fixed_intercept':float(fit.intercept_[0]),'fixed_texture_coefficient':float(fit.coef_[0,0]),'scaling_mean':mu.tolist(),'scaling_sd':sd.tolist(),'seconds':time.monotonic()-start}
 (R/'results/continuous_witness.json').write_text(json.dumps(out,indent=2));(R/'results/continuous_records.json').write_text(json.dumps(records));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
