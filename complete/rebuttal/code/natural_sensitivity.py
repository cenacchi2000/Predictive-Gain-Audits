"""Review C: full split/cap factorial on natural labels; no selected settings."""
from pathlib import Path
import numpy as np,json,hashlib,time
from scipy.special import expit,logit,logsumexp
from scipy.optimize import minimize,brentq
from scipy.stats import beta
from sklearn.datasets import load_breast_cancer,load_digits,load_wine
from experiments import kl
R=Path(__file__).resolve().parents[1];SEEDS=[20260927,20261001,20261007];SPLITS=[(.3,.35),(.4,.3),(.5,.25)];CAPS=[1.,2.,3.];GAMMA=.025/81;ALPHA=.025/81
CFG={'seeds':SEEDS,'train_witness_splits':SPLITS,'caps':CAPS,'alpha':ALPHA,'gamma':GAMMA,'analyses':81,'joint_error':.05,'candidate':'same Gaussian Laplace/512 draws as original study','context':'training-median binary context, same feature definitions as original','status':'full factorial saved before first execution; overlapping data, not independent external replication'}
def bound(T,a,c,alpha):
 fractions=np.array([.01,.025,.05,.1,.25,.5,.9])
 def f(m):return float(logsumexp(np.log1p(fractions[:,None]*(T[None,:]-m)/(m-a)).sum(1))-np.log(7)-np.log(1/alpha))
 return brentq(f,a+1e-10,c) if f(a+1e-10)>0 else a

def main():
 proto=R/'results/natural_sensitivity_protocol.json'
 if proto.exists():raise RuntimeError('Protocol already exists')
 proto.write_text(json.dumps(CFG|{'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2));rows=[];records=[]
 for name,loader in [('WDBC',load_breast_cancer),('Digits',load_digits),('Wine',load_wine)]:
  ds=loader();X=ds.data.astype(float);Y=(ds.target>=5).astype(int) if name=='Digits' else (ds.target==0).astype(int);hraw=X[:,1] if name=='WDBC' else X.mean(1) if name=='Digits' else X[:,0]
  for trainfrac,witfrac in SPLITS:
   for seed in SEEDS:
    rng=np.random.default_rng(seed);idx=rng.permutation(len(Y));nt=int(trainfrac*len(Y));nw=int(witfrac*len(Y));train=idx[:nt];wit=idx[nt:nt+nw];audit=idx[nt+nw:];n=len(audit);threshold=np.median(hraw[train]);H=(hraw>threshold).astype(int);mu=X[train].mean(0);sd=X[train].std(0);sd[sd==0]=1;Z=np.c_[np.ones(len(X)),(X-mu)/sd];zt=Z[train];yt=Y[train];dim=Z.shape[1]
    def objective(w):
     z=np.einsum('ij,j->i',zt,w);return float(np.logaddexp(0,z).sum()-np.einsum('i,i->',yt,z)+.5*np.einsum('i,i->',w,w)),np.einsum('ij,i->j',zt,expit(z)-yt)+w
    opt=minimize(objective,np.zeros(dim),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-8,'ftol':1e-12});assert opt.success;pp=expit(np.einsum('ij,j->i',zt,opt.x));cov=np.linalg.inv(np.eye(dim)+np.einsum('ij,i,ik->jk',zt,pp*(1-pp),zt));draws=rng.multivariate_normal(opt.x,cov,512);cand=expit(np.einsum('ij,sj->is',Z,draws)).mean(1);q0=[];ws={key:[] for key in ['family81','original_level']};counts=[]
    for h in [0,1]:
     tr=train[H[train]==h];wi=wit[H[wit]==h];r=float(np.clip((Y[tr].sum()+.5)/(len(tr)+1),.02,.98));q0.append(r);k=int(Y[wi].sum());m=len(wi);counts.append([m,k])
     for key,gamma in [('family81',GAMMA),('original_level',.025/9)]:
      lo=0 if k==0 else beta.ppf(gamma/4,k,m-k+1);hi=1 if k==m else beta.ppf(1-gamma/4,k+1,m-k);ws[key].append(max(float(kl(lo,r)),float(kl(hi,r))))
    r=np.array(q0)[H[audit]];y=Y[audit]
    for B in CAPS:
     q=expit(logit(r)+np.clip(logit(np.clip(cand[audit],1e-10,1-1e-10))-logit(r),-B,B));D=np.where(y,np.log(q/r),np.log((1-q)/(1-r)));row={'dataset':name,'train_fraction':trainfrac,'witness_fraction':witfrac,'seed':seed,'cap':B,'train_n':nt,'witness_n':nw,'audit_n':n,'threshold':float(threshold),'gain':float(D.mean()),'counts':counts}
     for key,alpha in [('family81',ALPHA),('original_level',.025/9)]:
      bw=np.array(ws[key])[H[audit]];support=[]
      for r0,b0 in zip(q0,ws[key]):
       for shift in [-B,B]:
        qv=expit(logit(r0)+shift);support.extend([np.log(qv/r0)-b0,np.log((1-qv)/(1-r0))-b0])
      a=float(min(support));c=float(max(support));lower=bound(D-bw,a,c,alpha);row[key+'_lower']=float(lower);row[key+'_witness']=float(bw.mean());assert np.all((D-bw>=a-1e-10)&(D-bw<=c+1e-10))
     rows.append(row);records.append({'key':[name,trainfrac,seed,B],'train':train.tolist(),'witness':wit.tolist(),'audit':audit.tolist(),'y':y.tolist(),'H':H[audit].tolist(),'q0':r.tolist(),'q1':q.tolist(),'witnesses':ws,'D':D.tolist()})
   print(name,trainfrac,flush=True)
 (R/'results/natural_sensitivity.json').write_text(json.dumps({'config':CFG,'rows':rows},indent=2));(R/'results/natural_sensitivity_records.json').write_text(json.dumps(records))
 print('Family81 positives',sum(x['family81_lower']>0 for x in rows),'original level',sum(x['original_level_lower']>0 for x in rows),flush=True)
if __name__=='__main__':main()
