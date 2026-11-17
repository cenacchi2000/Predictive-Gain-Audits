"""SEB-Cert reference implementation. Probabilities predict fixed-label correctness."""
import numpy as np
from scipy.stats import rankdata

EPS=1e-6

def logloss(c,r):
    r=np.clip(np.asarray(r,float),EPS,1-EPS); c=np.asarray(c)
    return -c*np.log(r)-(1-c)*np.log1p(-r)

def bundle(x,bands=8,tau=1e-12):
    """29 entries: energy8, entropy1, peaks3, phase8, missing8, zero1.
    Input (N,C,T) or (N,T). Support threshold is relative to total power.
    FFT convention is the ordinary unweighted positive-frequency spectrum.
    """
    x=np.asarray(x,float)
    if x.ndim==2:x=x[:,None,:]
    if x.ndim!=3 or x.shape[-1]<2 or not np.isfinite(x).all():
        raise ValueError('Need finite N,C,T array, T>=2')
    x=x-x.mean(-1,keepdims=True)
    scale=np.max(np.abs(x),axis=(1,2),keepdims=True)
    x=np.divide(x,scale,out=np.zeros_like(x),where=scale>0)
    f=np.fft.rfft(x,axis=-1)[...,1:]; power=np.abs(f)**2
    total=power.sum((1,2)); q=np.divide(power.sum(1),total[:,None],out=np.zeros((len(x),f.shape[-1])),where=total[:,None]>0)
    groups=np.array_split(np.arange(f.shape[-1]),bands)
    energy=np.stack([q[:,ix].sum(1) for ix in groups],1)
    logq=np.zeros_like(q);np.log(q,out=logq,where=q>0)
    ent=-(q*logq).sum(1)
    if f.shape[-1]>1:ent/=np.log(f.shape[-1])
    else:ent[:]=0
    ordered=np.sort(q,axis=1)[:,::-1]
    peaks=np.stack([ordered[:,:j].sum(1) for j in (1,3,5)],1)
    valid=power>tau*total[:,None,None]
    unit=np.divide(f,np.abs(f),out=np.zeros_like(f),where=np.abs(f)>0)
    phase=[]; missing=[]
    for ix in groups:
        if x.shape[1]==1:
            a=ix[:-1]; b=ix[1:]
            v=valid[:,0,a]&valid[:,0,b]
            z=unit[:,0,b]*unit[:,0,a].conj()
            count=v.sum(1); num=(z*v).sum(1)
        else:
            # O(C*T), no explicit quadratic channel-pair array.
            v=valid[:,:,ix]; u=unit[:,:,ix]*v
            count=(v.sum(1)*(v.sum(1)-1)).sum(1)
            num=(np.abs(u.sum(1))**2-v.sum(1)).sum(1)
        phase.append(np.divide(np.abs(num),count,out=np.zeros(len(x)),where=count>0))
        missing.append((count==0).astype(float))
    return np.column_stack([energy,ent,peaks,np.stack(phase,1),np.stack(missing,1),total==0])

def auc(c,r):
    c=np.asarray(c); n1=c.sum();n0=len(c)-n1
    if not n1 or not n0:return float('nan')
    ranks=rankdata(r,method='average')
    return float((ranks[c==1].sum()-n1*(n1+1)/2)/(n1*n0))

def metrics(c,r):
    c=np.asarray(c);r=np.asarray(r);n=len(c)
    order=np.lexsort((np.arange(n),-r));err=1-c[order]
    risks=np.cumsum(err)/np.arange(1,n+1)
    bins=np.array_split(np.argsort(r,kind='stable'),min(15,n))
    ece=sum(len(z)/n*abs(c[z].mean()-r[z].mean()) for z in bins)
    prevalence=float(1-c.mean())
    oracle=np.r_[np.zeros(int(c.sum())),np.ones(n-int(c.sum()))]
    oa=float(np.mean(np.cumsum(oracle)/np.arange(1,n+1)))
    m={'n':n,'errors':int((1-c).sum()),'accuracy':float(c.mean()),'auc':auc(c,r),'nll':float(logloss(c,r).mean()),'brier':float(np.mean((c-r)**2)),'ece':float(ece),'fce':float(np.mean(r[c==0]>=.9)) if (c==0).any() else None,'aurc':float(risks.mean()),'excess_aurc':float(risks.mean()-oa),'prevalence':prevalence}
    # Exact expectation under uniformly random ordering inside every score tie.
    expected=np.cumsum(err).astype(float);s=np.asarray(r)[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];ends=np.r_[starts[1:],n]
    for start,end in zip(starts,ends):
        before=0 if start==0 else expected[start-1]
        expected[start:end]=before+np.arange(1,end-start+1)*err[start:end].sum()/(end-start)
    m['aurc_tie_averaged']=float(np.mean(expected/np.arange(1,n+1)))
    m['excess_aurc_tie_averaged']=m['aurc_tie_averaged']-oa
    m['augrc']=float((expected.sum()-expected[-1]/2)/(n*n))
    m['oracle_aurc']=oa
    for cov in (.7,.8,.9,.95):
        k=int(np.floor(cov*n)); retained=order[:k];rev=order[k:];cap=int((1-c[rev]).sum())
        m['utility_'+str(cov)]={'coverage':k/n,'risk':float((1-c[retained]).mean()) if k else None,'precision':cap/len(rev) if len(rev) else None,'recall':cap/(1-c).sum() if (1-c).sum() else None,'reviewed':len(rev),'captured':cap}
        retained_errors=expected[k-1] if k else 0.;captured=float((1-c).sum()-retained_errors)
        m['utility_tie_'+str(cov)]={'coverage':k/n,'risk':float(retained_errors/k) if k else None,'precision':captured/(n-k) if n>k else None,'recall':captured/(1-c).sum() if (1-c).sum() else None,'reviewed':n-k,'captured':captured,'oracle_risk':max(0,k-c.sum())/k if k else None}
    return m
