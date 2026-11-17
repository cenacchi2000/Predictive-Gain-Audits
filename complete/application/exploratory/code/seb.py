"""SEB-Cert reference implementation. Probabilities predict fixed-label correctness."""
import math
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

FAMILIES=[list(range(8))+[28],[8],list(range(9,12)),list(range(12,28))]

def shapley(values):
    values=np.asarray(values); ans=[]
    for j in range(4):
        z=np.zeros(values.shape[1:])
        for s in range(16):
            if not s&(1<<j):
                k=s.bit_count();w=math.factorial(k)*math.factorial(3-k)/24
                z=z+w*(values[s|(1<<j)]-values[s])
        ans.append(z)
    return np.asarray(ans)

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
    for cov in (.7,.8,.9,.95):
        k=int(np.floor(cov*n)); retained=order[:k];rev=order[k:];cap=int((1-c[rev]).sum())
        m['utility_'+str(cov)]={'coverage':k/n,'risk':float((1-c[retained]).mean()) if k else None,'precision':cap/len(rev) if len(rev) else None,'recall':cap/(1-c).sum() if (1-c).sum() else None,'reviewed':len(rev),'captured':cap}
    return m

def gate(c,r0,r1,ref,t1,tref,family_size=1,alpha=.05,margin=.01,min_class=20):
    """Seven one-sided bounds per frozen configuration; no bootstrap.
    Rows must be iid original subjects and independent of every fitted object.
    """
    c=np.asarray(c);n=len(c); n1=int(c.sum()); n0=n-n1
    z=math.log(7*family_size/alpha); L=math.log((1-EPS)/EPS)
    value=float(np.mean(logloss(c,r0)-logloss(c,r1)))
    lower=value-L*math.sqrt(2*z/n)
    result={'n':n,'errors':n0,'correct':n1,'value':value,'value_lcb':lower,'authorized':False}
    if n0<min_class or n1<min_class:
        result['reason']='fewer_than_20_distinct_cases_in_a_correctness_class';return result
    ag=auc(c,r1)-auc(c,ref); al=ag-math.sqrt(2*(1/n0+1/n1)*z)
    fg=float(np.mean((r1[c==0]>=.9).astype(float)-(ref[c==0]>=.9).astype(float)))
    fu=fg+math.sqrt(2*z/n0); rad=math.sqrt(z/(2*n))
    a1=np.asarray(r1)>=t1; ar=np.asarray(ref)>=tref
    e1=float(np.mean((1-c)*a1)); er=float(np.mean((1-c)*ar));p1=float(a1.mean());pr=float(ar.mean())
    upper= min(1,(e1+rad)/(p1-rad)) if p1>rad else 1.
    low=max(0,er-rad)/min(1,pr+rad)
    ru=upper-low
    result.update(auc_gap=ag,auc_lcb=al,fce_gap=fg,fce_ucb=fu,risk_gap_ucb=ru,candidate_coverage=p1,reference_coverage=pr)
    result['authorized']=bool(lower>0 and al>margin and fu<=0 and ru<=0 and p1>rad and pr>0)
    result['reason']='authorized' if result['authorized'] else 'one_or_more_simultaneous_bounds_failed'
    return result
