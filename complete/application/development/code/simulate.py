"""Power and null checks for frozen paired-score tests; no benchmark tuning."""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp, logit

def main():
    rng=np.random.default_rng(20270925);repeats=5000;sizes=[100,250,500,1000,2000]
    reference=.7;candidate=.8;b=float(logit(candidate)-logit(reference))
    d1=np.log(candidate/reference);d0=np.log((1-candidate)/(1-reference))
    null_p=-d0/(d1-d0);out=[]
    for name,p in [('negative',.7),('boundary_null',null_p),('small_gain',.77),('positive',.8)]:
        counts={m:{n:np.zeros(3,dtype=int) for n in sizes} for m in [1,48]}
        for offset in range(0,repeats,100):
            c=rng.binomial(1,p,size=(100,max(sizes)));d=np.where(c,d1,d0)
            sums=np.cumsum(d,1);squares=np.cumsum(d*d,1)
            logs=np.zeros((*d.shape,9))
            for j,f in enumerate(2.**np.arange(-8,1)):
                logs[:,:,j]=np.cumsum(np.log1p(d*f/b),1)
            ps=np.c_[np.zeros(len(d)),sums[:,:-1]];pv=np.c_[np.zeros(len(d)),squares[:,:-1]]
            lam=np.clip(ps/(2*pv+b*b),0,.5/b)
            adaptive=np.cumsum(np.log1p(lam*d),1)
            for n in sizes:
                le=logsumexp(np.c_[logs[:,n-1],adaptive[:,n-1]],axis=1)-np.log(10)
                mean=sums[:,n-1]/n;var=(squares[:,n-1]-sums[:,n-1]**2/n)/(n-1)
                for m in counts:
                    l=np.log(2*m/.05)
                    eb=mean-np.sqrt(2*np.maximum(var,0)*l/n)-14*b*l/(3*(n-1))
                    hoeff=mean-b*np.sqrt(2*np.log(m/.05)/n)
                    counts[m][n]+=np.array([(le>=np.log(m/.05)).sum(),(eb>0).sum(),(hoeff>0).sum()])
        for m in counts:
            for n in sizes:
                out.append({'scenario':name,'true_p':p,'true_gain':p*d1+(1-p)*d0,'bound':b,'n':n,'family':m,
                            'betting_rate':counts[m][n][0]/repeats,'bernstein_rate':counts[m][n][1]/repeats,'hoeffding_rate':counts[m][n][2]/repeats,'repeats':repeats})
    path=Path(__file__).resolve().parents[1]/'results'/'simulations.json';path.write_text(json.dumps(out,indent=2))
    print(path)

if __name__=='__main__':main()
