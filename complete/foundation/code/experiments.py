"""Exact and Monte Carlo checks for approximation-aware information auditing."""
from pathlib import Path
import json
import numpy as np
from scipy.special import xlogy,xlog1py,expit,logsumexp,ndtri
from scipy.optimize import brentq,minimize
from scipy.stats import rankdata
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';P=ROOT/'paper';OUT.mkdir(exist_ok=True);P.mkdir(exist_ok=True)

def kl(p,q):
 return xlogy(p,p)-xlogy(p,q)+xlogy(1-p,1-p)-xlogy(1-p,1-q)

def reverse_interval(q,delta):
 if not 0<=q<=1 or delta<0:raise ValueError('Invalid budget or probability')
 if delta==0:return q,q
 if q==0:return 0.,float(-np.expm1(-delta))
 if q==1:return float(np.exp(-delta)),1.
 # Solve in log odds to avoid a fixed endpoint epsilon truncating the feasible set.
 from scipy.special import logit
 a=float(logit(q))
 def fun(t):return float(xlogy(q,q)+xlogy(1-q,1-q)+np.logaddexp(0.,t)-q*t-delta)
 left=a-1.;right=a+1.
 while fun(left)<0:left=a-2*(a-left)
 while fun(right)<0:right=a+2*(right-a)
 return float(expit(brentq(fun,left,a,xtol=1e-12))),float(expit(brentq(fun,a,right,xtol=1e-12)))

def witness(q,delta,reported=None):
 lo,hi=reverse_interval(q,delta);r=q if reported is None else reported
 return float(max(0.,kl(lo,r),kl(hi,r)))

def eb(count,n,p0,p1,alpha=.05):
 d1=np.log(p1/p0);d0=np.log((1-p1)/(1-p0));mean=(count*d1+(n-count)*d0)/n
 v=(d1-d0)**2*(count/n)*(1-count/n)*n/(n-1)
 return mean-np.sqrt(2*v*np.log(2/alpha)/n)-7*abs(d1-d0)*np.log(2/alpha)/(3*(n-1))

def rhat(a):
 n=a.shape[1];w=np.var(a,axis=1,ddof=1).mean();b=n*np.var(a.mean(1),ddof=1)
 return float(np.sqrt(((n-1)*w/n+b/n)/w))

def main():
 rng=np.random.default_rng(2027);report={}
 # Asymptotic separation: shrinking reverse KL and standard error, diverging gain.
 separation=[]
 for n in [10**k for k in range(2,13)]:
  p=n**(-.25);logq=-np.sqrt(n);q=np.exp(logq);log1q=np.log1p(-q)
  mean=float(xlogy(p,p)+(1-p)*np.log1p(-p)-p*logq-(1-p)*log1q)
  rev=float(xlogy(q,q)-q*np.log(p)+(1-q)*(log1q-np.log1p(-p)))
  d1=np.log(p)-logq;d0=np.log1p(-p)-log1q;se=float(np.sqrt(p*(1-p)*(d1-d0)**2/n))
  separation.append({'n':n,'true_information':0.,'p':p,'log_q':float(logq),'reverse_kl':rev,'fitted_gain':mean,'true_mc_standard_error':se})
 report['separation']=separation
 # Actual Gaussian VI on a separated two-component posterior, with exact quadrature.
 nodes,weights=hermgauss(100);zz=np.sqrt(2)*nodes;ww=weights/np.sqrt(np.pi);delta=.01;mu=16.
 def logphi(x,m):return -.5*(x-m)**2-.5*np.log(2*np.pi)
 def objective(v):
  loc,logsd=v;sd=np.exp(logsd);x=loc+sd*zz;lq=-.5*zz**2-.5*np.log(2*np.pi)-logsd
  lp=logsumexp(np.stack([np.log1p(-delta)+logphi(x,0),np.log(delta)+logphi(x,mu)]),axis=0)
  return float(np.dot(ww,lq-lp))
 fits=[minimize(objective,init,method='L-BFGS-B',bounds=[(-30,30),(-4,4)],options={'ftol':1e-13,'maxiter':1000}) for init in [[0,0],[16,0],[8,2],[0,1]]]
 best=min(fits,key=lambda a:a.fun);loc,logsd=best.x;r0=float(ww@expit(4*(loc+np.exp(logsd)*zz-8)));common=float(ww@expit(4*(zz-8)));rare=float(ww@expit(4*(mu+zz-8)));truth=(1-delta)*common+delta*rare
 # For the declared reference N(0,1), pi >= (1-delta)q gives an analytic posterior reverse-KL upper bound.
 r0=common;budget=-np.log1p(-delta);b=witness(r0,budget)
 report['mixture_vi']={'mixture_weight':delta,'separation':mu,'optimizer_success':[bool(a.success) for a in fits],'best_gaussian_mean':float(loc),'best_gaussian_sd':float(np.exp(logsd)),'best_gaussian_reverse_kl_quadrature':float(best.fun),'reference_probability':r0,'true_probability':float(truth),'true_information':0.,'posterior_reverse_kl_upper':float(budget),'predictive_forward_kl':float(kl(truth,r0)),'binary_witness':b,'quadrature_nodes':100,'reference':'N(0,1), matching the numerically optimized Gaussian to reported precision'}
 # Fixed-model sampling uncertainty only: compare valid fitted-gain and information claims.
 sims=[];trials=5000
 for scenario,p0,p1,truep,bias,trueinfo in [('redundant_evidence',r0,truth,truth,b,0.),('genuine_evidence',.5,.85,.9,witness(.5,.01),float(np.log(2)+.9*np.log(.9)+.1*np.log(.1)))]:
  for n in [100,500,2000,10000]:
   count=rng.binomial(n,truep,trials);lower=eb(count,n,p0,p1)
   sims.append({'scenario':scenario,'n':n,'trials':trials,'true_information':trueinfo,'baseline_witness':float(bias),'fitted_gain_authorization_rate':float(np.mean(lower>0)),'information_authorization_rate':float(np.mean(lower-bias>0)),'mean_fitted_lower':float(lower.mean()),'mean_information_lower':float((lower-bias).mean())})
 report['audit_simulations']=sims
 # Mean-field VI is not invariant to a rotation, although the exact Gaussian model is.
 gaussian=[]
 for rho in [.1,.3,.5,.7,.9,.99]:
  exactvar=2*(1+rho);mfvar=2*(1-rho*rho);gain=.5*(exactvar/mfvar-1-np.log(exactvar/mfvar))
  gaussian.append({'rho':rho,'posterior_reverse_kl':float(-.5*np.log1p(-rho*rho)),'predictive_forward_kl':float(gain),'rotated_meanfield_forward_kl':0.,'true_new_information':0.})
 report['gaussian_reparameterization']=gaussian
 # A stationary-target, slowly switching sampler; all chains deliberately start in common mode.
 n=10000;chains=4;gamma=1e-5;mode=np.zeros(chains,dtype=int);draws=np.empty((chains,n));switches=0
 for t in range(n):
  prob=np.where(mode==0,gamma*delta,gamma*(1-delta));switch=rng.random(chains)<prob;mode=np.where(switch,1-mode,mode);switches+=int(switch.sum());draws[:,t]=rng.normal(mu*mode,1)
 split=np.concatenate([draws[:,:n//2],draws[:,n//2:]],axis=0);ranks=rankdata(split.ravel());normal=ndtri((ranks-.375)/(len(ranks)+.25)).reshape(split.shape);fold=abs(split-np.median(split));ranksf=rankdata(fold.ravel());normalf=ndtri((ranksf-.375)/(len(ranksf)+.25)).reshape(split.shape)
 report['slow_switch_sampler']={'chains':chains,'draws_per_chain':n,'switch_probability_scale':gamma,'mode_switches':switches,'rank_split_rhat':max(rhat(normal),rhat(normalf)),'sample_mean':float(draws.mean()),'true_mean':delta*mu,'within_mode_iid_standard_error':float(draws.std(ddof=1)/np.sqrt(draws.size)),'probability_all_chains_stay_common':float((1-gamma*delta)**(chains*(n-1))),'initialization':'all chains in common mode; this is a deliberately adverse diagnostic example, not a representative MCMC benchmark'}
 # Numerical checks of the one-dimensional sharp witness and MC interval extension.
 checked=0
 for _ in range(2000):
  q=float(expit(rng.uniform(-25,25)));d=float(10**rng.uniform(-6,0));lo,hi=reverse_interval(q,d);bound=witness(q,d)
  ps=np.linspace(lo,hi,101);assert np.max(kl(ps,q))<=bound+1e-9
  inner=ps[(ps>lo)&(ps<hi)];assert not len(inner) or np.max(kl(q,inner))<=d+1e-8
  checked+=1
 report['tests']={'binary_envelope_cases':checked,'passed':True}
 (OUT/'experiments.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,ax=plt.subplots(1,3,figsize=(7.1,2.3));ns=[a['n'] for a in separation]
 ax[0].loglog(ns,[a['reverse_kl'] for a in separation],label='Reverse KL');ax[0].loglog(ns,[a['true_mc_standard_error'] for a in separation],label='MC standard error');ax[0].loglog(ns,[a['fitted_gain'] for a in separation],label='Apparent gain');ax[0].set_xlabel('Evaluation sample size');ax[0].set_title('(a) True information = 0');ax[0].legend(fontsize=7,frameon=False)
 for scenario,label,axis in [('redundant_evidence','(b) Redundant evidence',ax[1]),('genuine_evidence','(c) Genuine evidence',ax[2])]:
  data=[x for x in sims if x['scenario']==scenario]
  axis.semilogx([x['n'] for x in data],[x['fitted_gain_authorization_rate'] for x in data],marker='o',ms=3,label='Fitted-gain test')
  axis.semilogx([x['n'] for x in data],[x['information_authorization_rate'] for x in data],marker='s',ms=3,label='Corrected information test');axis.set_ylim(-.03,1.03);axis.set_xlabel('Independent audit labels');axis.set_title(label);axis.legend(fontsize=7,frameon=False)
 fig.tight_layout(pad=.5);fig.savefig(P/'diagnostic_separation.pdf',bbox_inches='tight');plt.close(fig)
 print(json.dumps({k:v for k,v in report.items() if k not in ['separation','audit_simulations','gaussian_reparameterization']},indent=2))
 print('SIMULATIONS',sims)
if __name__=='__main__':main()
