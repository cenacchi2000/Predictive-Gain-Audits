"""Calibration selects the information audit, before independent audit labels.
Constructed heterogeneous problem; no claim of a real-data inference benchmark.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit,logit,xlogy
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1]
SEED=27092026;SS=[16,32,64,128,256,512,1024,2048];BETAS=np.array([.0001,.001,.01,.03,.1]);BS=np.array([.5,1.,2.,3.]);TRIALS=500;NCAL=300;N=2000;EPS=.01;L=np.log(1/EPS);ALPHA=.05
U=np.array([.5,.05]);P0=np.array([.5,.1]);DELTA=np.array([.005,float(kl(.05,.1))+.005]);EFFECT=np.array([.35,.08])
def entropy(p):return -xlogy(p,p)-xlogy(1-p,1-p)
TRUE_I=float(np.mean(entropy(P0)-.5*(entropy(P0+EFFECT)+entropy(P0-EFFECT))))
def lookup(S):
 out=np.empty((len(BETAS),2,S+1));reports=np.clip(np.arange(S+1)/S,EPS,1-EPS)
 for ib,beta in enumerate(BETAS):
  rad=np.sqrt(np.log(2/beta)/(2*S))
  for h in range(2):
   for k in range(S+1):
    low=reverse_interval(max(0,k/S-rad),DELTA[h])[0];high=reverse_interval(min(1,k/S+rad),DELTA[h])[1]
    out[ib,h,k]=max(0,float(kl(low,reports[k])),float(kl(high,reports[k])))
 return reports,out

def draw(rng,n,S,genuine,reports,look):
 h=rng.integers(0,2,n);g=rng.integers(0,2,n);truth=P0[h]+(2*g-1)*EFFECT[h] if genuine else P0[h]
 y=rng.random(n)<truth;k=rng.binomial(S,U[h]);r=reports[k];b=look[:,h,k]
 z=logit(r);target=logit(truth);q=expit(z[None,:]+np.clip(target-z,-BS[:,None],BS[:,None]))
 D=np.where(y[None,:],np.log(q/r),np.log((1-q)/(1-r)))
 assert np.all(np.abs(D)<=BS[:,None]+1e-10)
 T=D[:,None,:]-b[None,:,:]
 return D,T,b

def lower(T,audit_n):
 return T.mean(-1)-np.sqrt(2*T.var(-1,ddof=1)*np.log(2/ALPHA)/audit_n)-7*(2*BS[:,None]+L)*np.log(2/ALPHA)/(3*(audit_n-1))-BETAS[None,:]*L

def main():
 rng=np.random.default_rng(SEED);raw=[];summ=[]
 for S in SS:
  reports,look=lookup(S)
  for genuine in [False,True]:
   records={k:[] for k in ['fixed','gain_selected','witness_selected','bound_selected']}
   for trial in range(TRIALS):
    dcal,tcal,bcal=draw(rng,NCAL,S,genuine,reports,look)
    # Same paired samples for all selectors. No audit label enters selection.
    selected={'fixed':(2,2),'gain_selected':(int(np.argmax(dcal.mean(-1))),2),'witness_selected':(2,int(np.argmax(lower(tcal,N)[2]))),'bound_selected':np.unravel_index(np.argmax(lower(tcal,N)),(len(BS),len(BETAS)))}
    D,T,b=draw(rng,N,S,genuine,reports,look);low=lower(T,N)
    for method,(i,j) in selected.items():
     record={'S':S,'genuine':genuine,'trial':trial,'method':method,'lower':float(low[i,j]),'gain':float(D[i].mean()),'witness':float(b[j].mean()),'beta':float(BETAS[j]),'B':float(BS[i]),'true_information':TRUE_I if genuine else 0.}
     records[method].append(record);raw.append(record)
   for method,rr in records.items():
    l=np.array([x['lower'] for x in rr]);summ.append({'S':S,'genuine':genuine,'method':method,'trials':TRIALS,'positive_rate':float(np.mean(l>0)),'coverage_failure_rate':float(np.mean(l>(TRUE_I if genuine else 0))),'mean_lower':float(l.mean()),'lower_quantiles':np.quantile(l,[.1,.25,.5,.75,.9]).tolist(),'mean_gain':float(np.mean([x['gain'] for x in rr])),'mean_witness':float(np.mean([x['witness'] for x in rr])),'mean_beta':float(np.mean([x['beta'] for x in rr])),'mean_B':float(np.mean([x['B'] for x in rr]))})
   print('Finished',S,genuine,flush=True)
 out={'seed':SEED,'calibration_cases':NCAL,'audit_cases':N,'trials':TRIALS,'epsilon':EPS,'alpha':ALPHA,'beta_grid':BETAS.tolist(),'B_grid':BS.tolist(),'approximate_predictive':U.tolist(),'true_baseline_predictive':P0.tolist(),'reverse_budgets':DELTA.tolist(),'true_information_positive':TRUE_I,'summary':summ,'note':'Controlled heterogeneous distribution. Candidate target probabilities are specified by the generator, not fitted from a real dataset. Calibration and audit samples are independent. Selection maximizes a calibration proxy; only the audit bound is a confidence statement.'}
 (R/'results/certificate_design.json').write_text(json.dumps(out,indent=2));(R/'results/certificate_design_trials.json').write_text(json.dumps(raw))
 print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
