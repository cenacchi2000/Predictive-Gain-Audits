"""Independent numerical checks of the main mathematical constructions."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from experiments import kl,witness,reverse_interval
R=Path(__file__).resolve().parents[1]
a=json.loads((R/'results/experiments.json').read_text())
q,err=quad(lambda x:norm.pdf(x)*expit(4*(x-8)),-14,14,epsabs=1e-20,epsrel=1e-10)
p=.99*q+.01*(1-q)
assert abs(q-a['mixture_vi']['reference_probability'])<1e-15
assert abs(float(kl(p,q))-a['mixture_vi']['predictive_forward_kl'])<1e-7
assert witness(q,-np.log(.99))>=kl(p,q)-1e-10
rng=np.random.default_rng(71)
for _ in range(10000):
 w=rng.dirichlet(np.ones(4)).reshape(2,2)
 p0=float(w[:,1].sum());p1=w[:,1]/w.sum(1)
 q0=rng.uniform(.01,.99);q1=rng.uniform(.01,.99,2)
 D=sum(w[g,y]*np.log((q1[g] if y else 1-q1[g])/(q0 if y else 1-q0)) for g in range(2) for y in range(2))
 I=float(w.sum(1)@kl(p1,p0));A0=float(kl(p0,q0));A1=float(w.sum(1)@kl(p1,q1))
 assert abs(D-I-A0+A1)<1e-12
 # Pointwise inequality driving the failure-aware theorem.
 eps=.01;L=np.log(1/eps);a0=A0;b=rng.uniform(0,L)
 assert a0-b<=L*(a0>b)+1e-12
out={'decomposition_and_failure_inequality_cases':10000,'adaptive_quadrature_reference_probability':q,'quad_reported_absolute_error':err,'difference_from_100_node_quadrature':q-a['mixture_vi']['reference_probability'],'mixture_gain_independent_quadrature':float(kl(p,q)),'passed':True}
(R/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
