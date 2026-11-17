"""Outward interval enclosures for a binary witness at exact binary64 inputs.
This certifies root conversion only, not upstream model budgets or log-score sums.
"""
from fractions import Fraction
from pathlib import Path
import json
import numpy as np
from mpmath import iv
iv.dps=60
R=Path(__file__).resolve().parents[1]
def interval(x):
 f=x if isinstance(x,Fraction) else Fraction.from_float(float(x));return iv.mpf(f.numerator)/iv.mpf(f.denominator)
def ikl(p,q):
 p0=interval(p);q0=interval(q)
 if p==0:return -iv.log(1-q0)
 if p==1:return -iv.log(q0)
 return p0*iv.log(p0/q0)+(1-p0)*iv.log((1-p0)/(1-q0))
def upper_float(v):return float(np.nextafter(float(v.b),np.inf))
def brackets(u,delta,steps=160):
 u=Fraction.from_float(float(u));delta=Fraction.from_float(float(delta))
 assert 0<=u<=1 and delta>=0
 if delta==0:return [u]*4
 if u==0:
  c=1-iv.exp(-interval(delta));lo=Fraction.from_float(max(0,float(np.nextafter(float(c.a),-np.inf))));hi=Fraction.from_float(min(1,float(np.nextafter(float(c.b),np.inf))));return [Fraction(0),Fraction(0),lo,hi]
 if u==1:
  c=iv.exp(-interval(delta));lo=Fraction.from_float(max(0,float(np.nextafter(float(c.a),-np.inf))));hi=Fraction.from_float(min(1,float(np.nextafter(float(c.b),np.inf))));return [lo,hi,Fraction(1),Fraction(1)]
 roots=[]
 for left,right,increasing in [(Fraction(0),u,False),(u,Fraction(1),True)]:
  for _ in range(steps):
   mid=(left+right)/2;v=ikl(u,mid)-interval(delta)
   if bool(v.a>0):
    if increasing:right=mid
    else:left=mid
   elif bool(v.b<0):
    if increasing:left=mid
    else:right=mid
   else:break
  roots.extend([left,right])
 return roots

def witness(u,delta,r=None,steps=160):
 r=Fraction.from_float(float(u if r is None else r));assert 0<r<1
 roots=brackets(u,delta,steps);b=max(upper_float(ikl(roots[0],r)),upper_float(ikl(roots[3],r)))
 return b,[float(z) for z in roots]

def witness_interval(low,high,delta,r):
 assert 0<=low<=high<=1
 a=brackets(low,delta)[0];c=brackets(high,delta)[3];return max(upper_float(ikl(a,r)),upper_float(ikl(c,r)))

def main():
 from experiments import kl,reverse_interval
 rng=np.random.default_rng(29092027);rows=[]
 for _ in range(200):
  u=float(expit(rng.uniform(-15,15)) if False else 1/(1+np.exp(-rng.uniform(-12,12))));delta=float(10**rng.uniform(-7,-.1));r=float(np.clip(u+rng.normal(0,.02),.001,.999));b,br=witness(u,delta,r)
  old=max(float(kl(p,r)) for p in reverse_interval(u,delta));assert b+1e-10>=old
  rows.append({'u':u,'delta':delta,'r':r,'interval_upper':b,'float_estimate':old,'root_enclosures':br})
 out={'cases':len(rows),'precision_decimal_digits':60,'bisection_steps':160,'scope':'Outward interval root and KL evaluation for exact binary64 inputs; recorded float root endpoints are for display only. The witness upper bound is rounded toward positive infinity. Does not validate an upstream KL budget or other floating-point operations.','rows':rows};(R/'results/interval_witness_checks.json').write_text(json.dumps(out,indent=2));print('Checked 200 outward interval witnesses.')
if __name__=='__main__':main()
