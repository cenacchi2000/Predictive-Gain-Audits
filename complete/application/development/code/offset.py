"""Regularized logistic correction with the reference log odds as an offset."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler

class OffsetLogistic:
    def __init__(self,C=.1):self.C=C
    def design(self,r,g,fit=False):
        a=logit(np.clip(r,1e-6,1-1e-6))
        # Interactions allow spectral evidence to modify confidence differently.
        z=np.c_[g,g*a[:,None]]
        if fit:self.scaler=StandardScaler().fit(z)
        z=self.scaler.transform(z)
        return a,np.c_[np.ones(len(z)),z]
    def fit(self,r,g,c):
        a,z=self.design(r,g,True);n=len(c)
        def fun(w):
            v=a+z@w
            penalty=np.dot(w,w)/(2*self.C*n)
            return np.mean(np.logaddexp(0,v)-c*v)+penalty,z.T@(expit(v)-c)/n+w/(self.C*n)
        result=minimize(fun,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-11})
        if not result.success:raise RuntimeError(result.message)
        self.coef=result.x;return self
    def predict(self,r,g):
        a,z=self.design(r,g);return np.clip(expit(a+z@self.coef),1e-6,1-1e-6)
