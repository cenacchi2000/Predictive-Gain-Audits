import unittest
import numpy as np
from scipy.special import expit, logit
from study import bounded_update, betting
from seb import logloss
from offset import OffsetLogistic

class UpdateTests(unittest.TestCase):
    def test_logscore_budget_and_fallback(self):
        rng=np.random.default_rng(61)
        reference=expit(rng.uniform(-13,13,10000));candidate=expit(rng.uniform(-13,13,10000))
        for b in [.25,.5,1,2]:
            q=bounded_update(reference,candidate,b)
            self.assertLessEqual(np.max(np.abs(logit(q)-logit(reference))),b+1e-9)
            for c in [np.zeros(len(q)),np.ones(len(q))]:
                d=logloss(c,reference)-logloss(c,q)
                self.assertLessEqual(np.max(np.abs(d)),b+1e-10)
            zero=betting(np.zeros(100),b)
            self.assertFalse(zero['authorized'])
            self.assertAlmostEqual(zero['log_e'],0)

    def test_simple_null_expectation(self):
        # Exact enumeration, rather than an approximate Monte Carlo type-I check.
        import itertools
        for p in [.3,.5]:
            total=0
            for bits in itertools.product([0,1],repeat=8):
                x=np.array(bits); d=2*x-1
                weight=p**x.sum()*(1-p)**(len(x)-x.sum())
                total+=weight*np.exp(betting(d,1)['log_e'])
            self.assertLessEqual(total,1+1e-12)

    def test_positive_signal_is_detected(self):
        self.assertTrue(betting(np.full(100,.2),.25)['authorized'])
        self.assertFalse(betting(np.full(100,-.2),.25)['authorized'])

    def test_offset_learns_a_real_signal(self):
        rng=np.random.default_rng(78);g=rng.normal(size=(2000,2));r=np.full(2000,.5)
        c=rng.binomial(1,expit(2*g[:,0]));m=OffsetLogistic(1).fit(r[:1000],g[:1000],c[:1000])
        q=m.predict(r[1000:],g[1000:])
        self.assertLess(logloss(c[1000:],q).mean(),logloss(c[1000:],r[1000:]).mean()-.1)

if __name__=='__main__':unittest.main()
