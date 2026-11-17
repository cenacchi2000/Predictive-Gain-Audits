import unittest,itertools
import numpy as np
from study import betting,bounded_update,roles,ordered_gate
from seb import logloss
class GateTests(unittest.TestCase):
 def test_null_expected_wealth(self):
  for fraction in [0,.125,.5]:
   wealth=[]
   for signs in itertools.product([-1,1],repeat=8):
    wealth.append(np.exp(betting(np.array(signs)*.25,.25,fraction)['log_e']))
   self.assertLessEqual(np.mean(wealth),1+1e-12)
 def test_negative_null(self):
  self.assertLessEqual(betting(np.full(100,-.1),.25,.5)['log_e'],1e-12)
 def test_positive(self):
  self.assertTrue(betting(np.full(100,.1),.25,.5)['authorized'])
 def test_no_update(self):
  self.assertAlmostEqual(betting(np.zeros(100),.125,0)['log_e'],0)
 def test_budget(self):
  rng=np.random.default_rng(4);r=rng.uniform(.001,.999,1000);s=rng.uniform(.001,.999,1000);c=rng.integers(0,2,1000)
  for b in [.125,.25,.5,1,2]:
   q=bounded_update(r,s,b);self.assertLessEqual(np.max(abs(logloss(c,r)-logloss(c,q))),b+1e-12)
 def test_roles(self):
  r=roles('example',100);self.assertEqual(sum(r=='calibration'),30);self.assertEqual(sum(r=='gate'),50);self.assertEqual(sum(r=='assessment'),20)
  self.assertEqual(set(ordered_gate('example',r)),set(np.flatnonzero(r=='gate')))
if __name__=='__main__':unittest.main()
