import unittest
import numpy as np
from seb import *
class TestSEB(unittest.TestCase):
 def test_invariance(self):
  rng=np.random.default_rng(7)
  for channels in (1,3):
   x=rng.normal(size=(8,channels,97));g=bundle(x)
   for a in (.001,.1,2,1000):
    for shift in (1,17,31):np.testing.assert_allclose(g,bundle(a*np.roll(x,shift,-1)),atol=1e-10)
 def test_zero(self):
  g=bundle(np.ones((3,2,32)));self.assertTrue(np.isfinite(g).all());self.assertTrue((g[:,-1]==1).all())
 def test_short(self):
  for n in (2,3,8):self.assertTrue(np.isfinite(bundle(np.arange(n)[None,:])).all())
 def test_channel_permutation(self):
  x=np.random.default_rng(3).normal(size=(3,4,55));np.testing.assert_allclose(bundle(x),bundle(x[:,::-1]),atol=1e-12)
 def test_auc_ties(self):
  self.assertEqual(auc(np.array([0,1]),[.5,.5]),.5)
 def test_error_free_metrics(self):
  m=metrics(np.ones(10),np.ones(10)*.8);self.assertIsNone(m['fce']);self.assertTrue(np.isnan(m['auc']));self.assertEqual(m['aurc'],0);self.assertGreater(m['nll'],0)
 def test_tie_averaged_risk(self):
  c=np.array([0,0,1,1]);m=metrics(c,np.ones(4)*.7)
  self.assertAlmostEqual(m['aurc_tie_averaged'],.5)
  self.assertAlmostEqual(m['augrc'],.25)
  self.assertAlmostEqual(m['utility_tie_0.8']['risk'],.5)
 def test_augrc_identity(self):
  c=np.array([0,1,1,0,1]*20);r=np.repeat(np.arange(20),5);m=metrics(c,r/20)
  a=m['accuracy'];self.assertAlmostEqual(m['augrc'],(1-m['auc'])*a*(1-a)+.5*(1-a)**2)
 def test_utility_identity(self):
  c=np.array([0,1,1,0,1]*20);m=metrics(c,np.linspace(0,1,100))
  for k,v in m.items():
   if k.startswith('utility'):self.assertAlmostEqual(v['coverage']*v['risk']+(1-v['coverage'])*v['precision'],m['prevalence'])
if __name__=='__main__':unittest.main()
