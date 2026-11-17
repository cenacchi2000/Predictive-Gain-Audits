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
 def test_ledger(self):
  v=np.random.default_rng(4).normal(size=(16,20));v[0]=0;np.testing.assert_allclose(shapley(v).sum(0),v[15],atol=1e-12)
 def test_auc_ties(self):
  self.assertEqual(auc(np.array([0,1]),[.5,.5]),.5)
 def test_error_free_metrics(self):
  m=metrics(np.ones(10),np.ones(10)*.8);self.assertIsNone(m['fce']);self.assertTrue(np.isnan(m['auc']));self.assertEqual(m['aurc'],0);self.assertGreater(m['nll'],0)
 def test_utility_identity(self):
  c=np.array([0,1,1,0,1]*20);m=metrics(c,np.linspace(0,1,100))
  for k,v in m.items():
   if k.startswith('utility'):self.assertAlmostEqual(v['coverage']*v['risk']+(1-v['coverage'])*v['precision'],m['prevalence'])
 def test_identical_gate(self):
  c=np.array([0,1]*100);r=np.ones(200)*.7;g=gate(c,r,r,r,.6,.6);self.assertFalse(g['authorized']);self.assertEqual(g['value'],0)
 def test_known_large_gain_authorizes(self):
  c=np.tile([0,1],50000);base=np.full(len(c),.95);candidate=np.where(c==1,.9,.1)
  self.assertTrue(gate(c,base,candidate,base,.5,.5)['authorized'])
 def test_harmful_fce_rejected(self):
  c=np.tile([0,1],50000);ref=np.where(c==1,.3,.4);candidate=np.where(c==1,.95,.92)
  g=gate(c,ref,candidate,ref,.2,.2);self.assertGreater(g['auc_lcb'],.01);self.assertFalse(g['authorized']);self.assertGreater(g['fce_ucb'],0)
 def test_missing_classes_fallback(self):
  c=np.ones(40);r=np.ones(40)*.8;self.assertFalse(gate(c,r,r,r,.5,.5)['authorized'])
if __name__=='__main__':unittest.main()
