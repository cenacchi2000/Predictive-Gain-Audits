from pathlib import Path
import json,hashlib,numpy as np
from scipy.integrate import quad
from scipy.special import expit,logsumexp
from scipy.stats import norm,beta
from sklearn.datasets import load_breast_cancer
R=Path(__file__).resolve().parents[1];d=json.loads((R/'results/continuous_witness.json').read_text());ds=load_breast_cancer();tr=np.array(d['training_indices']);X=(ds.data[:,[1,0]]-d['scaling_mean'])/d['scaling_sd'];off=d['fixed_intercept']+d['fixed_texture_coefficient']*X[tr,0];x=X[tr,1];y=(ds.target[tr]==0);m=d['Q_mode'];sd=d['Q_sd']
def V(t):
 eta=off+t*x;return .125*t*t+np.sum(np.logaddexp(0,eta)-y*eta)
vm=V(m);Z,err=quad(lambda t:np.exp(vm-V(t)),m-20,m+20,epsabs=1e-11,epsrel=1e-11);logZ=np.log(Z)-vm;ev,eerr=quad(lambda z:V(m+sd*z)*norm.pdf(z),-12,12,epsabs=1e-10,epsrel=1e-11);trueKL=ev-d['entropyQ']+logZ
assert ev<=d['EQ_V_upper']+1e-9;assert logZ<=d['logZ_upper']+1e-9;assert trueKL<=d['delta_upper']+1e-9
rows=json.loads((R/'results/natural_sensitivity.json').read_text())['rows'];recs=json.loads((R/'results/natural_sensitivity_records.json').read_text());scoreerr=0
for row,r in zip(rows,recs):
 sets=[set(r[k]) for k in ['train','witness','audit']];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i));y=np.array(r['y']);q0=np.array(r['q0']);q1=np.array(r['q1']);D=np.where(y,np.log(q1/q0),np.log((1-q1)/(1-q0)));assert np.isfinite(D).all();scoreerr=max(scoreerr,float(np.max(abs(D-r['D']))));assert abs(D.mean()-row['gain'])<1e-12
 assert all(np.isfinite(row[k]) for k in ['family81_lower','original_level_lower'])
assert len(rows)==81
for key,script in [('natural_sensitivity','natural_sensitivity.py'),('continuous','continuous_witness.py'),('matched','matched_categorical.py')]:
 p=json.loads((R/'results'/f'{key}_protocol.json').read_text());assert hashlib.sha256((R/'code'/script).read_bytes()).hexdigest()==p['code_sha256']
out={'natural_analyses':81,'natural_disjoint_splits':True,'natural_score_replay_max_error':scoreerr,'all_natural_scores_and_bounds_finite':True,'posterior_posthoc_check_only':{'logZ_reference':logZ,'logZ_upper':d['logZ_upper'],'EQ_V_reference':ev,'EQ_V_upper':d['EQ_V_upper'],'KL_reference':trueKL,'KL_upper':d['delta_upper'],'quadrature_error':err},'note':'Reference integration is only an independent post-execution numerical check, never an input to the witness or audit. Real-arithmetic guarantees are not full floating-point interval certification.'}
(R/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
# Paired comparisons and matched-component accounting.
c=json.loads((R/'results/matched_categorical.json').read_text());assert len(c['trials'])==4800 and len(c['rows'])==48
from matched_categorical import bounds
replay=0.
for t in c['trials']:
 for k,v in t['lower'].items():
  if k in t['bounds']:assert abs(v-(t['gain_lower']-t['bounds'][k]))<1e-12
 assert t['lower']['CP_full']>=t['lower']['CP_exact_envelope']-1e-12
 if t['counts'] is not None and t['rep']%20==0:
  vals,active=bounds(np.array(t['counts']),'CP');assert active==t['active']['CP'];replay=max(replay,max(abs(vals[k]-t['bounds']['CP_'+k]) for k in vals))
assert replay<1e-12
out.update({'matched_trials':4800,'matched_configurations':48,'matched_component_replay_max_error':replay,'same_CP_added_positive':sum(x['CP_full_vs_envelope_gained'] for x in c['rows']),'known_vector_added_positive':sum(x['known_exact_vs_generic_gained'] for x in c['rows']),'information_bound_violations_in_controlled_grid':sum(v['violations'] for x in c['rows'] for v in x['methods'].values())})
(R/'results/verification.json').write_text(json.dumps(out,indent=2));print('Matched checks passed',out['same_CP_added_positive'],out['known_vector_added_positive'])
