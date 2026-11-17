"""Post-hoc sensitivity of saved bounds; not new real-world validation."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
x=json.loads((R/'results/bayesian_application_trials.json').read_text());rhos=[0,.01,.025,.05];rows=[]
for mass in [.9,.99,.999]:
 for S in [256,1024,4096]:
  for n in [2000,10000]:
   vals=np.array([r['information_lower'] for r in x if r['genuine'] and r['mass']==mass and r['S']==S and r['n']==n]);assert len(vals)==200
   rows.append(dict(mass=mass,S=S,n=n,median=float(np.median(vals)),positive_percent=[float(100*np.mean(vals>rho)) for rho in rhos]))
(R/'results/witness_sensitivity.json').write_text(json.dumps({'analysis':'Post-hoc deterministic subtraction from saved model-relative bounds. No distribution-shift experiment and no estimated real-world discrepancy.','rho_nats':rhos,'rows':rows},indent=2))
s=r'''\begin{table}[htbp]
\centering\small
\caption{\textbf{Sensitivity to additional witness error.} All 18 positive-control Bayesian settings, 200 saved repetitions each. Entries are percentages with $L_I>\rho$, where $L_I$ is the original untruncated bound. This is a post-hoc robustness calculation, not a distribution-shift experiment. The extra allowance $\rho$ is hypothetical; no real-world discrepancy of this size has been established.}
\label{tab:witness_sensitivity}
\begin{tabular}{rrrrrrrr}
\toprule
Retained & Draws & Audit & Median & \multicolumn{4}{c}{Positive after additional allowance (\%)}\\
mass & $S$ & $n$ & $L_I$ & $\rho=0$ & $.01$ & $.025$ & $.05$\\
\midrule
'''
for r in rows:s+=' & '.join([f"{r['mass']:.3f}",str(r['S']),str(r['n']),f"{r['median']:.3f}"]+[f'{v:.1f}' for v in r['positive_percent']])+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'+'\n';(R/'paper/witness_sensitivity_table.tex').write_text(s)
print('Computed all18 settings from3600 saved positive-control repetitions.')
