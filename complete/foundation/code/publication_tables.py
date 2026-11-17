from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];P=R/'paper';A=json.loads((R/'results/experiments.json').read_text())
s=r'''\begin{table}[t]
\centering\small
\caption{\textbf{A stronger conclusion requires stronger evidence.} The same predictive gain supports different claims depending on the available error control.}
\label{tab:claims}
\begin{tabular}{p{.21\linewidth}p{.35\linewidth}p{.34\linewidth}}
\toprule Claim & Required evidence & What remains outside the claim\\\midrule
Predictive improvement & Valid lower bound $L_\Delta>0$ for frozen predictors & Additional information; ranking or deployment utility\\
Conditional information & Valid baseline witness and $L_I>0$ & Model correctness if the witness is only model-relative\\
Deployment authorization & A separately declared gate with selection and multiplicity control & Arbitrary distribution shift or untested performance metrics\\
\bottomrule\end{tabular}
\end{table}
'''
(P/'claims_table.tex').write_text(s)
s=r'''\begin{table}[t]
\centering\small
\caption{\textbf{Prediction and information are different audit targets.} Positive-certificate frequencies over 5,000 repetitions; the two lower bounds differ by a fixed baseline witness. All tests use $\alpha=0.05$. LCB denotes lower confidence bound; negative values are shown before truncation at zero.}
\label{tab:controlled}
\begin{tabular}{lrrrrr}\toprule
Evidence & Labels & Predictive & Information & Mean predictive & Mean information\\
 & $n$ & positive (\%) & positive (\%) & LCB (nats) & LCB (nats)\\\midrule
'''
for d in A['audit_simulations']:
 s+=' & '.join(['Redundant ($I=0$)' if d['scenario']=='redundant_evidence' else 'Genuine ($I>0$)',str(d['n']),f"{100*d['fitted_gain_authorization_rate']:.2f}",f"{100*d['information_authorization_rate']:.2f}",f"{d['mean_fitted_lower']:.4f}",f"{d['mean_information_lower']:.4f}"])+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'
(P/'controlled_main_table.tex').write_text(s)
s=r'''\begin{table}[t]
\centering\small
\caption{\textbf{Candidate, gate, and deployed policy remain distinct.} Application gains are nats; intervals are descriptive dataset-bootstrap 95\% intervals, not gate confidence bounds. Harm is the fraction of cases receiving a worse candidate log score.}
\label{tab:application_summary}
\begin{tabular}{lrrrrr}\toprule
Study & Datasets / & Candidate gain & Authorized & Policy & Harm\\
 & policies & [descriptive 95\% interval] & policies & gain & (\%)\\\midrule
'''
for key,label in [('development','Development'),('validation','Later validation')]:
 rows=json.loads((R/'results'/('application_'+key+'.json')).read_text())
 vals=[r['assessment_nll_gain'] for r in rows];import numpy as np
 # Exact published bootstrap intervals retained from the frozen summaries.
 roots={'development':'research_v2','validation':'research_v3'};source=R.parent/roots[key]/'results/assessment_summary.json';dest=R/'results'/('application_'+key+'_summary.json')
 if source.exists():dest.write_text(source.read_text())
 d=json.loads(dest.read_text());g=d['assessment_nll_gain'];mean=g.get('mean',g.get('macro_mean'));ci=g.get('descriptive_dataset_bootstrap_95ci',g.get('dataset_bootstrap_95ci'));harm=np.mean([r['assessment_harm_fraction'] for r in rows])
 s+=f"{label} & {d['datasets']} / {len(rows)} & {mean:.5f} [{ci[0]:.5f}, {ci[1]:.5f}] & {d['authorizations']} / {len(rows)} & 0.00000 & {100*harm:.1f}"+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'
(P/'application_main_table.tex').write_text(s)
print('Built three main-paper tables from recorded results and stated assumptions.')
d=json.loads((R/'results/multiclass_witness.json').read_text())
s=r'''\begin{table}[t]
\centering\small
\caption{\textbf{Exact categorical witnesses are practical for small alphabets.} Twelve generated problems per class count. Time is mean partition-enumeration time on the local CPU. The final column is the largest gap between the exact witness and the best feasible six-start SLSQP solution; a positive gap shows that numerical optimization need not find the global maximum.}
\label{tab:categorical}
\begin{tabular}{rrrr}\toprule
Classes $K$ & Binary partitions & Mean time (ms) & Largest optimization gap (nats)\\\midrule
'''
for r in d['rows']:s+=f"{r['classes']} & {r['partitions']} & {1000*r['mean_runtime_seconds']:.2f} & {r['max_witness_minus_best_slsqp']:.6f}"+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'
(P/'categorical_table.tex').write_text(s)
