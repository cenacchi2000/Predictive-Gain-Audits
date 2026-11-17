from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];P=R/'paper';d=json.loads((R/'results/categorical_scalable.json').read_text())
s=r'''\begin{table}[t]\centering\small
\caption{\textbf{Exact categorical conversion scales beyond enumeration.} Thirty problems per class count, across budgets $.001,.01,.1$. Each solve scans $q$ and computes one root. The extremizer attains the returned value and reverse-KL budget. The new formula agrees with all 48 earlier enumeration cases to $1.1\times10^{-15}$. Timings describe this implementation, not an optimized solver competition.}
\label{tab:categorical}
\begin{tabular}{rrrrr}\toprule
Classes & Mean time (ms) & 95th percentile (ms) & Max budget error & Max attainment error\\\midrule
'''
for k in [10,100,1000,10000,100000]:
 rr=[x for x in d['rows'] if x['K']==k];times=np.array([x['runtime_seconds'] for x in rr])*1000
 s+=f"{k:,} & {times.mean():.3f} & {np.quantile(times,.95):.3f} & {max(abs(x['reverse_kl']-x['delta']) for x in rr):.1e} & {max(abs(x['forward_kl']-x['witness']) for x in rr):.1e}"+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}';(P/'categorical_scaling_table.tex').write_text(s)
d=json.loads((R/'results/representation_audit.json').read_text())
s=r'''\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}
\caption{\textbf{A learned representation can improve prediction without adding information.} A trained teacher generates labels on continuous 64-dimensional inputs. Both probes are fitted independently of the audit. Rates pool 50 repetitions for each of three fitted-probe seeds (150 per row). Prediction and corrected empirical Bernstein (EB) use fixed-sample bounds; betting uses the corrected score. HRT uses the exact conditional feature sampler. Its rejection tests independence; our bound also quantifies information.}
\label{tab:representation}
\begin{tabular}{lrrrrrr}\toprule
Added input & Audit & Mean gain & \multicolumn{4}{c}{Positive conclusion / rejection (\%)}\\
 & cases & (nats) & Prediction & Corrected EB & Betting & HRT\\\midrule
'''
for r in d['rows']:
 label='Deterministic layer' if not r['genuine'] else 'Layer + new signal'
 s+=f"{label} & {r['n']} & {r['mean_gain']:.4f} & {100*r['predictive_positive']:.1f} & {100*r['eb_positive']:.1f} & {100*r['betting_positive']:.1f} & {100*r['hrt_rejection']:.1f}"+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}';(P/'representation_table.tex').write_text(s)
# Fresh selection table, retaining the exploratory table separately.
p=P/'design_results_table.tex';fresh=json.loads((R/'results/certificate_design_fresh.json').read_text());t=p.read_text();a=t.index('16 &');b=t.index('\\bottomrule',a);body=''
for S in sorted({x['S'] for x in fresh['summary']}):
 rr=[next(x for x in fresh['summary'] if x['S']==S and x['genuine'] and x['method']==m) for m in ['fixed','gain_selected','witness_selected','bound_selected']];body+=' & '.join([str(S)]+[f"{100*x['positive_rate']:.1f}" for x in rr]+[f"{x['mean_lower']:.4f}" for x in rr])+r'\\'+'\n'
t=t[:a]+body+t[b:];t=t.replace('Complete positive-information selection experiment','Fresh fitted-candidate selection experiment');(P/'design_fresh_table.tex').write_text(t)
print('Built scaling, representation, and fresh-selection tables.')
