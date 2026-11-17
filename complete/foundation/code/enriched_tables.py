from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];P=R/'paper';d=json.loads((R/'results/multiclass_witness.json').read_text())
s=r'''\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{\textbf{Tightness, cost, and numerical optimization are separate comparisons.} Four generated problems per row. Exact is the mean sharp bound in nats. Generic and Pinsker report their mean bound divided by the exact bound (1 is best). Time measures exact enumeration. Gap is the largest exact-minus-SLSQP value over six starts; it is not a statistical error bar.}
\label{tab:categorical}
\begin{tabular}{rrrrrrr}\toprule
Classes & Budget & Exact & Generic & Pinsker & Time & Optimizer\\
$K$ & $\delta$ & (nats) & / exact & / exact & (ms) & gap (nats)\\\midrule
'''
for K in [3,5,8,10]:
 for delta in [.01,.05,.2]:
  rr=[x for x in d['bound_comparisons'] if x['classes']==K and x['budget']==delta]
  vals=[K,delta,np.mean([x['exact_witness'] for x in rr]),np.mean([x['generic_divergence_bound']/x['exact_witness'] for x in rr]),np.mean([x['pinsker_bound']/x['exact_witness'] for x in rr]),1000*np.mean([x['runtime_seconds'] for x in rr]),max(0,max(x['exact_witness']-x['best_slsqp'] for x in rr))]
  s+=f'{K} & {delta:.2f} & {vals[2]:.4f} & {vals[3]:.2f} & {vals[4]:.2f} & {vals[5]:.2f} & {vals[6]:.5f}'+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}';(P/'categorical_table.tex').write_text(s)
rows=json.loads((R/'results/application_validation.json').read_text());short={'MixedShapesRegularTrain':'MixedShapes','SemgHandGenderCh2':'SemgHandGender'}
s=r'''\begin{table}[t]
\centering\small\setlength{\tabcolsep}{2.2pt}
\caption{\textbf{Every later-validation policy, including harm and failed approval.} Classifier identifies the model whose correctness is predicted. Approval/test gives the numbers of disjoint records used to approve and evaluate the correction. Gains use assessment outcomes; positive is better. AUROC measures how well scores rank correct versus incorrect predictions. Approval requires betting evidence $\log E>5.298$. None passes, so every deployed-policy gain is zero. Harm is the percentage of assessment cases given a worse log score. AUROC is undefined for the error-free case.}
\label{tab:application_summary}
\begin{tabular}{llrrrrr}\toprule
Dataset & Classifier & Approval/test & Score gain & AUROC gain & $\log E$ & Harm (\%)\\\midrule
'''
for x in rows:
 au=x['assessment_auc_gain'];at=f'{au:.4f}' if au is not None and np.isfinite(au) else '--'
 s+=f"{short.get(x['dataset'],x['dataset'])} & {x['backbone']} & {x['gate_n']}/{x['assessment_n']} & {x['assessment_nll_gain']:.5f} & {at} & {x['gate_log_e']:.3f} & {100*x['assessment_harm_fraction']:.1f}"+r'\\'+'\n'
s+=r'\midrule'+'\n'
for key,label in [('development','Development (48)'),('validation','Validation (10)')]:
 dd=json.loads((R/'results'/('application_'+key+'.json')).read_text());gain=np.mean([x['assessment_nll_gain'] for x in dd]);au=np.nanmean([float('nan') if x['assessment_auc_gain'] is None else x['assessment_auc_gain'] for x in dd]);harm=np.mean([x['assessment_harm_fraction'] for x in dd]);s+=f'{label} & -- & -- & {gain:.5f} & {au:.4f} & 0 approved & {100*harm:.1f}'+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}';(P/'application_main_table.tex').write_text(s.replace('-0.00000','0.00000').replace('-0.000 &','0.000 &'))
print('Enriched Tables 3 and 4 from all declared comparison problems and all later-validation policies.')
x=json.loads((R/'results/certificate_design.json').read_text());t=r'''\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3.5pt}
\caption{Complete positive-information selection experiment: 500 repetitions per setting. Rates are percentages of positive certificates; lower bounds are mean nats before truncation. Confidence-only selection changes $\beta$ while keeping $B=2$. Every method has zero observed coverage failures in both controls at every setting.}
\begin{tabular}{rrrrrrrrr}\toprule
Draws & \multicolumn{4}{c}{Positive certificates (\%)} & \multicolumn{4}{c}{Mean information lower bound}\\
$S$ & Fixed & Gain & $\beta$ only & Joint & Fixed & Gain & $\beta$ only & Joint\\\midrule
'''
for S in sorted({r['S'] for r in x['summary']}):
 rr=[next(r for r in x['summary'] if r['S']==S and r['genuine'] and r['method']==m) for m in ['fixed','gain_selected','witness_selected','bound_selected']];t+=' & '.join([str(S)]+[f"{100*r['positive_rate']:.1f}" for r in rr]+[f"{r['mean_lower']:.4f}" for r in rr])+r'\\'+'\n'
t+=r'\bottomrule\end{tabular}\end{table}';(P/'design_results_table.tex').write_text(t)
