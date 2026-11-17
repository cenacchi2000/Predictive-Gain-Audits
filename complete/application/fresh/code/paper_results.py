from pathlib import Path
import json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from table_helpers import table,f,macro,METHODS,esc
from study import NEW,OUT
P=Path(__file__).resolve().parents[1]/'paper'
SHORT={'MixedShapesRegularTrain':'MixedShapes','SemgHandGenderCh2':'SemgGender','CinCECGTorso':'CinCECGTorso'}
def main():
 rows=json.loads((OUT/'assessment_metrics.json').read_text());c=json.loads((OUT/'assessment_comparisons.json').read_text());s=json.loads((OUT/'assessment_summary.json').read_text());sim=json.loads((OUT/'calibration_betting_simulation.json').read_text())
 (P/'abstract_results.tex').write_text('A twelve-dataset development study finds a small average candidate benefit, but no authorized policy among 48 configurations. After revising selection and the test, a frozen evaluation on ten fresh datasets again authorizes none: mean log-loss reduction is .00124 nats, with a descriptive 95\\% interval spanning zero. Controlled examples show that the method can detect strong spectral signals. The real-data results instead delimit the evidence for deployment beyond strong non-spectral references.\n')
 text=r'''Neither the revised gate nor the previous mixture authorizes any of the
ten fresh policies. The deployed policy therefore equals its selected
reference in every dataset. Candidate assessment log-score gain averages
$.00124$ nats, with descriptive dataset-bootstrap 95\% interval
$[-.00004,.00266]$. Five datasets have positive observed gain.
The AUROC change averages $.00187$ across nine defined datasets, with
interval $[-.00086,.00620]$. These results do not establish a dependable
spectral benefit beyond the selected controls.

The correction increases realized loss by more than $10^{-12}$ nats on
34.4\% of fresh assessment examples on average. Two calibration-selected
policies choose no update. Candidate improvements and harms must therefore
be distinguished from deployment: fallback prevents these proposed
changes from being used, but does not certify the absolute quality of the
reference. Table~\ref{tab:fresh} gives the aggregate scores; the supplement
reports every dataset and gate, including near-threshold results.
'''
 table_rows=[]
 for key,label in [('raw','Raw output'),('intercept','Intercept'),('reference','Reference'),('candidate','Candidate'),('policy','Gated policy')]:table_rows.append([label]+[f(macro(rows,k,key)) for k in ['nll','auc','aurc_tie_averaged','fce']])
 text+=table('Fresh assessment means. Lower is better except AUROC. FalseConf is the fraction of errors assigned score at least .9. One error-free assessment dataset is omitted only from AUROC and FalseConf.',['Score','NLL','AUROC','AURC','FalseConf'],table_rows,'tab:fresh').replace('\\centering\\small','\\centering\\small\\setlength{\\tabcolsep}{3pt}')
 text+=r'''\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{evidence_and_power.pdf}
\caption{Left: fresh assessment gain of the selected candidate and gated policy; all policies fall back. Right: constructed power comparison for the same fixed scores, family size, and gate sample size. Improved power in this example does not establish benefit on the fresh datasets.}\label{fig:fresh}\end{figure*}
'''
 (P/'fresh_results.tex').write_text(text)
 (P/'fresh_conclusion.tex').write_text('Spectral corrections require evidence beyond plausibility or improvement over raw confidence. Across development and fresh validation, the tested candidates show small or uncertain average gains and no authorized deployment benefit. Bounded updates and an independent gate make that outcome explicit rather than concealing it in an aggregate performance claim.\n')
 (P/'fresh_simulation_text.tex').write_text('For fixed reference .7 and candidate .8, a separate simulation learns the fixed bet from 500 calibration labels. At correctness prevalence .8 and 1,000 gate labels, authorization occurs in 49.4\\% of 5,000 trials with the revised rule, versus 39.0\\% with the old mixture at the same family size of ten. At 100 labels, the rates are .04\\% and .24\\%, respectively. The revision does not dominate at every sample size.\n')
 # Full selected-policy identities and evidence.
 t='';rr=[]
 for x in c:
  rr.append([SHORT.get(x['dataset'],x['dataset']),x['backbone'],f(x['assessment_accuracy']),f(x['assessment_nll_gain'],5),f(x['assessment_auc_gain'],4),f(x['gate_log_e'],3),f(x['old_gate_log_e'],3)])
 t+=table('Every fresh policy. Both gates use threshold $\\log(10/.05)=5.2983$; neither authorizes any row. Gain and AUROC gap compare the candidate with its selected reference on assessment.',['Dataset','Backbone','Accuracy','Gain','AUROC gap','New log $E$','Old log $E$'],rr)
 rr=[]
 for x in c:rr.append([SHORT.get(x['dataset'],x['dataset']),esc(x['reference']),esc(x['candidate']),f(x['bound'],3),f(x['fraction'],3),x['gate_n']])
 t+=table('Calibration-selected identities, update budgets, fixed betting fractions, and independent gate sizes. No update is a mathematically identical reference correction.',['Dataset','Reference','Candidate','$B$','$\\rho$','Gate $n$'],rr).replace('\\centering\\small','\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}')
 vals=collections.defaultdict(dict)
 for r in rows:vals[r['dataset']][r['method']]=r
 for ds,d in vals.items():
  if 'constant' in d:
   for key,label in METHODS:
    if '/' in key and key not in d:rows.append({**d['constant'],'method':key})
 rr=[]
 for key,label in METHODS:rr.append([label]+[f(macro(rows,k,key)) for k in ['nll','auc','brier','ece','aurc_tie_averaged','augrc','fce']])
 t+=table('Fresh comparison against fixed members of the reference library. These fixed comparators are not selected using assessment.',['Score','NLL','AUROC','Brier','ECE','AURC','AUGRC','FalseConf'],rr)
 rr=[]
 for x in c:rr.append([SHORT.get(x['dataset'],x['dataset']),f(x['assessment_harm_fraction']),f(x['assessment_gain_on_correct'],4),f(x['assessment_gain_on_errors'],4),f(x['assessment_error_upcross_09'],4),f(x['assessment_error_downcross_09'],4)])
 t+=table('Fresh harm and signed changes. Upcross and downcross are fractions of errors crossing the .9 confidence threshold. Undefined error-conditional cells are shown as dashes.',['Dataset','Harm','Correct gain','Error gain','Upcross','Downcross'],rr)
 rr=[]
 for x in c:rr.append([SHORT.get(x['dataset'],x['dataset'])]+[f(q,4) for q in x['assessment_gain_quantiles']])
 t+=table('Fresh realized log-score gain quantiles in nats.',['Dataset','Min.','5\\%','25\\%','50\\%','75\\%','95\\%','Max.'],rr)
 rr=[]
 for key,label in [('intercept','Intercept'),('reference','Reference'),('candidate','Candidate'),('policy','Policy')]:
  for cov in ['0.7','0.8','0.9','0.95']:
   rr.append([label,cov]+[f(macro(rows,'utility_tie_'+cov+'.'+k,key)) for k in ['coverage','risk','precision','recall','oracle_risk']])
 t+=table('Fresh absolute review utility; ties are averaged uniformly. Oracle risk uses the same fixed predictions and correctness counts.',['Score','Nominal','Actual','Risk','Precision','Recall','Oracle risk'],rr)
 counts=[]
 for key in ['nll','auc','fce','aurc_tie_averaged']:
  selected=[r for r in rows if r['method']=='reference'];counts.append(key+': '+str(sum(r[key] is not None for r in selected)))
 t+='Valid dataset counts are '+', '.join(x.replace('_',r'\_') for x in counts)+'. Undefined cells are excluded only from their own endpoint.\n'
 (P/'fresh_supp_tables.tex').write_text(t)
 rr=[]
 for name in NEW:
  m=json.loads((OUT/name/'source.json').read_text());x=next(a for a in c if a['dataset']==name);rr.append([SHORT.get(name,name),m['train'],m['audit_pool'],m['eligible_audit_n'],m['length'],m['classes'],x['calibration_n'],x['gate_n'],x['assessment_n']])
 (P/'fresh_dataset_table.tex').write_text(table('Fresh archive sizes and role counts after input-only filtering. All ten datasets are univariate. MixedShapes abbreviates MixedShapesRegularTrain and SemgGender abbreviates SemgHandGenderCh2.',['Dataset','Train','Pool','Eligible','$T$','Classes','Cal.','Gate','Assess.'],rr).replace('\\centering\\small','\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}'))
 rr=[[x['scenario'].replace('_',' '),x['gate_n'],f(x['expected_gain'],4),f(x['new_rate'],4),f(x['old_rate'],4)] for x in sim]
 (P/'fresh_simulation_table.tex').write_text(table('Calibration-informed betting in 5,000 trials per row. Each uses 500 independent calibration labels, then fresh gate labels. Both rules have family size ten and threshold 200. A zero frequency is not a zero-probability assertion.',['Scenario','Gate $n$','True gain','New rate','Old rate'],rr))
 plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42});fig,(a,b)=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw={'width_ratios':[1.2,1]})
 yy=np.arange(len(c));a.axvline(0,color='#64748b',lw=.7);a.scatter([x['assessment_nll_gain'] for x in c],yy-.12,label='Candidate',s=18,color='#1d4ed8');a.scatter(np.zeros(len(c)),yy+.12,label='Gated policy',s=16,marker='s',color='#b45309');a.set_yticks(yy,[SHORT.get(x['dataset'],x['dataset']) for x in c]);a.invert_yaxis();a.set_xlabel('Assessment log-score gain (nats)');a.set_title('(a) Fresh validation',loc='left',fontsize=8);a.legend(fontsize=7,frameon=False)
 ss=[x for x in sim if x['scenario']=='positive']
 for key,label,col in [('new_rate','Calibration-informed','#1d4ed8'),('old_rate','Previous mixture','#b45309')]:b.plot([x['gate_n'] for x in ss],[x[key] for x in ss],label=label,color=col,marker='o',ms=3)
 b.set_ylim(-.02,1.02);b.set_xlabel('Independent gate labels');b.set_ylabel('Authorization frequency');b.set_title('(b) Constructed gain: 0.0257 nats',loc='left',fontsize=8);b.legend(fontsize=7,frameon=False);b.grid(axis='y',alpha=.15);fig.tight_layout(pad=.6,w_pad=1.2);fig.savefig(P/'evidence_and_power.pdf',bbox_inches='tight');plt.close(fig)
 print('Fresh paper numbers and figure generated.')
if __name__=='__main__':main()
