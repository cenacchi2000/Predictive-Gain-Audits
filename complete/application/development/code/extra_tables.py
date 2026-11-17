"""Generate diagnostic tables from saved records without changing any model."""
from pathlib import Path
import json
import numpy as np
from tables import ROOT,P,R,NEW,SHORT,f,table,macro,esc
from seb import logloss

def main():
    sim=json.loads((R/'simulations.json').read_text());cn=json.loads((R/'conditional_null.json').read_text());pos=json.loads((R/'positive_control.json').read_text())
    text=table('Authorization frequencies in 5,000 independent simulations per row; family size 48. Zero observed events do not imply zero population probability.',['Scenario','True gain','Gate $n$','Betting','Bernstein','Hoeffding'],[[s['scenario'].replace('_',' '),f(s['true_gain'],4),s['n'],f(s['betting_rate'],4),f(s['bernstein_rate'],4),f(s['hoeffding_rate'],4)] for s in sim if s['family']==48])
    text+=table('The conditional-information null holds in every row. Authorization of positive fitted gain is valid for the stated score-improvement target.',['Output reference','Gate $n$','True gain','Authorization'],[['Misspecified' if 'misspecified' in s['scenario'] else 'Correct',s['n'],f(s['true_fitted_gain'],4),f(s['authorization_rate'],3)] for s in cn])
    records=pos['records'];rows=[]
    for n in sorted(set(x['n'] for x in records)):
        a=[x for x in records if x['n']==n];rows.append([n,len(a),f(np.mean([x['exact_fitted_gain'] for x in a]),4),f(np.mean([x['authorized'] for x in a]))])
    text+=table('Constructed spectral positive control. Exact fitted gain averages the known four-state population; authorization uses independent gate labels.',['Gate $n$','Fitting trials','Mean exact gain','Authorization'],rows)
    (P/'supp_simulation_tables.tex').write_text(text)
    comp=json.loads((R/'assessment_comparisons.json').read_text());metrics=json.loads((R/'assessment_metrics.json').read_text());text=''
    rows=[]
    for name in NEW:
        a=[x for x in comp if x['dataset']==name]
        keys=['assessment_gain_on_correct','assessment_gain_on_errors','assessment_error_upcross_09','assessment_error_downcross_09']
        rows.append([SHORT.get(name,name)]+[f(np.mean([x[k] for x in a if x[k] is not None]),4) for k in keys])
    text+=table('Signed gain by outcome, and fractions of errors crossing the .9 confidence threshold upward or downward. Positive gain is better. Each entry averages the four configurations.',['Dataset','Correct gain','Error gain','Upcross','Downcross'],rows)
    rows=[]
    for name in NEW:
        a=[x for x in comp if x['dataset']==name];qq=np.mean([x['assessment_gain_quantiles'] for x in a],axis=0)
        rows.append([SHORT.get(name,name)]+[f(x,4) for x in qq])
    text+=table('Mean within-configuration quantiles of realized log-score gain (nats). These are averages of four quantiles, not pooled quantiles.',['Dataset','Min.','5\%','25\%','50\%','75\%','95\%','Max.'],rows)
    examples=[]
    for x in comp:
        a=np.load(R/x['dataset']/f"{x['backbone']}_{x['seed']}"/'records.npz');sc=dict(zip(a['score_names'],a['scores']));ids=np.flatnonzero(a['roles']=='assessment');c=a['correct'][ids];d=logloss(c,sc['reference'][ids])-logloss(c,sc['candidate'][ids]);j=int(np.argmin(d));examples.append((float(d[j]),x, int(ids[j]),int(c[j]),float(sc['reference'][ids[j]]),float(sc['candidate'][ids[j]])))
    examples.sort(key=lambda a:a[0]);rows=[]
    for gain,x,i,c,r,q in examples[:3]:rows.append([SHORT.get(x['dataset'],x['dataset']),x['backbone'],x['seed'],i,c,f(r,4),f(q,4),f(gain,4)])
    text+=table('Three adverse examples selected after assessment as the largest per-configuration losses. IDs are zero-based archive rows. Selection makes these illustrative cases, not estimates of typical harm.',['Dataset','Backbone','Seed','ID','Correct','Ref.','Cand.','Gain'],rows)
    ref=[x for x in metrics if x['method']=='reference'];n_auc=sum(x['auc'] is not None for x in ref);n_fce=sum(x['fce'] is not None for x in ref)
    text+=f'All 48 configurations contribute to probability losses and AURC; {n_auc} contribute to AUROC and {n_fce} to error-conditional false confidence. Undefined metrics are omitted only from their own summaries.\n'
    text+=r'''\subsection{Every authorization decision}
All 48 gates use $\log(M/\alpha)=6.8669$, with $\alpha=.05$.
The table identifies the fitted reference and candidate selected using calibration;
$B$ is the update budget and $n$ is the gate size. Every decision is fallback.
Model names use lr for logistic regression and gb for gradient boosting.
{\scriptsize
\begin{longtable}{llr p{1.15in}p{1.3in}rrr}
\caption{Complete frozen model identities and gate evidence.}\\
\toprule Dataset & Backbone & Seed & Reference & Candidate & $B$ & $n$ & $\log E$\\\midrule
\endfirsthead
\toprule Dataset & Backbone & Seed & Reference & Candidate & $B$ & $n$ & $\log E$\\\midrule
\endhead
'''
    for x in comp:
        g=json.loads((R/x['dataset']/f"{x['backbone']}_{x['seed']}"/'gate.json').read_text());text+=' & '.join([SHORT.get(x['dataset'],x['dataset']),x['backbone'],str(x['seed']),esc(x['reference']),esc(x['candidate']),f(x['bound'],2),str(g['n']),f(g['log_e'])])+r'\\'+'\n'
    text+=r'\bottomrule\end{longtable}}'+'\n'
    (P/'supp_diagnostics.tex').write_text(text)
    if (R/'stress_metrics.json').exists():
        sm=json.loads((R/'stress_metrics.json').read_text())
        if len(sm)==216:
            rows=[]
            for kind,label in [('dominant_band','Dominant frequency bins'),('random_band','Random frequency band'),('time_block','Time block')]:
                subset=[s for s in sm if s['stress']==kind]
                for method in ['reference','candidate','policy']:
                    rows.append([label,method]+[f(macro(subset,k,method),4) for k in ['accuracy','nll','auc','aurc_tie_averaged','fce']])
            text=r'''\subsection{Targeted stress after fitting}
We remove the 20\% most energetic positive-frequency bins (pooled over
channels), a randomly located contiguous band covering 20\% of positive
frequencies, or a contiguous time block covering 20\% of the series.
All clean models and gate decisions remain frozen. Inputs are normalized
again after intervention. Each dataset uses both backbones at seed 17;
interventions are applied only to assessment rows. Archive labels are
retained for diagnostic scoring, without assuming these interventions
preserve the true semantic label. These results therefore cannot establish
robustness or transport the clean-population certificate
\citep{ovadia2019trust}. The filename \texttt{dominant\_band} denotes
selected dominant bins, not a contiguous band.
'''
            text+=table('Stress diagnostics: dataset macro means over two backbones. The policy remains the clean reference because all clean gates failed.',['Intervention','Score','Accuracy','NLL','AUROC','AURC','FalseConf'],rows)
            (P/'stress_results.tex').write_text(text)
    print('Diagnostic tables generated.')

if __name__=='__main__':main()
