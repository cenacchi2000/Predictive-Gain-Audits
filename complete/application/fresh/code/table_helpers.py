"""All manuscript numbers derive from released assessment records and simulations."""
from pathlib import Path
import json,collections
import numpy as np
from study import NEW

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';R=ROOT/'results'
SHORT={'ChlorineConcentration':'ChlorineConc.','NonInvasiveFetalECGThorax1':'FetalECGThorax1','UWaveGestureLibraryAll':'UWaveAll','InsectWingbeatSound':'InsectWingbeat'}
METHODS=[('raw','Raw output'),('intercept','Smoothed intercept'),('output/lr1','Output logistic'),('output/gb15','Output boosting'),('time/gb15','Time boosting'),('representation/gb15','Representation boosting'),('agreement/lr1','Agreement logistic'),('atypicality/lr1','Atypicality logistic'),('all_controls/gb15','Combined-control boosting'),('reference','Selected reference'),('candidate','Bounded spectral candidate'),('policy','Authorized policy')]

def f(x,d=3):return '--' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
def esc(s):return str(s).replace('_',r'\_')
def cell(row,key):
    if key.startswith('utility_'):
        parent,child=key.rsplit('.',1)
        return row[parent][child]
    x=row
    for k in key.split('.'):x=x[k]
    return x
def macro(rows,key,method=None):
    groups=collections.defaultdict(list)
    for row in rows:
        if method is not None and row['method']!=method:continue
        value=cell(row,key)
        if value is not None and np.isfinite(value):groups[row['dataset']].append(value)
    return float(np.mean([np.mean(a) for a in groups.values()])) if groups else None
def table(caption,headers,rows,label='',wide=False):
    env='table*' if wide else 'table';out=f'\\begin{{{env}}}[t]\n\\centering\\small\n\\caption{{{caption}}}\n'
    if label:out+=f'\\label{{{label}}}\n'
    out+='\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'
    out+='\n'.join(' & '.join(str(x) for x in row)+r'\\' for row in rows)
    return out+'\n\\bottomrule\\end{tabular}\n'+f'\\end{{{env}}}\n'

def main():
    metrics=json.loads((R/'assessment_metrics.json').read_text());comp=json.loads((R/'assessment_comparisons.json').read_text());summary=json.loads((R/'assessment_summary.json').read_text())
    # Constant fallback represents every learner in a calibration cell with <3 in either class.
    augmented=list(metrics)
    byconfig=collections.defaultdict(dict)
    for row in metrics:byconfig[(row['dataset'],row['backbone'],row['seed'])][row['method']]=row
    for key,values in byconfig.items():
        if 'constant' in values:
            for method,label in METHODS:
                if '/' in method and method not in values:augmented.append({**values['constant'],'method':method})
    metrics=augmented
    mean=summary['assessment_nll_gain']['macro_mean'];lo,hi=summary['assessment_nll_gain']['dataset_bootstrap_95ci'];aucd=summary['assessment_auc_gain']['macro_mean'];auth=summary['authorizations']
    harms=summary['assessment_harm_fraction']['macro_mean'];positive=sum(x['assessment_nll_gain']>0 for x in comp)
    abstext=(f'On twelve public datasets with MiniRocket and Hydra, the bounded candidate changes mean assessment log loss by {f(-mean,4)} nats relative to a calibration-selected non-spectral reference; {auth} of 48 configurations are authorized. Signed per-example changes, dataset-level uncertainty, and exact fallback are reported. Constructed positive and conditional-null controls verify the distinction between fitted value and conditional information. The released prediction records permit direct replay of the claims.\n')
    (P/'abstract_results.tex').write_text(abstext)
    text=(f'The candidate\'s macro mean assessment log-score gain is ${mean:+.4f}$ nats, with a descriptive dataset-bootstrap 95\\% interval $[{lo:+.4f},{hi:+.4f}]$. Its correctness-AUROC change is ${aucd:+.4f}$. Positive assessment log-score gain occurs in {positive}/48 configurations, whereas the independent gate authorizes {auth}/48. Authorization uses the gate alone; these assessment gains do not retroactively change deployment. Table~\\ref{{tab:primary}} distinguishes the bounded candidate from the resulting policy.\n\n')
    if auth==0:text+='Every deployed policy equals its selected reference exactly. This is a failure to establish deployable spectral improvement on this suite, not evidence that all true gains are zero. The policy column must not be described as a successful spectral model.\n\n'
    text+=(f'The candidate worsens realized log score by more than $10^{{-12}}$ nats on {100*harms:.1f}\\% of assessment examples on average. The supplement reports signed-gain quantiles, changes separately on correct and incorrect predictions, crossings of the .9 confidence threshold by errors, and absolute review utility with intercept and oracle context. These comparisons use the same class decisions and label prevalence for every score in a configuration.\n')
    mainrows=[]
    for key,label in [('raw','Raw output'),('output/gb15','Output boosting'),('reference','Reference'),('candidate','Spectral candidate'),('policy','Gated policy')]:
        mainrows.append([label]+[f(macro(metrics,m,key)) for m in ['nll','auc','aurc_tie_averaged','fce']])
    text+=table('Assessment macro means; lower is better except AUROC. AURC averages uniformly within score ties. FalseConf is conditional on errors. The supplement reports metric-specific counts.',['Score','NLL','AUROC','AURC','FalseConf'],mainrows,'tab:primary')
    text+=r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{evidence_and_power.pdf}\caption{Left: assessment log-score gain, averaging four configurations within each dataset. Right: controlled power comparison at a fixed positive gain and the same family-adjusted error budget. The simulation does not establish spectral benefit in the real datasets.}\label{fig:evidence}\end{figure*}'+'\n'
    (P/'assessment_results.tex').write_text(text)
    supp=''
    fullrows=[]
    for key,label in METHODS:fullrows.append([label]+[f(macro(metrics,m,key)) for m in ['nll','auc','brier','ece','aurc_tie_averaged','augrc','fce']])
    supp+=table('Complete aggregate score comparison. Fixed logistic controls use $C=1$; fixed boosting controls use 15 leaves. Selection of the reference uses the full calibration library.',['Score','NLL','AUROC','Brier','ECE','AURC','AUGRC','FalseConf'],fullrows)
    dsrows=[]
    for name in NEW:
        cc=[a for a in comp if a['dataset']==name]; mm=[a for a in metrics if a['dataset']==name]
        aucs=[a['assessment_auc_gain'] for a in cc if a['assessment_auc_gain'] is not None]
        dsrows.append([SHORT.get(name,name),f(np.mean([a['assessment_accuracy'] for a in cc])),f(macro(mm,'nll','reference')),f(np.mean([a['assessment_nll_gain'] for a in cc]),4),f(np.mean(aucs) if aucs else None,4),f(np.mean([a['assessment_harm_fraction'] for a in cc])),str(sum(a['authorized'] for a in cc))+'/4'])
    supp+=table('Dataset-level results. Gain is reference loss minus candidate loss; positive is better. Harm is the fraction with realized gain below $-10^{-12}$ nats.',['Dataset','Accuracy','Ref. NLL','Gain','AUROC gap','Harm','Authorized'],dsrows)
    utility=[]
    for key,label in [('intercept','Intercept'),('reference','Reference'),('candidate','Candidate'),('policy','Policy')]:
        for cov in ['0.7','0.8','0.9','0.95']:
            prefix='utility_tie_'+cov+'.';utility.append([label,cov]+[f(macro(metrics,prefix+m,key)) for m in ['coverage','risk','precision','recall','oracle_risk']])
    supp+=table('Absolute review utility with uniform random tie breaking. Oracle risk uses the same class predictions and outcome counts. Nominal and actual coverage differ because retained counts are rounded down.',['Score','Nominal','Actual','Risk','Precision','Recall','Oracle risk'],utility)
    (P/'supp_assessment_tables.tex').write_text(supp)
    print('Wrote main and supplementary assessment tables.')

if __name__=='__main__':main()
