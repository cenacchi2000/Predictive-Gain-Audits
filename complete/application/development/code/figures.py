from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper'
def main():
    summary=json.loads((ROOT/'results'/'assessment_summary.json').read_text())
    sims=json.loads((ROOT/'results'/'simulations.json').read_text())
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,(a,b)=plt.subplots(1,2,figsize=(6.6,3.1),gridspec_kw={'width_ratios':[1.2,1]})
    data=summary['assessment_nll_gain']['by_dataset'];policy=summary['assessment_policy_nll_gain']['by_dataset']
    labels={'ElectricDevices':'ElectricDevices','StarLightCurves':'StarLightCurves','Crop':'Crop','ChlorineConcentration':'ChlorineConc.','NonInvasiveFetalECGThorax1':'FetalECGThorax1','Phoneme':'Phoneme','UWaveGestureLibraryAll':'UWaveAll','InsectWingbeatSound':'InsectWingbeat','FaceAll':'FaceAll','TwoPatterns':'TwoPatterns','PenDigits':'PenDigits','LSST':'LSST'}
    names=list(data);y=np.arange(len(names))
    a.axvline(0,color='#64748b',linewidth=.7)
    a.scatter(list(data.values()),y-.12,s=18,color='#1d4ed8',label='Candidate')
    a.scatter([policy[k] for k in names],y+.12,s=18,marker='s',color='#b45309',label='Gated policy')
    a.set_yticks(y,[labels[n] for n in names]);a.invert_yaxis();a.set_xlabel('Log-score gain over reference (nats)');a.set_title('(a) Assessment: four fits per dataset',loc='left',fontsize=8);a.legend(frameon=False,fontsize=7,loc='best')
    for field,color,label in [('betting_rate','#1d4ed8','Betting mixture'),('bernstein_rate','#b45309','Empirical Bernstein'),('hoeffding_rate','#475569','Range-aware Hoeffding')]:
        vals=[s for s in sims if s['scenario']=='positive' and s['family']==48]
        b.plot([v['n'] for v in vals],[v[field] for v in vals],marker='o',markersize=3,color=color,label=label)
    b.set_ylim(-.02,1.02);b.set_xlabel('Independent gate labels');b.set_ylabel('Authorization frequency');b.set_title('(b) Fixed positive gain: 0.0257 nats',loc='left',fontsize=8);b.legend(frameon=False,fontsize=7);b.grid(axis='y',alpha=.15)
    fig.tight_layout(pad=.6,w_pad=1.3);fig.savefig(OUT/'evidence_and_power.pdf',bbox_inches='tight');fig.savefig(OUT/'evidence_and_power.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
