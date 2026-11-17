"""Vector publication figures; all empirical values read from saved result files."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scipy.stats import norm
from experiments import kl,reverse_interval
R=Path(__file__).resolve().parents[1];P=R/'paper';A=json.loads((R/'results/experiments.json').read_text());M=json.loads((R/'results/monte_carlo_audit.json').read_text())
BLUE='#2765A0';TEAL='#007F78';ORANGE='#C56A20';PURPLE='#79529B';GRAY='#64717D';RED='#AB4145'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#87919A','axes.linewidth':.6,'grid.color':'#E6E9ED','grid.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'legend.fontsize':7})
def save(fig,name):
 fig.savefig(P/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(P/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(P/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.04);plt.close(fig)
def wilson(rate,n):
 z=norm.ppf(.975);d=1+z*z/n;c=(rate+z*z/(2*n))/d;h=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/d;return c-h,c+h
# Regenerate the original scientific architecture diagram.
import runpy
runpy.run_path(str(Path(__file__).with_name('overview_visual.py')))
# Stronger main evidence plot, preserving every tested sample size and adding uncertainty.
fig,ax=plt.subplots(1,3,figsize=(5.5,2.1),layout='constrained')
ns=np.array([r['n'] for r in A['separation']])
for key,label,c,ls in [('fitted_gain','Predictive gain',ORANGE,'-'),('reverse_kl','Reverse KL',TEAL,'--'),('true_mc_standard_error','True standard error',BLUE,':')]:ax[0].loglog(ns,[r[key] for r in A['separation']],color=c,ls=ls,lw=1.8,label=label)
ax[0].set(title='a  Diagnostic separation',xlabel='Evaluation size n',ylabel='Nats / standard error');ax[0].set_xticks([1e2,1e7,1e12]);ax[0].legend(loc='upper left',fontsize=6.4);ax[0].grid(axis='y',which='major')
for axis,scenario,title in [(ax[1],'redundant_evidence','b  Redundant: I = 0'),(ax[2],'genuine_evidence','c  Informative: I = 0.368')]:
 rows=[r for r in A['audit_simulations'] if r['scenario']==scenario]
 for key,label,c,marker in [('fitted_gain_authorization_rate','Predictive',ORANGE,'o'),('information_authorization_rate','Information',TEAL,'s')]:
  n=np.array([r['n'] for r in rows]);p=np.array([r[key] for r in rows]);lo,hi=wilson(p,5000);axis.fill_between(n,lo,hi,color=c,alpha=.18);axis.plot(n,p,marker=marker,color=c,lw=1.5,ms=4,label=label)
 axis.set_xscale('log');axis.set(xlabel='Audit outcomes n',ylim=(-.035,1.07),title=title);axis.set_yticks([0,.5,1]);axis.grid(axis='y');axis.legend(loc='center right',fontsize=6.3)
ax[1].set_ylabel('Fraction positive');save(fig,'diagnostic_separation')
# Cost/evidence tradeoff, with no invented uncertainty intervals for unretained repetitions.
fig,ax=plt.subplots(1,3,figsize=(5.5,2.1),layout='constrained');rows=[r for r in M['rows'] if r['genuine_information']];S=np.array([r['draws_per_case'] for r in rows])
ax[0].plot(S,[r['mean_gain'] for r in rows],'-o',c=ORANGE,label='Fitted gain',ms=4);ax[0].plot(S,[r['mean_witness'] for r in rows],'-s',c=TEAL,label='Error witness',ms=4);ax[0].axhline(.01*np.log(20),c=GRAY,ls=':',label='Failure cost βL');ax[0].set(title='a  Error accounting',ylabel='Nats');ax[0].legend(fontsize=6.3)
for key,label,c,marker in [('mean_failure_aware_lower','Per-case β',TEAL,'o'),('mean_simultaneous_lower','β/n allowance',PURPLE,'s')]:ax[1].plot(S,[r[key] for r in rows],'-'+marker,c=c,label=label,ms=4)
ax[1].axhline(0,c=GRAY,lw=.8);ax[1].set(title='b  Mean lower bound',ylabel='Nats');ax[1].legend(fontsize=6.3)
for key,label,c,marker in [('failure_aware_positive_fraction','Per-case β',TEAL,'o'),('simultaneous_interval_positive_fraction','β/n allowance',PURPLE,'s')]:
 p=np.array([r[key] for r in rows]);lo,hi=wilson(p,1000);ax[2].fill_between(S,lo,hi,color=c,alpha=.15);ax[2].plot(S,p,'-'+marker,c=c,ms=4,label=label)
ax[2].set(title='c  Certificate power',ylabel='Fraction positive',ylim=(-.04,1.06));ax[2].legend(fontsize=6.3)
for a in ax:a.set_xscale('log',base=2);a.set_xticks(S,labels=S);a.set_xlabel('Draws per case S');a.grid(axis='y')
save(fig,'monte_carlo_tradeoff')
# Mechanistic figure, appendix: actual specified posterior and sharp witness geometry.
fig,ax=plt.subplots(1,3,figsize=(7.15,2.5),layout='constrained');x=np.linspace(-5,21,1200);density=.99*norm.pdf(x)+.01*norm.pdf(x,16)
ax[0].semilogy(x,density,c=TEAL,label='Exact mixture');ax[0].semilogy(x,norm.pdf(x),c=ORANGE,ls='--',label='Gaussian approximation');ax[0].set(ylim=(1e-6,1),xlabel='Latent value θ',ylabel='Density (log scale)',title='a  The rare mode is missed');ax[0].legend(fontsize=6.2)
u=.1;delta=.05;lo,hi=reverse_interval(u,delta);p=np.linspace(.001,.5,1000);ax[1].plot(p,kl(u,p),c=BLUE,label='Reverse KL');ax[1].plot(p,kl(p,u),c=TEAL,label='Forward KL');ax[1].axhline(delta,c=BLUE,ls=':',lw=.8);ax[1].axvspan(lo,hi,color=BLUE,alpha=.08);ax[1].scatter([lo,hi],kl(np.array([lo,hi]),u),color=TEAL,s=20,zorder=3);ax[1].set(ylim=(0,.4),xlabel='Unknown true probability p',ylabel='Nats',title='b  Exact binary witness');ax[1].legend(fontsize=6.2)
rr=A['gaussian_reparameterization'];ax[2].plot([r['rho'] for r in rr],[r['predictive_forward_kl'] for r in rr],'-o',c=ORANGE,ms=4,label='Gain after rotation');ax[2].set_yscale('log');ax[2].set(xlabel='Posterior correlation ρ',ylabel='Predictive gain (nats)',title='c  Same evidence, better inference');ax[2].text(.04,.9,'No new observations\nI = 0 throughout',transform=ax[2].transAxes,fontsize=7);ax[2].grid(axis='y')
save(fig,'mechanism_geometry')
# Full application results copied into foundation/results for portable graphics regeneration.
for version,label in [('research_v2','development'),('research_v3','validation')]:
 f=R.parent/version/'results/assessment_comparisons.json';dest=R/'results'/('application_'+label+'.json')
 if f.exists():dest.write_text(f.read_text())
fig,axes=plt.subplots(1,2,figsize=(7.15,4.1),layout='constrained')
for axis,label,title in zip(axes,['development','validation'],['a  Development: 48 configurations','b  Later validation: 10 policies']):
 data=json.loads((R/'results'/('application_'+label+'.json')).read_text());names=sorted({r['dataset'] for r in data});short={'NonInvasiveFetalECGThorax1':'FetalECGThorax1','UWaveGestureLibraryAll':'UWaveAll','MixedShapesRegularTrain':'MixedShapes','SemgHandGenderCh2':'SemgHandGender'}
 for i,name in enumerate(names):
  vals=np.array([r['assessment_nll_gain'] for r in data if r['dataset']==name]);axis.scatter(vals,np.repeat(i,len(vals))+np.linspace(-.11,.11,len(vals)),s=17,color=BLUE,alpha=.7);axis.scatter([vals.mean()],[i],marker='D',s=22,color=ORANGE,zorder=3)
 axis.axvline(0,c=GRAY,lw=.8);axis.set_yticks(range(len(names)),labels=[short.get(n,n) for n in names]);axis.invert_yaxis();axis.set(xlabel='Candidate log-score gain (nats)',title=title);axis.grid(axis='x');axis.text(.98,.02,'0 authorizations\nPolicy gain = 0',ha='right',transform=axis.transAxes,fontsize=7,color=PURPLE,bbox=dict(fc='white',ec='none',alpha=.9))
save(fig,'application_all_cases')
print('Created five vector figure families (PDF/SVG/PNG).')
