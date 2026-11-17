"""Paired finite-sample audit evidence, without decorative or simulated extra data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[1];P=R/'paper'
d=json.loads((R/'results/certificate_design_fresh.json').read_text());raw=json.loads((R/'results/certificate_design_fresh_trials.json').read_text());S=sorted({x['S'] for x in d['summary']})
methods=['fixed','gain_selected','witness_selected','bound_selected'];names=['Fixed','Prediction','Confidence','Joint'];blue,teal,purple,orange='#326DB0','#007F78','#79529B','#C56A20'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,'axes.titleweight':'bold','axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.fontsize':6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(6.8,4.65));gs=fig.add_gridspec(2,2,left=.095,right=.965,bottom=.10,top=.94,hspace=.57,wspace=.36)
a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);e=fig.add_subplot(gs[1,1])
def records(s,m,genuine=True):return sorted([x for x in raw if x['S']==s and x['method']==m and x['genuine']==genuine],key=lambda x:x['trial'])
# Full factorial power surface: no repeated, nearly coincident curves.
power=np.array([[next(x['positive_rate'] for x in d['summary'] if x['S']==s and x['genuine'] and x['method']==m) for s in S] for m in methods])*100
im=a.imshow(power,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
for i in range(4):
 for j in range(len(S)):
  v=power[i,j];a.text(j,i,f'{v:.0f}' if v in [0,100] else f'{v:.1f}',ha='center',va='center',fontsize=6.2,color='white' if v>55 else '#263C45')
 good=np.flatnonzero(power[i]>=90)
 if len(good):a.plot(good[0],i-.38,marker='v',ms=4,color='white',mec='#263C45',mew=.5)
a.set(xticks=range(len(S)),xticklabels=[str(s) for s in S],yticks=range(4),yticklabels=names,title='a  Certification across the compute budget',xlabel='Posterior draws per case');a.tick_params(axis='x',rotation=35,length=0);a.tick_params(axis='y',length=0)
a.set_xticks(np.arange(-.5,len(S),1),minor=True);a.set_yticks(np.arange(-.5,4,1),minor=True);a.grid(which='minor',color='white',lw=.8);a.tick_params(which='minor',length=0)
a.set_title('a  Certification across the compute budget',pad=19)
a.text(0,1.02,'Positive information bounds (%)',transform=a.transAxes,fontsize=6.3)
# Count paired decisions directly: gained and lost positive conclusions are visible.
transition_rows=[]
for budget in [512,1024,2048]:
 for target,reference,step in [('witness_selected','fixed','confidence'),('bound_selected','witness_selected','joint')]:
  old=np.array([v['lower'] for v in records(budget,reference)])>0
  new=np.array([v['lower'] for v in records(budget,target)])>0
  gained=int(np.sum(~old & new));lost=int(np.sum(old & ~new))
  transition_rows.append({'draws':budget,'step':step,'gained':gained,'lost':lost,'unchanged':int(np.sum(old==new))})
for i,v in enumerate(transition_rows):
 b.barh(i,v['gained'],height=.57,color=blue)
 b.barh(i,-v['lost'],height=.57,color=orange)
 if v['gained']:b.text(v['gained']+7,i,str(v['gained']),va='center',fontsize=6.8,color=blue)
 if v['lost']:b.text(-v['lost']-7,i,str(v['lost']),va='center',ha='right',fontsize=6.8,color=orange)
 if not v['gained'] and not v['lost']:b.text(7,i,'0 changes',va='center',fontsize=6.5,color='#66737D')
b.axvline(0,c='#66737D',lw=.8)
b.set(yticks=range(6),yticklabels=[f"{v['draws']:,} / {v['step']}" for v in transition_rows],xlim=(-85,460),ylim=(5.7,-1.2),xlabel='Changed conclusions out of 500 audits')
b.set_title('b  Which audits change their conclusion?',pad=19)
b.set_xticks([-50,0,100,200,300,400],labels=['50 lost','0','100','200','300','400'])
b.tick_params(axis='y',length=0);b.grid(axis='x',lw=.4,alpha=.18)
for yy in [1.5,3.5]:b.axhline(yy,c='#D5DBDF',lw=.6)
b.legend(handles=[Patch(fc=blue,label='Gained positive'),Patch(fc=orange,label='Lost positive')],loc='upper right',ncol=2,fontsize=5.8)
(R/'results/figure2_decision_transitions.json').write_text(json.dumps(transition_rows,indent=2))
# Exact accounting of the three allowances.
rows=[next(x for x in d['summary'] if x['S']==1024 and x['genuine'] and x['method']==m) for m in methods]
for i,r in enumerate(rows):
 base=r['mean_witness'];fail=r['mean_beta']*np.log(1/d['epsilon']);sampling=r['mean_gain']-r['mean_lower']-base-fail
 for left,width,col in [(0,base,blue),(base,fail,purple),(base+fail,sampling,'#D5DBDF')]:c.barh(i,width,left=left,height=.52,color=col)
 c.scatter(r['mean_gain'],i,c=orange,marker='D',s=22,zorder=5)
 c.annotate(f"{r['mean_lower']:+.3f}",(max(r['mean_gain'],base+fail+sampling)+.006,i),va='center',fontsize=6.4,color=teal if r['mean_lower']>0 else '#59646E')
c.set_yticks(range(4),labels=names);c.set(title='c  Accounting for the final bound',xlabel='Mean nats at 1,024 draws',xlim=(0,.23),ylim=(3.7,-1.6))
c.legend(handles=[Patch(color=blue,label='Baseline error'),Patch(color=purple,label='Bound failures'),Patch(color='#D5DBDF',label='Audit sampling'),Line2D([],[],color=orange,marker='D',ls='',ms=3,label='Prediction gain')],loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,fontsize=5.7)
# Paired joint distributions: every point is an observed repetition.
allv=[]
for genuine,col,label in [(False,purple,'No added information'),(True,teal,'Added information')]:
 rr=records(1024,'bound_selected',genuine);x=np.array([r['gain'] for r in rr]);y=np.array([r['lower'] for r in rr]);allv.append((x,y,col))
 e.scatter(x,y,s=5,alpha=.20,c=col,edgecolors='none',rasterized=False)
 e.scatter(np.median(x),np.median(y),s=24,c=col,marker='D',ec='white',lw=.6,zorder=5,label=label)
xx=np.concatenate([v[0] for v in allv]);yy=np.concatenate([v[1] for v in allv]);xmin=min(0,xx.min()-.008);xmax=xx.max()+.01;ymin=yy.min()-.008;ymax=max(yy.max()+.008,.035)
e.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='Predictive gain (nats)',ylabel='Information lower bound (nats)',title='d  The gain and claim in each audit')
e.set_xticks([0,.05,.10,.15,.20]);e.set_xticklabels(['0.00','0.05','0.10','0.15','0.20'])
e.axhspan(0,ymax,color=teal,alpha=.05);e.axhline(0,c='#44535B',lw=.8);e.plot([0,min(xmax,ymax)],[0,min(xmax,ymax)],c='#939CA3',ls=':',lw=.8)
e.legend(loc='lower right',fontsize=5.8,handletextpad=.2)
e.text(.02,.97,'Positive information claim',transform=e.transAxes,va='top',fontsize=6,color=teal)
# Marginal densities inside the plotting frame, preserving point pairing in the central panel.
for x,y,col in allv:
 grid=np.linspace(xmin,xmax,240);den=gaussian_kde(x)(grid);e.fill_between(grid,ymin,ymin+den/den.max()*(ymax-ymin)*.08,color=col,alpha=.25,lw=0)
 grid=np.linspace(ymin,ymax,240);den=gaussian_kde(y)(grid);e.fill_betweenx(grid,xmax-den/den.max()*(xmax-xmin)*.07,xmax,color=col,alpha=.25,lw=0)
for ext in ['pdf','svg','png']:fig.savefig(P/('certificate_design.'+ext),bbox_inches='tight',pad_inches=.04,dpi=250)
plt.close(fig)
print('Rendered power surface, paired ablations, correction accounting, and paired audit outcomes.')
