"""Original vector diagram of the audit's three coupled requirements."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parents[1]/'paper'
blue,teal,purple,orange,gray='#326DB0','#007F78','#79529B','#C56A20','#66737D'
fig=plt.figure(figsize=(7.2,3.45));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,c):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.012',fc=c+'09',ec=c,lw=.9))
def txt(x,y,t,**kw):ax.text(x,y,t,va='top',fontsize=8,**kw)
def arrow(x,y,xx,yy,c=gray):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=10,lw=1,color=c))
txt(.005,.98,'WHAT MUST BE ACCOUNTED FOR BEFORE A GAIN BECOMES AN INFORMATION CLAIM?',weight='bold',color='#223846',fontsize=8.5) if False else None
ax.text(.005,.98,'THREE REQUIREMENTS, ONE AUDIT TARGET',va='top',weight='bold',fontsize=10,color='#223846')
# Three scientific components, with the mathematical handoff visible.
for x,c,title in [(0.01,orange,'1  ATTRIBUTE THE GAIN'),(.35,blue,'2  BOUND BASELINE ERROR'),(.69,teal,'3  AUDIT THE CORRECTED SCORE')]:
 box(x,.39,.295,.49,c);ax.text(x+.011,.852,title,va='top',fontsize=7.7,weight='bold',color=c)
ax.text(.028,.76,r'$H\ \longrightarrow\ q_0(Y\mid H)$',fontsize=10)
ax.text(.028,.67,r'$(H,G)\ \longrightarrow\ q_1(Y\mid H,G)$',fontsize=10)
ax.text(.028,.56,r'$\Delta=I(Y;G\mid H)+A_0-A_1$',fontsize=10)
txt(.028,.475,'Better approximation alone can\nexplain a gain when I = 0.',linespacing=1.45)
ax.text(.368,.755,r'$\mathrm{KL}(Q\Vert P)\leq\delta$ supplies the budget',fontsize=7.4)
# Actual analytic geometry, not a decorative miniature plot.
import numpy as np
from experiments import kl,reverse_interval
u=.1;delta=.05;lo,hi=reverse_interval(u,delta);pp=np.linspace(.002,.36,350)
inset=fig.add_axes([.395,.52,.23,.18]);inset.plot(pp,kl(u,pp),c=blue,lw=.95,label='Reverse KL');inset.plot(pp,kl(pp,u),c=orange,lw=.95,label='Forward KL');inset.axhline(delta,c=blue,ls=':',lw=.7);inset.axvspan(lo,hi,color=blue,alpha=.10);inset.scatter([lo,hi],[float(kl(lo,u)),float(kl(hi,u))],c=orange,s=9,zorder=4)
inset.set(xlim=(0,.36),ylim=(0,.24));inset.set_xticks([.1,.3]);inset.set_yticks([0,.2]);inset.tick_params(labelsize=5.8,pad=1,length=2);inset.spines[['top','right']].set_visible(False);inset.set_xlabel('Unknown outcome probability',fontsize=6,labelpad=0);inset.legend(fontsize=5.1,loc='upper left',frameon=False,handlelength=1)
ax.text(.368,.437,'Endpoint maximum gives bound b.',fontsize=7.2)
ax.text(.368,.402,'Categorical: one root at min q.',fontsize=7.2)
ax.text(.708,.755,r'$T_i=\log(q_{1,i}(Y_i)/q_{0,i}(Y_i))-b_i$',fontsize=9)
ax.text(.708,.65,r'$L_I=\overline{T}-\beta L-r_\alpha(T,n)$',fontsize=11)
txt(.708,.565,'Subtract both remaining allowances:',fontsize=8) if False else None
ax.text(.708,.565,'Subtract both remaining allowances:',fontsize=7.6)
txt(.708,.505,'βL: failures of the error bound\nrα: finite audit-sample uncertainty',linespacing=1.6)
arrow(.309,.62,.344,.62,orange);arrow(.649,.62,.684,.62,blue)
# Calibration acts on the whole bound; audit results never feed back.
box(.01,.205,.63,.105,gray)
ax.text(.025,.278,'CALIBRATION',fontsize=8,weight='bold',color=gray)
ax.text(.19,.278,'Choose B and β to improve the final bound.',fontsize=7.8)
ax.text(.19,.232,'Freeze predictions, bounds, and settings.',fontsize=7.6)
arrow(.16,.385,.16,.318,gray);arrow(.49,.385,.49,.318,gray)
arrow(.649,.257,.691,.257,teal)
box(.70,.205,.285,.105,teal);ax.text(.71,.277,'INDEPENDENT AUDIT LABELS',fontsize=8,weight='bold',color=teal);ax.text(.71,.231,'Evaluate once; never retune.',fontsize=7.8)
arrow(.84,.388,.84,.319,teal)
ax.plot([.01,.985],[.155,.155],c='#D4DBDF',lw=.8)
ax.text(.014,.108,r'Valid witness and $L_I > 0$',color=teal,fontsize=8.4,weight='bold');ax.text(.014,.045,'→ conditional information under stated assumptions',fontsize=8)
ax.text(.55,.108,r'No valid witness, or $L_I \leq 0$',color=gray,fontsize=8.4,weight='bold');ax.text(.55,.045,'→ no positive information claim; report prediction',fontsize=8)
for ext in ['pdf','svg','png']:fig.savefig(P/('design_overview.'+ext),bbox_inches='tight',pad_inches=.035,dpi=230)
plt.close(fig)
