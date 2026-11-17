"""Fresh, modest UCR study; not a reconstruction of the original 480 runs."""
from pathlib import Path
import json,hashlib,io,zipfile,urllib.request,time,platform,sys
import numpy as np
from scipy.ndimage import convolve1d
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.neighbors import NearestNeighbors
import sklearn,scipy
from seb import *
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CACHE=Path(__file__).resolve().parents[3]/'work'/'datasets';CACHE.mkdir(parents=True,exist_ok=True)
DATASETS=['ECG200','FordA','Wafer','ECG5000','ItalyPowerDemand','TwoLeadECG']
SEEDS=[17,29,43];TOTAL=len(DATASETS)*len(SEEDS)*2

def dump(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=True))
def norm(x):
 x=x-x.mean(1,keepdims=True);return x/np.maximum(x.std(1,keepdims=True),1e-12)
def load(name):
 url=f'https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip';p=CACHE/(name+'.zip')
 if not p.exists():urllib.request.urlretrieve(url,p)
 z=zipfile.ZipFile(p); out=[]
 for role in ('TRAIN','TEST'):
  b=z.read(name+'_'+role+'.txt');a=np.loadtxt(io.BytesIO(b));out.append((norm(a[:,1:]),a[:,0]))
 meta={'dataset':name,'url':url,'zip_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'train':len(out[0][0]),'test':len(out[1][0]),'length':out[0][0].shape[1]}
 return *out,meta

def conv(x,seed=2027):
 rng=np.random.default_rng(seed);out=[]
 for j in range(64):
  length=int(rng.choice([7,9,11]));d=int(rng.integers(1,max(2,x.shape[1]//length)))
  w=rng.normal(size=length);w-=w.mean();w/=np.linalg.norm(w)
  kernel=np.zeros((length-1)*d+1);kernel[::d]=w
  a=convolve1d(x,kernel,axis=1,mode='constant')+rng.uniform(-1,1)
  out.extend([(a>0).mean(1),a.max(1)])
 return np.stack(out,1)
def time_features(x):
 chunks=np.array_split(x,8,axis=1);out=[z.mean(1) for z in chunks]+[z.std(1) for z in chunks]
 out.extend([np.quantile(x,q,axis=1) for q in (.05,.25,.5,.75,.95)])
 out.extend([np.mean(x*x,1),np.mean(abs(np.diff(x,axis=1)),1),np.max(x,1),np.min(x,1),np.mean(x**3,1),np.mean(x**4,1),np.mean(x[:,1:]*x[:,:-1],1),np.mean(x[:,2:]*x[:,:-2],1)])
 return np.stack(out,1)
def fullprob(m,x,k):
 out=np.zeros((len(x),k));out[:,np.asarray(m.classes_,int)]=m.predict_proba(x);return out
def cues(p):
 p=np.clip(p,EPS,1-EPS);p=p/p.sum(1,keepdims=True);l=np.log(p);l-=l.mean(1,keepdims=True)
 s=np.sort(l,axis=1);return np.column_stack([l,np.eye(p.shape[1])[p.argmax(1)],p.max(1),s[:,-1]-s[:,-2],-(p*np.log(p)).sum(1)])
def fit_reliability(x,c):
 if len(np.unique(c))<2:
  model=DummyClassifier(strategy='constant',constant=int(c[0])).fit(x,c);return model,np.full(len(c),float(c[0])),None
 folds=min(3,int(np.bincount(c.astype(int)).min()));best=None
 for reg in (.01,.1,1.,10.):
  m=make_pipeline(StandardScaler(),LogisticRegression(C=reg,max_iter=1500,solver='lbfgs'))
  if folds>=2:
   pred=cross_val_predict(m,x,c,cv=StratifiedKFold(folds,shuffle=True,random_state=2027),method='predict_proba')[:,1]
  else:
   pred=np.full(len(c),c.mean())
  key=(logloss(c,pred).mean(),reg)
  if best is None or key<best[0]:best=(key,m,pred)
 return best[1].fit(x,c),best[2],best[0][1]
def predict(m,x):
 p=m.predict_proba(x);cl=m.classes_
 return np.clip(p[:,list(cl).index(1)] if 1 in cl else np.zeros(len(x)),EPS,1-EPS)
def stress(x,kind):
 if kind=='block':
  a=x.copy();start=x.shape[1]//3;a[:,start:start+x.shape[1]//5]=0;return norm(a)
 f=np.fft.rfft(x,axis=1);m=f.shape[1]-1;k=max(1,int(.2*m))
 if kind=='dominant':
  ix=np.argsort(abs(f[:,1:]),axis=1)[:,-k:]+1;np.put_along_axis(f,ix,0,axis=1)
 else:
  rng=np.random.default_rng(2027);start=rng.integers(1,m-k+2,size=len(x))
  for i,s in enumerate(start):f[i,s:s+k]=0
 return norm(np.fft.irfft(f,n=x.shape[1],axis=1))

if __name__=='__main__':
 start=time.time(); rows=[];configs=[];sources=[]
 dump(OUT/'protocol.json',{'datasets':DATASETS,'seeds':SEEDS,'backbones':['ExtraTrees','RandomConv'],'family_size':TOTAL,'alpha':.05,'clip':EPS,'split':[.5,.25,.25],'role_assignment':'sha256(dataset:train:id:SEB2027); same across seeds/backbones','gate':'7 one-sided bounds/configuration, global Bonferroni','stress':['block','dominant','band'],'new_study':True})
 for name in DATASETS:
  (x,y),(xt,yt),meta=load(name);classes=np.unique(y);y=np.searchsorted(classes,y);yt=np.searchsorted(classes,yt);sources.append(meta);dump(OUT/'sources.json',sources)
  ids=sorted(range(len(x)),key=lambda i:hashlib.sha256(f'{name}:train:{i}:SEB2027'.encode()).hexdigest())
  nf=int(.5*len(x));nc=int(.25*len(x));fi=np.array(ids[:nf]);ci=np.array(ids[nf:nf+nc]);gi=np.array(ids[nf+nc:])
  allx=np.concatenate([x,xt]);cx=conv(allx);gg=bundle(allx);tf=time_features(allx)
  rp=np.random.default_rng(2027).normal(size=(x.shape[1],29))/np.sqrt(x.shape[1]);proj=allx@rp
  nn=NearestNeighbors(n_neighbors=1).fit(x[fi]);dist=np.log1p(nn.kneighbors(allx)[0]);freq=np.bincount(y[fi],minlength=len(classes))/len(fi)
  for seed in SEEDS:
   models={
    'ExtraTrees':ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,max_features='sqrt',random_state=seed,n_jobs=4).fit(x[fi],y[fi]),
    'RandomConv':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=seed)).fit(cx[fi],y[fi])}
   prob={k:fullprob(m,allx if k=='ExtraTrees' else cx,len(classes)) for k,m in models.items()}
   for backbone,m in models.items():
    folder=OUT/f'{name}_{backbone}_{seed}';folder.mkdir(exist_ok=True)
    p=prob[backbone];other=prob['RandomConv' if backbone=='ExtraTrees' else 'ExtraTrees'];pred=p.argmax(1);c=(pred==np.r_[y,yt]).astype(int);h=cues(p)
    ag=np.column_stack([other.max(1),(p.argmax(1)==other.argmax(1)).astype(float),(p*other).sum(1)])
    atyp=np.column_stack([dist,-np.log(np.maximum(freq[p.argmax(1)],EPS))])
    extra=np.column_stack([tf,proj,ag,atyp]);fixedproj=np.random.default_rng(2027).normal(size=(extra.shape[1],29))/np.sqrt(extra.shape[1])
    feats={'output':h,'output_quadratic':PolynomialFeatures(2,include_bias=False).fit_transform(h),'time':np.c_[h,tf],'random_projection':np.c_[h,proj],'agreement':np.c_[h,ag],'atypicality':np.c_[h,atyp],'all_features':np.c_[h,extra@fixedproj],'spectral':np.c_[h,gg]}
    fit={};cv={};reg={};scores={}
    for key,f in feats.items():
     fit[key],cv[key],reg[key]=fit_reliability(f[ci],c[ci]);scores[key]=predict(fit[key],f)
    non=[k for k in feats if k!='spectral'];eligible=[k for k in non if logloss(c[ci],cv[k]).mean()<=logloss(c[ci],cv['output']).mean()+1e-12]
    reference=min(eligible,key=lambda k:(-auc(c[ci],cv[k]) if np.isfinite(auc(c[ci],cv[k])) else 0,logloss(c[ci],cv[k]).mean(),k))
    t1=float(np.quantile(cv['spectral'],.2));tr=float(np.quantile(cv[reference],.2))
    gate_result=gate(c[gi],scores['output'][gi],scores['spectral'][gi],scores[reference][gi],t1,tr,TOTAL)
    scores['intercept']=np.full(len(allx),np.clip(c[ci].mean(),EPS,1-EPS));scores['raw']=np.clip(p.max(1),EPS,1-EPS)
    scores['reference']=scores[reference].copy();scores['policy']=scores['spectral'].copy() if gate_result['authorized'] else scores['reference'].copy()
    # All coalition models trained on calibration only.
    coal=np.zeros((16,len(allx)));coal[0]=scores['output'];coal[15]=scores['spectral']
    for mask in range(1,15):
     js=sum([FAMILIES[j] for j in range(4) if mask&(1<<j)],[]);f=np.c_[h,gg[:,js]]
     cm,_,_=fit_reliability(f[ci],c[ci]);coal[mask]=predict(cm,f)
    vv=logloss(c[None,:],coal[0][None,:])-logloss(c[None,:],coal);phi=shapley(vv)
    assert np.max(abs(phi.sum(0)-vv[15]))<1e-10
    if not gate_result['authorized']:assert np.array_equal(scores['policy'],scores['reference'])
    roles=np.full(len(allx),'test',dtype='<U12');roles[fi]='fit';roles[ci]='calibration';roles[gi]='gate'
    np.savez_compressed(folder/'predictions.npz',correct=c,label=np.r_[y,yt],predicted=pred,roles=roles,ids=np.arange(len(allx)),logits=np.log(np.clip(p,EPS,1)),bundle=gg,coalitions=coal,shapley=phi,**scores)
    conf={'dataset':name,'backbone':backbone,'seed':seed,'fit_n':len(fi),'cal_n':len(ci),'gate_n':len(gi),'test_n':len(xt),'reference':reference,'regularization':reg,'gate':gate_result,'threshold_spectral':t1,'threshold_reference':tr,'test_log_gain':float(vv[15,len(x):].mean()),'test_harm_fraction':float((vv[15,len(x):]<0).mean()),'class_prevalence':freq.tolist(),'fit_missing_classes':int((freq==0).sum()),'coalition_closure':float(np.max(abs(phi.sum(0)-vv[15])))}
    configs.append(conf);dump(folder/'config.json',conf)
    for key,r in scores.items():rows.append(dict(dataset=name,backbone=backbone,seed=seed,condition='clean',method=key,**metrics(c[len(x):],r[len(x):])))
    # Test corruption: recompute both classifiers and every evidence feature.
    for kind in ('block','dominant','band'):
     xx=stress(xt,kind);ccx=conv(xx);pp={k:fullprob(mm,xx if k=='ExtraTrees' else ccx,len(classes)) for k,mm in models.items()};p2=pp[backbone];o2=pp['RandomConv' if backbone=='ExtraTrees' else 'ExtraTrees'];c2=(p2.argmax(1)==yt).astype(int);h2=cues(p2)
     ag2=np.column_stack([o2.max(1),(p2.argmax(1)==o2.argmax(1)).astype(float),(p2*o2).sum(1)]);at2=np.column_stack([np.log1p(nn.kneighbors(xx)[0]),-np.log(np.maximum(freq[p2.argmax(1)],EPS))]);tf2=time_features(xx);pr2=xx@rp;g2=bundle(xx)
     ff={'output':h2,'output_quadratic':PolynomialFeatures(2,include_bias=False).fit_transform(h2),'time':np.c_[h2,tf2],'random_projection':np.c_[h2,pr2],'agreement':np.c_[h2,ag2],'atypicality':np.c_[h2,at2],'all_features':np.c_[h2,np.c_[tf2,pr2,ag2,at2]@fixedproj],'spectral':np.c_[h2,g2]}
     ss={k:predict(fit[k],f) for k,f in ff.items()};ss['intercept']=np.full(len(xx),np.clip(c[ci].mean(),EPS,1-EPS));ss['raw']=np.clip(p2.max(1),EPS,1-EPS);ss['reference']=ss[reference].copy();ss['policy']=ss['spectral'].copy() if gate_result['authorized'] else ss['reference'].copy()
     np.savez_compressed(folder/(kind+'.npz'),correct=c2,label=yt,predicted=p2.argmax(1),**ss)
     for key,r in ss.items():rows.append(dict(dataset=name,backbone=backbone,seed=seed,condition=kind,method=key,**metrics(c2,r)))
    dump(OUT/'metrics.json',rows);dump(OUT/'configurations.json',configs)
    print(name,backbone,seed,'reference',reference,'gate',gate_result['authorized'],'test gain',round(conf['test_log_gain'],5),flush=True)
 dump(OUT/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'machine':platform.machine(),'elapsed_seconds':time.time()-start,'cpu_threads_per_forest':4})
 print('COMPLETE',len(configs),'configurations',time.time()-start,flush=True)
