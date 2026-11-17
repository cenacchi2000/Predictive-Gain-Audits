"""SEB bounded-update study. Assessment labels are used only in report.py.

Development datasets were inspected in the earlier study. New datasets are
declared in protocol.json before their assessment labels are evaluated.
"""
from pathlib import Path
import argparse, hashlib, io, json, time, urllib.request, zipfile, gc, sys, joblib
import numpy as np
from scipy.special import expit, logit, softmax, logsumexp
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from aeon.transformations.collection.convolution_based import MiniRocket, HydraTransformer
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.datasets import load_from_ts_file
from threadpoolctl import threadpool_limits
from seb import bundle, logloss, metrics, EPS
from offset import OffsetLogistic

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'results'
DEVELOPMENT=['FordA','ECG5000','Wafer']
NEW=['ElectricDevices','StarLightCurves','Crop','ChlorineConcentration',
     'NonInvasiveFetalECGThorax1','Phoneme','UWaveGestureLibraryAll',
     'InsectWingbeatSound','FaceAll','TwoPatterns','PenDigits','LSST']
SEEDS=[17,29]
FAMILY=48
ALPHA=.05
SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def normalized(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==2:x=x[:,None,:]
    if not np.isfinite(x).all():
        # Missing entries use each series/channel's observed mean; no labels.
        mu=np.nanmean(np.where(np.isfinite(x),x,np.nan),axis=-1,keepdims=True)
        x=np.where(np.isfinite(x),x,np.nan_to_num(mu)).astype(np.float32)
    x=x-x.mean(-1,keepdims=True)
    return x/np.maximum(x.std(-1,keepdims=True),1e-6)

def load(name):
    DATA.mkdir(exist_ok=True); path=DATA/(name+'.zip')
    old=ROOT.parent/'datasets'/(name+'.zip')
    if not path.exists():
        if old.exists():path.write_bytes(old.read_bytes())
        else:urllib.request.urlretrieve('https://www.timeseriesclassification.com/aeon-toolkit/'+name+'.zip',path)
    z=zipfile.ZipFile(path); values=[]
    for role in ['TRAIN','TEST']:
        names=z.namelist(); txt=next((s for s in names if s.endswith(name+'_'+role+'.txt')),None)
        if txt:
            a=np.loadtxt(io.BytesIO(z.read(txt))); values.append((normalized(a[:,1:]),a[:,0].astype(str)))
        else:
            ts=next(s for s in names if s.endswith(name+'_'+role+'.ts'))
            local=DATA/(name+'_'+role+'.ts');local.write_bytes(z.read(ts))
            x,y=load_from_ts_file(local);values.append((normalized(x),np.asarray(y).astype(str)))
    enc=LabelEncoder().fit(values[0][1])
    x,y=values[0]; xt,yt=values[1]
    meta={'dataset':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'train':len(x),'audit_pool':len(xt),'channels':x.shape[1],'length':x.shape[2],
          'classes':len(enc.classes_),'source':'https://www.timeseriesclassification.com/aeon-toolkit/'+name+'.zip'}
    train_hash={hashlib.sha256(a.tobytes()).digest() for a in x};seen=set();eligible=[]
    for i,a in enumerate(xt):
        key=hashlib.sha256(a.tobytes()).digest()
        if key not in train_hash and key not in seen:eligible.append(i)
        seen.add(key)
    meta['eligible_audit_ids']=eligible
    meta['eligible_audit_n']=len(eligible)
    meta['audit_exclusion_rule']='First original row per exact normalized float32 waveform; exclude any waveform matching official TRAIN. Labels do not affect this rule.'
    return x,enc.transform(y),xt,enc.transform(yt),meta

def roles(name,n,eligible=None):
    ids=sorted(range(n) if eligible is None else eligible,key=lambda i:hashlib.sha256(f'SEB-v2:{name}:audit:{i}'.encode()).hexdigest())
    count=len(ids)
    r=np.full(n,'excluded',dtype='<U12');r[ids]='assessment';r[ids[:int(.4*count)]]='calibration';r[ids[int(.4*count):int(.7*count)]]='gate'
    return r

def ordered_gate(name,role_array):
    return np.array(sorted(np.flatnonzero(role_array=='gate'),key=lambda i:hashlib.sha256(f'SEB-v2:gate-order:{name}:{i}'.encode()).hexdigest()),dtype=int)

def logits_matrix(a):
    return np.c_[-a/2,a/2] if a.ndim==1 else a

def fit_backbone(name,seed,x,y,xt):
    folder=OUT/name/f'backbones_{seed}';folder.mkdir(parents=True,exist_ok=True)
    for kind in ['MiniRocket','Hydra']:
        dest=folder/(kind+'.npz')
        if dest.exists():continue
        t=time.time();print(name,seed,kind,'fitting',flush=True)
        # MiniRocket requires nine time points. Reflect padding is classifier-only.
        xx=x if x.shape[-1]>=9 else np.pad(x,((0,0),(0,0),(0,9-x.shape[-1])),mode='reflect')
        xa=xt if xt.shape[-1]>=9 else np.pad(xt,((0,0),(0,0),(0,9-xt.shape[-1])),mode='reflect')
        if kind=='MiniRocket':
            transform=MiniRocket(n_kernels=10000,n_jobs=4,random_state=seed)
            train=transform.fit_transform(xx); scaler=StandardScaler(with_mean=False)
            train=scaler.fit_transform(train).astype(np.float32)
        else:
            transform=HydraTransformer(n_jobs=4,random_state=seed)
            train=transform.fit_transform(xx);scaler=_SparseScaler()
            train=scaler.fit_transform(train).numpy()
        model=RidgeClassifierCV(alphas=np.logspace(-3,3,10)).fit(train,y)
        # Fixed coordinate subset is a representation control, not a full ConfidNet replica.
        ix=np.random.default_rng(2027).choice(train.shape[1],min(128,train.shape[1]),replace=False)
        nn=NearestNeighbors(n_neighbors=1,n_jobs=4).fit(train[:,ix])
        logits=[]; reps=[]; distances=[]
        for start in range(0,len(xa),256):
            f=scaler.transform(transform.transform(xa[start:start+256]))
            if hasattr(f,'numpy'):f=f.numpy()
            f=np.asarray(f,dtype=np.float32)
            logits.append(logits_matrix(model.decision_function(f)))
            reps.append(f[:,ix]);distances.append(np.log1p(nn.kneighbors(f[:,ix])[0]))
        assert all(np.isfinite(a).all() for a in logits+reps+distances)
        np.savez_compressed(dest,logits=np.concatenate(logits),representation=np.concatenate(reps),distance=np.concatenate(distances))
        joblib.dump({'transform':transform,'scaler':scaler,'model':model,'indices':ix,'nn':nn},folder/(kind+'.joblib'),compress=3)
        dump(folder/(kind+'.json'),{'seconds':time.time()-t,'alpha':float(model.alpha_),'features':train.shape[1], 'representation_indices':ix.tolist()})
        del train,transform,model;gc.collect()

def output_features(z):
    z=z-z.mean(1,keepdims=True);p=softmax(z,axis=1);s=np.sort(z,axis=1)
    return np.c_[z,np.eye(z.shape[1])[z.argmax(1)],p.max(1),s[:,-1]-s[:,-2],-(p*np.log(np.maximum(p,EPS))).sum(1)]

def time_features(x):
    x=x.mean(1);parts=np.array_split(x,8,axis=-1)
    return np.column_stack([a.mean(1) for a in parts]+[a.std(1) for a in parts]+[np.quantile(x,q,axis=1) for q in [.05,.25,.5,.75,.95]]+
        [np.mean(np.abs(np.diff(x)),1),np.mean(x**3,1),np.mean(x**4,1),np.max(x,1),np.min(x,1),np.mean(x[:,1:]*x[:,:-1],1),np.mean(x[:,2:]*x[:,:-2],1),np.mean(x*x,1)])

def probability(model,x):
    return np.clip(model.predict_proba(x)[:,1],EPS,1-EPS)

def learners():
    return {'lr01':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=600)),
            'lr1':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=600)),
            'gb7':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=10,min_samples_leaf=30,early_stopping=False,random_state=2027),
            'gb15':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=10,min_samples_leaf=30,early_stopping=False,random_state=2027)}

def fit_group(features,c,cal):
    cc=c[cal];nclass=np.bincount(cc,minlength=2);cvout={};out={};models={}
    if nclass.min()<3:
        prior=(cc.sum()+.5)/(len(cc)+1)
        return {'constant':np.full(len(cc),prior)},{'constant':np.full(len(c),prior)},{'constant':prior}
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2027).split(np.zeros(len(cc)),cc))
    for family,feat in features.items():
        for alg,model in learners().items():
            key=family+'/'+alg;pred=np.zeros(len(cc))
            for tr,va in folds:
                fitted=clone(model).fit(feat[cal][tr],cc[tr]);pred[va]=probability(fitted,feat[cal][va])
            fitted=clone(model).fit(feat[cal],cc)
            cvout[key]=pred;out[key]=probability(fitted,feat);models[key]=fitted
    return cvout,out,models

def bounded_update(reference,candidate,bound):
    a=logit(np.clip(reference,EPS,1-EPS));b=logit(np.clip(candidate,EPS,1-EPS))
    return expit(a+np.clip(b-a,-bound,bound))

def offset_candidates(features,models,reference,scores,g,c,cal):
    """Outer-fold reference refits prevent validation labels entering offsets."""
    if np.bincount(c[cal],minlength=2).min()<3:return {},{},{}
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2027).split(np.zeros(len(cal)),c[cal]))
    refs=[]
    for tr,va in folds:
        tr=cal[tr];va=cal[va]
        if reference in ['raw','intercept'] or reference=='constant':
            if reference=='raw':rt=scores['raw'][tr];rv=scores['raw'][va]
            else:rt=np.full(len(tr),(c[tr].sum()+.5)/(len(tr)+1));rv=np.full(len(va),rt[0])
        else:
            f=features[reference.split('/')[0]]
            m=clone(models[reference]).fit(f[tr],c[tr]);rt=probability(m,f[tr]);rv=probability(m,f[va])
        refs.append((rt,rv))
    cv={};out={};fit={}
    for C in [.01,.1,1.]:
        key='offset_spectral/'+str(C);pred=np.zeros(len(cal))
        for (tr,va),(rt,rv) in zip(folds,refs):
            m=OffsetLogistic(C).fit(rt,g[cal[tr]],c[cal[tr]])
            pred[va]=m.predict(rv,g[cal[va]])
        m=OffsetLogistic(C).fit(scores[reference][cal],g[cal],c[cal])
        cv[key]=pred;out[key]=m.predict(scores[reference],g);fit[key]=m
    return cv,out,fit

def betting(gain,bound,family=FAMILY):
    # Nine fixed bets and one predictable variance-adaptive bet.
    fractions=2.**np.arange(-8,1);factors=1+gain[:,None]*fractions[None,:]/bound
    if factors.min() < -1e-10:raise ValueError('Bound violation')
    with np.errstate(divide='ignore'):
        logs=np.log(np.maximum(factors,0)).sum(0)
    past_sum=np.r_[0,np.cumsum(gain[:-1])]
    past_square=np.r_[0,np.cumsum(gain[:-1]**2)]
    lam=np.clip(past_sum/(2*past_square+bound**2),0,.5/bound)
    adaptive=float(np.log1p(lam*gain).sum())
    loge=float(logsumexp(np.r_[logs,adaptive])-np.log(len(logs)+1))
    n=len(gain);v=float(np.var(gain,ddof=1)) if n>1 else 0
    eb=float(np.mean(gain)-np.sqrt(2*v*np.log(2*family/ALPHA)/n)-14*bound*np.log(2*family/ALPHA)/(3*(n-1))) if n>1 else -2*bound
    return {'log_e':loge,'log_threshold':float(np.log(family/ALPHA)),'authorized':bool(loge>=np.log(family/ALPHA)),
            'mean_gain':float(np.mean(gain)),'empirical_bernstein_lcb':eb,'n':n,'bound':bound}

def reliability(name,seed,xt,yt,meta):
    directory=OUT/name;rr=roles(name,len(xt),meta['eligible_audit_ids']);cal=np.flatnonzero(rr=='calibration');gate=ordered_gate(name,rr)
    rec={k:dict(np.load(directory/f'backbones_{seed}'/(k+'.npz'))) for k in ['MiniRocket','Hydra']}
    g=bundle(xt);tf=time_features(xt)
    for kind in rec:
        path=directory/f'{kind}_{seed}';path.mkdir(exist_ok=True)
        if (path/'records.npz').exists():continue
        z=rec[kind]['logits'];h=output_features(z);other=rec['Hydra' if kind=='MiniRocket' else 'MiniRocket']['logits']
        p=softmax(z,1);po=softmax(other,1);c=(z.argmax(1)==yt).astype(int)
        # Only calibration correctness is available to fitting and selection.
        cfit=np.zeros_like(c);cfit[cal]=c[cal]
        ag=np.c_[(z.argmax(1)==other.argmax(1)),po.max(1),(p*po).sum(1)]
        features={'output':h,'time':np.c_[h,tf],'representation':np.c_[h,rec[kind]['representation']],
                  'agreement':np.c_[h,ag],'atypicality':np.c_[h,rec[kind]['distance']],
                  'all_controls':np.c_[h,tf,rec[kind]['representation'],ag,rec[kind]['distance']]}
        cv,scores,models=fit_group(features,cfit,cal)
        # Raw softmax of ridge decisions is explicitly a score, not trained class probabilities.
        cv['raw']=p[cal].max(1);scores['raw']=p.max(1)
        prior=(c[cal].sum()+.5)/(len(cal)+1)
        cv['intercept']=np.full(len(cal),prior);scores['intercept']=np.full(len(c),prior)
        if np.bincount(c[cal],minlength=2).min()>=3:
            for tr,va in StratifiedKFold(3,shuffle=True,random_state=2027).split(np.zeros(len(cal)),c[cal]):
                cv['intercept'][va]=(c[cal[tr]].sum()+.5)/(len(tr)+1)
        loss=lambda a:float(logloss(c[cal],a).mean())
        reference=min(cv,key=lambda k:(loss(cv[k]),k))
        scv,sscores,smodels=fit_group({'spectral':np.c_[h,g],'spectral_controls':np.c_[features['all_controls'],g]},cfit,cal)
        ocv,oscores,omodels=offset_candidates(features,models,reference,scores,g,cfit,cal)
        scv.update(ocv);sscores.update(oscores);smodels.update(omodels)
        candidates=[(loss(bounded_update(cv[reference],pred,b)),key,b) for key,pred in scv.items() for b in [.25,.5,1.,2.]]
        _,chosen,b=min(candidates)
        candidate=bounded_update(scores[reference],sscores[chosen],b)
        # Save frozen selection before reading any gate correctness.
        frozen={'dataset':name,'backbone':kind,'seed':seed,'reference':reference,'candidate':chosen,'bound':b,
                'calibration_reference_nll':loss(cv[reference]),'calibration_candidate_nll':loss(bounded_update(cv[reference],scv[chosen],b)),
                'calibration_n':len(cal),'gate_n':len(gate),'assessment_n':int(np.sum(rr=='assessment')),
                'calibration_errors':int((1-c[cal]).sum()),
                'code_sha256':SOURCE_HASH}
        dump(path/'frozen.json',frozen)
        joblib.dump({'reference':models.get(reference,prior),'candidate':smodels[chosen]},path/'reliability.joblib',compress=3)
        gain=logloss(c[gate],scores[reference][gate])-logloss(c[gate],candidate[gate])
        decision=betting(gain,b)
        dump(path/'gate.json',decision)
        # Store assessment labels without computing metrics; report.py is a separate release step.
        score_names=list(scores)+list(sscores)+['reference','candidate','policy']
        mat=[scores[k] for k in scores]+[sscores[k] for k in sscores]+[scores[reference],candidate,candidate if decision['authorized'] else scores[reference]]
        np.savez_compressed(path/'records.npz',roles=rr,correct=c,label=yt,predicted=z.argmax(1),scores=np.asarray(mat),score_names=np.array(score_names),bundle=g)
        print(name,seed,kind,'frozen; gate log-e',round(decision['log_e'],3),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=DEVELOPMENT);parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS);parser.add_argument('--download-only',action='store_true');args=parser.parse_args()
    for name in args.datasets:
        x,y,xt,yt,meta=load(name);dump(OUT/name/'source.json',meta)
        if args.download_only:print(meta,flush=True);continue
        for seed in args.seeds:
            with threadpool_limits(limits=4):
                fit_backbone(name,seed,x,y,xt);reliability(name,seed,xt,yt,meta)

if __name__=='__main__':main()
