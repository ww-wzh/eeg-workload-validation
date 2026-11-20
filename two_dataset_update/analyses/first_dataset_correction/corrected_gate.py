"""CB-SF with an explicitly signed loss cutoff and unchanged frozen constants."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

FEATURES=['feature_shift','disagreement','entropy_nn','entropy_fuse','confidence_nn']


def loss_labels(gains, loss_cutoff=-0.05):
    if not np.isfinite(loss_cutoff) or not -1 <= loss_cutoff < 0:
        raise ValueError('loss_cutoff must be an explicitly negative gain, e.g. -0.05')
    values=np.asarray(gains,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite gain')
    return (values<=loss_cutoff).astype(np.int64)


def fit_gate(training,target,*,loss_cutoff=-0.05):
    # Target API accepts only the five label-free diagnostics.
    if set(target)!=set(FEATURES):
        raise ValueError('Target must contain exactly five label-free diagnostics')
    x=training[FEATURES].to_numpy(dtype=np.float64)
    gains=training.benefit.to_numpy(dtype=np.float64)
    t=np.array([[target[k] for k in FEATURES]],dtype=np.float64)
    if len(x)<5 or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError('Invalid gate inputs')
    event=loss_labels(gains,loss_cutoff)
    center=np.median(x,axis=0)
    q1,q3=np.quantile(x,[.25,.75],axis=0)
    scale=np.where(q3-q1>=1e-8,q3-q1,np.std(x,axis=0,ddof=0))
    scale=np.where(scale>=1e-8,scale,1.)
    tz=(t-center)/scale
    max_z=float(np.abs(tz).max())
    ood=bool(max_z>10.)
    xz=np.clip((x-center)/scale,-5.,5.)
    tz=np.clip(tz,-5.,5.)
    benefit_model=Ridge(alpha=1.).fit(xz,gains)
    raw=float(benefit_model.predict(tz)[0])
    benefit=float(np.clip(raw,max(-1.,float(gains.min())),min(1.,float(gains.max()))))
    fitted=len(np.unique(event))==2
    if fitted:
        model=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=0).fit(xz,event)
        score=float(model.predict_proba(tz)[0,1])
    else:
        score=float(event.mean())
    weight=0. if ood else .8/(1+np.exp(-np.clip((benefit-.05*score)/.02,-40.,40.)))
    return {'weight':float(np.clip(weight,0,.8)),'predicted_benefit':benefit,
        'predicted_benefit_raw':raw,'predicted_catastrophe':score,
        'ood_fallback':ood,'max_abs_robust_z':max_z,'training_rows':len(training),
        'training_subjects':int(training.meta_subject.nunique()),'loss_events':int(event.sum()),
        'non_loss_events':int(len(event)-event.sum()),'event_model_fitted':fitted,
        'loss_cutoff':float(loss_cutoff)}


def exact_sign_flip(values):
    """Two-sided tail, including all equally OR more extreme signed sums."""
    values=np.asarray(values,dtype=float)
    nz=values[np.abs(values)>1e-12]
    if not len(nz):
        return 1.
    patterns=np.arange(1<<len(nz),dtype=np.uint32)[:,None]
    signs=2*((patterns>>np.arange(len(nz)))&1).astype(np.int8)-1
    null=signs@nz
    return float(np.mean(np.abs(null)>=abs(nz.sum())-1e-12))
