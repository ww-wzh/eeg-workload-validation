"""Repair first-dataset CB-SF postprocessing, preserving all archived sources.

No EEG branch training or remote writes. All source files are hashed and checked
again at completion. Each metric retains its original ordering and BH family.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from corrected_gate import FEATURES,fit_gate,loss_labels,exact_sign_flip

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'shared/first_dataset'
NEUTRAL=BASE/'neutral_predictions'
ORD=BASE/'heldout_gain_inputs'
BAL=HERE.parent/'first_dataset_balanced_gain_inputs'
META=BASE/'loso_gain_inputs'
SEEDS=[1335,1388,1441,1494,1547]
SOURCES={}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pin(p):
    SOURCES[str(p.relative_to(ROOT))]=sha(p)
    return p


def read(p):
    return pd.read_csv(pin(p),float_precision='round_trip')


def load_module(path,name):
    pin(path)
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save(frame,p):
    frame.to_csv(p,index=False,lineterminator='\n')


h=load_module(BASE/'heldout_gain_helpers.py','legacy_heldout_helpers')
pin(ROOT/'shared/revision_pipeline/risk_gate.py')


def compare_records(a,b):
    keys=['protocol','direction','subject','recording_id','method']
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_array_equal(a.pred_label,b.pred_label)
    e=float(np.abs(a[['p0','p1']].to_numpy()-b[['p0','p1']].to_numpy()).max())
    assert e<1e-10,e
    return e


def replay_original(core):
    folder=NEUTRAL/'04_cbsf'
    config=json.loads(pin(BASE/'seed_aggregation_protocol.json').read_text(encoding='utf-8'))
    params=config['cbsf_stability']['parameters']
    assert params['catastrophe_threshold']==.05
    weights=read(folder/'gate_weights.csv');targets=read(folder/'target_diagnostics.csv')
    tables={k:read(folder/f'gate_training_rows_{suffix}.csv') for k,suffix in
            [('cross_task','cross_task'),('arithmetic','loso_arithmetic'),('stroop','loso_stroop')]}
    rows=[];windows=[]
    for row in targets.itertuples():
        if row.protocol=='cross_task':
            tr=tables['cross_task'].query('subject != @row.subject').copy()
            tr['meta_subject']=tr.subject.astype(int)
            assert len(tr)==24 and tr.meta_subject.nunique()==12
        else:
            tr=tables[row.direction].query('outer_subject == @row.subject').copy()
        target={k:float(getattr(row,k)) for k in FEATURES}
        gate=fit_gate(tr,target)
        old=weights[(weights.protocol==row.protocol)&(weights.subject==row.subject)&(weights.direction==row.direction)]
        assert len(old)==1
        err=max(abs(gate[k]-float(old.iloc[0][k])) for k in ['weight','predicted_catastrophe','predicted_benefit'])
        assert err<1e-10,err
        windows.append(h.make_rows(h.target_window(core,row.protocol,row.subject,row.direction),'cbsf',weight=gate['weight']))
        rows.append({'protocol':row.protocol,'direction':row.direction,'subject':row.subject,
                     'max_replay_error':err,**gate})
    rec=h.recording_scores(pd.concat(windows,ignore_index=True))
    error=compare_records(rec,read(folder/'recording_predictions.csv'))
    save(pd.DataFrame(rows),HERE/'original_proxy_gate_replay.csv')
    return {'cells':len(rows),'recordings':len(rec),'threshold_argument':.05,
        'max_weight_event_benefit_error':max(r['max_replay_error'] for r in rows),
        'max_record_probability_error':error,'class_predictions_unchanged':True,
        'primary_BH65_predictions_unchanged':True}


def metric(frame,balanced):
    assert set(frame.true_label)=={0,1}
    return float(frame.groupby('true_label').correct.mean().mean() if balanced else frame.correct.mean())


def seed_response_tables(meta):
    cross=[]
    for subject in range(1,14):
        for direction in ['arithmetic_to_stroop','stroop_to_arithmetic']:
            for seed in SEEDS:
                path=NEUTRAL/'01_core/cross_task/checkpoints'/f's{subject:02d}_{direction}_seed{seed}.csv'
                raw=read(path);raw=raw[raw.method.isin(['always_nn','always_fuse'])]
                rr=h.recording_scores(raw)
                for balanced in [False,True]:
                    gain=metric(rr[rr.method=='always_fuse'],balanced)-metric(rr[rr.method=='always_nn'],balanced)
                    cross.append({'response_metric':'balanced' if balanced else 'ordinary',
                        'subject':subject,'direction':direction,'seed':seed,'benefit':gain})
    loso=[]
    for (task,outer,subject,seed),rr in meta[meta.protocol=='loso_meta_pair'].groupby(['direction','outer_subject','meta_subject','seed'],sort=True):
        for balanced in [False,True]:
            gain=metric(rr[rr.method=='always_fuse'],balanced)-metric(rr[rr.method=='always_nn'],balanced)
            loso.append({'response_metric':'balanced' if balanced else 'ordinary','task':task,
                'outer_subject':int(outer),'meta_subject':int(subject),'seed':int(seed),'benefit':gain})
    return pd.DataFrame(loso),pd.DataFrame(cross)


def reconstruct(variant,core,seed_loso,seed_cross):
    balanced=variant=='balanced'
    out=HERE/(variant+'_heldout');out.mkdir(exist_ok=True)
    legacy_diag=read(BAL/'balanced_gain_gate_diagnostics.csv' if balanced else ORD/'heldout_gate_diagnostics.csv')
    legacy_rec=read(BAL/'balanced_gain_recording_predictions.csv' if balanced else ORD/'heldout_gate_recording_predictions.csv')
    loso=read(BAL/'balanced_gain_training_rows_loso.csv' if balanced else META/'04_all_meta_median_seed.csv')
    cross=read(BAL/'balanced_gain_training_rows_cross_task.csv' if balanced else ORD/'cross_task_cbsf_training_rows.csv')
    sl=seed_loso[seed_loso.response_metric==variant]
    sc=seed_cross[seed_cross.response_metric==variant]
    assert not sl.duplicated(['task','outer_subject','meta_subject','seed']).any()
    assert not sc.duplicated(['subject','direction','seed']).any()
    gl=sl.groupby(['task','outer_subject','meta_subject']).benefit.median()
    gc=sc.groupby(['subject','direction']).benefit.median()
    for r in loso.itertuples():assert abs(r.benefit-gl.loc[(r.task,r.outer_subject,r.meta_subject)])<1e-12
    for r in cross.itertuples():assert abs(r.benefit-gc.loc[(r.subject,r.direction)])<1e-12
    for table in [loso,cross]:table['loss_event']=loss_labels(table.benefit).astype(bool)
    save(loso,out/'training_rows_loso.csv');save(cross,out/'training_rows_cross_task.csv')
    save(sl,out/'loso_seed_responses.csv');save(sc,out/'cross_task_seed_responses.csv')
    windows=[];old_windows=[];diagnostics=[];seed_rows=[];deletions=[];seed_records=[]
    for _,row in legacy_diag.iterrows():
        is_cross=row.setting=='cross_task'
        direction=str(row.direction) if is_cross else ('arithmetic' if 'arithmetic' in row.setting else 'stroop')
        protocol='cross_task' if is_cross else 'loso'
        subject=int(row.subject)
        selected=h.target_window(core,protocol,subject,direction)
        base={'method':row.method,'setting':row.setting,'protocol':protocol,'direction':direction,'subject':subject}
        if str(row.method).startswith('dasf'):
            w=h.make_rows(selected,row.method,select_fuse=bool(row.selected_fusion))
            windows.append(w);old_windows.append(w)
            diagnostics.append({**row.to_dict(),**base});continue
        tr=cross[cross.subject!=subject].copy() if is_cross else loso[(loso.task==direction)&(loso.outer_subject==subject)].copy()
        assert subject not in set(tr.meta_subject)
        assert len(tr)==(24 if is_cross else (14 if direction=='arithmetic' else 12))
        target={k:float(row[k]) for k in FEATURES}
        check=h.diagnostics(selected,target['feature_shift'])
        for k in FEATURES:assert abs(check[k]-target[k])<1e-12
        gate=fit_gate(tr,target)
        pos=h.fit_predict_gate_robust(tr,pd.Series(target),catastrophe_threshold=.05,
            benefit_alpha=1.,catastrophe_c=1.,risk_penalty=.05,temperature=.02,max_weight=.8)
        for k in ['weight','predicted_benefit','predicted_catastrophe']:assert abs(gate[k]-getattr(pos,k))<1e-12
        old=h.fit_predict_gate_robust(tr,pd.Series(target),catastrophe_threshold=-.05,
            benefit_alpha=1.,catastrophe_c=1.,risk_penalty=.05,temperature=.02,max_weight=.8)
        assert abs(old.weight-row.weight)<1e-12
        old_windows.append(h.make_rows(selected,row.method,weight=old.weight))
        windows.append(h.make_rows(selected,row.method,weight=gate['weight']))
        diagnostics.append({**base,**target,**gate,'legacy_weight':float(row.weight)})
        # Gain-only sensitivity: keep median diagnostic rows and outer ensemble fixed.
        for seed in SEEDS:
            ts=tr.copy()
            lookup=(sc[sc.seed==seed].set_index(['subject','direction']).benefit if is_cross else
                    sl[(sl.seed==seed)&(sl.task==direction)&(sl.outer_subject==subject)].set_index('meta_subject').benefit)
            ts['benefit']=[lookup.loc[(int(r.subject),r.direction)] if is_cross else lookup.loc[int(r.meta_subject)] for r in ts.itertuples()]
            sg=fit_gate(ts,target)
            seed_rows.append({**base,'seed':seed,**sg})
            sr=h.recording_scores(h.make_rows(selected,row.method,weight=sg['weight']))
            sr['seed']=seed;seed_records.append(sr)
        for deleted in sorted(tr.meta_subject.unique()):
            reduced=tr[tr.meta_subject!=deleted].copy()
            assert len(reduced)==len(tr)-(2 if is_cross else 1)
            assert int(deleted) not in set(reduced.meta_subject)
            if is_cross:
                assert len(reduced)==22 and reduced.meta_subject.nunique()==11
                assert (reduced.groupby('meta_subject').direction.nunique()==2).all()
            dg=fit_gate(reduced,target)
            deletions.append({**base,'deleted_meta_subject':int(deleted),
                'full_weight':gate['weight'],'deletion_weight':dg['weight'],
                'weight_delta':dg['weight']-gate['weight'],
                'full_training_rows':len(tr),'deletion_training_rows':len(reduced),
                'full_training_subjects':tr.meta_subject.nunique(),'deletion_training_subjects':reduced.meta_subject.nunique(),
                'full_ood_fallback':gate['ood_fallback'],'deletion_ood_fallback':dg['ood_fallback'],
                'full_predicted_catastrophe':gate['predicted_catastrophe'],'deletion_predicted_catastrophe':dg['predicted_catastrophe'],
                'full_loss_events':gate['loss_events'],'deletion_loss_events':dg['loss_events']})
    old_error=compare_records(h.recording_scores(pd.concat(old_windows,ignore_index=True)),legacy_rec)
    window=pd.concat(windows,ignore_index=True);rec=h.recording_scores(window)
    assert len(window)==50332 and len(rec)==302 and len(seed_rows)==270 and len(deletions)==678
    d=pd.DataFrame(diagnostics);se=pd.DataFrame(seed_rows);de=pd.DataFrame(deletions)
    save(window,out/'window_predictions.csv');save(rec,out/'recording_predictions.csv');save(d,out/'gate_diagnostics.csv')
    save(se,out/'seed_sensitivity.csv');save(de,out/'participant_deletion.csv')
    save(pd.concat(seed_records,ignore_index=True),out/'seed_sensitivity_recording_predictions.csv')
    join=rec.merge(legacy_rec,on=['protocol','direction','subject','recording_id','method'],suffixes=('_new','_old'),validate='one_to_one')
    cb=d[d.method.str.startswith('cbsf')]
    summary={'legacy_record_probability_replay_max_error':old_error,'gate_cells':len(d),'cbsf_cells':len(cb),
        'changed_cbsf_weights':int((abs(cb.weight-cb.legacy_weight)>1e-12).sum()),
        'changed_record_class_predictions':int((join.pred_label_new!=join.pred_label_old).sum()),
        'seed_sensitivity_rows':len(se),'participant_deletion_rows':len(de),
        'loss_events_loso_candidate_rows':int(loso.loss_event.sum()),'loso_candidate_rows':len(loso),
        'loss_events_cross_candidate_rows':int(cross.loss_event.sum()),'cross_candidate_rows':len(cross)}
    dump(out/'reconstruction_checks.json',summary)
    print(variant,json.dumps(summary),flush=True)
    return summary


def statistics():
    stat=load_module(BASE/'compute_heldout_statistics.py','heldout_statistics')
    stat.NEW_PRED=HERE/'ordinary_heldout/recording_predictions.csv';stat.OUT=HERE/'ordinary_heldout/statistics'
    for p in [stat.CORE,stat.OLD_DASF,stat.OLD_CBSF]:pin(p)
    stat.main()
    metrics=load_module(BASE/'compute_gate_metrics.py','heldout_metrics')
    for variant in ['ordinary','balanced']:
        out=HERE/(variant+'_heldout')
        metrics.PRED=out/'recording_predictions.csv';metrics.DIAG=out/'gate_diagnostics.csv'
        metrics.TRAIN=out/'training_rows_cross_task.csv';metrics.OUT=out/'metrics';metrics.main()
    unified=load_module(BASE/'compute_bh79.py','heldout_bh79')
    unified.LEGACY=BASE/'primary_statistics/participant_statistics/primary_comparisons_BH65.csv'
    pin(unified.LEGACY)
    unified.NEW=stat.OUT/'comparisons_BH14.csv';unified.OUT=HERE/'ordinary_heldout/BH79';unified.main()
    balanced=load_module(BAL/'compute_balanced_gain_stats.py','balanced_statistics')
    out=HERE/'balanced_heldout'
    values=balanced.participant_metrics(pd.read_csv(out/'recording_predictions.csv',float_precision='round_trip'))
    save(values,out/'participant_metrics.csv')
    balanced.OUT=out/'statistics';balanced.OUT.mkdir(exist_ok=True)
    balanced.GATE=out/'participant_metrics.csv';balanced.exact_sign_flip=exact_sign_flip
    balanced.main()


def self_checks():
    assert loss_labels([-.1,-.0500001,-.05,-.0499999,0.,.05]).tolist()==[1,1,1,0,0,0]
    try:loss_labels([0.],.05)
    except ValueError:pass
    else:raise AssertionError('Positive cutoff accepted')
    assert exact_sign_flip([1.,-1.])==1.
    assert exact_sign_flip([1.,1.])==.5
    assert exact_sign_flip([1.,1.,-1.])==1.
    assert exact_sign_flip([0.,0.])==1.
    assert exact_sign_flip([1.,1.,1.])==.25


def main():
    self_checks()
    pin(Path(__file__));pin(HERE/'corrected_gate.py')
    policy=json.loads(pin(BASE/'gate_protocol/nested_gate_protocol.json').read_text(encoding='utf-8'))
    assert policy['gate_rules']['cbsf']['event_threshold']==-.05
    core=read(NEUTRAL/'01_core/core_window_ensemble.csv')
    meta=read(META/'02_all_meta_recording_seed.csv')
    original=replay_original(core)
    print('Original proxy CB-SF replay passed',original,flush=True)
    sl,sc=seed_response_tables(meta)
    checks={variant:reconstruct(variant,core,sl,sc) for variant in ['ordinary','balanced']}
    statistics()
    for name,expected in SOURCES.items():assert sha(ROOT/name)==expected,name
    report={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
        'corrections':['signed loss cutoff','participant deletion of both cross-task directions',
                       'direction-aware seed response keys','two-sided sign-flip tail includes more extreme sums'],
        'original_proxy_replay':original,'heldout_checks':checks,'source_files_sha256':SOURCES,
        'EEG_base_models_retrained':0,'historical_sources_unchanged':True,
        'seed_scope':'Gain-only sensitivity with median diagnostics and final outer ensemble fixed',
        'deletion_scope':'Participant deletion, removing both directions in cross-task fitting'}
    dump(HERE/'correction_verification.json',report)
    print('Correction and statistics complete; source hashes unchanged.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
