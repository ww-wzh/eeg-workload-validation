"""Reproduce historical CB-SF outputs and isolate a signed-threshold bug.

Read-only toward historical files. Diagnostic replays fit small historical gates,
not EEG branches or new COG models. Corrected numbers here are impact diagnostics,
not final corrected estimates or revised multiple-comparison tests.
"""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'shared/first_dataset'
BAL=ROOT/'analyses/first_dataset_balanced_gain_inputs'
ORD=BASE/'heldout_gain_inputs'
META=BASE/'loso_gain_inputs'
CORE=BASE/'neutral_predictions/01_core/core_window_ensemble.csv'
sys.path.insert(0,str(ROOT/'shared'))
from revision_pipeline.risk_gate import fit_predict_gate_robust,GATE_FEATURES

def read(p):return pd.read_csv(p,float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fit(tr,target,argument):
    return fit_predict_gate_robust(tr,target,catastrophe_threshold=argument,benefit_alpha=1,
                                  catastrophe_c=1,risk_penalty=.05,temperature=.02,max_weight=.8)

def aggregate(frame,weight):
    keys=['protocol','subject','direction','recording_id','window_id','true_label']
    nn=frame[frame.method=='always_nn'];fu=frame[frame.method=='always_fuse']
    m=nn[keys+['p0','p1']].merge(fu[keys+['p0','p1']],on=keys,suffixes=('_nn','_fu'),validate='one_to_one')
    p=(1-weight)*m[['p0_nn','p1_nn']].to_numpy()+weight*m[['p0_fu','p1_fu']].to_numpy()
    p/=p.sum(axis=1,keepdims=True)
    m=m[keys].copy();m['p0']=p[:,0];m['p1']=p[:,1]
    rows=[]
    for rid,g in m.groupby('recording_id',sort=True):
        v=np.exp(np.log(np.clip(g[['p0','p1']].to_numpy(),1e-12,1)).mean(axis=0));v/=v.sum()
        y=int(g.true_label.iloc[0]);pred=int(v.argmax())
        rows.append({'recording_id':rid,'true_label':y,'p0':float(v[0]),'p1':float(v[1]),'pred_label':pred,'correct':int(pred==y)})
    return pd.DataFrame(rows)

def ba(frame):return float(np.mean([frame[frame.true_label==c].correct.mean() for c in [0,1]]))

def main():
    files=[CORE,ORD/'heldout_gate_diagnostics.csv',ORD/'heldout_gate_recording_predictions.csv',
           ORD/'cross_task_cbsf_training_rows.csv',META/'04_all_meta_median_seed.csv',
           BAL/'balanced_gain_gate_diagnostics.csv',BAL/'balanced_gain_recording_predictions.csv',
           BAL/'balanced_gain_training_rows_loso.csv',BAL/'balanced_gain_training_rows_cross_task.csv',
           BAL/'balanced_gain_participant_deletion.csv',BAL/'balanced_gain_cross_task_seed_responses.csv',
           BAL/'run_balanced_gain_sensitivity.py',BASE/'heldout_gain_helpers.py',
           BASE/'gate_protocol/nested_gate_protocol.json',
           ROOT/'shared/revision_pipeline/risk_gate.py']
    pinned={str(p.relative_to(ROOT)):sha(p) for p in files}
    core=read(CORE)
    variants={
        'ordinary_heldout':(read(ORD/'heldout_gate_diagnostics.csv'),read(META/'04_all_meta_median_seed.csv'),read(ORD/'cross_task_cbsf_training_rows.csv'),read(ORD/'heldout_gate_recording_predictions.csv')),
        'balanced_heldout':(read(BAL/'balanced_gain_gate_diagnostics.csv'),read(BAL/'balanced_gain_training_rows_loso.csv'),read(BAL/'balanced_gain_training_rows_cross_task.csv'),read(BAL/'balanced_gain_recording_predictions.csv'))}
    replay=[];predictions=[];metrics=[]
    for variant,(diagnostics,loso,cross,old_records) in variants.items():
        for _,target in diagnostics[diagnostics.method.str.startswith('cbsf')].iterrows():
            subject=int(target.subject)
            is_cross=target.setting=='cross_task'
            task=str(target.direction) if is_cross else ('arithmetic' if 'arithmetic' in target.setting else 'stroop')
            tr=cross[cross.subject!=subject].copy() if is_cross else loso[(loso.task==task)&(loso.outer_subject==subject)].copy()
            old=fit(tr,target,-.05);correct=fit(tr,target,.05)
            e1=abs(old.weight-float(target.weight));e2=abs(old.predicted_catastrophe-float(target.predicted_catastrophe))
            assert max(e1,e2)<1e-11,(variant,subject,task,e1,e2)
            selected=core[(core.protocol==('cross_task' if is_cross else 'loso'))&(core.subject==subject)&(core.direction==task)]
            rec_old=aggregate(selected,old.weight);rec_correct=aggregate(selected,correct.weight);rec_nn=aggregate(selected,0)
            saved=old_records[(old_records.method==target.method)&(old_records.subject==subject)&(old_records.direction==task)]
            cmp=rec_old.merge(saved,on='recording_id',suffixes=('_replayed','_saved'),validate='one_to_one')
            assert len(cmp)==len(rec_old)>0
            err=float(abs(cmp.p1_replayed-cmp.p1_saved).max())
            assert err<1e-11 and (cmp.pred_label_replayed==cmp.pred_label_saved).all()
            replay.append({'analysis':variant,'setting':task,'subject':subject,'rows':len(tr),
                           'declared_loss_events':int((tr.benefit<=-.05).sum()),'actual_fit_events':int((tr.benefit<=.05).sum()),
                           'saved_weight':float(target.weight),'replayed_bug_weight':old.weight,'correct_cutoff_weight':correct.weight,
                           'weight_change':correct.weight-old.weight,'saved_event_score':float(target.predicted_catastrophe),
                           'correct_cutoff_event_score':correct.predicted_catastrophe,'saved_weight_replay_error':e1,
                           'saved_event_score_replay_error':e2,'saved_record_probability_replay_error':err,
                           'record_predictions_changed':int((rec_old.pred_label.to_numpy()!=rec_correct.pred_label.to_numpy()).sum())})
            metrics.append({'analysis':variant,'setting':task,'subject':subject,'historical_ba':ba(rec_old),
                            'sign_corrected_ba':ba(rec_correct),'reference_mlp_ba':ba(rec_nn)})
            for version,recs in [('historical_replayed',rec_old),('sign_corrected_diagnostic',rec_correct)]:
                predictions.extend({'analysis':variant,'setting':task,'subject':subject,'version':version,**r} for r in recs.to_dict('records'))
    replay=pd.DataFrame(replay);metrics=pd.DataFrame(metrics)
    replay.to_csv(HERE/'threshold_replay_by_gate.csv',index=False)
    metrics.to_csv(HERE/'threshold_impact_by_participant.csv',index=False)
    pd.DataFrame(predictions).to_csv(HERE/'threshold_impact_record_predictions.csv',index=False)
    summary=metrics.groupby(['analysis','setting'])[['historical_ba','sign_corrected_ba','reference_mlp_ba']].mean().reset_index()
    summary.to_csv(HERE/'threshold_impact_summary.csv',index=False)
    # Historical cross-task deletion used only one direction, unlike the 24-row full gate.
    deletion=read(BAL/'balanced_gain_participant_deletion.csv')
    dd=variants['balanced_heldout'][0]
    cross_del=deletion[deletion.setting=='cross_task']
    lookup=dd[dd.setting=='cross_task'][['subject','direction','weight']]
    joined=cross_del.merge(lookup,on=['subject','direction'],validate='many_to_one')
    full_gap=abs(joined.full_weight-joined.weight)
    # Historical seed code keyed gains by subject only and current target direction.
    seed_rows=read(BAL/'balanced_gain_cross_task_seed_responses.csv')
    pivot=seed_rows.pivot(index=['subject','seed'],columns='direction',values='benefit')
    differing=int((abs(pivot.arithmetic_to_stroop-pivot.stroop_to_arithmetic)>1e-12).sum())
    followup={'participant_deletion':{'actual_scope':'12 other participants in target direction only; omitting one yields11 rows',
              'main_gate_scope':'24 rows from12 other participants in BOTH directions; omitting one should yield22 rows',
              'cross_task_deletion_rows':len(joined),'rows_where_reported_full_weight_differs_from_main_gate':int((full_gap>1e-10).sum()),'max_full_weight_discrepancy':float(full_gap.max())},
              'seed_sensitivity':{'code_issue':'Both-direction training rows receive gain from current target direction through a subject-only lookup.',
              'subject_seed_pairs_with_direction_dependent_gain':differing,'total_subject_seed_pairs':len(pivot)},
              'action':'Rebuild these sensitivity outputs with direction-aware response keys and participant-wise removal of BOTH directions; not corrected by changing event sign alone.'}
    dump(HERE/'sensitivity_scope_audit.json',followup)
    for name,h in pinned.items():assert sha(ROOT/name)==h
    report={'completed_utc':datetime.now(timezone.utc).isoformat(),
            'confirmed_issue':'Historical config cutoff -0.05 is passed to API that internally negates argument; actual loss event was gain<=+0.05.',
            'replayed_existing_gate_cells':len(replay),'replayed_analysis_variants':list(variants),
            'max_saved_weight_replay_error':float(replay.saved_weight_replay_error.max()),
            'max_saved_record_probability_replay_error':float(replay.saved_record_probability_replay_error.max()),
            'gates_with_changed_weight_under_intended_cutoff':int((abs(replay.weight_change)>1e-12).sum()),
            'changed_record_predictions_across_both_analyses':int(replay.record_predictions_changed.sum()),
            'EEG_base_models_retrained':0,'COG_models_fitted':0,'historical_files_unchanged':True,
            'corrected_outputs_role':'Historical sign-only diagnostic. The complete corrected analysis is provided in the first_dataset_correction directory.',
            'unaffected_by_this_bug':'Fixed Logistic split/session experiments and DASF decision formula do not call the CB-SF event model.',
            'scope_limit':'Audited heldout ordinary/balanced reconstructions and balanced sensitivities; original internal-proxy method not yet globally replayed.',
            'source_sha256':pinned}
    dump(HERE/'audit_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
    print(summary.to_string(index=False));print(json.dumps(followup,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
