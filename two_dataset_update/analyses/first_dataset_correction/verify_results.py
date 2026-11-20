"""Check repaired results through independent aggregation and legacy gate replay."""
from pathlib import Path
import hashlib
import csv
import itertools
import json
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'shared'))
from revision_pipeline.risk_gate import fit_predict_gate_robust,GATE_FEATURES

def read(p):return pd.read_csv(p,float_precision='round_trip')
def fit(tr,target):
    return fit_predict_gate_robust(tr,target,catastrophe_threshold=.05,benefit_alpha=1.,
        catastrophe_c=1.,risk_penalty=.05,temperature=.02,max_weight=.8)
def exact(d):
    sums=[sum(a*b for a,b in zip(signs,d)) for signs in itertools.product([-1,1],repeat=len(d))]
    return float(np.mean(np.abs(sums)>=abs(sum(d))-1e-12))

def main():
    report=json.loads((HERE/'correction_verification.json').read_text(encoding='utf-8'))
    with (ROOT/'SOURCE_INDEX.csv').open(encoding='utf-8',newline='') as stream:
        index={r['publication_path']:r for r in csv.DictReader(stream)}
    for name,h in report['source_files_sha256'].items():
        name=name.replace('\\','/')
        assert index[name]['original_sha256']==h,name
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==index[name]['publication_sha256'],name
    summaries={};count_seed=0;count_deletion=0
    for variant in ['ordinary','balanced']:
        folder=HERE/(variant+'_heldout');windows=read(folder/'window_predictions.csv');rec=read(folder/'recording_predictions.csv')
        for key,g in windows.groupby(['protocol','direction','subject','recording_id','method']):
            p=np.exp(np.log(np.clip(g[['p0','p1']].to_numpy(),1e-12,1)).mean(axis=0));p/=p.sum()
            rr=rec
            for col,value in zip(['protocol','direction','subject','recording_id','method'],key):rr=rr[rr[col]==value]
            assert len(rr)==1
            np.testing.assert_allclose(p,rr[['p0','p1']].iloc[0].to_numpy(),rtol=0,atol=1e-12)
            assert int(p.argmax())==int(rr.pred_label.iloc[0])
        diag=read(folder/'gate_diagnostics.csv');diag=diag[diag.method.str.startswith('cbsf')]
        loso=read(folder/'training_rows_loso.csv');cross=read(folder/'training_rows_cross_task.csv')
        sl=read(folder/'loso_seed_responses.csv');sc=read(folder/'cross_task_seed_responses.csv')
        seeds=read(folder/'seed_sensitivity.csv');deletion=read(folder/'participant_deletion.csv')
        for target in diag.itertuples():
            is_cross=target.protocol=='cross_task';subject=int(target.subject)
            tr=(cross[cross.subject!=subject] if is_cross else loso[(loso.task==target.direction)&(loso.outer_subject==subject)]).copy()
            x=pd.Series({k:getattr(target,k) for k in GATE_FEATURES})
            full=fit(tr,x)
            assert abs(full.weight-target.weight)<1e-12
            se=seeds[(seeds.subject==subject)&(seeds.protocol==target.protocol)&(seeds.direction==target.direction)]
            for r in se.itertuples():
                if is_cross:
                    ts=tr.drop(columns='benefit').merge(sc[sc.seed==r.seed][['subject','direction','benefit']],on=['subject','direction'],validate='one_to_one',sort=False)
                else:
                    response=sl[(sl.seed==r.seed)&(sl.task==target.direction)&(sl.outer_subject==subject)]
                    ts=tr.drop(columns='benefit').merge(response[['meta_subject','benefit']],on='meta_subject',validate='one_to_one',sort=False)
                assert len(ts)==len(tr) and not ts.benefit.isna().any()
                result=fit(ts,x)
                assert abs(result.weight-r.weight)<1e-12
                assert abs(result.predicted_catastrophe-r.predicted_catastrophe)<1e-12
                count_seed+=1
            dd=deletion[(deletion.subject==subject)&(deletion.protocol==target.protocol)&(deletion.direction==target.direction)]
            for r in dd.itertuples():
                assert abs(r.full_weight-full.weight)<1e-12
                reduced=tr[tr.meta_subject!=r.deleted_meta_subject]
                assert len(reduced)==r.deletion_training_rows
                if is_cross:assert len(reduced)==22 and reduced.meta_subject.nunique()==11
                result=fit(reduced,x)
                assert abs(result.weight-r.deletion_weight)<1e-12
                count_deletion+=1
        seed_range=seeds.groupby(['protocol','direction','subject']).weight.agg(['min','max']).reset_index()
        seed_range['range']=seed_range['max']-seed_range['min']
        seed_range.to_csv(folder/'seed_weight_ranges.csv',index=False)
        ds=[]
        for (protocol,direction),g in deletion.groupby(['protocol','direction']):
            sr=seed_range[(seed_range.protocol==protocol)&(seed_range.direction==direction)]
            ds.append({'protocol':protocol,'direction':direction,'deletion_refits':len(g),
                'deletion_changes_over_point1':int((abs(g.weight_delta)>.1).sum()),
                'seed_cells':len(sr),'seed_ranges_over_point5':int((sr['range']>.5).sum()),
                'min_weight':float(sr['min'].min()),'max_weight':float(sr['max'].max())})
        pd.DataFrame(ds).to_csv(folder/'sensitivity_summary.csv',index=False)
        summaries[variant]=ds
    # Independent exact enumeration retains zero differences and includes tail.
    vals=read(HERE/'ordinary_heldout/statistics/participant_setting_metrics.csv')
    comp=read(HERE/'ordinary_heldout/statistics/comparisons_BH14.csv')
    for r in comp.itertuples():
        a=vals[(vals.setting==r.setting)&(vals.method==r.method)].set_index('subject').recording_balanced_accuracy
        b=vals[(vals.setting==r.setting)&(vals.method==r.reference_method)].set_index('subject').recording_balanced_accuracy
        d=(a-b).sort_index().to_numpy()
        assert len(d)==r.n_participants and abs(d.mean()*100-r.mean_difference_pp)<1e-12
        assert exact(d)==r.exact_sign_flip_p
    np.testing.assert_allclose(multipletests(comp.exact_sign_flip_p,method='fdr_bh')[1],comp.bh_adjusted_p_14_family,rtol=0,atol=1e-12)
    bal=read(HERE/'balanced_heldout/statistics/balanced_gain_method_comparisons_BH7.csv')
    np.testing.assert_allclose(multipletests(bal.exact_sign_flip_p,method='fdr_bh')[1],bal.bh_q_7_family,rtol=0,atol=1e-12)
    bvals=read(HERE/'balanced_heldout/participant_metrics.csv')
    for r in bal.itertuples():
        if r.setting=='cross_task_bidirectional':
            actual=bvals[(bvals.protocol=='cross_task')&(bvals.method==r.method)].groupby('subject').balanced_accuracy.mean()
        else:
            actual=bvals[(bvals.protocol+'_'+bvals.direction==r.setting)&(bvals.method==r.method)].set_index('subject').balanced_accuracy
        ref=vals[(vals.setting==r.setting)&(vals.method=='feature_mlp')].set_index('subject').recording_balanced_accuracy
        d=(actual-ref).sort_index().to_numpy()
        assert abs(d.mean()*100-r.mean_difference_pp)<1e-12 and exact(d)==r.exact_sign_flip_p
    result={'status':'passed','seed_refits_independently_replayed':count_seed,
        'participant_deletion_refits_independently_replayed':count_deletion,'recording_aggregation_verified':604,
        'exact_tests_verified':21,'BH14_and_BH7_verified_with_statsmodels':True,
        'sources_unchanged':True,'sensitivity_summary':summaries}
    (HERE/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
