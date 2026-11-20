"""Independent checks of raw decoding, saved models, split membership and metrics."""
from __future__ import annotations
import argparse
import json
import tempfile
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from audit_inputs import HERE, ARCHIVES, CHANNELS, dump, sha

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def preflight():
    import mne
    from run_validation import preprocess_support, score
    inventory=load(HERE/'input_inventory.json')['records']
    checks=[]
    for subject,stem in [(1,'MATBeasy'),(10,'zeroBACK')]:
        r=next(x for x in inventory if x['subject']==subject and x['session']==1 and x['stem']==stem)
        with zipfile.ZipFile(ARCHIVES/f'sub-{subject:02d}.zip') as z:
            raw=z.read(r['fdt_member'])
            native=np.frombuffer(raw,dtype='<f4').reshape(r['pnts'],r['nbchan'])[:,r['selected_channel_indices']]
            with tempfile.TemporaryDirectory(prefix='raw_decoder_check_',dir=HERE) as td:
                base=Path(td)
                (base/f'{stem}.set').write_bytes(z.read(r['set_member']))
                (base/f'{stem}.fdt').write_bytes(raw)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    reference=mne.io.read_raw_eeglab(base/f'{stem}.set',preload=False,verbose='ERROR')
                sample=reference.get_data(picks=CHANNELS,start=0,stop=10000).T*1e6
                diff=float(np.max(abs(sample-native[:10000])))
                assert np.allclose(sample,native[:10000],atol=1e-7,rtol=1e-12)
            a,b=r['raw_supports'][0]
            feature=preprocess_support(native[a:b])
            assert feature.shape==(272,) and np.isfinite(feature).all()
            checks.append({'record_id':r['record_id'],'mne_max_abs_difference_native_units':diff,
                           'original_channels':r['nbchan'],'feature_shape':list(feature.shape)})
    # The endpoint aggregates probabilities only within each held-out class record.
    p=np.array([[.9,.1]]*10+[[.1,.9]]*10)
    assert score(p)[0]==1.0 and score(p)[1]==1.0
    assert score(np.full((20,2),.5))[0]==.5
    for r in inventory:
        supports=np.asarray(r['raw_supports'])
        assert np.all(supports[:,1]-supports[:,0]==6250)
        assert np.all(supports[1:,0]>=supports[:-1,1])
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'raw_decoding_checks':checks,
            'source_support_geometry_checked_records':len(inventory),'metric_synthetic_checks_passed':True,
            'new_models_fitted':0}
    dump(HERE/'preflight_checks.json',result)
    print(json.dumps(result,indent=2))

def verify():
    from run_validation import frozen
    frozen()
    assert load(HERE/'preflight_checks.json')['new_models_fitted']==0
    start=load(HERE/'fit_started.json')
    assert start['freeze_sha256']==sha(HERE/'freeze.json')
    assert start['analysis_freeze_sha256']==sha(HERE/'analysis_freeze.json')
    assert start['pair_plan_sha256']==sha(HERE/'pair_plan.json')
    assert start['feature_audit_sha256']==sha(HERE/'feature_audit.json')
    reports=load(HERE/'feature_audit.json')['records']
    arrays={};supports={}
    for r in reports:
        p=HERE/'features'/(r['record_id']+'.npz')
        assert sha(p)==r['feature_sha256']
        with np.load(p) as data:
            arrays[r['record_id']]=data['features'];supports[r['record_id']]=data['raw_supports']
        assert np.isfinite(arrays[r['record_id']]).all()
    excluded=set(load(HERE/'duplicate_provenance_audit.json')['affected_participant_labels'])
    assert all(p['subject'] not in excluded for p in load(HERE/'pair_plan.json')['pairs'])
    for s in range(1,30):
        r=load(HERE/'features'/f'sub-{s:02d}_complete.json')
        assert r['archive_md5']==r['archive_official_md5']
    rows=[];max_error=0;models=0;warnings_found=[];iterations=[]
    target_signatures={}
    for pair in load(HERE/'pair_plan.json')['pairs']:
        result=load(HERE/'fits'/(pair['pair_id']+'.json'))
        path=HERE/'fits'/(pair['pair_id']+'.npz')
        assert sha(path)==result['npz_sha256']
        warnings_found.extend(result['warnings'])
        with np.load(path) as data:
            for j,seed in enumerate(result['seeds']):
                test=data['test_indices'][j];same=data['train_same_indices'][j];cross=data['train_cross_indices'][j]
                signature=(pair['subject'],pair['task'],pair['target'],seed)
                current=np.concatenate([test.ravel(),same.ravel()]).tolist()
                assert target_signatures.setdefault(signature,current)==current
                for c in [0,1]:
                    assert len(set(test[c]))==10 and len(set(same[c]))==10 and len(set(cross[c]))==10
                    assert not set(test[c]).intersection(same[c])
                    rid=pair['target_records'][c]
                    a=supports[rid][same[c]];b=supports[rid][test[c]]
                    assert not np.any((a[:,None,0]<b[None,:,1])&(b[None,:,0]<a[:,None,1]))
                xt=np.concatenate([arrays[rid][ii] for rid,ii in zip(pair['target_records'],test)])
                for k,arm in enumerate(['same_record','cross_session']):
                    rids=pair['target_records'] if k==0 else pair['source_records']
                    inds=same if k==0 else cross
                    if k==1:
                        assert pair['source']!=pair['target'] and set(rids).isdisjoint(pair['target_records'])
                    x=np.concatenate([arrays[rid][ii] for rid,ii in zip(rids,inds)])
                    m=data['mask'][j,k];mean=data['mean'][j,k];scale=data['scale'][j,k]
                    assert np.array_equal(x.std(axis=0)>1e-8,m)
                    assert np.allclose(x[:,m].mean(axis=0,dtype=np.float64),mean[m],rtol=1e-12,atol=1e-12)
                    sd=np.sqrt(x[:,m].var(axis=0,dtype=np.float64))
                    assert np.allclose(sd,scale[m],rtol=1e-10,atol=1e-12)
                    transformed=xt.copy();transformed-=mean;transformed/=scale
                    prob1=expit(transformed[:,m]@data['coef'][j,k,m]+data['intercept'][j,k])
                    prob=np.column_stack([1-prob1,prob1])
                    error=float(np.max(abs(prob-data['prob'][j,k])));max_error=max(max_error,error)
                    assert error<1e-9
                    logs=np.log(np.maximum(data['prob'][j,k],1e-12)).reshape(2,10,2).mean(axis=1)
                    prediction=logs.argmax(axis=1)
                    record_ba=float((prediction==np.array([0,1])).mean())
                    window_ba=float((data['prob'][j,k].argmax(axis=1)==np.repeat([0,1],10)).mean())
                    written=next(s for s in result['scores'] if s['seed']==seed and s['arm']==arm)
                    assert record_ba==written['record_ba'] and window_ba==written['window_ba']
                    rows.append({'subject':pair['subject'],'task':pair['task'],'primary':pair['primary'],'arm':arm,'record_ba':record_ba,'window_ba':window_ba})
                    models+=1;iterations.append(int(data['iterations'][j,k]))
    table=pd.DataFrame(rows)
    recomputed=table[table.primary].groupby(['task','subject','arm'])[['record_ba','window_ba']].mean().reset_index()
    published=pd.read_csv(HERE/'participant_scores.csv')
    published=published[published.population=='primary']
    merged=recomputed.merge(published,on=['task','subject','arm'],suffixes=('_new','_saved'),validate='one_to_one')
    for metric in ['record_ba','window_ba']:
        assert np.allclose(merged[metric+'_new'],merged[metric+'_saved'],atol=1e-14,rtol=0)
    summary=pd.read_csv(HERE/'summary.csv')
    for task in ['NBACK','MATB']:
        new=recomputed[recomputed.task==task].pivot(index='subject',columns='arm',values='record_ba')
        old=summary[(summary.task==task)&(summary.population=='primary')].iloc[0]
        assert len(new)==old.n_participants
        assert abs((new.same_record-new.cross_session).mean()-old.difference)<1e-14
    report={'verified_utc':datetime.now(timezone.utc).isoformat(),'model_count':models,'raw_archives_official_MD5_verified':29,
            'features_checked':len(arrays),'max_saved_model_probability_error':max_error,
            'max_solver_iterations':max(iterations),'convergence_warnings':warnings_found,
            'train_test_sample_support_overlap':0,'target_sets_identical_across_source_choices':True,
            'saved_probabilities_metrics_and_participant_summaries_recomputed':True,
            'scope':'Raw decoding checked against MNE on two layouts; official archive hashes verified during extraction. Every saved model prediction, split support and participant metric checked. Models not independently retrained.'}
    dump(HERE/'verification.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    else:verify()
