"""Frozen, checkpointed COG-BCI session validation. No hyperparameter search."""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
import platform
import sys
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.signal import iirnotch, resample_poly, sosfiltfilt, tf2sos
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from audit_inputs import HERE, ROOT, ARCHIVES, DATA, TASKS, CHANNELS, dump, sha
from neutral_feature_extractor import VariantConfig, extract_features

BANDS = [(0.5,4),(4,8),(8,12),(12,18),(18,30),(30,55)]
CODE_FILES = ['audit_inputs.py','neutral_feature_extractor.py','run_validation.py','verify_results.py']

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8], 'little')

def frozen():
    f = load(HERE/('analysis_freeze.json' if (HERE/'analysis_freeze.json').exists() else 'freeze.json'))
    for name, expected in f['sha256'].items():
        assert sha(HERE/name) == expected, f'Frozen input changed: {name}'
    return f

def freeze_analysis():
    assert not (HERE/'fit_started.json').exists(), 'Analysis amendment must precede fitting'
    assert not list((HERE/'fits').glob('*.npz'))
    initial=load(HERE/'freeze.json')
    for name in ['run_validation.py','verify_results.py','protocol.json']:
        assert sha(HERE/'initial_freeze_sources'/name)==initial['sha256'][name]
    for name in ['neutral_feature_extractor.py','audit_inputs.py','input_inventory.json']:
        assert sha(HERE/name)==initial['sha256'][name]
    names=CODE_FILES+['protocol.json','input_inventory.json','freeze.json','feature_audit.json',
                     'verify_provenance.py','duplicate_provenance_audit.json','official_duplicate_source_checks.json']
    f={'frozen_utc':datetime.now(timezone.utc).isoformat(),'sha256':{n:sha(HERE/n) for n in names},
       'new_models_fitted':0,'reason':'Verified cross-ID duplicates; conservative full-label quarantine. Correct verification to test no duplicate label admitted rather than assume source contains no duplicates.',
       'unchanged':'Raw extraction, feature definitions, learner, sampling geometry, endpoints and statistics.',
       'quarantined_participant_labels':load(HERE/'duplicate_provenance_audit.json')['affected_participant_labels']}
    dump(HERE/'analysis_freeze.json',f)
    print('Post-source-audit analysis freeze saved; zero new models fitted.',flush=True)

def freeze():
    if (HERE/'freeze.json').exists():
        frozen()
        print('Existing freeze verified; not replaced.', flush=True)
        return
    assert not list((HERE/'fits').glob('*.npz')), 'Cannot freeze after fitting'
    original = ROOT/'shared/first_dataset/preprocessing_sensitivity.py'
    def fn_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='extract_features'))
    assert fn_ast(original) == fn_ast(HERE/'neutral_feature_extractor.py'), 'Feature implementation changed'
    names = CODE_FILES + ['protocol.json','input_inventory.json']
    f = {'frozen_utc':datetime.now(timezone.utc).isoformat(), 'sha256':{n:sha(HERE/n) for n in names},
         'original_feature_extractor':str(original.relative_to(ROOT)), 'original_feature_extractor_sha256':sha(original),
         'feature_function_AST_identical':True,
         'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'platform':platform.platform()},
         'models_fitted_in_new_experiment_at_freeze':0}
    dump(HERE/'freeze.json',f)
    print('Protocol, source inventory and executable code frozen before fitting.', flush=True)

def archive_digest(path):
    md5, s = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            md5.update(block); s.update(block)
    return md5.hexdigest(), s.hexdigest()

def preprocess_support(signal):
    # All filtering support is local to this disjoint 12.5-second raw tile.
    down = resample_poly(signal, 1, 2, axis=0).astype(np.float32)
    b,a = iirnotch(50/(250/2),30)
    filtered = sosfiltfilt(tf2sos(b,a),down,axis=0).astype(np.float32)
    f,_ = extract_features(filtered, sfreq=250, win_sec=8.5, stride_sec=0.5,
                           bands=BANDS, config=VariantConfig('neutral','none',1,1,1),
                           device=torch.device('cpu'), chunk_size=16)
    assert f.shape == (9,272)
    # 9 candidate starts: 0, .5, ..., 4 seconds; take only the central one (2 s).
    return f[4]

def feature_subject(subject):
    torch.set_num_threads(1)
    frozen()
    dest = HERE/'features'; dest.mkdir(exist_ok=True)
    finished = dest/f'sub-{subject:02d}_complete.json'
    if finished.exists():
        old = load(finished)
        assert old['freeze_sha256'] == sha(HERE/'freeze.json')
        for name,h in old['files_sha256'].items():
            assert sha(dest/name)==h
        return {'subject':subject,'reused':True}
    rows = [r for r in load(HERE/'input_inventory.json')['records'] if r['subject']==subject]
    archive = ARCHIVES/f'sub-{subject:02d}.zip'
    manifest = load(archive.with_suffix('.zip.manifest.json'))
    md5, checksum = archive_digest(archive)
    assert md5==manifest['md5'] and archive.stat().st_size==manifest['bytes'], f'Archive changed: {subject}'
    reports=[]; files={}; started=time.perf_counter()
    with zipfile.ZipFile(archive) as z, threadpool_limits(limits=1):
        for r in rows:
            if not r['header_eligible']:
                reports.append(r); continue
            raw = z.read(r['fdt_member'])  # zipfile verifies member CRC while reading.
            fdt_sha=hashlib.sha256(raw).hexdigest()
            x=np.frombuffer(raw,dtype='<f4').reshape(r['pnts'],r['nbchan'])[:,r['selected_channel_indices']]
            features=[]; supports=[]; rejected=[]; ptp=[]
            for i,(a,b) in enumerate(r['raw_supports']):
                signal=x[a:b]
                reason=''
                if not np.isfinite(signal).all(): reason='nonfinite_raw'
                elif np.any(signal.std(axis=0,dtype=np.float64)<=1e-8): reason='flat_selected_channel'
                if reason:
                    rejected.append({'tile':i,'reason':reason}); continue
                f=preprocess_support(signal)
                if not np.isfinite(f).all():
                    rejected.append({'tile':i,'reason':'nonfinite_features'}); continue
                features.append(f); supports.append((a,b)); ptp.append(np.ptp(signal,axis=0))
            array=np.asarray(features,dtype=np.float32).reshape(-1,272)
            support_array=np.asarray(supports,dtype=np.int64).reshape(-1,2)
            name=r['record_id']+'.npz'
            np.savez_compressed(dest/name,features=array,raw_supports=support_array,
                                amplitude_ptp_native=np.asarray(ptp,dtype=np.float64))
            files[name]=sha(dest/name)
            reports.append({**r,'fdt_sha256':fdt_sha,'n_valid_tiles':len(array),'rejected_tiles':rejected,
                            'eligible_signal':len(array)>=20,'feature_sha256':files[name],
                            'ptp_native_min_max':[float(np.min(ptp)),float(np.max(ptp))] if ptp else None})
    payload={'subject':subject,'freeze_sha256':sha(HERE/'freeze.json'),'archive_official_md5':manifest['md5'],
             'archive_md5':md5,'archive_sha256':checksum,'records':reports,'files_sha256':files,'seconds':time.perf_counter()-started}
    dump(finished,payload)
    return {'subject':subject,'records':len(reports),'valid_tiles':sum(r.get('n_valid_tiles',0) for r in reports),'seconds':round(payload['seconds'],1)}

def features(workers):
    frozen()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(feature_subject,s) for s in range(1,30)]
        for future in as_completed(futures): print(json.dumps(future.result()),flush=True)
    rows=[r for s in range(1,30) for r in load(HERE/'features'/f'sub-{s:02d}_complete.json')['records']]
    by_hash={}
    for r in rows:
        if 'fdt_sha256' in r: by_hash.setdefault(r['fdt_sha256'],[]).append(r['record_id'])
    dup=[v for v in by_hash.values() if len(v)>1]
    dump(HERE/'feature_audit.json',{'records':rows,'duplicate_selected_FDT_groups':dup,
                                    'rejected_tile_count':sum(len(r.get('rejected_tiles',[])) for r in rows)})
    print('Feature audit complete: '+json.dumps({'records':len(rows),'duplicates':dup,'ineligible':[r['record_id'] for r in rows if not r.get('eligible_signal',False)]}),flush=True)

def pair_plan():
    audit=load(HERE/'feature_audit.json')
    dup={rid for g in audit['duplicate_selected_FDT_groups'] for rid in g}
    excluded_labels={r['subject'] for r in audit['records'] if r['record_id'] in dup}
    records={(r['subject'],r['session'],r['task'],r['label']):r for r in audit['records']}
    pairs=[]; excluded=[]
    for subject in range(1,30):
        if subject in excluded_labels:
            excluded.append({'subject':subject,'reason':'Full participant-label quarantine: cross-ID duplicate EEG provenance unresolved'})
            continue
        for task in TASKS:
            good=[]
            for ses in range(1,4):
                rr=[records[(subject,ses,task,c)] for c in [0,1]]
                if all(r.get('eligible_signal',False) and r['record_id'] not in dup for r in rr): good.append(ses)
                else: excluded.append({'subject':subject,'task':task,'session':ses,'reason':'record eligibility/duplicates','records':[r['record_id'] for r in rr]})
            for source in good:
                for target in good:
                    if source==target: continue
                    sr=[records[(subject,source,task,c)] for c in [0,1]]
                    tr=[records[(subject,target,task,c)] for c in [0,1]]
                    cadence=True
                    if task=='NBACK':
                        cadence=all(np.ptp(r['median_isi_seconds'])<=.02 for r in sr+tr)
                        cadence=cadence and all(abs(np.median(a['median_isi_seconds'])-np.median(b['median_isi_seconds']))<=.02 for a,b in zip(sr,tr))
                    clean=not any(r['interrupted'] for r in sr+tr)
                    pairs.append({'pair_id':f'sub-{subject:02d}_{task}_S{source}_to_S{target}',
                                  'subject':subject,'task':task,'source':source,'target':target,
                                  'source_records':[r['record_id'] for r in sr], 'target_records':[r['record_id'] for r in tr],
                                  'matching_cadence':bool(cadence),'uninterrupted':clean,'primary':bool(clean and cadence)})
    dump(HERE/'pair_plan.json',{'pairs':pairs,'excluded_sessions':excluded,'prepared_utc':datetime.now(timezone.utc).isoformat()})
    return pairs

def select_indices(pair,seed,arrays):
    test=[]; same=[]; cross=[]
    for c in [0,1]:
        target=arrays[pair['target_records'][c]]
        source=arrays[pair['source_records'][c]]
        perm=np.random.default_rng(stable_seed('target',pair['subject'],pair['task'],pair['target'],c,seed)).permutation(len(target))
        test.append(perm[:10]); same.append(perm[10:20])
        rng=np.random.default_rng(stable_seed('source',pair['subject'],pair['task'],pair['source'],pair['target'],c,seed))
        cross.append(rng.choice(len(source),size=10,replace=False))
    return np.array(test),np.array(same),np.array(cross)

def fit_one(x,y,xt):
    mask=x.std(axis=0)>1e-8
    assert mask.any()
    scaler=StandardScaler().fit(x[:,mask])
    a=scaler.transform(x[:,mask]); b=scaler.transform(xt[:,mask])
    params=load(HERE/'protocol.json')['fixed_learner'].copy();params.pop('model')
    if tuple(map(int,sklearn.__version__.split('.')[:2]))>=(1,8):
        params.pop('penalty');params['l1_ratio']=0.0
    model=LogisticRegression(**params)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always');warnings.filterwarnings('error',category=ConvergenceWarning)
        model.fit(a,y)
    assert np.array_equal(model.classes_,[0,1])
    # Full 272-column arrays make saved models independent of Python pickle files.
    coef=np.zeros(272);mean=np.zeros(272);scale=np.ones(272)
    coef[mask]=model.coef_[0];mean[mask]=scaler.mean_;scale[mask]=scaler.scale_
    return {'mask':mask,'mean':mean,'scale':scale,'coef':coef,'intercept':model.intercept_[0],
            'prob':model.predict_proba(b),'iterations':int(model.n_iter_[0]),
            'warnings':[str(w.message) for w in caught]}

def score(prob):
    rp=np.mean(np.log(np.clip(prob.reshape(2,10,2),1e-12,1)),axis=1)
    rp=np.exp(rp-rp.max(axis=1,keepdims=True));rp/=rp.sum(axis=1,keepdims=True)
    pred=rp.argmax(axis=1)
    record_ba=float(np.mean(pred==[0,1]))
    y=np.repeat([0,1],10)
    window_ba=float(np.mean(prob.argmax(axis=1)==y))
    return record_ba,window_ba,rp

def fit(workers):
    frozen()
    # Before fitting, verify all generated feature files and freeze the pair list.
    for s in range(1,30):
        complete=load(HERE/'features'/f'sub-{s:02d}_complete.json')
        assert complete['freeze_sha256']==sha(HERE/'freeze.json')
        for name,h in complete['files_sha256'].items(): assert sha(HERE/'features'/name)==h
    if (HERE/'pair_plan.json').exists(): pairs=load(HERE/'pair_plan.json')['pairs']
    else: pairs=pair_plan()
    if not (HERE/'fit_started.json').exists():
        dump(HERE/'fit_started.json',{'utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(HERE/'freeze.json'),
                                     'analysis_freeze_sha256':sha(HERE/'analysis_freeze.json'),
                                     'feature_audit_sha256':sha(HERE/'feature_audit.json'),'pair_plan_sha256':sha(HERE/'pair_plan.json')})
    start=load(HERE/'fit_started.json')
    assert start['pair_plan_sha256']==sha(HERE/'pair_plan.json') and start['feature_audit_sha256']==sha(HERE/'feature_audit.json')
    seeds=load(HERE/'protocol.json')['seeds']
    arrays={rid:np.load(HERE/'features'/f'{rid}.npz')['features'] for rid in {r for p in pairs for r in p['source_records']+p['target_records']}}
    dest=HERE/'fits';dest.mkdir(exist_ok=True)
    t=time.perf_counter()
    for num,pair in enumerate(pairs,1):
        path=dest/(pair['pair_id']+'.npz');summary=dest/(pair['pair_id']+'.json')
        if path.exists() and summary.exists():
            old=load(summary);assert old['npz_sha256']==sha(path);continue
        saved={key:[] for key in ['test_indices','train_same_indices','train_cross_indices','prob','mask','mean','scale','coef','intercept','iterations']}
        scores=[];warning_list=[]
        for seed in seeds:
            test,same,cross=select_indices(pair,seed,arrays)
            xt=np.concatenate([arrays[rid][ii] for rid,ii in zip(pair['target_records'],test)])
            y=np.repeat([0,1],10)
            models=[]
            for arm,rids,inds in [('same_record',pair['target_records'],same),('cross_session',pair['source_records'],cross)]:
                x=np.concatenate([arrays[rid][ii] for rid,ii in zip(rids,inds)])
                model=fit_one(x,y,xt);models.append(model)
                rb,wb,rp=score(model['prob'])
                scores.append({'seed':seed,'arm':arm,'record_ba':rb,'window_ba':wb,'record_probabilities':rp.tolist()})
                warning_list.extend(model['warnings'])
            saved['test_indices'].append(test);saved['train_same_indices'].append(same);saved['train_cross_indices'].append(cross)
            for key in ['prob','mask','mean','scale','coef','intercept','iterations']:saved[key].append([m[key] for m in models])
        np.savez_compressed(path,**{k:np.asarray(v) for k,v in saved.items()})
        dump(summary,{'pair':pair,'seeds':seeds,'scores':scores,'npz_sha256':sha(path),'warnings':warning_list,
                      'freeze_sha256':sha(HERE/'freeze.json')})
        if num%12==0 or num==len(pairs):print(f'Fitted {num}/{len(pairs)} ordered pairs, {num*10} LR fits; elapsed {time.perf_counter()-t:.1f}s',flush=True)
    print('All fits complete.',flush=True)

def bh(values):
    p=np.asarray(values);order=np.argsort(p);out=np.empty(len(p));rank=np.arange(1,len(p)+1)
    out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/rank)[::-1])[::-1]);return out

def summarize():
    frozen()
    rows=[]
    for p in load(HERE/'pair_plan.json')['pairs']:
        r=load(HERE/'fits'/(p['pair_id']+'.json'))
        rows.extend({**{k:v for k,v in p.items() if k not in ['source_records','target_records']},**{k:v for k,v in s.items() if k!='record_probabilities'}} for s in r['scores'])
    df=pd.DataFrame(rows);df.to_csv(HERE/'fold_scores.csv',index=False)
    participants=[];summary=[]
    for population,mask in [('primary',df.primary),('all_sessions_matching_cadence',df.matching_cadence),('all_sessions_all_pairs',np.ones(len(df),dtype=bool))]:
        for task in TASKS:
            current=df[mask & (df.task==task)]
            group=current.groupby(['subject','arm'])[['record_ba','window_ba']].mean().reset_index()
            group.insert(0,'population',population);group.insert(1,'task',task)
            participants.append(group)
            pivot=group.pivot(index='subject',columns='arm',values='record_ba').dropna()
            a=pivot.same_record.to_numpy();b=pivot.cross_session.to_numpy();d=a-b;n=len(d)
            rng=np.random.default_rng(20261110);draws=rng.integers(n,size=(10000,n))
            ci=lambda v:np.quantile(v[draws].mean(axis=1),[.025,.975]).tolist()
            row={'population':population,'task':task,'n_participants':n,'n_ordered_pairs':current.pair_id.nunique(),
                 'n_unique_sessions':len({(r.subject,r.target) for r in current.itertuples()}),
                 'same_record_ba':float(a.mean()),'same_record_ci_low':ci(a)[0],'same_record_ci_high':ci(a)[1],
                 'cross_session_ba':float(b.mean()),'cross_session_ci_low':ci(b)[0],'cross_session_ci_high':ci(b)[1],
                 'difference':float(d.mean()),'difference_ci_low':ci(d)[0],'difference_ci_high':ci(d)[1],
                 'positive_participants':int(np.sum(d>1e-12)),'zero_participants':int(np.sum(abs(d)<=1e-12)),'negative_participants':int(np.sum(d < -1e-12))}
            if population=='primary':
                rng=np.random.default_rng(20261110)
                simulated=(rng.choice([-1,1],size=(100000,n))*d).mean(axis=1)
                row['p_signflip']=(1+int(np.sum(abs(simulated)>=abs(d.mean())-1e-15)))/100001
            summary.append(row)
    out=pd.DataFrame(summary)
    primary=out.population=='primary';out.loc[primary,'p_BH2']=bh(out.loc[primary,'p_signflip'].to_numpy())
    out.to_csv(HERE/'summary.csv',index=False)
    allp=pd.concat(participants,ignore_index=True);allp.to_csv(HERE/'participant_scores.csv',index=False)
    seed=df[df.primary].groupby(['task','seed','subject','arm'])[['record_ba','window_ba']].mean().reset_index()
    seed.groupby(['task','seed','arm'])[['record_ba','window_ba']].mean().reset_index().to_csv(HERE/'seed_sensitivity.csv',index=False)
    deletion=[]
    for task in TASKS:
        part=allp[(allp.population=='primary')&(allp.task==task)].pivot(index='subject',columns='arm',values='record_ba')
        diff=part.same_record-part.cross_session
        for s in part.index:deletion.append({'task':task,'deleted_subject':s,'remaining_n':len(diff)-1,'difference':float(diff.drop(s).mean())})
    pd.DataFrame(deletion).to_csv(HERE/'participant_deletion.csv',index=False)
    pd.DataFrame(load(HERE/'pair_plan.json')['pairs']).to_csv(HERE/'pair_eligibility.csv',index=False)
    print(out.to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','features','freeze-analysis','fit','summarize']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='freeze':freeze()
    elif args.phase=='freeze-analysis':freeze_analysis()
    elif args.phase=='features':features(args.workers)
    elif args.phase=='fit':fit(args.workers)
    else:summarize()

if __name__=='__main__':main()
