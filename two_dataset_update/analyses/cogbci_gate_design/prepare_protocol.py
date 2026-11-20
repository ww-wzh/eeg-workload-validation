"""B1: verify sources, construct deterministic cohorts and freeze gate audit design.

No classifier, stacker, gain predictor or gate is fitted by this script.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import platform
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'cogbci_session_validation'
ARCHIVES=ROOT/'data/cogbci/raw_archives'
sys.path.insert(0,str(PARENT))
from neutral_feature_extractor import VariantConfig,extract_features

BANDS=[(.5,4),(4,8),(8,12),(12,18),(18,30),(30,55)]

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(content,encoding='utf-8');temp.replace(p)

def source_checks():
    lines=(PARENT/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
    for line in lines:
        expected,name=line.split('  ',1)
        assert sha(PARENT/name)==expected,f'Parent artifact changed: {name}'
    report=load(PARENT/'verification.json')
    assert report['max_saved_model_probability_error']==0 and report['train_test_sample_support_overlap']==0
    names=['protocol.json','input_inventory.json','feature_audit.json','analysis_freeze.json',
           'duplicate_provenance_audit.json','verification.json','neutral_feature_extractor.py','MANIFEST.sha256']
    code=[ROOT/'shared/revision_pipeline'/name for name in ['models.py','risk_gate.py','splits.py','gate_pilot.py']]
    code += [ROOT/'shared/first_dataset/gate_pilot_protocol.json',
             ROOT/'shared/first_dataset/gate_protocol/nested_gate_protocol.json']
    audit=HERE.parent/'first_dataset_implementation_audit'
    audit_files=[audit/n for n in ['audit_legacy_gate.py','audit_report.json','sensitivity_scope_audit.json',
                                  'threshold_replay_by_gate.csv','threshold_impact_summary.csv']]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [*(PARENT/n for n in names),*code,*audit_files]}
    protocol=load(HERE/'protocol.json')
    original=load(code[-2])['model_config']
    for model in ['mlp','rf','extra_trees','stacker']:
        for key,value in original[model].items():
            if key=='n_jobs':continue
            assert protocol['base_models'][model][key]==value,(model,key)
    original_gates=load(code[-1])['gate_rules']['cbsf']
    for key in ['benefit_alpha','event_C','risk_penalty','temperature','max_weight']:
        assert protocol['gate_rules']['cbsf'][key]==original_gates[key]
    assert protocol['gate_rules']['cbsf']['loss_event_threshold']==original_gates['event_threshold']
    return {'verified_utc':datetime.now(timezone.utc).isoformat(),'parent_manifest_files_verified':len(lines),
            'sources_sha256':inputs,'historical_declared_model_and_gate_constants_match':True,
            'legacy_implementation_caveat':'Protocol uses signed event cutoff -0.05, but legacy fitting API internally negates its argument. New runner must pass magnitude +0.05 or implement gain<=-0.05 directly. Historical replay is audited separately.'}

def build_cohort():
    protocol=load(HERE/'protocol.json')
    records=load(PARENT/'feature_audit.json')['records']
    excluded=set(protocol['cohort']['excluded_subjects_provenance'])
    expected=set(protocol['cohort']['expected_subjects'])
    assert excluded==set(load(PARENT/'duplicate_provenance_audit.json')['affected_participant_labels'])
    kept=[]; excluded_rows=[]
    for r in records:
        reason=''
        if r['subject'] in excluded:reason='participant_provenance_quarantine'
        elif r['interrupted']:reason='notebook_interrupted_session'
        elif r['task']=='NBACK' and not all(abs(v-2.008)<=.02 for v in r['median_isi_seconds']):reason='noncanonical_nback_stimulus_interval'
        elif not r.get('eligible_signal',False) or r['n_valid_tiles']<20:reason='source_feature_feasibility_failure'
        row={k:r[k] for k in ['record_id','subject','session','task','label','stem','fdt_member','fdt_sha256','nbchan','pnts','selected_channel_indices']}
        row['source_feature_sha256']=r['feature_sha256']
        if reason:
            excluded_rows.append({**row,'reason':reason});continue
        assert r['subject'] in expected
        indices=np.linspace(0,r['n_valid_tiles']-1,20,dtype=int)
        assert len(np.unique(indices))==20
        row['selected_cache_indices']=indices.tolist()
        row['valid_tiles_before_selection']=r['n_valid_tiles']
        kept.append(row)
    kept.sort(key=lambda r:(r['task'],r['subject'],r['session'],r['label']))
    by_subject=[]
    for task in ['NBACK','MATB']:
        rr=[r for r in kept if r['task']==task]
        assert {r['subject'] for r in rr}==expected
        target=protocol['cohort']['expected_counts'][task]
        assert len(rr)==target['records'] and 20*len(rr)==target['windows']
        assert len({(r['subject'],r['session']) for r in rr})==target['sessions']
        for s in sorted(expected):
            cur=[r for r in rr if r['subject']==s]
            counts=Counter(r['label'] for r in cur)
            assert counts[0]==counts[1] and counts[0]>=1
            sessions=sorted({r['session'] for r in cur})
            assert all({r['label'] for r in cur if r['session']==ses}=={0,1} for ses in sessions)
            by_subject.append({'task':task,'subject':s,'sessions':','.join(map(str,sessions)),
                               'n_sessions':len(sessions),'records_per_class':counts[0],
                               'n_records':len(cur),'n_windows':len(cur)*20})
    dump(HERE/'record_selection.json',{'records':kept,'excluded_records':excluded_rows})
    pd.DataFrame(by_subject).to_csv(HERE/'participant_inventory.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['selected_channel_indices','selected_cache_indices']} for r in excluded_rows]).to_csv(HERE/'excluded_records.csv',index=False)
    return kept,by_subject,excluded_rows

def reconstruct_subject(subject):
    torch.set_num_threads(1)
    records=[r for r in load(HERE/'record_selection.json')['records'] if r['subject']==subject]
    dest=HERE/load(HERE/'protocol.json')['inputs']['prepared_input_directory'];dest.mkdir(exist_ok=True)
    identity={'preparation_code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'protocol.json'),
              'selection_sha256':sha(HERE/'record_selection.json'),'extractor_sha256':sha(PARENT/'neutral_feature_extractor.py')}
    complete=dest/f'sub-{subject:02d}_complete.json'
    if complete.exists():
        cached=load(complete);assert cached['identity']==identity
        for name,h in cached['files_sha256'].items():assert sha(dest/name)==h
        return {'subject':subject,'reused':True,'records':len(records)}
    reports=[];files={};started=time.perf_counter()
    b,a=iirnotch(50/(250/2),30);notch=tf2sos(b,a)
    band=butter(4,[.5/(250/2),55/(250/2)],btype='band',output='sos')
    config=VariantConfig('neutral','none',1,1,1)
    with zipfile.ZipFile(ARCHIVES/f'sub-{subject:02d}.zip') as z,threadpool_limits(limits=1):
        for r in records:
            cache=PARENT/'features'/(r['record_id']+'.npz')
            assert sha(cache)==r['source_feature_sha256']
            with np.load(cache,allow_pickle=False) as d:
                ii=np.array(r['selected_cache_indices'])
                reference=d['features'][ii];supports=d['raw_supports'][ii];ptp=d['amplitude_ptp_native'][ii]
            assert np.all(supports[1:,0]>=supports[:-1,1])
            raw=z.read(r['fdt_member'])
            assert hashlib.sha256(raw).hexdigest()==r['fdt_sha256']
            signal=np.frombuffer(raw,dtype='<f4').reshape(r['pnts'],r['nbchan'])[:,r['selected_channel_indices']]
            rebuilt=[];covariances=[];matrix_errors=[];legacy_asymmetry=[]
            for start,stop in supports:
                local=signal[start:stop]
                assert len(local)==6250
                down=resample_poly(local,1,2,axis=0).astype(np.float32)
                filtered=sosfiltfilt(notch,down,axis=0).astype(np.float32)
                f,c=extract_features(filtered,sfreq=250,win_sec=8.5,stride_sec=.5,bands=BANDS,
                                     config=config,device=torch.device('cpu'),chunk_size=16)
                assert f.shape==(9,272) and c.shape==(9,8,8)
                # Repeat the extractor's bandpass and central window exactly, then
                # increase only the accumulation precision of the same second moment.
                central=sosfiltfilt(band,filtered,axis=0).astype(np.float32)[500:2625].T.astype(np.float64)
                assert central.shape==(8,2125)
                c64=central@central.T/2125+1e-4*np.eye(8)
                scale=np.linalg.norm(c64,'fro')
                relative_error=np.linalg.norm(c[4].astype(np.float64)-c64,'fro')/scale
                assert relative_error<=1e-5,(r['record_id'],start,relative_error)
                assert np.array_equal(c64,c64.T)
                assert np.linalg.eigvalsh(c64).min()>=-1e-12*scale
                matrix_errors.append(float(relative_error))
                legacy_asymmetry.append(float(np.linalg.norm(c[4].astype(np.float64)-c[4].T.astype(np.float64),'fro')/scale))
                rebuilt.append(f[4]);covariances.append(c64)
            rebuilt=np.asarray(rebuilt,dtype=np.float32);covariances=np.asarray(covariances,dtype=np.float64)
            assert np.isfinite(rebuilt).all() and np.isfinite(covariances).all()
            difference=np.abs(rebuilt.astype(np.float64)-reference.astype(np.float64))
            assert np.allclose(rebuilt,reference,atol=2e-4,rtol=2e-4),r['record_id']
            assert np.array_equal(covariances,covariances.transpose(0,2,1))
            name=r['record_id']+'.npz'
            np.savez_compressed(dest/name,features=reference,covariances=covariances,raw_supports=supports,
                                selected_cache_indices=ii,amplitude_ptp_native=ptp)
            files[name]=sha(dest/name)
            reports.append({'record_id':r['record_id'],'n_windows':20,'max_feature_abs_difference':float(difference.max()),
                            'feature_allclose':True,'fdt_sha256':r['fdt_sha256'],'output_sha256':files[name],
                            'covariance_dtype':'float64','max_float32_to_float64_relative_fro_error':max(matrix_errors),
                            'max_legacy_relative_fro_asymmetry':max(legacy_asymmetry),
                            'exact_covariance_symmetry':True,'numerical_psd_check':True})
    dump(complete,{'subject':subject,'identity':identity,'files_sha256':files,'records':reports,
                   'elapsed_seconds':time.perf_counter()-started,'models_fitted':0})
    return {'subject':subject,'records':len(records),'seconds':round(time.perf_counter()-started,1),'models_fitted':0}

def split_checks_and_job_plan(participants):
    protocol=load(HERE/'protocol.json');subjects=protocol['cohort']['expected_subjects'];seeds=protocol['seeds']
    selected=load(HERE/'record_selection.json')['records']
    rows_by_task={task:[r for r in selected if r['task']==task] for task in ['NBACK','MATB']}
    jobs=[];checks=Counter();stage_counts=Counter()
    for task,rows in rows_by_task.items():
        by_s={s:[r for r in rows if r['subject']==s] for s in subjects}
        ids={s:{r['record_id'] for r in by_s[s]} for s in subjects}
        def verify_training(train_ids,held_ids):
            assert not set(train_ids)&set(held_ids)
            train_records=set().union(*(ids[s] for s in train_ids))
            held_records=set().union(*(ids[s] for s in held_ids))
            assert train_records.isdisjoint(held_records)
            yy=[r['label'] for s in train_ids for r in by_s[s]]
            assert yy.count(0)==yy.count(1)>0
            checks['participant_record_and_class_checks']+=1
        for outer in subjects:
            train=[s for s in subjects if s!=outer]
            verify_training(train,[outer])
            for inner in train:verify_training([s for s in train if s!=inner],[outer,inner])
            for seed in seeds:
                jobs.append({'kind':'outer','task':task,'excluded_subjects':[outer],'prediction_subjects':[outer],
                             'training_subjects':train,'seed':seed,'inner_participant_folds':24,'fit_stages':76,
                             'job_id':f'{task}_outer_s{outer:02d}_seed{seed}'})
                stage_counts.update({'mlp':25,'rf':25,'extra_trees':25,'stacker':1})
        for first,second in itertools.combinations(subjects,2):
            train=[s for s in subjects if s not in [first,second]]
            verify_training(train,[first,second])
            for inner in train:verify_training([s for s in train if s!=inner],[first,second,inner])
            # Both directed uses share exactly the same source cohort; score files remain separated.
            assert train==[s for s in subjects if s not in [second,first]]
            checks['symmetric_exclusion_pair_checks']+=1
            for seed in seeds:
                jobs.append({'kind':'double_exclusion','task':task,'excluded_subjects':[first,second],
                             'prediction_subjects':[first,second],'training_subjects':train,'seed':seed,
                             'inner_participant_folds':23,'fit_stages':73,
                             'job_id':f'{task}_exclude_s{first:02d}_s{second:02d}_seed{seed}',
                             'response_routing':[{'outer':first,'meta':second},{'outer':second,'meta':first}]})
                stage_counts.update({'mlp':24,'rf':24,'extra_trees':24,'stacker':1})
    # Exhaust every binary record prediction for every attainable balanced record count.
    for half in [1,2,3]:
        y=np.repeat([0,1],half)
        for pred in itertools.product([0,1],repeat=2*half):
            correct=np.array(pred)==y
            accuracy=correct.mean();ba=.5*(correct[y==0].mean()+correct[y==1].mean())
            assert abs(accuracy-ba)<1e-14
            checks['accuracy_BA_prediction_patterns']+=1
    assert len(jobs)==3250 and sum(j['fit_stages'] for j in jobs)==238000
    expected=protocol['execution']['expected_fit_stages_without_additional_deduplication']
    assert all(stage_counts[k]==expected[k] for k in stage_counts)
    dump(HERE/'job_plan.json',{'jobs':jobs,'scientific_scope':'Two task-specific participant-LOSO experiments; all sessions grouped by participant.','models_fitted':0})
    dump(HERE/'split_feasibility.json',{'status':'passed_metadata_design_checks_not_training_validation',
         'checks':dict(checks),'job_count':len(jobs),'fit_stage_counts':dict(stage_counts),
         'total_fit_stages':sum(stage_counts.values()),'outer_target_cells':50,'meta_rows_per_outer':24,
         'total_correlated_meta_rows':1200,'meta_rows_per_seed_across_tasks':1200,
         'total_meta_seed_rows':6000,'independent_participants_per_task':25,
         'untrained_runner_note':'Membership design has been checked; future fitting code must enforce the same exclusions at runtime.',
         'models_fitted':0})

def create_report(participants,excluded):
    by_task={task:[r for r in participants if r['task']==task] for task in ['NBACK','MATB']}
    counts={task:{'participants':len(rr),'sessions':sum(r['n_sessions'] for r in rr),'records':sum(r['n_records'] for r in rr),'windows':sum(r['n_windows'] for r in rr)} for task,rr in by_task.items()}
    protocol=load(HERE/'protocol.json')
    dest=HERE/protocol['inputs']['prepared_input_directory']
    prep=[load(dest/f'sub-{s:02d}_complete.json') for s in protocol['cohort']['expected_subjects']]
    max_diff=max(r['max_feature_abs_difference'] for s in prep for r in s['records'])
    max_matrix_error=max(r['max_float32_to_float64_relative_fro_error'] for s in prep for r in s['records'])
    report={'status':'B1_complete_design_and_inputs_ready_no_training','counts':counts,
            'excluded_record_counts':dict(Counter(r['reason'] for r in excluded)),
            'prepared_records':sum(len(s['records']) for s in prep),'prepared_windows':sum(r['n_windows'] for s in prep for r in s['records']),
            'max_reconstructed_feature_abs_difference':max_diff,'all_covariances_reconstructed_from_hash_verified_raw_supports':True,
            'prepared_input_directory':dest.name,'covariance_accumulation':'float64 from the same float32 filtered samples',
            'max_float32_to_float64_relative_fro_error':max_matrix_error,'exact_covariance_symmetry':True,
            'models_fitted':0,'model_fit_scope':'New COG gate-audit study only; historical small-gate diagnostic replays are separate.',
            'historical_cbsf_issue_report':'../first_dataset_implementation_audit/audit_report.json',
            'runtime':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},
            'completed_utc':datetime.now(timezone.utc).isoformat()}
    dump(HERE/'readiness_report.json',report)
    text=f'''# COG-BCI gate input preparation

This report describes the design and input-preparation stage, before training.
Completed analysis results, when available, are in
`../cogbci_gate_evaluation/analysis/`.

| Task | Participants | Sessions | Records | Windows |
|---|---:|---:|---:|---:|
| N-back | {counts['NBACK']['participants']} | {counts['NBACK']['sessions']} | {counts['NBACK']['records']} | {counts['NBACK']['windows']} |
| MATB | {counts['MATB']['participants']} | {counts['MATB']['sessions']} | {counts['MATB']['records']} | {counts['MATB']['windows']} |

Each target participant is excluded from fitting, selection and gate-response
construction. An additional participant is excluded for each independent
meta-response. Each final gate has 24 responses. Record selection, exclusions,
binary labels, comparisons and inference rules are specified in the protocol.
Each record contributes 20 windows with independent raw and filtering supports.
Reconstructed features match the reference cache with maximum absolute difference
{max_diff:.9g}. The preparation report records the covariance accumulation type
and precision checks. The event definition is gain <= -0.05.
'''
    (HERE/'DESIGN_SUMMARY.md').write_text(text,encoding='utf-8')
    return report

def freeze():
    sources=load(HERE/'source_checks.json')['sources_sha256']
    for name,h in sources.items():assert sha(ROOT/name)==h,f'Protected input changed: {name}'
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['design_freeze.json','MANIFEST.sha256'])
    checksums={p.relative_to(HERE).as_posix():sha(p) for p in paths}
    payload={'frozen_utc':datetime.now(timezone.utc).isoformat(),'phase':'B1 design/input freeze before new model fitting',
             'new_gate_study_model_fits':0,'sha256':checksums,'source_sha256':sources,
             'execution_requirement':'Freeze the runner and runtime before fitting.'}
    dump(HERE/'design_freeze.json',payload)
    checksums['design_freeze.json']=sha(HERE/'design_freeze.json')
    (HERE/'MANIFEST.sha256').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(checksums.items())),encoding='utf-8')
    for name,h in checksums.items():assert sha(HERE/name)==h

def main():
    if (HERE/'design_freeze.json').exists():
        f=load(HERE/'design_freeze.json')
        for name,h in f['sha256'].items():assert sha(HERE/name)==h
        for name,h in f['source_sha256'].items():assert sha(ROOT/name)==h
        print('B1 frozen output verified; no files or models changed.',flush=True);return
    source=source_checks();dump(HERE/'source_checks.json',source)
    kept,participants,excluded=build_cohort()
    split_checks_and_job_plan(participants)
    print('Cohort and all nested participant membership checks passed; reconstructing missing DASF inputs.',flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(reconstruct_subject,s) for s in load(HERE/'protocol.json')['cohort']['expected_subjects']]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    report=create_report(participants,excluded)
    freeze()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
