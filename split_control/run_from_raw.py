"""Refit frozen split-control jobs from original raw files into a separate output directory."""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
import argparse
import csv
import hashlib
import json
import os
import platform
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent


@contextmanager
def output_lock(folder):
    lock = folder / 'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='Directory containing Arithmetic_Data and Stroop_Data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--task', choices=['arithmetic', 'stroop'])
    parser.add_argument('--subject', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--limit', type=int, help='Optional technical validation subset; omit for all selected jobs')
    args = parser.parse_args()
    import numpy as np
    import scipy
    import sklearn
    import torch
    from scipy.signal import iirnotch, tf2sos, sosfiltfilt
    from threadpoolctl import threadpool_limits
    import model_core as core
    from channel_selection import select_openbci_eeg_channels
    from feature_extractor import VariantConfig, extract_features
    raw_root, output = args.raw_root.resolve(), args.output.resolve()
    if output == BASE or BASE in output.parents:
        raise ValueError('Use an output directory outside the distributed evidence')
    output.mkdir(parents=True, exist_ok=True)
    core.HERE = output
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    jobs = [json.loads(line) for line in (BASE/'planned_folds.jsonl').read_text(encoding='utf-8').splitlines()]
    jobs = [j for j in jobs if (args.task is None or j['task'] == args.task)
            and (args.subject is None or j['subject'] == args.subject)
            and (args.seed is None or j['seed'] == args.seed)]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        jobs = jobs[:args.limit]
    if not jobs:
        raise ValueError('No planned jobs match the requested filters')
    with (BASE/'raw_input_identity.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = {r['recording_id']: r for r in csv.DictReader(stream)}
    required = sorted({p['recording_id'] for j in jobs for p in j['training'] + j['testing']})
    for rid in required:
        if core.sha(raw_root/records[rid]['raw_path']) != records[rid]['raw_sha256']:
            raise ValueError('Original raw-file identity mismatch: ' + rid)
    identity = {'scope': 'Portable re-execution of frozen memberships',
                'selected_job_ids': [j['job_id'] for j in jobs],
                'plan_sha256': core.sha(BASE/'planned_folds.jsonl'),
                'raw_sha256': {rid: records[rid]['raw_sha256'] for rid in required},
                'sources_sha256': {p.name: core.sha(p) for p in [Path(__file__), BASE/'model_core.py',
                    BASE/'feature_extractor.py', BASE/'channel_selection.py']},
                'versions': {'python':platform.python_version(), 'numpy':np.__version__,
                             'scipy':scipy.__version__, 'sklearn':sklearn.__version__, 'torch':torch.__version__},
                'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                'numeric_threads': 1}
    identity_sha = core.digest(identity)
    results = []
    b, a = iirnotch(50.0 / 125.0, Q=30.0)
    notch = tf2sos(b, a)
    config = VariantConfig('neutral','none',1.0,1.0,1.0)
    bands = [(0.5,4),(4,8),(8,12),(12,18),(18,30),(30,55)]
    with output_lock(output), threadpool_limits(limits=1):
        identity_path = output/'run_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Output belongs to a different execution identity')
        else:
            core.write_json(identity_path,identity)
        # Work by participant to bound feature memory. Feature cache completion
        # hashes cover every filtering interval used by the selected jobs.
        for task, subject in sorted({(j['task'],j['subject']) for j in jobs}):
            cell_jobs=[j for j in jobs if (j['task'],j['subject'])==(task,subject)]
            intervals=sorted({(p['recording_id'],*p['filter_interval']) for j in cell_jobs for p in j['training']+j['testing']})
            cache=output/'features'/f'{task}_s{subject:02d}'
            cached=core.validate_completion(cache,identity_sha)
            arrays={}
            file_hashes={}
            if cached:
                for rid,start,stop in intervals:
                    name=f'{rid.replace(":","_")}_{start}_{stop}.npz'
                    with np.load(cache/name,allow_pickle=False) as saved:
                        arrays[(rid,start,stop)]=saved['features'].copy()
            else:
                cache.mkdir(parents=True,exist_ok=True)
                for rid in sorted({x[0] for x in intervals}):
                    values=np.loadtxt(raw_root/records[rid]['raw_path'],dtype=np.float32,delimiter=',',usecols=range(9),ndmin=2)
                    eeg,report=select_openbci_eeg_channels(values)
                    if not report.packet_counter_removed or len(eeg)!=int(records[rid]['samples']):
                        raise ValueError('Channel/sample mismatch: '+rid)
                    for _,start,stop in (x for x in intervals if x[0]==rid):
                        filtered=sosfiltfilt(notch,eeg[start:stop],axis=0).astype(np.float32)
                        features,_=extract_features(filtered,sfreq=250.,win_sec=8.5,stride_sec=.5,
                                                    bands=bands,config=config,device=device,chunk_size=64)
                        if features.shape[1]!=272 or not np.isfinite(features).all():
                            raise ValueError('Invalid feature extraction: '+rid)
                        arrays[(rid,start,stop)]=features
                        name=f'{rid.replace(":","_")}_{start}_{stop}.npz'
                        core.write_npz(cache/name,features=features)
                        file_hashes[name]=core.sha(cache/name)
                core.write_json(cache/'complete.json',{'identity_sha256':identity_sha,'files_sha256':file_hashes})
            for job in cell_jobs:
                result,reused=core.run_job(job,arrays,identity_sha)
                results.append({'job_id':job['job_id'],'reused':reused,'fold_metrics':result['fold_metrics']})
                print(f'{len(results)}/{len(jobs)} {task} S{subject:02d} {job["arm"]}: '+('reused' if reused else 'fit'),flush=True)
        core.write_json(output/'execution_summary.json',{'identity_sha256':identity_sha,'selected_jobs':len(jobs),
            'new_fits':sum(not r['reused'] for r in results),'reused_fits':sum(r['reused'] for r in results),
            'jobs_by_arm':dict(Counter(j['arm'] for j in jobs)), 'results':results})


if __name__=='__main__':
    main()
