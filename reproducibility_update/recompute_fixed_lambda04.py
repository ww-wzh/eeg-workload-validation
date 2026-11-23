"""Reconstruct a post-hoc fixed 0.4 blend from archived window probabilities.

No model is trained and no original result is overwritten. Run from any folder.
The input is the Supplementary Data S1 ZIP distributed with release V1.2.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd

SEEDS = (1335, 1388, 1441, 1494, 1547)


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    if not (np.isfinite(p).all() and (p >= 0).all() and (p.sum(axis=1) > 0).all()):
        raise ValueError('Invalid probability matrix')
    return p / p.sum(axis=1, keepdims=True)


def aggregate(p, records):
    ids = np.unique(records)
    values = np.array([np.exp(np.log(np.clip(p[records == r], 1e-12, 1)).mean(axis=0)) for r in ids])
    return ids, normalize(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-zip', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='New output directory')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    records_out, people_out, summaries = [], [], []
    with zipfile.ZipFile(args.input_zip) as z:
        ref = pd.read_csv(io.BytesIO(z.read('inputs/gate_run/analysis/record_scores.csv')))
        ref = ref[ref['mode'] == 'ensemble']
        gates = pd.read_csv(io.BytesIO(z.read('inputs/gate_run/analysis/gate_diagnostics.csv')))
        gates = gates[(gates['mode'] == 'ensemble') & (gates.method == 'cbsf_heldout')]
        max_error = 0.0
        for task in ('NBACK', 'MATB'):
            subjects = sorted(ref[ref.task == task].subject.unique())
            assert len(subjects) == 25
            for subject in subjects:
                inputs = []
                for seed in SEEDS:
                    name = f'inputs/gate_run/jobs/{task}_outer_s{subject:02d}_seed{seed}/target_s{subject:02d}_predictions.npz'
                    with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as a:
                        inputs.append({k: a[k] for k in a.files})
                for a in inputs[1:]:
                    assert np.array_equal(a['record_ids'], inputs[0]['record_ids'])
                    assert np.array_equal(a['window_ids'], inputs[0]['window_ids'])
                records = inputs[0]['record_ids']
                neural = normalize(np.median([a['feature_mlp'] for a in inputs], axis=0))
                fused = normalize(np.median([a['always_fuse'] for a in inputs], axis=0))
                w = float(gates[(gates.task == task) & (gates.subject == subject)].iloc[0].weight)
                for method, weight in [('fixed_04', .4), ('fixed_half', .5), ('cbsf_heldout', w), ('feature_mlp', 0.), ('always_fuse', 1.)]:
                    ids, rp = aggregate(normalize((1-weight)*neural+weight*fused), records)
                    reference = ref[(ref.task == task) & (ref.subject == subject) & (ref.method == ('feature_mlp' if method == 'fixed_04' else method))].set_index('record_id').loc[ids]
                    labels = reference.label.to_numpy(dtype=int)
                    if method != 'fixed_04':
                        error = float(np.abs(rp-reference[['p0','p1']].to_numpy()).max())
                        max_error = max(max_error, error)
                        assert error < 2e-10, (task, subject, method, error)
                        assert np.array_equal(rp.argmax(axis=1), reference.prediction.to_numpy())
                    pred = rp.argmax(axis=1)
                    ba = .5*((pred[labels == 0] == 0).mean()+(pred[labels == 1] == 1).mean())
                    people_out.append(dict(task=task, subject=int(subject), method=method, balanced_accuracy=float(ba)))
                    if method in ('fixed_04','cbsf_heldout'):
                        records_out.extend(dict(task=task, subject=int(subject), method=method, record_id=str(r), label=int(y), p0=float(p[0]), p1=float(p[1]), prediction=int(p.argmax())) for r,y,p in zip(ids,labels,rp))
            frame = pd.DataFrame(people_out)
            wide = frame[frame.task == task].pivot(index='subject', columns='method', values='balanced_accuracy').loc[subjects]
            draws = np.random.default_rng(20261110).integers(25, size=(10000,25))
            ci = lambda v: np.quantile(np.asarray(v)[draws].mean(axis=1), [.025,.975])
            fixed, cb = wide.fixed_04.to_numpy(), wide.cbsf_heldout.to_numpy()
            d = np.round(cb-fixed, 12)
            fl,fh = ci(fixed); dl,dh = ci(d)
            summaries.append(dict(task=task,n=25,fixed_ba=float(fixed.mean()),fixed_ci_low=float(fl),fixed_ci_high=float(fh),cbsf_ba=float(cb.mean()),difference=float(d.mean()),ci_low=float(dl),ci_high=float(dh),better=int((d>0).sum()),tied=int((d==0).sum()),worse=int((d<0).sum())))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records_out).to_csv(args.out/'record_predictions.csv', index=False)
    pd.DataFrame(people_out).to_csv(args.out/'participant_scores.csv', index=False)
    pd.DataFrame(summaries).to_csv(args.out/'summary.csv', index=False)
    report = dict(status='passed',analysis='Post-hoc descriptive fixed lambda=0.4 blend; outside BH12; no new significance tests',aggregation='Seed-median normalized window probabilities, window-level blend, geometric aggregation within each recording',bootstrap_draws=10000,bootstrap_seed=20261110,max_original_probability_error=max_error,input_zip_sha256=hashlib.file_digest(args.input_zip.open('rb'),'sha256').hexdigest(),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (args.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
