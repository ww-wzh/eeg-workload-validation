"""Run one COG-BCI session pair from original EEG, using the archived code.

This engineering check is restricted to one participant, pair, and seed. It is
not a cohort analysis. No network request is made and existing files stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sampling_audit(rv, code_dir, audited):
    """Rebuild sampling metadata without training additional models."""
    pairs = [p for p in rv.load(code_dir/'pair_plan.json')['pairs'] if p['primary']]
    seeds = rv.load(code_dir/'protocol.json')['seeds']
    support_map = {}
    for rid,row in audited.items():
        rejected = {r['tile'] for r in row.get('rejected_tiles',[])}
        support_map[rid] = np.asarray([s for i,s in enumerate(row['raw_supports']) if i not in rejected],dtype=np.int64)
    arrays = {rid:np.empty((len(s),0)) for rid,s in support_map.items()}
    seen, rows = {}, []
    for pair in pairs:
        for seed in seeds:
            test,same,_ = rv.select_indices(pair,seed,arrays)
            for c,rid in enumerate(pair['target_records']):
                key = (pair['task'],rid,seed)
                signature = (tuple(test[c]),tuple(same[c]))
                if key in seen:
                    assert seen[key] == signature, 'Target windows changed with source session'
                    continue
                seen[key] = signature
                starts = support_map[rid][:,0]/500.0
                for idx in test[c]:
                    nearest = float(np.min(np.abs(starts[same[c]]-starts[idx])))
                    assert nearest >= 12.5-1e-9
                    rows.append(dict(task=pair['task'],subject=pair['subject'],target_session=pair['target'],record_id=rid,seed=seed,test_support_index=int(idx),nearest_center_distance_s=nearest,feature_window_edge_gap_s=nearest-8.5,raw_support_edge_gap_s=nearest-12.5))
    frame = pd.DataFrame(rows)
    summary=[]
    for task,g in frame.groupby('task',sort=False):
        x=g.feature_window_edge_gap_s.to_numpy()
        summary.append(dict(task=task,n=len(x),min=float(x.min()),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75)),max=float(x.max()),all_source_session_reuse_checks_passed=True))
    return frame,pd.DataFrame(summary)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True, help='Original sub-02.zip from COG-BCI Version 4')
    ap.add_argument('--out', type=Path, required=True, help='Empty output directory')
    ap.add_argument('--code-dir', type=Path, default=root/'two_dataset_update/analyses/cogbci_session_validation')
    ap.add_argument('--reference-cache', type=Path, help='Optional original local session run containing features/ and fits/')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.code_dir.resolve()))
    import run_validation as rv
    import audit_inputs as audit
    import scipy
    import sklearn
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    pair_id, seed = 'sub-02_NBACK_S2_to_S3', 1335
    pair = next(p for p in rv.load(args.code_dir/'pair_plan.json')['pairs'] if p['pair_id'] == pair_id)
    assert pair['primary'] and pair['subject'] == 2
    inventory = {r['record_id']:r for r in rv.load(args.code_dir/'input_inventory.json')['records']}
    audited = {r['record_id']:r for r in rv.load(args.code_dir/'feature_audit.json')['records']}
    arrays, supports, hashes = {}, {}, {}
    ref_errors = []
    with zipfile.ZipFile(args.archive) as z, threadpool_limits(limits=1):
        for rid in pair['source_records']+pair['target_records']:
            row = inventory[rid]
            header = audit.header(z, row['set_member'])
            assert hashlib.sha256(z.read(row['set_member'])).hexdigest() == row['set_sha256']
            assert audit.segmentation(header, row['stem'])['raw_supports'] == [tuple(s) for s in row['raw_supports']]
            labels = [str(c.labels).strip() for c in np.atleast_1d(header['chanlocs'])]
            assert [labels.index(c) for c in audit.CHANNELS] == row['selected_channel_indices']
            raw = z.read(row['fdt_member'])
            hashes[rid] = hashlib.sha256(raw).hexdigest()
            assert hashes[rid] == audited[rid]['fdt_sha256']
            x = np.frombuffer(raw,dtype='<f4').reshape(row['pnts'],row['nbchan'])[:,row['selected_channel_indices']]
            feat, sup = [], []
            for start,stop in row['raw_supports']:
                signal = x[start:stop]
                if not np.isfinite(signal).all() or np.any(signal.std(axis=0,dtype=np.float64)<=1e-8):
                    continue
                f = rv.preprocess_support(signal)
                if np.isfinite(f).all():
                    feat.append(f); sup.append((start,stop))
            arrays[rid] = np.asarray(feat,dtype=np.float32).reshape(-1,272)
            supports[rid] = np.asarray(sup,dtype=np.int64)
            assert len(feat) == audited[rid]['n_valid_tiles'] and len(feat)>=20
            if args.reference_cache:
                with np.load(args.reference_cache/'features'/f'{rid}.npz',allow_pickle=False) as saved:
                    assert np.array_equal(supports[rid],saved['raw_supports'])
                    err = float(np.max(np.abs(arrays[rid].astype(float)-saved['features'])))
                    ref_errors.append(err)
                    assert err < 1e-5, (rid,err)
    test,same,cross = rv.select_indices(pair,seed,arrays)
    for c in (0,1):
        assert set(test[c]).isdisjoint(same[c])
    # Explicitly construct both test arrays rather than trusting a shared name.
    xt_same = np.concatenate([arrays[r][ii] for r,ii in zip(pair['target_records'],test)])
    xt_cross = np.concatenate([arrays[r][ii] for r,ii in zip(pair['target_records'],test)])
    assert np.array_equal(xt_same,xt_cross) and len(xt_same)==20
    window_rows, record_rows, fitted = [], [], []
    with threadpool_limits(limits=1):
        for arm,rids,indices,xt in [('same_record',pair['target_records'],same,xt_same),('cross_session',pair['source_records'],cross,xt_cross)]:
            train = np.concatenate([arrays[r][ii] for r,ii in zip(rids,indices)])
            model = rv.fit_one(train,np.repeat([0,1],10),xt)
            rb,wb,rp = rv.score(model['prob'])
            fitted.append(model['prob'])
            for c,rid in enumerate(pair['target_records']):
                record_rows.append(dict(pair_id=pair_id,seed=seed,arm=arm,record_id=rid,label=c,p0=float(rp[c,0]),p1=float(rp[c,1]),prediction=int(rp[c].argmax()),record_balanced_accuracy=rb,window_balanced_accuracy=wb))
                for j,idx in enumerate(test[c]):
                    start,stop=supports[rid][idx]
                    window_rows.append(dict(arm=arm,record_id=rid,label=c,support_index=int(idx),raw_start=int(start),raw_stop=int(stop),p0=float(model['prob'][c*10+j,0]),p1=float(model['prob'][c*10+j,1])))
    prediction_error = None
    if args.reference_cache:
        with np.load(args.reference_cache/'fits'/f'{pair_id}.npz',allow_pickle=False) as saved:
            si = rv.load(args.code_dir/'protocol.json')['seeds'].index(seed)
            assert np.array_equal(test,saved['test_indices'][si])
            assert np.array_equal(same,saved['train_same_indices'][si])
            assert np.array_equal(cross,saved['train_cross_indices'][si])
            prediction_error = float(np.max(np.abs(np.asarray(fitted)-saved['prob'][si])))
            assert prediction_error < 1e-8
    summary = dict(status='passed',scope='One participant, one directed session pair, one seed; engineering reproduction only',pair_id=pair_id,seed=seed,test_windows_per_arm=20,training_windows_per_arm=20,test_windows_identical=True,raw_supports_disjoint_within_target=True,feature_max_abs_difference=max(ref_errors) if ref_errors else None,archived_prediction_max_abs_difference=prediction_error,raw_fdt_sha256=hashes,code_sha256={name:digest(args.code_dir/name) for name in ['run_validation.py','audit_inputs.py','neutral_feature_extractor.py','protocol.json']},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__))
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(window_rows).to_csv(args.out/'window_predictions.csv',index=False)
    pd.DataFrame(record_rows).to_csv(args.out/'record_predictions.csv',index=False)
    (args.out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    gaps,gap_summary=sampling_audit(rv,args.code_dir,audited)
    gaps.to_csv(args.out/'sampling_distances.csv',index=False)
    gap_summary.to_csv(args.out/'sampling_distance_summary.csv',index=False)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
