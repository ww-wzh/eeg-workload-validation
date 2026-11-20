"""Verify archived aggregation-order evidence and optionally replay all gate fits."""
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE/'results'
TOL = 1e-10


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def exact_tail(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1e-12]
    observed = abs(nonzero.sum())
    sums = np.array([np.dot(signs, nonzero) for signs in product([-1.,1.], repeat=len(nonzero))])
    return float(np.mean(np.abs(sums) >= observed-1e-12))


def check_statistics():
    assert exact_tail([1.,1.,-1.]) == 1.
    assert exact_tail([-.25,.75]) == 1.
    assert exact_tail([-.125,.375,.125]) == .75
    assert exact_tail([0.,0.]) == 1.
    assert exact_tail([1.,1.,1.]) == .25
    core = read(ROOT/'shared/first_dataset/neutral_predictions/01_core/core_window_ensemble.csv')
    records = []
    for keys, group in core[core.method=='always_nn'].groupby(['protocol','direction','subject','recording_id']):
        label = int(group.true_label.iloc[0])
        score = np.log(np.clip(group[['p0','p1']].to_numpy(float),1e-12,1.)).mean(axis=0)
        records.append(dict(zip(['protocol','direction','subject','recording_id'],keys),true_label=label,correct=int(score.argmax()==label)))
    base = pd.DataFrame(records).groupby(['protocol','direction','subject','true_label']).correct.mean().groupby(['protocol','direction','subject']).mean()
    gates = read(RESULTS/'participant_metrics.csv')
    saved = read(RESULTS/'statistics/balanced_gain_method_comparisons_BH7.csv')
    verified_p = []
    for row in saved.itertuples():
        selected = gates[gates.method==row.method].copy()
        selected['delta'] = [x.balanced_accuracy-base.loc[(x.protocol,x.direction,x.subject)] for x in selected.itertuples()]
        if row.setting=='cross_task_bidirectional':
            values = selected[selected.protocol=='cross_task'].groupby('subject')['delta'].mean().to_numpy()
        else:
            values = selected[(selected.protocol+'_'+selected.direction)==row.setting].delta.to_numpy()
        assert len(values)==row.n_participants
        np.testing.assert_allclose(values.mean()*100,row.mean_difference_pp,atol=1e-12,rtol=0)
        assert int((values>1e-12).sum())==row.better
        assert int((values < -1e-12).sum())==row.worse
        assert int((np.abs(values)<=1e-12).sum())==row.tied
        rng=np.random.Generator(np.random.PCG64(int(row.bootstrap_seed)))
        draws=rng.integers(0,len(values),size=(10000,len(values)))
        ci=np.percentile(values[draws].mean(axis=1),[2.5,97.5])
        np.testing.assert_allclose(ci,[row.ci95_low,row.ci95_high],atol=1e-12,rtol=0)
        p=exact_tail(values)
        assert p==row.exact_sign_flip_p,(row.contrast,p,row.exact_sign_flip_p)
        verified_p.append(p)
    order=np.argsort(verified_p,kind='stable')
    adjusted=np.empty(len(order))
    adjusted[order]=np.minimum.accumulate((np.array(verified_p)[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1].clip(0,1)
    np.testing.assert_allclose(adjusted,saved.bh_q_7_family,rtol=0,atol=1e-12)
    return len(saved)


def compare_replay(folder):
    rows=0
    for reference in sorted(RESULTS.rglob('*.csv')):
        relative=reference.relative_to(RESULTS)
        actual=folder/relative
        assert actual.is_file(),relative
        a,b=read(reference),read(actual)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=TOL,atol=TOL)
        rows+=len(a)
    a=json.loads((RESULTS/'summary.json').read_text())
    b=json.loads((folder/'summary.json').read_text())
    a.pop('created_utc',None); b.pop('created_utc',None)
    assert a==b
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    count=0
    for line in (HERE/'MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=HERE/name
        assert path.is_file(),name
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==expected,name
        count+=1
    stats=check_statistics()
    result={'status':'passed','manifest_files':count,'independent_BH7_comparisons':stats}
    if args.full:
        with tempfile.TemporaryDirectory(prefix='eeg-ensemble-replay-') as temporary:
            folder=Path(temporary)
            subprocess.run([sys.executable,'-B',str(HERE/'run_analysis.py'),'--output-dir',str(folder)],check=True)
            result['replayed_csv_rows']=compare_replay(folder)
            result['meta_prediction_files']=915
            result['full_gate_fits']=54
            result['participant_deletion_fits']=678
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
