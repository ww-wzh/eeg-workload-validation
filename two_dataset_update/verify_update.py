"""Read-only verification of the portable two-dataset evidence update.

Checks saved results without raw EEG, CUDA, or training checkpoints.
The optional full gate replay uses the original independently checked function.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
JOURNAL = HERE / 'analyses'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_files():
    entries = (HERE / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
    for line in entries:
        expected, name = line.split('  ', 1)
        path = HERE / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed package file: {name}')
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checksums-only', action='store_true')
    parser.add_argument('--full-gates', action='store_true')
    args = parser.parse_args()
    result = {'manifest_files_checked': verify_files()}
    print('Package checksums verified.', flush=True)
    if args.checksums_only:
        print(json.dumps({**result, 'status': 'passed'}, indent=2))
        return

    import numpy as np
    import pandas as pd
    from scipy.special import expit
    from statsmodels.stats.multitest import multipletests
    from threadpoolctl import threadpool_limits

    def read(path):
        return pd.read_csv(path, float_precision='round_trip')

    def load(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def close(a, b, tol=1e-12):
        np.testing.assert_allclose(a, b, rtol=0, atol=tol)

    correction = JOURNAL / 'first_dataset_correction'
    spec = importlib.util.spec_from_file_location('published_corrected_gate', correction / 'corrected_gate.py')
    gate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate_module)
    FEATURES = gate_module.FEATURES
    fit_gate = gate_module.fit_gate
    with (HERE / 'SOURCE_INDEX.csv').open(encoding='utf-8', newline='') as stream:
        source_index = {r['publication_path']: r for r in csv.DictReader(stream)}
    for name, expected in load(correction / 'correction_verification.json')['source_files_sha256'].items():
        name = name.replace('\\', '/')
        source = source_index[name]
        assert source['original_sha256'] == expected, name
        assert digest(HERE / name) == source['publication_sha256'], name
    aggregation_count = 0
    for variant in ['ordinary', 'balanced']:
        folder = correction / f'{variant}_heldout'
        windows, records = read(folder / 'window_predictions.csv'), read(folder / 'recording_predictions.csv')
        keys = ['protocol', 'direction', 'subject', 'recording_id', 'method']
        expected = records.set_index(keys)
        assert not expected.index.duplicated().any()
        for key, group in windows.groupby(keys):
            prob = np.exp(np.log(np.clip(group[['p0', 'p1']].to_numpy(), 1e-12, 1)).mean(axis=0))
            prob /= prob.sum()
            row = expected.loc[key]
            close(prob, row[['p0', 'p1']].to_numpy(dtype=float))
            assert int(prob.argmax()) == int(row.pred_label)
            aggregation_count += 1
    ordinary = read(correction / 'ordinary_heldout/statistics/comparisons_BH14.csv')
    balanced = read(correction / 'balanced_heldout/statistics/balanced_gain_method_comparisons_BH7.csv')
    assert len(ordinary) == 14 and len(balanced) == 7
    close(multipletests(ordinary.exact_sign_flip_p, method='fdr_bh')[1], ordinary.bh_adjusted_p_14_family)
    close(multipletests(balanced.exact_sign_flip_p, method='fdr_bh')[1], balanced.bh_q_7_family)
    result['first_dataset_record_aggregations'] = aggregation_count
    result['first_dataset_BH_families'] = [14, 7]
    print('First-dataset aggregation and corrected BH families verified.', flush=True)

    session = JOURNAL / 'cogbci_session_validation'
    scores = read(session / 'participant_scores.csv')
    summary = read(session / 'summary.csv')
    assert set(scores.subject.unique()) == set(range(1, 30)) - {17, 25, 27, 28}
    for row in summary.itertuples():
        wide = scores[(scores.population == row.population) & (scores.task == row.task)].pivot(
            index='subject', columns='arm', values='record_ba').sort_index()
        a, b = wide.same_record.to_numpy(), wide.cross_session.to_numpy()
        d = a-b
        assert len(wide) == row.n_participants == 25
        close([a.mean(), b.mean(), d.mean()], [row.same_record_ba, row.cross_session_ba, row.difference])
        draws = np.random.default_rng(20261110).integers(len(d), size=(10000, len(d)))
        for values, prefix in [(a, 'same_record'), (b, 'cross_session'), (d, 'difference')]:
            close(np.quantile(values[draws].mean(axis=1), [.025, .975]),
                  [getattr(row, prefix+'_ci_low'), getattr(row, prefix+'_ci_high')])
        if row.population == 'primary':
            signs = np.random.default_rng(20261110).choice([-1, 1], size=(100000, len(d)))
            p = (1 + np.sum(np.abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-15))/100001
            close(p, row.p_signflip)
    primary = summary[summary.population == 'primary']
    close(multipletests(primary.p_signflip, method='fdr_bh')[1], primary.p_BH2)
    result['matched_session_summary_rows'] = len(summary)
    print('COG-BCI matched-session means, intervals, and BH2 verified.', flush=True)

    audit = JOURNAL / 'cogbci_gate_design'
    folder = JOURNAL / 'cogbci_gate_evaluation/analysis'
    protocol = load(audit / 'protocol.json')
    subjects = protocol['cohort']['expected_subjects']
    assert len(subjects) == 25 and set(subjects) == set(scores.subject.unique())
    participant = read(folder / 'participant_scores.csv')
    records = read(folder / 'record_scores.csv')
    for row in participant[participant['mode'] == 'ensemble'].itertuples():
        g = records[(records.task == row.task) & (records.subject == row.subject) &
                    (records.method == row.method) & (records['mode'] == 'ensemble')]
        assert len(g) > 0 and not g.record_id.duplicated().any()
        close(g[['p0', 'p1']].sum(axis=1), 1)
        assert np.array_equal(g[['p0', 'p1']].to_numpy().argmax(axis=1), g.prediction)
        correct = g.prediction.to_numpy() == g.label.to_numpy()
        ba = .5*(correct[g.label.to_numpy() == 0].mean()+correct[g.label.to_numpy() == 1].mean())
        close(ba, row.balanced_accuracy)
    main_scores = participant[participant['mode'] == 'ensemble']
    means = read(folder / 'method_summary.csv')
    comparisons = read(folder / 'comparisons_BH12.csv')
    rng = np.random.default_rng(protocol['statistics']['inference_seed'])
    bootstrap = rng.integers(0, len(subjects), (10000, len(subjects)))
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(100000, len(subjects)))
    for task in protocol['tasks']:
        wide = main_scores[main_scores.task == task].pivot(
            index='subject', columns='method', values='balanced_accuracy').loc[subjects]
        for row in means[means.task == task].itertuples():
            v = wide[row.method].to_numpy()
            close(v.mean(), row.mean)
            close(np.quantile(v[bootstrap].mean(axis=1), [.025, .975]), [row.ci_low, row.ci_high])
        for row in comparisons[comparisons.task == task].itertuples():
            left, right = row.contrast.split(' - ')
            d = (wide[left]-wide[right]).to_numpy()
            close(d.mean(), row.mean_difference)
            close(np.quantile(d[bootstrap].mean(axis=1), [.025, .975]), [row.ci_low, row.ci_high])
            null = np.abs((signs @ d)/len(d))
            close((np.sum(null >= abs(d.mean())-1e-12)+1)/100001, row.p_two_sided)
    assert len(comparisons) == 12
    close(multipletests(comparisons.p_two_sided, method='fdr_bh')[1], comparisons.p_BH12)
    meta, seed_rows = read(folder / 'meta_rows.csv'), read(folder / 'meta_seed_rows.csv')
    assert len(meta) == 2400 and len(seed_rows) == 12000
    median = seed_rows.groupby(['task', 'outer_subject', 'meta_subject', 'kind'])[
        ['neural_ba', 'fused_ba', 'benefit', *FEATURES]].median().sort_index()
    close(median.to_numpy(), meta.set_index(['task', 'outer_subject', 'meta_subject', 'kind'])[
        median.columns].sort_index().to_numpy())
    result['COG_gate_primary_comparisons'] = 12
    print('COG-BCI recording scores, gate summaries, intervals, and BH12 verified.', flush=True)
    if args.full_gates:
        scope = {'np': np, 'pd': pd, 'expit': expit, 'FEATURES': FEATURES, 'fit_gate': fit_gate}
        source = (HERE / 'verification/cog_gate_replay.py').read_text(encoding='utf-8')
        exec(compile(source, 'verification/cog_gate_replay.py', 'exec'), scope)
        print('Replaying all saved COG-BCI gate fits and decisions on CPU...', flush=True)
        with threadpool_limits(limits=1):
            result['COG_gate_replay'] = scope['check_gates'](
                meta, seed_rows, read(folder / 'gate_diagnostics.csv'),
                read(folder / 'meta_participant_deletion.csv'), subjects)
    result.update(status='passed', raw_EEG_used=False, EEG_models_retrained=0,
                  verification_scope='Saved probabilities, response tables, and participant-level statistics. Training checkpoints and raw-signal extraction are outside this portable check.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
