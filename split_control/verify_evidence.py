"""Verify saved split-control plans, membership hashes and participant statistics."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def read_rows(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def canonical_sha(value):
    value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    for line in (HERE / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    import plan_geometry as geometry
    import verify_statistics
    records = read_rows('raw_input_identity.csv')
    for row in records:
        for key in ['subject', 'raw_level', 'binary_label', 'samples', 'frozen_windows',
                    'early_candidate_windows', 'late_candidate_windows', 'nonoverlap_candidate_windows']:
            row[key] = int(row[key])
        row['duration_seconds_at_250_hz'] = float(row['duration_seconds_at_250_hz'])
    geometry.self_check()
    plans = [json.loads(line) for line in (HERE / 'planned_folds.jsonl').read_text(encoding='utf-8').splitlines()]
    rebuilt = geometry.build_jobs(records, geometry.geometry(records))
    assert plans == rebuilt
    planned_audits = geometry.validate_jobs(plans, records)
    saved_audits = read_rows('planned_fit_membership_audit.csv')
    assert len(planned_audits) == len(saved_audits) == 1580
    for a, b in zip(planned_audits, saved_audits):
        assert all(str(value) == b[key] for key, value in a.items())
    hashes = {row['job_id']: row for row in read_rows('executed_membership_hashes.csv')}
    fits = {row['job_id']: row for row in read_rows('verified_fit_metrics.csv')}
    completed = {row['job_id']: row for row in read_rows('completed_job_index.csv')}
    predictions = read_rows('verified_recording_predictions.csv')
    ids = {j['job_id'] for j in plans}
    assert ids == set(hashes) == set(fits) == set(completed)
    assert len(predictions) == 5800
    assert {r['job_id'] for r in predictions} == ids
    for job in plans:
        job_id = job['job_id']
        assert job_id == canonical_sha({k: v for k, v in job.items() if k != 'job_id'})
        assert hashes[job_id]['job_canonical_sha256'] == job_id
        for role, short in [('training', 'train'), ('testing', 'test')]:
            membership = [{'recording_id': p['recording_id'], 'raw_level': p['raw_level'],
                           'binary_label': p['binary_label'], 'start': start,
                           'filter_interval': p['filter_interval']}
                          for p in sorted(job[role], key=lambda p: p['raw_level'])
                          for start in sorted(p['window_starts'])]
            assert canonical_sha(membership) == hashes[job_id][short + '_membership_sha256']
            count = len(membership)
            assert count == int(hashes[job_id][short + '_windows']) == int(fits[job_id][short + '_windows'])
            assert count == int(completed[job_id][short + '_windows'])
        for prediction in (r for r in predictions if r['job_id'] == job_id):
            part = next(p for p in job['testing'] if p['recording_id'] == prediction['recording_id'])
            assert int(prediction['binary_label']) == part['binary_label']
            assert int(prediction['windows']) == len(part['window_starts'])
            p0, p1 = float(prediction['probability_0']), float(prediction['probability_1'])
            assert 0 <= p0 <= 1 and 0 <= p1 <= 1 and abs(p0 + p1 - 1) < 1e-12
            assert int(prediction['prediction']) == int(p1 > p0)
    verify_statistics.main()
    print(json.dumps({'status': 'passed', 'planned_and_executed_jobs': len(ids),
        'recording_predictions': len(predictions), 'plan_rebuilt_exactly': True,
        'membership_hashes_verified': 3160, 'new_model_fits': 0}, indent=2))


if __name__ == '__main__':
    main()
