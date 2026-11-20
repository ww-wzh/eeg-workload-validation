"""Read-only source audit and outcome-independent task segmentation."""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'cogbci'
ARCHIVES = DATA / 'raw_archives'
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C4']
TASKS = {'NBACK': ['zeroBACK', 'twoBACK'], 'MATB': ['MATBeasy', 'MATBdiff']}
FS = 500
TILE = 6250

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def header(z, member):
    return loadmat(io.BytesIO(z.read(member)), squeeze_me=True, struct_as_record=False)

def events(d):
    return [(str(e.type).strip(), float(e.latency) - 1.0) for e in np.atleast_1d(d['event'])]

def segmentation(d, stem):
    ev = sorted(events(d), key=lambda x: x[1])
    n = int(d['pnts'])
    boundaries = [int(np.ceil(v)) for k, v in ev if k.lower() == 'boundary' and 0 < v < n]
    intervals = []
    detail = {}
    if stem in TASKS['NBACK']:
        prefix = '60' if stem == 'zeroBACK' else '62'
        onset_codes = {prefix + '21', prefix + '22', prefix + '23'}
        onsets = [v for k, v in ev if k in onset_codes]
        ends = [v for k, v in ev if k == prefix + '12']
        other_codes = {'6021', '6022', '6023', '6121', '6122', '6123', '6221', '6222', '6223'} - onset_codes
        if any(k in other_codes for k, v in ev):
            raise ValueError('Unexpected task-level onset code')
        if len(ends) != 3:
            raise ValueError(f'Expected 3 block ends, found {len(ends)}')
        previous = -1.0
        counts, isi = [], []
        for end in ends:
            ons = np.array([v for v in onsets if previous < v < end])
            if len(ons) < 20 or np.max(np.diff(ons)) > 5 * FS:
                raise ValueError('Insufficient onsets or unexplained within-block gap')
            intervals.append((int(np.ceil(ons[0])), min(n, int(np.floor(end)))))
            counts.append(len(ons))
            isi.append(float(np.median(np.diff(ons)) / FS))
            previous = end
        if sum(counts) != len(onsets):
            raise ValueError('Unassigned onset')
        detail = {'block_onset_counts': counts, 'median_isi_seconds': isi, 'interval_rule': 'first stimulus onset to explicit block end; start markers unused'}
    else:
        token = 'easy' if stem == 'MATBeasy' else 'diff'
        marked = [v for k, v in ev if token in k.split()]
        if len(marked) < 20:
            raise ValueError('Missing nominal MATB condition markers')
        other = {'easy', 'diff', 'med'} - {token}
        if any(other.intersection(k.split()) for k, v in ev):
            raise ValueError('Conflicting MATB condition markers')
        ends = [v for k, v in ev if k.lower() == (stem + 'end').lower()]
        if len(ends) != 1:
            raise ValueError(f'Expected one task end, found {len(ends)}')
        intervals = [(max(0, int(np.ceil(min(marked)))), min(n, int(np.floor(ends[0]))))]
        detail = {'condition_marker_count': len(marked), 'interval_rule': 'first condition marker to explicit task end'}
    # Never let a support tile cross a discontinuity in the delivered EEGLAB file.
    continuous = []
    for start, stop in intervals:
        cuts = [start] + sorted(set(b for b in boundaries if start < b < stop)) + [stop]
        continuous.extend(zip(cuts[:-1], cuts[1:]))
    supports = [(a + k*TILE, a + (k+1)*TILE) for a,b in continuous for k in range((b-a)//TILE)]
    return {'active_intervals_raw': intervals, 'continuous_intervals_raw': list(continuous),
            'raw_supports': supports, 'n_tiles_before_signal_qc': len(supports),
            'internal_boundaries': boundaries, **detail}

def inventory():
    n = loadmat(DATA / 'resources/notebook.mat', squeeze_me=True, struct_as_record=False)['notebook']
    sessions, records, problems = [], [], []
    for subject in range(1,30):
        with zipfile.ZipFile(ARCHIVES / f'sub-{subject:02d}.zip') as z:
            names = z.namelist()
            for ses in range(1,4):
                entry = getattr(getattr(n, f'SBJ_{subject}'), f'SESS_{ses}')
                sessions.append({'subject': subject, 'session': ses, 'interrupted': bool(entry.interrupted), 'notes': str(getattr(entry, 'Notes', ''))})
                for task, stems in TASKS.items():
                    for label, stem in enumerate(stems):
                        suffix = f'/ses-S{ses}/eeg/{stem}.set'
                        matches = [m for m in names if m.endswith(suffix)]
                        if len(matches) != 1:
                            raise ValueError((subject, suffix, matches))
                        member = matches[0]
                        d = header(z, member)
                        labels = [str(c.labels).strip() for c in np.atleast_1d(d['chanlocs'])]
                        row = {'record_id': f'sub-{subject:02d}_S{ses}_{stem}', 'subject': subject, 'session': ses, 'task': task, 'label': label,
                               'stem': stem, 'set_member': member, 'fdt_member': member[:-4] + '.fdt',
                               'pnts': int(d['pnts']), 'nbchan': int(d['nbchan']), 'sampling_rate': float(d['srate']),
                               'selected_channel_indices': [labels.index(c) for c in CHANNELS],
                               'all_channel_labels': labels, 'interrupted': bool(entry.interrupted),
                               'set_sha256': hashlib.sha256(z.read(member)).hexdigest()}
                        assert row['sampling_rate'] == FS and len(set(labels)) == len(labels)
                        assert row['fdt_member'] in names
                        assert z.getinfo(row['fdt_member']).file_size == row['nbchan'] * row['pnts'] * 4
                        try:
                            row.update(segmentation(d, stem))
                            row['header_eligible'] = True
                        except ValueError as exc:
                            row['header_eligible'] = False
                            row['header_exclusion_reason'] = str(exc)
                            problems.append((row['record_id'], str(exc)))
                        records.append(row)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'records': records, 'sessions': sessions,
              'resource_sha256': {p.name: sha(p) for p in [DATA/'resources/notebook.mat', DATA/'resources/triggerlist.txt']},
              'statement': 'No model fitted; eligibility is based on source headers, event markers and acquisition notebook.'}
    dump(HERE/'input_inventory.json', report)
    print(json.dumps({'records':len(records), 'header_problems':problems,
                      'flagged_sessions':[(s['subject'],s['session']) for s in sessions if s['interrupted']],
                      'tile_count_range':[min(r['n_tiles_before_signal_qc'] for r in records if r['header_eligible']), max(r['n_tiles_before_signal_qc'] for r in records if r['header_eligible'])],
                      'isi_counts':dict(Counter(round(x,3) for r in records for x in r.get('median_isi_seconds',[])))}, indent=2))

if __name__ == '__main__':
    inventory()
