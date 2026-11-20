"""Export machine-readable COG-BCI source-audit tables from saved audit records."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(output_dir):
    verification = load(HERE / 'verification.json')
    if not (verification['model_count'] == 3000
            and verification['max_saved_model_probability_error'] == 0):
        raise ValueError('Source-table export requires the verified reference run.')
    audit = load(HERE / 'feature_audit.json')
    duplicates = load(HERE / 'duplicate_provenance_audit.json')
    excluded = set(duplicates['affected_participant_labels'])
    group_ids = {record: index + 1 for index, group in enumerate(duplicates['groups'])
                 for record in group['records']}
    records = []
    fields = ['record_id', 'subject', 'session', 'task', 'label', 'pnts',
              'nbchan', 'sampling_rate', 'interrupted', 'n_valid_tiles', 'fdt_sha256']
    for record in audit['records']:
        row = {key: record[key] for key in fields}
        row.update({
            'quarantined_participant': record['subject'] in excluded,
            'duplicate_group': group_ids.get(record['record_id'], ''),
            'stimulus_interval_seconds': (np.median(record['median_isi_seconds'])
                                         if record['task'] == 'NBACK' else ''),
            'set_member': record['set_member'],
            'fdt_member': record['fdt_member'],
        })
        records.append(row)
    pairs = [{'record_a': group['records'][0], 'record_b': group['records'][1],
              'bytes': group['fdt_bytes'], 'sha256': group['fdt_sha256'],
              'byte_equal': group['fdt_byte_for_byte_equal'],
              'set_byte_equal': group['set_files_byte_for_byte_equal']}
             for group in duplicates['groups']]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / 'source_record_audit.csv', index=False)
    pd.DataFrame(pairs).to_csv(output_dir / 'duplicate_pairs.csv', index=False)
    return {'source_records': len(records), 'duplicate_pairs': len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=HERE / 'recomputed' / 'source_audit')
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir), indent=2))


if __name__ == '__main__':
    main()
