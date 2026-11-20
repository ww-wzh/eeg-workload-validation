"""Reconstruct every published split from recording metadata and fixed seeds."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype}|{value.shape}|".encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def reconstruct(root=ROOT):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(root / "recordings.csv")
    arrays = {}
    for task, expected_n in protocol["data"]["tasks"].items():
        frame = metadata.loc[metadata.task.eq(task)].sort_values("record_index")
        assert frame.record_index.tolist() == list(range(4 * expected_n))
        ids = frame.recording_id.to_numpy(dtype=str)
        subjects = frame.subject.to_numpy(dtype=int)
        labels = frame.binary_label.to_numpy(dtype=np.int64)
        record_indices = np.arange(len(frame), dtype=np.int64)
        window_records = np.repeat(record_indices, frame.n_windows.to_numpy(dtype=int))
        starts = np.r_[0, np.cumsum(frame.n_windows.to_numpy(dtype=int))[:-1]]
        np.testing.assert_array_equal(starts, frame.first_window_row)
        arrays[f"{task}__window_record_index"] = window_records
        arrays[f"{task}__recording_ids"] = ids

        def store(key, train, test):
            train = np.sort(np.asarray(train, dtype=np.int64))
            test = np.sort(np.asarray(test, dtype=np.int64))
            assert not np.intersect1d(train, test).size
            assert len(train) and len(test)
            arrays[key + "__train"] = train
            arrays[key + "__test"] = test

        for subject in range(1, expected_n + 1):
            rec = record_indices[subjects == subject]
            assert len(rec) == 4 and labels[rec].tolist() == [0, 0, 1, 1]
            rows = np.flatnonzero(np.isin(window_records, rec))
            for seed in protocol["splits"]["seeds"]:
                by_class = [np.asarray(sorted(ids[rec][labels[rec] == cls]), dtype=object)
                            for cls in (0, 1)]
                rng = np.random.default_rng(seed)
                for group in by_class:
                    rng.shuffle(group)
                for fold in range(2):
                    test_ids = {by_class[0][fold], by_class[1][fold]}
                    test_rec = rec[np.isin(ids[rec], list(test_ids))]
                    train_rec = rec[~np.isin(ids[rec], list(test_ids))]
                    key = f"rd/{task}/{subject:02d}/5/{seed}/{fold}"
                    store(key, np.flatnonzero(np.isin(window_records, train_rec)),
                          np.flatnonzero(np.isin(window_records, test_rec)))
                for strategy, fraction in (("random_window_88_12", 0.12),
                                           ("random_window_50_50", 0.5)):
                    splitter = StratifiedShuffleSplit(n_splits=2, test_size=fraction,
                                                     random_state=seed)
                    y = labels[window_records[rows]]
                    for draw, (train, test) in enumerate(splitter.split(np.zeros(len(rows)), y)):
                        assert set(window_records[rows[train]]) == set(rec)
                        assert set(window_records[rows[test]]) == set(rec)
                        store(f"{strategy}/{task}/{subject:02d}/{seed}/{draw}",
                              rows[train], rows[test])
            train = np.flatnonzero(subjects[window_records] != subject)
            store(f"pd/{task}/{subject:02d}/0000", train, rows)
    return arrays


def check(root=ROOT):
    arrays = reconstruct(root)
    requests = json.loads((root / "fold_requests.json").read_text(encoding="utf-8"))
    diagnostics = pd.read_csv(root / "fold_diagnostics.csv").set_index("fit_key")
    assert len(requests) == len(diagnostics) == 868
    assert {k.removesuffix("__train") for k in arrays if k.endswith("__train")} == set(requests)
    with np.load(root / "window_indices.npz", allow_pickle=False) as saved:
        assert set(arrays) == set(saved.files)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, saved[key], err_msg=key)
    for key, item in requests.items():
        request = item["request"]
        task = request["task"]
        ids = arrays[f"{task}__recording_ids"]
        mapping = arrays[f"{task}__window_record_index"]
        labels = np.array([int(r.rsplit(":r", 1)[1]) >= 2 for r in ids], dtype=np.int64)
        for part in ("train", "test"):
            rows = arrays[f"{key}__{part}"]
            assert array_hash(rows) == request[f"{part}_rows_sha256"], key
            assert len(rows) == request[f"{part}_windows"], key
            actual_ids = ids[np.unique(mapping[rows])].tolist()
            assert actual_ids == request[f"{part}_recordings"], key
        train_labels = labels[mapping[arrays[key + "__train"]]]
        assert array_hash(train_labels) == diagnostics.loc[key, "training_window_labels_sha256"]
    return arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new NPZ path for reconstructed indices")
    args = parser.parse_args()
    result = check()
    if args.output:
        if args.output.exists():
            raise FileExistsError("Choose a new output path; existing files are preserved")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **result)
    print("PASS: all 868 splits match the saved indices and executed request hashes.")
