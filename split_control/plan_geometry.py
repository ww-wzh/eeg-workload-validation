"""Frozen window geometry and deterministic plan construction."""
from __future__ import annotations
import hashlib
import json
import itertools
from collections import Counter, defaultdict
import numpy as np
FS, WINDOW, STRIDE, EDGE, GAP = 250, 2125, 125, 1250, 2500
SEEDS = [1335, 1388, 1441, 1494, 1547]


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def grid(start, stop, step=STRIDE):
    """Window starts inside half-open [start, stop), with complete windows only."""
    return list(range(start, stop - WINDOW + 1, step)) if stop - start >= WINDOW else []


def blocks(n, gap=GAP, edge=EDGE):
    # Align all starts to the existing 125-sample cache lattice.
    if gap % (2 * STRIDE) or edge % STRIDE:
        raise ValueError("Gap/edge must preserve the frozen cache lattice")
    middle = (n // 2 // STRIDE) * STRIDE
    early_end, late_start = middle - gap // 2, middle + gap // 2
    return {
        "early_filter_interval": [0, early_end],
        "late_filter_interval": [late_start, n],
        "early": grid(edge, early_end - edge),
        "late": grid(late_start + edge, n - edge),
    }


def rng_for(*parts):
    seed = int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed)


def choose(values, count, *parts):
    if len(values) < count:
        raise ValueError("Insufficient candidate windows; never silently resample or shorten a fold")
    return sorted(rng_for(*parts).choice(values, size=count, replace=False).astype(int).tolist())


def overlap_pairs(a, b):
    # Half-open intervals; touching edges are not shared samples.
    return sum(abs(int(x) - int(y)) < WINDOW for x in a for y in b)


def self_check():
    assert overlap_pairs([0], [WINDOW]) == 0
    assert overlap_pairs([0], [WINDOW - 1]) == 1
    assert grid(0, WINDOW) == [0]
    assert grid(0, WINDOW - 1) == []
    b = blocks(25000)
    assert b["late_filter_interval"][0] - b["early_filter_interval"][1] == GAP
    assert not overlap_pairs(b["early"], b["late"])
    assert all(s % STRIDE == 0 for s in b["early"] + b["late"])
    assert choose(range(100), 8, "check") == choose(range(100), 8, "check")
    assert not blocks(6500)["early"]
    return 9


def geometry(records):
    rows = []
    for task, count in [("arithmetic", 15), ("stroop", 13)]:
        for subject in range(1, count + 1):
            current = [r for r in records if r["task"] == task and r["subject"] == subject]
            q = min(min(r["early_candidate_windows"], r["late_candidate_windows"]) for r in current)
            nonoverlap_q = min(r["nonoverlap_candidate_windows"] // 2 for r in current)
            rows.append({
                "task": task, "subject": subject,
                "temporal_eligible": bool(q > 0), "temporal_windows_per_record_per_partition": q,
                "temporal_train_windows_per_fit": 4 * q, "temporal_test_windows_per_fit": 4 * q,
                "nonoverlap_eligible": bool(nonoverlap_q > 0),
                "nonoverlap_windows_per_record_per_partition": nonoverlap_q,
                "nonoverlap_train_windows_per_fit": 4 * nonoverlap_q,
                "nonoverlap_test_windows_per_fit": 4 * nonoverlap_q,
                "shortest_record_seconds": min(r["duration_seconds_at_250_hz"] for r in current),
                "temporal_exclusion_recordings": ";".join(r["recording_id"] for r in current
                    if not r["early_candidate_windows"] or not r["late_candidate_windows"]),
            })
    return rows


def partition(entry, starts, filter_interval, role):
    return {"recording_id": entry["recording_id"], "binary_label": entry["binary_label"],
            "raw_level": entry["raw_level"], "role": role,
            "window_starts": list(starts), "filter_interval": list(filter_interval)}


def job(task, subject, seed, fold, panel, arm, training, testing):
    value = {"task": task, "subject": subject, "seed": seed, "fold": fold,
             "panel": panel, "arm": arm, "training": training, "testing": testing}
    value["job_id"] = digest(value)
    return value


def build_jobs(records, participants):
    jobs = []
    for person in participants:
        task, subject = person["task"], person["subject"]
        current = [r for r in records if r["task"] == task and r["subject"] == subject]
        for seed in SEEDS:
            if person["temporal_eligible"]:
                q = person["temporal_windows_per_record_per_partition"]
                selections = {}
                for r in current:
                    rid, n = r["recording_id"], r["samples"]
                    b = blocks(n)
                    early = choose(b["early"], q, "temporal", rid, seed, "early")
                    late = choose(b["late"], q, "temporal", rid, seed, "late")
                    draw = rng_for("random-block-pool", rid, seed).permutation(b["early"] + b["late"])
                    selections[rid] = (b, early, late, sorted(draw[:q].astype(int).tolist()),
                                       sorted(draw[q:2*q].astype(int).tolist()))
                # Predefined label-balanced complementary recording folds.
                low = rng_for("record-pair", task, subject, seed, 0).permutation([0, 1]).tolist()
                high = rng_for("record-pair", task, subject, seed, 1).permutation([2, 3]).tolist()
                for fold in range(2):
                    for arm in ["random_block_pool_full", "temporal_block_full", "temporal_block_local"]:
                        train, test = [], []
                        for r in current:
                            b, early, late, random_a, random_b = selections[r["recording_id"]]
                            full = [0, r["samples"]]
                            if arm == "random_block_pool_full":
                                a, z, fa, fz = random_a, random_b, full, full
                            else:
                                a, z = early, late
                                fa, fz = (b["early_filter_interval"], b["late_filter_interval"]) if arm.endswith("local") else (full, full)
                            if fold:
                                a, z, fa, fz = z, a, fz, fa
                            train.append(partition(r, a, fa, "train"))
                            test.append(partition(r, z, fz, "test"))
                        jobs.append(job(task, subject, seed, fold, "temporal", arm, train, test))
                    train, test = [], []
                    train_levels = {low[fold], high[fold]}
                    for r in current:
                        starts = choose(grid(EDGE, r["samples"] - EDGE), 2*q,
                                        "record-size-match", r["recording_id"], seed)
                        role = "train" if r["raw_level"] in train_levels else "test"
                        target = train if role == "train" else test
                        target.append(partition(r, starts, [0, r["samples"]], role))
                    jobs.append(job(task, subject, seed, fold, "temporal", "recording_holdout_full", train, test))
            if person["nonoverlap_eligible"]:
                q = person["nonoverlap_windows_per_record_per_partition"]
                for arm, step in [("random_dense_matched", STRIDE), ("random_nonoverlap_matched", WINDOW)]:
                    selections = {}
                    for r in current:
                        pool = grid(EDGE, r["samples"] - EDGE, step)
                        draw = rng_for("nonoverlap", arm, r["recording_id"], seed).permutation(pool)
                        selections[r["recording_id"]] = (sorted(draw[:q].astype(int).tolist()),
                                                         sorted(draw[q:2*q].astype(int).tolist()))
                    for fold in range(2):
                        train, test = [], []
                        for r in current:
                            a, z = selections[r["recording_id"]]
                            if fold:
                                a, z = z, a
                            full = [0, r["samples"]]
                            train.append(partition(r, a, full, "train"))
                            test.append(partition(r, z, full, "test"))
                        jobs.append(job(task, subject, seed, fold, "nonoverlap", arm, train, test))
    return jobs


def validate_jobs(jobs, records):
    lookup = {r["recording_id"]: r for r in records}
    audit = []
    for j in jobs:
        train = {p["recording_id"]: p for p in j["training"]}
        test = {p["recording_id"]: p for p in j["testing"]}
        tr_n = sum(len(p["window_starts"]) for p in train.values())
        te_n = sum(len(p["window_starts"]) for p in test.values())
        assert tr_n == te_n and tr_n > 0
        counts = []
        for role, parts in [("train", train), ("test", test)]:
            label_counts = Counter()
            for rid, part in parts.items():
                r = lookup[rid]
                assert r["task"] == j["task"] and r["subject"] == j["subject"]
                assert part["role"] == role and part["binary_label"] == r["binary_label"]
                starts = part["window_starts"]
                assert starts == sorted(set(starts)) and starts
                a, b = part["filter_interval"]
                assert 0 <= a < b <= r["samples"]
                assert all(a + EDGE <= s and s + WINDOW <= b - EDGE and s % STRIDE == 0 for s in starts)
                label_counts[r["binary_label"]] += len(starts)
            assert label_counts[0] == label_counts[1] > 0
            counts.append(dict(label_counts))
        shared = sorted(set(train) & set(test))
        pairs = sum(overlap_pairs(train[r]["window_starts"], test[r]["window_starts"]) for r in shared)
        assert all(not (set(train[r]["window_starts"]) & set(test[r]["window_starts"])) for r in shared)
        if j["arm"] in {"temporal_block_full", "temporal_block_local", "random_nonoverlap_matched", "recording_holdout_full"}:
            assert pairs == 0
        if j["arm"] == "temporal_block_local":
            for r in shared:
                a, b = train[r]["filter_interval"], test[r]["filter_interval"]
                assert max(a[0], b[0]) - min(a[1], b[1]) >= GAP
        if j["arm"] == "recording_holdout_full":
            assert not shared and len(train) == len(test) == 2
        if j["arm"] == "random_nonoverlap_matched":
            for r in shared:
                combined = sorted(train[r]["window_starts"] + test[r]["window_starts"])
                assert all(b-a >= WINDOW for a, b in zip(combined, combined[1:]))
        audit.append({"job_id": j["job_id"], "task": j["task"], "subject": j["subject"],
                      "panel": j["panel"], "arm": j["arm"], "seed": j["seed"], "fold": j["fold"],
                      "train_windows": tr_n, "test_windows": te_n,
                      "shared_recordings": len(shared), "cross_partition_overlapping_window_pairs": pairs})
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    by_key = {}
    for j in jobs:
        key = (j["task"], j["subject"], j["seed"], j["fold"], j["panel"])
        by_key.setdefault(key, []).append(j)
    for group in by_key.values():
        # Counts match exactly across arms within each panel and binary class.
        assert len({sum(len(p["window_starts"]) for p in j["training"]) for j in group}) == 1
        arms = {j["arm"]: j for j in group}
        if "temporal_block_full" in arms:
            a, b = arms["temporal_block_full"], arms["temporal_block_local"]
            for role in ["training", "testing"]:
                assert [(p["recording_id"], p["window_starts"]) for p in a[role]] == [
                    (p["recording_id"], p["window_starts"]) for p in b[role]]
    # Both complementary folds reverse roles exactly, including filter intervals.
    by_pair = {}
    for j in jobs:
        key = (j["task"], j["subject"], j["seed"], j["panel"], j["arm"])
        by_pair.setdefault(key, {})[j["fold"]] = j
    for pair in by_pair.values():
        assert set(pair) == {0, 1}
        for left, right in [("training", "testing"), ("testing", "training")]:
            def canonical(parts):
                return sorted((p["recording_id"], tuple(p["window_starts"]), tuple(p["filter_interval"])) for p in parts)
            assert canonical(pair[0][left]) == canonical(pair[1][right])
    return audit
