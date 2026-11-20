"""Independently rederive aggregation, intervals, exact tests and BH8."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def group(values, keys):
    result = defaultdict(list)
    for row in values:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def balanced_accuracy(values):
    confusion = np.zeros((2, 2), dtype=int)
    for row in values:
        confusion[int(row["binary_label"]), int(row["prediction"])] += 1
    assert np.all(confusion.sum(axis=1) > 0)
    return float((confusion.diagonal() / confusion.sum(axis=1)).mean())


def close(a, b, tolerance=1e-10):
    assert abs(float(a) - float(b)) <= tolerance, (a, b)


def main():
    manifest = read(HERE / "split_control_analysis_manifest.json")
    recording = rows("verified_recording_predictions.csv")
    key = ["task", "subject", "panel", "arm"]
    seeds = {}
    for identity, values in group(recording, key + ["seed"]).items():
        if identity[3] == "recording_holdout_full":
            assert len(values) == len({r["recording_id"] for r in values}) == 4
            score = balanced_accuracy(values)
        else:
            per_fold = group(values, ["fold"])
            assert len(per_fold) == 2
            score = sum(balanced_accuracy(v) for v in per_fold.values()) / 2
        seeds[identity] = score
    for row in rows("split_control_seed_metrics.csv"):
        close(row["balanced_accuracy"], seeds[tuple(row[k] for k in key + ["seed"])])
    participant = {}
    for identity in {k[:-1] for k in seeds}:
        values = [v for k, v in seeds.items() if k[:-1] == identity]
        assert len(values) == 5
        participant[identity] = sum(values) / 5
    for row in rows("split_control_participant_metrics.csv"):
        close(row["balanced_accuracy"], participant[tuple(row[k] for k in key)])
    indices = {}
    for label, resampling in manifest["resampling"].items():
        task, panel = label.split("/")
        subjects = sorted({int(k[1]) for k in participant if (k[0], k[2]) == (task, panel)})
        assert subjects == resampling["subjects"]
        seed = int.from_bytes(hashlib.sha256(f"Sensors_split_controls_v1|{task}|{panel}".encode()).digest()[:8], "little")
        assert seed == resampling["seed"]
        generator = np.random.Generator(np.random.PCG64(seed))
        regenerated = generator.integers(len(subjects), size=(10000, len(subjects)), dtype=np.int64)
        assert sha(HERE / resampling["matrix"]) == resampling["matrix_sha256"]
        stored = np.load(HERE / resampling["matrix"], allow_pickle=False)
        assert np.array_equal(stored, regenerated)
        indices[(task, panel)] = stored
    for row in rows("split_control_arm_summary.csv"):
        task, panel, arm = row["task"], row["panel"], row["arm"]
        subjects = manifest["resampling"][f"{task}/{panel}"]["subjects"]
        scores = np.array([participant[(task, str(s), panel, arm)] for s in subjects])
        distribution = np.mean(scores[indices[(task, panel)]], axis=1) * 100
        low, high = np.percentile(distribution, [2.5, 97.5], method="linear")
        assert len(subjects) == int(row["participants"])
        close(row["balanced_accuracy_percent"], np.mean(scores) * 100)
        close(row["ci95_lower_percent"], low)
        close(row["ci95_upper_percent"], high)
    comparisons = rows("split_control_comparisons_BH8.csv")
    differences = {(r["task"], r["method_a"], r["method_b"], r["subject"]): r for r in rows("split_control_participant_contrasts.csv")}
    pvalues = []
    for row in comparisons:
        task, panel, a, b = row["task"], row["panel"], row["method_a"], row["method_b"]
        subjects = manifest["resampling"][f"{task}/{panel}"]["subjects"]
        d = []
        for subject in subjects:
            va, vb = participant[(task, str(subject), panel, a)], participant[(task, str(subject), panel, b)]
            saved = differences[(task, a, b, str(subject))]
            close(saved["balanced_accuracy_a"], va)
            close(saved["balanced_accuracy_b"], vb)
            close(saved["difference_pp"], 100 * (va - vb))
            d.append(va - vb)
        d = np.array(d)
        close(row["difference_pp_a_minus_b"], d.mean() * 100)
        rounded = np.round(d, 12)
        nonzero = rounded[rounded != 0]
        # Deliberately use explicit products, rather than the analyzer's bit array.
        if len(nonzero):
            observed = abs(sum(nonzero))
            extreme = sum(abs(sum(sign * value for sign, value in zip(signs, nonzero))) >= observed - 1e-12
                          for signs in itertools.product([-1, 1], repeat=len(nonzero)))
            p = extreme / (2 ** len(nonzero))
        else:
            p = 1.0
        close(row["p_exact_two_sided"], p, 1e-15)
        pvalues.append(p)
        bootstrap = [float(np.mean(d[ix]) * 100) for ix in indices[(task, panel)]]
        low, high = np.percentile(bootstrap, [2.5, 97.5], method="linear")
        close(row["ci95_lower_pp"], low)
        close(row["ci95_upper_pp"], high)
        assert int(row["improve_a"]) == sum(d > 1e-12)
        assert int(row["tie"]) == sum(abs(d) <= 1e-12)
        assert int(row["decline_a"]) == sum(d < -1e-12)
    assert len(pvalues) == 8
    ranked = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    for rank, (original, _) in enumerate(ranked, 1):
        adjusted = min(1.0, min(p * 8 / position for position, (_, p) in enumerate(ranked, 1) if position >= rank))
        close(comparisons[original]["q_BH8"], adjusted, 1e-15)
    workload = group(recording, key + ["raw_level"])
    workload_people = rows("split_control_workload_by_participant.csv")
    for row in workload_people:
        values = workload[tuple(row[k] for k in key + ["raw_level"])]
        correctness = [int(r["prediction"]) == int(r["binary_label"]) for r in values]
        close(row["correctness"], sum(correctness) / len(values))
        close(row["probability_class1"], np.mean([float(r["probability_1"]) for r in values]))
        close(row["probability_true_class"], np.mean([float(r[f"probability_{r['binary_label']}"]) for r in values]))
    for row in rows("split_control_workload_summary.csv"):
        selected = [p for p in workload_people if all(row[k] == p[k] for k in ["task", "panel", "arm", "raw_level", "workload"])]
        assert len(selected) == int(row["participants"])
        close(row["correctness_percent"], np.mean([float(p["correctness"]) for p in selected]) * 100)
        for field in ["probability_class1", "probability_true_class"]:
            close(row[field], np.mean([float(p[field]) for p in selected]))
    fit_metrics = rows("verified_fit_metrics.csv")
    for row in rows("split_control_temporal_direction_by_participant.csv"):
        selected = [v for v in fit_metrics if all(v[k] == row[k] for k in key + ["fold"])]
        assert len(selected) == 5
        close(row["balanced_accuracy"], np.mean([float(v["balanced_accuracy"]) for v in selected]))
    direction_people = rows("split_control_temporal_direction_by_participant.csv")
    for row in rows("split_control_temporal_direction_summary.csv"):
        selected = [p for p in direction_people if all(p[k] == row[k] for k in ["task", "panel", "arm", "fold", "direction"])]
        assert len(selected) == int(row["participants"])
        close(row["balanced_accuracy_percent"], np.mean([float(v["balanced_accuracy"]) for v in selected]) * 100)
    for row in rows("split_control_sample_size_summary.csv"):
        selected = [v for v in fit_metrics if all(v[k] == row[k] for k in ["task", "panel", "arm"])]
        counts = {v["subject"]: int(v["train_windows"]) for v in selected}
        assert len(counts) == int(row["participants"]) and len(selected) == int(row["fits"])
        assert int(row["train_windows_min"]) == min(counts.values())
        assert int(row["train_windows_max"]) == max(counts.values())
        close(row["train_windows_median_across_participants"], np.median(list(counts.values())))
    print(json.dumps({"status": "saved_split_statistics_verified", "seed_rows": len(seeds),
        "participant_rows": len(participant), "BH_comparisons": len(comparisons),
        "participant_contrasts": len(differences), "new_model_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
