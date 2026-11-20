"""Analyze all frozen split controls at participant level, without fitting."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
HERE = BASE
PLAN = BASE
KEYS = ["task", "subject", "panel", "arm"]
LEVELS = {0: "Natural", 1: "Low", 2: "Medium", 3: "High"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, f"Existing analysis output differs: {path}"
    else:
        with path.open("xb") as stream:
            stream.write(content)


def save_csv(name, values):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
    save(HERE / name, buffer.getvalue().encode("utf-8-sig"))


def grouped(values, fields):
    result = defaultdict(list)
    for value in values:
        result[tuple(value[k] for k in fields)].append(value)
    return sorted(result.items())


def score(values):
    correctness = np.array([int(r["prediction"]) == int(r["binary_label"]) for r in values])
    labels = np.array([int(r["binary_label"]) for r in values])
    assert set(labels) == {0, 1}
    return {"balanced_accuracy": float(np.mean([correctness[labels == c].mean() for c in [0, 1]])),
            "accuracy": float(correctness.mean())}


def exact_sign_flip(differences):
    d = np.round(np.asarray(differences, dtype=float), 12)
    d = d[d != 0]
    if not len(d):
        return 1.0
    signs = 1.0 - 2.0 * ((np.arange(2 ** len(d), dtype=np.uint64)[:, None] >> np.arange(len(d), dtype=np.uint64)) & 1)
    sums = signs @ d
    return float(np.mean(abs(sums) >= abs(d.sum()) - 1e-12))


def bootstrap_ci(values, indices):
    distribution = np.asarray(values)[indices].mean(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def main():
    protocol = read(PLAN / "split_control_protocol.json")
    predictions = rows(HERE / "verified_recording_predictions.csv")
    fits = rows(HERE / "verified_fit_metrics.csv")
    for row in predictions + fits:
        for key in ["subject", "seed", "fold"]:
            row[key] = int(row[key])
    seed_rows = []
    for key, values in grouped(predictions, KEYS + ["seed"]):
        task, subject, panel, arm, seed = key
        folds = dict(grouped(values, ["fold"]))
        assert set(folds) == {(0,), (1,)}
        if arm == "recording_holdout_full":
            assert len(values) == len({r["recording_id"] for r in values}) == 4
            result = score(values)
            order = "pool complementary record folds, score, then mean seeds"
        else:
            assert len(values) == 8 and all(len(v) == 4 for v in folds.values())
            per_fold = [score(v) for v in folds.values()]
            result = {k: float(np.mean([s[k] for s in per_fold])) for k in ["balanced_accuracy", "accuracy"]}
            order = "score each fold, mean folds, then mean seeds"
        seed_rows.append(dict(zip(KEYS + ["seed"], key)) | result | {"aggregation": order})
    assert len(seed_rows) == 790
    participant_rows = []
    for key, values in grouped(seed_rows, KEYS):
        assert sorted(v["seed"] for v in values) == protocol["seeds"]
        participant_rows.append(dict(zip(KEYS, key)) |
                                {k: float(np.mean([v[k] for v in values])) for k in ["balanced_accuracy", "accuracy"]} |
                                {"seeds": 5, "folds_per_seed": 2})
    assert len(participant_rows) == 158
    for row in participant_rows:
        assert row["balanced_accuracy"] == row["accuracy"], "These controls have class-balanced record scoring"
    resampling = {}
    matrices = {}
    for (task, panel), values in grouped(participant_rows, ["task", "panel"]):
        subjects = sorted({r["subject"] for r in values})
        expected = list(range(1, 16)) if task == "arithmetic" else list(range(1, 14))
        expected.remove(7 if task == "arithmetic" else 4) if task == "arithmetic" or panel == "temporal" else None
        assert subjects == expected
        text = f"Sensors_split_controls_v1|{task}|{panel}"
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        indices = np.random.default_rng(seed).integers(0, len(subjects), size=(10000, len(subjects)), dtype=np.int64)
        buffer = io.BytesIO()
        np.save(buffer, indices, allow_pickle=False)
        name = f"resampling/{task}_{panel}_participant_indices.npy"
        save(HERE / name, buffer.getvalue())
        resampling[f"{task}/{panel}"] = {"subjects": subjects, "seed": seed, "matrix": name,
                                          "matrix_sha256": sha(HERE / name), "repeats": 10000,
                                          "shared_across_all_arms_and_contrasts": True}
        matrices[(task, panel)] = indices
    arm_rows, size_rows = [], []
    for key, values in grouped(participant_rows, ["task", "panel", "arm"]):
        task, panel, arm = key
        values.sort(key=lambda r: r["subject"])
        ba = [r["balanced_accuracy"] for r in values]
        lo, hi = bootstrap_ci(ba, matrices[(task, panel)])
        arm_rows.append(dict(zip(["task", "panel", "arm"], key)) |
                        {"participants": len(ba), "balanced_accuracy_percent": 100 * float(np.mean(ba)),
                         "ci95_lower_percent": 100 * lo, "ci95_upper_percent": 100 * hi})
        current_fits = [r for r in fits if (r["task"], r["panel"], r["arm"]) == key]
        sizes = [int(r["train_windows"]) for r in current_fits]
        # Counts are constant over seeds/folds for a participant: summarize people once.
        person_sizes = {r["subject"]: int(r["train_windows"]) for r in current_fits}
        size_rows.append(dict(zip(["task", "panel", "arm"], key)) |
                         {"participants": len(ba), "fits": len(current_fits), "train_windows_min": min(sizes),
                          "train_windows_median_across_participants": float(np.median(list(person_sizes.values()))),
                          "train_windows_max": max(sizes), "test_windows_equal_train": True})
    lookup = {(r["task"], r["panel"], r["arm"], r["subject"]): r["balanced_accuracy"] for r in participant_rows}
    comparisons, differences = [], []
    for task in ["arithmetic", "stroop"]:
        for a, b in protocol["statistics"]["comparisons_per_task"]:
            panel = "nonoverlap" if a == "random_dense_matched" else "temporal"
            subjects = resampling[f"{task}/{panel}"]["subjects"]
            d = np.array([lookup[(task, panel, a, s)] - lookup[(task, panel, b, s)] for s in subjects])
            lo, hi = bootstrap_ci(d, matrices[(task, panel)])
            p = exact_sign_flip(d)
            comparisons.append({"task": task, "panel": panel, "method_a": a, "method_b": b,
                                "participants": len(subjects), "difference_pp_a_minus_b": float(d.mean() * 100),
                                "ci95_lower_pp": 100 * lo, "ci95_upper_pp": 100 * hi,
                                "improve_a": int(np.sum(d > 1e-12)), "tie": int(np.sum(abs(d) <= 1e-12)),
                                "decline_a": int(np.sum(d < -1e-12)), "p_exact_two_sided": p})
            for s, value in zip(subjects, d):
                differences.append({"task": task, "panel": panel, "method_a": a, "method_b": b,
                                    "subject": s, "balanced_accuracy_a": lookup[(task, panel, a, s)],
                                    "balanced_accuracy_b": lookup[(task, panel, b, s)], "difference_pp": float(value * 100)})
    pvalues = np.array([r["p_exact_two_sided"] for r in comparisons])
    order = np.argsort(pvalues, kind="stable")
    ordered_q = np.minimum.accumulate((pvalues[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1]
    q = np.empty(len(order))
    q[order] = np.minimum(ordered_q, 1)
    for row, value in zip(comparisons, q):
        row["q_BH8"] = float(value)
    workload_people = []
    for key, values in grouped(predictions, KEYS + ["raw_level"]):
        level = int(key[-1])
        assert len({v["recording_id"] for v in values}) == 1
        expected_count = 5 if key[3] == "recording_holdout_full" else 10
        assert len(values) == expected_count
        correct = [int(v["prediction"]) == int(v["binary_label"]) for v in values]
        positive = [float(v["probability_1"]) for v in values]
        true_prob = [float(v[f"probability_{v['binary_label']}"]) for v in values]
        workload_people.append(dict(zip(KEYS, key[:-1])) |
                               {"raw_level": level, "workload": LEVELS[level], "correctness": float(np.mean(correct)),
                                "probability_class1": float(np.mean(positive)), "probability_true_class": float(np.mean(true_prob)),
                                "scored_predictions_over_folds_and_seeds": len(values)})
    workload_summary = []
    for key, values in grouped(workload_people, ["task", "panel", "arm", "raw_level", "workload"]):
        workload_summary.append(dict(zip(["task", "panel", "arm", "raw_level", "workload"], key)) |
                                {"participants": len(values), "correctness_percent": float(np.mean([r["correctness"] for r in values]) * 100),
                                 "probability_class1": float(np.mean([r["probability_class1"] for r in values])),
                                 "probability_true_class": float(np.mean([r["probability_true_class"] for r in values]))})
    directions = []
    temporal = [r for r in predictions if r["arm"] in {"temporal_block_full", "temporal_block_local"}]
    for key, values in grouped(temporal, KEYS + ["fold"]):
        per_seed = [score(v) for _, v in grouped(values, ["seed"])]
        assert len(per_seed) == 5
        directions.append(dict(zip(KEYS + ["fold"], key)) |
                          {"direction": "Early to late" if key[-1] == 0 else "Late to early",
                           "balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in per_seed]))})
    direction_summary = []
    for key, values in grouped(directions, ["task", "panel", "arm", "fold", "direction"]):
        direction_summary.append(dict(zip(["task", "panel", "arm", "fold", "direction"], key)) |
                                 {"participants": len(values), "balanced_accuracy_percent": float(np.mean([r["balanced_accuracy"] for r in values]) * 100)})
    outputs = {"split_control_seed_metrics.csv": seed_rows,
               "split_control_participant_metrics.csv": participant_rows,
               "split_control_arm_summary.csv": arm_rows,
               "split_control_sample_size_summary.csv": size_rows,
               "split_control_comparisons_BH8.csv": comparisons,
               "split_control_participant_contrasts.csv": differences,
               "split_control_workload_by_participant.csv": workload_people,
               "split_control_workload_summary.csv": workload_summary,
               "split_control_temporal_direction_by_participant.csv": directions,
               "split_control_temporal_direction_summary.csv": direction_summary}
    for name, values in outputs.items():
        save_csv(name, values)
    manifest = {"status": "all_preplanned_descriptives_and_eight_comparisons_computed",
                "source_sha256": sha(Path(__file__)), "protocol_file_sha256": sha(PLAN / "split_control_protocol.json"),
                "input_sha256": {n: sha(HERE / n) for n in ["independent_full_fit_verification.json", "verified_recording_predictions.csv", "verified_fit_metrics.csv"]},
                "output_sha256": {n: sha(HERE / n) for n in outputs}, "output_rows": {n: len(v) for n, v in outputs.items()},
                "resampling": resampling, "sign_flip": protocol["statistics"]["sign_flip"],
                "limits": ["Exploratory after review, not preregistered or confirmatory.",
                           "Participant is the statistical unit; seeds, folds and windows are not independent inferential samples.",
                           "Tasks analyzed separately; cross-task natural duplicates are not independent replication.",
                           "Bootstrap intervals are conditional on these saved fits; sign-flip assumes sign-exchangeability.",
                           "Recording context and within-class workload levels cannot be causally separated by this dataset.",
                           "Nonoverlap contrast changes sampling density and temporal coverage, as well as shared raw samples.",
                           "A fixed logistic learner was used; these new controls do not re-estimate the original Feature-MLP gap."]}
    save(HERE / "split_control_analysis_manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"arm_summaries": arm_rows, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    HERE = args.output.resolve()
    if HERE == BASE:
        raise ValueError("Choose a separate analysis output directory")
    HERE.mkdir(parents=True, exist_ok=True)
    for name in ["verified_recording_predictions.csv", "verified_fit_metrics.csv", "independent_full_fit_verification.json"]:
        shutil.copyfile(BASE / name, HERE / name)
    main()
