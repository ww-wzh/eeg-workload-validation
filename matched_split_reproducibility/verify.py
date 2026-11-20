"""Verify split membership and recompute statistics from saved recording predictions."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from reconstruct_splits import check as check_splits

ROOT = Path(__file__).resolve().parent
METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision",
           "f1", "auroc", "pr_auc", "average_precision", "log_loss", "brier_score"]


def read(name):
    return pd.read_csv(ROOT / name, float_precision="round_trip")


def compare(actual, expected, keys):
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    assert actual.index.is_unique and expected.index.is_unique
    assert actual.index.equals(expected.index), keys
    assert set(actual.columns) == set(expected.columns)
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-10,
                                       atol=1e-12, equal_nan=True, err_msg=column)
        else:
            assert actual[column].tolist() == expected[column].tolist(), column


def metrics(group):
    y = group.true_label.to_numpy(int)
    pred = group.pred_label.to_numpy(int)
    raw = group.p1.to_numpy(float)
    p = np.clip(raw, 1e-12, 1 - 1e-12)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(n_recordings=len(group), n_class_0=int(sum(y == 0)), n_class_1=int(sum(y == 1)),
                tn=tn, fp=fp, fn=fn, tp=tp, accuracy=float(np.mean(y == pred)),
                balanced_accuracy=(sensitivity + specificity) / 2,
                sensitivity=sensitivity, specificity=specificity,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                f1=2 * tp / (2 * tp + fp + fn), auroc=roc_auc_score(y, p),
                pr_auc=auc(recall, precision), average_precision=average_precision_score(y, p),
                log_loss=float(-np.mean(np.log(np.where(y == 1, p, 1 - p)))),
                brier_score=float(np.mean((raw - y) ** 2)))


def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    values = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(1, values)
    return result


def tests(delta):
    delta = np.round(np.asarray(delta, dtype=float), 12)
    nonzero = delta[np.abs(delta) > 1e-15]
    if len(nonzero):
        result = wilcoxon(delta, zero_method="pratt", correction=False,
                          alternative="two-sided", method="approx")
        statistic, p = float(result.statistic), float(result.pvalue)
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(nonzero))), dtype=float)
        exact = float(np.mean(np.abs(signs @ nonzero) >= abs(float(nonzero.sum())) - 1e-12))
    else:
        statistic, p, exact = 0.0, 1.0, 1.0
    return dict(wilcoxon_statistic=statistic, wilcoxon_raw_p=p, exact_sign_flip_p=exact,
                n_zero_differences=int(sum(delta == 0)), n_nonzero_differences=int(sum(delta != 0)))


def recompute():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    predictions = read("recording_predictions.csv")
    assert len(predictions) == 2912 and predictions.fit_key.nunique() == 868
    requests = json.loads((ROOT / "fold_requests.json").read_text(encoding="utf-8"))
    assert set(predictions.fit_key) == set(requests)
    for key, group in predictions.groupby("fit_key"):
        request = requests[key]["request"]
        expected = dict(zip(request["test_recordings"], request["test_labels"]))
        assert group.recording_id.is_unique
        assert dict(zip(group.recording_id, group.true_label)) == expected
        assert group.task.eq(request["task"]).all()
        assert group.subject.eq(request["subject"]).all()
        assert group.strategy.eq(request["setting"]).all()
    probs = predictions[["p0", "p1"]].to_numpy()
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_array_equal(probs.argmax(axis=1), predictions.pred_label)
    keys = ["task", "strategy", "subject", "evaluation_block"]
    rows = []
    for key, group in predictions.groupby(keys, sort=True):
        assert len(group) == group.recording_id.nunique() == 4
        assert sorted(group.true_label) == [0, 0, 1, 1]
        rows.append(dict(zip(keys, key)) | metrics(group))
    blocks = pd.DataFrame(rows)
    assert len(blocks) == 728
    compare(blocks, read("block_metrics.csv"), keys)
    rows = []
    for (task, strategy, subject), group in blocks.groupby(keys[:3], sort=True):
        count = 1 if strategy == "participant_disjoint" else 5 if strategy == "recording_disjoint" else 10
        assert len(group) == count
        row = dict(task=task, strategy=strategy, subject=subject, evaluation_blocks=count)
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[metric + "_valid_blocks"] = len(values)
        rows.append(row)
    participant = pd.DataFrame(rows)
    assert len(participant) == 112
    compare(participant, read("participant_metrics.csv"), keys[:3])
    summary, contrasts, differences = [], [], []
    for task, n in protocol["data"]["tasks"].items():
        frame = participant.loc[participant.task.eq(task)]
        pivot = frame.pivot(index="subject", columns="strategy", values="balanced_accuracy").sort_index()
        assert pivot.index.tolist() == list(range(1, n + 1)) and not pivot.isna().any().any()
        namespace = protocol["statistics"]["seed_namespace"]
        seed = int.from_bytes(hashlib.sha256(f"{namespace}|{task}".encode()).digest()[:8], "little")
        index = np.random.default_rng(seed).integers(0, n, (10000, n))
        for strategy in protocol["strategy_display"]:
            group = frame.loc[frame.strategy.eq(strategy)].set_index("subject").sort_index()
            for metric in METRICS:
                values = group[metric].to_numpy(float)
                finite = np.isfinite(values)
                sample = values[index]
                counts = np.isfinite(sample).sum(axis=1)
                valid = counts > 0
                boot = np.nansum(sample[valid], axis=1) / counts[valid]
                low, high = np.quantile(boot, [0.025, 0.975], method="linear") if len(boot) else (np.nan, np.nan)
                summary.append(dict(task=task, strategy=strategy,
                                    strategy_display=protocol["strategy_display"][strategy], metric=metric,
                                    n_participants=n, n_valid_participants=int(sum(finite)),
                                    n_valid_bootstrap_draws=int(sum(valid)),
                                    participant_macro_mean=float(values[finite].mean()) if finite.any() else np.nan,
                                    participant_sd_proportion=float(values[finite].std(ddof=1)) if sum(finite) > 1 else np.nan,
                                    ci95_low=low, ci95_high=high))
        for family, name in [("primary_six", "primary_comparisons"),
                             ("fraction_sensitivity_six", "sensitivity_comparisons")]:
            for left, right in protocol["statistics"][name]:
                delta = (pivot[left] - pivot[right]).to_numpy(float)
                low, high = np.quantile(delta[index].mean(axis=1) * 100, [0.025, 0.975], method="linear")
                contrasts.append(dict(task=task, family=family, left=left, right=right,
                                      n_participants=n, left_mean=float(pivot[left].mean()),
                                      right_mean=float(pivot[right].mean()), mean_difference_pp=float(delta.mean() * 100),
                                      ci95_low_pp=low, ci95_high_pp=high) | tests(delta))
                differences.extend(dict(task=task, family=family, left=left, right=right,
                                        subject=subject, paired_difference_pp=float(d * 100))
                                   for subject, d in zip(pivot.index, delta))
    contrasts = pd.DataFrame(contrasts)
    for column, stem in [("wilcoxon_raw_p", "wilcoxon"), ("exact_sign_flip_p", "sign_flip")]:
        contrasts[stem + "_bh_family6"] = np.nan
        for _, indices in contrasts.groupby("family").groups.items():
            assert len(indices) == 6
            contrasts.loc[indices, stem + "_bh_family6"] = bh(contrasts.loc[indices, column])
        contrasts[stem + "_bh_all12_sensitivity"] = bh(contrasts[column])
    summary = pd.DataFrame(summary)
    differences = pd.DataFrame(differences)
    compare(summary, read("summary_intervals.csv"), ["task", "strategy", "metric"])
    compare(contrasts, read("comparisons.csv"), ["task", "family", "left", "right"])
    compare(differences, read("participant_differences.csv"), ["task", "family", "left", "right", "subject"])
    table = read("split_summary_display.csv")
    from decimal import Decimal, ROUND_HALF_UP
    fmt = lambda x: str(Decimal(str(x * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    for _, row in table.iterrows():
        match = summary.loc[summary.task.eq(row.Task.lower()) & summary.strategy_display.eq(row['Split strategy'])
                            & summary.metric.eq('balanced_accuracy')].iloc[0]
        value = f"{fmt(match.participant_macro_mean)} ({fmt(match.ci95_low)} to {fmt(match.ci95_high)})"
        assert value == row['Balanced accuracy percent (95% CI)'], (value, row.to_dict())
    return {"block_metrics.csv": blocks, "participant_metrics.csv": participant,
            "summary_intervals.csv": summary, "comparisons.csv": contrasts,
            "participant_differences.csv": differences}


def verify_manifest():
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional new directory for recomputed tables")
    args = parser.parse_args()
    verify_manifest()
    check_splits()
    result = recompute()
    report = dict(status="passed", folds=868, recording_predictions=2912,
                  evaluation_blocks=728, participant_strategy_scores=112, metric_summaries=88,
                  paired_comparisons=12, split_summary_rows=8,
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  scipy=scipy.__version__, scikit_learn=sklearn.__version__)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, frame in result.items():
            frame.to_csv(args.output_dir / name, index=False, float_format="%.17g")
        (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
