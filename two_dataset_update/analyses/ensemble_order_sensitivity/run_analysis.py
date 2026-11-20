"""Recompute the balanced-gain gate response in final five-seed order."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "shared/first_dataset"
META = BASE / "loso_gain_inputs"
NEUTRAL = BASE / "neutral_predictions"
PREVIOUS = HERE.parent / "first_dataset_correction/balanced_heldout"
OUT = HERE / "recomputed"
SEEDS = {1335, 1388, 1441, 1494, 1547}
TOL = 1e-12
sys.path.insert(0, str(HERE.parent / "first_dataset_correction"))
from corrected_gate import FEATURES, fit_gate, exact_sign_flip


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def verify_saved_jobs() -> list[Path]:
    import gzip
    index = json.loads((HERE / "inputs/meta_index.json").read_text(encoding="utf-8"))
    assert index["completed_jobs"] == 915 and len(index["jobs"]) == 915
    paths = []
    for item in index["jobs"]:
        path = HERE / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Compressed prediction checksum mismatch: {item['job_id']}")
        with gzip.open(path, "rb") as stream:
            original = hashlib.sha256(stream.read()).hexdigest()
        if original != item["uncompressed_sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {item['job_id']}")
        paths.append(path)
    assert len(set(paths)) == 915
    return paths



def balanced_accuracy(frame: pd.DataFrame) -> float:
    recalls = frame.groupby("true_label").correct.mean()
    if set(recalls.index.astype(int)) != {0, 1}:
        raise ValueError("Each response unit must contain both classes")
    return float(recalls.mean())


def recording_predictions(windows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in windows.groupby(keys, sort=False, dropna=False):
        values = np.clip(group[["p0", "p1"]].to_numpy(dtype=np.float64), 1e-12, 1.0)
        probability = np.exp(np.mean(np.log(values), axis=0))
        probability /= probability.sum()
        label = int(group.true_label.iloc[0])
        prediction = int(np.argmax(probability))
        rows.append({**dict(zip(keys, key)), "true_label": label, "pred_label": prediction,
                     "correct": int(prediction == label), "p0": probability[0], "p1": probability[1],
                     "n_windows": int(len(group))})
    return pd.DataFrame(rows)


def loso_response_rows(files: list[Path]) -> pd.DataFrame:
    meta = read_csv(META / "02_all_meta_recording_seed.csv")
    labels = meta.drop_duplicates(
        ["direction", "outer_subject", "meta_subject", "recording_id"]
    )[["direction", "outer_subject", "meta_subject", "recording_id", "true_label"]].rename(
        columns={"direction": "task"}
    )
    grouped: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    for path in files:
        job_id = path.parent.name
        match = re.fullmatch(r"p(\d+)_(\d+)_s(\d+)", job_id)
        if match:
            # The task is encoded by the target prediction rows, not by the pair directory.
            sample = read_csv(path)
            task = str(sample.task.iloc[0])
            grouped.setdefault((task, int(match.group(1)), int(match.group(2))), []).append(
                (int(match.group(3)), path)
            )
            continue
        # The archived first pair predates the current zero-padded directory names.
        sample = read_csv(path)
        task = str(sample.task.iloc[0])
        outer = sorted({int(sample.outer_subject.min()), int(sample.outer_subject.max())})
        grouped.setdefault((task, outer[0], outer[1]), []).append((int(sample.seed.iloc[0]), path))

    rows = []
    expected_pairs = {("arithmetic", i, j) for i in range(1, 16) for j in range(i + 1, 16)}
    expected_pairs |= {("stroop", i, j) for i in range(1, 14) for j in range(i + 1, 14)}
    if set(grouped) != expected_pairs:
        raise ValueError(f"Unexpected pair grid: {len(grouped)} groups")

    for pair, members in sorted(grouped.items()):
        if {seed for seed, _ in members} != SEEDS or len(members) != 5:
            raise ValueError(f"Incomplete seeds for pair {pair}")
        raw = pd.concat([read_csv(path) for _, path in sorted(members)], ignore_index=True)
        raw = raw[raw.method.isin(["always_nn", "always_fuse"])].copy()
        if set(raw.method) != {"always_nn", "always_fuse"}:
            raise ValueError(f"Unexpected branch set for pair {pair}")
        keys = ["task", "outer_subject", "meta_subject", "method", "recording_id", "window_id"]
        seed_counts = raw.groupby(keys, dropna=False).seed.nunique()
        if not (seed_counts == 5).all():
            raise ValueError(f"A window lacks five seed predictions for pair {pair}")
        windows = raw.groupby(keys, as_index=False, dropna=False)[["p0", "p1"]].median()
        mass = windows[["p0", "p1"]].sum(axis=1)
        windows[["p0", "p1"]] = windows[["p0", "p1"]].div(mass, axis=0)
        windows = windows.merge(labels, on=["task", "outer_subject", "meta_subject", "recording_id"],
                                validate="many_to_one")
        rec = recording_predictions(
            windows,
            ["task", "outer_subject", "meta_subject", "method", "recording_id"],
        )
        for (task, outer, meta_subject), group in rec.groupby(["task", "outer_subject", "meta_subject"]):
            score = group.groupby("method").apply(balanced_accuracy, include_groups=False)
            rows.append({"task": task, "outer_subject": int(outer), "meta_subject": int(meta_subject),
                         "benefit": float(score["always_fuse"] - score["always_nn"]),
                         "aggregation": "median_seed_per_window_then_recording_geometric"})
    result = pd.DataFrame(rows)
    if len(result) != 366 or result.duplicated(["task", "outer_subject", "meta_subject"]).any():
        raise ValueError("Unexpected LOSO response grid")
    return result.sort_values(["task", "outer_subject", "meta_subject"]).reset_index(drop=True)


def cross_response_rows() -> pd.DataFrame:
    source = read_csv(HERE / "inputs/seed_predictions.csv.gz")
    source = source[(source.protocol == "cross_task") & source.method.isin(["always_nn", "always_fuse"])]
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method"]
    seed_counts = source.groupby(keys, dropna=False).seed.nunique()
    if not (seed_counts == 5).all():
        raise ValueError("Cross-task window does not have all five seed predictions")
    label_counts = source.groupby(keys, dropna=False).true_label.nunique()
    if not (label_counts == 1).all():
        raise ValueError("Cross-task window labels differ across seeds")
    labels = source.groupby(keys, as_index=False, dropna=False).true_label.first()
    windows = source.groupby(keys, as_index=False, dropna=False)[["p0", "p1"]].median().merge(
        labels, on=keys, validate="one_to_one"
    )
    mass = windows[["p0", "p1"]].sum(axis=1)
    windows[["p0", "p1"]] = windows[["p0", "p1"]].div(mass, axis=0)
    rec = recording_predictions(windows, ["protocol", "subject", "direction", "recording_id", "method"])
    rows = []
    for (subject, direction), group in rec.groupby(["subject", "direction"]):
        score = group.groupby("method").apply(balanced_accuracy, include_groups=False)
        rows.append({"subject": int(subject), "direction": direction,
                     "benefit": float(score["always_fuse"] - score["always_nn"]),
                     "aggregation": "median_seed_per_window_then_recording_geometric"})
    result = pd.DataFrame(rows)
    if len(result) != 26 or result.duplicated(["subject", "direction"]).any():
        raise ValueError("Unexpected cross-task response grid")
    return result.sort_values(["subject", "direction"]).reset_index(drop=True)


def fit_new_gates(loso_response: pd.DataFrame, cross_response: pd.DataFrame,
                  core: pd.DataFrame, helper) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_loso = read_csv(PREVIOUS / "training_rows_loso.csv")
    old_cross = read_csv(PREVIOUS / "training_rows_cross_task.csv")
    diag = read_csv(PREVIOUS / "gate_diagnostics.csv")

    loso = old_loso.drop(columns=["benefit", "loss_event"], errors="ignore").merge(
        loso_response[["task", "outer_subject", "meta_subject", "benefit"]],
        on=["task", "outer_subject", "meta_subject"], validate="one_to_one",
    )
    loso["loss_event"] = loso.benefit <= -0.05
    cross = old_cross.drop(columns=["benefit", "loss_event"], errors="ignore").merge(
        cross_response[["subject", "direction", "benefit"]],
        on=["subject", "direction"], validate="one_to_one",
    )
    cross["loss_event"] = cross.benefit <= -0.05

    output_rows, prediction_rows, window_rows, deletion_rows = [], [], [], []
    for row in diag[diag.method.str.startswith("cbsf")].itertuples(index=False):
        target = {key: float(getattr(row, key)) for key in FEATURES}
        if row.protocol == "cross_task":
            train = cross[cross.subject.astype(int) != int(row.subject)].copy()
            if int(row.subject) in set(train.meta_subject.astype(int)):
                raise ValueError("Cross-task outer participant entered gate training")
        else:
            train = loso[(loso.task == row.direction) &
                         (loso.outer_subject.astype(int) == int(row.subject))].copy()
            if int(row.subject) in set(train.meta_subject.astype(int)):
                raise ValueError("LOSO outer participant entered gate training")
        gate = fit_gate(train, target)
        output_rows.append({"method": row.method, "setting": row.setting, "protocol": row.protocol,
                            "direction": row.direction, "subject": int(row.subject), **target, **gate})
        selected = helper.target_window(core, row.protocol, int(row.subject), row.direction)
        window_rows.append(helper.make_rows(selected, row.method, weight=gate["weight"]))

        for deleted in sorted(train.meta_subject.astype(int).unique()):
            reduced = train[train.meta_subject.astype(int) != deleted].copy()
            fit = fit_gate(reduced, target)
            deletion_rows.append({"method": row.method, "setting": row.setting, "protocol": row.protocol,
                                  "direction": row.direction, "subject": int(row.subject),
                                  "deleted_meta_subject": int(deleted), "full_weight": gate["weight"],
                                  "deletion_weight": fit["weight"],
                                  "weight_delta": fit["weight"] - gate["weight"],
                                  "full_loss_events": gate["loss_events"],
                                  "deletion_loss_events": fit["loss_events"],
                                  "full_training_rows": len(train), "deletion_training_rows": len(reduced)})

    predictions = pd.concat([helper.recording_scores(pd.concat(window_rows, ignore_index=True)),
                             read_csv(PREVIOUS / "recording_predictions.csv").query(
                                 "method == 'dasf_balanced_gain'" )], ignore_index=True)
    return loso, cross, (pd.DataFrame(output_rows), predictions, pd.DataFrame(deletion_rows))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    job_files = verify_saved_jobs()
    helper = load_module(BASE / "heldout_gain_helpers.py", "heldout_helpers_order")
    core_path = NEUTRAL / "01_core" / "core_window_ensemble.csv"
    core = read_csv(core_path)
    loso_response = loso_response_rows(job_files)
    cross_response = cross_response_rows()
    loso, cross, (gate_diag, predictions, deletions) = fit_new_gates(
        loso_response, cross_response, core, helper
    )

    old_loso = read_csv(PREVIOUS / "training_rows_loso.csv")
    old_cross = read_csv(PREVIOUS / "training_rows_cross_task.csv")
    old_gate = read_csv(PREVIOUS / "gate_diagnostics.csv")
    old_pred = read_csv(PREVIOUS / "recording_predictions.csv")
    summary = {}
    for name, new, old, keys in [
        ("loso", loso_response, old_loso, ["task", "outer_subject", "meta_subject"]),
        ("cross_task", cross_response, old_cross, ["subject", "direction"]),
    ]:
        joined = new.merge(old[keys + ["benefit"]].rename(columns={"benefit": "old_benefit"}),
                           on=keys, validate="one_to_one")
        summary[name] = {
            "rows": int(len(joined)),
            "changed_response_rows": int((np.abs(joined.benefit - joined.old_benefit) > TOL).sum()),
            "max_absolute_response_change": float(np.abs(joined.benefit - joined.old_benefit).max()),
            "old_loss_events": int((joined.old_benefit <= -0.05).sum()),
            "new_loss_events": int((joined.benefit <= -0.05).sum()),
        }

    gate_join = gate_diag.merge(
        old_gate[old_gate.method.str.startswith("cbsf")][
            ["method", "protocol", "direction", "subject", "weight"]
        ].rename(columns={"weight": "old_weight"}),
        on=["method", "protocol", "direction", "subject"], validate="one_to_one",
    )
    old_cbsf = old_pred[old_pred.method.str.startswith("cbsf")]
    new_cbsf = predictions[predictions.method.str.startswith("cbsf")]
    pred_join = new_cbsf.merge(
        old_cbsf[["protocol", "subject", "direction", "recording_id", "method", "pred_label"]]
        .rename(columns={"pred_label": "old_pred_label"}),
        on=["protocol", "subject", "direction", "recording_id", "method"], validate="one_to_one",
    )
    summary["gate_and_predictions"] = {
        "gate_targets": int(len(gate_diag)),
        "changed_weights": int((np.abs(gate_join.weight - gate_join.old_weight) > TOL).sum()),
        "max_absolute_weight_change": float(np.abs(gate_join.weight - gate_join.old_weight).max()),
        "changed_recording_class_predictions": int((pred_join.pred_label != pred_join.old_pred_label).sum()),
        "deletion_rows": int(len(deletions)),
    }

    stats = load_module(HERE.parent / "first_dataset_balanced_gain_inputs" /
                        "compute_balanced_gain_stats.py", "ensemble_order_balanced_stats")
    participant = stats.participant_metrics(predictions)
    stats.GATE = OUT / "participant_metrics.csv"
    stats.CORE = core_path
    stats.OUT = OUT / "statistics"
    stats.OUT.mkdir(exist_ok=True)

    pd.concat([helper.make_rows(helper.target_window(core, row.protocol, int(row.subject), row.direction),
                                row.method, weight=row.weight)
               for row in gate_diag.itertuples(index=False)], ignore_index=True).to_csv(
                   OUT / "window_predictions.csv", index=False, lineterminator="\n")
    loso_response.to_csv(OUT / "response_rows_loso.csv", index=False, lineterminator="\n")
    cross_response.to_csv(OUT / "response_rows_cross_task.csv", index=False, lineterminator="\n")
    loso.to_csv(OUT / "training_rows_loso.csv", index=False, lineterminator="\n")
    cross.to_csv(OUT / "training_rows_cross_task.csv", index=False, lineterminator="\n")
    gate_diag.to_csv(OUT / "gate_diagnostics.csv", index=False, lineterminator="\n")
    predictions.to_csv(OUT / "recording_predictions.csv", index=False, lineterminator="\n")
    deletions.to_csv(OUT / "participant_deletion.csv", index=False, lineterminator="\n")
    participant.to_csv(OUT / "participant_metrics.csv", index=False, lineterminator="\n")
    stats.exact_sign_flip = exact_sign_flip
    stats.main()
    summary["status"] = "passed"
    summary["response_definition"] = (
        "median of five seed probabilities per window; geometric probability aggregation per recording; "
        "participant response is fused minus neural participant-macro recording balanced accuracy"
    )
    summary["independent_unit"] = "participant"
    summary["base_model_fits"] = 0
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_paths = [Path(__file__), HERE.parent / "first_dataset_correction/corrected_gate.py",
                    core_path, META / "02_all_meta_recording_seed.csv", HERE / "inputs/meta_index.json",
                    HERE / "inputs/seed_predictions.csv.gz", PREVIOUS / "training_rows_loso.csv",
                    PREVIOUS / "training_rows_cross_task.csv", PREVIOUS / "gate_diagnostics.csv",
                    PREVIOUS / "recording_predictions.csv"]
    manifest = {
        "analysis_script": Path(__file__).name,
        "input_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths},
        "verified_window_prediction_files": len(job_files),
        "output_sha256": {path.relative_to(OUT).as_posix(): sha256(path) for path in sorted(OUT.rglob("*"))
                          if path.is_file() and path.name != "manifest.json"},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT == (HERE / "results").resolve():
        parser.error("Use a separate output directory; published results are read-only references.")
    with threadpool_limits(limits=1):
        main()
