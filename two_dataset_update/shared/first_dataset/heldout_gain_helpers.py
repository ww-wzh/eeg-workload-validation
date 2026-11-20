"""Generate held-out-gain DASF/CB-SF predictions from completed 181 outputs.

This is a deterministic post-processing step. It does not retrain EEGNet,
EEG-Conformer, Riemannian models, or the four base branches from 181.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from revision_pipeline.risk_gate import (  # noqa: E402
    GATE_FEATURES,
    blend_probabilities,
    fit_predict_gate_robust,
    robust_feature_shift,
)

POLICY = HERE / "gate_protocol" / "nested_gate_protocol.json"
NEUTRAL = HERE / "neutral_predictions"
CORE = NEUTRAL / "01_core" / "core_window_ensemble.csv"
META_REC = HERE / "loso_gain_inputs" / "02_all_meta_recording_seed.csv"
META = HERE / "loso_gain_inputs" / "03_all_meta_seed.csv"
META_MEDIAN = HERE / "loso_gain_inputs" / "04_all_meta_median_seed.csv"
FEATURE_ROOT = NEUTRAL / "neutral_feature_cache" / "neutral"
OUT = HERE / "heldout_gain_reconstruction"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundles() -> dict[str, dict[str, np.ndarray]]:
    bundles: dict[str, dict[str, np.ndarray]] = {}
    for task, n_subjects in (("arithmetic", 15), ("stroop", 13)):
        features, covs, subjects, labels, records, windows = [], [], [], [], [], []
        for subject in range(1, n_subjects + 1):
            for rec in range(4):
                path = FEATURE_ROOT / f"{task}_s{subject:02d}_r{rec}.npz"
                if not path.exists():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as z:
                    f = np.asarray(z["features"], dtype=np.float64)
                    c = np.asarray(z["covariances"], dtype=np.float64)
                n = len(f)
                features.append(f); covs.append(c)
                subjects.extend([subject] * n)
                labels.extend([0 if rec < 2 else 1] * n)
                rid = f"{task}:s{subject:02d}:r{rec}"
                records.extend([rid] * n)
                windows.extend([f"{rid}:w{i:05d}" for i in range(n)])
        bundles[task] = {
            "features": np.concatenate(features), "covariances": np.concatenate(covs),
            "subjects": np.asarray(subjects), "labels": np.asarray(labels),
            "records": np.asarray(records, dtype=object), "windows": np.asarray(windows, dtype=object),
        }
    return bundles


def geometric(group: pd.DataFrame) -> tuple[np.ndarray, int]:
    p = group[["p0", "p1"]].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    out = np.exp(np.mean(np.log(p), axis=0)); out /= out.sum()
    return out, int(np.argmax(out))


def recording_scores(window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["protocol", "subject", "direction", "recording_id", "method"]
    for key, g in window.groupby(keys, sort=False):
        p, pred = geometric(g)
        rows.append(dict(zip(keys, key), p0=p[0], p1=p[1], pred_label=pred,
                            true_label=int(g.true_label.iloc[0]), correct=int(pred == int(g.true_label.iloc[0])),
                            n_windows=len(g)))
    return pd.DataFrame(rows)


def target_window(core: pd.DataFrame, protocol: str, subject: int, direction: str) -> pd.DataFrame:
    g = core[(core.protocol == protocol) & (core.subject.astype(int) == subject) & (core.direction == direction)]
    if g.empty:
        raise ValueError(f"No core target rows: {protocol}/{subject}/{direction}")
    return g


def diagnostics(g: pd.DataFrame, feature_shift: float) -> dict[str, float]:
    nn = g[g.method == "always_nn"].sort_values(["recording_id", "window_id"])
    fu = g[g.method == "always_fuse"].sort_values(["recording_id", "window_id"])
    keys = ["recording_id", "window_id"]
    m = nn[keys + ["p0", "p1"]].merge(fu[keys + ["p0", "p1"]], on=keys, suffixes=("_nn", "_fuse"), validate="one_to_one")
    pn = m[["p0_nn", "p1_nn"]].to_numpy(float); pf = m[["p0_fuse", "p1_fuse"]].to_numpy(float)
    def ent(p):
        q = np.clip(p, 1e-12, 1.0); return float(np.mean(-np.sum(q * np.log(q), axis=1)))
    return {"feature_shift": float(feature_shift),
            "disagreement": float(np.mean(np.abs(pn[:, 1] - pf[:, 1]))),
            "entropy_nn": ent(pn), "entropy_fuse": ent(pf),
            "confidence_nn": float(np.mean(np.max(pn, axis=1)))}


def feature_rows(bundle: dict[str, np.ndarray], subjects: set[int], records: set[int] | None = None):
    mask = np.isin(bundle["subjects"], sorted(subjects))
    if records is not None:
        recnum = np.array([int(str(x).rsplit(":r", 1)[1]) for x in bundle["records"]])
        mask &= np.isin(recnum, sorted(records))
    return bundle["features"][mask], bundle["covariances"][mask]


def cross_shift(bundles, subject: int, direction: str) -> float:
    source_task, target_task = ("arithmetic", "stroop") if direction == "arithmetic_to_stroop" else ("stroop", "arithmetic")
    xs, _ = feature_rows(bundles[source_task], {subject}, {1, 2, 3})
    xt, _ = feature_rows(bundles[target_task], {subject}, {1, 2, 3})
    return robust_feature_shift(xs, xt)


def loso_shift(bundles, task: str, subject: int) -> float:
    b = bundles[task]; n = int(b["subjects"].max())
    ref, _ = feature_rows(b, set(range(1, n + 1)) - {subject})
    tgt, _ = feature_rows(b, {subject})
    return robust_feature_shift(ref, tgt)


def loso_cov_shift(bundles, task: str, subject: int) -> float:
    b = bundles[task]; n = int(b["subjects"].max())
    _, refc = feature_rows(b, set(range(1, n + 1)) - {subject})
    _, tgtc = feature_rows(b, {subject})
    a, c = refc.mean(axis=0), tgtc.mean(axis=0)
    den = 0.5 * (np.linalg.norm(a, "fro") + np.linalg.norm(c, "fro"))
    return float(np.linalg.norm(a - c, "fro") / den) if den > 1e-12 else float("nan")


def make_rows(g: pd.DataFrame, method: str, weight: float | None = None, select_fuse: bool = False) -> pd.DataFrame:
    nn = g[g.method == "always_nn"].copy(); fu = g[g.method == "always_fuse"].copy()
    keys = ["protocol", "subject", "direction", "recording_id", "window_id", "true_label"]
    m = nn[keys + ["p0", "p1"]].merge(fu[keys + ["p0", "p1"]], on=keys, suffixes=("_nn", "_fuse"), validate="one_to_one")
    pn = m[["p0_nn", "p1_nn"]].to_numpy(float); pf = m[["p0_fuse", "p1_fuse"]].to_numpy(float)
    if select_fuse:
        out = pf
    elif weight is None:
        out = pn
    else:
        out = blend_probabilities(pn, pf, float(weight))
    result = m[["protocol", "subject", "direction", "recording_id", "window_id", "true_label"]].copy()
    result["method"] = method; result["p0"] = out[:, 0]; result["p1"] = out[:, 1]; result["ensemble_size"] = 5
    return result


def fit_cbsf(training: pd.DataFrame, target: dict[str, float], cfg: dict) -> tuple[float, dict]:
    x = pd.Series({k: target[k] for k in GATE_FEATURES})
    pred = fit_predict_gate_robust(training[GATE_FEATURES + ["benefit", "meta_subject"]], x,
        catastrophe_threshold=cfg["event_threshold"], benefit_alpha=cfg["benefit_alpha"],
        catastrophe_c=cfg["event_C"], risk_penalty=cfg["risk_penalty"],
        temperature=cfg["temperature"], max_weight=cfg["max_weight"])
    return pred.weight, {"weight": pred.weight, "predicted_benefit": pred.predicted_benefit,
                         "predicted_benefit_raw": pred.predicted_benefit_raw,
                         "predicted_catastrophe": pred.predicted_catastrophe,
                         "ood_fallback": pred.ood_fallback, "max_abs_robust_z": pred.max_abs_robust_z,
                         "training_rows": len(training), "training_subjects": int(training.meta_subject.nunique())}


def main() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert sha(META) == sha(META)  # explicit input existence/readability gate
    core = pd.read_csv(CORE, float_precision="round_trip")
    meta_rec = pd.read_csv(META_REC, float_precision="round_trip")
    meta = pd.read_csv(META, float_precision="round_trip")
    median = pd.read_csv(META_MEDIAN, float_precision="round_trip")
    bundles = load_bundles()
    assert len(meta) == 1830 and len(median) == 366
    assert set(core.method.astype(str)) == {"always_nn", "always_fuse", "rf", "extra_trees"}
    cfg = policy["gate_rules"]["cbsf"]
    generated, gate_rows = [], []

    # New LOSO CB-SF and DASF, 112 recording rows each.
    for task, n in (("arithmetic", 15), ("stroop", 13)):
        direction = task
        train_all = median[median.task == task].copy()
        for subject in range(1, n + 1):
            target_core = target_window(core, "loso", subject, direction)
            target = diagnostics(target_core, loso_shift(bundles, task, subject))
            tr = train_all[train_all.outer_subject.astype(int) == subject].copy()
            tr["meta_subject"] = tr.meta_subject.astype(int)
            w, gd = fit_cbsf(tr, target, cfg)
            out = make_rows(target_core, "cbsf_double_exclusion_gain", weight=w)
            generated.append(out); gate_rows.append({"method": "cbsf_double_exclusion_gain", "setting": f"{task}_loso", "subject": subject, **target, **gd})

            # DASF gain: median seed fusion accuracy minus median seed neural accuracy.
            r = meta_rec[(meta_rec.dataset == "openbci") & (meta_rec.outer_subject.astype(int) == subject)
                         & (meta_rec.direction == direction) & (meta_rec.protocol == "loso_meta_pair")]
            seed_acc = r.groupby(["seed", "method"], as_index=False).correct.mean()
            nn_med = float(seed_acc[seed_acc.method == "always_nn"].correct.median())
            fu_med = float(seed_acc[seed_acc.method == "always_fuse"].correct.median())
            gain = fu_med - nn_med
            margin = float(np.clip(0.02 + 0.2 * loso_cov_shift(bundles, task, subject), 0.0, 0.15))
            choose = bool(np.isfinite(gain) and gain > margin)
            generated.append(make_rows(target_core, "dasf_double_exclusion_gain", select_fuse=choose))
            gate_rows.append({"method": "dasf_double_exclusion_gain", "setting": f"{task}_loso", "subject": subject,
                              "gain": gain, "delta_cov": loso_cov_shift(bundles, task, subject), "margin": margin, "selected_fusion": choose})

    # Cross-task held-out target-task CB-SF, 78 recording rows.
    cross_rows = []
    for subject in range(1, 14):
        for direction in ("arithmetic_to_stroop", "stroop_to_arithmetic"):
            g = target_window(core, "cross_task", subject, direction)
            source_task, target_task = ("arithmetic", "stroop") if direction == "arithmetic_to_stroop" else ("stroop", "arithmetic")
            cross_rows.append({"subject": subject, "direction": direction, **diagnostics(g, cross_shift(bundles, subject, direction))})
    cross_target = pd.DataFrame(cross_rows)
    training_rows = []
    for row in cross_rows:
        g = target_window(core, "cross_task", row["subject"], row["direction"])
        rec = recording_scores(g)
        nn = rec[rec.method == "always_nn"].set_index("recording_id")
        fu = rec[rec.method == "always_fuse"].set_index("recording_id")
        benefit = float(fu.correct.mean() - nn.correct.mean())
        training_rows.append({**{k: row[k] for k in ["direction", "subject"]}, **{k: row[k] for k in GATE_FEATURES}, "benefit": benefit, "meta_subject": row["subject"]})
    cross_training = pd.DataFrame(training_rows)
    for row in cross_rows:
        target = {k: row[k] for k in GATE_FEATURES}
        tr = cross_training[cross_training.subject.astype(int) != int(row["subject"])].copy()
        w, gd = fit_cbsf(tr, target, cfg)
        generated.append(make_rows(target_window(core, "cross_task", row["subject"], row["direction"]), "cbsf_target_task_gain", weight=w))
        gate_rows.append({"method": "cbsf_target_task_gain", "setting": "cross_task", "subject": row["subject"], "direction": row["direction"], **target, **gd})

    windows = pd.concat(generated, ignore_index=True)
    if windows.duplicated(["protocol", "subject", "direction", "recording_id", "window_id", "method"]).any():
        raise AssertionError("Duplicate held-out gate window predictions")
    recs = recording_scores(windows)
    if len(recs) != 302:
        raise AssertionError(f"Expected 302 recording predictions, got {len(recs)}")
    OUT.mkdir(parents=True, exist_ok=True)
    windows.to_csv(OUT / "heldout_gate_window_predictions.csv", index=False, lineterminator="\n")
    recs.to_csv(OUT / "heldout_gate_recording_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(gate_rows).to_csv(OUT / "heldout_gate_diagnostics.csv", index=False, lineterminator="\n")
    cross_training.to_csv(OUT / "cross_task_cbsf_training_rows.csv", index=False, lineterminator="\n")
    manifest = {"status": "passed", "protocol_version": policy["schema"], "full_gate_fitted": True,
                "recording_rows": len(recs), "window_rows": len(windows), "gate_cells": len(gate_rows),
                "methods": sorted(pd.DataFrame(gate_rows).method.unique()),
                "expected": {"recording_rows": 302, "loso_cbsf": 112, "loso_dasf": 112, "cross_task_cbsf": 78},
                "inputs": {"core_sha256": sha(CORE), "meta_recording_sha256": sha(META_REC),
                           "meta_seed_sha256": sha(META), "meta_median_sha256": sha(META_MEDIAN)},
                "outputs_sha256": {p.name: sha(p) for p in OUT.glob("*.csv")}}
    (OUT / "185_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("185 held-out-gain post-processing completed")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
