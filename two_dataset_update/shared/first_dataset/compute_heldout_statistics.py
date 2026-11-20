"""186: participant-level statistics for the held-out-gain gate audit.

Reads frozen neutral-preprocessing predictions only.  It does not retrain any
base learner or overwrite previous results.  The inferential unit is the
participant; recording balanced accuracy is calculated first, and cross-task
bidirectional values are the mean of the two direction-specific participant
values.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
BASE = HERE
NEUTRAL = BASE / "neutral_predictions"
NEW = BASE / "heldout_gain_reconstruction"
OUT = BASE / "heldout_gain_statistics"
CORE = NEUTRAL / "01_core" / "core_recording_predictions.csv"
OLD_DASF = NEUTRAL / "03_dasf" / "recording_predictions.csv"
OLD_CBSF = NEUTRAL / "04_cbsf" / "recording_predictions.csv"
NEW_PRED = NEW / "heldout_gate_recording_predictions.csv"
BOOTSTRAP_REPEATS = 10_000
BASE_SEED = 20260910
EPS = 1e-15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed(*parts: object) -> int:
    s = "|".join(map(str, (BASE_SEED, *parts)))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") % (2**32)


def read_prediction(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {"protocol", "subject", "direction", "recording_id", "method", "true_label", "pred_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    d["subject"] = d["subject"].astype(int)
    d["true_label"] = d["true_label"].astype(int)
    d["pred_label"] = d["pred_label"].astype(int)
    return d


def recording_ba(d: pd.DataFrame) -> float:
    if d.empty:
        raise ValueError("empty recording set")
    truth, pred = [], []
    for rid, g in d.groupby("recording_id", sort=True):
        if g["true_label"].nunique() != 1:
            raise ValueError(f"recording {rid} has multiple labels")
        truth.append(int(g["true_label"].iloc[0]))
        pred.append(int(g["pred_label"].iloc[0]))
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    out = []
    for cls in (0, 1):
        mask = truth == cls
        if not mask.any():
            return float("nan")
        out.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(out))


def participant_values(all_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, direction, subject, method), g in all_predictions.groupby(
        ["protocol", "direction", "subject", "method"], sort=True
    ):
        expected = 3 if protocol == "cross_task" else 4
        if len(g) != expected:
            raise AssertionError(f"{protocol}/{direction}/S{subject:02d}/{method}: expected {expected}, got {len(g)}")
        rows.append({
            "protocol": protocol, "direction": direction, "subject": int(subject),
            "method": method, "recording_balanced_accuracy": recording_ba(g),
        })
    return pd.DataFrame(rows)


def exact_sign_flip(x: np.ndarray) -> tuple[float, int, int]:
    x = np.asarray(x, dtype=float)
    nz = x[np.abs(x) > EPS]
    if len(nz) == 0:
        return 1.0, int(len(x)), 0
    null = np.array([np.sum(np.asarray(s) * nz) for s in itertools.product((-1.0, 1.0), repeat=len(nz))])
    observed = float(np.sum(nz))
    p = float(np.mean(np.abs(null) >= abs(observed) - EPS))
    return p, int(np.sum(np.abs(x) <= EPS)), int(len(nz))


def pratt(x: np.ndarray) -> tuple[float, float]:
    if np.all(np.abs(x) <= EPS):
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wilcoxon(x, zero_method="pratt", correction=False, alternative="two-sided", method="approx")
    return float(r.statistic), float(r.pvalue)


def rank_biserial(x: np.ndarray) -> float:
    x = np.asarray(x)
    ranks = rankdata(np.abs(x), method="average")
    pos, neg = ranks[x > EPS].sum(), ranks[x < -EPS].sum()
    return float((pos - neg) / (pos + neg)) if pos + neg else 0.0


def bootstrap_ci(x: np.ndarray, key: str) -> tuple[float, float]:
    rng = np.random.default_rng(seed("bootstrap", key))
    draws = x[rng.integers(0, len(x), size=(BOOTSTRAP_REPEATS, len(x)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975], method="linear"))


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.clip(ranked, 0, 1)
    return out


def main() -> None:
    paths = [CORE, OLD_DASF, OLD_CBSF, NEW_PRED]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + "; ".join(missing))
    core = read_prediction(CORE)
    dasf = read_prediction(OLD_DASF)
    cbsf = read_prediction(OLD_CBSF)
    new = read_prediction(NEW_PRED)
    # Keep only one copy of shared branches; method names are explicitly mapped.
    core = core[core.method == "always_nn"].copy()
    dasf = dasf[dasf.method == "dasf_clean"].copy()
    cbsf = cbsf[cbsf.method == "cbsf"].copy()
    new_dasf = new[new.method == "dasf_double_exclusion_gain"].copy()
    new_cbsf = new[new.method == "cbsf_double_exclusion_gain"].copy()
    new_target = new[new.method == "cbsf_target_task_gain"].copy()
    frames = []
    for d, method in [(core, "feature_mlp"), (dasf, "dasf_original"), (cbsf, "cbsf_original"),
                      (new_dasf, "dasf_heldout_gain"), (new_cbsf, "cbsf_heldout_gain"),
                      (new_target, "cbsf_target_task_gain")]:
        d = d.copy(); d["method"] = method; frames.append(d)
    pred = pd.concat(frames, ignore_index=True)
    key = ["protocol", "direction", "subject", "recording_id"]
    if pred.duplicated(key + ["method"]).any():
        raise AssertionError("duplicate prediction key")
    vals = participant_values(pred)
    OUT.mkdir(parents=True, exist_ok=True)
    vals.to_csv(OUT / "participant_direction_metrics.csv", index=False)

    # Build the five analysis settings.
    settings = [
        ("loso_arithmetic", "loso", "arithmetic"), ("loso_stroop", "loso", "stroop"),
        ("cross_task_arithmetic_to_stroop", "cross_task", "arithmetic_to_stroop"),
        ("cross_task_stroop_to_arithmetic", "cross_task", "stroop_to_arithmetic"),
    ]
    setting_rows = []
    for name, protocol, direction in settings:
        q = vals[(vals.protocol == protocol) & (vals.direction == direction)].copy()
        q["setting"] = name; setting_rows.append(q)
    # Pooled bidirectional cross-task: average direction-specific BA per participant/method.
    q = vals[vals.protocol == "cross_task"].groupby(["subject", "method"], as_index=False)["recording_balanced_accuracy"].mean()
    q["setting"] = "cross_task_bidirectional"; q["protocol"] = "cross_task"; q["direction"] = "bidirectional"; setting_rows.append(q)
    values = pd.concat(setting_rows, ignore_index=True)
    values.to_csv(OUT / "participant_setting_metrics.csv", index=False)

    contrasts = []
    def add(setting, a, b, label):
        x = values[(values.setting == setting) & (values.method == a)].set_index("subject")["recording_balanced_accuracy"]
        y = values[(values.setting == setting) & (values.method == b)].set_index("subject")["recording_balanced_accuracy"]
        z = (x - y).dropna().sort_index().to_numpy()
        if len(z) < 2: raise AssertionError(f"too few paired participants for {label}")
        low, high = bootstrap_ci(z, label); sign_p, nzero, nnz = exact_sign_flip(z); wstat, wp = pratt(z)
        contrasts.append({"contrast": label, "setting": setting, "method": a, "reference_method": b,
            "n_participants": len(z), "method_mean": float(x.mean()), "reference_mean": float(y.mean()),
            "mean_difference": float(z.mean()), "mean_difference_pp": float(100*z.mean()),
            "median_difference": float(np.median(z)), "sd_difference": float(np.std(z, ddof=1)),
            "ci95_low": low, "ci95_high": high, "ci95_low_pp": 100*low, "ci95_high_pp": 100*high,
            "better": int(np.sum(z > EPS)), "tied": int(np.sum(np.abs(z) <= EPS)), "worse": int(np.sum(z < -EPS)),
            "rank_biserial": rank_biserial(z), "exact_sign_flip_p": sign_p,
            "wilcoxon_pratt_statistic": wstat, "wilcoxon_pratt_raw_p": wp,
            "n_zero_differences": nzero, "n_nonzero_differences": nnz})
    for s in ("loso_arithmetic", "loso_stroop"):
        add(s, "dasf_heldout_gain", "feature_mlp", f"{s}:heldout DASF vs Feature-MLP")
        add(s, "dasf_heldout_gain", "dasf_original", f"{s}:heldout DASF vs original DASF")
        add(s, "cbsf_heldout_gain", "feature_mlp", f"{s}:heldout CB-SF vs Feature-MLP")
        add(s, "cbsf_heldout_gain", "cbsf_original", f"{s}:heldout CB-SF vs original CB-SF")
    for s in ("cross_task_arithmetic_to_stroop", "cross_task_stroop_to_arithmetic", "cross_task_bidirectional"):
        add(s, "cbsf_target_task_gain", "feature_mlp", f"{s}:target-task-gain CB-SF vs Feature-MLP")
        add(s, "cbsf_target_task_gain", "cbsf_original", f"{s}:target-task-gain CB-SF vs original CB-SF")
    comp = pd.DataFrame(contrasts)
    # The frozen protocol defines exact sign-flip as the primary test.  The
    # Pratt Wilcoxon p-values are retained as a separately labelled sensitivity
    # family and must not replace the primary p-values.
    comp["bh_adjusted_p_exact_sign_flip_14_family"] = bh(comp["exact_sign_flip_p"].to_numpy())
    comp["bh_adjusted_p_wilcoxon_14_family_sensitivity"] = bh(comp["wilcoxon_pratt_raw_p"].to_numpy())
    comp["bh_adjusted_p_14_family"] = comp["bh_adjusted_p_exact_sign_flip_14_family"]
    comp["bh_significant_0_05"] = comp["bh_adjusted_p_exact_sign_flip_14_family"] < 0.05
    comp.to_csv(OUT / "comparisons_BH14.csv", index=False)
    manifest = {"status": "passed", "analysis": "held-out-gain participant-level contrasts", "n_contrasts": int(len(comp)),
        "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "unit": "whole participant", "interval": "percentile 95%"},
        "primary_inference": "exact two-sided participant sign-flip; Wilcoxon Pratt approximate sensitivity",
        "multiple_comparisons": {
            "primary": "all 14 exact-sign-flip raw p-values in one BH family",
            "sensitivity": "all 14 Wilcoxon-Pratt raw p-values in one separately labelled BH family",
        }, "inputs": {str(p): sha256(p) for p in paths},
        "methods": {"feature_mlp": "core always_nn", "dasf_original": "03_dasf dasf_clean", "cbsf_original": "04_cbsf cbsf",
                    "dasf_heldout_gain": "185 double-exclusion DASF", "cbsf_heldout_gain": "185 double-exclusion CB-SF",
                    "cbsf_target_task_gain": "185 target-task-gain CB-SF"}}
    # Re-write the finalized frames after all validation has completed.
    vals.to_csv(OUT / "participant_direction_metrics.csv", index=False)
    values.to_csv(OUT / "participant_setting_metrics.csv", index=False)
    comp.to_csv(OUT / "comparisons_BH14.csv", index=False)
    (OUT / "186_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = comp[["contrast", "n_participants", "mean_difference_pp", "ci95_low_pp", "ci95_high_pp", "exact_sign_flip_p", "bh_adjusted_p_14_family", "bh_significant_0_05"]]
    summary.to_string(OUT / "statistics_summary.txt", index=False)
    print("186 participant-level held-out-gain statistics completed")
    print(summary.to_string(index=False))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
