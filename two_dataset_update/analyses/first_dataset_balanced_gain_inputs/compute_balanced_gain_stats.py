"""Participant-level inference for the balanced-accuracy gain sensitivity.

This file derives a separately labelled seven-comparison sensitivity family from
the saved predictions.  It does not retrain any model.  The cross-task
bidirectional row is formed by averaging the two direction-specific differences
within participant before inference, so the inferential unit remains participant.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "shared" / "first_dataset"
NEUTRAL = BASE / "neutral_predictions"
OUT = Path(__file__).resolve().parent
GATE = OUT / "balanced_gain_participant_metrics.csv"
CORE = NEUTRAL / "01_core" / "core_window_ensemble.csv"
N_BOOT = 10_000
TOL = 1e-12


def geometric_recording_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["protocol", "subject", "direction", "recording_id", "method"]
    for key, group in frame.groupby(keys, sort=False):
        p = np.clip(group[["p0", "p1"]].to_numpy(float), 1e-12, 1.0)
        prob = np.exp(np.mean(np.log(p), axis=0))
        pred = int(np.argmax(prob))
        true = int(group.true_label.iloc[0])
        rows.append(dict(zip(keys, key), true_label=true, correct=int(pred == true), n_windows=len(group)))
    return pd.DataFrame(rows)


def balanced_accuracy(frame: pd.DataFrame) -> float:
    values = []
    for label in (0, 1):
        part = frame[frame.true_label.astype(int) == label]
        if part.empty:
            return float("nan")
        values.append(float(part.correct.astype(float).mean()))
    return float(np.mean(values))


def participant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["protocol", "direction", "subject", "method"], sort=True):
        rows.append({
            "protocol": keys[0], "direction": keys[1], "subject": int(keys[2]),
            "method": keys[3], "balanced_accuracy": balanced_accuracy(group),
        })
    return pd.DataFrame(rows)


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    nz = values[np.abs(values) > TOL]
    if len(nz) == 0:
        return 1.0
    observed = abs(float(values.sum()))
    counts = 0
    total = 1 << len(nz)
    for mask in range(total):
        signs = np.where(((mask >> np.arange(len(nz))) & 1) == 1, 1.0, -1.0)
        if abs(float(np.abs(np.dot(signs, nz))) - observed) <= TOL:
            counts += 1
    return float(counts / total)


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(("Sensors_balanced_gain_stats_v1|" + label).encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def bootstrap_ci(values: np.ndarray, label: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.Generator(np.random.PCG64(stable_seed(label)))
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    q = np.empty(len(p), dtype=float)
    running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * len(p) / (rank + 1))
        q[i] = running
    return q.tolist()


def contrast_row(setting: str, method: str, label: str, gate: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    g = gate[(gate.protocol + "_" + gate.direction == setting) & (gate.method == method)].copy()
    b = baseline[baseline.protocol + "_" + baseline.direction == setting].copy()
    merged = g[["subject", "balanced_accuracy"]].merge(
        b[["subject", "balanced_accuracy"]], on="subject", suffixes=("_gate", "_baseline"), validate="one_to_one"
    )
    differences = (merged.balanced_accuracy_gate - merged.balanced_accuracy_baseline).to_numpy(float)
    ci_low, ci_high = bootstrap_ci(differences, setting + "|" + method)
    return {
        "contrast": label,
        "family": "balanced_gain_sensitivity_7",
        "setting": setting,
        "method": method,
        "reference_method": "always_nn_feature_mlp",
        "n_participants": int(len(differences)),
        "method_mean": float(merged.balanced_accuracy_gate.mean()),
        "reference_mean": float(merged.balanced_accuracy_baseline.mean()),
        "mean_difference": float(differences.mean()),
        "mean_difference_pp": float(100 * differences.mean()),
        "median_difference": float(np.median(differences)),
        "sd_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "ci95_low_pp": 100 * ci_low,
        "ci95_high_pp": 100 * ci_high,
        "better": int(np.sum(differences > TOL)),
        "tied": int(np.sum(np.abs(differences) <= TOL)),
        "worse": int(np.sum(differences < -TOL)),
        "exact_sign_flip_p": exact_sign_flip(differences),
        "n_zero_differences": int(np.sum(np.abs(differences) <= TOL)),
        "n_nonzero_differences": int(np.sum(np.abs(differences) > TOL)),
    }


def main() -> None:
    core = pd.read_csv(CORE, float_precision="round_trip")
    baseline = participant_metrics(geometric_recording_scores(core[core.method == "always_nn"]))
    gate = pd.read_csv(GATE, float_precision="round_trip")

    rows = []
    specs = [
        ("loso_arithmetic", "cbsf_balanced_gain", "Arithmetic LOSO: CB-SF vs Feature-MLP"),
        ("loso_arithmetic", "dasf_balanced_gain", "Arithmetic LOSO: DASF vs Feature-MLP"),
        ("loso_stroop", "cbsf_balanced_gain", "Stroop LOSO: CB-SF vs Feature-MLP"),
        ("loso_stroop", "dasf_balanced_gain", "Stroop LOSO: DASF vs Feature-MLP"),
        ("cross_task_arithmetic_to_stroop", "cbsf_balanced_target_task_gain", "Arithmetic-to-Stroop: CB-SF vs Feature-MLP"),
        ("cross_task_stroop_to_arithmetic", "cbsf_balanced_target_task_gain", "Stroop-to-Arithmetic: CB-SF vs Feature-MLP"),
    ]
    for setting, method, label in specs:
        rows.append(contrast_row(setting, method, label, gate, baseline))

    # Bidirectional cross-task contrast: average the two direction-specific effects
    # per participant, then infer over the 13 participants.
    d1 = gate[(gate.protocol == "cross_task") & (gate.direction == "arithmetic_to_stroop") & (gate.method == "cbsf_balanced_target_task_gain")][["subject", "balanced_accuracy"]]
    d2 = gate[(gate.protocol == "cross_task") & (gate.direction == "stroop_to_arithmetic") & (gate.method == "cbsf_balanced_target_task_gain")][["subject", "balanced_accuracy"]]
    b1 = baseline[(baseline.protocol == "cross_task") & (baseline.direction == "arithmetic_to_stroop")][["subject", "balanced_accuracy"]]
    b2 = baseline[(baseline.protocol == "cross_task") & (baseline.direction == "stroop_to_arithmetic")][["subject", "balanced_accuracy"]]
    x1 = d1.merge(b1, on="subject", suffixes=("_gate", "_base")); x1["d1"] = x1.balanced_accuracy_gate - x1.balanced_accuracy_base
    x2 = d2.merge(b2, on="subject", suffixes=("_gate", "_base")); x2["d2"] = x2.balanced_accuracy_gate - x2.balanced_accuracy_base
    differences = x1[["subject", "d1"]].merge(x2[["subject", "d2"]], on="subject").assign(difference=lambda x: (x.d1 + x.d2) / 2).difference.to_numpy(float)
    ci_low, ci_high = bootstrap_ci(differences, "cross_task_bidirectional|cbsf_balanced_target_task_gain")
    rows.append({
        "contrast": "Bidirectional cross-task: CB-SF vs Feature-MLP",
        "family": "balanced_gain_sensitivity_7", "setting": "cross_task_bidirectional",
        "method": "cbsf_balanced_target_task_gain", "reference_method": "always_nn_feature_mlp",
        "n_participants": int(len(differences)),
        "method_mean": float((d1.balanced_accuracy.mean() + d2.balanced_accuracy.mean()) / 2),
        "reference_mean": float((b1.balanced_accuracy.mean() + b2.balanced_accuracy.mean()) / 2),
        "mean_difference": float(differences.mean()), "mean_difference_pp": float(100 * differences.mean()),
        "median_difference": float(np.median(differences)),
        "sd_difference": float(np.std(differences, ddof=1)),
        "ci95_low": ci_low, "ci95_high": ci_high, "ci95_low_pp": 100 * ci_low, "ci95_high_pp": 100 * ci_high,
        "better": int(np.sum(differences > TOL)), "tied": int(np.sum(np.abs(differences) <= TOL)),
        "worse": int(np.sum(differences < -TOL)), "exact_sign_flip_p": exact_sign_flip(differences),
        "n_zero_differences": int(np.sum(np.abs(differences) <= TOL)),
        "n_nonzero_differences": int(np.sum(np.abs(differences) > TOL)),
    })
    q = bh_adjust([r["exact_sign_flip_p"] for r in rows])
    for r, value in zip(rows, q):
        r["bh_q_7_family"] = value
        r["bootstrap_resamples"] = N_BOOT
        r["bootstrap_seed"] = stable_seed(r["setting"] + "|" + r["method"])
        r["interval_basis"] = "saved fitted predictions; whole-participant percentile bootstrap"
        r["sign_flip_basis"] = "two-sided exact participant sign-flip test"
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "balanced_gain_method_comparisons_BH7.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "passed", "analysis": "exploratory balanced-accuracy gain response sensitivity",
        "does_not_retrain_base_models": True, "comparisons": 7, "bootstrap_resamples": N_BOOT,
        "family": "BH7; separate from the ordinary-accuracy held-out-gain BH14 family",
        "core_sha256": hashlib.sha256(CORE.read_bytes()).hexdigest(),
        "gate_metrics_sha256": hashlib.sha256(GATE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256((OUT / "balanced_gain_method_comparisons_BH7.csv").read_bytes()).hexdigest(),
        "notes": [
            "The bidirectional row averages the two direction-specific differences within each participant before inference.",
            "Intervals are conditional on saved fits and do not include complete retraining uncertainty.",
            "Exact sign-flip inference depends on the assumed exchangeability/symmetry of paired participant differences.",
        ],
    }
    (OUT / "balanced_gain_stats_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out[["contrast", "n_participants", "mean_difference_pp", "ci95_low_pp", "ci95_high_pp", "better", "tied", "worse", "exact_sign_flip_p", "bh_q_7_family"]].to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
