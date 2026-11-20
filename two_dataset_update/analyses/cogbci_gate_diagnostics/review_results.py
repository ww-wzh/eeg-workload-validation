"""Read-only review of frozen, verified COG-BCI results; no new model fits/tests."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "cogbci_gate_evaluation"
DATA = BASE / "analysis"
METHODS = ["feature_mlp", "rf", "extra_trees", "always_fuse", "fixed_half",
           "dasf_proxy", "dasf_heldout", "cbsf_proxy", "cbsf_heldout",
           "cbsf_response_bridge", "cbsf_heldout_no_risk_penalty", "cbsf_heldout_no_ood_guard"]
TOL = 1e-12


def read(name):
    return pd.read_csv(DATA / f"{name}.csv")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def as_bool(series):
    return series.astype(str).str.lower().eq("true")


def counts(values):
    a = np.asarray(values, dtype=float)
    return {"positive": int((a > TOL).sum()), "zero": int((np.abs(a) <= TOL).sum()),
            "negative": int((a < -TOL).sum())}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main():
    manifest = json.loads((DATA / "verified_outputs_manifest.json").read_text(encoding="utf-8"))
    assert all(digest(BASE / filename) == expected for filename, expected in manifest.items())
    verification = json.loads((DATA / "independent_verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    markers = list((BASE / "jobs").glob("*/complete.json"))
    assert len(markers) == 3250
    summary = read("method_summary")
    contrast = read("comparisons_BH12")
    score = read("participant_scores")
    ens = score[score["mode"].eq("ensemble")]
    records = read("record_scores")
    gates = read("gate_diagnostics")
    main_gates = gates[gates["mode"].eq("ensemble")]
    meta = read("meta_rows")
    pairs = read("paired_response_changes")
    branch = read("branch_complementarity")
    deletion = read("meta_participant_deletion")
    prediction = read("outer_gain_prediction_errors")
    for row in summary.itertuples():
        values = ens[(ens.task == row.task) & (ens.method == row.method)]
        assert values.subject.nunique() == 25
        assert abs(values.balanced_accuracy.mean() - row.mean) < TOL

    review = {
        "review_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive diagnostics of frozen results, with no model retraining, new hypothesis tests or threshold selection.",
        "integrity": {"completed_jobs": len(markers), "output_hashes_matched": len(manifest), "independent_verification": verification},
        "method_summary": summary.to_dict("records"),
        "primary_contrasts": contrast.to_dict("records"),
        "significant_BH12": int((contrast.p_BH12 < .05).sum()),
        "tasks": {},
    }
    for task in ["NBACK", "MATB"]:
        ss = ens[ens.task.eq(task)]
        wide = ss.pivot(index="subject", columns="method", values="balanced_accuracy")
        rr = records[records.task.eq(task)]
        gg = main_gates[main_gates.task.eq(task)]
        pp = pairs[pairs.task.eq(task)]
        bb = branch[branch.task.eq(task)]
        out = {"participants": 25, "records": int(rr[rr.method.eq("feature_mlp")].shape[0])}
        out["method_participant_comparisons"] = []
        for left, right in [("cbsf_heldout", "feature_mlp"), ("cbsf_heldout", "cbsf_proxy"),
                            ("cbsf_heldout", "always_fuse"), ("cbsf_heldout", "fixed_half"),
                            ("cbsf_heldout", "extra_trees"), ("always_fuse", "feature_mlp"),
                            ("cbsf_response_bridge", "cbsf_proxy"),
                            ("cbsf_heldout", "cbsf_response_bridge"),
                            ("cbsf_heldout_no_risk_penalty", "cbsf_heldout"),
                            ("cbsf_heldout_no_ood_guard", "cbsf_heldout")]:
            delta = wide[left] - wide[right]
            joined = rr[rr.method.eq(left)].merge(rr[rr.method.eq(right)], on=["subject", "record_id", "label"], suffixes=("_l", "_r"), validate="one_to_one")
            changed = joined.prediction_l != joined.prediction_r
            out["method_participant_comparisons"].append({
                "contrast": left + " - " + right, "mean_difference_pp": float(delta.mean()*100),
                **counts(delta),
                "positive_subjects": [int(v) for v in delta.index[delta > TOL]],
                "negative_subjects": [int(v) for v in delta.index[delta < -TOL]],
                "changed_record_decisions": int(changed.sum()),
                "repaired_records": int((changed & (joined.prediction_l == joined.label)).sum()),
                "harmed_records": int((changed & (joined.prediction_r == joined.label)).sum()),
            })
        out["gate_summary"] = []
        for method, frame in gg.groupby("method", sort=True):
            entry = {"method": method, "weight_min": frame.weight.min(), "weight_median": frame.weight.median(),
                     "weight_max": frame.weight.max(), "weight_mean": frame.weight.mean(),
                     "zero_weight_count": int((frame.weight.abs() <= TOL).sum())}
            if method.startswith("cbsf"):
                entry.update({"loss_events_min": frame.loss_events.min(), "loss_events_max": frame.loss_events.max(),
                              "loss_events_median": frame.loss_events.median(),
                              "constant_event_fallbacks": int(as_bool(frame.constant_event_fallback).sum()),
                              "ood_fallbacks": int(as_bool(frame.ood_fallback).sum())})
            else:
                entry.update({"selected_fusion_count": int((frame.weight > .5).sum()),
                              "pooled_gain_min": frame.pooled_gain.min(), "pooled_gain_max": frame.pooled_gain.max(),
                              "margin_min": frame.margin.min(), "margin_max": frame.margin.max()})
            out["gate_summary"].append(entry)
        weights = gg.pivot(index="subject", columns="method", values="weight")
        out["changed_proxy_to_heldout_weights"] = int(((weights.cbsf_proxy - weights.cbsf_heldout).abs() > TOL).sum())
        out["meta_response_summary"] = []
        for kind, frame in meta[meta.task.eq(task)].groupby("kind", sort=True):
            out["meta_response_summary"].append({"kind": kind, "rows_correlated": len(frame),
                "mean_benefit_pp": frame.benefit.mean()*100, "median_benefit_pp": frame.benefit.median()*100,
                **counts(frame.benefit), "loss_events": int((frame.benefit <= -.05).sum())})
        out["paired_response_changes"] = {
            "paired_rows_correlated": len(pp), "changed_gain_rows": int((pp.proxy_minus_heldout.abs() > TOL).sum()),
            "mean_proxy_minus_heldout_pp": pp.proxy_minus_heldout.mean()*100,
            "event_changed_rows": int(as_bool(pp.event_changed).sum()),
        }
        out["branch_complementarity"] = {key: int(bb[key].sum()) for key in ["both_correct", "neural_only", "fused_only", "both_wrong"]}
        out["branch_complementarity"].update({"participant_gain_counts": counts(bb.gain),
            "hard_branch_oracle_mean_ba": bb.participant_hard_oracle_ba.mean(),
            "mean_neural_ba": bb.neural_ba.mean(), "mean_fused_ba": bb.fused_ba.mean()})
        out["seed_sensitivity"] = []
        out["seed_participant_changes"] = []
        for mode in ["complete_seed", "gate_only_seed"]:
            rows = score[(score.task == task) & score["mode"].eq(mode)]
            by_seed = rows.groupby(["seed", "method"]).balanced_accuracy.mean().unstack("method")
            for seed, frame in rows.groupby("seed"):
                sw = frame.pivot(index="subject", columns="method", values="balanced_accuracy")
                for method in ["cbsf_heldout", "cbsf_proxy", "always_fuse"]:
                    out["seed_participant_changes"].append({"mode": mode, "seed": int(seed), "method": method,
                        **counts(sw[method]-sw.feature_mlp)})
            for method in ["feature_mlp", "extra_trees", "always_fuse", "fixed_half", "dasf_heldout", "cbsf_proxy", "cbsf_heldout"]:
                out["seed_sensitivity"].append({"mode": mode, "method": method,
                    "means_by_seed": {int(seed): float(value) for seed, value in by_seed[method].items()},
                    "difference_from_mlp_pp_by_seed": {int(seed): float(value*100) for seed, value in (by_seed[method]-by_seed.feature_mlp).items()}})
        out["meta_deletion"] = []
        for method, frame in deletion[deletion.task.eq(task)].groupby("method", sort=True):
            f = frame.merge(wide[method].rename("main_ba"), left_on="subject", right_index=True, validate="many_to_one")
            f = f.merge(weights[method].rename("main_weight"), left_on="subject", right_index=True, validate="many_to_one")
            changed = (f.balanced_accuracy-f.main_ba).abs() > TOL
            ranges = f.groupby("subject").weight.agg(["min", "max"])
            delta = f.groupby("subject").balanced_accuracy.agg(["min", "max"])
            # Target-specific extrema are a sensitivity envelope, not a simultaneous cohort result.
            out["meta_deletion"].append({"method": method, "refits": len(frame),
                "changed_score_refits": int(changed.sum()), "targets_with_changed_score": int(f.loc[changed, "subject"].nunique()),
                "refit_score_change_pp_min": float(((f.balanced_accuracy-f.main_ba)*100).min()),
                "refit_score_change_pp_max": float(((f.balanced_accuracy-f.main_ba)*100).max()),
                "maximum_within_target_weight_range": float((ranges["max"]-ranges["min"]).max()),
                "mean_of_target_min_ba": float(delta["min"].mean()),
                "mean_of_target_max_ba": float(delta["max"].mean()),
                "ood_fallbacks": int(as_bool(frame.ood_fallback).sum()),
            })
        out["benefit_prediction"] = []
        for method, frame in prediction[prediction.task.eq(task)].groupby("method", sort=True):
            out["benefit_prediction"].append({"method": method,
                "benefit_mae_pp": frame.benefit_error.abs().mean()*100,
                "constant_mean_mae_pp": frame.constant_benefit_error.abs().mean()*100,
                "benefit_rmse_pp": np.sqrt((frame.benefit_error**2).mean())*100,
                "constant_mean_rmse_pp": np.sqrt((frame.constant_benefit_error**2).mean())*100,
                "event_score_squared_error": frame.event_squared_error.mean(),
                "constant_prevalence_squared_error": frame.constant_event_squared_error.mean(),
                "observed_outer_loss_events": int(as_bool(frame.realized_loss_event).sum())})
        out["class_collapse"] = []
        for method, frame in ss.groupby("method", sort=True):
            out["class_collapse"].append({"method": method,
                "sensitivity_mean": frame.sensitivity.mean(), "specificity_mean": frame.specificity.mean(),
                "zero_sensitivity_participants": int((frame.sensitivity <= TOL).sum()),
                "zero_specificity_participants": int((frame.specificity <= TOL).sum()),
                "log_loss": frame.log_loss.mean(), "brier": frame.brier.mean()})
        review["tasks"][task] = out
    HERE.mkdir(exist_ok=True)
    (HERE / "descriptive_review.json").write_text(json.dumps(clean(review), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    compact = {"integrity": review["integrity"], "significant_BH12": review["significant_BH12"], "tasks": {}}
    for task, out in review["tasks"].items():
        compact["tasks"][task] = {k: v for k, v in out.items() if k not in ["seed_sensitivity", "class_collapse"]}
        compact["tasks"][task]["seed_sensitivity"] = [v for v in out["seed_sensitivity"] if v["method"] in ["cbsf_heldout", "cbsf_proxy"]]
        compact["tasks"][task]["class_collapse"] = [v for v in out["class_collapse"] if v["method"] in ["feature_mlp", "extra_trees", "always_fuse", "cbsf_heldout"]]
    print(json.dumps(clean(compact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
