"""B3-B5 analysis of a complete frozen COG gate run; never selects models.

No partial-cohort inferential results are generated. Original per-window
predictions are converted to participant-record scores only after all 3250
model-set receipts pass integrity checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits
import run_nested as r
from revision_pipeline.risk_gate import robust_feature_shift, binary_entropy

FEATURES = ["feature_shift", "disagreement", "entropy_nn", "entropy_fuse", "confidence_nn"]
OUT = r.HERE / "analysis"


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    assert np.isfinite(p).all() and np.all(p >= 0) and np.all(p.sum(axis=1) > 0)
    return p / p.sum(axis=1, keepdims=True)


def aggregate(p, records):
    ids = np.unique(records)
    # Exactly the clipping and geometric normalization in the pinned trainer.
    values = [np.exp(np.mean(np.log(np.clip(p[records == rec], 1e-12, 1.)), axis=0)) for rec in ids]
    return ids, normalize(np.array(values))


def scores(p, records, labels):
    ids, rp = aggregate(p, records)
    ry = np.array([labels[records == rec][0] for rec in ids])
    assert all(len(np.unique(labels[records == rec])) == 1 for rec in ids)
    pred = rp.argmax(axis=1)
    assert np.sum(ry == 0) == np.sum(ry == 1)
    ba = .5 * (np.mean(pred[ry == 0] == 0) + np.mean(pred[ry == 1] == 1))
    assert abs(ba - np.mean(pred == ry)) < 1e-14
    wp = p.argmax(axis=1)
    wb = .5 * (np.mean(wp[labels == 0] == 0) + np.mean(wp[labels == 1] == 1))
    return {"balanced_accuracy": float(ba), "window_balanced_accuracy": float(wb),
            "log_loss": float(-np.log(np.clip(rp[np.arange(len(ry)), ry], 1e-12, 1)).mean()),
            "brier": float(np.mean((rp[:, 1] - ry) ** 2)),
            "sensitivity": float(np.mean(pred[ry == 1] == 1)),
            "specificity": float(np.mean(pred[ry == 0] == 0)),
            "record_ids": ids, "record_probability": rp, "record_labels": ry}


def diagnostics(neural, fused, x_source, x_target):
    return {"feature_shift": robust_feature_shift(x_source, x_target),
            "disagreement": float(np.abs(neural[:, 1] - fused[:, 1]).mean()),
            "entropy_nn": binary_entropy(neural), "entropy_fuse": binary_entropy(fused),
            "confidence_nn": float(np.max(neural, axis=1).mean())}


def fit_gate(rows, target, *, risk_penalty=.05, ood_guard=True):
    # Explicit signed cutoff prevents recurrence of the historical double minus.
    assert set(target) == set(FEATURES)
    assert rows.meta_subject.nunique() == len(rows)
    x = rows[FEATURES].to_numpy(dtype=np.float64)
    gains = rows.benefit.to_numpy(dtype=np.float64)
    t = np.array([[target[k] for k in FEATURES]])
    assert len(x) >= 5 and np.isfinite(x).all() and np.isfinite(gains).all() and np.isfinite(t).all()
    event = (gains <= -.05).astype(np.int64)
    center = np.median(x, axis=0)
    q1, q3 = np.quantile(x, [.25, .75], axis=0)
    scale = np.where(q3-q1 >= 1e-8, q3-q1, x.std(axis=0, ddof=0))
    scale = np.where(scale >= 1e-8, scale, 1.)
    tz = (t-center)/scale
    max_z = float(np.abs(tz).max())
    xz = np.clip((x-center)/scale, -5, 5)
    tz = np.clip(tz, -5, 5)
    with threadpool_limits(limits=1):
        benefit_model = Ridge(alpha=1.).fit(xz, gains)
        raw = float(benefit_model.predict(tz)[0])
        benefit = float(np.clip(raw, max(-1., gains.min()), min(1., gains.max())))
        if len(np.unique(event)) == 2:
            em = LogisticRegression(C=1., class_weight="balanced", max_iter=2000, random_state=0).fit(xz, event)
            assert np.all(em.n_iter_ < 2000), "Event fit reached iteration cap"
            ev = float(em.predict_proba(tz)[0, 1])
        else:
            ev = float(event.mean())
    ood = bool(ood_guard and max_z > 10.)
    weight = 0. if ood else .8 * expit(np.clip((benefit-risk_penalty*ev)/.02, -40., 40.))
    return {"weight": float(weight), "predicted_benefit": benefit, "raw_benefit": raw,
            "predicted_event_score": ev, "max_abs_robust_z": max_z, "ood_fallback": ood,
            "loss_events": int(event.sum()), "nonevents": int(len(event)-event.sum()),
            "constant_event_fallback": bool(len(np.unique(event)) == 1),
            "training_mean_benefit": float(gains.mean()), "training_mean_event": float(event.mean()),
            "positive_gains": int((gains > 0).sum()), "negative_gains": int((gains < 0).sum()),
            "zero_gains": int((gains == 0).sum()), "training_rows": len(rows),
            "loss_cutoff": -.05, "risk_penalty": risk_penalty, "ood_guard": ood_guard}


def verify_all_jobs():
    freeze = r.load(r.HERE / "execution_freeze.json")
    assert freeze["identity"]["code_sha256"][str(Path(r.__file__).relative_to(r.ROOT))] == r.sha(Path(r.__file__))
    jobs = r.load(r.DESIGN / "job_plan.json")["jobs"]
    missing = [j["job_id"] for j in jobs if not (r.HERE / "jobs" / j["job_id"] / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"Full study is incomplete: {len(missing)} of {len(jobs)} jobs missing. No group results produced.")
    checked = 0
    for j in jobs:
        path = r.HERE / "jobs" / j["job_id"]
        receipt = r.load(path / "complete.json")
        assert receipt["identity"] == {"execution": freeze["identity_sha256"], "job": j}
        for name, expected in receipt["files_sha256"].items():
            assert r.sha(path / name) == expected
            checked += 1
            if name.startswith("stages/"):
                stage = path / name
                for payload, hp in r.load(stage)["files_sha256"].items():
                    assert r.sha(stage.parent / payload) == hp
                    checked += 1
    return {"jobs_verified": len(jobs), "hashed_payloads": checked,
            "execution": freeze["identity_sha256"]}


def job_path(task, excluded, seed):
    excluded = sorted(excluded)
    if len(excluded) == 1:
        name = f"{task}_outer_s{excluded[0]:02d}_seed{seed}"
    else:
        name = f"{task}_exclude_s{excluded[0]:02d}_s{excluded[1]:02d}_seed{seed}"
    return r.HERE / "jobs" / name


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def cov_shift(data, source_idx, target_idx):
    # Each record contributes exactly 20 windows. Preserve the specified
    # window-weighted source/target second moments, not participant weighting.
    mats = []
    for row in data.record_rows:
        with np.load(r.DESIGN / "prepared_records_float64" / (row["record_id"] + ".npz")) as arr:
            mats.append(arr["covariances"])
    cov = np.concatenate(mats)
    cs, ct = cov[source_idx].mean(axis=0), cov[target_idx].mean(axis=0)
    return float(np.linalg.norm(ct-cs, "fro") / (.5*(np.linalg.norm(ct, "fro")+np.linalg.norm(cs, "fro"))))


def build_responses(task, outer, data, seeds, subjects):
    seed_rows = []
    outer_predictions = []
    oi = data.indices([outer])
    train_idx = data.indices([s for s in subjects if s != outer])
    for seed in seeds:
        path = job_path(task, [outer], seed)
        target = arrays(path / f"target_s{outer:02d}_predictions.npz")
        assert np.array_equal(target["window_ids"], data.window[oi])
        outer_predictions.append({k: target[k] for k in [*r.BRANCHES, "always_fuse"]})
        proxy = arrays(path / "oof_predictions.npz")
        assert np.array_equal(proxy["window_ids"], data.window[train_idx])
        for meta in subjects:
            if meta == outer:
                continue
            mi = data.indices([meta])
            ref = data.indices([s for s in subjects if s not in [outer, meta]])
            pm = proxy["subject"] == meta
            assert np.array_equal(proxy["window_ids"][pm], data.window[mi])
            held = arrays(job_path(task, [outer, meta], seed) / f"target_s{meta:02d}_predictions.npz")
            assert np.array_equal(held["window_ids"], data.window[mi])
            for kind, neural, fused in [("proxy", proxy["feature_mlp"][pm], proxy["always_fuse"][pm]),
                                         ("heldout", held["feature_mlp"], held["always_fuse"])]:
                ns = scores(neural, data.record[mi], data.y[mi])["balanced_accuracy"]
                fs = scores(fused, data.record[mi], data.y[mi])["balanced_accuracy"]
                seed_rows.append({"task": task, "outer_subject": outer, "meta_subject": meta,
                    "seed": seed, "kind": kind, "neural_ba": ns, "fused_ba": fs, "benefit": fs-ns,
                    "source_participants": len(subjects)-2,
                    **diagnostics(neural, fused, data.x[ref], data.x[mi])})
    frame = pd.DataFrame(seed_rows)
    # Compute each participant's gain within seed BEFORE taking its median.
    rows = frame.groupby(["task", "outer_subject", "meta_subject", "kind"], sort=True)[
        ["neural_ba", "fused_ba", "benefit", *FEATURES]].median().reset_index()
    assert len(rows) == 48 and len(frame) == 240
    ensemble = {k: normalize(np.median([p[k] for p in outer_predictions], axis=0))
                for k in outer_predictions[0]}
    return frame, rows, outer_predictions, ensemble, train_idx, oi


def dasf(rows, kind, shift):
    # Seed means are equal-weight means of participants. Median of branch means
    # is intentionally different from mean/median of per-participant gain.
    pooled = rows[rows.kind == kind].groupby("seed")[["neural_ba", "fused_ba"]].mean()
    gain = float(pooled.fused_ba.median() - pooled.neural_ba.median())
    margin = float(np.clip(.02+.2*shift, 0, .15))
    return {"weight": float(gain > margin), "pooled_gain": gain, "margin": margin, "covariance_shift": shift}


def branch_blend(neural, fused, weight):
    return normalize((1-weight)*neural+weight*fused)


def evaluate_gates(meta, seed_rows, target_diagnostics, shift, outer_probs, ablations=False):
    methods = {k: v for k, v in outer_probs.items()}
    methods["fixed_half"] = branch_blend(outer_probs["feature_mlp"], outer_probs["always_fuse"], .5)
    decisions = []
    for kind in ["proxy", "heldout"]:
        gate = dasf(seed_rows, kind, shift)
        name = "dasf_"+kind
        methods[name] = branch_blend(outer_probs["feature_mlp"], outer_probs["always_fuse"], gate["weight"])
        decisions.append({"method": name, **gate})
    for name in ["cbsf_proxy", "cbsf_heldout", "cbsf_response_bridge"]:
        kind = "proxy" if name == "cbsf_proxy" else "heldout"
        train = meta[meta.kind == kind].sort_values("meta_subject").copy()
        if name == "cbsf_response_bridge":
            proxy = meta[meta.kind == "proxy"].set_index("meta_subject")
            train["benefit"] = train.meta_subject.map(proxy.benefit)
        gate = fit_gate(train, target_diagnostics)
        methods[name] = branch_blend(outer_probs["feature_mlp"], outer_probs["always_fuse"], gate["weight"])
        decisions.append({"method": name, **gate})
    if ablations:
        train = meta[meta.kind == "heldout"].copy()
        for name, kwargs in [("cbsf_heldout_no_risk_penalty", {"risk_penalty": 0.}),
                             ("cbsf_heldout_no_ood_guard", {"ood_guard": False})]:
            gate = fit_gate(train, target_diagnostics, **kwargs)
            methods[name] = branch_blend(outer_probs["feature_mlp"], outer_probs["always_fuse"], gate["weight"])
            decisions.append({"method": name, **gate})
    return methods, decisions


def add_scores(destination, records_dest, task, outer, mode, seed, methods, data, oi):
    for name, p in methods.items():
        sc = scores(p, data.record[oi], data.y[oi])
        common = {"task": task, "subject": outer, "mode": mode, "seed": seed, "method": name}
        destination.append({**common, **{k: v for k, v in sc.items() if not k.startswith("record_")}})
        if records_dest is not None:
            for rec, y, prob in zip(sc["record_ids"], sc["record_labels"], sc["record_probability"]):
                records_dest.append({**common, "record_id": rec, "label": int(y),
                                     "p0": prob[0], "p1": prob[1], "prediction": int(prob.argmax())})


def primary_statistics(scores_frame, protocol):
    rng = np.random.default_rng(protocol["statistics"]["inference_seed"])
    subjects = protocol["cohort"]["expected_subjects"]
    # Same matrix across both tasks and every comparison.
    bootstrap = rng.integers(0, len(subjects), (10000, len(subjects)))
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(100000, len(subjects)))
    result, means = [], []
    main = scores_frame[scores_frame["mode"] == "ensemble"]
    for task in protocol["tasks"]:
        wide = main[main.task == task].pivot(index="subject", columns="method", values="balanced_accuracy").loc[subjects]
        assert len(wide) == 25 and not wide.isna().any().any()
        for name in wide.columns:
            values = wide[name].to_numpy()
            lo, hi = np.quantile(values[bootstrap].mean(axis=1), [.025, .975])
            means.append({"task": task, "method": name, "n": 25, "mean": values.mean(), "ci_low": lo, "ci_high": hi})
        for contrast in protocol["statistics"]["primary_contrasts_per_task"]:
            left, right = contrast.split(" - ")
            difference = (wide[left]-wide[right]).to_numpy()
            lo, hi = np.quantile(difference[bootstrap].mean(axis=1), [.025, .975])
            statistic = abs(difference.mean())
            null = np.abs((signs @ difference)/len(difference))
            p = (np.sum(null >= statistic-1e-12)+1)/(len(null)+1)
            result.append({"task": task, "contrast": contrast, "n": 25, "mean_difference": difference.mean(),
                           "ci_low": lo, "ci_high": hi, "p_two_sided": p,
                           "positive": int((difference > 0).sum()), "zero": int((difference == 0).sum()),
                           "negative": int((difference < 0).sum())})
    assert len(result) == 12
    ps = np.array([a["p_two_sided"] for a in result])
    order = np.argsort(ps)
    adjusted = np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
    for idx, q in zip(order, np.minimum(adjusted, 1)):
        result[idx]["p_BH12"] = q
    return pd.DataFrame(means), pd.DataFrame(result)


def freeze_analysis():
    ident = {"analysis_sha256": r.sha(Path(__file__)),
             "protocol_sha256": r.sha(r.DESIGN / "protocol.json"),
             "risk_helpers_sha256": r.sha(r.SOURCE / "revision_pipeline/risk_gate.py")}
    path = r.HERE / "analysis_freeze.json"
    if path.exists():
        assert r.load(path)["identity"] == ident, "Analysis code changed after freeze"
    else:
        r.dump(path, {"identity": ident, "frozen_utc": r.now(), "new_gate_outcomes_examined": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "analyze"])
    args = parser.parse_args()
    freeze_analysis()
    if args.mode == "freeze":
        print("Gate scoring and diagnostics code frozen; no outcomes scored.")
        return
    checks = verify_all_jobs()
    protocol = r.load(r.DESIGN / "protocol.json")
    subjects, seeds = protocol["cohort"]["expected_subjects"], protocol["seeds"]
    all_seed_rows, all_meta, all_scores, record_scores, all_gates, deletion, complementary = [], [], [], [], [], [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for task in protocol["tasks"]:
        data = r.TaskData(task)
        for outer in subjects:
            sr, meta, seeded, ensemble, ti, oi = build_responses(task, outer, data, seeds, subjects)
            all_seed_rows.append(sr)
            all_meta.append(meta)
            diag = diagnostics(ensemble["feature_mlp"], ensemble["always_fuse"], data.x[ti], data.x[oi])
            shift = cov_shift(data, ti, oi)
            methods, decisions = evaluate_gates(meta, sr, diag, shift, ensemble, ablations=True)
            add_scores(all_scores, record_scores, task, outer, "ensemble", -1, methods, data, oi)
            ns = scores(ensemble["feature_mlp"], data.record[oi], data.y[oi])
            fs = scores(ensemble["always_fuse"], data.record[oi], data.y[oi])
            realized = fs["balanced_accuracy"] - ns["balanced_accuracy"]
            for g in decisions:
                all_gates.append({"task": task, "subject": outer, "mode": "ensemble", "seed": -1,
                                  "realized_branch_gain": realized, "realized_loss_event": realized <= -.05,
                                  **diag, **g})
            nc = ns["record_probability"].argmax(axis=1) == ns["record_labels"]
            fc = fs["record_probability"].argmax(axis=1) == fs["record_labels"]
            complementary.append({"task": task, "subject": outer, "both_correct": int((nc & fc).sum()),
                "neural_only": int((nc & ~fc).sum()), "fused_only": int((~nc & fc).sum()),
                "both_wrong": int((~nc & ~fc).sum()), "neural_ba": ns["balanced_accuracy"],
                "fused_ba": fs["balanced_accuracy"], "gain": realized,
                "participant_hard_oracle_ba": max(ns["balanced_accuracy"], fs["balanced_accuracy"])})
            # Two explicitly separate seed sensitivities: full per-seed pipeline
            # versus varying meta evidence with the final ensemble held fixed.
            for seed, prob in zip(seeds, seeded):
                single = sr[sr.seed == seed].copy()
                for mode, pp in [("complete_seed", prob), ("gate_only_seed", ensemble)]:
                    dd = diagnostics(pp["feature_mlp"], pp["always_fuse"], data.x[ti], data.x[oi])
                    sm, sg = evaluate_gates(single, single, dd, shift, pp)
                    add_scores(all_scores, None, task, outer, mode, seed, sm, data, oi)
                    all_gates.extend({"task": task, "subject": outer, "mode": mode, "seed": seed, **dd, **g} for g in sg)
            # Participant deletion modifies only gate fitting, including robust
            # transforms and one-class event fallbacks. It never retrains EEG.
            for omitted in subjects:
                if omitted == outer:
                    continue
                dm, dg = evaluate_gates(meta[meta.meta_subject != omitted], sr[sr.meta_subject != omitted],
                                       diag, shift, ensemble)
                for g in dg:
                    sc = scores(dm[g["method"]], data.record[oi], data.y[oi])
                    deletion.append({"task": task, "subject": outer, "omitted_meta_subject": omitted,
                                     "balanced_accuracy": sc["balanced_accuracy"], **g})
            print(f"Scored complete outer participant: {task} {outer:02d}", flush=True)
    frames = {"meta_seed_rows": pd.concat(all_seed_rows), "meta_rows": pd.concat(all_meta),
              "participant_scores": pd.DataFrame(all_scores), "record_scores": pd.DataFrame(record_scores),
              "gate_diagnostics": pd.DataFrame(all_gates), "meta_participant_deletion": pd.DataFrame(deletion),
              "branch_complementarity": pd.DataFrame(complementary)}
    assert len(frames["meta_seed_rows"]) == 12000  # 6000 paired rows, two kinds
    assert len(frames["meta_rows"]) == 2400  # 1200 paired rows, two kinds
    # Explicit paired evidence changes; keep outer/meta participant identifiers.
    paired = frames["meta_rows"].pivot(index=["task", "outer_subject", "meta_subject"],
        columns="kind", values="benefit").reset_index()
    paired["proxy_minus_heldout"] = paired.proxy-paired.heldout
    paired["proxy_loss_event"] = paired.proxy <= -.05
    paired["heldout_loss_event"] = paired.heldout <= -.05
    paired["event_changed"] = paired.proxy_loss_event != paired.heldout_loss_event
    frames["paired_response_changes"] = paired
    frames["response_changes_by_outer"] = paired.groupby(["task", "outer_subject"])[
        ["proxy_minus_heldout", "proxy_loss_event", "heldout_loss_event", "event_changed"]].mean().reset_index()
    gates = frames["gate_diagnostics"]
    primary_gates = gates[(gates["mode"] == "ensemble") & gates.method.str.startswith("cbsf_")].copy()
    primary_gates["benefit_error"] = primary_gates.predicted_benefit-primary_gates.realized_branch_gain
    primary_gates["constant_benefit_error"] = primary_gates.training_mean_benefit-primary_gates.realized_branch_gain
    primary_gates["event_squared_error"] = (primary_gates.predicted_event_score-primary_gates.realized_loss_event.astype(float))**2
    primary_gates["constant_event_squared_error"] = (primary_gates.training_mean_event-primary_gates.realized_loss_event.astype(float))**2
    frames["outer_gain_prediction_errors"] = primary_gates
    frames["method_summary"], frames["comparisons_BH12"] = primary_statistics(frames["participant_scores"], protocol)
    for name, frame in frames.items():
        frame.to_csv(OUT / (name+".csv"), index=False, float_format="%.17g")
    r.dump(OUT / "analysis_status.json", {"status": "analysis_complete_pending_independent_verification",
        "utc": r.now(), "input_integrity": checks,
        "paired_response_rows": 1200, "paired_seed_response_rows": 6000,
        "independent_participants": 25, "participants_per_task": 25,
        "row_counts": {k: len(v) for k, v in frames.items()},
        "source_sha256": r.sha(Path(__file__))})


if __name__ == "__main__":
    main()
