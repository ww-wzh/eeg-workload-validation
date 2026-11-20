"""Independent verification from saved COG window probabilities and responses.

This verifier reconstructs recording probabilities with a log-softmax formula,
checks participant scores and the BH12 family, and replays every fitted gate
against the separately audited first-dataset implementation. It never refits
EEG models or changes published evidence.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from statsmodels.stats.multitest import multipletests
import run_nested as r

sys.path.insert(0, str(r.HERE.parent / "first_dataset_correction"))
from corrected_gate import fit_gate, FEATURES
from revision_pipeline.risk_gate import robust_feature_shift, binary_entropy

OUT = r.HERE / "analysis"


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arr:
        return {k: arr[k] for k in arr.files}


def score_independent(prob, recordings, labels):
    ids = sorted(set(recordings))
    rp = np.array([softmax(np.log(np.clip(prob[recordings == rec], 1e-12, 1)).mean(axis=0)) for rec in ids])
    y = np.array([labels[recordings == rec][0] for rec in ids])
    correct = rp.argmax(axis=1) == y
    assert (y == 0).sum() == (y == 1).sum()
    return ids, rp, float(.5*(correct[y==0].mean()+correct[y==1].mean()))


def check_gates(meta, seed_rows, gates, deletion, subjects):
    checked = 0
    max_error = 0.
    for row in pd.concat([gates, deletion], ignore_index=True).to_dict("records"):
        task = row["task"]
        subject = int(row["subject"])
        mode = row.get("mode")
        deleting = pd.notna(row.get("omitted_meta_subject"))
        evidence = meta if deleting or mode == "ensemble" else seed_rows[seed_rows.seed == int(row["seed"])]
        evidence = evidence[(evidence.task == task) & (evidence.outer_subject == subject)]
        if deleting:
            evidence = evidence[evidence.meta_subject != int(row["omitted_meta_subject"])]
            # Target diagnostics do not change when a meta participant is removed.
            full_target = gates[(gates.task == task) & (gates.subject == subject) &
                                (gates["mode"] == "ensemble") & (gates.method == row["method"])].iloc[0]
            target = {k: float(full_target[k]) for k in FEATURES}
        else:
            target = {k: float(row[k]) for k in FEATURES}
        if row["method"].startswith("cbsf"):
            kind = "proxy" if row["method"] == "cbsf_proxy" else "heldout"
            train = evidence[evidence.kind == kind].sort_values("meta_subject").copy()
            if row["method"] == "cbsf_response_bridge":
                proxy = evidence[evidence.kind == "proxy"].set_index("meta_subject")
                train["benefit"] = train.meta_subject.map(proxy.benefit)
            expected = set(subjects)-{subject}
            if deleting:
                expected.remove(int(row["omitted_meta_subject"]))
            assert set(train.meta_subject) == expected and len(train) == len(expected)
            ref = fit_gate(train, target, loss_cutoff=-.05)
            weight = ref["weight"]
            if row["method"] == "cbsf_heldout_no_risk_penalty":
                weight = 0. if ref["ood_fallback"] else .8*expit(np.clip(ref["predicted_benefit"]/.02,-40,40))
            if row["method"] == "cbsf_heldout_no_ood_guard":
                weight = .8*expit(np.clip((ref["predicted_benefit"]-.05*ref["predicted_catastrophe"])/.02,-40,40))
            errs = [abs(weight-row["weight"]), abs(ref["predicted_benefit"]-row["predicted_benefit"]),
                    abs(ref["predicted_catastrophe"]-row["predicted_event_score"])]
            assert max(errs) < 2e-10
            max_error = max(max_error, *errs)
            assert int(row["loss_events"]) == ref["loss_events"]
            assert int(row["training_rows"]) == len(expected)
        else:
            evidence = seed_rows[(seed_rows.task == task) & (seed_rows.outer_subject == subject)]
            if not deleting and mode != "ensemble":
                evidence = evidence[evidence.seed == int(row["seed"])]
            if deleting:
                evidence = evidence[evidence.meta_subject != int(row["omitted_meta_subject"])]
            kind = row["method"].removeprefix("dasf_")
            pool = evidence[evidence.kind == kind].groupby("seed")[["fused_ba","neural_ba"]].mean()
            gain = float(np.median(pool.fused_ba)-np.median(pool.neural_ba))
            assert abs(gain-row["pooled_gain"]) < 1e-12
            assert float(gain > row["margin"]) == row["weight"]
        checked += 1
    return {"gate_fits_and_decisions_replayed": checked, "max_gate_error": max_error}


def main():
    status = r.load(OUT / "analysis_status.json")
    assert status["status"] == "analysis_complete_pending_independent_verification"
    protocol = r.load(r.DESIGN / "protocol.json")
    subjects = protocol["cohort"]["expected_subjects"]
    seeds = protocol["seeds"]
    meta = pd.read_csv(OUT / "meta_rows.csv", float_precision="round_trip")
    seed_rows = pd.read_csv(OUT / "meta_seed_rows.csv", float_precision="round_trip")
    scored = pd.read_csv(OUT / "participant_scores.csv", float_precision="round_trip")
    records = pd.read_csv(OUT / "record_scores.csv", float_precision="round_trip")
    gates = pd.read_csv(OUT / "gate_diagnostics.csv", float_precision="round_trip")
    deletion = pd.read_csv(OUT / "meta_participant_deletion.csv", float_precision="round_trip")
    comparisons = pd.read_csv(OUT / "comparisons_BH12.csv", float_precision="round_trip")
    assert len(meta) == 2400 and len(seed_rows) == 12000 and len(deletion) == 6000
    # Paired kind rows remain 1200, drawn from the same 25 people.
    assert meta[["task","outer_subject","meta_subject"]].drop_duplicates().shape[0] == 1200
    expected_medians = seed_rows.groupby(["task","outer_subject","meta_subject","kind"])[
        ["neural_ba","fused_ba","benefit",*FEATURES]].median().sort_index()
    got = meta.set_index(["task","outer_subject","meta_subject","kind"])[expected_medians.columns].sort_index()
    np.testing.assert_allclose(expected_medians, got, atol=1e-14, rtol=0)
    max_prediction_error = 0.
    checks = {"independent_participants":25,"tasks":2,"record_aggregations_checked":0,
              "seed_response_gains_recomputed":0,"nested_memberships_checked":0}
    for job in r.load(r.DESIGN / "job_plan.json")["jobs"]:
        path = r.HERE / "jobs" / job["job_id"]
        summary = r.load(path / "job_summary.json")
        assert summary["target_labels_received"] is False
        assert summary["inner_coverage_min"] == summary["inner_coverage_max"] == 1
        assert summary["final_epoch"] == max(1,int(np.median(summary["inner_best_epochs"])))
        for subject in job["training_subjects"]:
            for branch in r.BRANCHES:
                fit = r.load(path / "stages" / f"inner_s{subject:02d}_{branch}" / "fit.json")
                assert set(fit["train_membership"]["subjects"]) == set(job["training_subjects"])-{subject}
                assert fit["validation_membership"]["subjects"] == [subject]
                assert fit["prediction_membership"]["subjects"] == job["prediction_subjects"]
                assert fit["target_labels_received"] is False
                checks["nested_memberships_checked"] += 1
    for task in protocol["tasks"]:
        data = r.TaskData(task)
        task_seed_rows = seed_rows[seed_rows.task == task]
        for outer in subjects:
            oi = data.indices([outer])
            ti = data.indices([s for s in subjects if s != outer])
            saved_seeds = []
            for seed in seeds:
                path = r.HERE / "jobs" / f"{task}_outer_s{outer:02d}_seed{seed}"
                pp = read_arrays(path/f"target_s{outer:02d}_predictions.npz")
                assert np.array_equal(pp["window_ids"], data.window[oi])
                saved_seeds.append(pp)
                proxy = read_arrays(path/"oof_predictions.npz")
                assert np.array_equal(proxy["window_ids"], data.window[ti])
                for meta_subject in subjects:
                    if meta_subject == outer:
                        continue
                    first,second = sorted([outer,meta_subject])
                    held_path = r.HERE / "jobs" / f"{task}_exclude_s{first:02d}_s{second:02d}_seed{seed}"
                    held = read_arrays(held_path/f"target_s{meta_subject:02d}_predictions.npz")
                    mi = data.indices([meta_subject])
                    assert np.array_equal(held["window_ids"],data.window[mi])
                    pm = proxy["subject"] == meta_subject
                    reference_idx = data.indices([s for s in subjects if s not in [outer,meta_subject]])
                    for kind,pn,pf in [("proxy",proxy["feature_mlp"][pm],proxy["always_fuse"][pm]),
                                      ("heldout",held["feature_mlp"],held["always_fuse"])]:
                        nn = score_independent(pn,data.record[mi],data.y[mi])[2]
                        ff = score_independent(pf,data.record[mi],data.y[mi])[2]
                        row = task_seed_rows[(task_seed_rows.outer_subject==outer)&(task_seed_rows.meta_subject==meta_subject)&
                            (task_seed_rows.seed==seed)&(task_seed_rows.kind==kind)].iloc[0]
                        assert max(abs(row.benefit-(ff-nn)),abs(row.neural_ba-nn),abs(row.fused_ba-ff)) < 1e-13
                        dd = [robust_feature_shift(data.x[reference_idx],data.x[mi]),float(np.abs(pn[:,1]-pf[:,1]).mean()),
                              binary_entropy(pn),binary_entropy(pf),float(np.max(pn,axis=1).mean())]
                        np.testing.assert_allclose(row[FEATURES].to_numpy(dtype=float),dd,atol=2e-13,rtol=0)
                        checks["seed_response_gains_recomputed"] += 1
            ensemble = {}
            for name in [*r.BRANCHES,"always_fuse"]:
                p = np.median([v[name] for v in saved_seeds],axis=0)
                ensemble[name] = p/p.sum(axis=1,keepdims=True)
            participant_rows = scored[(scored.task==task)&(scored.subject==outer)]
            for row in participant_rows.to_dict("records"):
                if row["mode"] in ["ensemble","gate_only_seed"]:
                    base = ensemble
                else:
                    base = saved_seeds[seeds.index(int(row["seed"]))]
                if row["method"] in [*r.BRANCHES,"always_fuse"]:
                    pred = base[row["method"]]
                else:
                    if row["method"] == "fixed_half":
                        weight = .5
                    else:
                        gr = gates[(gates.task==task)&(gates.subject==outer)&(gates["mode"]==row["mode"])&
                                   (gates.seed==row["seed"])&(gates.method==row["method"])]
                        assert len(gr)==1
                        weight = float(gr.iloc[0].weight)
                    pred = (1-weight)*base["feature_mlp"]+weight*base["always_fuse"]
                    pred = pred/pred.sum(axis=1,keepdims=True)
                ids,rp,ba = score_independent(pred,data.record[oi],data.y[oi])
                assert abs(row["balanced_accuracy"]-ba)<1e-12
                checks["record_aggregations_checked"] += 1
                if row["mode"]=="ensemble":
                    saved_records = records[(records.task==task)&(records.subject==outer)&(records.method==row["method"])].set_index("record_id").loc[ids]
                    err = float(np.max(np.abs(saved_records[["p0","p1"]].to_numpy()-rp)))
                    assert err < 2e-12
                    max_prediction_error=max(max_prediction_error,err)
        print(f"Independent prediction/gain reconstruction complete: {task}",flush=True)
    checks.update(check_gates(meta,seed_rows,gates,deletion,subjects))
    rng=np.random.default_rng(protocol["statistics"]["inference_seed"])
    bootstrap=rng.integers(0,25,(10000,25))
    signs=rng.choice(np.array([-1,1],dtype=np.int8),size=(100000,25))
    for row in comparisons.to_dict("records"):
        left,right=row["contrast"].split(" - ")
        source=scored[(scored.task==row["task"])&(scored["mode"]=="ensemble")]
        wide=source.pivot(index="subject",columns="method",values="balanced_accuracy").loc[subjects]
        delta=(wide[left]-wide[right]).to_numpy()
        null=np.abs((delta[None,:]*signs).mean(axis=1))
        p=(1+int((null>=abs(delta.mean())-1e-12).sum()))/100001
        ci=np.percentile(delta[bootstrap].mean(axis=1),[2.5,97.5])
        assert abs(p-row["p_two_sided"])<1e-13
        np.testing.assert_allclose(ci,[row["ci_low"],row["ci_high"]],atol=1e-13,rtol=0)
        assert abs(delta.mean()-row["mean_difference"])<1e-13
    np.testing.assert_allclose(comparisons.p_BH12,multipletests(comparisons.p_two_sided,method="fdr_bh")[1],atol=1e-13,rtol=0)
    checks.update({"status":"passed","utc":r.now(),"max_record_probability_error":max_prediction_error,
        "primary_contrasts_checked":12,"BH12_verified_with_statsmodels":True,
        "analysis_sha256":r.sha(r.HERE/"analyze_gates.py"),"verifier_sha256":r.sha(Path(__file__)),
        "remote_files_modified":False})
    r.dump(OUT/"independent_verification.json",checks)
    manifest={p.relative_to(r.HERE).as_posix():r.sha(p) for p in OUT.iterdir() if p.is_file()}
    r.dump(OUT/"verified_outputs_manifest.json",manifest)
    print(checks,flush=True)


if __name__=="__main__":
    main()
