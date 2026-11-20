# Exact function excerpt from the included original COG verifier.
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
