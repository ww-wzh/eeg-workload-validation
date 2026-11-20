"""Scientific edge cases and independent gate/statistics equivalence checks."""
import sys
import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax
from statsmodels.stats.multitest import multipletests
import analyze_gates as a

sys.path.insert(0, str(a.r.HERE.parent / "first_dataset_correction"))
from corrected_gate import fit_gate as reference_gate, loss_labels


class AnalysisTests(unittest.TestCase):
    def test_signed_event_boundary_and_independent_gate_equivalence(self):
        rng = np.random.default_rng(211)
        for gains in [np.tile([-.1, -.05, -.049, 0., .05, .1], 4), np.zeros(24), -np.ones(24)*.1]:
            frame = pd.DataFrame(rng.normal(size=(24, 5)), columns=a.FEATURES)
            frame["meta_subject"] = np.arange(24)
            frame["benefit"] = gains
            for x in [np.zeros(5), np.ones(5)*100]:
                target = dict(zip(a.FEATURES, x))
                actual = a.fit_gate(frame, target)
                ref = reference_gate(frame, target)
                self.assertEqual(actual["loss_events"], int(loss_labels(gains).sum()))
                self.assertAlmostEqual(actual["weight"], ref["weight"], places=13)
                self.assertAlmostEqual(actual["predicted_benefit"], ref["predicted_benefit"], places=13)
                self.assertAlmostEqual(actual["predicted_event_score"], ref["predicted_catastrophe"], places=13)
                self.assertEqual(actual["ood_fallback"], ref["ood_fallback"])

    def test_geometric_record_aggregation_and_order(self):
        p = np.array([[.7,.3],[.2,.8],[.9,.1],[.8,.2]])
        rec = np.array(["a","a","b","b"])
        y = np.array([0,0,1,1])
        ids, actual = a.aggregate(p, rec)
        independent = np.vstack([softmax(np.log(p[rec == v]).mean(axis=0)) for v in ids])
        np.testing.assert_allclose(actual, independent, atol=1e-15)
        swapped = np.array([3, 1, 0, 2])
        np.testing.assert_allclose(a.aggregate(p[swapped], rec[swapped])[1], actual, atol=1e-15)
        self.assertEqual(a.scores(p, rec, y)["balanced_accuracy"], 0.)

    def test_dasf_participant_macro_and_median_order(self):
        neural = [[.1,.2],[.5,.5],[.9,.8],[.4,.5],[.7,.6]]
        fused = [[.9,.8],[.5,.6],[.2,.1],[.8,.9],[.3,.4]]
        frame = pd.DataFrame([{"kind":"heldout","seed":s,"meta_subject":m,"neural_ba":n,"fused_ba":f}
            for s in range(5) for m,(n,f) in enumerate(zip(neural[s],fused[s]))])
        expected = np.median(np.mean(fused,axis=1))-np.median(np.mean(neural,axis=1))
        gate = a.dasf(frame,"heldout",.5)
        self.assertAlmostEqual(gate["pooled_gain"],expected)
        self.assertAlmostEqual(gate["margin"],.12)
        self.assertEqual(gate["weight"],float(expected>.12))

    def test_seed_and_deletion_paths_retain_all_participants(self):
        rng = np.random.default_rng(44)
        rows=[]
        for seed in [1335,1388,1441,1494,1547]:
            for meta in range(2,26):
                for kind in ["proxy","heldout"]:
                    n,f=rng.choice([0,.5,1],2)
                    rows.append({"kind":kind,"meta_subject":meta,"seed":seed,"neural_ba":n,
                                 "fused_ba":f,"benefit":f-n,**dict(zip(a.FEATURES,rng.random(5)))})
        seed_rows=pd.DataFrame(rows)
        med=seed_rows.groupby(["kind","meta_subject"])[["neural_ba","fused_ba","benefit",*a.FEATURES]].median().reset_index()
        probs={k:a.normalize(rng.random((40,2))) for k in [*a.r.BRANCHES,"always_fuse"]}
        dd=dict(zip(a.FEATURES,[.2]*5))
        for omitted in [None,2]:
            mm=med if omitted is None else med[med.meta_subject!=omitted]
            ss=seed_rows if omitted is None else seed_rows[seed_rows.meta_subject!=omitted]
            methods,gates=a.evaluate_gates(mm,ss,dd,.1,probs,ablations=True)
            self.assertEqual(len(methods),12)
            self.assertEqual(len(gates),7)
            for g in gates:
                if g["method"].startswith("cbsf"):
                    self.assertEqual(g["training_rows"],24 if omitted is None else 23)
        single=seed_rows[seed_rows.seed==1335]
        methods,gates=a.evaluate_gates(single,single,dd,.1,probs)
        self.assertEqual(len(methods),10)

    def test_BH12_with_ties_against_statsmodels(self):
        protocol=a.r.load(a.r.DESIGN/"protocol.json")
        rng=np.random.default_rng(99)
        frame=pd.DataFrame([{"mode":"ensemble","task":task,"subject":s,"method":method,
            "balanced_accuracy":float(rng.choice([0,.25,.5,.75,1]))}
            for task in protocol["tasks"] for s in protocol["cohort"]["expected_subjects"]
            for method in protocol["scored_methods"]])
        frame.loc[frame.method=="dasf_heldout","balanced_accuracy"] = frame.loc[frame.method=="dasf_proxy","balanced_accuracy"].to_numpy()
        means,comparisons=a.primary_statistics(frame,protocol)
        self.assertEqual(len(comparisons),12)
        self.assertTrue(np.all(comparisons.loc[comparisons.contrast=="dasf_heldout - dasf_proxy","p_two_sided"]==1))
        np.testing.assert_allclose(comparisons.p_BH12,multipletests(comparisons.p_two_sided,method="fdr_bh")[1],atol=1e-15)


if __name__=="__main__":
    unittest.main(verbosity=2)
