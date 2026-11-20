"""187: descriptive metrics and gate-event audit for 185 predictions."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score, precision_recall_curve, auc

HERE = Path(__file__).resolve().parent
BASE = HERE
NEW = BASE / "heldout_gain_reconstruction"
OUT = BASE / "heldout_gain_metrics"
PRED = NEW / "heldout_gate_recording_predictions.csv"
DIAG = NEW / "heldout_gate_diagnostics.csv"
TRAIN = NEW / "cross_task_cbsf_training_rows.csv"
EPS = 1e-12

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def metrics(g: pd.DataFrame) -> dict:
    y = g.true_label.to_numpy(int); yp = g.pred_label.to_numpy(int); p = np.clip(g.p1.to_numpy(float), EPS, 1-EPS)
    neg, pos = y == 0, y == 1
    tn, fp = int(np.sum(neg & (yp==0))), int(np.sum(neg & (yp==1)))
    fn, tp = int(np.sum(pos & (yp==0))), int(np.sum(pos & (yp==1)))
    sens = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    prec = tp/(tp+fp) if tp+fp else np.nan
    out = {"n_recordings":len(g),"n_class_0":int(neg.sum()),"n_class_1":int(pos.sum()),"accuracy":float(np.mean(y==yp)),
           "balanced_accuracy":float(np.nanmean([sens,spec])),"sensitivity":sens,"specificity":spec,"precision":prec,
           "f1":(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan),"log_loss":float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
           "brier_score":float(np.mean((g.p1.to_numpy(float)-y)**2))}
    if np.unique(y).size == 2:
        out["auroc"] = float(roc_auc_score(y,p)); pr, rec, _ = precision_recall_curve(y,p); out["pr_auc"] = float(auc(rec,pr)); out["average_precision"] = float(average_precision_score(y,p))
    else: out.update({"auroc":np.nan,"pr_auc":np.nan,"average_precision":np.nan})
    return out

def main() -> None:
    for p in (PRED, DIAG, TRAIN):
        if not p.exists(): raise FileNotFoundError(p)
    d = pd.read_csv(PRED); diag = pd.read_csv(DIAG); train = pd.read_csv(TRAIN)
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for keys,g in d.groupby(["protocol","direction","method"], sort=True): rows.append(dict(zip(["protocol","direction","method"],keys), **metrics(g)))
    for keys,g in d.groupby(["protocol","direction","subject","method"], sort=True): rows.append(dict(zip(["protocol","direction","subject","method"],keys), **metrics(g)))
    pd.DataFrame(rows).to_csv(OUT/"187_recording_level_and_participant_metrics.csv",index=False,float_format="%.17g")
    # Event counts use the actual frozen training rows.  The threshold is <= -0.05.
    train["loss_event"] = train["benefit"] <= -0.05
    event = train.groupby("direction", as_index=False).agg(training_rows=("benefit","size"), loss_events=("loss_event","sum"), non_loss_events=("loss_event",lambda x:int((~x).sum())), min_benefit=("benefit","min"), max_benefit=("benefit","max"))
    event["threshold"]=-0.05; event["event_definition"]="benefit <= -0.05"
    overall = pd.DataFrame([{"direction":"pooled","training_rows":len(train),"loss_events":int(train.loss_event.sum()),"non_loss_events":int((~train.loss_event).sum()),"min_benefit":float(train.benefit.min()),"max_benefit":float(train.benefit.max()),"threshold":-0.05,"event_definition":"benefit <= -0.05"}])
    pd.concat([event,overall],ignore_index=True).to_csv(OUT/"gate_event_counts.csv",index=False,float_format="%.17g")
    cell = diag.groupby(["method","setting"],as_index=False).agg(gate_cells=("subject","size"),selected_fusion=("selected_fusion","sum"),ood_fallback=("ood_fallback","sum"),mean_weight=("weight","mean"),mean_predicted_catastrophe=("predicted_catastrophe","mean"),mean_max_abs_robust_z=("max_abs_robust_z","mean"))
    cell.to_csv(OUT/"gate_selection_and_fallback.csv",index=False,float_format="%.17g")
    manifest={"status":"passed","prediction_sha256":sha256(PRED),"diagnostics_sha256":sha256(DIAG),"training_rows_sha256":sha256(TRAIN),"metrics_scope":"185 held-out-gain methods; descriptive recording/participant metrics","event_threshold":"benefit <= -0.05","event_counts_pooled":{"training_rows":int(len(train)),"loss_events":int(train.loss_event.sum()),"non_loss_events":int((~train.loss_event).sum())},"no_retraining":True}
    (OUT/"187_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("187 complete metrics and gate-event audit completed")
    print(event.to_string(index=False)); print("Pooled loss events:",int(train.loss_event.sum()),"of",len(train)); print("Output:",OUT)

if __name__ == "__main__": main()
