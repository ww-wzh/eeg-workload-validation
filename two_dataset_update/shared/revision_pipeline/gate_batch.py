"""Full frozen LOSO meta-prediction schedule with explicit completed-pair adoption."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import gate_pairs as pairs


p = pairs.pilot
SCHEMA = "full-loso-meta-batch-v1"
FIRST_SUMMARY_SHA = "321d82082bf370f79eeb85a0378b7a95323394708b3f3c449379aada0f3c61c6"


class FeaturePool:
    def __init__(self, cache, cohorts, hashes):
        self.rows = {}
        self.counts = {}
        for task, cohort in cohorts.items():
            for subject in cohort:
                rows = p.load_cached_rows(cache, task, [subject], hashes, training=False)
                for field in ("features", "subjects", "recordings", "windows"):
                    getattr(rows, field).setflags(write=False)
                self.rows[(task, subject)] = rows
            self.counts[task] = {"participants": len(cohort), "recordings": len(cohort)*4,
                                 "windows": sum(len(self.rows[(task,s)].features) for s in cohort)}

    def __call__(self, job):
        retained = [self.rows[(job["task"], subject)] for subject in job["training_subjects"]]
        arrays = {field: np.concatenate([getattr(rows, field) for rows in retained])
                  for field in ("features", "subjects", "recordings", "windows")}
        # Only retained training recordings receive labels in the fitting input.
        labels = np.asarray([int(str(recording)[-1]) >= 2 for recording in arrays["recordings"]], dtype=np.int64)
        train = p.TrainingRows(**arrays, labels=labels)
        targets = {s:self.rows[(job["task"], s)] for s in job["excluded_subjects"]}
        pairs.validate_job(job, train, targets)
        return train, targets


def pair_identity(job, train, targets, config, context, backend_tag):
    return {"schema": pairs.SCHEMA, "job": job, "config": config, "backend": backend_tag,
            "training_rows": train.fingerprint(),
            "label_free_targets": {str(s):targets[s].fingerprint() for s in job["excluded_subjects"]},
            "context": context}


class ReadonlyCheckpoints(p.CheckpointStore):
    def __init__(self, root, identity):
        self.root = root
        self.identity = identity
        self.identity_digest = hashlib.sha256(p.json_bytes(identity)).hexdigest()
        self.directory = root / "checkpoints"
        saved = pairs.read_json(root / "01_run_identity.json")
        if saved.get("identity") != identity or saved.get("identity_sha256") != self.identity_digest:
            raise p.CheckpointError("Saved job identity differs from its current expected inputs.")


def read_fitted_pair(job, train, targets, config, context, output, backend_tag=pairs.PairBackend.tag):
    """Reconstruct evidence from committed stages without executing any model."""
    pairs.validate_job(job, train, targets)
    identity = pair_identity(job, train, targets, config, context, backend_tag)
    store = ReadonlyCheckpoints(output, identity)
    log = pairs.read_json(output / "02_training_membership.json")
    if (log["job_id"] != job["job_id"] or log["excluded_subjects"] != job["excluded_subjects"]
            or log["prediction_requests"] != job["prediction_requests"]
            or log["completed_stages"] != job["fit_stages"] or log["expected_stages"] != job["fit_stages"]):
        raise p.CheckpointError("The completed job membership log is incomplete.")
    stages = log["stages"]
    by_name = {row["stage"]:row for row in stages}
    if len(by_name) != len(stages) or len(stages) != job["fit_stages"]:
        raise p.CheckpointError("Unexpected or duplicate stages in completed job.")
    if {path.stem for path in store.directory.glob("*.npz")} != set(by_name):
        raise p.CheckpointError("Checkpoint data files do not match the stage inventory.")
    if {path.name.removesuffix(".done.json") for path in store.directory.glob("*.done.json")} != set(by_name):
        raise p.CheckpointError("Checkpoint commit markers do not match the stage inventory.")
    seen = []

    def load(name, operation, n):
        stage = by_name[name]
        if stage["operation"] != operation:
            raise p.CheckpointError(f"Actual fitting/selection slice changed: {name}")
        cached = store.load(name, operation, n)
        if cached is None:
            raise p.CheckpointError(f"Missing committed stage: {name}")
        values, epoch, seconds = cached
        if epoch != stage["selected_epoch"] or seconds != stage["fit_seconds"]:
            raise p.CheckpointError("Stage timing/epoch metadata differs from the checkpoint.")
        if operation["model"] == "always_nn":
            if not 1 <= epoch <= config["mlp"]["max_epochs"]:
                raise p.CheckpointError("MLP selected epoch is outside the frozen range.")
        elif epoch != 0:
            raise p.CheckpointError("A non-MLP checkpoint reported an epoch.")
        seen.append(name)
        return values, epoch

    a,b = job["excluded_subjects"]
    seed = int(job["seed"])
    inner = {name:np.full((len(train.features),2), np.nan) for name in p.BRANCHES}
    epochs = []
    coverage = np.zeros(len(train.features), dtype=np.int64)
    for index, fold in enumerate(p.splits.subject_loso_folds(train.subjects)):
        held = int(np.unique(train.subjects[fold.validation]).item())
        member = p.membership(train, fold.train, fold.validation, a,b)
        fold_seed = seed + 1009*(index+1)
        coverage[fold.validation] += 1
        inner["always_nn"][fold.validation], epoch = load(f"inner_s{held:02d}_mlp",
            {**member,"model":"always_nn","seed":fold_seed},len(fold.validation))
        epochs.append(epoch)
        for offset,name in enumerate(("rf","extra_trees"),1):
            inner[name][fold.validation],_ = load(f"inner_s{held:02d}_{name}",
                {**member,"model":name,"seed":fold_seed+offset*37},len(fold.validation))
    if not np.all(coverage == 1) or any(not np.isfinite(values).all() for values in inner.values()):
        raise p.CheckpointError("Inner predictions are not complete held-participant slices.")
    selected = max(1,int(np.median(epochs)))
    member = {**p.membership(train,np.arange(len(train.features)),None,a,b), "separate_prediction_targets":[
        {"subject":s,"windows":len(targets[s].features),"fingerprint":targets[s].fingerprint()} for s in (a,b)]}
    predictions = {a:{},b:{}}
    n_a = len(targets[a].features)
    total = n_a + len(targets[b].features)

    def assign(name,values):
        predictions[a][name] = values[:n_a].copy()
        predictions[b][name] = values[n_a:].copy()

    values, epoch = load("final_mlp",{**member,"model":"always_nn","seed":seed,"fixed_epochs":selected},total)
    if epoch != selected:
        raise p.CheckpointError("Final epoch differs from inner-fold selection.")
    assign("always_nn",values)
    for offset,name in enumerate(("rf","extra_trees"),1):
        values,_ = load("final_"+name,{**member,"model":name,"seed":seed+offset*37},total)
        assign(name,values)
    operation = {**member,"model":"always_fuse","seed":seed,
                 "inner_probability_sha256":{name:p.array_digest(values) for name,values in inner.items()},
                 "target_probability_sha256":{str(s):{name:p.array_digest(values) for name,values in pred.items()}
                                               for s,pred in predictions.items()}}
    values,_ = load("final_stacker",operation,total)
    assign("always_fuse",values)
    if seen != [row["stage"] for row in stages]:
        raise p.CheckpointError("The saved stage order changed.")
    result = pairs.PairResult(predictions,selected,epochs,stages,store.identity_digest)
    expected = pairs.window_frame(job,targets,result)
    actual = pd.read_csv(output / "03_label_free_meta_window_predictions.csv",float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(expected,actual,check_dtype=False,check_exact=True)
    except AssertionError as error:
        raise p.CheckpointError("Window output does not reproduce the aligned checkpoint predictions.") from error
    return result


def scoring_labels(targets):
    return {s:{str(r):int(int(str(r)[-1])>=2) for r in np.unique(rows.recordings)} for s,rows in targets.items()}


def audit_complete_pair(job, loader, config, context, output, backend_tag=pairs.PairBackend.tag):
    summary = pairs.read_json(output / "07_pair_summary.json")
    if summary.get("status") not in ("completed_first_pair_validation_not_full_gate", "completed_full_schedule_meta_pair"):
        raise p.CheckpointError("Unexpected completed-pair status.")
    required = {"01_run_identity.json","02_training_membership.json","03_label_free_meta_window_predictions.csv",
                "05_meta_recording_predictions.csv","06_meta_rows.json"}
    if not required.issubset(summary["outputs_sha256"]):
        raise p.CheckpointError("Completed-pair checksum manifest is incomplete.")
    for name,digest in summary["outputs_sha256"].items():
        if Path(name).name != name or p.file_digest(output/name) != digest:
            raise p.CheckpointError("A completed-pair output changed; it will not be overwritten.")
    train, targets = loader(job)
    result = read_fitted_pair(job,train,targets,config,context,output,backend_tag)
    expected, meta = pairs.score_pair(job,train,targets,result,scoring_labels(targets))
    actual = pd.read_csv(output / "05_meta_recording_predictions.csv",float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(expected,actual,check_dtype=False,check_exact=True)
    except AssertionError as error:
        raise p.CheckpointError("Completed recording output does not reproduce its checkpoints.") from error
    if meta != pairs.read_json(output / "06_meta_rows.json"):
        raise p.CheckpointError("Completed meta diagnostics do not reproduce their checkpoints.")
    expected_fields = {"job_id":job["job_id"],"pair_identity_sha256":result.identity_sha256,
        "completed_fit_stages":job["fit_stages"],"selected_epoch":result.selected_epoch,
        "inner_selected_epochs":result.inner_epochs,"recording_rows":32,"meta_rows":2,
        "window_rows":sum(len(t.features) for t in targets.values())*4,
        "completed_fit_seconds":sum(s["fit_seconds"] for s in result.stages),"full_gate_fitted":False,
        "protected_inputs_unchanged":True}
    if any(summary.get(key) != value for key,value in expected_fields.items()):
        raise p.CheckpointError("Completed-pair summary disagrees with its evidence.")
    return {"job":job,"output":output,"result":result,"records":actual,"meta":meta,"summary":summary}


def adopt_first(job, loader, config, runtime, first_output, pilot_root, atol):
    if job["job_id"] != pairs.FIRST_JOB or p.file_digest(first_output / "07_pair_summary.json") != FIRST_SUMMARY_SHA:
        raise p.CheckpointError("The previously verified first paired result changed.")
    saved = pairs.read_json(first_output / "01_run_identity.json")
    old_context = saved["identity"]["context"]
    if old_context["runtime"] != runtime or old_context["design_sha256"] != pairs.DESIGN_SHA or old_context["schedule_sha256"] != pairs.JOBS_SHA:
        raise p.CheckpointError("First paired result uses a different frozen design or runtime.")
    pairs.verify_hashes(old_context["source_input_hashes"])
    evidence = audit_complete_pair(job,loader,config,old_context,first_output)
    _,targets = loader(job)
    parity = pairs.compare_pilot(job,evidence["result"],targets,pilot_root,first_output,atol=atol)
    if parity != pairs.read_json(first_output / "04_pilot_equivalence.json") or parity["status"] != "passed":
        raise p.CheckpointError("The first pair's pilot parity proof failed.")
    return evidence


def batch_identity_file(root, identity, *, create):
    digest = hashlib.sha256(p.json_bytes(identity)).hexdigest()
    path = root / "00_execution_identity.json"
    if path.exists():
        saved = pairs.read_json(path)
        if saved.get("identity") != identity or saved.get("identity_sha256") != digest:
            raise p.CheckpointError("Batch source, data or runtime changed; existing jobs are preserved without restart.")
    else:
        contents = list(root.iterdir()) if root.exists() else []
        if any(pth.name != ".run.lock" for pth in contents):
            raise p.CheckpointError("Batch files exist without an execution identity; no automatic reuse or overwrite.")
        if create:
            p.atomic_json(path,{"identity":identity,"identity_sha256":digest,"created_utc":datetime.now(timezone.utc).isoformat()})
    return {"batch_identity_sha256":digest,"schema":SCHEMA}


def complete_new_pair(job, loader, config, device, output, context, hashes, *, emit, backend=None):
    train,targets = loader(job)
    result = pairs.run_pair(job,train,targets,config=config,device=device,output=output,context=context,backend=backend,emit=emit)
    tag = backend.tag if backend is not None else pairs.PairBackend.tag
    checked = read_fitted_pair(job,train,targets,config,context,output,tag)
    recordings,meta = pairs.score_pair(job,train,targets,checked,scoring_labels(targets))
    pairs.verify_hashes(hashes)
    p.atomic_bytes(output / "05_meta_recording_predictions.csv",recordings.to_csv(index=False,lineterminator="\n").encode("utf-8"))
    p.atomic_json(output / "06_meta_rows.json",meta)
    names = ["01_run_identity.json","02_training_membership.json","03_label_free_meta_window_predictions.csv",
             "05_meta_recording_predictions.csv","06_meta_rows.json"]
    summary = {"status":"completed_full_schedule_meta_pair","job_id":job["job_id"],
        "pair_identity_sha256":result.identity_sha256,"completed_fit_stages":len(result.stages),
        "reused_fit_stages":sum(s["reused"] for s in result.stages),
        "completed_fit_seconds":sum(s["fit_seconds"] for s in result.stages),
        "selected_epoch":result.selected_epoch,"inner_selected_epochs":result.inner_epochs,
        "recording_rows":len(recordings),"meta_rows":len(meta),"window_rows":sum(len(t.features) for t in targets.values())*4,
        "protected_inputs_unchanged":True,"full_gate_fitted":False,
        "outputs_sha256":{name:p.file_digest(output/name) for name in names}}
    p.atomic_json(output / "07_pair_summary.json",summary)
    return {"job":job,"output":output,"result":result,"records":recordings,"meta":meta,"summary":summary}


def compact_evidence(evidence):
    return {key:value for key,value in evidence.items() if key != "result"}


def scan_completed(schedule, loader, config, context, root, first, *, backend_tag=pairs.PairBackend.tag, emit=None):
    completed = {first["job"]["job_id"]:compact_evidence(first)}
    if pairs.job_directory(root/"jobs",first["job"]).exists():
        raise p.CheckpointError("The adopted first pair must not also have a new training directory.")
    expected_directories = {pairs.job_directory(root/"jobs",job) for job in schedule[1:]}
    for path in (root/"jobs").glob("*/*"):
        if path not in expected_directories:
            raise p.CheckpointError("Unexpected pair directory outside the frozen schedule.")
    for job in schedule[1:]:
        output = pairs.job_directory(root/"jobs",job)
        if (output / "07_pair_summary.json").exists():
            completed[job["job_id"]] = compact_evidence(audit_complete_pair(job,loader,config,context,output,backend_tag))
            if emit and len(completed) % 25 == 0:
                emit(f"[verify] {len(completed)}/{len(schedule)} completed pairs checked without fitting")
    return completed


def progress(root,schedule,completed,status,*,current=None,error=None):
    by_task = {task:sum(row["job"]["task"] == task for row in completed.values()) for task in ("arithmetic","stroop")}
    value = {"status":status,"planned_jobs":len(schedule),"completed_jobs":len(completed),
             "remaining_jobs":len(schedule)-len(completed),"completed_by_task":by_task,
             "current_job":current,"error":error,"updated_utc":datetime.now(timezone.utc).isoformat(),
             "full_gate_fitted":False,"current_main_predictions_unchanged":True,
             "completed_job_ids":[job["job_id"] for job in schedule if job["job_id"] in completed]}
    p.atomic_json(root / "01_progress.json",value)
    return value


def aggregate_full(schedule,completed,protocol,root):
    if set(completed) != {job["job_id"] for job in schedule}:
        raise p.CheckpointError("Cannot publish full meta outputs from an incomplete schedule.")
    records = pd.concat([completed[job["job_id"]]["records"] for job in schedule],ignore_index=True)
    meta_rows = []
    for job in schedule:
        for row in completed[job["job_id"]]["meta"]:
            meta_rows.append({"task":job["task"],"outer_subject":row["outer_subject"],"meta_subject":row["meta_subject"],
                "seed":job["seed"],"benefit":row["ordinary_recording_gain"],
                "loss_event_le_minus_0_05":row["loss_event_le_minus_0_05"],**row["label_free_diagnostics"]})
    meta = pd.DataFrame(meta_rows)
    keys = ["task","outer_subject","meta_subject"]
    if meta.duplicated(keys+["seed"]).any() or records.duplicated(["direction","outer_subject","meta_subject","seed","recording_id","method"]).any():
        raise p.CheckpointError("Duplicate meta-evidence keys in the final aggregate.")
    expected = {(task,t,m,seed) for task,cohort in protocol["inputs"]["cohorts"].items()
                for t in cohort for m in cohort if t != m for seed in protocol["inputs"]["seeds"]}
    if set(meta[keys+["seed"]].itertuples(index=False,name=None)) != expected:
        raise p.CheckpointError("The full seed/participant meta-evidence grid is incomplete.")
    if len(records) != protocol["loso"]["expected_meta_recording_predictions_all_four_branches"]:
        raise p.CheckpointError("Unexpected total meta-recording prediction count.")
    expected_methods = set(p.METHODS)
    for _,group in records.groupby(["direction","outer_subject","meta_subject","seed"]):
        if len(group) != 16 or set(group.method) != expected_methods or group.groupby("method").recording_id.nunique().ne(4).any():
            raise p.CheckpointError("Each meta seed row must have all four branches on four recordings.")
    columns = list(p.risk_gate.GATE_FEATURES)+["benefit"]
    median = meta.groupby(keys,as_index=False)[columns].median()
    if len(meta) != protocol["loso"]["expected_all_seed_meta_rows"] or len(median) != protocol["loso"]["expected_median_seed_meta_rows"]:
        raise p.CheckpointError("Unexpected per-seed or median-seed meta row count.")
    output_names = []
    for name,frame in (("02_all_meta_recording_seed.csv",records),("03_all_meta_seed.csv",meta),("04_all_meta_median_seed.csv",median)):
        p.atomic_bytes(root/name,frame.to_csv(index=False,lineterminator="\n").encode("utf-8"))
        output_names.append(name)
    result = {"status":"completed_full_loso_meta_predictions_gates_and_statistics_pending",
              "completed_jobs":len(completed),"adopted_prior_jobs":1,"meta_recording_rows":len(records),
              "meta_seed_rows":len(meta),"meta_median_seed_rows":len(median),"full_gate_fitted":False,
              "downstream_outputs":["DASF/CB-SF target predictions and cross-task variant","14 paired comparisons and gate stability summaries"],
              "outputs_sha256":{name:p.file_digest(root/name) for name in output_names},
              "source_pair_summaries":{job["job_id"]:{"path":str(completed[job["job_id"]]["output"] / "07_pair_summary.json"),
                                                       "sha256":p.file_digest(completed[job["job_id"]]["output"] / "07_pair_summary.json")}
                                       for job in schedule}}
    p.atomic_json(root / "05_full_meta_summary.json",result)
    return result


def run_schedule(schedule,loader,protocol,device,root,identity,hashes,first,*,task="all",max_jobs=None,emit=print,backend_factory=None):
    if task not in ("all","arithmetic","stroop") or (max_jobs is not None and max_jobs < 1):
        raise ValueError("Invalid task selection or job limit.")
    if not schedule or schedule[0]["job_id"] != first["job"]["job_id"] or len({j["job_id"] for j in schedule}) != len(schedule):
        raise ValueError("The canonical unique job order and adopted first job are required.")
    with p.run_lock(root):
        pairs.verify_hashes(hashes)
        context = batch_identity_file(root,identity,create=True)
        config = protocol["model_config"]
        tag = backend_factory().tag if backend_factory else pairs.PairBackend.tag
        completed = {first["job"]["job_id"]:compact_evidence(first)}
        advanced = 0
        current = None
        try:
            completed = scan_completed(schedule,loader,config,context,root,first,backend_tag=tag,emit=emit)
            progress(root,schedule,completed,"running")
            for index,job in enumerate(schedule):
                current = job["job_id"]
                if current in completed:
                    emit(f"[job {index+1}/{len(schedule)}] verified complete: {current}")
                    continue
                if task != "all" and job["task"] != task:
                    continue
                if max_jobs is not None and advanced >= max_jobs:
                    break
                progress(root,schedule,completed,"running",current=current)
                pairs.verify_hashes(hashes)
                emit(f"[job {index+1}/{len(schedule)}] start/resume: {current}")
                evidence = complete_new_pair(job,loader,config,device,pairs.job_directory(root/"jobs",job),context,hashes,
                    emit=emit,backend=backend_factory() if backend_factory else None)
                completed[current] = compact_evidence(evidence)
                advanced += 1
                progress(root,schedule,completed,"running")
                emit(f"[done {len(completed)}/{len(schedule)}] {current}")
            pairs.verify_hashes(hashes)
            if len(completed) == len(schedule):
                aggregate_full(schedule,completed,protocol,root)
                state = progress(root,schedule,completed,"completed_full_meta_predictions")
            else:
                state = progress(root,schedule,completed,"paused_after_requested_scope")
            return {**state,"jobs_advanced_this_invocation":advanced}
        except BaseException as error:
            progress(root,schedule,completed,"interrupted" if isinstance(error,KeyboardInterrupt) else "stopped_on_error",
                     current=current,error=str(error))
            raise
