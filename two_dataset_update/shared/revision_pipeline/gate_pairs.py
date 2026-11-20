"""Checkpointed paired LOSO meta predictions; no target labels in fitting APIs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import gate_pilot as pilot


SCHEMA = "paired-exclusion-meta-predictions-v1"
FIRST_JOB = "arithmetic_exclude_s01_s02_seed1335"
DESIGN_SHA = "76e75284af130a3ac246caef0994b42bce61cb19f0a3926aa64b2c80a74edf1f"
JOBS_SHA = "02e4060ac630d048dd0a23a5cc18696f381c5160965151eee4884e7ad2379272"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(hashes):
    for name, digest in hashes.items():
        if pilot.file_digest(Path(name)) != digest:
            raise pilot.CheckpointError(f"Protected input or source changed: {name}")


def load_design(folder):
    protocol_path = folder / "nested_gate_protocol.json"
    jobs_path = folder / "paired_training_plan.json"
    if pilot.file_digest(protocol_path) != DESIGN_SHA or pilot.file_digest(jobs_path) != JOBS_SHA:
        raise pilot.CheckpointError("The frozen 174 design or paired-job schedule changed.")
    manifest = read_json(folder / "design_manifest.json")
    if manifest["status"] != "frozen_design_only_no_training":
        raise pilot.CheckpointError("Unexpected design freeze status.")
    for name, digest in manifest["artifact_sha256"].items():
        if Path(name).name != name or pilot.file_digest(folder / name) != digest:
            raise pilot.CheckpointError("A frozen protocol artifact changed.")
    sums = {**manifest["artifact_sha256"], "design_manifest.json": pilot.file_digest(folder / "design_manifest.json")}
    expected = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    if (folder / "SHA256SUMS.txt").read_text(encoding="utf-8") != expected:
        raise pilot.CheckpointError("The frozen checksum list changed.")
    protected = read_json(folder / "source_fingerprints.json")
    verify_hashes(protected)
    protocol, schedule = read_json(protocol_path), read_json(jobs_path)
    if len(schedule) != 915 or len({row["job_id"] for row in schedule}) != 915:
        raise pilot.CheckpointError("The frozen job schedule is incomplete.")
    return protocol, schedule, protected


def select_jobs(schedule, requested_ids, approved_ids):
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Select a nonempty, unique batch of jobs.")
    if set(requested_ids) - set(approved_ids):
        raise PermissionError("This batch has not been approved. Only the first paired validation is currently open.")
    mapping = {row["job_id"]: row for row in schedule}
    if set(requested_ids) - set(mapping):
        raise ValueError("An unknown job was requested.")
    return [row for row in schedule if row["job_id"] in requested_ids]


def validate_job(job, train, targets):
    pair = job["excluded_subjects"]
    if len(pair) != 2 or pair != sorted(set(pair)) or set(targets) != set(pair):
        raise ValueError("Exactly two distinct, sorted excluded participants are required.")
    a, b = pair
    pilot.validate_unit(train, targets[b], a, b)
    pilot.validate_unit(train, targets[a], b, a)
    actual = sorted(int(value) for value in np.unique(train.subjects))
    if actual != job["training_subjects"] or actual != job["inner_validation_subject_order"]:
        raise ValueError("Actual training participants do not match the job.")
    if len(np.unique(train.recordings)) != job["training_recordings"] or job["inner_folds"] != len(actual):
        raise ValueError("Training-recording or fold count mismatch.")
    for rows in (train, *targets.values()):
        if not all(str(value).startswith(job["task"] + ":") for value in rows.recordings):
            raise ValueError("A different task entered the paired job.")
    expected_roles = [{"outer_target": a, "meta_validation": b}, {"outer_target": b, "meta_validation": a}]
    if job["prediction_requests"] != expected_roles:
        raise ValueError("The paired outer/meta roles are inconsistent.")
    expected_id = f"{job['task']}_exclude_s{a:02d}_s{b:02d}_seed{job['seed']}"
    if job["job_id"] != expected_id:
        raise ValueError("Job identifier does not match its participants or seed.")
    if job["fit_stages"] != 3 * len(actual) + 4 or job["expected_recording_prediction_rows"] != 32:
        raise ValueError("Unexpected paired-job stage or prediction counts.")


class PairBackend(pilot.CoreBackend):
    tag = "frozen-models-separate-target-prediction-calls-v1"

    def final_mlp(self, train, targets, config, seed, device, epoch):
        fitted, selected = pilot.models.fit_mlp(train.features, train.labels, config, seed, device, fixed_epochs=epoch)
        return {subject: fitted.predict_proba(rows.features) for subject, rows in sorted(targets.items())}, selected

    def final_forest(self, name, train, targets, config, seed):
        fitted = pilot.models._make_forest(name, config, seed)
        fitted.fit(train.features, train.labels)
        return {subject: fitted.predict_proba(rows.features) for subject, rows in sorted(targets.items())}, 0

    def final_stack(self, train, inner, predictions, config, seed):
        _, inverse, counts = np.unique(train.recordings, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inverse]
        weight /= weight.mean()
        fitted = LogisticRegression(C=float(config["C"]), class_weight="balanced",
                                    max_iter=int(config["max_iter"]), random_state=int(seed))
        fitted.fit(pilot.models._stack_features(*(inner[name] for name in pilot.BRANCHES)),
                   train.labels, sample_weight=weight)
        return {subject: fitted.predict_proba(pilot.models._stack_features(*(values[name] for name in pilot.BRANCHES)))
                for subject, values in sorted(predictions.items())}, 0


@dataclass
class PairResult:
    probabilities: dict[int, dict[str, np.ndarray]]
    selected_epoch: int
    inner_epochs: list[int]
    stages: list[dict]
    identity_sha256: str


def run_pair(job, train, targets, *, config, device, output, context, backend=None, emit=print):
    validate_job(job, train, targets)
    backend = backend or PairBackend()
    a, b = job["excluded_subjects"]
    pair = [a, b]
    seed = int(job["seed"])
    identity = {"schema": SCHEMA, "job": job, "config": config, "backend": backend.tag,
                "training_rows": train.fingerprint(),
                "label_free_targets": {str(subject): targets[subject].fingerprint() for subject in pair},
                "context": context}
    stages = []
    target_sizes = {subject: len(targets[subject].features) for subject in pair}
    total_target_rows = sum(target_sizes.values())
    target_operation = {"separate_prediction_targets": [
        {"subject": subject, "windows": target_sizes[subject], "fingerprint": targets[subject].fingerprint()}
        for subject in pair]}
    with pilot.run_lock(output):
        store = pilot.CheckpointStore(output, identity)
        folds = pilot.splits.subject_loso_folds(train.subjects)

        def stage(name, operation, n, action):
            cached = store.load(name, operation, n)
            reused = cached is not None
            emit(f"[{len(stages)+1}/{job['fit_stages']}] {'reuse' if reused else 'fit'} {name}")
            if reused:
                probability, epoch, seconds = cached
            else:
                started = time.monotonic()
                probability, epoch = action()
                probability = np.asarray(probability, dtype=np.float64)
                seconds = time.monotonic() - started
                pilot.validate_probability(probability, n)
                epoch = int(epoch)
            if operation["model"] == "always_nn":
                if not 1 <= epoch <= int(config["mlp"]["max_epochs"]):
                    raise pilot.CheckpointError("Invalid selected epoch; stage not accepted.")
                if "fixed_epochs" in operation and epoch != operation["fixed_epochs"]:
                    raise pilot.CheckpointError("Final epoch does not match inner selection.")
            elif epoch != 0:
                raise pilot.CheckpointError("A non-MLP stage reported an epoch.")
            if not reused:
                store.save(name, operation, probability, epoch, seconds)
                emit(f"    saved ({seconds:.1f}s, epoch={epoch})")
            stages.append({"stage": name, "operation": operation, "reused": reused,
                           "selected_epoch": epoch, "fit_seconds": seconds})
            pilot.atomic_json(output / "02_training_membership.json", {
                "job_id": job["job_id"], "excluded_subjects": pair,
                "prediction_requests": job["prediction_requests"], "expected_stages": job["fit_stages"],
                "completed_stages": len(stages), "stages": stages})
            return probability, epoch

        inner = {name: np.full((len(train.features), 2), np.nan) for name in pilot.BRANCHES}
        epochs = []
        for index, fold in enumerate(folds):
            held = int(np.unique(train.subjects[fold.validation]).item())
            member = pilot.membership(train, fold.train, fold.validation, a, b)
            fold_seed = seed + 1009 * (index + 1)
            inner["always_nn"][fold.validation], epoch = stage(
                f"inner_s{held:02d}_mlp", {**member, "model": "always_nn", "seed": fold_seed}, len(fold.validation),
                lambda: backend.mlp(train.features[fold.train], train.labels[fold.train], train.recordings[fold.train],
                                    train.features[fold.validation], train.labels[fold.validation], train.recordings[fold.validation],
                                    train.features[fold.validation], config["mlp"], fold_seed, device, None))
            epochs.append(epoch)
            for offset, name in enumerate(("rf", "extra_trees"), 1):
                forest_seed = fold_seed + offset * 37
                inner[name][fold.validation], _ = stage(
                    f"inner_s{held:02d}_{name}", {**member, "model": name, "seed": forest_seed}, len(fold.validation),
                    lambda: backend.forest(name, train.features[fold.train], train.labels[fold.train],
                                           train.features[fold.validation], config[name], forest_seed))
        if any(not np.isfinite(values).all() for values in inner.values()):
            raise pilot.CheckpointError("Inner validation predictions are incomplete.")
        selected_epoch = max(1, int(np.median(epochs)))
        member = {**pilot.membership(train, np.arange(len(train.features)), None, a, b), **target_operation}
        predictions = {subject: {} for subject in pair}

        def pack_fit(action):
            by_target, epoch = action()
            if set(by_target) != set(pair):
                raise pilot.CheckpointError("A final model did not predict both held-out participants.")
            for subject in pair:
                pilot.validate_probability(np.asarray(by_target[subject]), target_sizes[subject])
            # Concatenation is checkpoint storage only; models predicted each person separately.
            return np.concatenate([by_target[subject] for subject in pair]), epoch

        def assign(name, values):
            start = 0
            for subject in pair:
                predictions[subject][name] = values[start:start + target_sizes[subject]].copy()
                start += target_sizes[subject]

        values, _ = stage("final_mlp", {**member, "model": "always_nn", "seed": seed, "fixed_epochs": selected_epoch},
                          total_target_rows, lambda: pack_fit(lambda: backend.final_mlp(train, targets, config["mlp"], seed, device, selected_epoch)))
        assign("always_nn", values)
        for offset, name in enumerate(("rf", "extra_trees"), 1):
            forest_seed = seed + offset * 37
            values, _ = stage(f"final_{name}", {**member, "model": name, "seed": forest_seed}, total_target_rows,
                              lambda: pack_fit(lambda: backend.final_forest(name, train, targets, config[name], forest_seed)))
            assign(name, values)
        operation = {**member, "model": "always_fuse", "seed": seed,
                     "inner_probability_sha256": {name: pilot.array_digest(values) for name, values in inner.items()},
                     "target_probability_sha256": {str(subject): {name: pilot.array_digest(values) for name, values in probabilities.items()}
                                                   for subject, probabilities in predictions.items()}}
        values, _ = stage("final_stacker", operation, total_target_rows,
                          lambda: pack_fit(lambda: backend.final_stack(train, inner, predictions, config["stacker"], seed)))
        assign("always_fuse", values)
        result = PairResult(predictions, selected_epoch, epochs, stages, store.identity_digest)
        window = window_frame(job, targets, result)
        pilot.atomic_bytes(output / "03_label_free_meta_window_predictions.csv", window.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        return result


def window_frame(job, targets, result):
    frames = []
    for role in job["prediction_requests"]:
        subject = role["meta_validation"]
        target = targets[subject]
        for method, values in result.probabilities[subject].items():
            frames.append(pd.DataFrame({"task": job["task"], "outer_subject": role["outer_target"],
                          "meta_subject": subject, "seed": job["seed"], "method": method,
                          "recording_id": target.recordings, "window_id": target.windows,
                          "p0": values[:, 0], "p1": values[:, 1]}))
    frame = pd.concat(frames, ignore_index=True)
    keys = ["task", "outer_subject", "meta_subject", "seed", "method", "window_id"]
    expected = sum(len(rows.features) for rows in targets.values()) * 4
    if len(frame) != expected or frame.duplicated(keys).any():
        raise pilot.CheckpointError("Paired window output is incomplete or contains duplicate keys.")
    return frame


def compare_pilot(job, result, targets, pilot_root, pair_output, *, atol):
    if job["job_id"] != FIRST_JOB:
        raise ValueError("The 171 parity check is restricted to the original pilot pair and seed.")
    old = read_json(pilot_root / "05_pilot_summary.json")
    old_identity = read_json(pilot_root / "01_run_identity.json")
    current_identity = read_json(pair_output / "01_run_identity.json")
    for field in ("config", "training_rows"):
        if old_identity["identity"][field] != current_identity["identity"][field]:
            raise pilot.CheckpointError(f"Pilot fitting inputs differ: {field}")
    if old_identity["identity"]["target_rows_without_labels"] != targets[2].fingerprint():
        raise pilot.CheckpointError("S02 pilot target inputs changed.")
    issues = []
    if old["selected_epoch"] != result.selected_epoch or old["inner_selected_epochs"] != result.inner_epochs:
        issues.append("Selected epochs differ from the pilot.")
    old_stages = {row["stage"]: row for row in read_json(pilot_root / "02_training_membership.json")["stages"]}
    rows = []
    for stage in result.stages:
        name = stage["stage"]
        if name not in old_stages:
            raise pilot.CheckpointError("A paired model stage has no matching pilot stage.")
        old_operation = old_stages[name]["operation"]
        for key, value in old_operation.items():
            if key not in ("inner_probability_sha256", "target_probability_sha256") and stage["operation"].get(key) != value:
                issues.append(f"Training/selection membership differs: {name}/{key}")
        with np.load(pilot_root / "checkpoints" / (name + ".npz"), allow_pickle=False) as archive:
            reference = archive["probability"]
        if name.startswith("inner_"):
            with np.load(pair_output / "checkpoints" / (name + ".npz"), allow_pickle=False) as archive:
                observed = archive["probability"]
        else:
            observed = result.probabilities[2][stage["operation"]["model"]]
        pilot.validate_probability(reference, len(observed))
        difference = float(np.max(np.abs(observed - reference)))
        matches = bool(np.allclose(observed, reference, atol=atol, rtol=0))
        rows.append({"stage": name, "max_absolute_probability_difference": difference, "passed": matches})
        if not matches:
            issues.append(f"Probability difference exceeds frozen tolerance: {name}")
    old_windows = pd.read_csv(pilot_root / "03_label_free_meta_window_predictions.csv", float_precision="round_trip")
    old_recordings = pd.read_csv(pilot_root / "04_meta_recording_predictions.csv", float_precision="round_trip")
    for name in pilot.METHODS:
        window_rows = old_windows[old_windows.method.eq(name)]
        if not np.array_equal(window_rows.window_id.to_numpy(), targets[2].windows):
            raise pilot.CheckpointError("Pilot window order differs from the paired target order.")
        for recording in np.unique(targets[2].recordings):
            values = result.probabilities[2][name][targets[2].recordings == recording]
            geometric = np.exp(np.mean(np.log(np.clip(values, 1e-12, 1.0)), axis=0))
            reference = old_recordings[old_recordings.method.eq(name) & old_recordings.recording_id.eq(recording)]
            if len(reference) != 1 or int(np.argmax(geometric)) != int(reference.pred_label.iloc[0]):
                issues.append(f"Recording prediction differs: {name}/{recording}")
    return {"status": "passed" if not issues else "failed", "job_id": job["job_id"],
            "pair_identity_sha256": result.identity_sha256, "probability_atol": atol, "probability_rtol": 0,
            "checked_stages": len(rows), "stage_probability_checks": rows,
            "max_absolute_probability_difference": max(row["max_absolute_probability_difference"] for row in rows),
            "selected_epochs_match": old["selected_epoch"] == result.selected_epoch and old["inner_selected_epochs"] == result.inner_epochs,
            "recording_labels_required_exact": True, "issues": issues,
            "criterion": "Numerical equivalence and isolation, never improved accuracy.",
            "scope": "S02 parity against171; both-role training isolation is checked for the complete paired job. Not a full gate or full-cohort result."}


def score_pair(job, train, targets, result, labels):
    if set(labels) != set(targets):
        raise ValueError("Scoring labels must be supplied separately for each held-out participant.")
    frames, summaries = [], []
    for role in job["prediction_requests"]:
        subject = role["meta_validation"]
        one = pilot.PilotResult(result.probabilities[subject], result.selected_epoch, result.inner_epochs, result.stages)
        recordings, summary = pilot.score_predictions(one, train, targets[subject], labels[subject],
            outer=role["outer_target"], meta=subject, seed=job["seed"], task=job["task"])
        recordings["protocol"] = "loso_meta_pair"
        recordings["outer_subject"] = role["outer_target"]
        recordings["meta_subject"] = subject
        recordings["seed"] = job["seed"]
        frames.append(recordings)
        summaries.append({"outer_subject": role["outer_target"], "meta_subject": subject,
                          "ordinary_recording_gain": summary["ordinary_recording_gain"],
                          "loss_event_le_minus_0_05": summary["loss_event_le_minus_0_05"],
                          "label_free_diagnostics": summary["label_free_diagnostics"],
                          "accuracy_for_engineering_inspection_only": summary["accuracy_for_engineering_inspection_only"]})
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != job["expected_recording_prediction_rows"] or frame.duplicated(["outer_subject", "meta_subject", "method", "recording_id"]).any():
        raise pilot.CheckpointError("Incomplete paired recording predictions.")
    return frame, summaries


def job_directory(root, job):
    a, b = job["excluded_subjects"]
    if job["task"] not in ("arithmetic", "stroop") or not (0 < a < b < 100):
        raise ValueError("Unsafe paired job output name.")
    return root / job["task"] / f"p{a:02d}_{b:02d}_s{int(job['seed'])}"


def execute_validation_batch(jobs, train_loader, *, config, device, root, context, hashes, pilot_root, atol, emit=print):
    if [job["job_id"] for job in jobs] != [FIRST_JOB]:
        raise PermissionError("This validation entry point accepts only the predefined first paired job; use the batch runner for full execution.")
    with pilot.run_lock(root):
        job = jobs[0]
        train, targets = train_loader(job)
        verify_hashes(hashes)
        output = job_directory(root, job)
        result = run_pair(job, train, targets, config=config, device=device, output=output, context=context, emit=emit)
        parity = compare_pilot(job, result, targets, pilot_root, output, atol=atol)
        pilot.atomic_json(output / "04_pilot_equivalence.json", parity)
        if parity["status"] != "passed":
            raise pilot.CheckpointError("Paired/pilot equivalence failed. Evidence is preserved; no further job or score was run.")
        # Each person's labels enter only downstream of both prediction calls and the parity check.
        labels = {subject: {str(recording): int(int(str(recording)[-1]) >= 2) for recording in np.unique(rows.recordings)}
                  for subject, rows in targets.items()}
        recordings, meta = score_pair(job, train, targets, result, labels)
        verify_hashes(hashes)
        pilot.atomic_bytes(output / "05_meta_recording_predictions.csv", recordings.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        pilot.atomic_json(output / "06_meta_rows.json", meta)
        names = ["01_run_identity.json", "02_training_membership.json", "03_label_free_meta_window_predictions.csv",
                 "04_pilot_equivalence.json", "05_meta_recording_predictions.csv", "06_meta_rows.json"]
        summary = {"status": "completed_first_pair_validation_not_full_gate", "job_id": job["job_id"],
                   "pair_identity_sha256": result.identity_sha256,
                   "completed_fit_stages": len(result.stages), "reused_fit_stages": sum(s["reused"] for s in result.stages),
                   "completed_fit_seconds": sum(s["fit_seconds"] for s in result.stages),
                   "meta_rows": len(meta), "recording_rows": len(recordings),
                   "window_rows": sum(len(rows.features) for rows in targets.values()) * 4,
                   "selected_epoch": result.selected_epoch, "inner_selected_epochs": result.inner_epochs,
                   "pilot_equivalence": "passed", "max_pilot_probability_difference": parity["max_absolute_probability_difference"],
                   "protected_inputs_unchanged": True, "full_gate_fitted": False, "remaining_jobs_started": 0,
                   "outputs_sha256": {name: pilot.file_digest(output / name) for name in names}}
        pilot.atomic_json(output / "07_pair_summary.json", summary)
        emit("First paired validation completed; no other job was started.")
        return summary, output
