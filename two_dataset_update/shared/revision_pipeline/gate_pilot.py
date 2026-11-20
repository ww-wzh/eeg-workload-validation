"""Resumable, target-label-free fitting for one doubly excluded LOSO unit."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import torch

from . import aggregation, models, risk_gate, splits


BRANCHES = ("always_nn", "rf", "extra_trees")
METHODS = (*BRANCHES, "always_fuse")
SCHEMA = "double-exclusion-gate-pilot-v1"


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    values = np.asarray(values)
    digest = hashlib.sha256(json_bytes({"shape": list(values.shape), "dtype": str(values.dtype)}))
    if values.dtype.kind in "OU":
        digest.update(json_bytes(values.tolist()))
    else:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, json_bytes(value))


@contextmanager
def run_lock(root: Path):
    """OS-owned lock releases on interruption; a leftover file is harmless."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This pilot output is already locked by another process.") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class PredictionRows:
    features: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray
    windows: np.ndarray

    def fingerprint(self):
        return {name: array_digest(getattr(self, name)) for name in ("features", "subjects", "recordings", "windows")}


@dataclass(frozen=True)
class TrainingRows(PredictionRows):
    labels: np.ndarray

    def fingerprint(self):
        return {**super().fingerprint(), "labels": array_digest(self.labels)}


def validate_rows(rows: PredictionRows) -> None:
    n = len(rows.features)
    if n == 0 or rows.features.ndim != 2 or not np.isfinite(rows.features).all():
        raise ValueError("Features must be a nonempty finite matrix.")
    if any(np.asarray(getattr(rows, field)).shape != (n,) for field in ("subjects", "recordings", "windows")):
        raise ValueError("Row metadata has inconsistent lengths.")
    if len(set(rows.windows.tolist())) != n:
        raise ValueError("Window identifiers are not unique.")
    for subject, recording, window in zip(rows.subjects, rows.recordings, rows.windows):
        match = re.fullmatch(r"(arithmetic|stroop):s(\d+):r([0-3])", str(recording))
        if match is None or int(match.group(2)) != int(subject) or not str(window).startswith(str(recording) + ":w"):
            raise ValueError("Recording/window metadata disagrees with participant membership.")
    if isinstance(rows, TrainingRows):
        if rows.labels.shape != (n,) or set(np.unique(rows.labels)) != {0, 1}:
            raise ValueError("Training labels must contain both binary classes.")
        for recording in np.unique(rows.recordings):
            if len(np.unique(rows.labels[rows.recordings == recording])) != 1:
                raise ValueError("A training recording contains multiple labels.")


def validate_unit(train: TrainingRows, target: PredictionRows, outer: int, meta: int) -> None:
    validate_rows(train)
    validate_rows(target)
    if outer == meta or {outer, meta} & set(train.subjects.tolist()):
        raise ValueError("Outer or meta-validation participant entered training.")
    if isinstance(target, TrainingRows):
        raise ValueError("The fitting API accepts label-free target rows only.")
    if set(target.subjects.tolist()) != {meta} or train.features.shape[1] != target.features.shape[1]:
        raise ValueError("Target participant or feature dimensions are inconsistent.")
    if set(train.recordings) & set(target.recordings) or set(train.windows) & set(target.windows):
        raise ValueError("Training and target rows overlap.")
    if len(np.unique(target.recordings)) != 4:
        raise ValueError("A pilot meta-validation participant must have four complete recordings.")
    if len(np.unique(train.subjects)) < 3:
        raise ValueError("At least three retained participants are needed for inner LOSO.")


def load_cached_rows(cache: Path, task: str, subjects: list[int], pinned: dict[str, str], *, training: bool):
    features, subject_ids, recordings, windows, labels = [], [], [], [], []
    for subject in subjects:
        for level in range(4):
            path = cache / f"{task}_s{subject:02d}_r{level}.npz"
            if str(path) not in pinned or file_digest(path) != pinned[str(path)]:
                raise ValueError(f"Missing or changed verified feature cache: {path}")
            with np.load(path, allow_pickle=False) as archive:
                matrix = archive["features"].astype(np.float32)
                metadata = json.loads(str(archive["metadata_json"].item()))
                variant = metadata["variant"]
                if matrix.ndim != 2 or matrix.shape[1] != 272 or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid feature matrix: {path}")
                if (metadata["sfreq"], metadata["window_seconds"], metadata["stride_seconds"]) != (250, 8.5, 0.5):
                    raise ValueError("Cache window protocol changed.")
                if variant["normalization"] != "none" or any(variant[key] != 1.0 for key in
                        ("line_attenuation_50hz", "frontal_multiplier", "theta_alpha_multiplier")):
                    raise ValueError("This pilot requires the verified fixed-preprocessing caches.")
            count = len(matrix)
            recording = f"{task}:s{subject:02d}:r{level}"
            features.append(matrix)
            subject_ids.extend([subject] * count)
            recordings.extend([recording] * count)
            windows.extend(f"{recording}:w{index:05d}" for index in range(count))
            if training:
                labels.extend([int(level >= 2)] * count)
    values = dict(features=np.concatenate(features), subjects=np.asarray(subject_ids, dtype=np.int64),
                  recordings=np.asarray(recordings), windows=np.asarray(windows))
    result = TrainingRows(**values, labels=np.asarray(labels, dtype=np.int64)) if training else PredictionRows(**values)
    validate_rows(result)
    return result


class CheckpointError(RuntimeError):
    pass


def validate_probability(values: np.ndarray, n: int) -> None:
    if values.shape != (n, 2) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise CheckpointError("Prediction checkpoint contains invalid probabilities or dimensions.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise CheckpointError("Prediction rows are not normalized.")


class CheckpointStore:
    def __init__(self, root: Path, identity: dict):
        self.root = root
        self.identity = identity
        self.identity_digest = hashlib.sha256(json_bytes(identity)).hexdigest()
        self.directory = root / "checkpoints"
        manifest = root / "01_run_identity.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved.get("identity") != identity or saved.get("identity_sha256") != self.identity_digest:
                raise CheckpointError("Protocol, inputs, code or runtime changed. Existing checkpoints were preserved; no automatic restart is allowed.")
        else:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise CheckpointError("Checkpoint files exist without their run identity; refusing to reuse or overwrite them.")
            atomic_json(manifest, {"identity": identity, "identity_sha256": self.identity_digest,
                                   "created_utc": datetime.now(timezone.utc).isoformat()})
        self.directory.mkdir(parents=True, exist_ok=True)

    def _paths(self, name):
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError("Unsafe stage name.")
        return self.directory / (name + ".npz"), self.directory / (name + ".done.json")

    def load(self, name, operation, n):
        data_path, done_path = self._paths(name)
        if not done_path.exists():
            return None
        try:
            marker = json.loads(done_path.read_text(encoding="utf-8"))
            if marker["identity_sha256"] != self.identity_digest or marker["operation"] != operation:
                raise CheckpointError(f"Stage identity mismatch: {name}")
            if file_digest(data_path) != marker["data_sha256"]:
                raise CheckpointError(f"Stage checksum mismatch: {name}")
            with np.load(data_path, allow_pickle=False) as archive:
                probability = archive["probability"].astype(np.float64)
                epoch = int(archive["epoch"].item())
            validate_probability(probability, n)
            if epoch < 0:
                raise CheckpointError("Invalid cached epoch.")
            fit_seconds = float(marker["fit_seconds"])
            if not np.isfinite(fit_seconds) or fit_seconds < 0:
                raise CheckpointError("Invalid cached fitting duration.")
            return probability, epoch, fit_seconds
        except CheckpointError:
            raise
        except (OSError, ValueError, KeyError, EOFError) as error:
            raise CheckpointError(f"Incomplete or damaged committed checkpoint: {name}. No files were deleted.") from error

    def save(self, name, operation, probability, epoch, fit_seconds):
        data_path, done_path = self._paths(name)
        validate_probability(probability, len(probability))
        temporary = data_path.with_name(data_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, probability=probability, epoch=np.asarray(epoch, dtype=np.int64))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, data_path)
        atomic_json(done_path, {"identity_sha256": self.identity_digest, "operation": operation,
                               "fit_seconds": float(fit_seconds),
                               "data_sha256": file_digest(data_path), "committed_utc": datetime.now(timezone.utc).isoformat()})


class CoreBackend:
    """Reuse the frozen model implementations; only orchestration is new."""
    tag = "existing-revision-pipeline-models"

    def mlp(self, x, y, recordings, validation_x, validation_y, validation_recordings, prediction_x,
            config, seed, device, fixed_epochs):
        fitted, epoch = models.fit_mlp(x, y, config, seed, device,
                                      x_validation=validation_x, y_validation=validation_y,
                                      validation_recordings=validation_recordings, fixed_epochs=fixed_epochs)
        return fitted.predict_proba(prediction_x), epoch

    def forest(self, method, x, y, prediction_x, config, seed):
        fitted = models._make_forest(method, config, seed)
        fitted.fit(x, y)
        return fitted.predict_proba(prediction_x), 0

    def stack(self, inner_probabilities, labels, recordings, target_probabilities, config, seed):
        _, inverse, counts = np.unique(recordings, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inverse]
        weight /= np.mean(weight)
        fitted = LogisticRegression(C=float(config["C"]), class_weight="balanced",
                                    max_iter=int(config["max_iter"]), random_state=int(seed))
        fitted.fit(models._stack_features(*(inner_probabilities[name] for name in BRANCHES)), labels, sample_weight=weight)
        return fitted.predict_proba(models._stack_features(*(target_probabilities[name] for name in BRANCHES))), 0


def membership(train: TrainingRows, indices: np.ndarray, validation: np.ndarray | None, outer: int, meta: int):
    subjects = sorted(int(value) for value in np.unique(train.subjects[indices]))
    records = sorted(str(value) for value in np.unique(train.recordings[indices]))
    if {outer, meta} & set(subjects):
        raise ValueError("Excluded participant in an actual model fitting slice.")
    result = {"training_subjects": subjects, "training_recordings": records, "training_windows": len(indices),
              "training_window_sha256": array_digest(train.windows[indices]),
              "training_features_sha256": array_digest(train.features[indices]),
              "training_labels_sha256": array_digest(train.labels[indices]),
              "training_class_counts": np.bincount(train.labels[indices], minlength=2).tolist()}
    if min(result["training_class_counts"]) == 0:
        raise ValueError("A model fitting slice lacks a class.")
    if validation is not None:
        valid_subjects = sorted(int(value) for value in np.unique(train.subjects[validation]))
        if {outer, meta} & set(valid_subjects) or set(subjects) & set(valid_subjects):
            raise ValueError("Invalid early-stopping/inner-validation participants.")
        result.update({"validation_subjects": valid_subjects,
                       "validation_recordings": sorted(str(value) for value in np.unique(train.recordings[validation])),
                       "validation_window_sha256": array_digest(train.windows[validation]),
                       "validation_features_sha256": array_digest(train.features[validation]),
                       "validation_labels_sha256": array_digest(train.labels[validation])})
    return result


def runtime_identity(device: torch.device):
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_threads": torch.get_num_threads(),
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@dataclass
class PilotResult:
    probabilities: dict[str, np.ndarray]
    selected_epoch: int
    inner_epochs: list[int]
    stages: list[dict]


def run_unit(train: TrainingRows, target: PredictionRows, *, outer: int, meta: int, seed: int,
             config: dict, device: torch.device, output: Path, context: dict,
             backend: CoreBackend | None = None, emit: Callable[[str], None] = print) -> PilotResult:
    validate_unit(train, target, outer, meta)
    backend = backend or CoreBackend()
    identity = {"schema": SCHEMA, "outer_target": outer, "meta_validation": meta, "seed": seed,
                "config": config, "backend": backend.tag, "training_rows": train.fingerprint(),
                "target_rows_without_labels": target.fingerprint(), "context": context}
    stages = []
    with run_lock(output):
        store = CheckpointStore(output, identity)
        folds = splits.subject_loso_folds(train.subjects)
        total = 3 * len(folds) + 4
        inner = {name: np.full((len(train.features), 2), np.nan, dtype=np.float64) for name in BRANCHES}
        epochs = []

        def stage(name, operation, n, action):
            number = len(stages) + 1
            values = store.load(name, operation, n)
            reused = values is not None
            emit(f"[{number}/{total}] {'reuse' if reused else 'fit'} {name}")
            if not reused:
                started = time.monotonic()
                probability, epoch = action()
                probability = np.asarray(probability, dtype=np.float64)
                validate_probability(probability, n)
                values = probability, int(epoch), time.monotonic() - started
            probability, epoch, fit_seconds = values
            if operation["model"] == "always_nn" and not 1 <= epoch <= int(config["mlp"]["max_epochs"]):
                raise CheckpointError("Cached selected epoch is outside the frozen model range.")
            if "fixed_epochs" in operation and epoch != operation["fixed_epochs"]:
                raise CheckpointError("Final MLP epoch does not match inner-fold selection.")
            if not reused:
                store.save(name, operation, probability, epoch, fit_seconds)
                emit(f"    saved ({fit_seconds:.1f}s, epoch={epoch})")
            stages.append({"stage": name, "reused": reused, "operation": operation,
                           "selected_epoch": epoch, "fit_seconds": fit_seconds})
            atomic_json(output / "02_training_membership.json", {"outer_target": outer, "meta_validation": meta,
                        "expected_stages": total, "completed_stages": len(stages), "stages": stages})
            return probability, epoch

        for fold_number, fold in enumerate(folds):
            training, validation = fold.train, fold.validation
            held = int(np.unique(train.subjects[validation]).item())
            member = membership(train, training, validation, outer, meta)
            fold_seed = int(seed) + 1009 * (fold_number + 1)
            probability, epoch = stage(f"inner_s{held:02d}_mlp", {**member, "model": "always_nn", "seed": fold_seed}, len(validation),
                lambda: backend.mlp(train.features[training], train.labels[training], train.recordings[training],
                                    train.features[validation], train.labels[validation], train.recordings[validation],
                                    train.features[validation], config["mlp"], fold_seed, device, None))
            inner["always_nn"][validation] = probability
            epochs.append(epoch)
            for offset, name in enumerate(("rf", "extra_trees"), start=1):
                forest_seed = fold_seed + offset * 37
                probability, _ = stage(f"inner_s{held:02d}_{name}", {**member, "model": name, "seed": forest_seed}, len(validation),
                    lambda: backend.forest(name, train.features[training], train.labels[training], train.features[validation], config[name], forest_seed))
                inner[name][validation] = probability
        if any(not np.isfinite(values).all() for values in inner.values()):
            raise CheckpointError("Inner predictions do not cover the complete retained training set.")

        selected_epoch = max(1, int(np.median(epochs)))
        all_rows = np.arange(len(train.features))
        member = membership(train, all_rows, None, outer, meta)
        predictions = {}
        predictions["always_nn"], _ = stage("final_mlp", {**member, "model": "always_nn", "seed": int(seed), "fixed_epochs": selected_epoch}, len(target.features),
            lambda: backend.mlp(train.features, train.labels, train.recordings, None, None, None,
                                target.features, config["mlp"], int(seed), device, selected_epoch))
        for offset, name in enumerate(("rf", "extra_trees"), start=1):
            forest_seed = int(seed) + offset * 37
            predictions[name], _ = stage(f"final_{name}", {**member, "model": name, "seed": forest_seed}, len(target.features),
                lambda: backend.forest(name, train.features, train.labels, target.features, config[name], forest_seed))
        operation = {**member, "model": "always_fuse", "seed": int(seed),
                     "inner_probability_sha256": {name: array_digest(inner[name]) for name in BRANCHES},
                     "target_probability_sha256": {name: array_digest(predictions[name]) for name in BRANCHES}}
        predictions["always_fuse"], _ = stage("final_stacker", operation, len(target.features),
            lambda: backend.stack(inner, train.labels, train.recordings, predictions, config["stacker"], int(seed)))
        result = PilotResult(predictions, selected_epoch, epochs, stages)
        write_label_free_predictions(output, result, target, outer, meta, seed)
        return result


def write_label_free_predictions(output, result, target, outer, meta, seed):
    rows = []
    for name, probability in result.probabilities.items():
        for index in range(len(target.features)):
            rows.append({"outer_subject": outer, "meta_subject": meta, "seed": seed, "method": name,
                         "recording_id": str(target.recordings[index]), "window_id": str(target.windows[index]),
                         "p0": probability[index, 0], "p1": probability[index, 1]})
    content = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")
    atomic_bytes(output / "03_label_free_meta_window_predictions.csv", content)


def score_predictions(result: PilotResult, train: TrainingRows, target: PredictionRows,
                      labels_by_recording: dict[str, int], *, outer: int, meta: int, seed: int, task: str):
    """Labels enter only here, after target probabilities have been committed."""
    if set(labels_by_recording) != set(target.recordings.tolist()):
        raise ValueError("Scoring labels do not match the target recordings.")
    if set(labels_by_recording.values()) != {0, 1}:
        raise ValueError("Scoring requires both workload classes.")
    frames = []
    for name, probability in result.probabilities.items():
        frames.append(pd.DataFrame({"dataset": "openbci", "protocol": "loso_meta_pilot", "subject": meta,
                     "direction": task, "recording_id": target.recordings, "window_id": target.windows,
                     "method": name, "true_label": [labels_by_recording[str(value)] for value in target.recordings],
                     "p0": probability[:, 0], "p1": probability[:, 1], "ensemble_size": 1}))
    recordings = aggregation.aggregate_recordings(pd.concat(frames, ignore_index=True))
    if len(recordings) != 16:
        raise ValueError("Expected four methods times four meta-validation recordings.")
    accuracy = recordings.groupby("method")["correct"].mean().to_dict()
    diagnostics = risk_gate.label_free_diagnostics(result.probabilities["always_nn"], result.probabilities["always_fuse"], train.features, target.features)
    diagnostics["feature_shift"] = risk_gate.robust_feature_shift(train.features, target.features)
    gain = float(accuracy["always_fuse"] - accuracy["always_nn"])
    summary = {"status": "completed_single_unit_engineering_pilot_not_paper_result",
               "outer_subject": outer, "meta_subject": meta, "seed": seed, "task": task,
               "selected_epoch": result.selected_epoch, "inner_selected_epochs": result.inner_epochs,
               "completed_fit_stages": len(result.stages), "reused_fit_stages": sum(stage["reused"] for stage in result.stages),
               "completed_stage_fit_seconds": sum(stage["fit_seconds"] for stage in result.stages),
               "this_invocation_fit_seconds": sum(stage["fit_seconds"] for stage in result.stages if not stage["reused"]),
               "duration_scope": "Completed fit stages only; excludes interrupted work, input loading and scoring.",
               "recording_rows": len(recordings), "accuracy_for_engineering_inspection_only": accuracy,
               "ordinary_recording_gain": gain, "loss_event_le_minus_0_05": gain <= -0.05,
               "label_free_diagnostics": diagnostics, "gate_fitted": False,
               "full_five_seed_ensemble": False, "population_inference_performed": False}
    return recordings, summary


def resolve_device(selection: str) -> torch.device:
    if selection == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device("cuda:0" if selection == "cuda" or (selection == "auto" and torch.cuda.is_available()) else "cpu")
