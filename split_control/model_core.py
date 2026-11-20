"""Original split-control fitting and aggregation functions with caller-configured output paths."""
from __future__ import annotations
import hashlib
import io
import json
import os
import time
import uuid
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_bytes(path, data):
    """Unique temp name, durable close, then bounded Windows sharing retries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.1 * 2**attempt, 2.0))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def write_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def validate_completion(folder, identity_sha, job_id=None):
    completion = folder / "complete.json"
    if not completion.exists():
        return None
    value = json.loads(completion.read_text(encoding="utf-8"))
    if value["identity_sha256"] != identity_sha:
        raise ValueError(f"Checkpoint identity mismatch: {folder}")
    if job_id is not None and value["job_id"] != job_id:
        raise ValueError("Checkpoint job ID mismatch")
    for name, expected in value["files_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {folder / name}")
    return value


def assemble(parts, arrays):
    features, labels, membership = [], [], []
    for part in sorted(parts, key=lambda item: item["raw_level"]):
        start, stop = part["filter_interval"]
        cache = arrays[(part["recording_id"], start, stop)]
        starts = sorted(part["window_starts"])
        indices = np.asarray([(value - start) // 125 for value in starts], dtype=int)
        if any((value - start) % 125 for value in starts) or (indices < 0).any() or (indices >= len(cache)).any():
            raise ValueError("Window start is outside its filtering unit/cache lattice")
        features.append(cache[indices])
        labels.extend([part["binary_label"]] * len(starts))
        membership.extend({"recording_id": part["recording_id"], "raw_level": part["raw_level"],
                           "binary_label": part["binary_label"], "start": value,
                           "filter_interval": part["filter_interval"]} for value in starts)
    return np.concatenate(features), np.asarray(labels, dtype=int), membership


def aggregate(probabilities, membership):
    rows = []
    for rid in sorted({row["recording_id"] for row in membership}):
        indices = [i for i, row in enumerate(membership) if row["recording_id"] == rid]
        log_probability = np.log(np.clip(probabilities[indices], 1e-12, 1.0)).mean(axis=0)
        p = np.exp(log_probability - np.max(log_probability))
        p /= p.sum()
        part = membership[indices[0]]
        rows.append({"recording_id": rid, "raw_level": part["raw_level"],
                     "binary_label": part["binary_label"], "windows": len(indices),
                     "probability_0": float(p[0]), "probability_1": float(p[1]),
                     "prediction": int(p.argmax())})
    return rows


def metrics(rows):
    correct = np.asarray([row["prediction"] == row["binary_label"] for row in rows])
    labels = np.asarray([row["binary_label"] for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Scoring needs both classes")
    return {"accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean([correct[labels == c].mean() for c in [0, 1]])),
            "recordings": len(rows)}


def run_job(job, arrays, identity_sha):
    folder = HERE / "jobs" / job["job_id"]
    complete = validate_completion(folder, identity_sha, job["job_id"])
    if complete:
        return json.loads((folder / "result.json").read_text(encoding="utf-8")), True
    folder.mkdir(parents=True, exist_ok=True)
    x_train, y_train, train_membership = assemble(job["training"], arrays)
    x_test, y_test, test_membership = assemble(job["testing"], arrays)
    train_std = x_train.std(axis=0, ddof=0)
    mask = train_std > 1e-8
    if not mask.any() or not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Invalid features or empty training-only feature mask")
    scaler = StandardScaler().fit(x_train[:, mask])
    transformed_train = scaler.transform(x_train[:, mask])
    transformed_test = scaler.transform(x_test[:, mask])
    kwargs = {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 4000,
              "tol": 1e-4, "random_state": 0, "fit_intercept": True}
    if tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 8):
        kwargs["l1_ratio"] = 0.0
    else:
        kwargs["penalty"] = "l2"
    model = LogisticRegression(**kwargs)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=ConvergenceWarning)
        pools = threadpool_info()
        model.fit(transformed_train, y_train)
        probabilities = model.predict_proba(transformed_test)
    elapsed = time.perf_counter() - started
    if not np.array_equal(model.classes_, [0, 1]) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid class ordering or probabilities")
    record_rows = aggregate(probabilities, test_membership)
    result = {"identity_sha256": identity_sha, "job_id": job["job_id"],
              "job": job, "started_after_inputs_verified": True,
              "train_membership": train_membership, "test_membership": test_membership,
              "fit_seconds": elapsed, "n_iter": model.n_iter_.tolist(),
              "feature_mask_count": int(mask.sum()), "parameters": kwargs,
              "warnings": [{"category": type(w.message).__name__, "text": str(w.message)} for w in caught],
              "numeric_threadpools_during_fit": pools,
              "recording_predictions": record_rows, "fold_metrics": metrics(record_rows)}
    write_npz(folder / "model_and_predictions.npz", x_train=x_train, x_test=x_test,
              y_train=y_train, y_test=y_test, feature_mask=mask, train_std=train_std,
              scaler_mean=scaler.mean_, scaler_var=scaler.var_, scaler_scale=scaler.scale_,
              scaler_samples_seen=np.asarray(scaler.n_samples_seen_),
              coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_,
              test_probabilities=probabilities, n_iter=model.n_iter_)
    write_json(folder / "result.json", result)
    write_json(folder / "complete.json", {"identity_sha256": identity_sha, "job_id": job["job_id"],
               "files_sha256": {name: sha(folder / name) for name in ["model_and_predictions.npz", "result.json"]}})
    return result, False
