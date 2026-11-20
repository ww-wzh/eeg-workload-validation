"""Execute the frozen COG-BCI gate study, with label-free prediction APIs.

Each completed fit stage is committed by a SHA-256 receipt. A restart verifies
the receipt and exact execution identity before reusing its predictions. The
historical model implementation is imported unchanged. No target labels are
passed to a fitting or prediction function. This script does not score gates.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = HERE.parent / "cogbci_gate_design"
SOURCE = ROOT / "shared"
sys.path.insert(0, str(SOURCE))
from revision_pipeline import models

BRANCHES = ("feature_mlp", "rf", "extra_trees")
PILOT = "NBACK_exclude_s01_s02_seed1335"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "joblib": joblib.__version__,
            "device": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
            "torch_threads": 1, "tree_threads": 8, "deterministic_algorithms": True,
            "allow_tf32": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen execution requires the verified CUDA device")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_design():
    freeze = load(DESIGN / "design_freeze.json")
    for name, expected in freeze["sha256"].items():
        if sha(DESIGN / name) != expected:
            raise ValueError(f"Frozen B1 input changed: {name}")
    for name, expected in freeze["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Historical source changed: {name}")
    return freeze


def execution_identity():
    verify_design()
    code = [Path(__file__), SOURCE / "revision_pipeline/models.py",
            SOURCE / "revision_pipeline/splits.py"]
    identity = {"code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
                "design_freeze_sha256": sha(DESIGN / "design_freeze.json"),
                "protocol_sha256": sha(DESIGN / "protocol.json"),
                "runtime": runtime(),
                "prediction_checkpoint_policy": "Save every stage's predictions, membership, seed, epoch and train scaler. Retain all fitted models for the ID-selected engineering pilot; later fits are reproducible from pinned sources and inputs. No mid-epoch resume.",
                "optimization": "Only frozen unordered-pair symmetry reuse; no scientific design changes."}
    dest = HERE / "execution_freeze.json"
    if dest.exists():
        old = load(dest)
        if old["identity"] != identity:
            raise ValueError("Execution code/input/runtime changed; preserve outputs and create an explicit new version")
    else:
        dump(dest, {"frozen_utc": now(), "identity": identity, "identity_sha256": digest(identity),
                    "models_fitted_before_freeze": 0})
    return digest(identity)


class TaskData:
    def __init__(self, task):
        records = [r for r in load(DESIGN / "record_selection.json")["records"] if r["task"] == task]
        records.sort(key=lambda r: (r["subject"], r["session"], r["label"]))
        xx, yy, ss, rr, ww = [], [], [], [], []
        self.record_rows = records
        for r in records:
            path = DESIGN / "prepared_records_float64" / (r["record_id"] + ".npz")
            with np.load(path, allow_pickle=False) as arr:
                features = arr["features"]
                starts = arr["raw_supports"][:, 0]
                assert features.shape == (20, 272) and np.all(np.diff(starts) > 0)
                xx.append(features)
                yy.extend([r["label"]] * 20)
                ss.extend([r["subject"]] * 20)
                rr.extend([r["record_id"]] * 20)
                ww.extend([f'{r["record_id"]}:{int(s)}' for s in starts])
        self.x = np.concatenate(xx).astype(np.float32)
        self.y = np.asarray(yy, dtype=np.int64)
        self.subject = np.asarray(ss, dtype=np.int64)
        self.record = np.asarray(rr)
        self.window = np.asarray(ww)
        assert np.isfinite(self.x).all() and len(set(ww)) == len(ww)
        for s in np.unique(self.subject):
            counts = np.bincount(self.y[self.subject == s], minlength=2)
            assert counts[0] == counts[1] > 0

    def indices(self, subjects):
        return np.flatnonzero(np.isin(self.subject, subjects))

    def training(self, indices):
        return {"x": self.x[indices], "y": self.y[indices], "records": self.record[indices],
                "subjects": self.subject[indices], "windows": self.window[indices]}

    def prediction(self, indices):
        # Deliberately no labels; this is the only target-data API used by fits.
        return {"x": self.x[indices], "records": self.record[indices],
                "subjects": self.subject[indices], "windows": self.window[indices]}


def membership(data):
    return {"subjects": sorted(set(data["subjects"].tolist())),
            "records": sorted(set(data["records"].tolist())),
            "windows_sha256": digest(data["windows"].tolist()), "n_windows": len(data["x"])}


def check_isolation(train, predict, validation=None):
    assert "y" not in predict
    for other in [predict] + ([] if validation is None else [validation]):
        for field in ("subjects", "records", "windows"):
            assert set(train[field]).isdisjoint(other[field]), f"Overlap in {field}"
    assert set(np.unique(train["y"])) == {0, 1}
    if validation is not None:
        for field in ("subjects", "records", "windows"):
            assert set(validation[field]).isdisjoint(predict[field])
        assert set(np.unique(validation["y"])) == {0, 1}


def check_probs(p, rows):
    assert p.shape == (rows, 2) and np.isfinite(p).all()
    assert np.all(p >= 0) and np.all(p <= 1)
    assert np.allclose(p.sum(axis=1), 1, atol=2e-7, rtol=0)


def read_stage(path, identity):
    receipt = path / "complete.json"
    if not receipt.exists():
        return None
    saved = load(receipt)
    assert saved["identity"] == identity, f"Stage identity mismatch: {path}"
    for name, expected in saved["files_sha256"].items():
        assert sha(path / name) == expected, f"Corrupt checkpoint: {path / name}"
    with np.load(path / "predictions.npz", allow_pickle=False) as arr:
        result = {k: arr[k] for k in arr.files}
    return saved, result


def commit_stage(path, identity, result, metadata, fitted=None):
    path.mkdir(parents=True, exist_ok=True)
    save_npz(path / "predictions.npz", **result)
    dump(path / "fit.json", metadata)
    names = ["predictions.npz", "fit.json"]
    if fitted is not None:
        temp = path / "model.joblib.tmp"
        joblib.dump(fitted, temp, compress=3)
        temp.replace(path / "model.joblib")
        names.append("model.joblib")
    dump(path / "complete.json", {"identity": identity, "completed_utc": now(),
                                  "files_sha256": {n: sha(path / n) for n in names}})
    return read_stage(path, identity)


def fit_branch(path, branch, train, predict, config, seed, execution, validation=None,
               fixed_epochs=None, retain_model=False):
    check_isolation(train, predict, validation)
    identity = {"execution": execution, "branch": branch, "seed": seed,
                "train": membership(train), "predict": membership(predict),
                "validation": None if validation is None else membership(validation),
                "config": config, "fixed_epochs": fixed_epochs,
                "train_labels_sha256": digest(train["y"].tolist()),
                "validation_labels_sha256": None if validation is None else digest(validation["y"].tolist())}
    cached = read_stage(path, identity)
    if cached is not None:
        return cached[1], load(path / "fit.json"), True
    # Keep uncommitted artifacts for diagnosis instead of silently overwriting.
    if path.exists() and any(path.iterdir()):
        preserved = path.with_name(path.name + "_uncommitted_" + str(time.time_ns()))
        path.rename(preserved)
    started = time.perf_counter()
    kwargs = {} if validation is None else {"x_validation": validation["x"],
        "y_validation": validation["y"], "validation_recordings": validation["records"]}
    with threadpool_limits(limits=1):
        if branch == "feature_mlp":
            fitted, epoch = models.fit_mlp(train["x"], train["y"], config, seed,
                                          torch.device("cuda"), fixed_epochs=fixed_epochs, **kwargs)
            metadata = {"best_epoch": epoch, "scaler_mean": fitted.scaler.mean_.tolist(),
                        "scaler_scale": fitted.scaler.scale_.tolist(),
                        "scaler_var": fitted.scaler.var_.tolist(),
                        "scaler_n_samples_seen": int(fitted.scaler.n_samples_seen_)}
            assert np.array_equal(fitted.scaler.mean_, np.mean(train["x"], axis=0, dtype=np.float64))
        else:
            fitted = models._make_forest(branch, {**config, "n_jobs": 8}, seed)
            fitted.fit(train["x"], train["y"])
            assert np.array_equal(fitted.classes_, [0, 1])
            metadata = {"n_estimators": len(fitted.estimators_),
                        "n_features_in": int(fitted.n_features_in_)}
        result = {"prediction": fitted.predict_proba(predict["x"]),
                  "prediction_windows": predict["windows"]}
        check_probs(result["prediction"], len(predict["x"]))
        if validation is not None:
            result["validation"] = fitted.predict_proba(validation["x"])
            result["validation_windows"] = validation["windows"]
            check_probs(result["validation"], len(validation["x"]))
        metadata.update({"elapsed_seconds": time.perf_counter() - started,
                         "train_membership": membership(train), "prediction_membership": membership(predict),
                         "validation_membership": None if validation is None else membership(validation),
                         "seed": seed, "branch": branch, "target_labels_received": False})
        commit_stage(path, identity, result, metadata, fitted=fitted if retain_model else None)
    return result, metadata, False


def stack_x(branches):
    return models._stack_features(*(branches[name] for name in BRANCHES))


def progress(job_id, stage, cached, seconds):
    value = {"utc": now(), "job_id": job_id, "stage": stage,
             "checkpoint_reused": cached, "fit_seconds": round(seconds, 3)}
    dump(HERE / "progress.json", value)
    print(json.dumps(value), flush=True)


def run_job(job, data, protocol, execution):
    path = HERE / "jobs" / job["job_id"]
    job_identity = {"execution": execution, "job": job}
    if (path / "complete.json").exists():
        receipt = load(path / "complete.json")
        assert receipt["identity"] == job_identity
        for name, expected in receipt["files_sha256"].items():
            assert sha(path / name) == expected, f"Job checkpoint corrupted: {path / name}"
        progress(job["job_id"], "complete", True, 0)
        return
    started = time.perf_counter()
    train_idx = data.indices(job["training_subjects"])
    target_idx = data.indices(job["prediction_subjects"])
    train = data.training(train_idx)
    target = data.prediction(target_idx)
    assert sorted(set(target["subjects"])) == job["excluded_subjects"]
    assert sorted(set(train["subjects"])) == job["training_subjects"]
    assert not set(job["training_subjects"]) & set(job["excluded_subjects"])
    oof = {k: np.full((len(train_idx), 2), np.nan) for k in BRANCHES}
    coverage = np.zeros(len(train_idx), dtype=int)
    epochs = []
    receipts = []
    retain = job["job_id"] == PILOT
    for fold, subject in enumerate(job["training_subjects"]):
        vi = np.flatnonzero(train["subjects"] == subject)
        ti = np.flatnonzero(train["subjects"] != subject)
        inner = {k: v[ti] for k, v in train.items()}
        validation = {k: v[vi] for k, v in train.items()}
        coverage[vi] += 1
        for offset, branch in enumerate(BRANCHES):
            stage_path = path / "stages" / f"inner_s{subject:02d}_{branch}"
            cfg = protocol["base_models"]["mlp" if branch == "feature_mlp" else branch]
            result, meta, cached = fit_branch(stage_path, branch, inner, target, cfg,
                int(job["seed"]) + 1009 * (fold + 1) + 37 * offset, execution,
                validation=validation, retain_model=retain)
            assert np.array_equal(result["validation_windows"], train["windows"][vi])
            oof[branch][vi] = result["validation"]
            if branch == "feature_mlp":
                epochs.append(meta["best_epoch"])
            receipts.append(stage_path / "complete.json")
            progress(job["job_id"], stage_path.name, cached, meta["elapsed_seconds"])
    assert np.all(coverage == 1)
    for values in oof.values():
        check_probs(values, len(train_idx))
    selected = max(1, int(np.median(epochs)))
    test = {}
    for offset, branch in enumerate(BRANCHES):
        stage_path = path / "stages" / ("final_" + branch)
        cfg = protocol["base_models"]["mlp" if branch == "feature_mlp" else branch]
        result, meta, cached = fit_branch(stage_path, branch, train, target, cfg,
            int(job["seed"]) + 37 * offset, execution,
            fixed_epochs=selected if branch == "feature_mlp" else None, retain_model=retain)
        assert np.array_equal(result["prediction_windows"], target["windows"])
        test[branch] = result["prediction"]
        receipts.append(stage_path / "complete.json")
        progress(job["job_id"], stage_path.name, cached, meta["elapsed_seconds"])
    # Stacker learns from internal branch inputs only. Its OOF-input scores are
    # deliberately kept as dependent proxy responses for the audit.
    stack_path = path / "stages" / "stacker"
    stack_identity = {"execution": execution, "job": job, "branch_receipts":
                      {p.relative_to(path).as_posix(): sha(p) for p in receipts}}
    cached = read_stage(stack_path, stack_identity)
    if cached is None:
        start_stack = time.perf_counter()
        cfg = protocol["base_models"]["stacker"]
        stacker = LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=cfg["max_iter"],
                                     solver=cfg["solver"], fit_intercept=True, random_state=job["seed"])
        unique, counts = np.unique(train["records"], return_counts=True)
        weights = dict(zip(unique, counts))
        sample_weights = np.array([1 / weights[r] for r in train["records"]])
        sample_weights /= sample_weights.mean()
        with threadpool_limits(limits=1):
            stacker.fit(stack_x(oof), train["y"], sample_weight=sample_weights)
        if np.any(stacker.n_iter_ >= cfg["max_iter"]):
            raise RuntimeError(f"Stacker iteration limit reached: {job['job_id']}")
        result = {"prediction": stacker.predict_proba(stack_x(test)),
                  "prediction_windows": target["windows"], "proxy": stacker.predict_proba(stack_x(oof)),
                  "proxy_windows": train["windows"]}
        meta = {"coefficient": stacker.coef_.tolist(), "intercept": stacker.intercept_.tolist(),
                "classes": stacker.classes_.tolist(), "n_iter": stacker.n_iter_.tolist(),
                "elapsed_seconds": time.perf_counter() - start_stack,
                "train_membership": membership(train), "target_labels_received": False}
        cached = commit_stage(stack_path, stack_identity, result, meta, fitted=stacker if retain else None)
    test["always_fuse"] = cached[1]["prediction"]
    oof["always_fuse"] = cached[1]["proxy"]
    for values in [*test.values(), *oof.values()]:
        check_probs(values, len(target_idx) if values is test.get("always_fuse") else len(values))
    path.mkdir(parents=True, exist_ok=True)
    save_npz(path / "oof_predictions.npz", window_ids=train["windows"], subject=train["subjects"],
             record_ids=train["records"], **oof)
    files = ["oof_predictions.npz"]
    for subject in job["prediction_subjects"]:
        mask = target["subjects"] == subject
        name = f"target_s{subject:02d}_predictions.npz"
        save_npz(path / name, window_ids=target["windows"][mask], record_ids=target["records"][mask],
                 **{k: v[mask] for k, v in test.items()})
        files.append(name)
    dump(path / "job_summary.json", {"job": job, "inner_best_epochs": epochs, "final_epoch": selected,
         "elapsed_seconds_this_invocation": time.perf_counter() - started,
         "inner_coverage_min": int(coverage.min()), "inner_coverage_max": int(coverage.max()),
         "target_labels_received": False, "symmetric_routes": job.get("response_routing", [])})
    files.append("job_summary.json")
    receipts.append(stack_path / "complete.json")
    files.extend(p.relative_to(path).as_posix() for p in receipts)
    dump(path / "complete.json", {"identity": job_identity, "completed_utc": now(),
         "files_sha256": {name: sha(path / name) for name in files}})
    progress(job["job_id"], "complete", False, time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "pilot", "all", "job"])
    parser.add_argument("--job-id")
    args = parser.parse_args()
    configure()
    execution = execution_identity()
    protocol = load(DESIGN / "protocol.json")
    jobs = load(DESIGN / "job_plan.json")["jobs"]
    if args.mode == "freeze":
        print("Execution identity frozen and all B1 inputs verified.", flush=True)
        return
    if args.mode == "pilot":
        jobs = [j for j in jobs if j["job_id"] == PILOT]
    elif args.mode == "job":
        jobs = [j for j in jobs if j["job_id"] == args.job_id]
        assert len(jobs) == 1
    else:
        pilot_report = load(HERE / "pilot_verification.json")
        assert pilot_report["status"] == "passed" and pilot_report["execution"] == execution
        # Complete one seed across tasks before the next. This is scheduling only;
        # every preregistered seed/job remains required for analysis.
        jobs.sort(key=lambda j: (j["seed"], j["task"], j["kind"], j["excluded_subjects"]))
    datasets = {}
    for index, job in enumerate(jobs):
        if job["task"] not in datasets:
            datasets[job["task"]] = TaskData(job["task"])
        if shutil.disk_usage(HERE).free < 5 * 1024 ** 3:
            raise RuntimeError("Less than 5 GiB free: stop safely with completed stage checkpoints intact")
        run_job(job, datasets[job["task"]], protocol, execution)
        dump(HERE / "run_status.json", {"status": "running", "mode": args.mode,
             "last_job": job["job_id"], "jobs_visited_this_run": index + 1,
             "planned_jobs_this_run": len(jobs), "utc": now()})
    dump(HERE / "run_status.json", {"status": "selected_jobs_complete", "mode": args.mode,
         "jobs_visited_this_run": len(jobs), "utc": now()})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        dump(HERE / "last_failure.json", {"utc": now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise
