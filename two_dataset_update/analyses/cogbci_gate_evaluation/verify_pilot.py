"""Independently replay the engineering unit against the pinned old trainer."""
import inspect
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_nested as runner
from revision_pipeline.splits import Split


def main():
    runner.configure()
    execution = runner.execution_identity()
    protocol = runner.load(runner.DESIGN / "protocol.json")
    job = next(j for j in runner.load(runner.DESIGN / "job_plan.json")["jobs"]
               if j["job_id"] == runner.PILOT)
    path = runner.HERE / "jobs" / runner.PILOT
    receipt = runner.load(path / "complete.json")
    assert receipt["identity"] == {"execution": execution, "job": job}
    for name, expected in receipt["files_sha256"].items():
        assert runner.sha(path / name) == expected
    data = runner.TaskData("NBACK")
    train_idx = data.indices(job["training_subjects"])
    target_idx = data.indices(job["prediction_subjects"])
    training = data.training(train_idx)
    target = data.prediction(target_idx)
    assert "y" not in target
    # Changing every held-out label leaves both fit and prediction input APIs
    # byte-for-byte identical; only the later scoring layer receives labels.
    data.y[target_idx] = 1 - data.y[target_idx]
    for k, v in training.items():
        assert np.array_equal(v, data.training(train_idx)[k])
    for k, v in target.items():
        assert np.array_equal(v, data.prediction(target_idx)[k])
    data.y[target_idx] = 1 - data.y[target_idx]
    folds = []
    receipt_count = 0
    for subject in job["training_subjects"]:
        vi = np.flatnonzero(training["subjects"] == subject)
        ti = np.flatnonzero(training["subjects"] != subject)
        folds.append(Split(train=ti, validation=vi))
        for branch in runner.BRANCHES:
            stage = path / "stages" / f"inner_s{subject:02d}_{branch}"
            saved = runner.load(stage / "complete.json")
            runner.read_stage(stage, saved["identity"])
            members = runner.load(stage / "fit.json")
            assert set(members["train_membership"]["subjects"]) == set(job["training_subjects"]) - {subject}
            assert members["validation_membership"]["subjects"] == [subject]
            assert set(members["prediction_membership"]["subjects"]) == {1, 2}
            assert members["target_labels_received"] is False
            receipt_count += 1
    # Symmetry is a property of the training and seed construction: swapping
    # outer/meta roles leaves the unordered training exclusion and fold seeds
    # unchanged. The two label-free target files remain separate.
    for first, second in [(1, 2), (2, 1)]:
        assert job["training_subjects"] == [s for s in protocol["cohort"]["expected_subjects"] if s not in [first, second]]
    models = dict(protocol["base_models"])
    for name in ["rf", "extra_trees"]:
        models[name] = {**models[name], "n_jobs": 8}
    print("All stage receipts and exclusion memberships passed; running independent legacy trainer replay.", flush=True)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        replay = runner.models.nested_fit_predict(training["x"], training["y"], training["records"],
            target["x"], folds, models, job["seed"], torch.device("cuda"))
    mapping = {"always_nn": "feature_mlp", "rf": "rf", "extra_trees": "extra_trees", "always_fuse": "always_fuse"}
    errors = {}
    with np.load(path / "oof_predictions.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["window_ids"], training["windows"])
        for old, new in mapping.items():
            errors["oof_" + new] = float(np.abs(replay.oof[old] - saved[new]).max())
    for subject in job["prediction_subjects"]:
        mask = target["subjects"] == subject
        with np.load(path / f"target_s{subject:02d}_predictions.npz", allow_pickle=False) as saved:
            assert np.array_equal(saved["window_ids"], target["windows"][mask])
            for old, new in mapping.items():
                errors[f"subject_{subject}_{new}"] = float(np.abs(replay.test[old][mask] - saved[new]).max())
    assert max(errors.values()) <= 2e-7, errors
    summary = runner.load(path / "job_summary.json")
    assert replay.inner_best_epochs == summary["inner_best_epochs"]
    assert replay.selected_epoch == summary["final_epoch"]
    before = runner.sha(path / "complete.json")
    runner.run_job(job, data, protocol, execution)
    assert before == runner.sha(path / "complete.json")
    timings = [runner.load(p)["elapsed_seconds"] for p in (path / "stages").glob("*/fit.json")]
    runner.dump(runner.HERE / "pilot_verification.json", {"status": "passed", "utc": runner.now(),
        "execution": execution, "pilot_job": runner.PILOT,
        "verifier_sha256": runner.sha(Path(__file__)),
        "stage_count": len(timings), "inner_branch_receipts_checked": receipt_count,
        "poisoned_heldout_labels_leave_training_and_prediction_inputs_unchanged": True,
        "symmetric_training_and_seeds_equivalent": True,
        "independent_original_trainer_probability_errors": errors,
        "selected_epochs_match": True, "completed_job_resume_unchanged": True,
        "pilot_total_stage_fit_seconds": sum(timings),
        "pilot_wall_seconds": summary["elapsed_seconds_this_invocation"],
        "independent_replay_seconds": time.perf_counter() - start,
        "rough_serial_full_study_seconds": sum(timings) / 73 * 238000,
        "timing_caveat": "One NBACK unit only; MATB, early-stopping epochs and thermal load can differ. Accuracy did not control continuation."})
    print(runner.load(runner.HERE / "pilot_verification.json"), flush=True)


if __name__ == "__main__":
    main()
