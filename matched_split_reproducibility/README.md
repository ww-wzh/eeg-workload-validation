# Matched split reproducibility evidence

Saved evidence for the Arithmetic/Stroop fixed-learner comparison. It covers random
window 88/12, recording-disjoint, participant-disjoint and random window 50/50
evaluation: 868 fits, 2,912 recording predictions, 728 scoring blocks and 112
participant-strategy scores. The data include 15 Arithmetic and 13 Stroop
participants, four recordings per participant and task, and 19,125 windows.

## Run the checks

Use Python 3.12 in a separate environment. From this directory:

```text
python -m pip install -r requirements.txt
python reconstruct_splits.py
python verify.py
```

The pinned requirements record the original analysis environment. The supplied
checks also passed with Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.17.1
and scikit-learn 1.7.2. This verification used the saved predictions and did not
retrain models in that environment.

The first script generates all split memberships from `recordings.csv` and
the fixed random seeds, then compares every index with `window_indices.npz`
and the executed request hashes. It needs no EEG samples or model files.
The second script also verifies the package manifest and recalculates metrics
from the saved recording probabilities: block metrics, participant averages,
88 metric summaries with bootstrap intervals, twelve paired comparisons,
Wilcoxon and exact sign-flip tests, both BH correction scopes and the eight
displayed split-summary rows. Seeds and scoring blocks are averaged within
participant before statistical inference.

To save reconstructed outputs, choose paths that do not already exist:

```text
python reconstruct_splits.py --output reconstructed/window_indices.npz
python verify.py --output-dir recomputed_statistics
```

## Files and indexing

| File | Contents |
|---|---|
| `recordings.csv` | Task-specific recording order, participant, workload label, first window row and window count |
| `window_indices.npz` | Executed train/test index arrays for every fit and row-to-recording mappings |
| `fold_requests.json` | Executed fit requests, recording IDs, class labels and index/transform hashes |
| `fold_diagnostics.csv` | Sample counts, shared raw-sample diagnostics, convergence and training-label hashes |
| `recording_predictions.csv` | Saved binary class probabilities and decisions for each evaluated recording |
| `block_metrics.csv` | Metrics for the 728 four-recording scoring blocks |
| `participant_metrics.csv` | Within-participant averages for the four strategies |
| `summary_intervals.csv` | Participant-macro estimates and 95% bootstrap intervals |
| `comparisons.csv` | All twelve paired contrasts and multiple-testing corrections |
| `participant_differences.csv` | The 168 participant differences underlying the contrasts |
| `split_summary_display.csv` | Rounded balanced-accuracy estimates and intervals |
| `protocol.json` | Split, learning, aggregation and statistical settings |
| `execution_provenance.json` | Original execution metadata and source-to-package byte hashes |
| `MANIFEST.sha256` | Checksums of the distributed files |

Window rows are zero-based within task and are concatenated in the recording
order in `recordings.csv`. Within a recording, window k starts at sample 125k
of the preprocessed record and has 2,125 samples at 250 Hz. NPZ keys ending in
`__train` or `__test` contain those task-global row indices; the prefix is the
fit key in the request and prediction files. The other keys identify recording
IDs and the row-to-recording mapping. Array hashes cover the NumPy dtype,
shape and contiguous bytes, as implemented in `reconstruct_splits.py`.

## Reproducibility scope

This package reconstructs the executed splits and verifies the statistical
results from recording predictions. It does not refit the EEG classifier or
recreate probabilities from raw signals. Feature arrays, fitted estimators,
window-level probabilities and the training checkpoint databases are not
distributed here. Their execution hashes in the requests support provenance;
they are not a substitute for the missing numerical inputs. Repeating model
fitting requires the original EEG, the feature extraction implementation and
its dependencies, and the training settings recorded in `protocol.json`.

Raw EEG remains at https://doi.org/10.17632/kt38js3jv7.1. The existing code and
summary archive is https://doi.org/10.5281/zenodo.24328261. This package
supplies the corresponding split indices and detailed predictions. It contains no raw EEG. Code is covered by the accompanying MIT license; the original data retain
their own license. The copied numerical files and index archive retain their
original bytes. Public reconstruction scripts were written to verify those
saved artifacts and are not presented as the original training executables.
