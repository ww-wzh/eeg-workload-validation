# Reproducibility scope

## Predictions and statistics

`frozen/predictions_recording_route_a_v3.csv` has 2,660 recording-method rows,
14 methods and 190 rows per method. Its SHA-256 is
`cba069dd3837e9b3894dc2649aa6c8a9ac80812de996b2dc51cc0aa056f5c5cd`.

`matched_split_reproducibility/` reconstructs 868 split requests and verifies
2,912 recording predictions, 728 scoring blocks, 112 participant-strategy scores,
bootstrap intervals and twelve paired comparisons. It does not refit classifiers.

`split_control/` reconstructs the executed window geometry, verifies 3,160
train/test membership hashes and recomputes statistics for 1,580 fitted jobs.
Its separate raw-data runner requires the official EEG and a new output directory.

`two_dataset_update/analyses/first_dataset_correction/` contains corrected
ordinary-gain and balanced-gain responses, gate predictions and statistics.
`heldout_gate/` and `balanced_gain/` retain the earlier implementation for
comparison and must not be substituted for the corrected analysis.

`two_dataset_update/analyses/ensemble_order_sensitivity/` reconstructs responses
after the five-seed window-probability median and recording aggregation. Its
915 compressed LOSO prediction inputs and saved cross-task seed predictions
support response reconstruction and all gate refits. The verifier independently
enumerates the exact-test tails and optionally repeats the 54 full-input and
678 participant-deletion fits. Seven BH7-adjusted comparisons have q = 1.000.

COG-BCI has two designs: matched-session fixed-learner evaluation and nested
participant-disjoint gate evaluation. Their record-selection rules, exclusions,
probabilities and statistics are included. The full two-dataset verifier replays
8,850 COG-BCI gate fits from saved responses and diagnostics.

## Raw-data fitting

Raw EEG, generated feature arrays, fitted branch checkpoints and the local
training databases are separate inputs. A fresh training run needs the official
datasets, documented extraction and model dependencies, and a separate run
directory with a new execution identity. Historical execution hashes identify
completed experiments; they are not configuration files to overwrite.

## Hashes and distribution metadata

`PROVENANCE.csv` records source and current distribution hashes.
`two_dataset_update/SOURCE_INDEX.csv` retains the original execution hashes and
current file hashes required by the two-dataset verifier. Protocol descriptions
retain the actual exclusions, timing, exploratory status and parameter values.
Presentation-only metadata normalizations do not change numerical results.
Nested manifests cover their own directory scopes; `MANIFEST.sha256` covers
the entire current snapshot, excluding only itself.

Related evidence archives remain at https://doi.org/10.5281/zenodo.24328261 and
https://doi.org/10.5281/zenodo.24340688. The original datasets retain their own
versions, citations and licenses.
