# Reproducibility additions

These additions accompany the revision after the V1.2 archive. They do not
change its frozen branch predictions, cohort definitions, or BH12 tests.
The original release is V1.2, archived at https://doi.org/10.5281/zenodo.24434545.
The new fixed-weight comparison is post hoc and descriptive.

## Inputs and result locations

The current manuscript includes a new evaluation-design table, so the former
Tables 2, 7, and 8 are now Tables 3, 8, and 9. No result was renumbered in the
V1.2 archive itself.

| Current item | Input in V1.2 | Verification from repository root |
| --- | --- | --- |
| Table 3 and Figure 2 | `matched_split_reproducibility/recording_predictions.csv`, `recordings.csv`, `window_indices.npz`, `protocol.json` | `python matched_split_reproducibility/verify.py` |
| Table 8 and Figure 4 | `two_dataset_update/analyses/cogbci_session_validation/fold_scores.csv`, `participant_scores.csv`, `summary.csv`; `results/cogbci/paired_participant_differences.csv` | `python two_dataset_update/verify_update.py` |
| Original Table 9 rows and Table S31 | `two_dataset_update/analyses/cogbci_gate_evaluation/analysis/participant_scores.csv`, `method_summary.csv`, `comparisons_BH12.csv` | `python two_dataset_update/verify_update.py --full-gates` |
| New fixed 0.4 row and Table S42 | Original outer-fold window predictions in Supplementary Data S1 | Fixed-weight command below |
| Section 2.2.4 and Section S19 spacing | `pair_plan.json`, `feature_audit.json`, `protocol.json` and the archived sampling function | Single-pair command below also rebuilds sampling metadata without additional fits |

## Fixed weight reconstruction

Requires Python, NumPy, and pandas. Run from the repository root:

```powershell
python reproducibility_update/recompute_fixed_lambda04.py --input-zip "supplementary-data 50/Supplementary_Data_S1_COGBCI_sensitivities.zip" --out fixed04_check
```

This uses the median probability across five seeds at each window, normalizes
each branch, blends 0.6 times Feature-MLP with 0.4 times the fused branch, and
geometrically aggregates inside each recording. It checks the original
Feature-MLP, always-fuse, fixed 0.5, and held-out-response CB-SF predictions
before reporting the new comparison. Bootstrap intervals use 10,000 whole-
participant draws with seed 20261110. No new p values or BH tests are added.
Validated with Python 3.12.14, NumPy 2.3.5, and pandas 3.0.1.

## Minimal raw EEG example

Download `sub-02.zip` from COG-BCI Version 4 (https://doi.org/10.5281/zenodo.7413650)
under the original dataset terms. Do not put the raw archive in this repository.
Use the original experimental environment described in `two_dataset_update/ENVIRONMENT.md`
and the root `requirements.txt`. Run:

```powershell
python reproducibility_update/reproduce_single_pair_smoke.py --archive "D:/data/cogbci/sub-02.zip" --out single_pair_check
```

Only N-back participant 02, source session 2, target session 3, and seed 1335
are fitted. The script verifies source headers and FDT hashes, rebuilds the
valid support features, selects 10 training and 10 test windows per class,
and fits the two class-balanced logistic models. It explicitly checks that
both arms have identical test supports and features. Training and test supports
within a target recording are disjoint. The output contains window probabilities,
record probabilities and decisions, input/code hashes, and environment versions.
The separate sampling metadata audit reconstructs all primary test-window
indices but fits no additional models.

If original local feature/fit caches are available, `--reference-cache PATH`
also checks feature and probability agreement with them. This option is not
required for public reproduction. The supplied verification used this option:
maximum feature difference 0 and maximum prediction difference 0. Raw EEG and
feature caches are not redistributed. These single-pair scores are an engineering
check, not estimates of cohort performance.

Validated with Python 3.12.13, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0,
and PyTorch 2.9.0+cu129; feature extraction and logistic fitting ran on CPU.

## Output policy

Both scripts require a new or empty output directory. They neither replace
historical files nor access the network. `results/` contains the verified
outputs supplied with this addition. The fixed 0.4 contrast is separate from
the prespecified fixed 0.5 comparison in BH12. No plotting or manuscript-
generation scripts are included.
