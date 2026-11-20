# EEG workload validation

Experimental code, protocols, source audits, saved predictions and numerical
results for validation boundaries and selective fusion in Arithmetic/Stroop
and COG-BCI EEG workload data.

## Contents

| Directory | Contents |
|---|---|
| `frozen/` | Canonical 2,660 recording-method predictions for 14 methods and the neutral preprocessing protocol |
| `results/primary/` | Cohort summaries, method scores, split comparisons, calibration, permutation and stability results |
| `results/cogbci/` | Paired participant differences from the matched-session experiment |
| `first_dataset_audit/` | File hashes, duplicate groups, label and preprocessing sensitivity, numerical diagnostics |
| `matched_split_reproducibility/` | 868 fit requests, train/test indices, 2,912 recording predictions and statistical reconstruction |
| `split_control/` | Temporal, local-filter and non-overlap controls; 1,580 fits, saved predictions and raw-data runner |
| `two_dataset_update/analyses/first_dataset_correction/` | Corrected ordinary-gain and balanced-gain CB-SF results |
| `two_dataset_update/analyses/ensemble_order_sensitivity/` | Five-seed response aggregation, gate refitting and participant-deletion analysis |
| `two_dataset_update/analyses/cogbci_session_validation/` | Source audit and matched-session fixed-learner evaluation |
| `two_dataset_update/analyses/cogbci_gate_design/` | Eligibility, nested exclusions and training specifications |
| `two_dataset_update/analyses/cogbci_gate_evaluation/` | Nested model training, gate comparisons, bridges and ablations |
| `revision_pipeline/`, `route_a/`, `two_dataset_update/shared/` | Shared preprocessing, model, split and fusion implementations |
| `heldout_gate/`, `balanced_gain/` | Earlier gate implementations retained as experimental comparators |

## Verify saved evidence

Use Python 3.12 and the pinned CPU verification dependencies:

```text
python -m pip install -r two_dataset_update/requirements_verification.txt
python verify_release.py
python matched_split_reproducibility/reconstruct_splits.py
python matched_split_reproducibility/verify.py
python split_control/verify_evidence.py
python two_dataset_update/verify_update.py --full-gates
python two_dataset_update/analyses/ensemble_order_sensitivity/verify.py --full
```

These checks use included probabilities, indices and response tables. Raw EEG,
GPU hardware and base-model retraining are unnecessary for these checks.
`RUN_ORDER.md` lists computation stages; `REPRODUCIBILITY.md` states which
inputs are needed for new fits. `RESULTS_INDEX.csv` indexes the experiment groups.

## Source data and provenance

- Arithmetic/Stroop version 1: https://doi.org/10.17632/kt38js3jv7.1
- COG-BCI version 4: https://doi.org/10.5281/zenodo.7413650

Obtain raw EEG from the original repositories under their licenses.
Code uses the MIT License. Related experiment evidence was previously archived
at https://doi.org/10.5281/zenodo.24328261 and
https://doi.org/10.5281/zenodo.24340688. This distribution retains scientific
source hashes and the corrections recorded with each analysis.

`PROVENANCE.csv` maps distributed files to their prior-distribution hashes.
`MANIFEST.sha256` covers this snapshot. Historical execution hashes inside
protocols and reports identify the actual experiment files; the source index
records normalized distribution hashes separately.

## Statistical scope

Participants are the inference and bootstrap units. Windows, seeds, repeated
fits and task directions are dependent observations. Gain reconstruction and
split-control analyses are exploratory; bootstrap intervals condition on saved
model fits. Existing results do not establish a consistent fusion advantage.
