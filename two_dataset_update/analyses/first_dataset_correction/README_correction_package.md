# First-dataset CB-SF correction

The held-out-gain reconstruction used a negative loss threshold as an argument
to a function that negated it, producing the event `gain <= +0.05`. The corrected
implementation uses `gain <= -0.05`. The reconstruction also corrects cross-task
participant deletion, direction-aware seed-response lookup and the exact
two-sided sign-flip tail calculation. EEG base learners were not retrained.
The primary 14-method predictions were replayed and remain unchanged.

Cross-task gate training uses 24 rows from 12 people. Deleting one participant
removes both direction rows, leaving 22 rows from 11 people. Seed-response lookup
uses participant, direction and seed. BH14, BH7 and the combined BH79 sensitivity
families follow the corresponding frozen analysis definitions.

## Files

- `corrected_gate.py`: explicit signed-cutoff implementation.
- `rebuild_corrected_results.py`: reconstruction from saved inputs under
  `../../shared/first_dataset/` and `../first_dataset_balanced_gain_inputs/`.
- `verify_results.py`: independent gate, aggregation and statistical checks.
- `ordinary_heldout/`: ordinary-gain predictions, responses, metrics, comparisons
  and seed/participant-deletion analyses.
- `balanced_heldout/`: endpoint-aligned balanced-gain analysis and sensitivity.
- `correction_verification.json`, `independent_verification.json`: recorded
  reconstruction and verification results with original execution hashes.
- `original_proxy_gate_replay.csv`: replay of the original internal-proxy gates.

The sign-only diagnostic changed 107 of 108 earlier held-out CB-SF weights and
seven target decisions across the two gain definitions. The complete correction
changed three ordinary-gain and four balanced-gain recording decisions. The
ordinary-gain CB-SF scores were 55.00% for Arithmetic LOSO and 57.69% for Stroop
LOSO. No reconstructed contrast passed its BH-adjusted comparison family.

The portable package check is `../../verify_update.py`. Its source index maps
original execution hashes to publication files. The repository's `heldout_gate/`
and `balanced_gain/` directories preserve the preceding implementation's outputs
for comparison. The corrected results are in this directory.
Archive provenance is described in the root `REPRODUCIBILITY.md`.
