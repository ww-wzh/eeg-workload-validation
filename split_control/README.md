# Fixed-learner split controls

The saved experiment comprises 1,580 fits and 5,800 recording predictions.
`planned_folds.jsonl` specifies every raw-window start, class and filtering
interval. `executed_membership_hashes.csv` links each plan to the original
execution, its train/test membership digests and checkpoint hashes. Raw samples,
feature arrays and model checkpoints are obtained or regenerated separately.

## Verify the distributed results

From the repository root, use the verification environment documented in
`../two_dataset_update/requirements_verification.txt` and run:

```text
python split_control/verify_evidence.py
```

This rebuilds the frozen plan, checks both membership digests for all 1,580
fits, checks the saved prediction structure, and independently recomputes
recording-to-participant aggregation, intervals, exact sign-flip tests and BH8.
It uses the saved participant bootstrap matrices and verifies their deterministic
regeneration. It performs no model fitting and needs no raw EEG or GPU.

For a separate reconstruction of all descriptive outputs and statistics:

```text
python split_control/analyze_split_controls.py --output split_statistics_recomputed
```

CSV output precision is retained; rounded summaries use decimal round-half-up.
Different panels use their own feasibility cohorts: Arithmetic excludes S07 in
both panels; Stroop excludes S04 only in the temporal panel.

## Refit from raw EEG

Obtain the Arithmetic/Stroop version-1 data from
https://doi.org/10.17632/kt38js3jv7.1 and provide a directory containing
`Arithmetic_Data/` and `Stroop_Data/`. `raw_input_identity.csv` gives dataset
relative filenames, complete-file hashes and sample counts.

```text
python -m pip install -r split_control/requirements_training.txt
python split_control/run_from_raw.py --raw-root RAW_DATA_DIRECTORY --output split_refit --device cpu
```

The portable runner retains the original feature, fitting and aggregation
function bodies. It rebuilds features from raw filtering units, uses the exact
frozen window plans, and writes a new execution identity into a separate output
directory. It supports checksum-validated restart. `--task`, `--subject`,
`--seed` and `--limit` select a technical subset and must remain unchanged when
resuming an output directory. A stale `run.lock` requires checking that the
earlier process has ended before removing that lock.

The publication export uses new path handling and therefore has its own source
hashes. Original execution-source hashes and the original independent verifier
reports are preserved in `execution_provenance.json` and the two
`independent_*_verification.json` files. Those reports describe the historical
checks against the full local checkpoints; they are separate from the compact
saved-evidence verification above. Frozen protocol fields describe the scope at
the time these controls were designed.
