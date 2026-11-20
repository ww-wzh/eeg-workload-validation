# Five-seed aggregation-order sensitivity

This analysis tests the effect of defining CB-SF training responses in the same
aggregation order as final five-seed predictions.

For each branch, the five seed probabilities are combined by the component-wise
median at each window, normalized, then geometrically pooled within recordings.
The response is the fused branch's participant recording balanced accuracy minus
the neural branch's score. Loss events use gain <= -0.05. The five label-free
diagnostics, exclusion rules, fitted branch predictions and gate constants stay
fixed. DASF is retained as the unchanged comparator.

## Files

- `inputs/meta/`: 915 losslessly compressed prediction CSVs, comprising five
  seeds for each of 105 Arithmetic and 78 Stroop participant pairs.
- `inputs/meta_index.json`: relative paths, compressed and original byte hashes.
- `inputs/seed_predictions.csv.gz`: the saved seed prediction table used to
  reconstruct cross-task responses. It contains probabilities, not raw EEG.
- `results/`: response and training tables, gate diagnostics, window and recording
  predictions, participant metrics, deletion analysis and BH7 comparisons.
- `execution_provenance.json`: original execution hashes and publication changes.
- `run_analysis.py`: reproducible response reconstruction and gate refitting.
- `verify.py`: file checks, independent statistical checks and optional full replay.

The comparison responses and label-free diagnostics are read from the existing
`../first_dataset_correction/balanced_heldout/` evidence. Shared recording labels,
final window ensembles and aggregation helpers are already included in the
repository. Base models are not fitted again.

## Commands

From the repository root, install the pinned verification requirements, then run:

```text
python -m pip install -r two_dataset_update/requirements_verification.txt
python two_dataset_update/analyses/ensemble_order_sensitivity/verify.py
python two_dataset_update/analyses/ensemble_order_sensitivity/verify.py --full
```

The full check reconstructs all responses and repeats the 54 full-input and
678 participant-deletion gate fits in a temporary directory. It compares every
regenerated CSV to the released reference within numerical tolerance.

To retain a new computation in a separate directory:

```text
python two_dataset_update/analyses/ensemble_order_sensitivity/run_analysis.py --output-dir /path/to/new-results
```

## Interpretation and statistics

Changing aggregation order alters 67 of 366 LOSO responses and 6 of 26 cross-task
responses, 52 of 54 gate weights and four recording classifications. The seven
participant-level comparisons have BH7 q = 1.000. Bootstrap intervals condition
on the saved model fits. The bidirectional comparison averages the two task
directions within each participant before inference.

Exact two-sided sign-flip tests enumerate all signs of the nonzero participant
differences and count absolute sums at least as extreme as the observed sum.
The exact p-values are 1.000 for Stroop-to-Arithmetic and 0.750 for
the bidirectional contrast; all seven BH7 q-values remain 1.000.
