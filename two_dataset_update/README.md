# Two-dataset EEG audit evidence

This directory contains source and cohort audits, experimental protocols,
analysis code, saved predictions, gate-response tables and statistical results.

## Layout

- `analyses/first_dataset_correction/`: corrected ordinary- and balanced-gain
  CB-SF results, BH14/BH7 comparisons, seed and participant-deletion analyses.
- `analyses/first_dataset_implementation_audit/`: diagnosis of the earlier
  signed-cutoff implementation and the scope of its effects.
- `analyses/first_dataset_balanced_gain_inputs/`: preserved inputs used to
  reconstruct the balanced-gain analysis.
- `analyses/ensemble_order_sensitivity/`: response reconstruction in final
  five-seed aggregation order, gate refits and deletion analysis, saved seed
  probabilities, BH7 statistics and a full replay verifier.
- `analyses/cogbci_session_validation/`: source checks, eligibility, fixed
  Logistic learner, matched-session comparisons and sensitivity results.
- `analyses/cogbci_gate_design/`: participant exclusions, record selection,
  input precision checks, nested job specification and frozen protocol.
- `analyses/cogbci_gate_evaluation/`: training, restart, analysis and verification
  code; completed result tables are in `analysis/`.
- `analyses/cogbci_gate_diagnostics/`: descriptive summaries of the gate results.
- `shared/revision_pipeline/`: model, split, aggregation and gate implementations.
- `shared/first_dataset/`: reconstruction helpers, saved predictions and responses.

## Verification

From this directory:

```text
python -m pip install -r requirements_verification.txt
python verify_update.py --checksums-only
python verify_update.py
python verify_update.py --full-gates
python analyses/ensemble_order_sensitivity/verify.py --full
```

The default check reconstructs recording aggregation, first-dataset BH14/BH7,
matched-session means, bootstrap intervals and BH2, and gate BH12 statistics.
The full check also replays 8,850 COG-BCI gate fits and decisions from the saved
responses and diagnostics. Verification reads the evidence without modifying it.

## Publication files and execution provenance

Publication paths, explanatory text and terminal messages have been normalized.
`SOURCE_INDEX.csv` lists each distributed source file with its original byte hash
and its publication byte hash. Original execution hashes retained inside frozen
reports identify the artifacts used during the experiments. The current package
hashes are in `MANIFEST.sha256`; they are distinct from the original execution
hashes when a file's presentation or path text has changed. Prediction CSVs
retain their original content. The aggregation-order package documents its
two exact-test p-value corrections in the repository release notes.
Verification resolves original hashes through this source index. The training
entry point also treats a successful command-line help exit as normal completion.

The published training scripts retain their execution-identity checks. Running
from raw EEG requires the official data, acquisition notebook and trigger list,
feature preparation, and a separate run directory with a newly frozen identity.
Frozen historical reports are evidence of completed runs, not configuration
files to overwrite for a new run. This compact package excludes raw EEG, model
checkpoints and intermediate feature arrays. Use `verify_update.py` to check the
distributed saved evidence; checkpoint-dependent verifiers need those inputs.

Data sources: Arithmetic/Stroop https://doi.org/10.17632/kt38js3jv7.1 and
COG-BCI v4 https://doi.org/10.5281/zenodo.7413650.
See `RESULT_FILE_MAP.md`, `RUN_ORDER.md` and `ENVIRONMENT.md` for file mapping,
computation order and software versions.

## Source-audit export

`analyses/cogbci_session_validation/export_source_audit.py` exports
`source_record_audit.csv` and `duplicate_pairs.csv` from the included verified
audit records. Its default output is a separate `recomputed/source_audit/`
directory. The distribution provides experimental computation and numerical results.
