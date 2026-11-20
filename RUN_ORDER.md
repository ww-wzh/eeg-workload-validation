# Computation and verification

## Saved-evidence checks

```text
python verify_release.py
python matched_split_reproducibility/reconstruct_splits.py
python matched_split_reproducibility/verify.py
python split_control/verify_evidence.py
python two_dataset_update/verify_update.py --full-gates
python two_dataset_update/analyses/ensemble_order_sensitivity/verify.py --full
```

Install the pinned verification dependencies listed in the root README.

## Experimental stages

1. `run_corrected_experiments.py`: primary first-dataset model pipeline using
   `CBSF_DATA_ROOT` and the neutral preprocessing protocol.
2. `two_dataset_update/analyses/first_dataset_correction/rebuild_corrected_results.py`:
   reconstruct corrected ordinary-gain and balanced-gain analyses from saved inputs.
3. `two_dataset_update/analyses/cogbci_session_validation/`: audit sources,
   prepare features, specify the matched-session evaluation, fit and verify.
4. `two_dataset_update/analyses/cogbci_gate_design/`: eligibility and nested design.
5. `two_dataset_update/analyses/cogbci_gate_evaluation/`: nested fitting, restart,
   response construction, gate evaluation and ablation.
6. `two_dataset_update/analyses/ensemble_order_sensitivity/run_analysis.py`:
   reconstruct responses and refit gates from included saved probabilities.

For source-audit CSV reconstruction:

```text
python two_dataset_update/analyses/cogbci_session_validation/export_source_audit.py
```

The exporter writes to a separate `recomputed/source_audit/` directory.
Raw-data fitting requires the data, intermediates and new execution identity
described in `REPRODUCIBILITY.md`; saved-evidence verification does not.
