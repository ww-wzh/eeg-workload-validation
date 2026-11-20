# Computation order

Use `python verify_update.py --full-gates` for the portable saved-evidence check.

The original analysis stages were:

1. `analyses/first_dataset_correction/`: signed-cutoff implementation,
   reconstruction and independent checks.
2. `analyses/cogbci_session_validation/`: input audit, protocol freeze, feature
   preparation, provenance verification, analysis freeze, fitting and statistics.
3. `analyses/cogbci_gate_design/`: cohort, nested exclusions, input precision
   checks and protocol freeze.
4. `analyses/cogbci_gate_evaluation/run_nested.py`: nested model training with
   stage-level restart checks.
5. `analyses/cogbci_gate_evaluation/analyze_gates.py` and `verify_results.py`:
   response construction, gate evaluation, statistics and independent verification.
6. `analyses/cogbci_gate_diagnostics/review_results.py`: descriptive summaries.

Training from raw EEG requires additional data and generated intermediates.
The published historical execution identity is not interchangeable with a new
run's identity. See `README.md` for source-hash and training requirements.
