# Supporting audits for Arithmetic/Stroop

These files contain source, label, preprocessing, split and numerical audits. CSV files preserve the original bytes and numerical precision.
Original and distributed hashes are recorded in `../PROVENANCE.csv`.

| Files | Analysis scope |
|---|---|
| `raw_file_audit.csv`, `raw_duration_summary.csv`, `raw_duplicate_groups.csv` | All 120 deposited files, 104 distinct complete-file hashes and 16 duplicate pairs |
| `label_mapping_sensitivity.csv` | Earlier-preprocessing label sensitivity; select recording balanced accuracy |
| `preprocessing_sensitivity.csv` | Contrasts against that analysis's own earlier-preprocessing reference |
| `split_fold_overlap_audit.csv` | Fold membership, overlap and convergence for executed folds |
| `riemannian_*` | Numerical convergence checks and preservation of original recording decisions |
| `original_gate_event_support.csv` | Original proxy-response event counts and single-class fallbacks |

The raw-file audit preceded the corrected packet-counter parser. Its parser
diagnostic columns describe that historical audit; the complete-file hashes,
sample counts and duplicate groups remain applicable. The EEG models use the
eight columns after the packet counter, as specified by the current protocol.

Label and preprocessing sensitivity use earlier preprocessing. Event support
describes original proxy supervision.
Current corrected held-out-gain results are under
`../two_dataset_update/analyses/first_dataset_correction/`.
