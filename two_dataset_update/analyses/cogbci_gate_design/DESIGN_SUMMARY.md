# COG-BCI nested gate design

The design and input preparation were frozen on 2026-11-10 before new gate-study
model fitting. The completed results are in `../cogbci_gate_evaluation/analysis/`.

| Task | Participants | Sessions | Condition records | Windows |
|---|---:|---:|---:|---:|
| N-back | 25 | 63 | 126 | 2520 |
| MATB | 25 | 67 | 134 | 2680 |

Participant labels 17, 25, 27 and 28 were excluded because of source attribution
ambiguity. Interrupted sessions were excluded. N-back uses the selected 2.008 s
stimulus interval; participant 01 contributes session 3. Each record contributes
20 windows with separate raw-sample and filtering supports.

For each target participant, models and model selection exclude that participant.
An additional participant is excluded to obtain each independent gate-training
response. Each final gate uses 24 participant responses. The nested design has
250 outer and 3,000 unordered paired-exclusion model ensembles. Repeated fits are
computational units; inference uses the 25 participants.

Feature-MLP, Random Forest and Extra Trees supply the branches and heterogeneous
stacker. Comparisons include the neural branch, always-fused and 50% mixtures,
internal-proxy and independent-held-out DASF/CB-SF, and a response-only CB-SF
bridge. Six primary contrasts per task form the BH12 family. Recording balanced
accuracy is aggregated at participant level.

Second-moment matrices use float64 accumulation from the same float32 filtered
samples. Reconstructed features matched the previous cache with maximum absolute
difference zero. Matrices were exactly symmetric; the maximum relative Frobenius
difference from float32 accumulation was 3.08374241e-07. The float32 preparation
attempt is retained as historical input-precision evidence.

The loss event is `gain <= -0.05`. Binary input labels use an explicit 0/1 mapping.
DASF averages meta-participant scores with equal participant weights. Full record
selection, exclusions, protocol and preparation checks are provided in the JSON
and CSV files in this directory.
