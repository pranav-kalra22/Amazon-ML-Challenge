---
name: experiment-ledger
description: Tracks, versions, and logs all machine learning experiments, blocker configurations, hyperparameters, and evaluation metrics in a permanent ledger.
---

# Experiment Ledger

## Goal
Maintain a permanent, append-only historical record of all experiments, blocker configurations, feature sets, models, thresholds, and validation metrics in `experiments/experiment_log.csv`.

## When to Use
- Before launching a new experimental training run.
- Immediately after evaluating any model or blocker on the validation split.
- When promoting a new model configuration to "current best".

## Ledger File Location & Schema
The primary ledger is located at: `experiments/experiment_log.csv`

### Required Columns:
1. `experiment_id`: Unique identifier (e.g., `EXP_001_baseline_lgbm`).
2. `timestamp`: ISO-8601 UTC timestamp of completion.
3. `code_version`: Git commit SHA or clean code hash.
4. `validation_split`: Identification of validation split (e.g., `val_split_100k_s1_seed42`).
5. `blocker_config`: Summary or path of candidate blocker configuration.
6. `feature_config`: Summary or path of feature extraction pipeline.
7. `negative_sampling`: Hard negative mining ratio and strategy.
8. `model`: Model architecture name (e.g., `LightGBM`, `XGBoost`).
9. `hyperparameters`: Serialized JSON string of model hyperparameters.
10. `threshold_config`: Optimal threshold values ($\tau, \tau_{\text{singleton}}$).
11. `candidate_recall`: Blocker link recall on validation split.
12. `full_entity_coverage`: Percentage of entities with 100% true links retrieved.
13. `oracle_f05`: Upper bound F0.5 achievable from candidate pool.
14. `validation_macro_f05`: Primary competition metric on validation set.
15. `singleton_accuracy`: Percentage of true singletons correctly predicted empty.
16. `precision`: Entity-level macro precision.
17. `recall`: Entity-level macro recall.
18. `runtime_sec`: Total end-to-end execution runtime in seconds.
19. `memory_peak_mb`: Peak memory usage observed.
20. `is_current_best`: Boolean flag (`True` / `False`).
21. `notes`: Key hypotheses tested or observations.
22. `artifact_paths`: Paths to saved model weights, configs, and error reports.

## Procedure
1. Initialize `experiments/experiment_log.csv` with standard headers if absent.
2. Ensure previous rows are NEVER overwritten or deleted.
3. If an experiment surpasses the previous best validation Macro $F_{0.5}$, update the metadata tracker `experiments/best_experiment.json`.
4. Store trained model weights under `artifacts/<experiment_id>/`.

## Invariant Rules
- Every logged metric must be directly reproducible via saved code and config.
- Never overwrite previous experiment rows.
