---
name: threshold-optimizer
description: Optimizes decision thresholds, singleton guards, and post-processing policies directly for entity-level Macro F0.5.
---

# Threshold Optimizer

## Goal
Transform continuous model prediction probabilities $P(\text{match} \mid S_1, C)$ into discrete entity match sets that maximize the official **Entity-Level Macro $F_{0.5}$** metric on the validation split.

## When to Use
- After training or scoring any matching model.
- Before generating candidate predictions for evaluation or leaderboard submission.

## Why 0.5 is Not Optimal
Macro $F_{0.5}$ weights precision twice as heavily as recall:
- Merging two distinct entities (False Positive) imposes a catastrophic penalty.
- Singletons (5.6% of data) score 0.0 upon a single false merge.
- The default binary classification threshold ($\tau = 0.50$) is almost always too lenient for high-precision ER. Optimal thresholds typically lie significantly higher ($\tau \in [0.65, 0.90]$).

## Thresholding Policies & Search Space

1. **Global Static Threshold ($\tau$)**:
   - Single cutoff: predict match if $\hat{p} \ge \tau$.
2. **Source-Specific Thresholds ($\tau_{S2}, \tau_{S3}$)**:
   - Independent cutoffs for Source 2 and Source 3 candidates.
3. **Singleton Guard Threshold ($\tau_{\text{singleton}}$)**:
   - If the maximum candidate probability for an entity $\max_{c} \hat{p}(c) < \tau_{\text{singleton}}$, predict an empty match set $P_i = \emptyset$.
   - Directly protects the 1.0 reward for true singletons.
4. **Relative Margin / Score Gap Filter**:
   - Only admit additional candidates if $\hat{p}(c) \ge \max_{c'} \hat{p}(c') - \Delta_{\text{margin}}$.
5. **Exact Match Override**:
   - Deterministic rules for near-certain matches (e.g., exact brand + exact building number $\rightarrow$ auto-accept).

## Procedure
1. Load out-of-fold validation predictions or validation set pair probabilities.
2. Define a multi-dimensional grid search over candidate parameters:
   - $\tau_{\text{global}} \in [0.50, 0.95]$ (step 0.02)
   - $\tau_{\text{singleton}} \in [0.50, 0.95]$
   - Optional source-specific $\tau_{S2}, \tau_{S3}$
3. For each parameter combination, construct the entity-level prediction sets $\{S_1: \{c_1, \dots\}\}$.
4. Score the entire validation set using `macro-f05-evaluator`.
5. Select the configuration that strictly maximizes validation Macro $F_{0.5}$.
6. Save the optimal threshold configuration to `configs/optimal_thresholds.json`.

## Required Outputs
- `configs/optimal_thresholds.json`: Saved threshold parameters.
- Optimization trace plot or table logged in `reports/thresholds/`.

## Failure Conditions
- Optimizing threshold on pairwise $F_1$ or ROC-AUC rather than Macro $F_{0.5}$.
- Assuming $\tau = 0.5$ without validation search.
- Tuning thresholds directly on the test set.
