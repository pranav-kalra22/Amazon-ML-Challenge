---
name: macro-f05-evaluator
description: Authoritative local evaluator for entity-level set-based Macro F0.5 score, precision, recall, and singleton performance metrics.
---

# Macro F0.5 Evaluator

## Goal
Provide the authoritative, mathematically exact local implementation of the competition metric: **Entity-Level Macro $F_{0.5}$**, including precision, recall, singleton accuracy, and country/source diagnostics.

## When to Use
- Whenever evaluating candidate blockers (oracle candidate ceiling).
- During model validation and threshold tuning on held-out splits.
- For post-experiment benchmarking against baseline models.

## Metric Definition & Singleton Rules
For each Source 1 entity $i$, let $T_i$ be the set of true matching IDs and $P_i$ be the set of predicted IDs:

- If $|T_i| = 0$ and $|P_i| = 0$: $F_{0.5}^{(i)} = 1.0$ (Correct Singleton).
- If $|T_i| = 0$ and $|P_i| > 0$: $F_{0.5}^{(i)} = 0.0$ (False Merge on Singleton).
- If $|T_i| > 0$ and $|P_i| = 0$: $F_{0.5}^{(i)} = 0.0$ (Missed Non-Singleton).
- If $|T_i| > 0$ and $|P_i| > 0$:
  $$\text{Precision}_i = \frac{|T_i \cap P_i|}{|P_i|}, \quad \text{Recall}_i = \frac{|T_i \cap P_i|}{|T_i|}$$
  $$F_{0.5}^{(i)} = \frac{(1 + 0.5^2) \cdot \text{Precision}_i \cdot \text{Recall}_i}{0.5^2 \cdot \text{Precision}_i + \text{Recall}_i} = \frac{1.25 \cdot \text{Precision}_i \cdot \text{Recall}_i}{0.25 \cdot \text{Precision}_i + \text{Recall}_i}$$
  If $\text{Precision}_i + \text{Recall}_i = 0$, $F_{0.5}^{(i)} = 0.0$.

$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}^{(i)}$$

## Required Inputs
- Ground Truth dictionary or TSV: `{source1_entity_id: set(matched_entity_ids)}`
- Predictions dictionary or TSV: `{source1_entity_id: set(predicted_entity_ids)}`
- Optional metadata mapping: `{source1_entity_id: country}` for subgroup breakdowns.

## Procedure
1. Verify that all evaluation S1 entities are present in predictions.
2. Compute per-entity Precision, Recall, and $F_{0.5}^{(i)}$.
3. Compute macro statistics across all S1 entities:
   - Macro $F_{0.5}$
   - Macro Precision
   - Macro Recall
4. Compute singleton-specific metrics:
   - Singleton Count & Frequency
   - Singleton Accuracy (fraction of true singletons predicted as empty)
   - Singleton False Positive Rate (fraction of true singletons predicted with $\ge 1$ match)
5. Compute non-singleton-specific metrics ($F_{0.5}$ on entities where $|T_i| > 0$).
6. Compute country-level breakdown where ground-truth labels exist.

## Required Outputs
- Structured evaluation dictionary / report:
  ```json
  {
    "macro_f05": 0.xxxx,
    "macro_precision": 0.xxxx,
    "macro_recall": 0.xxxx,
    "singleton_accuracy": 0.xxxx,
    "singleton_count": 0,
    "non_singleton_f05": 0.xxxx,
    "country_metrics": { ... }
  }
  ```

## Mandatory Unit Tests
Every implementation must pass the following 6 test scenarios:
1. **Perfect Multi-Match**: $T = \{A, B\}, P = \{A, B\} \implies F_{0.5} = 1.0$
2. **Extra False Positive**: $T = \{A, B\}, P = \{A, B, C\} \implies P = 2/3, R = 1.0, F_{0.5} = 0.7142857$
3. **Missing True Positive**: $T = \{A, B\}, P = \{A\} \implies P = 1.0, R = 0.5, F_{0.5} = 0.8333333$
4. **Correct Singleton**: $T = \emptyset, P = \emptyset \implies F_{0.5} = 1.0$
5. **False Singleton Merge**: $T = \emptyset, P = \{A\} \implies F_{0.5} = 0.0$
6. **Missed Non-Singleton**: $T = \{A\}, P = \emptyset \implies F_{0.5} = 0.0$

## Failure Conditions
- Using pairwise sklearn metrics (`f1_score`, `roc_auc_score`, `accuracy_score`) instead of entity-level Macro $F_{0.5}$.
- Omitting singletons from the macro average.
