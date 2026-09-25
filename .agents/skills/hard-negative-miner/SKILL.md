---
name: hard-negative-miner
description: Mines informative, challenging negative candidate pairs matching inference-time error distributions to train robust matching models.
---

# Hard Negative Miner

## Goal
Construct realistic, balanced, and informative negative candidate pairs from the actual blocking distribution and specific lookalike failure modes, ensuring the matching classifier learns fine-grained discriminative boundaries.

## When to Use
- Preparing training sets for pairwise ML classifiers (LightGBM, XGBoost, CatBoost).
- Augmenting baseline training data when false-positive rates on lookalikes are high.

## Core Negative Sampling Strategies
1. **Blocking False Positives (Primary Distribution)**:
   - Run the exact candidate blocker on training S1 entities.
   - Any candidate $c \in C_i$ that is NOT in ground truth $T_i$ is a natural hard negative.
   - This directly aligns training distribution with inference-time candidate distribution.
2. **High-Name Lookalikes (Same Name, Different Entity)**:
   - Pairs sharing high name similarity or exact brand name, but situated at distinct physical addresses or different cities.
3. **Same-Building Anchor Lookalikes (Shared Address, Different Business)**:
   - Pairs sharing the same building number or postal code, but representing distinct businesses (e.g., adjacent office suites, mall storefronts).
4. **Near-Identical Legal/Brand Variants**:
   - Businesses with similar keywords (e.g., `Apex Logistics` vs `Apex Healthcare`).
5. **Random Negatives (Background Prior)**:
   - Controlled sample of random pairs within the same country to anchor low-similarity baselines.

## Procedure
1. Include $100\%$ of true positive matching pairs ($y = 1$).
2. From the blocker candidate pool on training entities, sample hard negatives ($y = 0$).
3. Maintain a documented negative-to-positive ratio (e.g., 3:1, 5:1, or 10:1).
4. Tag each negative pair with its mining origin (`source_type`: `blocker_candidate`, `same_address_hard`, `brand_lookalike`, `random`).
5. Record the negative sampling parameters in `experiments/experiment_log.csv`.

## Required Outputs
- Balanced feature matrix / training dataset: `artifacts/train_pairs_<strategy_id>.parquet`.
- Sampling metadata summary logging class counts and sampling ratios.

## Failure Conditions
- Training on random negatives only (causes massive false-positive explosion at inference time).
- Dropping any true positive pair from the training set.
- Leaking validation S1 entities into the training candidate pool.
