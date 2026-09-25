# Candidate Generation Benchmark Report — `EXP_002_corrected_lexical_blocker`

**Date:** 2026-09-25T16:14:31Z  
**Validation Split:** `val_50k_seed42` (50,000 Source 1 Entities)  
**Code Commit:** `5af2d35` (git_dirty: `False`)  
**Authoritative Config:** `configs/blocking/blocking_v02.yaml`  
**Candidate Search Space:** 10,320,219 Training S2 + S3 Records  
**Total Runtime:** 1283.01s | **Peak RAM:** 4629.47 MB (Initial: 84.35 MB, Final: 4629.47 MB)  

## 1. Primary Ceiling Metrics `[VALIDATION MEASUREMENT]`

| Metric | EXP_001 Baseline | EXP_002 Corrected | Delta | Target |
| :--- | :--- | :--- | :--- | :--- |
| **True Link Recall** | 83.390% | **57.566%** (99,434 / 172,729) | **-25.824%** | $\ge 98.0\%$ |
| **Full Entity Coverage** | 62.030% | **29.696%** (14,019 / 47,208) | **-32.334%** | Highest possible |
| **Oracle Macro $F_{0.5}$** | 0.925756 | **0.736050** | **-0.189706** | $\ge 0.985884$ |
| **Mean Candidates/S1** | 502.39 | **464.58** | -37.81 | Manageable volume |
| **Total Candidate Pairs** | 25,119,679 | 23,228,850 | -1,890,829 | Scalable volume |
| **Reduction Ratio** | 99.995132% | **99.995498%** | — | $> 99.99\%$ |

## 2. Candidate Volume Distribution per Entity

| Statistic | Value |
| :--- | :--- |
| Mean | 464.58 |
| Median (p50) | 62.0 |
| p90 | 1500.0 |
| p95 | 1500.0 |
| p99 | 1500.0 |
| Maximum | 1,500 |

## 3. Per-Channel Recall & Incremental Contribution

| Channel | Description | Link Recall | True Links Retrieved | Unique Links Added | Avg Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Exact Normalized Name | 39.69% | 68,553 | **+68,553** | 65.2 |
| **B** | Compact & Domain-Normalized | 41.98% | 72,510 | **+4,676** | 65.8 |
| **C2** | Candidate-DF-Aware Rare Token | 0.44% | 756 | **+230** | 0.0 |
| **D2** | True Address-Only Rescue | 14.04% | 24,250 | **+12,772** | 2.8 |
| **E2** | Symmetric Transliteration | 50.06% | 86,460 | **+13,198** | 457.3 |
| **F** | Order-Invariant Token Pairs | 0.03% | 59 | **+5** | 0.0 |
| **UNION** | **Channels A + B + C2 + D2 + E2 + F** | **57.566%** | **99,434** | **99,434** | **464.6** |

## 4. Deterministic Top-K Candidate Caps Benchmark

> **NOTE:** Candidates deterministically ranked by composite evidence score descending, entity ID ascending.

| Top-K Cap | Link Recall | Full Entity Coverage | Oracle Macro $F_{0.5}$ |
| :--- | :--- | :--- | :--- |
| `Top Uncapped` | 57.566% | 29.696% | **0.736050** |
| `Top 500` | 54.592% | 27.260% | **0.708837** |
| `Top 250` | 53.255% | 26.125% | **0.695850** |
| `Top 100` | 50.757% | 23.407% | **0.675658** |
| `Top 50` | 47.331% | 20.903% | **0.643396** |
| `Top 30` | 45.073% | 19.334% | **0.620002** |

## 5. Miss Analysis Breakdown (Measurable Recoverability Diagnostics)

- **Total Missed Links:** 73,295 (42.434% miss rate)
- **Miss Analysis CSV:** `reports/blocking\EXP_002_corrected_lexical_blocker_blocking_misses.csv`

| Heuristic Miss Category | Count | Percentage | Primary Observation | Proposed Recovery Channel |
| :--- | :--- | :--- | :--- | :--- |
| `address_shared_unindexed` | 55,682 | 75.97% | Measured in CSV | See recoverability analysis |
| `complex_transliteration_variant` | 14,440 | 19.70% | Measured in CSV | See recoverability analysis |
| `missing_address_candidate` | 2,942 | 4.01% | Measured in CSV | See recoverability analysis |
| `name_token_overlap_unindexed` | 211 | 0.29% | Measured in CSV | See recoverability analysis |
| `severe_alias_or_dba` | 20 | 0.03% | Measured in CSV | See recoverability analysis |

## 6. Subgroup Diagnostics

### Country Breakdown

- **US** (29,990 entities): Oracle Macro $F_{0.5} = 0.750478$, Precision = 0.830177, Recall = 0.615329
- **India** (20,010 entities): Oracle Macro $F_{0.5} = 0.714426$, Precision = 0.803698, Recall = 0.571121

### Source Diagnostics

- **Source 2 Links**: Retrieved = 51,492 / 83,561 (61.62% recall)
- **Source 3 Links**: Retrieved = 47,942 / 89,168 (53.77% recall)
