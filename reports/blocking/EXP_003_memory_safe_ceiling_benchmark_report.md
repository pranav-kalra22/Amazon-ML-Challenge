# Candidate Generation Benchmark Report — `EXP_003_memory_safe_ceiling`

**Date:** 2026-09-26T05:21:41Z  
**Validation Split:** `val_50k_seed42` (50,000 Source 1 Entities)  
**Code Commit:** `697d5e1` (git_dirty: `False`)  
**Authoritative Config:** `configs/blocking/blocking_v03.yaml`  
**Candidate Search Space:** 10,320,219 Training S2 + S3 Records  
**Total Runtime:** 1586.49s | **Peak RAM:** 1214.77 MB (Initial: 84.49 MB, Final: 1209.98 MB)  

## 1. True Unbounded Candidate Ceiling (Mode A) `[VALIDATION MEASUREMENT]`

> **NOTE:** Measured with zero candidate storage caps or streaming truncations. Reflects true mathematical upper bound of the blocking rules.

| Metric | EXP_001 Baseline | EXP_002 (Capped at 1500) | EXP_003 True Unbounded Ceiling | Delta vs EXP_001 | Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **True Link Recall** | 83.390% | 57.566% | **64.924%** (112,143 / 172,729) | **-18.466%** | $\ge 98.0\%$ |
| **Full Entity Coverage** | 62.030% | 29.696% | **30.486%** (14,392 / 47,208) | **-31.544%** | Highest possible |
| **Oracle Macro $F_{0.5}$** | 0.925756 | 0.736050 | **0.831634** | **-0.094122** | $\ge 0.985884$ |
| **Mean Candidates/S1** | 502.39 | 464.58 | **104.44** | -397.95 | Manageable volume |
| **Total Candidate Pairs** | 25,119,679 | 23,228,850 | 5,222,232 | -19,897,447 | Scalable volume |
| **Reduction Ratio** | 99.995132% | 99.995498% | **99.998988%** | — | $> 99.99\%$ |

## 2. Mode B — Ranked Production Candidates Benchmark `[VALIDATION MEASUREMENT]`

> **NOTE:** Evaluated across deterministic Top-K candidate caps using multi-channel evidence ranking.

| Top-K Cap | Link Recall | Full Entity Coverage | Oracle Macro $F_{0.5}$ | Mean Retained Candidates |
| :--- | :--- | :--- | :--- | :--- |
| `K = 2000` | **64.701%** | 30.253% | **0.830255** | 96.9 |
| `K = 1000` | **64.218%** | 29.864% | **0.826354** | 84.1 |
| `K = 500` | **62.942%** | 28.963% | **0.815007** | 60.4 |
| `K = 250` | **62.011%** | 28.474% | **0.804819** | 44.3 |
| `K = 100` | **59.680%** | 26.758% | **0.783284** | 29.7 |

## 3. EXP_002 Miss Reconsideration (Cap Impact Analysis)

- **EXP_002 Reported Misses:** 73,295
- **STREAMING_CAP_MISS:** 19,137 (26.11%) — true candidates matched blocker rules but were dropped by the 1,500 streaming cap.
- **BLOCKING_RULE_MISS:** 54,158 (73.89%) — genuine failure of lexical blocking rules.

## 4. Per-Channel Recall & Incremental Contribution (Mode A)

| Channel | Description | Link Recall | True Links Retrieved | Unique Links Added | Avg Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Exact Normalized Name | 47.53% | 82,104 | **+82,104** | 90.8 |
| **B** | Compact & Domain-Normalized | 50.33% | 86,928 | **+5,678** | 89.3 |
| **C2** | Candidate-DF-Aware Rare Token | 0.47% | 806 | **+245** | 0.0 |
| **D2** | True Address-Only Rescue | 17.14% | 29,598 | **+15,902** | 3.5 |
| **E2** | Symmetric Transliteration | 54.40% | 93,958 | **+8,207** | 94.6 |
| **F** | Order-Invariant Token Pairs | 0.04% | 61 | **+7** | 0.0 |
| **UNION** | **Channels A + B + C2 + D2 + E2 + F** | **64.924%** | **112,143** | **112,143** | **104.4** |

## 5. Subgroup Diagnostics (Mode A)

### Country Breakdown

- **US** (29,990 entities): Oracle Macro $F_{0.5} = 0.862490$, Precision = 0.944815, Recall = 0.708793
- **India** (20,010 entities): Oracle Macro $F_{0.5} = 0.785388$, Precision = 0.891804, Recall = 0.607158

### Source Diagnostics

- **Source 2 Links**: Retrieved = 55,194 / 83,561 (66.05% recall)
- **Source 3 Links**: Retrieved = 56,949 / 89,168 (63.87% recall)

## 6. Genuine Ceiling Miss Analysis

- **Total Missed Links:** 60,586 (35.076% miss rate)
- **Miss Analysis CSV:** `reports/blocking\EXP_003_memory_safe_ceiling_blocking_misses.csv`

| Heuristic Miss Category | Count | Percentage | Primary Observation | Proposed Recovery Channel |
| :--- | :--- | :--- | :--- | :--- |
| `address_shared_unindexed` | 44,988 | 74.25% | Measured in CSV | See recoverability analysis |
| `complex_transliteration_variant` | 12,828 | 21.17% | Measured in CSV | See recoverability analysis |
| `missing_address_candidate` | 2,570 | 4.24% | Measured in CSV | See recoverability analysis |
| `name_token_overlap_unindexed` | 182 | 0.30% | Measured in CSV | See recoverability analysis |
| `severe_alias_or_dba` | 18 | 0.03% | Measured in CSV | See recoverability analysis |
