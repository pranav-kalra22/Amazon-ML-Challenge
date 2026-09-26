# Candidate Generation Benchmark Report — `EXP_001_blocking_baseline`

**Date:** 2026-09-25T14:05:18Z  
**Validation Split:** `val_50k_seed42` (50,000 Source 1 Entities)  
**Candidate Search Space:** 10,320,219 Training S2 + S3 Records  
**Total Runtime:** 348.23s | **Peak RAM:** 3733.05 MB  

## 1. Primary Ceiling Metrics `[VALIDATION MEASUREMENT]`

| Metric | Value | Target |
| :--- | :--- | :--- |
| **True Link Recall** | **83.390%** (144,039 / 172,729) | $\ge 99.8\%$ |
| **Full Entity Coverage** | **62.030%** (29,283 / 47,208) | Highest possible |
| **Oracle Macro $F_{0.5}$** | **0.925756** | $\ge 0.995$ |
| **Total Candidate Pairs** | 25,119,441 | Scalable volume |
| **Reduction Ratio** | **99.995132%** | $> 99.99\%$ |

## 2. Candidate Volume Distribution per Entity

| Statistic | Value |
| :--- | :--- |
| Mean | 502.39 |
| Median (p50) | 87.0 |
| p90 | 877.0 |
| p95 | 1879.2 |
| p99 | 7956.0 |
| Maximum | 28,234 |

## 3. Per-Channel Recall & Incremental Contribution

| Channel | Description | Link Recall | True Links Retrieved | Unique Links Added | Avg Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Exact Normalized Name | 45.98% | 79,417 | **+79,417** | 80.7 |
| **B** | Compact & Domain-Normalized | 48.80% | 84,290 | **+5,617** | 80.5 |
| **C** | Rare Token Inverted Index | 27.66% | 47,781 | **+20,849** | 54.3 |
| **D** | Address Anchors | 60.96% | 105,298 | **+38,156** | 367.2 |
| **E** | Transliteration Lexical | 0.00% | 0 | **+0** | 0.0 |
| **UNION** | **Channels A + B + C + D + E** | **83.390%** | **144,039** | **144,039** | **502.4** |

## 4. Diagnostic Candidate Caps Benchmark

> **NOTE:** The candidate cap figures below were evaluated using arbitrary set-iteration truncation (`list(set(candidates))[:K]`) without candidate evidence ranking. They are **UNRANKED / NOT VALID FOR TOP-K DECISIONS** and are preserved strictly for historical baseline auditability. Ranked Top-K evaluation is introduced in EXP_002.

| Candidate Cap | Link Recall | Full Entity Coverage | Oracle Macro $F_{0.5}$ |
| :--- | :--- | :--- | :--- |
| `Uncapped` | 83.390% | 62.030% | **0.925756** |
| `200` | 69.143% | 47.130% | **0.809404** |
| `100` | 59.221% | 36.182% | **0.730959** |
| `50` | 48.186% | 27.514% | **0.629972** |
| `30` | 40.579% | 22.333% | **0.550970** |


## 5. Miss Analysis Breakdown

- **Total Missed Links:** 28,690 (16.610% miss rate)
- **Miss Analysis CSV:** `reports/blocking\EXP_001_blocking_baseline_blocking_misses.csv`

| Miss Category | Count | Percentage |
| :--- | :--- | :--- |
| `reordered_tokens_or_stopwords` | 11,744 | 40.93% |
| `transliteration` | 10,826 | 37.73% |
| `severe_typo_or_dba` | 3,904 | 13.61% |
| `missing_address` | 2,191 | 7.64% |
| `other` | 25 | 0.09% |

## 6. Subgroup Diagnostics

### Country Breakdown

- **US** (29,990 entities): Oracle Macro $F_{0.5} = 0.959855$, Precision = 0.988196, Recall = 0.896519
- **India** (20,010 entities): Oracle Macro $F_{0.5} = 0.874651$, Precision = 0.938381, Recall = 0.761769

### Source Diagnostics

- **Source 2 Links**: Retrieved = 69,190 / 83,561 (82.80% recall)
- **Source 3 Links**: Retrieved = 74,849 / 89,168 (83.94% recall)
