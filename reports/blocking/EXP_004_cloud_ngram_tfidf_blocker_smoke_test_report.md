# Blocking Benchmark Report: EXP_004_cloud_ngram_tfidf_blocker [SMOKE_TEST_ONLY]

- **Mode**: `smoke_test`
- **Git Commit**: `54b8bfa` (dirty: `True`)
- **Evaluated S1 Entities**: `500`
- **Evaluated Candidate Records**: `40,000`
- **Total True Links**: `1,798`

## 1. Primary Blocking Ceiling Metrics

| Metric | Score | Target Gate |
| :--- | :--- | :--- |
| **Oracle Macro F0.5** | **0.063179** | >= 0.9950 |
| True-Link Recall | 0.3337% | >= 95.0% |
| Full-Entity Coverage | 0.0000% | High |
| Total Candidate Pairs | 74,140 | Manageable |
| Mean Candidates / S1 | 148.3 | - |
| Peak RAM (RSS) | 246.1 MB | <= 56320 MB |
| Runtime | 47.15s | Scalable |

## 2. Incremental Miss Recovery Analysis

| Channel | Standalone Recall | EXP_003 Misses Recovered | % Misses Recovered | Candidate Pairs | Cands / Recovered Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_003_lexical` | 0.28% | 0 | 0.00% | 74,140 | - |
| `Name_Char_TFIDF` | 0.22% | 0 | 0.00% | 24,750 | 24750.0 |
| `Address_Char_TFIDF` | 0.28% | 1 | 0.06% | 4,985 | 4985.0 |
| `Transliterated_Char_TFIDF` | 0.22% | 0 | 0.00% | 25,392 | 25392.0 |
| `FINAL_UNION` | 0.33% | 1 | 0.06% | 74,140 | 74140.0 |

## 3. Entity Cardinality Performance Breakdown

| Ground-Truth Match Cardinality | Entity Count | True Links | Link Recall | Full Coverage | Oracle Macro F0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 matches (singletons)** | 28 | 0 | 100.00% | 100.00% | 1.000000 |
| **1 match** | 30 | 30 | 0.00% | 0.00% | 0.000000 |
| **2 matches** | 70 | 140 | 0.71% | 0.00% | 0.011905 |
| **3 matches** | 126 | 378 | 0.26% | 0.00% | 0.005669 |
| **4 matches** | 95 | 380 | 0.26% | 0.00% | 0.006579 |
| **5+ matches** | 151 | 870 | 0.34% | 0.00% | 0.009382 |

## 4. Zero-Candidate Entities Audit

- Total Zero-Candidate Entities: `0`
- Non-Singleton Zero-Candidate Entities (Severe Failures): `0`
- Singleton Zero-Candidate Entities (Correct Empties): `0`
- Singletons Receiving Candidates (False Positives): `28`

