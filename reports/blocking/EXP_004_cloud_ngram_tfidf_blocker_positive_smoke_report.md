# Blocking Benchmark Report: EXP_004_cloud_ngram_tfidf_blocker [SMOKE_TEST_ONLY]

- **Mode**: `positive_smoke`
- **Git Commit**: `6095972` (dirty: `True`)
- **Evaluated S1 Entities**: `150`
- **Evaluated Candidate Population**: `5,535`
- **Total True Links**: `535`

## 1. Primary Ceiling Metrics

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Oracle Macro F0.5** | **0.998470** | Entity-level macro F0.5 |
| True-Link Recall | 99.4393% | 532 / 535 |
| Full-Entity Coverage | 97.8723% | Non-singleton entities with 100% hits |
| Total Candidate Pairs | 3,182 | Manageable volume |
| Mean Candidates / S1 | 21.2 | Median: 19.0, P99: 80.0 |
| Peak RAM (RSS) | 188.4 MB | Initial: 148.6 MB |
| Total Runtime | 32.20s | Finished cleanly |

## 2. In-Memory Top-K & Similarity Threshold Sweep Results

| Top-K | Cosine Thresh | Links Hit | Recall | EXP_003 Misses Recovered | % Misses Recovered | Candidate Pairs | Cands / Rec Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 0.30 | 532 | 99.44% | 5 | 62.50% | 1,844 | 233.2 |
| 10 | 0.35 | 532 | 99.44% | 5 | 62.50% | 1,844 | 233.2 |
| 10 | 0.40 | 532 | 99.44% | 5 | 62.50% | 1,705 | 205.4 |
| 10 | 0.50 | 532 | 99.44% | 5 | 62.50% | 1,378 | 140.0 |
| 10 | 0.60 | 531 | 99.25% | 4 | 50.00% | 1,168 | 122.5 |
| 25 | 0.30 | 532 | 99.44% | 5 | 62.50% | 2,006 | 265.6 |
| 25 | 0.35 | 532 | 99.44% | 5 | 62.50% | 2,006 | 265.6 |
| 25 | 0.40 | 532 | 99.44% | 5 | 62.50% | 1,803 | 225.0 |
| 25 | 0.50 | 532 | 99.44% | 5 | 62.50% | 1,400 | 144.4 |
| 25 | 0.60 | 531 | 99.25% | 4 | 50.00% | 1,169 | 122.75 |
| 50 | 0.30 | 532 | 99.44% | 5 | 62.50% | 2,029 | 270.2 |
| 50 | 0.35 | 532 | 99.44% | 5 | 62.50% | 2,029 | 270.2 |
| 50 | 0.40 | 532 | 99.44% | 5 | 62.50% | 1,808 | 226.0 |
| 50 | 0.50 | 532 | 99.44% | 5 | 62.50% | 1,400 | 144.4 |
| 50 | 0.60 | 531 | 99.25% | 4 | 50.00% | 1,169 | 122.75 |
| 100 | 0.30 | 532 | 99.44% | 5 | 62.50% | 2,029 | 270.2 |
| 100 | 0.35 | 532 | 99.44% | 5 | 62.50% | 2,029 | 270.2 |
| 100 | 0.40 | 532 | 99.44% | 5 | 62.50% | 1,808 | 226.0 |
| 100 | 0.50 | 532 | 99.44% | 5 | 62.50% | 1,400 | 144.4 |
| 100 | 0.60 | 531 | 99.25% | 4 | 50.00% | 1,169 | 122.75 |

## 3. Entity Cardinality Breakdown

| Ground-Truth Match Cardinality | Entity Count | True Links | Link Recall | Full Coverage | Oracle Macro F0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 matches (singletons)** | 9 | 0 | 100.00% | 100.00% | 1.000000 |
| **1 match** | 8 | 8 | 100.00% | 100.00% | 1.000000 |
| **2 matches** | 24 | 48 | 100.00% | 100.00% | 1.000000 |
| **3 matches** | 39 | 117 | 98.29% | 94.87% | 0.995338 |
| **4 matches** | 26 | 104 | 100.00% | 100.00% | 1.000000 |
| **5+ matches** | 44 | 258 | 99.61% | 97.73% | 0.998918 |

## 4. Zero-Candidate Entities Audit

- Total Zero-Candidate Entities: `0`
- Non-Singleton Zero-Candidate Entities (Severe Failures): `0`
- Singleton Zero-Candidate Entities (Correct Empties): `0`
- Singletons Receiving Candidates (False Positives): `9`

