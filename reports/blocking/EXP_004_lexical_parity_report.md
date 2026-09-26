# Blocking Benchmark Report: EXP_004_cloud_ngram_tfidf_blocker [LEXICAL_PARITY_GATE]

- **Mode**: `parity_check`
- **Git Commit**: `5173e9e`
- **Evaluated S1 Entities**: `50,000`
- **Evaluated Candidate Population**: `10,320,219`
- **Total True Links**: `172,729`

## Lexical Parity Gate Verdict: **PASS**

| Metric | EXP_003 Authoritative Baseline | EXP_004 Lexical-Only Path | Match? |
| :--- | :--- | :--- | :--- |
| **True Links Hit** | 112,143 / 172,729 | 112,143 / 172,729 | YES |
| **True Link Recall** | 0.649242 (64.92%) | 0.649242 (64.92%) | YES |
| **Full Entity Coverage** | 0.304864 (30.49%) | 0.304864 (30.49%) | YES |
| **Oracle Macro F0.5** | **0.831634** | **0.831634** | YES |

## 1. Primary Ceiling Metrics

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Oracle Macro F0.5** | **0.831634** | Entity-level macro F0.5 |
| True-Link Recall | 64.9242% | 112,143 / 172,729 |
| Full-Entity Coverage | 30.4864% | Non-singleton entities with 100% hits |
| Total Candidate Pairs | 4,845,468 | Mode B Ranked Pairs |
| Mean Candidates / S1 | 96.9 | Median: 7.0, P99: 1592.0 |
| Peak RAM (RSS) | 1396.5 MB | Initial: 148.7 MB |
| Total Runtime | 1700.29s | Finished cleanly |

## 2. Entity Cardinality Breakdown

| Ground-Truth Match Cardinality | Entity Count | True Links | Link Recall | Full Coverage | Oracle Macro F0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 matches (singletons)** | 2,792 | 0 | 100.00% | 100.00% | 1.000000 |
| **1 match** | 2,753 | 2,753 | 65.17% | 65.17% | 0.651653 |
| **2 matches** | 8,537 | 17,074 | 64.52% | 44.50% | 0.778650 |
| **3 matches** | 12,046 | 36,138 | 64.68% | 32.48% | 0.823746 |
| **4 matches** | 10,926 | 43,704 | 64.97% | 24.57% | 0.847977 |
| **5+ matches** | 12,946 | 73,060 | 65.10% | 17.01% | 0.862081 |

## 3. Zero-Candidate Entities Audit

- Total Zero-Candidate Entities: `1,555`
- Non-Singleton Zero-Candidate Entities (Severe Failures): `990`
- Singleton Zero-Candidate Entities (Correct Empties): `565`
- Singletons Receiving Candidates (False Positives): `2,227`
