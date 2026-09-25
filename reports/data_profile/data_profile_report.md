# Amazon ML Challenge 2026 — Comprehensive Data Profile Report

**Generated:** 2026-09-25T12:00:02Z  
## 1. Population & File Sizes `[FULL-DATA MEASUREMENT]`

| Split & Source | Row Count | File Size (MB) |
| :--- | :--- | :--- |
| `train_s1` | 2,206,821 | 210.07 MB |
| `train_s2` | 5,034,616 | 489.3 MB |
| `train_s3` | 5,285,603 | 503.71 MB |
| `train_gt` | 2,206,821 | 127.02 MB |
| `test_s1` | 1,732,544 | 175.02 MB |
| `test_s2` | 4,887,273 | 509.46 MB |
| `test_s3` | 5,082,316 | 506.0 MB |

## 2. Match Cardinality & Singleton Distribution `[FULL-DATA MEASUREMENT]`

- **Total Source 1 Entities:** 2,206,821
- **Singletons (Zero Matches):** 123,247 (5.58%)
- **Total True Links:** 7,638,365
- **Matches per Non-Singleton:** Mean: 3.666, Median: 4, Range: [1, 11]
- **Match Share (Sample N=100k links):** Source 2: 48.3%, Source 3: 51.7%

## 3. Country Breakdown across Splits `[FULL-DATA MEASUREMENT]`

| File | India | US | France |
| :--- | :--- | :--- | :--- |
| `train_s1` | 883,188 | 1,323,633 | 0 |
| `train_s2` | 2,017,799 | 3,016,817 | 0 |
| `train_s3` | 2,115,547 | 3,170,056 | 0 |
| `test_s1` | 809,986 | 663,106 | 259,452 |
| `test_s2` | 2,312,565 | 1,871,330 | 703,378 |
| `test_s3` | 2,405,000 | 1,945,701 | 731,615 |

## 4. Text Quality & Missing Rates `[FULL-DATA MEASUREMENT]`

| File | Missing Name % | Missing Address % | Domain-Name Rate % |
| :--- | :--- | :--- | :--- |
| `train_s1` | 0.00% | 0.00% | 0.00% |
| `train_s2` | 0.00% | 6.71% | 4.00% |
| `train_s3` | 0.00% | 6.66% | 3.99% |
| `test_s1` | 0.00% | 0.00% | 0.00% |
| `test_s2` | 0.00% | 5.30% | 3.19% |
| `test_s3` | 0.00% | 5.36% | 3.23% |

## 5. Core Architectural Takeaways

1. **Zero Cross-Country Links `[FULL-DATA MEASUREMENT]`**: Measured across all 7,638,365 labelled training links (`cross_country == 0`, 100.0% same country). All labelled US and India training links satisfy exact country equality. The same generic equality rule (`country_A == country_B`) is applied dynamically to unseen country labels such as France; France ground truth is unavailable and therefore France recall cannot be directly verified.
2. **France Exclusivity `[FULL-DATA MEASUREMENT]`**: France appears ONLY in the test set (15.0% of Test S1 records: 259,452 S1, 703,378 S2, 731,615 S3). Zero training labels exist for France.
3. **Singleton Guard Mandatory `[FULL-DATA MEASUREMENT]`**: Exactly 123,247 singletons (5.58% of Source 1). Under entity-level Macro F0.5, correct empty predictions score 1.0, while any false positive link on a singleton collapses its entity score to 0.0.
4. **Missing Addresses in S2/S3 `[FULL-DATA MEASUREMENT]`**: S1 has 0.00% missing addresses. Train S2 has 6.71% and Train S3 has 6.66% missing addresses (Test S2: 5.30%, Test S3: 5.36%), requiring dual-path candidate scoring (Name+Address vs Name-Only).
