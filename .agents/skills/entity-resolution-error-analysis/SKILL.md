---
name: entity-resolution-error-analysis
description: Audits validation prediction errors, categorizing failure modes, generating error CSVs, and recommending targeted next experiments.
---

# Entity Resolution Error Analysis

## Goal
Systematically dissect and quantify all errors made by the current pipeline on the validation split. Isolate failure mechanisms into standardized categories and produce actionable recommendations for the next experiment.

## When to Use
- After running validation inference and threshold optimization on any model.
- Before formulating new features, blocking passes, or modeling iterations.

## Error Categories
Every error instance must be classified into one of these standard archetypes:
1. `same_name_diff_addr`: Same/similar business name located at completely different addresses (chain/branch or lookalike).
2. `same_addr_diff_biz`: Shared building number or complex housing two distinct business names.
3. `singleton_false_merge`: True singleton incorrectly assigned one or more matches.
4. `transliteration_gap`: Failure to bridge Indic script or French accent representations.
5. `dba_trade_name`: Unrelated legal name vs DBA trade name where only address anchors matched.
6. `severe_typo_leetspeak`: Unhandled character substitutions, OCR errors, or phonetic shifts.
7. `domain_alias_miss`: Unhandled website domain name format.
8. `missing_addr_ambiguity`: Candidate has empty address and name similarity is borderline.
9. `blocking_miss`: True match never retrieved into the candidate pool ($c \notin C_i$).
10. `threshold_miss`: True match was retrieved by blocker but scored below decision threshold ($c \in C_i, \hat{p}(c) < \tau$).

## Required Outputs
Export the following CSV diagnostics under `reports/error_analysis/<experiment_id>/`:
- `false_positives.csv`: Pairs predicted as matches that are not true matches.
- `false_negatives.csv`: True matching pairs that were not predicted.
- `singleton_false_positives.csv`: True singletons where false merges occurred.
- `blocking_misses.csv`: True links completely omitted by candidate generation.
- `threshold_misses.csv`: True links present in candidates but filtered by threshold.
- `error_summary_report.md`: Quantitative breakdown of failure counts by category and recommended next experiment.

## Procedure
1. Join validation predictions, candidates, and ground truth.
2. Partition discrepancies into False Positives, False Negatives, and Singleton False Merges.
3. Apply heuristic classifiers to tag each error with its failure archetype.
4. Summarize error distributions in a markdown report.
5. Formulate exactly one primary experiment hypothesis targeting the largest error category.

## Constraints
- Do not inspect test data (test has no ground truth).
- All error quantification must state exact sample sizes.
