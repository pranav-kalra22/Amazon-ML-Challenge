---
name: submission-auditor
description: Audits final matching_results.tsv and candidate_pairs.tsv files against all formatting, schema, and competition validator rules.
---

# Submission Auditor

## Goal
Enforce 100% strict compliance with all competition formatting rules for `matching_results.tsv` and `candidate_pairs.tsv`. Run the official `validate_submission.py` script and certify readiness before upload.

## When to Use
- Whenever generating submission files for the leaderboard.
- Before assembling the final submission zip archive.

## Mandatory Format Rules
1. **File Names & Folder Structure**:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`
2. **Tab Separator**:
   - Both files must be UTF-8 tab-separated (`\t`), never comma-separated.
3. **Exact Headers**:
   - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids`
   - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids`
4. **Complete Entity Coverage**:
   - Every single entity in `test_source1.tsv` (all 1,732,544 entities) must appear exactly once.
   - No extra entities; no missing entities; no duplicate rows.
5. **ID List Integrity**:
   - Singletons must have an empty string after the tab (no spaces, no quotes, no `None` or `NaN`).
   - Multiple IDs must be separated by commas without spaces or quotes.
   - IDs must carry only `S2-` or `S3-` prefixes (no self-matches to `S1-`).
   - No duplicate IDs within any single row list.
   - IDs must exist in `test_source2.tsv` or `test_source3.tsv`.
6. **Subset Invariant**:
   - Every entity ID present in `matching_results.tsv` must strictly appear in `candidate_pairs.tsv` for that Source 1 entity.

## Procedure
1. Execute local pre-flight programmatic checks on output files.
2. Run the official competition validator:
   ```bash
   python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```
3. A file is certified **SUBMITTABLE** if and only if the validator exits with code `0` and prints `PASS — no blocking issues found. Safe to submit.`

## Failure Conditions
- Any exit code other than 0 from the official validator.
- Missing any test S1 entity ID.
- Inclusion of any S1 self-match or non-existent S2/S3 ID.
