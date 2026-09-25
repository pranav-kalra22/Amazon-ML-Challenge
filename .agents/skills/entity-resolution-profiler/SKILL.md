---
name: entity-resolution-profiler
description: Profiles the Amazon ML Challenge dataset memory-efficiently, quantifying schemas, row counts, missing rates, cardinality, singletons, and noise patterns.
---

# Entity Resolution Profiler

## Goal
Perform rigorous, scalable, and memory-conscious data profiling on the multi-source dataset (Train & Test splits for Sources 1, 2, 3, and Ground Truth). Output structured diagnostic reports to `reports/data_profile/`.

## When to Use
- During initial workspace exploration.
- Whenever new data splits, partitions, or cleaned datasets are generated.
- To verify data integrity before designing blockers or feature pipelines.

## Required Inputs
- `dataset/train/train_source1.tsv`
- `dataset/train/train_source2.tsv`
- `dataset/train/train_source3.tsv`
- `dataset/train/train_ground_truth.tsv`
- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

## Procedure
1. **Streaming / Chunked File Inspection**:
   - Stream or chunk datasets to avoid loading 10M+ objects into unoptimized Python structures.
   - Use Polars, PyArrow, DuckDB, or chunked Pandas (`chunksize=250000`).
2. **Structural Profiling**:
   - Measure exact row counts, byte sizes, and column data types.
   - Detect and report delimiter anomalies or malformed rows.
3. **Completeness & Integrity**:
   - Measure null/empty string rates for `business_name`, `business_address`, and `country`.
   - Calculate duplicate rates on `entity_id` and raw text content within each source.
4. **Country Partition Analysis**:
   - Profile the exact frequency and percentage distribution of `country` across every file.
   - Systematically verify whether any ground truth matches cross country boundaries on the full training dataset.
5. **Cardinality & Singleton Profiling**:
   - Profile ground truth match cardinality (0, 1, 2, ..., N matches).
   - Compute exact singleton percentage and distribution.
   - Check if any Source 2 or Source 3 entity ID maps to multiple distinct Source 1 entities (many-to-one / many-to-many sharing).
6. **Lexical & Textual Profiling**:
   - Character and token length distributions for names and addresses.
   - Prevalence of non-ASCII / Indic Unicode characters and French accents.
   - Frequency of domain-like patterns (`.com`, `.in`, `.org`, etc.) in business names.
   - Distribution of noise patterns (e.g., `#`, `Door No`, `PMB`, state codes).

## Required Outputs
- `reports/data_profile/dataset_summary.json`: Machine-readable summary of counts, null rates, and cardinality.
- `reports/data_profile/data_profile_report.md`: Human-readable markdown report with exact evidence classifications (`FULL-DATA MEASUREMENT` vs `SAMPLE MEASUREMENT`).

## Validation Checks
- Output JSON must parse without errors.
- Every metric must explicitly declare whether it was evaluated on the full dataset or a sample ($N$).
- Memory consumption during execution must not exceed available system RAM.

## Failure Conditions
- Loading all source files uncompressed simultaneously into memory causing Out-Of-Memory (OOM) crashes.
- Generalizing a sample observation to a full dataset claim without a full measurement.

## Constraints
- External lookups are strictly prohibited.
- Do not modify or overwrite raw dataset files.
