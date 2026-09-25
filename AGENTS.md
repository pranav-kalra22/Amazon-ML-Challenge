# Amazon ML Challenge 2026 — Project Rules & Operating Directives

> **Core Axiom:** Measure first. Modify second. Validate third.

This project solves the **Business Entity Resolution** problem for the Amazon ML Challenge 2026. All agents and developers working in this workspace must strictly adhere to the following rules:

1. **Skill-First Workflow**: Before starting substantial ML work, determine which skill under `.agents/skills/` applies.
2. **Follow Skill Procedures**: Read and follow the skill's explicit procedure and constraints before execution.
3. **Official Metric**: Use the authoritative entity-level set-based Macro $F_{0.5}$ evaluator (`macro-f05-evaluator`).
4. **No Metric Substitution**: Never optimize only pairwise accuracy, ROC-AUC, or binary $F_1$. Macro $F_{0.5}$ weights precision 2x over recall.
5. **Strict Competition Integrity**: Never perform external entity lookup (no search engines, geocoding APIs, business registries, or scraping). All match evidence must be purely local.
6. **Preserve Raw Information**: Preserve raw fields alongside every normalized representation.
7. **Non-Destructive Normalization**: Never destructively normalize away potentially discriminative information (e.g., retain both raw and canonical tokens).
8. **Decoupled Evaluation**: Candidate-generation (blocking) performance and matching-model performance must be evaluated separately.
9. **Experiment Logging**: All major experiments must be logged in `experiments/experiment_log.csv` (`experiment-ledger`).
10. **Leakage-Safe Splits**: Validation splits must be split strictly by Source 1 entity (preserving complete match sets). Never randomly split candidate pairs.
11. **No Test Ground Truth**: Never assume or use test ground truth, as none exists.
12. **No Extrapolation to France**: Never claim France performance from training evidence; France appears only in the test set.
13. **Open-Set Country Handling**: Never hard-code countries (`US`, `India`). Country is an open-set string label.
14. **First-Class Singletons**: Treat singletons as a first-class prediction problem (correctly predicting empty yields 1.0; false merges on singletons yield 0.0).
15. **Preserve Best Models**: Never overwrite the best existing experiment or model artifacts.
16. **Pre-Submission Validation**: Before submitting, always execute the official validator script (`submission-auditor`).
17. **Candidate Integrity**: All final matches in `matching_results.tsv` must strictly belong to the candidate set in `candidate_pairs.tsv`.
18. **Empirical Justification**: No result may be called an "improvement" without a direct metric comparison against the previous best baseline.
19. **Evidence Discipline**: Always categorize findings as `FULL-DATA MEASUREMENT`, `VALIDATION MEASUREMENT`, `SAMPLE MEASUREMENT`, `HYPOTHESIS`, or `TARGET`. Never present samples as full-data facts.
20. **Reproducible Engineering**: Prefer standalone, deterministic, reproducible Python scripts over one-off manual or notebook calculations.
