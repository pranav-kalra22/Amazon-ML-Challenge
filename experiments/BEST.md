# Shared Best-Model Registry

This file is the single authoritative source of truth for the entire team regarding the current champion model.

---

### Current Best Experiment: Initial Lower-Bound Baseline
- **Experiment ID**: `EXP_000_all_singletons`
- **Git Commit**: `83bc810`
- **Branch**: `main`
- **Validation Macro F0.5**: `0.055840`
- **Candidate Recall**: `0.0000`
- **Oracle Candidate F0.5**: `0.055840`
- **Singleton Accuracy**: `1.0000` (2,792 / 2,792 singletons correctly identified)
- **Macro Precision**: `0.055840`
- **Macro Recall**: `0.055840`
- **Model**: `all_singletons_rule` (predict empty matches for all entities)
- **Threshold Configuration**: `N/A`
- **Validation Split**: `val_50k_seed42` ([split_metadata.json](file:///c:/Users/acer/Desktop/Amazon%20ML%20Challenge/artifacts/splits/split_metadata.json))
- **Date**: 2026-09-25
- **Notes**: Lower-bound baseline establishing exact empirical credit for singletons (5.584% of validation population).
