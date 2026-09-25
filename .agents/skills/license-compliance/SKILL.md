---
name: license-compliance
description: Audits software libraries, dependencies, and machine learning models for strict compliance with MIT/Apache-2.0 open-source rules and the 8B parameter limit.
---

# License Compliance

## Goal
Verify and document that all Python libraries, pre-trained model weights, and tools strictly comply with the Amazon ML Challenge 2026 rules:
- Open-source license (MIT, Apache 2.0, BSD).
- Maximum model size $\le 8$ Billion parameters.
- No commercial API dependencies.

## When to Use
- Before introducing any new external Python package or pre-trained model.
- Prior to assembling the final submission zip archive.

## Required Information per Dependency
Maintain an auditable table in `reports/license_compliance_report.md` capturing:
1. **Package / Model Name**
2. **Exact Version**
3. **Declared License Type** (e.g., MIT, Apache 2.0, BSD-3-Clause)
4. **Authoritative License Source / URL** (e.g., PyPI metadata, GitHub repository LICENSE file)
5. **Compliance Status**: `COMPLIANT` / `NON-COMPLIANT`
6. **Model Parameter Count** (if an ML model or embedding checkpoint: must be $\le 8\text{B}$)

## Procedure
1. Inspect the active Python environment and all pinned packages in `requirements.txt`.
2. Verify package licenses via `pip-licenses`, PyPI JSON API, or standard distribution metadata (`importlib.metadata`).
3. For pre-trained models:
   - Check the model card on Hugging Face / GitHub.
   - Confirm explicit permissive licensing (reject `CC-BY-NC`, `GPL`, or proprietary commercial terms).
   - Confirm total parameters $< 8,000,000,000$.
4. Generate the audit report: `reports/license_compliance_report.md`.

## Failure Conditions
- Using any dependency or model carrying a non-commercial (`NC`), copyleft (`GPL` without linking exception), or undocumented license.
- Employing any model exceeding 8 Billion parameters.
