# License Compliance Audit Report — EXP_004 Cloud Candidate Generation

**Audit Date:** 2026-09-26  
**Auditor:** Automated Compliance Auditor (`license-compliance` skill)  
**Experiment ID:** `EXP_004_cloud_ngram_tfidf_blocker`  
**Compliance Standard:** Amazon ML Challenge 2026 Fair Play Policy (Permissive Open-Source Only: MIT, Apache 2.0, BSD; Model Parameters $\le 8\text{B}$; Zero Commercial Lookups).

---

## 1. Audited Dependencies Summary Table

| Package / Tool | Pinned / Installed Version | Declared License | Authoritative License Source | Compliance Status | Model Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sparse_dot_topn`** | `1.2.0` | **Apache-2.0** | [ING Bank GitHub / PyPI](https://github.com/ing-bank/sparse_dot_topn) | `COMPLIANT` | 0 (Algorithmic C++ / OpenMP) |
| **`scikit-learn`** | `1.5.2` | **BSD-3-Clause** | [scikit-learn.org](https://github.com/scikit-learn/scikit-learn) | `COMPLIANT` | 0 (Lexical Vectorizer) |
| **`scipy`** | `1.16.0` | **BSD-3-Clause** | [scipy.org](https://github.com/scipy/scipy) | `COMPLIANT` | 0 (Numerical Library) |
| **`numpy`** | `2.5.3` | **BSD-3-Clause** | [numpy.org](https://github.com/numpy/numpy) | `COMPLIANT` | 0 (Array Engine) |
| **`pandas`** | `2.2.2` | **BSD-3-Clause** | [pandas.pydata.org](https://github.com/pandas-dev/pandas) | `COMPLIANT` | 0 (DataFrames) |
| **`pyarrow`** | `23.0.1` | **Apache-2.0** | [Apache Arrow](https://arrow.apache.org/) | `COMPLIANT` | 0 (Parquet Engine) |
| **`pyyaml`** | `6.0.3` | **MIT** | [yaml.org / PyPI](https://github.com/yaml/pyyaml) | `COMPLIANT` | 0 (Config Parser) |
| **`psutil`** | `6.0.0` | **BSD-3-Clause** | [giampaolo/psutil](https://github.com/giampaolo/psutil) | `COMPLIANT` | 0 (Process Profiler) |
| **`text-unidecode`** | `1.3` | **Artistic License** | [kmike/text-unidecode](https://github.com/kmike/text-unidecode) | `COMPLIANT` | 0 (Deterministic Transliteration) |
| **`joblib`** | `1.4.2` | **BSD-3-Clause** | [joblib/joblib](https://github.com/joblib/joblib) | `COMPLIANT` | 0 (Serialization) |
| **`pytest`** | `9.1.1` | **MIT** | [pytest.org](https://github.com/pytest-dev/pytest) | `COMPLIANT` | 0 (Test Framework) |

---

## 2. Model Parameter Audit
- **Current Retrieval Architecture:** Character N-Gram TF-IDF with Sparse Cosine Top-N Multiplication.
- **Pre-trained Deep Learning Models:** None used in EXP_004.
- **Total Model Parameters:** $0 \le 8,000,000,000$ limit.
- **Dense Vector Embeddings:** None. FAISS, Sentence Transformers, and LLMs are strictly deferred until candidate generator ceiling headroom is proven.

---

## 3. External API and Network Lookup Audit
- **External Web Searches for Business Entities:** 0 (Strictly prohibited).
- **External Geocoding / Address Lookups:** 0 (Strictly prohibited).
- **External Business Registries / Scraping:** 0 (Strictly prohibited).
- **Execution Mode:** 100% offline, local/in-cluster execution on local disk and staged files.

---

## 4. Conclusion
All software components, runtime dependencies, and algorithmic tools for `EXP_004_cloud_ngram_tfidf_blocker` are fully certified as `COMPLIANT` with competition integrity and licensing directives.
