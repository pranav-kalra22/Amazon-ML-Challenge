# License Compliance Audit Report — EXP_004 Cloud Candidate Generation

**Audit Date:** 2026-09-26  
**Auditor:** Compliance Auditor (`license-compliance` directive)  
**Experiment ID:** `EXP_004_cloud_ngram_tfidf_blocker`  
**Compliance Standard:** Amazon ML Challenge 2026 Rules & Integrity Directives (Offline execution; No external lookups; Permissive/open software dependencies; Model parameter count $\le 8\text{B}$).

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
| **`text-unidecode`** | `1.3` | **Artistic License / GPLv1+** | [kmike/text-unidecode](https://github.com/kmike/text-unidecode) | `NOTE ON SCOPE` (See Section 2) | 0 (Deterministic Transliteration) |
| **`joblib`** | `1.4.2` | **BSD-3-Clause** | [joblib/joblib](https://github.com/joblib/joblib) | `COMPLIANT` | 0 (Serialization) |
| **`pytest`** | `9.1.1` | **MIT** | [pytest.org](https://github.com/pytest-dev/pytest) | `COMPLIANT` | 0 (Test Framework) |

---

## 2. Dependency vs. Machine Learning Model License Distinction

The competition rules specify that pre-trained machine learning models and weights must carry permissive open-source licenses (MIT, Apache 2.0, BSD) and $\le 8\text{B}$ parameters.

- **EXP_004 Retrieval Model Parameters:** Exactly $0$. EXP_004 uses no pre-trained weights, embeddings, or neural checkpoints.
- **`sparse_dot_topn`:** Apache-2.0 licensed C++ sparse matrix top-N multiplication engine.
- **Core Numerical Stack:** `scikit-learn`, `scipy`, `numpy`, `pandas`, `psutil`, `joblib` are all BSD-3-Clause; `pyarrow` is Apache-2.0; `pyyaml` and `pytest` are MIT.
- **`text-unidecode` Clarification:** `text-unidecode` is licensed under the Artistic License and GPLv1+. It is a pure-Python deterministic string conversion library (not an ML model with weights). While compatible with local data preprocessing and non-distributed execution, it is not MIT/Apache/BSD. The project explicitly distinguishes software utility dependencies from final ML model licensing requirements. If strict MIT/Apache/BSD purity is demanded across every utility library for final submission, `text-unidecode` can be substituted with native Python unicode normalization (`unicodedata.normalize`) without model architectural impact.

---

## 3. Model Parameter Audit
- **Current Retrieval Architecture:** Character N-Gram TF-IDF with Sparse Cosine Top-N Multiplication.
- **Pre-trained Deep Learning Models:** None used in EXP_004.
- **Total Model Parameters:** $0 \le 8,000,000,000$ limit.
- **Dense Vector Embeddings:** None. FAISS, Sentence Transformers, and LLMs are deferred until lexical/sparse candidate generator ceiling headroom is proven.

---

## 4. External API and Network Lookup Audit
- **External Web Searches for Business Entities:** 0 (Strictly prohibited).
- **External Geocoding / Address Lookups:** 0 (Strictly prohibited).
- **External Business Registries / Scraping:** 0 (Strictly prohibited).
- **Execution Mode:** 100% offline, local/in-cluster execution on local disk and staged files.

---

## 5. Conclusion
Software dependencies, runtime components, and algorithmic tools for `EXP_004_cloud_ngram_tfidf_blocker` are fully documented with factual license designations and zero pre-trained model parameter footprint.

