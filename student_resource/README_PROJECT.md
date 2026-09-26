# Business Entity Resolution — ML Pipeline

## Overview

An ML-based Entity Resolution system that identifies matching business records across three independent data sources. Source 1 is the deduplicated reference; Sources 2 and 3 contain noisy/duplicate records. The system finds which S2/S3 records correspond to the same real-world business as each S1 entity.

## Project Structure

```
student_resource/
├── src/
│   ├── __init__.py          # Package init
│   ├── data_loader.py       # Load and validate TSV files
│   ├── normalization.py     # Business name, address, country normalization
│   ├── blocking.py          # Multi-strategy candidate generation
│   ├── features.py          # 40+ pair-level similarity features
│   ├── model.py             # LightGBM classifier + threshold optimization
│   ├── evaluation.py        # Entity-level F0.5 evaluation + error analysis
│   ├── inference.py         # Test-time inference + output file generation
│   └── main.py              # Full pipeline orchestrator
├── run_small.py             # Run pipeline on sampled data (for testing)
├── requirements.txt         # Python dependencies
├── dataset/
│   ├── train/               # Training TSVs + ground truth
│   └── test/                # Test TSVs (no ground truth)
├── utils/
│   └── validate_submission.py
└── output/                  # Generated output files
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- pandas, numpy — Data handling
- scikit-learn — TF-IDF vectorization, metrics
- lightgbm — Gradient boosted classifier
- rapidfuzz — Fast string similarity (Jaro-Winkler, Levenshtein, fuzzy ratios)
- tqdm — Progress bars
- joblib — Model serialization

## How to Run

### Quick Test (Small Sample — Recommended First)

```bash
# Sample 500 S1 entities and run full pipeline
python run_small.py --n 500

# Faster: disable TF-IDF blocking
python run_small.py --n 500 --no-tfidf-blocking

# Even smaller test
python run_small.py --n 100 --no-tfidf-blocking

# Train/validate only (no output files)
python run_small.py --n 500 --train-only
```

### Full Pipeline

```bash
# Train on full training data, then predict on test data
python -m src.main --data-dir dataset --output-dir output

# Train only (validation metrics, no test inference)
python -m src.main --data-dir dataset --train-only

# Test only (requires previously saved model)
python -m src.main --data-dir dataset --test-only --model-path models/model.joblib

# Faster: skip TF-IDF blocking
python -m src.main --data-dir dataset --no-tfidf-blocking
```

### Validate Submission

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Phases

### Phase 1 — Data Loading
Loads tab-separated source files. Ground truth parsed into match lists.

### Phase 2 — Normalization
- **Names**: lowercase, strip accents, remove punctuation, expand abbreviations (Pvt→Private, Ltd→Limited, Corp→Corporation)
- **Addresses**: lowercase, strip accents, expand abbreviations (Rd→Road, St→Street), apply city aliases (Bengaluru→Bangalore, Mumbai→Bombay)
- **Countries**: normalize to canonical form (USA/U.S.A→US, Bharat→India)

### Phase 3 — Candidate Generation (Blocking)
Multi-strategy blocking with union:
1. **Name token blocking** — shared important name tokens within same country
2. **Character n-gram blocking** — shared char 3-grams for fuzzy matches
3. **TF-IDF blocking** — top-K cosine-similar names via char n-gram TF-IDF
4. **Numeric token blocking** — shared house/postal numbers

### Phase 4 — Feature Engineering (40+ features)
For every candidate pair:
- **Name**: exact match, Jaro-Winkler, Levenshtein, fuzzy ratio, partial ratio, token sort/set ratio, Jaccard, overlap, Dice, sorted-token JW, prefix ratio, containment, TF-IDF cosine
- **Address**: same battery of similarity metrics
- **Country**: exact match
- **Postal/Numeric**: postal code match, numeric token Jaccard/overlap
- **Structural**: length differences, token count differences, shared token counts
- **Interaction**: avg/min/max of name+address Jaro-Winkler

### Phase 5 — ML Model (LightGBM)
- Binary classifier: P(match) for each candidate pair
- Handles class imbalance via `is_unbalance`
- Early stopping on validation loss

### Phase 6 — Threshold Optimization
- Entity-level macro-averaged F0.5 (matches competition scorer)
- Sweeps thresholds 0.30–0.95
- Selects threshold maximizing validation F0.5

### Phase 7 — Evaluation + Error Analysis
- Entity-level precision, recall, F0.5
- Categorizes: perfect match, partial match, true singleton, false positive/negative singleton
- Prints false positive/negative examples with original business names

### Phase 8 — Test Inference
- Runs full pipeline on test data
- Generates `matching_results.tsv` and `candidate_pairs.tsv`

## Output Files

### matching_results.tsv
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

### candidate_pairs.tsv
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00911
S1-00002	S3-00004,S3-00121
S1-00003	
```

## Evaluation Metric

**F₀.₅** (precision-weighted):
```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed per S1 entity, then macro-averaged. Singletons included.

## Key Design Decisions

1. **Entity-level train/val split** — prevents data leakage
2. **Conservative normalization** — preserves discriminative info
3. **Union of blocking strategies** — maximizes candidate recall
4. **Precision-oriented threshold** — F0.5 penalizes false merges more
5. **Open-set country handling** — no hardcoded country lists
6. **No external data** — purely based on provided datasets
