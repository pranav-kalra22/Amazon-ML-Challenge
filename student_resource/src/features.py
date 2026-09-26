"""
features.py — Pair-level feature engineering for entity resolution.

For every candidate pair (S1 entity, S2/S3 entity), computes a vector of
similarity features spanning business name, address, country, and structure.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from collections import defaultdict
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Low-level similarity functions
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Token-level Jaccard similarity."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def overlap_coefficient(set_a: set, set_b: set) -> float:
    """Overlap coefficient = |intersection| / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def dice_coefficient(set_a: set, set_b: set) -> float:
    """Dice/Sørensen coefficient = 2|intersection| / (|A| + |B|)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return 2 * len(intersection) / (len(set_a) + len(set_b))


def containment_similarity(s_short: str, s_long: str) -> float:
    """Check if the shorter string is contained in the longer one."""
    if not s_short or not s_long:
        return 0.0
    if len(s_short) > len(s_long):
        s_short, s_long = s_long, s_short
    return 1.0 if s_short in s_long else 0.0


def common_prefix_ratio(s1: str, s2: str) -> float:
    """Ratio of common prefix length to max string length."""
    if not s1 or not s2:
        return 0.0
    prefix_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return prefix_len / max(len(s1), len(s2))


def sorted_token_similarity(s1: str, s2: str) -> float:
    """Sort tokens alphabetically, then compute Jaro-Winkler similarity.
    Handles word-order variations."""
    tokens1 = " ".join(sorted(s1.split()))
    tokens2 = " ".join(sorted(s2.split()))
    return JaroWinkler.normalized_similarity(tokens1, tokens2)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF similarity engine
# ─────────────────────────────────────────────────────────────────────────────

class TfidfSimilarityEngine:
    """Pre-computed TF-IDF engine for fast pairwise cosine similarity lookups."""

    def __init__(self):
        self.name_vectorizer = None
        self.addr_vectorizer = None
        self.name_vectors = {}   # entity_id → sparse vector
        self.addr_vectors = {}   # entity_id → sparse vector

    def fit(self, all_entities: pd.DataFrame):
        """Fit TF-IDF on all entities' normalized names and addresses."""
        entity_ids = all_entities["entity_id"].values
        names = all_entities["name_norm"].fillna("").values
        addrs = all_entities["addr_norm"].fillna("").values

        # Name TF-IDF (character n-grams)
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=80000,
            sublinear_tf=True,
        )
        name_matrix = self.name_vectorizer.fit_transform(names)
        for i, eid in enumerate(entity_ids):
            self.name_vectors[eid] = name_matrix[i]

        # Address TF-IDF (character n-grams)
        self.addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=80000,
            sublinear_tf=True,
        )
        addr_matrix = self.addr_vectorizer.fit_transform(addrs)
        for i, eid in enumerate(entity_ids):
            self.addr_vectors[eid] = addr_matrix[i]

    def name_cosine(self, eid1: str, eid2: str) -> float:
        """Cosine similarity between name TF-IDF vectors."""
        v1 = self.name_vectors.get(eid1)
        v2 = self.name_vectors.get(eid2)
        if v1 is None or v2 is None:
            return 0.0
        return float(sklearn_cosine(v1, v2)[0, 0])

    def addr_cosine(self, eid1: str, eid2: str) -> float:
        """Cosine similarity between address TF-IDF vectors."""
        v1 = self.addr_vectors.get(eid1)
        v2 = self.addr_vectors.get(eid2)
        if v1 is None or v2 is None:
            return 0.0
        return float(sklearn_cosine(v1, v2)[0, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_features(
    row1: dict,
    row2: dict,
    tfidf_engine: TfidfSimilarityEngine = None,
) -> dict:
    """Compute all similarity features for a single candidate pair.

    Args:
        row1: dict with normalized fields for S1 entity
        row2: dict with normalized fields for S2/S3 entity
        tfidf_engine: optional pre-fitted TF-IDF engine

    Returns:
        dict of feature_name → float value
    """
    name1 = row1.get("name_norm", "")
    name2 = row2.get("name_norm", "")
    addr1 = row1.get("addr_norm", "")
    addr2 = row2.get("addr_norm", "")
    country1 = row1.get("country_norm", "")
    country2 = row2.get("country_norm", "")

    name_tokens1 = row1.get("name_tokens", set())
    name_tokens2 = row2.get("name_tokens", set())
    addr_tokens1 = row1.get("addr_tokens", set())
    addr_tokens2 = row2.get("addr_tokens", set())
    nums1 = row1.get("numeric_tokens", set())
    nums2 = row2.get("numeric_tokens", set())
    postal1 = row1.get("postal_code", "")
    postal2 = row2.get("postal_code", "")

    # Ensure tokens are sets
    if isinstance(name_tokens1, str):
        name_tokens1 = set(name_tokens1.split())
    if isinstance(name_tokens2, str):
        name_tokens2 = set(name_tokens2.split())
    if isinstance(addr_tokens1, str):
        addr_tokens1 = set(addr_tokens1.split())
    if isinstance(addr_tokens2, str):
        addr_tokens2 = set(addr_tokens2.split())
    if isinstance(nums1, str):
        nums1 = set(nums1.split())
    if isinstance(nums2, str):
        nums2 = set(nums2.split())

    features = {}

    # ── Name features ──────────────────────────────────────────────────────
    features["name_exact_match"] = 1.0 if name1 == name2 and name1 else 0.0
    features["name_jaro_winkler"] = JaroWinkler.normalized_similarity(name1, name2) if name1 and name2 else 0.0
    features["name_levenshtein"] = Levenshtein.normalized_similarity(name1, name2) if name1 and name2 else 0.0
    features["name_fuzz_ratio"] = fuzz.ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    features["name_fuzz_partial"] = fuzz.partial_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    features["name_fuzz_token_sort"] = fuzz.token_sort_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    features["name_fuzz_token_set"] = fuzz.token_set_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    features["name_jaccard"] = jaccard_similarity(name_tokens1, name_tokens2)
    features["name_overlap"] = overlap_coefficient(name_tokens1, name_tokens2)
    features["name_dice"] = dice_coefficient(name_tokens1, name_tokens2)
    features["name_sorted_jw"] = sorted_token_similarity(name1, name2) if name1 and name2 else 0.0
    features["name_prefix_ratio"] = common_prefix_ratio(name1, name2)
    features["name_containment"] = containment_similarity(name1, name2)

    # ── Address features ───────────────────────────────────────────────────
    features["addr_exact_match"] = 1.0 if addr1 == addr2 and addr1 else 0.0
    features["addr_jaro_winkler"] = JaroWinkler.normalized_similarity(addr1, addr2) if addr1 and addr2 else 0.0
    features["addr_levenshtein"] = Levenshtein.normalized_similarity(addr1, addr2) if addr1 and addr2 else 0.0
    features["addr_fuzz_ratio"] = fuzz.ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features["addr_fuzz_partial"] = fuzz.partial_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features["addr_fuzz_token_sort"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features["addr_fuzz_token_set"] = fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features["addr_jaccard"] = jaccard_similarity(addr_tokens1, addr_tokens2)
    features["addr_overlap"] = overlap_coefficient(addr_tokens1, addr_tokens2)
    features["addr_dice"] = dice_coefficient(addr_tokens1, addr_tokens2)
    features["addr_sorted_jw"] = sorted_token_similarity(addr1, addr2) if addr1 and addr2 else 0.0

    # ── Country features ───────────────────────────────────────────────────
    features["country_match"] = 1.0 if country1 == country2 and country1 else 0.0

    # ── Postal / numeric features ──────────────────────────────────────────
    features["postal_match"] = 1.0 if postal1 and postal2 and postal1 == postal2 else 0.0
    features["postal_both_present"] = 1.0 if postal1 and postal2 else 0.0
    features["numeric_jaccard"] = jaccard_similarity(nums1, nums2)
    features["numeric_overlap"] = overlap_coefficient(nums1, nums2)

    # ── Length / structural features ───────────────────────────────────────
    features["name_len_diff"] = abs(len(name1) - len(name2))
    features["name_len_ratio"] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
    features["addr_len_diff"] = abs(len(addr1) - len(addr2))
    features["addr_len_ratio"] = min(len(addr1), len(addr2)) / max(len(addr1), len(addr2), 1)
    features["name_token_count_diff"] = abs(len(name_tokens1) - len(name_tokens2))
    features["addr_token_count_diff"] = abs(len(addr_tokens1) - len(addr_tokens2))
    features["shared_name_tokens"] = len(name_tokens1 & name_tokens2)
    features["shared_addr_tokens"] = len(addr_tokens1 & addr_tokens2)
    features["shared_numeric_tokens"] = len(nums1 & nums2)

    # ── TF-IDF features (if engine provided) ───────────────────────────────
    if tfidf_engine is not None:
        eid1 = row1.get("entity_id", "")
        eid2 = row2.get("entity_id", "")
        features["name_tfidf_cosine"] = tfidf_engine.name_cosine(eid1, eid2)
        features["addr_tfidf_cosine"] = tfidf_engine.addr_cosine(eid1, eid2)
    else:
        features["name_tfidf_cosine"] = 0.0
        features["addr_tfidf_cosine"] = 0.0

    # ── Combined / interaction features ────────────────────────────────────
    features["name_addr_avg_jw"] = (
        features["name_jaro_winkler"] + features["addr_jaro_winkler"]
    ) / 2.0
    features["name_addr_min_jw"] = min(
        features["name_jaro_winkler"], features["addr_jaro_winkler"]
    )
    features["name_addr_max_jw"] = max(
        features["name_jaro_winkler"], features["addr_jaro_winkler"]
    )

    return features


def build_feature_matrix(
    candidates: dict,
    s1_lookup: dict,
    other_lookup: dict,
    tfidf_engine: TfidfSimilarityEngine = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix for all candidate pairs.

    Args:
        candidates: dict of s1_entity_id → set(candidate_entity_ids)
        s1_lookup: dict of entity_id → row dict for S1 entities
        other_lookup: dict of entity_id → row dict for S2/S3 entities
        tfidf_engine: optional TF-IDF engine

    Returns:
        DataFrame with columns: s1_id, s2s3_id, feature1, feature2, ...
    """
    rows = []
    total_pairs = sum(len(v) for v in candidates.items())

    if verbose:
        print(f"\nBuilding features for {total_pairs:,} candidate pairs...")

    pair_iter = (
        (s1_eid, cand_eid)
        for s1_eid, cand_set in candidates.items()
        for cand_eid in cand_set
    )

    if verbose:
        pair_iter = tqdm(pair_iter, total=total_pairs, desc="Feature engineering")

    for s1_eid, cand_eid in pair_iter:
        row1 = s1_lookup.get(s1_eid)
        row2 = other_lookup.get(cand_eid)
        if row1 is None or row2 is None:
            continue

        feats = compute_pair_features(row1, row2, tfidf_engine)
        feats["s1_id"] = s1_eid
        feats["s2s3_id"] = cand_eid
        rows.append(feats)

    df = pd.DataFrame(rows)
    if verbose:
        print(f"  Feature matrix shape: {df.shape}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature column names (exclude ID columns and label)."""
    exclude = {"s1_id", "s2s3_id", "label"}
    return [c for c in df.columns if c not in exclude]
