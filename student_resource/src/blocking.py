"""
blocking.py — Candidate generation through multi-strategy blocking.

Blocking reduces the O(N×M) comparison space to a manageable set of candidate
pairs by using multiple cheap filters whose union preserves high recall.

Strategies:
    1. Country blocking — only compare records in the same country
    2. Name-token blocking — shared important name tokens within same country
    3. Character n-gram blocking — shared character 3-grams for fuzzy matches
    4. TF-IDF blocking — top-K cosine-similar names via TF-IDF
    5. Address/numeric blocking — shared numeric tokens (house #, postal code)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


def _get_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def country_block(s1: pd.DataFrame, s_other: pd.DataFrame) -> dict:
    """Build country → entity_id index for blocking.

    Returns dict: s1_entity_id → set of candidate entity_ids from s_other
    that share the same country.
    """
    # Build country index for s_other
    country_index = defaultdict(set)
    for _, row in s_other.iterrows():
        country_index[row["country_norm"]].add(row["entity_id"])

    candidates = defaultdict(set)
    for _, row in s1.iterrows():
        country = row["country_norm"]
        if country in country_index:
            candidates[row["entity_id"]] = country_index[country].copy()

    return candidates


def name_token_block(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    min_shared_tokens: int = 1,
) -> dict:
    """Block by shared name tokens within the same country.

    Two records become candidates if they share at least `min_shared_tokens`
    important name tokens AND are in the same country.
    """
    from src.normalization import extract_name_tokens

    # Build inverted index: (country, token) → set of entity_ids
    token_index = defaultdict(set)
    for _, row in s_other.iterrows():
        important_tokens = extract_name_tokens(row["name_norm"])
        country = row["country_norm"]
        for token in important_tokens:
            if len(token) >= 2:  # Skip single-char tokens
                token_index[(country, token)].add(row["entity_id"])

    candidates = defaultdict(set)
    for _, row in s1.iterrows():
        important_tokens = extract_name_tokens(row["name_norm"])
        country = row["country_norm"]

        # Collect all entities that share at least one token
        candidate_counts = defaultdict(int)
        for token in important_tokens:
            if len(token) >= 2:
                for eid in token_index.get((country, token), set()):
                    candidate_counts[eid] += 1

        # Keep candidates with enough shared tokens
        for eid, count in candidate_counts.items():
            if count >= min_shared_tokens:
                candidates[row["entity_id"]].add(eid)

    return candidates


def ngram_block(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    n: int = 3,
    min_shared: int = 2,
) -> dict:
    """Block by shared character n-grams of business names within same country.

    Helps catch fuzzy matches missed by token-level blocking (e.g., misspellings).
    """
    # Build inverted index: (country, ngram) → set of entity_ids
    ngram_index = defaultdict(set)
    for _, row in s_other.iterrows():
        ngrams = _get_ngrams(row["name_norm"], n)
        country = row["country_norm"]
        for ng in ngrams:
            ngram_index[(country, ng)].add(row["entity_id"])

    candidates = defaultdict(set)
    for _, row in s1.iterrows():
        ngrams = _get_ngrams(row["name_norm"], n)
        country = row["country_norm"]

        candidate_counts = defaultdict(int)
        for ng in ngrams:
            for eid in ngram_index.get((country, ng), set()):
                candidate_counts[eid] += 1

        for eid, count in candidate_counts.items():
            if count >= min_shared:
                candidates[row["entity_id"]].add(eid)

    return candidates


def tfidf_block(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    top_k: int = 20,
    min_score: float = 0.15,
) -> dict:
    """Block using TF-IDF cosine similarity on normalized names.

    For each S1 entity, find the top-K most similar names from s_other
    (within the same country) based on character n-gram TF-IDF.
    """
    candidates = defaultdict(set)

    # Process per-country to enforce country blocking
    countries_s1 = s1["country_norm"].unique()

    for country in countries_s1:
        s1_country = s1[s1["country_norm"] == country].reset_index(drop=True)
        s_other_country = s_other[s_other["country_norm"] == country].reset_index(drop=True)

        if len(s_other_country) == 0 or len(s1_country) == 0:
            continue

        # Fit TF-IDF on combined names
        all_names = pd.concat([
            s1_country["name_norm"],
            s_other_country["name_norm"]
        ]).fillna("").values

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(all_names)

        n_s1 = len(s1_country)
        s1_vecs = tfidf_matrix[:n_s1]
        other_vecs = tfidf_matrix[n_s1:]

        # Compute similarities in batches to manage memory
        batch_size = 500
        for batch_start in range(0, n_s1, batch_size):
            batch_end = min(batch_start + batch_size, n_s1)
            sim = cosine_similarity(s1_vecs[batch_start:batch_end], other_vecs)

            for i in range(sim.shape[0]):
                s1_idx = batch_start + i
                s1_eid = s1_country.iloc[s1_idx]["entity_id"]

                # Get top-K indices
                row_sims = sim[i]
                if len(row_sims) <= top_k:
                    top_indices = np.where(row_sims >= min_score)[0]
                else:
                    top_indices = np.argpartition(row_sims, -top_k)[-top_k:]
                    top_indices = top_indices[row_sims[top_indices] >= min_score]

                for idx in top_indices:
                    candidates[s1_eid].add(s_other_country.iloc[idx]["entity_id"])

    return candidates


def numeric_token_block(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    min_shared: int = 1,
) -> dict:
    """Block by shared numeric tokens in addresses within the same country.

    Numeric tokens include house numbers, floor numbers, postal codes, etc.
    Only generates candidates when both records have numeric tokens.
    """
    # Build inverted index: (country, numeric_token) → set of entity_ids
    num_index = defaultdict(set)
    for _, row in s_other.iterrows():
        nums = row.get("numeric_tokens", set())
        if isinstance(nums, str):
            nums = set(nums.split())
        country = row["country_norm"]
        for num in nums:
            if len(num) >= 1:
                num_index[(country, num)].add(row["entity_id"])

    candidates = defaultdict(set)
    for _, row in s1.iterrows():
        nums = row.get("numeric_tokens", set())
        if isinstance(nums, str):
            nums = set(nums.split())
        country = row["country_norm"]

        if not nums:
            continue

        candidate_counts = defaultdict(int)
        for num in nums:
            for eid in num_index.get((country, num), set()):
                candidate_counts[eid] += 1

        for eid, count in candidate_counts.items():
            if count >= min_shared:
                candidates[row["entity_id"]].add(eid)

    return candidates


def union_candidates(*candidate_dicts) -> dict:
    """Take the union of multiple blocking strategies."""
    merged = defaultdict(set)
    for cand_dict in candidate_dicts:
        for s1_eid, cand_set in cand_dict.items():
            merged[s1_eid].update(cand_set)
    return dict(merged)


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    use_tfidf: bool = True,
    tfidf_top_k: int = 20,
    embedding_engine=None,
    embedding_top_k: int = 30,
    embedding_min_score: float = 0.3,
    verbose: bool = True,
) -> dict:
    """Run full multi-strategy blocking pipeline.

    Returns:
        dict: s1_entity_id → set of candidate entity_ids from S2 and S3
    """
    if verbose:
        print("\n=== Candidate Generation (Blocking) ===")

    all_candidates = defaultdict(set)

    n_strategies = 4 + (1 if embedding_engine is not None else 0)

    for label, s_other in [("S2", s2), ("S3", s3)]:
        if verbose:
            print(f"\n--- Blocking S1 vs {label} ---")

        # Strategy 1: Name token blocking
        if verbose:
            print(f"  [1/{n_strategies}] Name token blocking...")
        name_cands = name_token_block(s1, s_other, min_shared_tokens=1)
        if verbose:
            n_pairs = sum(len(v) for v in name_cands.values())
            print(f"        → {len(name_cands):,} S1 entities, {n_pairs:,} pairs")

        # Strategy 2: Character n-gram blocking
        if verbose:
            print(f"  [2/{n_strategies}] Character n-gram blocking...")
        ngram_cands = ngram_block(s1, s_other, n=3, min_shared=2)
        if verbose:
            n_pairs = sum(len(v) for v in ngram_cands.values())
            print(f"        → {len(ngram_cands):,} S1 entities, {n_pairs:,} pairs")

        # Strategy 3: TF-IDF blocking (optional, slower)
        tfidf_cands = {}
        if use_tfidf:
            if verbose:
                print(f"  [3/{n_strategies}] TF-IDF blocking...")
            tfidf_cands = tfidf_block(s1, s_other, top_k=tfidf_top_k, min_score=0.15)
            if verbose:
                n_pairs = sum(len(v) for v in tfidf_cands.values())
                print(f"        → {len(tfidf_cands):,} S1 entities, {n_pairs:,} pairs")

        # Strategy 4: Numeric token blocking
        if verbose:
            step_num = 4 if use_tfidf else 3
            print(f"  [{step_num}/{n_strategies}] Numeric token blocking...")
        num_cands = numeric_token_block(s1, s_other, min_shared=1)
        if verbose:
            n_pairs = sum(len(v) for v in num_cands.values())
            print(f"        → {len(num_cands):,} S1 entities, {n_pairs:,} pairs")

        # Strategy 5: Embedding-based semantic blocking (if engine provided)
        emb_cands = {}
        if embedding_engine is not None:
            if verbose:
                print(f"  [{n_strategies}/{n_strategies}] Embedding semantic blocking...")
            from src.embeddings import embedding_block
            emb_cands = embedding_block(
                s1, s_other, embedding_engine,
                top_k=embedding_top_k, min_score=embedding_min_score,
            )
            if verbose:
                n_pairs = sum(len(v) for v in emb_cands.values())
                print(f"        → {len(emb_cands):,} S1 entities, {n_pairs:,} pairs")

        # Union all strategies
        merged = union_candidates(name_cands, ngram_cands, tfidf_cands, num_cands, emb_cands)
        for s1_eid, cands in merged.items():
            all_candidates[s1_eid].update(cands)

    # Ensure every S1 entity has an entry (even if empty)
    for _, row in s1.iterrows():
        if row["entity_id"] not in all_candidates:
            all_candidates[row["entity_id"]] = set()

    if verbose:
        total_pairs = sum(len(v) for v in all_candidates.values())
        non_empty = sum(1 for v in all_candidates.values() if v)
        avg_cands = total_pairs / max(len(all_candidates), 1)
        print(f"\n  TOTAL: {len(all_candidates):,} S1 entities")
        print(f"         {non_empty:,} with candidates, "
              f"{len(all_candidates) - non_empty:,} without")
        print(f"         {total_pairs:,} total pairs, "
              f"{avg_cands:.1f} avg candidates per S1")

    return dict(all_candidates)


def measure_blocking_recall(candidates: dict, gt: pd.DataFrame) -> float:
    """Measure what fraction of true matches are present in candidate pairs.

    This is the most critical blocking metric — if a true match is not
    in the candidate set, the ML model can never recover it.
    """
    total_true = 0
    found_true = 0

    for _, row in gt.iterrows():
        s1_eid = row["source1_entity_id"]
        true_matches = row["matched_list"]
        if not true_matches:
            continue

        cand_set = candidates.get(s1_eid, set())
        for match_eid in true_matches:
            total_true += 1
            if match_eid in cand_set:
                found_true += 1

    recall = found_true / max(total_true, 1)
    print(f"\n  Blocking Recall: {recall:.4f} ({found_true}/{total_true})")
    return recall
