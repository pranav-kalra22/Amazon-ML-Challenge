"""
inference.py — Test-time inference and output file generation.

Takes the trained model and runs the full pipeline on test data:
    1. Normalize test records
    2. Generate candidate pairs via blocking
    3. Compute features for all pairs
    4. Predict match probabilities
    5. Apply threshold
    6. Generate matching_results.tsv and candidate_pairs.tsv
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict


def run_inference(
    model,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    candidates: dict,
    s1_lookup: dict,
    other_lookup: dict,
    tfidf_engine=None,
    embedding_engine=None,
    threshold: float = None,
    output_dir: str = "output",
    verbose: bool = True,
):
    """Run full inference pipeline and generate output TSV files.

    Args:
        model: Trained EntityMatchModel
        s1, s2, s3: Normalized test DataFrames
        candidates: dict of s1_entity_id → set of candidate IDs
        s1_lookup: dict of entity_id → row dict for S1
        other_lookup: dict of entity_id → row dict for S2/S3
        tfidf_engine: Pre-fitted TF-IDF engine
        threshold: Match probability threshold (uses model's best if None)
        output_dir: Directory to write output files

    Returns:
        dict of s1_entity_id → set of matched entity IDs
    """
    from src.features import build_feature_matrix, get_feature_columns

    threshold = threshold or model.best_threshold
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n=== Test Inference ===")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  S1 entities: {len(s1):,}")

    # ── Build features for all candidate pairs ────────────────────────────
    feature_df = build_feature_matrix(
        candidates, s1_lookup, other_lookup, tfidf_engine,
        embedding_engine=embedding_engine, verbose=verbose
    )

    # ── Predict match probabilities ──────────────────────────────────────
    if len(feature_df) > 0:
        probas = model.predict_proba(feature_df)
        feature_df["proba"] = probas
        feature_df["prediction"] = (probas >= threshold).astype(int)
    else:
        feature_df["proba"] = []
        feature_df["prediction"] = []

    # ── Build match sets per S1 entity ───────────────────────────────────
    match_dict = defaultdict(set)
    if len(feature_df) > 0:
        matches = feature_df[feature_df["prediction"] == 1]
        for _, row in matches.iterrows():
            match_dict[row["s1_id"]].add(row["s2s3_id"])

    # Ensure every S1 entity appears
    all_s1_ids = sorted(s1["entity_id"].unique())
    for s1_id in all_s1_ids:
        if s1_id not in match_dict:
            match_dict[s1_id] = set()

    # ── Generate matching_results.tsv ────────────────────────────────────
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = sorted(match_dict.get(s1_id, set()))
            matched_str = ",".join(matched) if matched else ""
            f.write(f"{s1_id}\t{matched_str}\n")

    if verbose:
        n_matched = sum(1 for v in match_dict.values() if v)
        total_matches = sum(len(v) for v in match_dict.values())
        print(f"\n  matching_results.tsv written to {matching_path}")
        print(f"    {len(all_s1_ids):,} S1 entities")
        print(f"    {n_matched:,} with matches, "
              f"{len(all_s1_ids) - n_matched:,} singletons")
        print(f"    {total_matches:,} total matches")

    # ── Generate candidate_pairs.tsv ─────────────────────────────────────
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = sorted(candidates.get(s1_id, set()))
            cands_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cands_str}\n")

    if verbose:
        total_cands = sum(len(v) for v in candidates.values())
        print(f"\n  candidate_pairs.tsv written to {candidate_path}")
        print(f"    {total_cands:,} total candidate pairs")

    return dict(match_dict)


def generate_submission_package(
    output_dir: str = "output",
    code_dir: str = ".",
    team_name: str = "team",
):
    """Print instructions for generating the submission zip."""
    print(f"\n=== Submission Package ===")
    print(f"  Create: {team_name}_submission.zip containing:")
    print(f"    output/")
    print(f"      matching_results.tsv")
    print(f"      candidate_pairs.tsv")
    print(f"    code/")
    print(f"      business_entity_resolution/")
    print(f"        src/")
    print(f"        README.md")
    print(f"        requirements.txt")
    print(f"    Documentation_template.md")
    print(f"\n  Validate with:")
    print(f"    python utils/validate_submission.py \\")
    print(f"      --matching {output_dir}/matching_results.tsv \\")
    print(f"      --candidate {output_dir}/candidate_pairs.tsv \\")
    print(f"      --test-dir dataset/test")
