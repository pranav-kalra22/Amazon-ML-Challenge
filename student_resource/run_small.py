"""
run_small.py — Run the full entity resolution pipeline on a small data sample.

Samples N Source-1 entities from training data and their corresponding
S2/S3 matches, then runs blocking → features → training → validation → output.

Usage:
    python run_small.py --n 500
    python run_small.py --n 1000 --no-tfidf-blocking
    python run_small.py --n 200 --train-only
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_source, load_ground_truth, build_lookup
from src.normalization import normalize_dataframe
from src.blocking import generate_candidates, measure_blocking_recall
from src.features import (
    build_feature_matrix,
    get_feature_columns,
    TfidfSimilarityEngine,
)
from src.model import EntityMatchModel
from src.evaluation import evaluate_predictions, error_analysis
from src.inference import run_inference


def sample_data(data_dir: str, n_s1: int = 500, seed: int = 42):
    """Load and sample a small subset of training data.

    Samples `n_s1` Source-1 entities, then keeps only the S2/S3 records
    that are referenced in the ground truth for those entities (plus a
    random sample of non-matching records to keep blocking realistic).

    Returns:
        s1, s2, s3, gt — sampled DataFrames
    """
    print(f"\n=== Sampling {n_s1} S1 entities from training data ===")
    train_dir = os.path.join(data_dir, "train")

    # Load ground truth first (small file)
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    # Sample S1 entity IDs
    rng = np.random.RandomState(seed)
    all_s1_ids = gt["source1_entity_id"].unique()
    if n_s1 >= len(all_s1_ids):
        sampled_s1_ids = set(all_s1_ids)
    else:
        sampled_s1_ids = set(rng.choice(all_s1_ids, size=n_s1, replace=False))

    gt_sampled = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)

    # Collect all S2/S3 IDs referenced in ground truth
    referenced_s2_ids = set()
    referenced_s3_ids = set()
    for _, row in gt_sampled.iterrows():
        for mid in row["matched_list"]:
            if mid.startswith("S2-"):
                referenced_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                referenced_s3_ids.add(mid)

    print(f"  Sampled S1 entities: {len(sampled_s1_ids)}")
    print(f"  Referenced S2 IDs in ground truth: {len(referenced_s2_ids)}")
    print(f"  Referenced S3 IDs in ground truth: {len(referenced_s3_ids)}")

    # Load source files using chunked reading to handle large files
    print("\n  Loading Source 1 (sampling)...")
    s1_chunks = []
    for chunk in pd.read_csv(
        os.path.join(train_dir, "train_source1.tsv"),
        sep="\t", dtype=str, keep_default_na=False, chunksize=50000
    ):
        filtered = chunk[chunk["entity_id"].isin(sampled_s1_ids)]
        if len(filtered) > 0:
            s1_chunks.append(filtered)
        # Stop early if we found all
        if sum(len(c) for c in s1_chunks) >= len(sampled_s1_ids):
            break
    s1 = pd.concat(s1_chunks, ignore_index=True) if s1_chunks else pd.DataFrame()
    for col in s1.columns:
        s1[col] = s1[col].str.strip()
    print(f"    → {len(s1)} S1 records loaded")

    # For S2: load referenced IDs + random sample of others for realistic blocking
    print("  Loading Source 2 (sampling)...")
    n_extra_s2 = max(len(referenced_s2_ids) * 5, 2000)  # 5x negatives or at least 2000
    s2_ref_chunks = []
    s2_extra_chunks = []
    extra_count = 0
    for chunk in pd.read_csv(
        os.path.join(train_dir, "train_source2.tsv"),
        sep="\t", dtype=str, keep_default_na=False, chunksize=50000
    ):
        # Keep referenced IDs
        ref = chunk[chunk["entity_id"].isin(referenced_s2_ids)]
        if len(ref) > 0:
            s2_ref_chunks.append(ref)

        # Sample extras (non-referenced)
        non_ref = chunk[~chunk["entity_id"].isin(referenced_s2_ids)]
        if extra_count < n_extra_s2 and len(non_ref) > 0:
            sample_n = min(len(non_ref), n_extra_s2 - extra_count)
            s2_extra_chunks.append(non_ref.sample(n=sample_n, random_state=seed))
            extra_count += sample_n

        if extra_count >= n_extra_s2 and sum(len(c) for c in s2_ref_chunks) >= len(referenced_s2_ids):
            break

    s2_parts = s2_ref_chunks + s2_extra_chunks
    s2 = pd.concat(s2_parts, ignore_index=True) if s2_parts else pd.DataFrame()
    for col in s2.columns:
        s2[col] = s2[col].str.strip()
    print(f"    → {len(s2)} S2 records ({len(referenced_s2_ids)} referenced + extras)")

    # For S3: same approach
    print("  Loading Source 3 (sampling)...")
    n_extra_s3 = max(len(referenced_s3_ids) * 5, 2000)
    s3_ref_chunks = []
    s3_extra_chunks = []
    extra_count = 0
    for chunk in pd.read_csv(
        os.path.join(train_dir, "train_source3.tsv"),
        sep="\t", dtype=str, keep_default_na=False, chunksize=50000
    ):
        ref = chunk[chunk["entity_id"].isin(referenced_s3_ids)]
        if len(ref) > 0:
            s3_ref_chunks.append(ref)

        non_ref = chunk[~chunk["entity_id"].isin(referenced_s3_ids)]
        if extra_count < n_extra_s3 and len(non_ref) > 0:
            sample_n = min(len(non_ref), n_extra_s3 - extra_count)
            s3_extra_chunks.append(non_ref.sample(n=sample_n, random_state=seed))
            extra_count += sample_n

        if extra_count >= n_extra_s3 and sum(len(c) for c in s3_ref_chunks) >= len(referenced_s3_ids):
            break

    s3_parts = s3_ref_chunks + s3_extra_chunks
    s3 = pd.concat(s3_parts, ignore_index=True) if s3_parts else pd.DataFrame()
    for col in s3.columns:
        s3[col] = s3[col].str.strip()
    print(f"    → {len(s3)} S3 records ({len(referenced_s3_ids)} referenced + extras)")

    return s1, s2, s3, gt_sampled


def build_ground_truth_dict(gt: pd.DataFrame) -> dict:
    """Convert ground truth DataFrame to dict: s1_id → set of match IDs."""
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches = row["matched_list"]
        gt_dict[s1_id] = set(matches)
    return gt_dict


def build_training_labels(feature_df: pd.DataFrame, gt_dict: dict) -> np.ndarray:
    """Create binary labels from ground truth."""
    labels = []
    for _, row in feature_df.iterrows():
        s1_id = row["s1_id"]
        s2s3_id = row["s2s3_id"]
        true_matches = gt_dict.get(s1_id, set())
        labels.append(1 if s2s3_id in true_matches else 0)
    return np.array(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Run Entity Resolution pipeline on a small data sample"
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of S1 entities to sample (default: 500)",
    )
    parser.add_argument(
        "--data-dir", default="dataset",
        help="Root dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Model save directory (default: models)",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train/validate, don't generate test output",
    )
    parser.add_argument(
        "--no-tfidf-blocking", action="store_true",
        help="Disable TF-IDF blocking (faster)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    use_tfidf = not args.no_tfidf_blocking
    start_time = time.time()

    print("=" * 60)
    print(f"Entity Resolution Pipeline — Small Sample Mode (N={args.n})")
    print("=" * 60)

    # ── Step 1: Sample Data ──────────────────────────────────────────────
    s1, s2, s3, gt = sample_data(args.data_dir, n_s1=args.n, seed=args.seed)

    if len(s1) == 0:
        print("ERROR: No S1 records loaded. Check your data directory.")
        sys.exit(1)

    # ── Step 2: Normalize ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Normalization")
    print("=" * 60)
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    print("  Done.")

    # ── Step 3: Train/Val Split (entity-level) ───────────────────────────
    gt_dict = build_ground_truth_dict(gt)
    rng = np.random.RandomState(args.seed)
    all_s1_ids = list(gt["source1_entity_id"].unique())
    rng.shuffle(all_s1_ids)

    split_idx = int(len(all_s1_ids) * 0.8)
    train_ids = set(all_s1_ids[:split_idx])
    val_ids = set(all_s1_ids[split_idx:])

    gt_train_dict = {k: v for k, v in gt_dict.items() if k in train_ids}
    gt_val_dict = {k: v for k, v in gt_dict.items() if k in val_ids}

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"\n  Train S1: {len(s1_train)}, Val S1: {len(s1_val)}")
    print(f"  Train matches: {sum(len(v) for v in gt_train_dict.values())}")
    print(f"  Val matches: {sum(len(v) for v in gt_val_dict.values())}")

    # ── Step 4: Candidate Generation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Candidate Generation (Blocking)")
    print("=" * 60)

    print("\n--- Training candidates ---")
    train_candidates = generate_candidates(
        s1_train, s2, s3, use_tfidf=use_tfidf, tfidf_top_k=15
    )
    gt_train_df = gt[gt["source1_entity_id"].isin(train_ids)]
    measure_blocking_recall(train_candidates, gt_train_df)

    print("\n--- Validation candidates ---")
    val_candidates = generate_candidates(
        s1_val, s2, s3, use_tfidf=use_tfidf, tfidf_top_k=15
    )
    gt_val_df = gt[gt["source1_entity_id"].isin(val_ids)]
    measure_blocking_recall(val_candidates, gt_val_df)

    # ── Step 5: Feature Engineering ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Feature Engineering")
    print("=" * 60)

    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1.iterrows()}
    other_lookup = build_lookup(s2, s3)

    print("  Fitting TF-IDF engine...")
    tfidf_engine = TfidfSimilarityEngine()
    all_entities = pd.concat([s1, s2, s3], ignore_index=True)
    tfidf_engine.fit(all_entities)

    print("\n--- Training features ---")
    train_features = build_feature_matrix(
        train_candidates, s1_lookup, other_lookup, tfidf_engine
    )
    train_labels = build_training_labels(train_features, gt_train_dict)
    train_features["label"] = train_labels
    print(f"  Pos: {train_labels.sum()}, Neg: {int((1-train_labels).sum())}")

    print("\n--- Validation features ---")
    val_features = build_feature_matrix(
        val_candidates, s1_lookup, other_lookup, tfidf_engine
    )
    val_labels = build_training_labels(val_features, gt_val_dict)
    val_features["label"] = val_labels

    # ── Step 6: Train Model ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Training ML Model")
    print("=" * 60)

    feature_cols = get_feature_columns(train_features)

    # Use fewer trees for small data
    params = EntityMatchModel._default_params()
    params["n_estimators"] = 500
    params["num_leaves"] = 31
    model = EntityMatchModel(params=params)

    model.train(
        X_train=train_features,
        y_train=train_labels,
        X_val=val_features,
        y_val=val_labels,
        feature_columns=feature_cols,
    )

    model.feature_importance(top_n=15)

    # ── Step 7: Threshold Optimization ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6: Threshold Optimization")
    print("=" * 60)

    model.optimize_threshold(
        X_val=val_features,
        y_val=val_labels,
        s1_ids=val_features["s1_id"].values,
        gt_dict=gt_val_dict,
    )

    # ── Step 8: Validation Evaluation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 7: Validation Evaluation")
    print("=" * 60)

    val_probas = model.predict_proba(val_features)
    val_preds = (val_probas >= model.best_threshold).astype(int)

    val_pred_dict = defaultdict(set)
    for i in range(len(val_features)):
        if val_preds[i] == 1:
            val_pred_dict[val_features.iloc[i]["s1_id"]].add(
                val_features.iloc[i]["s2s3_id"]
            )
    for s1_id in val_ids:
        if s1_id not in val_pred_dict:
            val_pred_dict[s1_id] = set()

    results = evaluate_predictions(dict(val_pred_dict), gt_val_dict)

    error_analysis(
        dict(val_pred_dict), gt_val_dict, s1_lookup, other_lookup, top_n=5
    )

    # ── Step 9: Save Model ───────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model.save(os.path.join(args.model_dir, "model.joblib"))

    # ── Step 10: Generate Output (on the sampled validation set) ─────────
    if not args.train_only:
        print("\n" + "=" * 60)
        print("PHASE 8: Generating Output Files (on sampled data)")
        print("=" * 60)

        # Use the full sampled S1 as "test" data to generate output
        os.makedirs(args.output_dir, exist_ok=True)
        all_candidates = generate_candidates(
            s1, s2, s3, use_tfidf=use_tfidf, tfidf_top_k=15, verbose=False
        )

        match_dict = run_inference(
            model=model,
            s1=s1,
            s2=s2,
            s3=s3,
            candidates=all_candidates,
            s1_lookup=s1_lookup,
            other_lookup=other_lookup,
            tfidf_engine=tfidf_engine,
            output_dir=args.output_dir,
        )

        # Evaluate on full sampled ground truth
        print("\n  Full-sample evaluation:")
        evaluate_predictions(match_dict, gt_dict)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")

    if not args.train_only:
        print(f"\nOutput files:")
        print(f"  {args.output_dir}/matching_results.tsv")
        print(f"  {args.output_dir}/candidate_pairs.tsv")
        print(f"\nTo validate:")
        print(f"  python utils/validate_submission.py \\")
        print(f"    --matching {args.output_dir}/matching_results.tsv \\")
        print(f"    --candidate {args.output_dir}/candidate_pairs.tsv \\")
        print(f"    --test-dir dataset/test")


if __name__ == "__main__":
    main()
