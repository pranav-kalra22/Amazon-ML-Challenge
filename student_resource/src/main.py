"""
main.py — End-to-end Entity Resolution pipeline.

Orchestrates the full workflow:
    Phase 1: Load data
    Phase 2: Normalize
    Phase 3: Candidate generation (blocking)
    Phase 4: Feature engineering
    Phase 5: Train/validate ML model
    Phase 6: Threshold optimization
    Phase 7: Error analysis
    Phase 8: Test inference → output TSVs

Usage:
    # Full pipeline (train + test)
    python -m src.main --data-dir dataset --output-dir output

    # Train only (validation)
    python -m src.main --data-dir dataset --train-only

    # Test only (requires saved model)
    python -m src.main --data-dir dataset --test-only --model-path models/model.joblib
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from src.data_loader import load_train_data, load_test_data, build_lookup
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


def build_ground_truth_dict(gt: pd.DataFrame) -> dict:
    """Convert ground truth DataFrame to dict: s1_id → set of match IDs."""
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches = row["matched_list"]
        gt_dict[s1_id] = set(matches)
    return gt_dict


def build_training_labels(
    feature_df: pd.DataFrame,
    gt_dict: dict,
) -> np.ndarray:
    """Create binary labels for the feature matrix based on ground truth.

    A pair (s1_id, s2s3_id) gets label 1 if s2s3_id ∈ gt_dict[s1_id].
    """
    labels = []
    for _, row in feature_df.iterrows():
        s1_id = row["s1_id"]
        s2s3_id = row["s2s3_id"]
        true_matches = gt_dict.get(s1_id, set())
        labels.append(1 if s2s3_id in true_matches else 0)
    return np.array(labels)


def split_by_entity(
    gt: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple:
    """Split S1 entity IDs into train/validation sets.

    Split is done at the entity level to prevent data leakage.
    """
    rng = np.random.RandomState(seed)
    s1_ids = gt["source1_entity_id"].unique()
    rng.shuffle(s1_ids)

    split_idx = int(len(s1_ids) * train_ratio)
    train_ids = set(s1_ids[:split_idx])
    val_ids = set(s1_ids[split_idx:])

    print(f"  Entity-level split: {len(train_ids):,} train, {len(val_ids):,} val")
    return train_ids, val_ids


def run_training_pipeline(
    data_dir: str,
    output_dir: str = "output",
    model_dir: str = "models",
    train_only: bool = False,
    use_tfidf_blocking: bool = True,
    tfidf_top_k: int = 20,
):
    """Run the full training + validation pipeline."""
    start = time.time()

    # ── Phase 1: Load Data ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Loading Data")
    print("=" * 60)
    s1, s2, s3, gt = load_train_data(data_dir)

    # ── Phase 2: Normalize ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Normalization")
    print("=" * 60)
    print("  Normalizing Source 1...")
    s1 = normalize_dataframe(s1)
    print("  Normalizing Source 2...")
    s2 = normalize_dataframe(s2)
    print("  Normalizing Source 3...")
    s3 = normalize_dataframe(s3)
    print("  Done.")

    # Build ground truth dict
    gt_dict = build_ground_truth_dict(gt)

    # ── Split into train / validation entities ───────────────────────────
    print("\n  Splitting data by entity...")
    train_ids, val_ids = split_by_entity(gt, train_ratio=0.8)

    # Filter ground truth
    gt_train_dict = {k: v for k, v in gt_dict.items() if k in train_ids}
    gt_val_dict = {k: v for k, v in gt_dict.items() if k in val_ids}

    # Filter S1 entities
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"  S1 train: {len(s1_train):,}")
    print(f"  S1 val:   {len(s1_val):,}")

    # ── Phase 3: Candidate Generation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Candidate Generation (Blocking)")
    print("=" * 60)

    # Generate candidates for training split
    print("\n--- Training candidates ---")
    train_candidates = generate_candidates(
        s1_train, s2, s3,
        use_tfidf=use_tfidf_blocking,
        tfidf_top_k=tfidf_top_k,
    )
    print("\n  Measuring blocking recall on training set:")
    gt_train_df = gt[gt["source1_entity_id"].isin(train_ids)]
    measure_blocking_recall(train_candidates, gt_train_df)

    # Generate candidates for validation split
    print("\n--- Validation candidates ---")
    val_candidates = generate_candidates(
        s1_val, s2, s3,
        use_tfidf=use_tfidf_blocking,
        tfidf_top_k=tfidf_top_k,
    )
    print("\n  Measuring blocking recall on validation set:")
    gt_val_df = gt[gt["source1_entity_id"].isin(val_ids)]
    measure_blocking_recall(val_candidates, gt_val_df)

    # ── Phase 4: Feature Engineering ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Feature Engineering")
    print("=" * 60)

    # Build lookups
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1.iterrows()}
    other_lookup = build_lookup(s2, s3)

    # Fit TF-IDF engine on all entities
    print("\n  Fitting TF-IDF engine...")
    tfidf_engine = TfidfSimilarityEngine()
    all_entities = pd.concat([s1, s2, s3], ignore_index=True)
    tfidf_engine.fit(all_entities)
    print("  Done.")

    # Build training feature matrix
    print("\n--- Training features ---")
    train_features = build_feature_matrix(
        train_candidates, s1_lookup, other_lookup, tfidf_engine
    )
    train_labels = build_training_labels(train_features, gt_train_dict)
    train_features["label"] = train_labels

    print(f"  Positives: {train_labels.sum():,} ({train_labels.mean()*100:.2f}%)")
    print(f"  Negatives: {(1 - train_labels).sum():,.0f}")

    # Build validation feature matrix
    print("\n--- Validation features ---")
    val_features = build_feature_matrix(
        val_candidates, s1_lookup, other_lookup, tfidf_engine
    )
    val_labels = build_training_labels(val_features, gt_val_dict)
    val_features["label"] = val_labels

    # ── Phase 5: Train Model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Training ML Model")
    print("=" * 60)

    feature_cols = get_feature_columns(train_features)
    model = EntityMatchModel()
    model.train(
        X_train=train_features,
        y_train=train_labels,
        X_val=val_features,
        y_val=val_labels,
        feature_columns=feature_cols,
    )

    # Feature importance
    model.feature_importance(top_n=20)

    # ── Phase 6: Threshold Optimization ──────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6: Threshold Optimization")
    print("=" * 60)

    model.optimize_threshold(
        X_val=val_features,
        y_val=val_labels,
        s1_ids=val_features["s1_id"].values,
        gt_dict=gt_val_dict,
    )

    # ── Phase 7: Validation Evaluation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 7: Validation Evaluation")
    print("=" * 60)

    val_probas = model.predict_proba(val_features)
    val_preds_binary = (val_probas >= model.best_threshold).astype(int)

    # Build predicted match sets
    val_pred_dict = defaultdict(set)
    for i in range(len(val_features)):
        if val_preds_binary[i] == 1:
            val_pred_dict[val_features.iloc[i]["s1_id"]].add(
                val_features.iloc[i]["s2s3_id"]
            )
    # Include singletons
    for s1_id in val_ids:
        if s1_id not in val_pred_dict:
            val_pred_dict[s1_id] = set()

    results = evaluate_predictions(dict(val_pred_dict), gt_val_dict)

    # Error analysis
    error_analysis(
        dict(val_pred_dict), gt_val_dict, s1_lookup, other_lookup, top_n=5
    )

    # ── Save Model ───────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    model.save(model_path)

    elapsed = time.time() - start
    print(f"\n  Training pipeline completed in {elapsed/60:.1f} minutes")

    return model, tfidf_engine, s1, s2, s3


def run_test_pipeline(
    data_dir: str,
    output_dir: str = "output",
    model_path: str = "models/model.joblib",
    model=None,
    tfidf_engine=None,
    train_s1=None,
    train_s2=None,
    train_s3=None,
    use_tfidf_blocking: bool = True,
    tfidf_top_k: int = 20,
):
    """Run inference on test data and generate output files."""
    start = time.time()

    print("\n" + "=" * 60)
    print("PHASE 8: Test Inference")
    print("=" * 60)

    # Load model if not provided
    if model is None:
        model = EntityMatchModel()
        model.load(model_path)

    # ── Load and normalize test data ─────────────────────────────────────
    s1_test, s2_test, s3_test = load_test_data(data_dir)
    print("\n  Normalizing test data...")
    s1_test = normalize_dataframe(s1_test)
    s2_test = normalize_dataframe(s2_test)
    s3_test = normalize_dataframe(s3_test)

    # ── Fit TF-IDF on test data if not provided ──────────────────────────
    if tfidf_engine is None:
        print("\n  Fitting TF-IDF on test data...")
        tfidf_engine = TfidfSimilarityEngine()
        all_test = pd.concat([s1_test, s2_test, s3_test], ignore_index=True)
        tfidf_engine.fit(all_test)

    # ── Generate candidates ──────────────────────────────────────────────
    test_candidates = generate_candidates(
        s1_test, s2_test, s3_test,
        use_tfidf=use_tfidf_blocking,
        tfidf_top_k=tfidf_top_k,
    )

    # Build lookups
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1_test.iterrows()}
    other_lookup = build_lookup(s2_test, s3_test)

    # ── Run inference ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    match_dict = run_inference(
        model=model,
        s1=s1_test,
        s2=s2_test,
        s3=s3_test,
        candidates=test_candidates,
        s1_lookup=s1_lookup,
        other_lookup=other_lookup,
        tfidf_engine=tfidf_engine,
        output_dir=output_dir,
    )

    elapsed = time.time() - start
    print(f"\n  Test inference completed in {elapsed/60:.1f} minutes")

    return match_dict


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="dataset",
        help="Root directory containing train/ and test/ subdirectories",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for output TSV files",
    )
    parser.add_argument(
        "--model-dir", "-m",
        default="models",
        help="Directory for saved models",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training and validation only (no test inference)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run test inference only (requires --model-path)",
    )
    parser.add_argument(
        "--model-path",
        default="models/model.joblib",
        help="Path to saved model for test-only mode",
    )
    parser.add_argument(
        "--no-tfidf-blocking",
        action="store_true",
        help="Disable TF-IDF blocking (faster but lower recall)",
    )
    parser.add_argument(
        "--tfidf-top-k",
        type=int,
        default=20,
        help="Number of top TF-IDF candidates per entity (default: 20)",
    )
    args = parser.parse_args()

    use_tfidf = not args.no_tfidf_blocking

    print("=" * 60)
    print("Business Entity Resolution Pipeline")
    print("=" * 60)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Model dir:   {args.model_dir}")
    print(f"  TF-IDF blocking: {'ON' if use_tfidf else 'OFF'}")
    print(f"  TF-IDF top-K: {args.tfidf_top_k}")

    if args.test_only:
        run_test_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            use_tfidf_blocking=use_tfidf,
            tfidf_top_k=args.tfidf_top_k,
        )
    else:
        # Training pipeline
        model, tfidf_engine, s1, s2, s3 = run_training_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            train_only=args.train_only,
            use_tfidf_blocking=use_tfidf,
            tfidf_top_k=args.tfidf_top_k,
        )

        if not args.train_only:
            # Test pipeline (reuses model and TF-IDF from training)
            run_test_pipeline(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                model=model,
                tfidf_engine=None,  # Refit on test data
                use_tfidf_blocking=use_tfidf,
                tfidf_top_k=args.tfidf_top_k,
            )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
