"""
evaluation.py — Evaluation utilities for entity resolution.

Computes entity-level F0.5 (macro-averaged over S1 entities) matching the
competition's scoring methodology. Also provides error analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_f05(precision: float, recall: float) -> float:
    """Compute F0.5 score."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_predictions(
    predictions: dict,
    ground_truth: dict,
    verbose: bool = True,
) -> dict:
    """Evaluate predicted matches against ground truth.

    Args:
        predictions: dict of s1_entity_id → set of predicted match IDs
        ground_truth: dict of s1_entity_id → set of true match IDs

    Returns:
        dict with macro-averaged precision, recall, F0.5, and per-entity details
    """
    entity_scores = []

    for s1_id in sorted(ground_truth.keys()):
        true_set = ground_truth[s1_id]
        pred_set = predictions.get(s1_id, set())

        if len(pred_set) == 0 and len(true_set) == 0:
            p, r, f05 = 1.0, 1.0, 1.0
            category = "true_singleton"
        elif len(pred_set) == 0 and len(true_set) > 0:
            p, r, f05 = 1.0, 0.0, 0.0  # vacuous precision, no recall
            category = "false_negative_singleton"
        elif len(pred_set) > 0 and len(true_set) == 0:
            p, r, f05 = 0.0, 1.0, 0.0  # false alarm
            category = "false_positive_singleton"
        else:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            f05 = compute_f05(p, r)
            if tp == len(true_set) and tp == len(pred_set):
                category = "perfect_match"
            elif tp > 0:
                category = "partial_match"
            else:
                category = "complete_mismatch"

        entity_scores.append({
            "s1_id": s1_id,
            "precision": p,
            "recall": r,
            "f05": f05,
            "category": category,
            "n_true": len(true_set),
            "n_pred": len(pred_set),
            "n_tp": len(pred_set & true_set) if pred_set and true_set else 0,
        })

    df_scores = pd.DataFrame(entity_scores)

    macro_precision = df_scores["precision"].mean()
    macro_recall = df_scores["recall"].mean()
    macro_f05 = df_scores["f05"].mean()

    results = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f05": macro_f05,
        "entity_scores": df_scores,
    }

    if verbose:
        print(f"\n=== Evaluation Results ===")
        print(f"  Macro Precision: {macro_precision:.4f}")
        print(f"  Macro Recall:    {macro_recall:.4f}")
        print(f"  Macro F0.5:      {macro_f05:.4f}")
        print(f"\n  Entity categories:")
        for cat, count in df_scores["category"].value_counts().items():
            print(f"    {cat}: {count}")

    return results


def error_analysis(
    predictions: dict,
    ground_truth: dict,
    s1_data: dict,
    other_data: dict,
    top_n: int = 10,
) -> dict:
    """Analyze false positives and false negatives.

    Args:
        predictions: dict of s1_entity_id → set of predicted match IDs
        ground_truth: dict of s1_entity_id → set of true match IDs
        s1_data: dict of entity_id → row dict for S1 entities
        other_data: dict of entity_id → row dict for S2/S3 entities
        top_n: Number of examples to show

    Returns:
        dict with 'false_positives' and 'false_negatives' lists
    """
    false_positives = []
    false_negatives = []

    for s1_id in ground_truth:
        true_set = ground_truth[s1_id]
        pred_set = predictions.get(s1_id, set())

        # False positives: predicted but not true
        fps = pred_set - true_set
        for fp_id in fps:
            s1_info = s1_data.get(s1_id, {})
            other_info = other_data.get(fp_id, {})
            false_positives.append({
                "s1_id": s1_id,
                "fp_id": fp_id,
                "s1_name": s1_info.get("business_name", ""),
                "other_name": other_info.get("business_name", ""),
                "s1_addr": s1_info.get("business_address", ""),
                "other_addr": other_info.get("business_address", ""),
                "s1_country": s1_info.get("country", ""),
                "other_country": other_info.get("country", ""),
            })

        # False negatives: true but not predicted
        fns = true_set - pred_set
        for fn_id in fns:
            s1_info = s1_data.get(s1_id, {})
            other_info = other_data.get(fn_id, {})
            false_negatives.append({
                "s1_id": s1_id,
                "fn_id": fn_id,
                "s1_name": s1_info.get("business_name", ""),
                "other_name": other_info.get("business_name", ""),
                "s1_addr": s1_info.get("business_address", ""),
                "other_addr": other_info.get("business_address", ""),
                "s1_country": s1_info.get("country", ""),
                "other_country": other_info.get("country", ""),
            })

    print(f"\n=== Error Analysis ===")
    print(f"  Total false positives: {len(false_positives)}")
    print(f"  Total false negatives: {len(false_negatives)}")

    if false_positives:
        print(f"\n  --- Top {min(top_n, len(false_positives))} False Positives ---")
        for fp in false_positives[:top_n]:
            print(f"    {fp['s1_id']} ↔ {fp['fp_id']}")
            print(f"      S1 Name:    {fp['s1_name']}")
            print(f"      Other Name: {fp['other_name']}")
            print(f"      S1 Addr:    {fp['s1_addr']}")
            print(f"      Other Addr: {fp['other_addr']}")
            print()

    if false_negatives:
        print(f"\n  --- Top {min(top_n, len(false_negatives))} False Negatives ---")
        for fn in false_negatives[:top_n]:
            print(f"    {fn['s1_id']} ↔ {fn['fn_id']}")
            print(f"      S1 Name:    {fn['s1_name']}")
            print(f"      Other Name: {fn['other_name']}")
            print(f"      S1 Addr:    {fn['s1_addr']}")
            print(f"      Other Addr: {fn['other_addr']}")
            print()

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
