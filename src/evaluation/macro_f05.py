#!/usr/bin/env python3
"""
Authoritative Entity-Level Macro F0.5 Evaluator for Amazon ML Challenge 2026.

Implements set-based macro evaluation across Source 1 entities with full singleton handling:
- Truth empty + Prediction empty -> 1.0
- Truth empty + Prediction non-empty -> 0.0
- Truth non-empty + Prediction empty -> 0.0
- Truth non-empty + Prediction non-empty -> standard F_0.5 formula
"""

from typing import Dict, Set, List, Optional, Any
import numpy as np


def compute_entity_f05(
    truth: Set[str],
    predicted: Set[str],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes precision, recall, and F_beta for a single Source 1 entity.
    
    Returns dict with keys: 'precision', 'recall', 'f05', 'is_singleton', 'correct_singleton'.
    """
    beta_sq = beta ** 2
    n_truth = len(truth)
    n_pred = len(predicted)
    
    # Singleton cases
    if n_truth == 0:
        if n_pred == 0:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f05": 1.0,
                "is_singleton": True,
                "correct_singleton": True
            }
        else:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f05": 0.0,
                "is_singleton": True,
                "correct_singleton": False
            }
            
    # Non-singleton, but prediction is empty
    if n_pred == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
            "is_singleton": False,
            "correct_singleton": False
        }
        
    # Non-singleton with predictions
    tp = len(truth & predicted)
    prec = tp / n_pred
    rec = tp / n_truth
    
    if prec + rec == 0:
        f_score = 0.0
    else:
        # F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
        f_score = ((1.0 + beta_sq) * prec * rec) / (beta_sq * prec + rec)
        
    return {
        "precision": prec,
        "recall": rec,
        "f05": f_score,
        "is_singleton": False,
        "correct_singleton": False
    }


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Computes macro-averaged F0.5 across all Source 1 entities in ground_truth.
    
    Parameters:
    - ground_truth: dict mapping source1_entity_id -> set of matched entity IDs
    - predictions: dict mapping source1_entity_id -> set of predicted entity IDs
    - entity_metadata: optional dict mapping source1_entity_id -> dict of metadata (e.g. {'country': 'US'})
    
    Returns:
    - Comprehensive evaluation dict with macro metrics and breakdowns.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        raise ValueError("Ground truth dictionary is empty.")
        
    precisions = []
    recalls = []
    f05_scores = []
    
    singleton_count = 0
    correct_singletons = 0
    non_singleton_f05s = []
    
    # Country-level breakdown tracking
    country_scores = {}
    # Source-level match counts
    s2_true, s2_pred, s2_tp = 0, 0, 0
    s3_true, s3_pred, s3_tp = 0, 0, 0
    
    for s1_id, truth_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        res = compute_entity_f05(truth_set, pred_set)
        
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f05_scores.append(res["f05"])
        
        if res["is_singleton"]:
            singleton_count += 1
            if res["correct_singleton"]:
                correct_singletons += 1
        else:
            non_singleton_f05s.append(res["f05"])
            
        # Source-level tracking
        for m in truth_set:
            if m.startswith("S2-"): s2_true += 1
            elif m.startswith("S3-"): s3_true += 1
        for p in pred_set:
            if p.startswith("S2-"): s2_pred += 1
            elif p.startswith("S3-"): s3_pred += 1
        for hit in (truth_set & pred_set):
            if hit.startswith("S2-"): s2_tp += 1
            elif hit.startswith("S3-"): s3_tp += 1
            
        # Country breakdown if metadata provided
        if entity_metadata and s1_id in entity_metadata:
            c = entity_metadata[s1_id].get("country", "Unknown")
            if c not in country_scores:
                country_scores[c] = {"f05": [], "prec": [], "rec": [], "singletons": 0, "correct_singletons": 0}
            country_scores[c]["f05"].append(res["f05"])
            country_scores[c]["prec"].append(res["precision"])
            country_scores[c]["rec"].append(res["recall"])
            if res["is_singleton"]:
                country_scores[c]["singletons"] += 1
                if res["correct_singleton"]:
                    country_scores[c]["correct_singletons"] += 1

    macro_f05 = float(np.mean(f05_scores))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    
    singleton_acc = (correct_singletons / singleton_count) if singleton_count > 0 else 0.0
    singleton_fp_rate = (1.0 - singleton_acc) if singleton_count > 0 else 0.0
    non_singleton_f05 = float(np.mean(non_singleton_f05s)) if non_singleton_f05s else 0.0
    
    result = {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "total_entities": total_entities,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
        "singleton_false_positive_rate": singleton_fp_rate,
        "non_singleton_count": len(non_singleton_f05s),
        "non_singleton_macro_f05": non_singleton_f05,
        "source_diagnostics": {
            "S2": {
                "true_links": s2_true,
                "predicted_links": s2_pred,
                "tp_links": s2_tp,
                "link_precision": (s2_tp / s2_pred) if s2_pred > 0 else 0.0,
                "link_recall": (s2_tp / s2_true) if s2_true > 0 else 0.0
            },
            "S3": {
                "true_links": s3_true,
                "predicted_links": s3_pred,
                "tp_links": s3_tp,
                "link_precision": (s3_tp / s3_pred) if s3_pred > 0 else 0.0,
                "link_recall": (s3_tp / s3_true) if s3_true > 0 else 0.0
            }
        }
    }
    
    if country_scores:
        result["country_breakdown"] = {}
        for c, stats in country_scores.items():
            result["country_breakdown"][c] = {
                "entity_count": len(stats["f05"]),
                "macro_f05": float(np.mean(stats["f05"])),
                "macro_precision": float(np.mean(stats["prec"])),
                "macro_recall": float(np.mean(stats["rec"])),
                "singleton_accuracy": (stats["correct_singletons"] / stats["singletons"]) if stats["singletons"] > 0 else 0.0
            }
            
    return result
