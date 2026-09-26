"""
model.py — ML matching model: training, prediction, and threshold optimization.

Uses LightGBM as the primary classifier with support for class imbalance
handling and F0.5-optimized threshold selection.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score
import joblib
import os


def compute_f05(precision: float, recall: float) -> float:
    """Compute F0.5 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


class EntityMatchModel:
    """LightGBM-based entity match classifier.

    Predicts P(match) for candidate pairs and supports threshold optimization.
    """

    def __init__(self, params: dict = None):
        self.model = None
        self.best_threshold = 0.5
        self.feature_columns = None
        self.params = params or self._default_params()

    @staticmethod
    def _default_params() -> dict:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "is_unbalance": True,    # Handle class imbalance
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        feature_columns: list = None,
    ):
        """Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels (0/1)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional)
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns or [
            c for c in X_train.columns if c not in {"s1_id", "s2s3_id", "label"}
        ]

        print(f"\nTraining LightGBM classifier...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Positive (match): {int(y_train.sum()):,} "
              f"({y_train.mean()*100:.1f}%)")
        print(f"  Negative (no match): {int((1-y_train).sum()):,} "
              f"({(1-y_train).mean()*100:.1f}%)")
        print(f"  Features: {len(self.feature_columns)}")

        self.model = lgb.LGBMClassifier(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val[self.feature_columns], y_val)]
            # Use a callback for early stopping
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100),
            ]
            print(f"  Validation samples: {len(X_val):,}")

        self.model.fit(
            X_train[self.feature_columns],
            y_train,
            **fit_params,
        )

        print(f"  Best iteration: {self.model.best_iteration_}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities for candidate pairs."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict binary match labels using the specified threshold."""
        threshold = threshold or self.best_threshold
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        s1_ids: np.ndarray,
        gt_dict: dict,
        thresholds: list = None,
    ) -> float:
        """Find the threshold that maximizes entity-level F0.5 on validation.

        Uses the actual entity-level evaluation (macro-averaged F0.5 over S1 entities)
        rather than pair-level metrics.

        Args:
            X_val: Validation feature matrix
            y_val: Validation labels
            s1_ids: S1 entity IDs for each row in X_val
            gt_dict: dict mapping S1 entity_id → set of true match IDs
            thresholds: list of thresholds to try

        Returns:
            Best threshold value
        """
        if thresholds is None:
            thresholds = [
                0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
            ]

        probas = self.predict_proba(X_val)
        s2s3_ids = X_val["s2s3_id"].values

        print(f"\n=== Threshold Optimization ===")
        print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F0.5':>10}")
        print("-" * 45)

        best_f05 = -1
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (probas >= thresh).astype(int)

            # Build predicted match sets per S1 entity
            pred_dict = {}
            for i, (s1_id, s2s3_id, pred) in enumerate(
                zip(s1_ids, s2s3_ids, preds)
            ):
                if s1_id not in pred_dict:
                    pred_dict[s1_id] = set()
                if pred == 1:
                    pred_dict[s1_id].add(s2s3_id)

            # Compute entity-level macro-averaged F0.5
            prec_sum = 0.0
            rec_sum = 0.0
            f05_sum = 0.0
            n_entities = 0

            for s1_id, true_set in gt_dict.items():
                pred_set = pred_dict.get(s1_id, set())
                n_entities += 1

                if len(pred_set) == 0 and len(true_set) == 0:
                    # Both empty: perfect
                    prec_sum += 1.0
                    rec_sum += 1.0
                    f05_sum += 1.0
                elif len(pred_set) == 0 and len(true_set) > 0:
                    # Missed everything
                    prec_sum += 1.0  # vacuous precision
                    rec_sum += 0.0
                    f05_sum += 0.0
                elif len(pred_set) > 0 and len(true_set) == 0:
                    # False alarm
                    prec_sum += 0.0
                    rec_sum += 1.0  # vacuous recall
                    f05_sum += 0.0
                else:
                    tp = len(pred_set & true_set)
                    p = tp / len(pred_set) if pred_set else 0.0
                    r = tp / len(true_set) if true_set else 0.0
                    f05 = compute_f05(p, r)
                    prec_sum += p
                    rec_sum += r
                    f05_sum += f05

            macro_prec = prec_sum / max(n_entities, 1)
            macro_rec = rec_sum / max(n_entities, 1)
            macro_f05 = f05_sum / max(n_entities, 1)

            print(f"{thresh:>10.2f} {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f05:>10.4f}")

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_thresh = thresh

        print(f"\n  Best threshold: {best_thresh:.2f} (F0.5 = {best_f05:.4f})")
        self.best_threshold = best_thresh
        return best_thresh

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        importance = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        print(f"\n=== Top {top_n} Features ===")
        for _, row in feat_imp.head(top_n).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:>6}")

        return feat_imp

    def save(self, path: str):
        """Save model, threshold, and feature columns."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "threshold": self.best_threshold,
            "feature_columns": self.feature_columns,
            "params": self.params,
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        """Load a previously saved model."""
        data = joblib.load(path)
        self.model = data["model"]
        self.best_threshold = data["threshold"]
        self.feature_columns = data["feature_columns"]
        self.params = data["params"]
        print(f"  Model loaded from {path}")
        print(f"  Threshold: {self.best_threshold:.2f}")
        print(f"  Features: {len(self.feature_columns)}")
