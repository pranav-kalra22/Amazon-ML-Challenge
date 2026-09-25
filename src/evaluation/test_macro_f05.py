#!/usr/bin/env python3
"""
Unit tests for Macro F0.5 Evaluator.
Validates the 6 mandatory competition scenarios and singleton rules.
"""

import math
import unittest
from src.evaluation.macro_f05 import compute_entity_f05, evaluate_predictions


class TestMacroF05Evaluator(unittest.TestCase):

    def test_case_1_perfect_multi_match(self):
        """Case 1: Perfect multi-match -> F0.5 = 1.0"""
        truth = {"S2-001", "S3-002"}
        pred = {"S2-001", "S3-002"}
        res = compute_entity_f05(truth, pred)
        self.assertAlmostEqual(res["precision"], 1.0)
        self.assertAlmostEqual(res["recall"], 1.0)
        self.assertAlmostEqual(res["f05"], 1.0)

    def test_case_2_extra_false_positive(self):
        """Case 2: Extra false positive -> P=2/3, R=1.0 -> F0.5 = 5/7 ~ 0.7142857"""
        truth = {"S2-001", "S3-002"}
        pred = {"S2-001", "S3-002", "S2-999"}
        res = compute_entity_f05(truth, pred)
        self.assertAlmostEqual(res["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(res["recall"], 1.0)
        expected_f05 = (1.25 * (2.0 / 3.0) * 1.0) / (0.25 * (2.0 / 3.0) + 1.0)
        self.assertAlmostEqual(res["f05"], expected_f05)
        self.assertAlmostEqual(res["f05"], 5.0 / 7.0)

    def test_case_3_missing_true_positive(self):
        """Case 3: Missing true positive -> P=1.0, R=0.5 -> F0.5 = 5/6 ~ 0.8333333"""
        truth = {"S2-001", "S3-002"}
        pred = {"S2-001"}
        res = compute_entity_f05(truth, pred)
        self.assertAlmostEqual(res["precision"], 1.0)
        self.assertAlmostEqual(res["recall"], 0.5)
        expected_f05 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        self.assertAlmostEqual(res["f05"], expected_f05)
        self.assertAlmostEqual(res["f05"], 5.0 / 6.0)

    def test_case_4_correct_singleton(self):
        """Case 4: Correct singleton -> truth empty, pred empty -> F0.5 = 1.0"""
        truth = set()
        pred = set()
        res = compute_entity_f05(truth, pred)
        self.assertTrue(res["is_singleton"])
        self.assertTrue(res["correct_singleton"])
        self.assertAlmostEqual(res["f05"], 1.0)

    def test_case_5_false_singleton_merge(self):
        """Case 5: False singleton merge -> truth empty, pred non-empty -> F0.5 = 0.0"""
        truth = set()
        pred = {"S2-001"}
        res = compute_entity_f05(truth, pred)
        self.assertTrue(res["is_singleton"])
        self.assertFalse(res["correct_singleton"])
        self.assertAlmostEqual(res["f05"], 0.0)

    def test_case_6_missed_non_singleton(self):
        """Case 6: Missed non-singleton -> truth non-empty, pred empty -> F0.5 = 0.0"""
        truth = {"S2-001"}
        pred = set()
        res = compute_entity_f05(truth, pred)
        self.assertFalse(res["is_singleton"])
        self.assertAlmostEqual(res["f05"], 0.0)

    def test_macro_aggregation_and_breakdown(self):
        """Test macro aggregation over mixed entities including country breakdown."""
        gt = {
            "S1-1": {"S2-10", "S3-20"},  # Perfect: 1.0
            "S1-2": {"S2-10", "S3-20"},  # FP: 5/7 ~ 0.7142857
            "S1-3": {"S2-10", "S3-20"},  # FN: 5/6 ~ 0.8333333
            "S1-4": set(),               # Correct singleton: 1.0
            "S1-5": set(),               # False singleton merge: 0.0
            "S1-6": {"S2-99"},           # Missed: 0.0
        }
        pred = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-10", "S3-20", "S2-999"},
            "S1-3": {"S2-10"},
            "S1-4": set(),
            "S1-5": {"S2-111"},
            "S1-6": set(),
        }
        metadata = {
            "S1-1": {"country": "US"},
            "S1-2": {"country": "US"},
            "S1-3": {"country": "US"},
            "S1-4": {"country": "India"},
            "S1-5": {"country": "India"},
            "S1-6": {"country": "India"},
        }
        results = evaluate_predictions(gt, pred, metadata)
        expected_macro = (1.0 + (5.0 / 7.0) + (5.0 / 6.0) + 1.0 + 0.0 + 0.0) / 6.0
        self.assertAlmostEqual(results["macro_f05"], expected_macro, places=5)
        self.assertEqual(results["total_entities"], 6)
        self.assertEqual(results["singleton_count"], 2)
        self.assertAlmostEqual(results["singleton_accuracy"], 0.5)
        self.assertAlmostEqual(results["singleton_false_positive_rate"], 0.5)
        self.assertIn("country_breakdown", results)
        self.assertIn("US", results["country_breakdown"])
        self.assertIn("India", results["country_breakdown"])


if __name__ == "__main__":
    unittest.main()
