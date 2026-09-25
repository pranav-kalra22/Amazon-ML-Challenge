#!/usr/bin/env python3
"""
Unit tests for non-destructive normalizer and multi-channel blocker.
"""

import sys
sys.path.insert(0, ".")
import unittest
from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive
)
from src.blocking.blocker import MultiChannelBlocker


class TestNormalizerAndBlocker(unittest.TestCase):

    def test_name_non_destructive_representations(self):
        """Verifies that all required name representations are independently preserved."""
        raw = "Bright Iheartmedia, Inc."
        rep = normalize_name_non_destructive(raw)

        self.assertEqual(rep["raw"], raw)
        self.assertEqual(rep["norm_unicode"], "bright iheartmedia, inc.")
        self.assertEqual(rep["punct_norm"], "bright iheartmedia inc")
        self.assertEqual(rep["compact_alnum"], "brightiheartmediainc")
        self.assertEqual(rep["legal_stripped"], "bright iheartmedia")
        self.assertEqual(rep["legal_stripped_compact"], "brightiheartmedia")
        self.assertIn("bright", rep["distinctive_tokens"])
        self.assertIn("iheartmedia", rep["distinctive_tokens"])
        self.assertNotIn("inc", rep["distinctive_tokens"])

    def test_domain_name_extraction(self):
        """Verifies domain root extraction from website-style names."""
        raw = "brightiheartmedia.com"
        rep = normalize_name_non_destructive(raw)
        self.assertEqual(rep["domain_root"], "brightiheartmedia")

        # Must match the legal_stripped_compact of "Bright Iheartmedia, Inc."
        other_rep = normalize_name_non_destructive("Bright Iheartmedia, Inc.")
        self.assertEqual(rep["domain_root"], other_rep["legal_stripped_compact"])

    def test_address_non_destructive_representations(self):
        """Verifies that both raw and numeric building numbers and postal codes are preserved."""
        raw = "10018C Broadway Ave, New York, NY 10018"
        rep = normalize_address_non_destructive(raw)

        self.assertEqual(rep["raw"], raw)
        self.assertEqual(rep["building_raw"], "10018c")
        self.assertEqual(rep["building_numeric"], "10018")
        self.assertEqual(rep["postal_code"], "10018")
        self.assertIn("broadway", rep["street_tokens"])
        self.assertIn("avenue", rep["road_standardized_tokens"])

    def test_indian_postal_pin_extraction(self):
        """Verifies 6-digit PIN code extraction for Indian addresses."""
        raw = "Plot 42, Sector 18, Gurugram, Haryana 122015"
        rep = normalize_address_non_destructive(raw)
        self.assertEqual(rep["postal_code"], "122015")
        self.assertEqual(rep["building_numeric"], "42")

    def test_unicode_transliteration(self):
        """Verifies ASCII transliteration as an additional representation."""
        raw = "Café de Paris SASU"
        rep = normalize_name_non_destructive(raw)
        self.assertEqual(rep["raw"], raw)
        self.assertIn("café", rep["norm_unicode"])
        self.assertEqual(rep["trans_compact"], "cafedeparissasu")
        self.assertEqual(rep["trans_stripped"], "cafe de paris")


    def test_blocker_channel_matches(self):
        """Verifies that multi-channel blocker connects entities across all channels."""
        s1_entities = [
            {
                "entity_id": "S1-001",
                "business_name": "Bright Iheartmedia, Inc.",
                "business_address": "10018C Broadway Ave, New York, NY 10018",
                "country": "US"
            },
            {
                "entity_id": "S1-002",
                "business_name": "Café de Paris SASU",
                "business_address": "15 Rue de Rivoli, Paris",
                "country": "France"
            }
        ]

        from src.blocking.blocker import CH_A, CH_B, CH_C, CH_D, CH_E
        blocker = MultiChannelBlocker(token_max_doc_freq=10, token_min_len=3)
        blocker.build_query_index(s1_entities)

        # Check Channel B domain root match
        self.assertIn(0, blocker.query_index[(CH_B, "US", "brightiheartmedia")])
        # Check Channel D address anchor match
        self.assertIn(0, blocker.query_index[(CH_D, "US", "b_10018_bright")])
        # Check Channel E transliteration match
        self.assertIn(1, blocker.query_index[(CH_E, "France", "cafe de paris")])
        # Check country isolation (India cannot match US key)
        self.assertNotIn(0, blocker.query_index[(CH_B, "India", "brightiheartmedia")])



if __name__ == "__main__":
    unittest.main()
