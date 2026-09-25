#!/usr/bin/env python3
"""
Unit tests for non-destructive normalizer and multi-channel candidate blocker — EXP_002.

Validates:
1. Non-destructive name and address representations
2. Symmetric cross-script transliteration retrieval (Latin S1 -> Indic Candidate)
3. Authoritative YAML configuration plumbing
4. Deterministic candidate evidence ranking
5. Candidate-population document frequency prioritization
6. True address-only rescue without name token prerequisite
"""

import sys
sys.path.insert(0, ".")
import os
import unittest
import tempfile
import yaml

from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive
)
from src.blocking.blocker import (
    MultiChannelBlocker,
    CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F
)


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
        self.assertIn("bright", rep["trans_distinctive_tokens"])

    def test_domain_name_extraction(self):
        """Verifies domain root extraction from website-style names."""
        raw = "brightiheartmedia.com"
        rep = normalize_name_non_destructive(raw)
        self.assertEqual(rep["domain_root"], "brightiheartmedia")

        # Must match the legal_stripped_compact of "Bright Iheartmedia, Inc."
        other_rep = normalize_name_non_destructive("Bright Iheartmedia, Inc.")
        self.assertEqual(rep["domain_root"], other_rep["legal_stripped_compact"])

    def test_address_non_destructive_representations(self):
        """Verifies that building numbers, postal codes, and distinctive tokens are preserved."""
        raw = "10018C Broadway Ave, New York, NY 10018"
        rep = normalize_address_non_destructive(raw)

        self.assertEqual(rep["raw"], raw)
        self.assertEqual(rep["building_raw"], "10018c")
        self.assertEqual(rep["building_numeric"], "10018")
        self.assertIn("10018", rep["all_building_numerics"])
        self.assertEqual(rep["postal_code"], "10018")
        self.assertIn("broadway", rep["street_tokens"])
        self.assertIn("broadway", rep["distinctive_tokens"])
        self.assertIn("avenue", rep["road_standardized_tokens"])

    def test_indian_postal_pin_extraction(self):
        """Verifies 6-digit PIN code extraction for Indian addresses."""
        raw = "Plot 42, Sector 18, Gurugram, Haryana 122015"
        rep = normalize_address_non_destructive(raw)
        self.assertEqual(rep["postal_code"], "122015")
        self.assertEqual(rep["building_numeric"], "42")
        self.assertIn("gurugram", rep["distinctive_tokens"])

    def test_unicode_transliteration(self):
        """Verifies ASCII transliteration as an additional representation."""
        raw = "Café de Paris SASU"
        rep = normalize_name_non_destructive(raw)
        self.assertEqual(rep["raw"], raw)
        self.assertIn("café", rep["norm_unicode"])
        self.assertEqual(rep["trans_compact"], "cafedeparissasu")
        self.assertEqual(rep["trans_stripped"], "cafe de paris")

    def test_transliteration_cross_script_retrieval(self):
        """
        CRITICAL TEST: Verifies that Latin-script S1 retrieves native Indic-script candidate
        through the transliteration channel (Channel E2).
        """
        s1_entities = [
            {
                "entity_id": "S1-INDIA-01",
                "business_name": "Sai Services",
                "business_address": "Main Road, Pune, Maharashtra",
                "country": "India"
            }
        ]
        blocker = MultiChannelBlocker()
        blocker.build_query_index(s1_entities)

        # Check that Latin S1 registered in transliteration space under Channel E2
        self.assertIn(0, blocker.query_index[(CH_E2, "India", "sai")])

        # Create temporary TSV candidate file with native Devanagari script: 'साई सर्विसेज'
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-DEV-999\tसाई सर्विसेज\tMain Road, Pune\tIndia\n")
            temp_path = f.name

        try:
            candidates = blocker.generate_candidates_streaming([temp_path])
            s1_cands = candidates[0]
            self.assertIn("S2-DEV-999", s1_cands, "Failed to retrieve native Indic candidate for Latin S1!")
            self.assertTrue(s1_cands["S2-DEV-999"] & CH_E2, "Candidate was not retrieved via Channel E2!")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_config_plumbing(self):
        """Verifies that changing YAML configuration parameters alters blocker construction."""
        s1_entities = [
            {
                "entity_id": "S1-TEST-01",
                "business_name": "Alpha Beta Gamma LLC",
                "business_address": "100 Broadway, NY 10001",
                "country": "US"
            }
        ]

        cfg_with_f = {
            "channels": {
                "channel_A_exact_name": {"enabled": True},
                "channel_F_order_invariant_name": {"enabled": True, "max_candidate_df": 50, "max_pairs_per_entity": 3}
            }
        }
        cfg_without_f = {
            "channels": {
                "channel_A_exact_name": {"enabled": True},
                "channel_F_order_invariant_name": {"enabled": False}
            }
        }

        b_f = MultiChannelBlocker(config=cfg_with_f)
        b_f.build_query_index(s1_entities)

        b_no_f = MultiChannelBlocker(config=cfg_without_f)
        b_no_f.build_query_index(s1_entities)

        # In b_f, Channel F keys must be present; in b_no_f, Channel F keys must NOT be present
        f_keys_with = [k for k in b_f.query_index if k[0] == CH_F]
        f_keys_without = [k for k in b_no_f.query_index if k[0] == CH_F]

        self.assertGreater(len(f_keys_with), 0, "Channel F keys should be present when enabled")
        self.assertEqual(len(f_keys_without), 0, "Channel F keys must be absent when disabled")

    def test_deterministic_candidate_ranking(self):
        """Verifies that candidate evidence ranking produces identical ordering across repeated runs."""
        blocker = MultiChannelBlocker()

        cand_dict = {
            "S2-005": CH_A,                     # Score 10.0
            "S2-001": CH_A | CH_B,              # Score 10.0 + 8.0 + bonus = 20.0
            "S2-004": CH_D2,                    # Score 6.0
            "S2-003": CH_C2,                    # Score 5.0
            "S2-002": CH_A | CH_B               # Same score 20.0 as S2-001, but 'S2-001' < 'S2-002'
        }

        order1 = blocker.rank_entity_candidates(cand_dict)
        order2 = blocker.rank_entity_candidates(cand_dict)
        order3 = blocker.rank_entity_candidates(cand_dict)

        self.assertEqual(order1, order2)
        self.assertEqual(order2, order3)

        # S2-001 and S2-002 have highest score; S2-001 breaks tie before S2-002
        self.assertEqual(order1[0], "S2-001")
        self.assertEqual(order1[1], "S2-002")
        self.assertEqual(order1[2], "S2-005")
        self.assertEqual(order1[3], "S2-004")
        self.assertEqual(order1[4], "S2-003")

    def test_candidate_df_ranking(self):
        """Verifies that synthetic candidate population frequency ranks genuinely rare tokens ahead of common tokens."""
        s1_entities = [
            {
                "entity_id": "S1-TEST-DF",
                "business_name": "Raretoken Commontoken Enterprises",
                "business_address": "123 Elm St",
                "country": "US"
            }
        ]

        # Create synthetic candidate file where 'commontoken' appears 10 times, 'raretoken' appears 1 time
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-01\tRaretoken Store\t123 Elm St\tUS\n")
            for i in range(10):
                f.write(f"S2-COMMON-{i}\tCommontoken Store\t123 Elm St\tUS\n")
            temp_path = f.name

        try:
            cfg = {
                "channels": {
                    "channel_C2_candidate_df_rare_token": {
                        "enabled": True, "min_token_len": 4, "max_candidate_df": 2, "max_tokens_per_entity": 1
                    }
                }
            }
            blocker = MultiChannelBlocker(config=cfg)
            blocker.build_query_index(s1_entities, candidate_file_paths=[temp_path])

            self.assertEqual(blocker.candidate_name_df["raretoken"], 1)
            self.assertEqual(blocker.candidate_name_df["commontoken"], 10)

            # Only 'raretoken' should be indexed in Channel C2 because commontoken DF (10) > max_candidate_df (2)
            c2_keys = [k[2] for k in blocker.query_index if k[0] == CH_C2]
            self.assertIn("raretoken", c2_keys)
            self.assertNotIn("commontoken", c2_keys)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_address_only_rescue(self):
        """
        Verifies that completely different business names but sufficiently strong address anchor
        generates a candidate pair via Channel D2 (True Address Rescue).
        """
        s1_entities = [
            {
                "entity_id": "S1-ADDR-01",
                "business_name": "Apex Industrial Tools",
                "business_address": "880 June Terrace, Lake Zurich, IL 60047",
                "country": "US"
            }
        ]
        blocker = MultiChannelBlocker()
        blocker.build_query_index(s1_entities)

        # Candidate with totally different name (DBA / trade name) at exact same address
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-DBA-888\tZenith Precision Logistics\t880 June Terrace, Lake Zurich, IL 60047\tUS\n")
            temp_path = f.name

        try:
            candidates = blocker.generate_candidates_streaming([temp_path])
            s1_cands = candidates[0]
            self.assertIn("S2-DBA-888", s1_cands, "Failed to rescue candidate with different name via address channel!")
            self.assertTrue(s1_cands["S2-DBA-888"] & CH_D2, "Address candidate was not retrieved via Channel D2!")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
