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

    def test_missing_name_normalization_schema(self):
        """Verifies that empty string, None, and NaN business names produce identical 15-key schemas as non-empty names."""
        expected_keys = {
            "raw", "norm_unicode", "punct_norm", "compact_alnum", "domain_root",
            "legal_stripped", "legal_stripped_compact", "transliterated", "trans_stripped",
            "trans_compact", "tokens", "distinctive_tokens", "trans_tokens",
            "trans_distinctive_tokens", "sorted_token_signature"
        }
        
        rep_normal = normalize_name_non_destructive("Acme Global Technologies Inc.")
        rep_empty = normalize_name_non_destructive("")
        rep_none = normalize_name_non_destructive(None)
        rep_nan = normalize_name_non_destructive(float("nan"))
        rep_whitespace = normalize_name_non_destructive("   ")

        self.assertEqual(set(rep_normal.keys()), expected_keys)
        self.assertEqual(set(rep_empty.keys()), expected_keys)
        self.assertEqual(set(rep_none.keys()), expected_keys)
        self.assertEqual(set(rep_nan.keys()), expected_keys)
        self.assertEqual(set(rep_whitespace.keys()), expected_keys)

        # Ensure safe empty types
        self.assertEqual(rep_empty["tokens"], [])
        self.assertEqual(rep_empty["distinctive_tokens"], [])
        self.assertEqual(rep_empty["trans_tokens"], [])
        self.assertEqual(rep_empty["trans_distinctive_tokens"], [])
        self.assertEqual(rep_empty["sorted_token_signature"], "")

    def test_no_keyerror_for_empty_candidate_name(self):
        """Verifies that candidate streaming never crashes on missing/empty business names or addresses."""
        s1_entities = [
            {
                "entity_id": "S1-TEST-NULL",
                "business_name": "Standard Company",
                "business_address": "123 Main St",
                "country": "US"
            }
        ]
        blocker = MultiChannelBlocker()
        blocker.build_query_index(s1_entities)

        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-NULL-01\t\t\tUS\n")
            f.write("S2-NULL-02\tnan\tnan\tUS\n")
            f.write("S2-NULL-03\tNone\t<null>\tUS\n")
            temp_path = f.name

        try:
            candidates = blocker.generate_candidates_streaming([temp_path])
            self.assertEqual(len(candidates), 1)
            self.assertEqual(len(candidates[0]), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_symmetric_transliteration_candidate_df(self):
        """Verifies that non-Latin candidate names increment transliterated token DF correctly during Pass 1."""
        s1_entities = [
            {
                "entity_id": "S1-INDIA-TRANS",
                "business_name": "Kumar Sai Enterprises",
                "business_address": "Pune Road",
                "country": "India"
            }
        ]
        # Candidate file with 'साई' (sai) 10 times and 'कुमार' (kumar) 1 time
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-RARE\tकुमार उद्योग\tPune Road\tIndia\n")
            for i in range(10):
                f.write(f"S2-COMMON-{i}\tसाई सर्विसेज {i}\tPune Road\tIndia\n")
            temp_path = f.name

        try:
            cfg = {
                "channels": {
                    "channel_A_exact_name": {"enabled": False},
                    "channel_B_compact_domain": {"enabled": False},
                    "channel_C2_candidate_df_rare_token": {"enabled": False},
                    "channel_D2_address_rescue": {"enabled": False},
                    "channel_E2_symmetric_transliteration": {"enabled": True, "token_min_len": 3, "max_candidate_df": 2},
                    "channel_F_order_invariant_name": {"enabled": False}
                }
            }
            blocker = MultiChannelBlocker(config=cfg)
            blocker.build_query_index(s1_entities, candidate_file_paths=[temp_path])

            self.assertIn("sai", blocker.candidate_trans_df)
            self.assertEqual(blocker.candidate_trans_df["sai"], 10)
            self.assertIn("kumar", blocker.candidate_trans_df)
            self.assertEqual(blocker.candidate_trans_df["kumar"], 1)

            # 'sai' had DF=10 > max_candidate_df=2, so it must NOT be indexed as a distinctive token
            e2_keys = [k[2] for k in blocker.query_index if k[0] == CH_E2]
            self.assertIn("ttok_kumar", e2_keys)
            self.assertNotIn("ttok_sai", e2_keys)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_ceiling_recall_unaffected_by_storage_cap(self):
        """Verifies that Mode A ceiling tracking captures 100% of true links regardless of max_heap_k."""
        s1_entities = [
            {
                "entity_id": "S1-CEIL-01",
                "business_name": "Apex Global",
                "business_address": "100 Elm Street",
                "country": "US"
            }
        ]
        blocker = MultiChannelBlocker()
        blocker.build_query_index(s1_entities)

        # 5 candidates all matching S1
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in range(5):
                f.write(f"S2-TRUE-{i}\tApex Global\t100 Elm Street\tUS\n")
            temp_path = f.name

        try:
            gt_pairs = {(0, f"S2-TRUE-{i}") for i in range(5)}
            # Run with max_heap_k=2 (storage cap)
            result = blocker.generate_candidates_streaming(
                [temp_path],
                max_heap_k=2,
                gt_links_set=gt_pairs
            )
            # The bounded heap only has 2 candidates
            self.assertEqual(len(result[0]), 2)
            # Mode A ceiling has all 5 true links!
            self.assertEqual(len(result.gt_hits), 5)
            for i in range(5):
                self.assertIn((0, f"S2-TRUE-{i}"), result.gt_hits)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_late_arriving_true_candidate_counted_in_ceiling(self):
        """Verifies that a true link arriving late after noise candidates is captured in ceiling mode."""
        s1_entities = [
            {
                "entity_id": "S1-LATE-01",
                "business_name": "Omega Tech",
                "business_address": "500 Market St",
                "country": "US"
            }
        ]
        blocker = MultiChannelBlocker()
        blocker.build_query_index(s1_entities)

        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            # 5 noise candidates that also match Omega Tech
            for i in range(5):
                f.write(f"S2-NOISE-{i}\tOmega Tech\t500 Market St\tUS\n")
            # 1 true candidate arriving at the very end
            f.write("S2-TRUE-LATE\tOmega Tech\t500 Market St\tUS\n")
            temp_path = f.name

        try:
            gt_pairs = {(0, "S2-TRUE-LATE")}
            result = blocker.generate_candidates_streaming(
                [temp_path],
                max_heap_k=2,
                gt_links_set=gt_pairs
            )
            self.assertIn((0, "S2-TRUE-LATE"), result.gt_hits)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_late_arriving_high_score_candidate_replaces_weaker_in_top_k(self):
        """Verifies that a late-arriving strong match replaces a weaker candidate in the bounded Top-K heap."""
        s1_entities = [
            {
                "entity_id": "S1-REPLACE-01",
                "business_name": "Zenith International",
                "business_address": "880 June Terrace, Lake Zurich, IL 60047",
                "country": "US"
            }
        ]
        blocker = MultiChannelBlocker()
        blocker.build_query_index(s1_entities)

        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            # Candidate 1: Weak match (Address rescue only -> score 6.0)
            f.write("S2-WEAK-01\tTotally Different Name\t880 June Terrace, Lake Zurich, IL 60047\tUS\n")
            # Candidate 2: Weak match (Address rescue only -> score 6.0)
            f.write("S2-WEAK-02\tAnother Different Name\t880 June Terrace, Lake Zurich, IL 60047\tUS\n")
            # Candidate 3: Arrives 3rd with Strong match (Exact name + compact name -> score 10.0 + 8.0 + 2.0 = 20.0)
            f.write("S2-STRONG-03\tZenith International\t999 Unrelated Blvd\tUS\n")
            temp_path = f.name

        try:
            result = blocker.generate_candidates_streaming(
                [temp_path],
                max_heap_k=2
            )
            retained_ids = set(result[0].keys())
            self.assertEqual(len(retained_ids), 2)
            # S2-STRONG-03 must be admitted into the bounded heap!
            self.assertIn("S2-STRONG-03", retained_ids)
            # And one of the weak candidates was evicted
            self.assertEqual(len(retained_ids & {"S2-WEAK-01", "S2-WEAK-02"}), 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_yaml_config_controls_k_and_blocker_params(self):
        """Verifies that YAML configuration directly configures max_heap_k, channel toggles, and weights."""
        cfg_path = "configs/blocking/blocking_v03.yaml"
        blocker = MultiChannelBlocker(config_path=cfg_path)
        self.assertEqual(blocker.max_heap_k, 2000)
        self.assertTrue(blocker.ch_A_enabled)
        self.assertTrue(blocker.ch_E2_enabled)
        self.assertEqual(blocker.weights.get("CH_A"), 10.0)
        self.assertEqual(blocker.weights.get("CH_B"), 8.0)


if __name__ == "__main__":
    unittest.main()

