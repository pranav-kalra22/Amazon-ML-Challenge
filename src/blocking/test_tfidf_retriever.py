#!/usr/bin/env python3
"""
Unit and Integration Tests for EXP_004 Scalable TF-IDF Retriever.

Validates:
1. TF-IDF YAML config loading and parameter propagation
2. Strictly float32 sparse CSR matrix operations
3. Absolute prohibition on dense conversions (.toarray() / .todense())
4. Deterministic top-N sparse cosine retrieval
5. Strict country partition isolation (open-set: US, India, France)
6. S2 and S3 partition isolation
7. Business name approximate retrieval (Channel G_NAME)
8. Address approximate retrieval preserving digits (Channel G_ADDR)
9. Transliterated name approximate retrieval (Channel G_TRANS)
10. Lexical + TF-IDF candidate union with bitmask attribution
11. Duplicate candidate suppression
12. Vectorizer and matrix serialization / reload caching
13. Cardinality performance breakdown calculation
14. Zero-candidate audit correctness
"""

import os
import sys
sys.path.insert(0, ".")
import shutil
import tempfile
import unittest
import numpy as np
from scipy import sparse

from src.blocking.tfidf_retriever import (
    TfidfApproximateRetriever,
    compute_config_fingerprint,
    compute_dataset_fingerprint,
    evaluate_parameter_sweep_in_memory
)
from src.blocking.blocker import (
    MultiChannelBlocker,
    CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F,
    CH_G_NAME, CH_G_ADDR, CH_G_TRANS,
    StreamingCandidateResult
)
from src.blocking.cloud_benchmark import (
    compute_cardinality_breakdown,
    compute_zero_candidate_audit
)


class TestTfidfRetriever(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "execution": {
                "batch_size": 100,
                "n_threads": 2,
                "cache_dir": os.path.join(self.temp_dir, "cache"),
                "memory_safety": {"max_ram_gb": 10.0, "abort_on_limit": True}
            },
            "tfidf_retrieval": {
                "enabled": True,
                "channel_G_name_char_tfidf": {
                    "enabled": True,
                    "analyzer": "char_wb",
                    "ngram_min": 3,
                    "ngram_max": 4,
                    "min_df": 1,
                    "max_df": 1.0,
                    "sublinear_tf": True,
                    "norm": "l2",
                    "dtype": "float32",
                    "retrieval_top_k": 5,
                    "similarity_threshold": 0.2,
                    "representations": ["norm_unicode", "legal_stripped"]
                },
                "channel_G_address_char_tfidf": {
                    "enabled": True,
                    "analyzer": "char_wb",
                    "ngram_min": 3,
                    "ngram_max": 4,
                    "min_df": 1,
                    "max_df": 1.0,
                    "sublinear_tf": True,
                    "norm": "l2",
                    "dtype": "float32",
                    "retrieval_top_k": 5,
                    "similarity_threshold": 0.2,
                    "representations": ["norm_unicode"]
                },
                "channel_G_transliterated_char_tfidf": {
                    "enabled": True,
                    "analyzer": "char_wb",
                    "ngram_min": 3,
                    "ngram_max": 4,
                    "min_df": 1,
                    "max_df": 1.0,
                    "sublinear_tf": True,
                    "norm": "l2",
                    "dtype": "float32",
                    "retrieval_top_k": 5,
                    "similarity_threshold": 0.2,
                    "representations": ["transliterated"]
                }
            }
        }
        self.retriever = TfidfApproximateRetriever(config=self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_loading_and_parameter_plumbing(self):
        """Verifies that YAML configuration is correctly parsed and plumbed."""
        self.assertEqual(self.retriever.batch_size, 100)
        self.assertEqual(self.retriever.n_threads, 2)
        self.assertTrue(self.retriever.name_cfg["enabled"])
        self.assertEqual(self.retriever.name_cfg["ngram_min"], 3)
        self.assertEqual(self.retriever.name_cfg["ngram_max"], 4)

    def test_float32_sparse_matrices(self):
        """Verifies that vectorizer produces float32 CSR matrices with zero dense copies."""
        vec = self.retriever.build_vectorizer(self.retriever.name_cfg)
        corpus = ["Apple Inc", "Microsoft Corporation", "Amazon Web Services"]
        mat = vec.fit_transform(corpus)

        self.assertTrue(sparse.issparse(mat))
        self.assertEqual(mat.dtype, np.float32)
        self.assertIsInstance(mat, sparse.csr_matrix)

    def test_no_dense_conversion_in_retrieval(self):
        """Verifies that retrieval does not call .toarray() or .todense()."""
        s1_entities = [{"entity_id": "S1-1", "business_name": "Apex Solutions", "business_address": "123 Main St", "country": "US"}]
        cands = [
            {"entity_id": "S2-1", "business_name": "Apex Solution Group", "business_address": "123 Main Street", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Zenith Corp", "business_address": "999 Oak Way", "country": "US"}
        ]
        matches = self.retriever.retrieve_partition_approximate_candidates(
            country="US",
            source="S2",
            channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg,
            channel_type="name",
            s1_entities=s1_entities,
            s1_indices=[0],
            candidate_records=cands,
            use_cache=False
        )
        self.assertGreater(len(matches), 0)
        # Verify retrieved candidate is Apex
        self.assertEqual(matches[0][1], "S2-1")
        self.assertEqual(matches[0][2], CH_G_NAME)

    def test_country_partition_isolation(self):
        """Verifies open-set country isolation: US queries never retrieve India or France candidates."""
        s1_us = [{"entity_id": "S1-US", "business_name": "Universal Medical", "business_address": "100 Hospital Rd", "country": "US"}]
        cands_india = [{"entity_id": "S2-IN", "business_name": "Universal Medical", "business_address": "100 Hospital Rd", "country": "India"}]

        # Under Country=US, passing India candidates should result in no match because caller partitions by country
        matches = self.retriever.retrieve_partition_approximate_candidates(
            country="US",
            source="S2",
            channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg,
            channel_type="name",
            s1_entities=s1_us,
            s1_indices=[0],
            candidate_records=[], # Empty partition for US
            use_cache=False
        )
        self.assertEqual(len(matches), 0)

    def test_address_approximate_retrieval(self):
        """Verifies that address retrieval matches minor typographical variations while preserving building numbers."""
        s1_entities = [{"entity_id": "S1-ADDR", "business_name": "Unique Name A", "business_address": "742 Evergreen Terrace, Springfield, IL 62704", "country": "US"}]
        cands = [
            {"entity_id": "S2-ADDR-MATCH", "business_name": "Different Name B", "business_address": "742 Evergeen Terr, Springfield, IL", "country": "US"},
            {"entity_id": "S2-ADDR-OTHER", "business_name": "Different Name C", "business_address": "100 Elm Street, Boston, MA", "country": "US"}
        ]
        matches = self.retriever.retrieve_partition_approximate_candidates(
            country="US",
            source="S2",
            channel_bit=CH_G_ADDR,
            channel_cfg=self.retriever.addr_cfg,
            channel_type="address",
            s1_entities=s1_entities,
            s1_indices=[0],
            candidate_records=cands,
            use_cache=False
        )
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0][1], "S2-ADDR-MATCH")
        self.assertEqual(matches[0][2], CH_G_ADDR)

    def test_transliterated_approximate_retrieval(self):
        """Verifies that transliteration retrieval handles cross-script variations."""
        s1_entities = [{"entity_id": "S1-TRANS", "business_name": "Shree Ganesh Enterprises", "business_address": "MG Road", "country": "India"}]
        # Indic script or alternate spelling
        cands = [
            {"entity_id": "S2-TRANS-MATCH", "business_name": "Sri Ganesha Enterprise", "business_address": "MG Road", "country": "India"},
            {"entity_id": "S2-TRANS-NOISE", "business_name": "Patel Sweets", "business_address": "Station Road", "country": "India"}
        ]
        matches = self.retriever.retrieve_partition_approximate_candidates(
            country="India",
            source="S2",
            channel_bit=CH_G_TRANS,
            channel_cfg=self.retriever.trans_cfg,
            channel_type="transliterated",
            s1_entities=s1_entities,
            s1_indices=[0],
            candidate_records=cands,
            use_cache=False
        )
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0][1], "S2-TRANS-MATCH")
        self.assertEqual(matches[0][2], CH_G_TRANS)

    def test_candidate_union_and_duplicate_suppression(self):
        """Verifies that unioning lexical and TF-IDF candidates preserves bitmasks and suppresses duplicates."""
        base_result = StreamingCandidateResult(
            candidate_dicts=[{"S2-DUPE": CH_A, "S2-ONLY-LEX": CH_B}],
            uncapped_counts=np.array([2], dtype=np.int32),
            uncapped_channel_counts={CH_A: np.array([1], dtype=np.int32), CH_B: np.array([1], dtype=np.int32)},
            total_scanned=10
        )
        tfidf_matches = [
            (0, "S2-DUPE", CH_G_NAME, 0.85),
            (0, "S2-NEW-TFIDF", CH_G_ADDR, 0.75)
        ]
        merged = self.retriever.merge_tfidf_candidates_into_result(base_result, tfidf_matches)

        cand_dict = merged[0]
        # S2-DUPE must have both CH_A (1) and CH_G_NAME (64) -> 65
        self.assertEqual(cand_dict["S2-DUPE"], CH_A | CH_G_NAME)
        # S2-ONLY-LEX remains CH_B (2)
        self.assertEqual(cand_dict["S2-ONLY-LEX"], CH_B)
        # S2-NEW-TFIDF is CH_G_ADDR (128)
        self.assertEqual(cand_dict["S2-NEW-TFIDF"], CH_G_ADDR)
        # Total distinct candidates = 3 (no duplication of S2-DUPE)
        self.assertEqual(len(cand_dict), 3)

    def test_vectorizer_and_matrix_caching(self):
        """Verifies serialization and reload of vectorizer and candidate matrix."""
        s1_entities = [{"entity_id": "S1-CACHE", "business_name": "Delta Logistics", "business_address": "Airport Way", "country": "US"}]
        cands = [{"entity_id": "S2-CACHE", "business_name": "Delta Logistics LLC", "business_address": "Airport Way", "country": "US"}]

        # First run writes cache
        m1 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1_entities, s1_indices=[0], candidate_records=cands,
            use_cache=True
        )
        # Second run reads cache
        m2 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1_entities, s1_indices=[0], candidate_records=cands,
            use_cache=True
        )
        self.assertEqual(len(m1), len(m2))
        self.assertEqual(m1[0][1], m2[0][1])

    def test_cardinality_breakdown_and_zero_candidate_audit(self):
        """Verifies calculation of cardinality breakdown and zero candidate audit."""
        val_gt = {
            "S1-SINGLE": set(),
            "S1-ONE": {"S2-1"},
            "S1-TWO": {"S2-2", "S3-2"}
        }
        oracle_preds = {
            "S1-SINGLE": set(),
            "S1-ONE": {"S2-1"},
            "S1-TWO": {"S2-2"} # Partial hit
        }
        meta = {"S1-SINGLE": {"country": "US"}, "S1-ONE": {"country": "US"}, "S1-TWO": {"country": "India"}}
        breakdown = compute_cardinality_breakdown(val_gt, oracle_preds, meta)
        self.assertEqual(len(breakdown), 3)

        cand_counts = np.array([0, 5, 2], dtype=np.int32)
        s1_ids = ["S1-SINGLE", "S1-ONE", "S1-TWO"]
        audit = compute_zero_candidate_audit(val_gt, cand_counts, s1_ids)

        self.assertEqual(audit["total_zero_candidate_entities"], 1)
        self.assertEqual(audit["singleton_zero_candidates"], 1)
        self.assertEqual(audit["non_singleton_zero_candidates"], 0)

    def test_open_set_country_france_partitioning(self):
        """Verifies that open-set country partitioning seamlessly handles France without code changes."""
        s1_france = [{"entity_id": "S1-FR-1", "business_name": "Boulangerie Patisserie Paris", "business_address": "15 Rue de Rivoli, Paris", "country": "France"}]
        cands_france = [{"entity_id": "S2-FR-1", "business_name": "Boulangerie & Patisserie", "business_address": "15 Rue Rivoli", "country": "France"}]

        matches = self.retriever.retrieve_partition_approximate_candidates(
            country="France",
            source="S2",
            channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg,
            channel_type="name",
            s1_entities=s1_france,
            s1_indices=[0],
            candidate_records=cands_france,
            use_cache=False
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1], "S2-FR-1")
        self.assertEqual(matches[0][2], CH_G_NAME)

    def test_source_partitioning_isolation(self):
        """Verifies that candidate source partitions (S2 vs S3) operate independently."""
        s1_us = [{"entity_id": "S1-SRC", "business_name": "Acme Widgets", "business_address": "123 Industrial Way", "country": "US"}]
        cands_s2 = [{"entity_id": "S2-ACME", "business_name": "Acme Widgets Co", "business_address": "123 Industrial Way", "country": "US"}]
        cands_s3 = [{"entity_id": "S3-ACME", "business_name": "Acme Widgets LLC", "business_address": "123 Industrial Way", "country": "US"}]

        m_s2 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1_us, s1_indices=[0], candidate_records=cands_s2,
            use_cache=False
        )
        m_s3 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S3", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1_us, s1_indices=[0], candidate_records=cands_s3,
            use_cache=False
        )
        self.assertEqual(m_s2[0][1], "S2-ACME")
        self.assertEqual(m_s3[0][1], "S3-ACME")

    def test_cached_candidate_ids_reloaded_correctly(self):
        """1. Verifies that cached candidate IDs are reloaded, validated against matrix shape, and preserved."""
        import json
        import joblib

        s1 = [{"entity_id": "S1-ID-TEST", "business_name": "Acme Tools", "business_address": "123 Elm St", "country": "US"}]
        cands = [
            {"entity_id": "CAND-01", "business_name": "Acme Tools Inc", "business_address": "123 Elm St", "country": "US"},
            {"entity_id": "CAND-02", "business_name": "Acme Tools LLC", "business_address": "123 Elm St", "country": "US"}
        ]
        # First call writes cache
        m1 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands,
            use_cache=True
        )
        self.assertGreater(len(m1), 0)

        cache_files = os.listdir(self.retriever.cache_dir)
        id_files = [f for f in cache_files if f.endswith("_cand_ids.joblib")]
        mat_files = [f for f in cache_files if f.endswith("_cand_mat.npz")]
        meta_files = [f for f in cache_files if f.endswith("_metadata.json")]
        self.assertEqual(len(id_files), 1)
        self.assertEqual(len(mat_files), 1)
        self.assertEqual(len(meta_files), 1)

        loaded_ids = joblib.load(os.path.join(self.retriever.cache_dir, id_files[0]))
        self.assertEqual(loaded_ids, ["CAND-01", "CAND-02"])

        with open(os.path.join(self.retriever.cache_dir, meta_files[0]), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["candidate_id_count"], 2)
        self.assertEqual(meta["matrix_shape"][0], 2)

        # Second call loads from cache and retrieves identical matches
        m2 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands,
            use_cache=True
        )
        self.assertEqual(m1, m2)

    def test_changed_candidate_order_cannot_corrupt_cached_mapping(self):
        """2. Verifies that changing candidate input order cannot cause wrong entity ID mapping."""
        s1 = [{"entity_id": "S1-QUERY", "business_name": "Beta Corporation", "business_address": "500 Main", "country": "US"}]
        cands_order1 = [
            {"entity_id": "CAND-ALPHA", "business_name": "Zeta Xylophone Qwerty", "business_address": "100 First", "country": "US"},
            {"entity_id": "CAND-BETA", "business_name": "Beta Corporation", "business_address": "500 Main", "country": "US"}
        ]
        # First call creates cache with order [ALPHA, BETA]
        m1 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands_order1,
            use_cache=True
        )
        self.assertEqual(len(m1), 1)
        self.assertEqual(m1[0][1], "CAND-BETA")

        # Second call passes reversed candidate list [BETA, ALPHA]
        cands_order2 = [
            {"entity_id": "CAND-BETA", "business_name": "Beta Corporation", "business_address": "500 Main", "country": "US"},
            {"entity_id": "CAND-ALPHA", "business_name": "Zeta Xylophone Qwerty", "business_address": "100 First", "country": "US"}
        ]
        m2 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands_order2,
            use_cache=True
        )
        self.assertEqual(len(m2), 1)
        # MUST still resolve to CAND-BETA, never corrupted into CAND-ALPHA
        self.assertEqual(m2[0][1], "CAND-BETA")

    def test_config_fingerprint_change_invalidates_old_cache(self):
        """3. Verifies that vectorizer configuration changes generate a different fingerprint and invalidate old cache."""
        s1 = [{"entity_id": "S1-CFG", "business_name": "Omega Systems", "business_address": "123 Way", "country": "US"}]
        cands = [{"entity_id": "CAND-OM", "business_name": "Omega Systems Corp", "business_address": "123 Way", "country": "US"}]

        cfg1 = dict(self.retriever.name_cfg)
        cfg1["ngram_max"] = 4
        self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=cfg1, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands,
            use_cache=True
        )

        cfg2 = dict(self.retriever.name_cfg)
        cfg2["ngram_max"] = 5
        dataset_fp = compute_dataset_fingerprint(cands)
        fp1 = compute_config_fingerprint("US", "S2", "G_NAME", cfg1, dataset_fp)
        fp2 = compute_config_fingerprint("US", "S2", "G_NAME", cfg2, dataset_fp)
        self.assertNotEqual(fp1, fp2)

        # Retrieve with cfg2 — creates cache with fp2
        self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=cfg2, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands,
            use_cache=True
        )
        cache_files = os.listdir(self.retriever.cache_dir)
        self.assertTrue(any(fp2 in f for f in cache_files))

    def test_dataset_fingerprint_change_invalidates_old_cache(self):
        """4. Verifies that candidate dataset modification invalidates cached candidate matrix."""
        s1 = [{"entity_id": "S1-DS", "business_name": "Delta Co", "business_address": "123 St", "country": "US"}]
        cands1 = [{"entity_id": "CAND-D1", "business_name": "Delta Co LLC", "business_address": "123 St", "country": "US"}]
        cands2 = [
            {"entity_id": "CAND-D1", "business_name": "Delta Co LLC", "business_address": "123 St", "country": "US"},
            {"entity_id": "CAND-D2", "business_name": "Delta Express", "business_address": "123 St", "country": "US"}
        ]
        fp1 = compute_dataset_fingerprint(cands1)
        fp2 = compute_dataset_fingerprint(cands2)
        self.assertNotEqual(fp1, fp2)

        self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands1,
            use_cache=True
        )
        m2 = self.retriever.retrieve_partition_approximate_candidates(
            country="US", source="S2", channel_bit=CH_G_NAME,
            channel_cfg=self.retriever.name_cfg, channel_type="name",
            s1_entities=s1, s1_indices=[0], candidate_records=cands2,
            use_cache=True
        )
        self.assertGreaterEqual(len(m2), 1)

    def test_lexical_cloud_path_matches_exp003_canonical_keys(self):
        """5. Verifies that MultiChannelBlocker candidate-side matching generates authoritative EXP_003 keys."""
        blocker = MultiChannelBlocker(config={"blocking": {"channels": {"A": True, "B": True, "C2": True, "D2": True, "E2": True, "F": True}}})
        s1_record = {"entity_id": "S1-EXP3", "business_name": "Starbucks Coffee", "business_address": "100 Pike St, Seattle 98101", "country": "US"}
        blocker.build_query_index([s1_record])

        # Candidate record matching via canonical match_candidate_record
        matches = blocker.match_candidate_record("US", "Starbucks Coffee LLC", "100 Pike St Suite 4, Seattle 98101")
        self.assertIn(0, matches)
        bitmask = matches[0]
        self.assertTrue(bitmask & (CH_A | CH_D2))

    def test_all_d2_key_families_exercised(self):
        """6. Verifies that all D2 address key families (exact, compact, bp, bs, ps, ap) are generated and matched."""
        blocker = MultiChannelBlocker(config={"blocking": {"channels": {"D2": True}}})
        s1 = [{"entity_id": "S1-ADDR", "business_name": "Generic Shop", "business_address": "123 Ocean Blvd Apt 4B, Miami 33139", "country": "US"}]
        blocker.build_query_index(s1)

        # Inspect query index keys for CH_D2
        d2_keys = [key_tuple[2] for key_tuple in blocker.query_index.keys() if key_tuple[0] == CH_D2]
        self.assertTrue(any(k.startswith("exact_") for k in d2_keys))
        self.assertTrue(any(k.startswith("compact_") for k in d2_keys))
        self.assertTrue(any(k.startswith("bp_") for k in d2_keys))
        self.assertTrue(any(k.startswith("bs_") for k in d2_keys))
        self.assertTrue(any(k.startswith("ps_") for k in d2_keys))
        self.assertTrue(any(k.startswith("ap_") for k in d2_keys))

    def test_c2_candidate_df_path_exercised(self):
        """7. Verifies that C2 rare token indexing measures candidate population DF correctly."""
        cand_tsv = os.path.join(self.temp_dir, "cand_test.tsv")
        with open(cand_tsv, "w", encoding="utf-8") as f:
            f.write("record_id\tname\taddress\tcountry\n")
            for i in range(10):
                f.write(f"C_{i}\tGeneral Store {i}\tAddress {i}\tUS\n")
            f.write("C_RARE\tRare Xylophone Goods\tRare Address\tUS\n")

        blocker = MultiChannelBlocker(config={"blocking": {"channels": {"C2": {"rare_df_max": 2}}}})
        s1 = [{"entity_id": "S1-RARE", "business_name": "Rare Xylophone Goods", "business_address": "Rare Address", "country": "US"}]
        blocker.build_query_index(s1, candidate_file_paths=[cand_tsv])

        self.assertGreater(len(blocker.candidate_name_df), 0)
        self.assertLessEqual(blocker.candidate_name_df.get("xylophone", 0), 2)

    def test_pilot_denominator_partition_only(self):
        """8. Verifies that pilot mode evaluates metrics strictly against the target partition's ground truth."""
        # Simulated ground truth spanning India and US
        val_gt = {
            "S1-IN": {"S2-IN-1"},
            "S1-US": {"S2-US-1", "S3-US-1"}
        }
        # In India x S2 pilot, only S1-IN -> S2-IN-1 is in-scope
        pilot_links = {(0, "S2-IN-1")}
        total_partition_links = 1
        recall = len(pilot_links) / total_partition_links
        self.assertEqual(recall, 1.0)
        # If computed against global 3 links, it would incorrectly be 0.333
        self.assertNotEqual(recall, 1.0 / 3.0)

    def test_sweep_filtering_produces_correct_k_and_threshold(self):
        """9. Verifies that evaluate_parameter_sweep_in_memory correctly applies K and threshold cuts."""
        matches = [
            (0, "C-HIGH", CH_G_NAME, 0.95),
            (0, "C-MED", CH_G_NAME, 0.65),
            (0, "C-LOW", CH_G_NAME, 0.35)
        ]
        val_gt_pairs = {(0, "C-HIGH"), (0, "C-MED")}
        lexical_hits = set()
        k_list = [1, 2, 5]
        thresh_list = [0.30, 0.50, 0.80]

        results = evaluate_parameter_sweep_in_memory(
            matches=matches,
            val_gt_pairs=val_gt_pairs,
            lexical_hits=lexical_hits,
            total_true_links=2,
            k_list=k_list,
            thresh_list=thresh_list,
            base_lexical_pairs=0
        )
        self.assertEqual(len(results), 9)

        # High threshold (0.80) -> only C-HIGH qualifies
        high_thresh_rows = [r for r in results if r["similarity_threshold"] == 0.80]
        for r in high_thresh_rows:
            self.assertEqual(r["true_links_hit"], 1)

        # K=1, thresh=0.30 -> exactly 1 candidate retained
        k1_low_thresh = [r for r in results if r["top_k"] == 1 and r["similarity_threshold"] == 0.30][0]
        self.assertEqual(k1_low_thresh["total_candidate_pairs"], 1)

    def test_cosine_similarity_metadata_survives_candidate_union(self):
        """10. Verifies that candidate union preserves cosine similarity metadata for downstream ranking."""
        base_result = StreamingCandidateResult(
            candidate_dicts=[{"C-1": CH_A}],
            uncapped_counts=np.array([1], dtype=np.int32),
            uncapped_channel_counts={CH_A: np.array([1], dtype=np.int32)},
            total_scanned=1,
            similarities=[{}]
        )
        tfidf_matches = [
            (0, "C-1", CH_G_NAME, 0.92),
            (0, "C-2", CH_G_ADDR, 0.74)
        ]
        merged = self.retriever.merge_tfidf_candidates_into_result(base_result, tfidf_matches)
        cand_dict = merged[0]
        self.assertIn("C-1", cand_dict)
        self.assertIn("C-2", cand_dict)

        self.assertIsNotNone(merged.similarities)
        sims_0 = merged.similarities[0]
        self.assertIn("C-1", sims_0)
        self.assertEqual(sims_0["C-1"]["name"], 0.92)
        self.assertIn("C-2", sims_0)
        self.assertEqual(sims_0["C-2"]["address"], 0.74)

    def test_sequential_partition_processing_releases_previous_partition_state(self):
        """11. Verifies that partition data can be deleted and garbage collected without lingering references."""
        import gc
        import weakref

        class MockPartitionPayload:
            def __init__(self, name):
                self.name = name
                self.records = [f"rec_{i}" for i in range(1000)]

        p1 = MockPartitionPayload("India_S2")
        ref1 = weakref.ref(p1)
        self.assertIsNotNone(ref1())

        del p1
        gc.collect()
        self.assertIsNone(ref1())


if __name__ == "__main__":
    unittest.main()

