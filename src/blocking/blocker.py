#!/usr/bin/env python3
"""
Multi-Channel Candidate Blocker for Business Entity Resolution — EXP_003.

Implements a union of independently measurable blocking channels:
- Channel A (bit 1): Exact normalized name (legal-stripped, punct-norm)
- Channel B (bit 2): Exact compact & domain-normalized name (compact alphanumeric, domain root)
- Channel C2 (bit 4): Candidate-population-aware rare token inverted index
- Channel D2 (bit 8): True address-only rescue (exact, compact, building+street, postal+street, address pairs)
- Channel E2 (bit 16): Symmetric cross-script transliteration and transliterated tokens
- Channel F (bit 32): Order-invariant distinctive token pairs

Strictly enforces open-set country equality (US -> US, India -> India, France -> France).
Supports authoritative YAML configuration, 2-pass candidate-DF streaming,
Mode A unbounded ceiling evaluation, and Mode B bounded evidence-ranked candidate heaps.
"""

import os
import sys
sys.path.insert(0, ".")
import re
import time
import heapq
import yaml
import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional

from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive,
    LEGAL_TERMS,
    COMMON_ADDR_STOP
)

# Channel bitmasks
CH_A = 1       # Exact normalized name
CH_B = 2       # Compact / domain name
CH_C2 = 4      # Candidate-DF-aware rare token
CH_D2 = 8      # True address-only rescue
CH_E2 = 16     # Symmetric cross-script transliteration
CH_F = 32      # Order-invariant distinctive token pairs
CH_G_NAME = 64 # Character n-gram TF-IDF approximate name
CH_G_ADDR = 128 # Character n-gram TF-IDF approximate address
CH_G_TRANS = 256 # Character n-gram TF-IDF approximate transliterated name

CHANNEL_NAMES = {
    CH_A: "A",
    CH_B: "B",
    CH_C2: "C2",
    CH_D2: "D2",
    CH_E2: "E2",
    CH_F: "F",
    CH_G_NAME: "G_NAME",
    CH_G_ADDR: "G_ADDR",
    CH_G_TRANS: "G_TRANS"
}


class CandidateHeapItem:
    """
    Candidate element stored in a bounded min-heap for Top-K evidence ranking.
    Min-heap ordering prioritizes evicting:
    1. Lower evidence scores first.
    2. For equal scores, lexicographically LARGER candidate IDs first (so smaller IDs are kept).
    """
    __slots__ = ("score", "cand_id", "bitmask")

    def __init__(self, score: float, cand_id: str, bitmask: int):
        self.score = score
        self.cand_id = cand_id
        self.bitmask = bitmask

    def __lt__(self, other: "CandidateHeapItem") -> bool:
        if self.score != other.score:
            return self.score < other.score
        return self.cand_id > other.cand_id

    def __gt__(self, other: "CandidateHeapItem") -> bool:
        if self.score != other.score:
            return self.score > other.score
        return self.cand_id < other.cand_id

    def __le__(self, other: "CandidateHeapItem") -> bool:
        return not (self > other)

    def __ge__(self, other: "CandidateHeapItem") -> bool:
        return not (self < other)

    def __eq__(self, other: "CandidateHeapItem") -> bool:
        return self.score == other.score and self.cand_id == other.cand_id


class StreamingCandidateResult(list):
    """
    Subclasses list so candidates[s1_idx] returns {cand_id: bitmask}
    maintaining 100% backward compatibility with all existing tests and scripts,
    while also exposing Mode A ceiling gt_hits, bounded heaps, and uncapped counts.
    """
    def __init__(
        self,
        candidate_dicts: List[Dict[str, int]],
        heaps: Optional[List[List[CandidateHeapItem]]] = None,
        gt_hits: Optional[Dict[Tuple[int, str], int]] = None,
        uncapped_counts: Optional[np.ndarray] = None,
        uncapped_channel_counts: Optional[Dict[int, np.ndarray]] = None,
        total_scanned: int = 0
    ):
        super().__init__(candidate_dicts)
        self.heaps = heaps if heaps is not None else []
        self.gt_hits = gt_hits if gt_hits is not None else {}
        self.uncapped_counts = uncapped_counts
        self.uncapped_channel_counts = uncapped_channel_counts if uncapped_channel_counts is not None else {}
        self.total_scanned = total_scanned


class MultiChannelBlocker:
    """
    Multi-channel candidate generator implementing Channels A, B, C2, D2, E2, F.
    Uses authoritative YAML configuration, streaming 2-pass candidate DF,
    Mode A unbounded ceiling evaluation, and Mode B bounded evidence-ranked heaps.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # Fallback default configuration
            self.config = {
                "name": "EXP_003_default",
                "channels": {
                    "channel_A_exact_name": {"enabled": True},
                    "channel_B_compact_domain": {"enabled": True},
                    "channel_C2_candidate_df_rare_token": {
                        "enabled": True, "min_token_len": 4, "max_candidate_df": 5, "max_tokens_per_entity": 2
                    },
                    "channel_D2_address_rescue": {
                        "enabled": True, "exact_normalized": True, "compact_normalized": True,
                        "building_and_street": True, "postal_and_street": True, "address_token_pair": True,
                        "address_token_max_candidate_df": 25, "address_token_min_len": 4
                    },
                    "channel_E2_symmetric_transliteration": {"enabled": True, "token_min_len": 3, "max_candidate_df": 5},
                    "channel_F_order_invariant_name": {
                        "enabled": True, "max_candidate_df": 25, "max_pairs_per_entity": 3
                    }
                },
                "candidate_ranking": {
                    "enabled": True,
                    "channel_weights": {
                        "CH_A": 10.0, "CH_B": 8.0, "CH_C2": 5.0, "CH_D2": 6.0, "CH_E2": 7.0, "CH_F": 5.0
                    },
                    "multi_channel_bonus": 2.0,
                    "tie_break": "entity_id_asc"
                },
                "constraints": {
                    "max_heap_k": 2000,
                    "production_caps": [2000, 1000, 500, 250, 100]
                }
            }

        # Parse config settings
        ch_cfg = self.config.get("channels", {})
        self.ch_A_enabled = ch_cfg.get("channel_A_exact_name", {}).get("enabled", True)
        self.ch_B_enabled = ch_cfg.get("channel_B_compact_domain", {}).get("enabled", True)
        
        c2_cfg = ch_cfg.get("channel_C2_candidate_df_rare_token", {})
        self.ch_C2_enabled = c2_cfg.get("enabled", True)
        self.c2_min_len = c2_cfg.get("min_token_len", 4)
        self.c2_max_df = c2_cfg.get("max_candidate_df", 5)
        self.c2_max_tokens = c2_cfg.get("max_tokens_per_entity", 2)

        d2_cfg = ch_cfg.get("channel_D2_address_rescue", {})
        self.ch_D2_enabled = d2_cfg.get("enabled", True)
        self.d2_exact = d2_cfg.get("exact_normalized", True)
        self.d2_compact = d2_cfg.get("compact_normalized", True)
        self.d2_bldg_street = d2_cfg.get("building_and_street", True)
        self.d2_postal_street = d2_cfg.get("postal_and_street", True)
        self.d2_addr_pair = d2_cfg.get("address_token_pair", True)
        self.d2_addr_max_df = d2_cfg.get("address_token_max_candidate_df", 25)
        self.d2_addr_min_len = d2_cfg.get("address_token_min_len", 4)

        e2_cfg = ch_cfg.get("channel_E2_symmetric_transliteration", {})
        self.ch_E2_enabled = e2_cfg.get("enabled", True)
        self.e2_min_len = e2_cfg.get("token_min_len", 3)
        self.e2_max_df = e2_cfg.get("max_candidate_df", self.c2_max_df)

        f_cfg = ch_cfg.get("channel_F_order_invariant_name", {})
        self.ch_F_enabled = f_cfg.get("enabled", True)
        self.f_max_df = f_cfg.get("max_candidate_df", 25)
        self.f_max_pairs = f_cfg.get("max_pairs_per_entity", 3)

        ranking_cfg = self.config.get("candidate_ranking", {})
        self.ranking_enabled = ranking_cfg.get("enabled", True)
        self.weights = ranking_cfg.get("channel_weights", {
            "CH_A": 10.0, "CH_B": 8.0, "CH_C2": 5.0, "CH_D2": 6.0, "CH_E2": 7.0, "CH_F": 5.0,
            "CH_G_NAME": 4.0, "CH_G_ADDR": 4.0, "CH_G_TRANS": 4.0
        })
        self.multi_channel_bonus = ranking_cfg.get("multi_channel_bonus", 2.0)

        constraints_cfg = self.config.get("constraints", {})
        self.max_heap_k = constraints_cfg.get("max_heap_k", 2000)

        # Inverted index: (channel_bit, country, key_str) -> list of s1_int_indices
        self.query_index = defaultdict(list)
        self.s1_id_to_idx = {}
        self.idx_to_s1_id = []
        self.s1_records = []
        
        # Candidate-population DF counters
        self.candidate_name_df = defaultdict(int)
        self.candidate_trans_df = defaultdict(int)
        self.candidate_addr_df = defaultdict(int)
        self.candidate_df_measured = False

    def scan_candidate_token_frequencies(
        self,
        candidate_file_paths: List[str],
        tracked_name_tokens: Set[str],
        tracked_trans_tokens: Set[str],
        tracked_addr_tokens: Set[str],
        progress_interval: int = 1000000
    ):
        """
        Pass 1: Streams through candidate source files (S2 and S3) and counts
        document frequencies strictly for S1 tracked name, transliterated, and address tokens.
        Ensures non-Latin candidates are transliterated through the exact same normalization pipeline.
        """
        print(f"\n[PASS 1] Scanning candidate token frequencies across {len(candidate_file_paths)} candidate files...", flush=True)
        print(f"  Tracking {len(tracked_name_tokens):,} name, {len(tracked_trans_tokens):,} transliterated, and {len(tracked_addr_tokens):,} address tokens...", flush=True)
        t0 = time.time()
        total_rows = 0
        self.candidate_df_measured = True

        for path in candidate_file_paths:
            t_file = time.time()
            rows_file = 0
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\n").split("\t")
                col_name = header.index("business_name") if "business_name" in header else 1
                col_addr = header.index("business_address") if "business_address" in header else 2

                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(col_name, col_addr):
                        continue
                    rows_file += 1
                    total_rows += 1

                    raw_name = parts[col_name]
                    raw_addr = parts[col_addr]

                    # Tokenize name
                    if raw_name:
                        if raw_name.isascii():
                            name_toks = set(re.findall(r'[a-z0-9]{3,}', raw_name.lower())) - LEGAL_TERMS
                            for tok in name_toks.intersection(tracked_name_tokens):
                                self.candidate_name_df[tok] += 1
                            for tok in name_toks.intersection(tracked_trans_tokens):
                                self.candidate_trans_df[tok] += 1
                        else:
                            # Non-Latin candidate: normalize through exact same pipeline
                            norm_cand = normalize_name_non_destructive(raw_name)
                            for tok in set(norm_cand["distinctive_tokens"]).intersection(tracked_name_tokens):
                                self.candidate_name_df[tok] += 1
                            for tok in set(norm_cand["trans_distinctive_tokens"]).intersection(tracked_trans_tokens):
                                self.candidate_trans_df[tok] += 1

                    # Tokenize address
                    if raw_addr and raw_addr.lower() not in {"", "nan", "<null>", "null", "none"}:
                        addr_toks = set(re.findall(r'[a-z0-9]{4,}', raw_addr.lower())) - COMMON_ADDR_STOP
                        for tok in addr_toks.intersection(tracked_addr_tokens):
                            self.candidate_addr_df[tok] += 1

                    if rows_file % progress_interval == 0:
                        now = time.time()
                        print(f"  Scanned {rows_file:,} rows from {os.path.basename(path)} ({rows_file/(now-t_file):.0f} rows/s)...", flush=True)

            print(f"  Completed {os.path.basename(path)}: {rows_file:,} rows in {time.time()-t_file:.2f}s", flush=True)

        print(f"[PASS 1 COMPLETE] Scanned {total_rows:,} total rows in {time.time()-t0:.2f}s.", flush=True)
        print(f"  Measured candidate DF for {len(self.candidate_name_df):,} name, {len(self.candidate_trans_df):,} transliterated, and {len(self.candidate_addr_df):,} address tokens.", flush=True)

    def build_query_index(
        self,
        s1_records: List[Dict[str, Any]],
        candidate_file_paths: Optional[List[str]] = None
    ):
        """
        Builds query inverted index from Source 1 entities with candidate-DF awareness.
        """
        print(f"\nBuilding query inverted index for {len(s1_records):,} Source 1 entities...", flush=True)
        t0 = time.time()

        self.s1_records = s1_records
        self.idx_to_s1_id = [r["entity_id"] for r in s1_records]
        self.s1_id_to_idx = {r["entity_id"]: idx for idx, r in enumerate(s1_records)}

        # Step 1: Pre-normalize all S1 records
        normalized_data = []
        tracked_name_tokens = set()
        tracked_trans_tokens = set()
        tracked_addr_tokens = set()

        for r in s1_records:
            n = normalize_name_non_destructive(r.get("business_name", ""))
            a = normalize_address_non_destructive(r.get("business_address", ""))
            normalized_data.append((n, a, r.get("country", "")))

            for tok in n["distinctive_tokens"]:
                if len(tok) >= self.c2_min_len and tok not in LEGAL_TERMS:
                    tracked_name_tokens.add(tok)
            for ttok in n["trans_distinctive_tokens"]:
                if len(ttok) >= self.e2_min_len and ttok not in LEGAL_TERMS:
                    tracked_trans_tokens.add(ttok)
            for tok in a["distinctive_tokens"]:
                if len(tok) >= self.d2_addr_min_len and tok not in COMMON_ADDR_STOP and tok not in LEGAL_TERMS:
                    tracked_addr_tokens.add(tok)
            for tok in a["street_tokens"]:
                if len(tok) >= 4 and tok not in COMMON_ADDR_STOP and tok not in LEGAL_TERMS:
                    tracked_addr_tokens.add(tok)

        # Step 2: Pass 1 — Stream candidate files to count frequencies if paths provided
        if candidate_file_paths and (self.ch_C2_enabled or self.ch_D2_enabled or self.ch_E2_enabled or self.ch_F_enabled):
            self.scan_candidate_token_frequencies(
                candidate_file_paths,
                tracked_name_tokens,
                tracked_trans_tokens,
                tracked_addr_tokens
            )

        # Step 3: Populate inverted index with bitmask channel keys
        ch_counts = defaultdict(int)

        for s1_idx, (n, a, c) in enumerate(normalized_data):
            # Channel A: Exact Normalized Name (Bit 1)
            if self.ch_A_enabled:
                if n["legal_stripped"]:
                    self.query_index[(CH_A, c, n["legal_stripped"])].append(s1_idx)
                    ch_counts[CH_A] += 1
                if n["punct_norm"] and n["punct_norm"] != n["legal_stripped"]:
                    self.query_index[(CH_A, c, n["punct_norm"])].append(s1_idx)
                    ch_counts[CH_A] += 1

            # Channel B: Compact & Domain-Normalized Name (Bit 2)
            if self.ch_B_enabled:
                if len(n["compact_alnum"]) >= 4:
                    self.query_index[(CH_B, c, n["compact_alnum"])].append(s1_idx)
                    ch_counts[CH_B] += 1
                if len(n["legal_stripped_compact"]) >= 4 and n["legal_stripped_compact"] != n["compact_alnum"]:
                    self.query_index[(CH_B, c, n["legal_stripped_compact"])].append(s1_idx)
                    ch_counts[CH_B] += 1
                if n["domain_root"] and len(n["domain_root"]) >= 4:
                    self.query_index[(CH_B, c, n["domain_root"])].append(s1_idx)
                    ch_counts[CH_B] += 1

            # Channel C2: Candidate-DF-Aware Rare Distinctive Name Tokens (Bit 4)
            if self.ch_C2_enabled:
                if self.candidate_df_measured:
                    c2_cand_tokens = [
                        t for t in n["distinctive_tokens"]
                        if len(t) >= self.c2_min_len and t not in LEGAL_TERMS and self.candidate_name_df[t] <= self.c2_max_df
                    ]
                else:
                    c2_cand_tokens = [
                        t for t in n["distinctive_tokens"]
                        if len(t) >= self.c2_min_len and t not in LEGAL_TERMS
                    ]
                if c2_cand_tokens:
                    c2_cand_tokens.sort(key=lambda t: (self.candidate_name_df[t], -len(t)))
                    for rt in c2_cand_tokens[:self.c2_max_tokens]:
                        self.query_index[(CH_C2, c, rt)].append(s1_idx)
                        ch_counts[CH_C2] += 1

            # Channel D2: True Address-Only Rescue (Bit 8)
            if self.ch_D2_enabled and a["raw"]:
                # 1. Exact normalized address (if substantial)
                if self.d2_exact and len(a["norm_unicode"]) >= 12:
                    self.query_index[(CH_D2, c, f"exact_{a['norm_unicode']}")].append(s1_idx)
                    ch_counts[CH_D2] += 1

                # 2. Compact normalized address
                if self.d2_compact and len(a["compact_norm"]) >= 15:
                    self.query_index[(CH_D2, c, f"compact_{a['compact_norm']}")].append(s1_idx)
                    ch_counts[CH_D2] += 1

                # 3. Building numeric + Postal code (highly discriminative physical location)
                if a["building_numeric"] and a["postal_code"] and len(a["postal_code"]) in (5, 6):
                    self.query_index[(CH_D2, c, f"bp_{a['building_numeric']}_{a['postal_code']}")].append(s1_idx)
                    ch_counts[CH_D2] += 1

                # 4. Building numeric + street token (ONLY if street token is selective)
                if self.d2_bldg_street and a["building_numeric"]:
                    for st in a["street_tokens"][:2]:
                        if len(st) >= 4 and st not in COMMON_ADDR_STOP:
                            if not self.candidate_df_measured or self.candidate_addr_df[st] <= self.d2_addr_max_df:
                                self.query_index[(CH_D2, c, f"bs_{a['building_numeric']}_{st}")].append(s1_idx)
                                ch_counts[CH_D2] += 1

                # 5. Postal / PIN + street token (ONLY if street token is selective)
                if self.d2_postal_street and a["postal_code"]:
                    for st in a["street_tokens"][:2]:
                        if len(st) >= 4 and st not in COMMON_ADDR_STOP:
                            if not self.candidate_df_measured or self.candidate_addr_df[st] <= self.d2_addr_max_df:
                                self.query_index[(CH_D2, c, f"ps_{a['postal_code']}_{st}")].append(s1_idx)
                                ch_counts[CH_D2] += 1

                # 6. Distinctive address token pairs (sorted, both selective)
                if self.d2_addr_pair:
                    if self.candidate_df_measured:
                        valid_addr_toks = [
                            t for t in a["distinctive_tokens"]
                            if len(t) >= self.d2_addr_min_len and t not in COMMON_ADDR_STOP and self.candidate_addr_df[t] <= 15
                        ]
                    else:
                        valid_addr_toks = [
                            t for t in a["distinctive_tokens"]
                            if len(t) >= self.d2_addr_min_len and t not in COMMON_ADDR_STOP
                        ]
                    if len(valid_addr_toks) >= 2:
                        valid_addr_toks.sort(key=lambda t: self.candidate_addr_df[t])
                        t1 = valid_addr_toks[0]
                        for t2 in valid_addr_toks[1:3]:
                            pair_key = f"ap_{min(t1, t2)}_{max(t1, t2)}"
                            self.query_index[(CH_D2, c, pair_key)].append(s1_idx)
                            ch_counts[CH_D2] += 1

            # Channel E2: Symmetric Cross-Script Transliteration (Bit 16)
            if self.ch_E2_enabled:
                # Transliterated legal-stripped name (registers even if S1 is already ASCII!)
                if n["trans_stripped"] and n["trans_stripped"] not in LEGAL_TERMS:
                    self.query_index[(CH_E2, c, n["trans_stripped"])].append(s1_idx)
                    ch_counts[CH_E2] += 1
                if len(n["trans_compact"]) >= 6:
                    self.query_index[(CH_E2, c, n["trans_compact"])].append(s1_idx)
                    ch_counts[CH_E2] += 1
                # Transliterated distinctive tokens (strictly selective with transliteration candidate DF)
                for ttok in n["trans_distinctive_tokens"]:
                    if len(ttok) >= self.e2_min_len and ttok not in LEGAL_TERMS:
                        if not self.candidate_df_measured or self.candidate_trans_df[ttok] <= self.e2_max_df:
                            self.query_index[(CH_E2, c, f"ttok_{ttok}")].append(s1_idx)
                            ch_counts[CH_E2] += 1

            # Channel F: Order-Invariant Name Token Pairs (Bit 32)
            if self.ch_F_enabled:
                if self.candidate_df_measured:
                    dtoks = [
                        t for t in n["distinctive_tokens"]
                        if len(t) >= 4 and t not in LEGAL_TERMS and self.candidate_name_df[t] <= self.f_max_df
                    ]
                else:
                    dtoks = [
                        t for t in n["distinctive_tokens"]
                        if len(t) >= 4 and t not in LEGAL_TERMS
                    ]
                if len(dtoks) >= 2:
                    dtoks.sort(key=lambda t: self.candidate_name_df[t])
                    # Require at least one token to have DF <= 10 when measured
                    if not self.candidate_df_measured or self.candidate_name_df[dtoks[0]] <= 10:
                        t1 = dtoks[0]
                        pairs_indexed = 0
                        for t2 in dtoks[1:]:
                            pair_str = f"pair_{min(t1, t2)}_{max(t1, t2)}"
                            self.query_index[(CH_F, c, pair_str)].append(s1_idx)
                            ch_counts[CH_F] += 1
                            pairs_indexed += 1
                            if pairs_indexed >= self.f_max_pairs:
                                break

        print(f"\n[QUERY INDEX BUILT] in {time.time()-t0:.2f}s across {len(self.query_index):,} total keys:", flush=True)
        for bit, ch_name in CHANNEL_NAMES.items():
            print(f"  Channel {ch_name:2s}: {ch_counts[bit]:,} keys", flush=True)

    def generate_candidates_streaming(
        self,
        candidate_file_paths: List[str],
        progress_interval: int = 1000000,
        max_streaming_candidates_per_entity: Optional[int] = None,
        max_heap_k: Optional[int] = 2000,
        gt_links_set: Optional[Set[Tuple[int, str]]] = None
    ) -> StreamingCandidateResult:
        """
        Pass 2: Streams through candidate files and queries the inverted index.
        Executes:
        - Mode A (Ceiling Mode): If gt_links_set provided, records true links hit unconditionally without any caps.
        - Mode B (Ranked Production Mode): Maintains bounded Top-K heaps per entity based on deterministic evidence score.
        - Exact uncapped candidate volume counting per entity.
        """
        n_s1 = len(self.idx_to_s1_id)
        # Determine heap K limit
        effective_k = max_heap_k if max_heap_k is not None else max_streaming_candidates_per_entity

        heaps: List[List[CandidateHeapItem]] = [[] for _ in range(n_s1)]
        gt_hits: Dict[Tuple[int, str], int] = defaultdict(int) if gt_links_set is not None else {}
        uncapped_counts = np.zeros(n_s1, dtype=np.int32)
        uncapped_channel_counts = {
            CH_A: np.zeros(n_s1, dtype=np.int32),
            CH_B: np.zeros(n_s1, dtype=np.int32),
            CH_C2: np.zeros(n_s1, dtype=np.int32),
            CH_D2: np.zeros(n_s1, dtype=np.int32),
            CH_E2: np.zeros(n_s1, dtype=np.int32),
            CH_F: np.zeros(n_s1, dtype=np.int32),
            CH_G_NAME: np.zeros(n_s1, dtype=np.int32),
            CH_G_ADDR: np.zeros(n_s1, dtype=np.int32),
            CH_G_TRANS: np.zeros(n_s1, dtype=np.int32),
        }

        total_scanned = 0
        t_start = time.time()

        print(f"\n[PASS 2] Streaming candidates and querying multi-channel index...", flush=True)
        if gt_links_set is not None:
            print(f"  Mode A (Ceiling Mode): tracking {len(gt_links_set):,} ground-truth pairs unconditionally.", flush=True)
        print(f"  Mode B (Ranked Heap Mode): bounded top-K heap per entity with K={effective_k}.", flush=True)

        for path in candidate_file_paths:
            print(f"Streaming candidate population from {path}...", flush=True)
            t_file = time.time()
            rows_in_file = 0

            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\n").split("\t")
                col_id = header.index("entity_id") if "entity_id" in header else 0
                col_name = header.index("business_name") if "business_name" in header else 1
                col_addr = header.index("business_address") if "business_address" in header else 2
                col_country = header.index("country") if "country" in header else 3

                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= col_country:
                        continue
                    rows_in_file += 1
                    total_scanned += 1

                    cand_id = parts[col_id]
                    c = parts[col_country]
                    raw_name = parts[col_name]
                    raw_addr = parts[col_addr]

                    n = normalize_name_non_destructive(raw_name)
                    row_s1_matches = {}

                    # Channel A: Exact Normalized Name (Bit 1)
                    if self.ch_A_enabled:
                        if n["legal_stripped"]:
                            for s1_idx in self.query_index.get((CH_A, c, n["legal_stripped"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_A
                        if n["punct_norm"] and n["punct_norm"] != n["legal_stripped"]:
                            for s1_idx in self.query_index.get((CH_A, c, n["punct_norm"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_A

                    # Channel B: Compact & Domain (Bit 2)
                    if self.ch_B_enabled:
                        if len(n["compact_alnum"]) >= 4:
                            for s1_idx in self.query_index.get((CH_B, c, n["compact_alnum"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_B
                        if len(n["legal_stripped_compact"]) >= 4 and n["legal_stripped_compact"] != n["compact_alnum"]:
                            for s1_idx in self.query_index.get((CH_B, c, n["legal_stripped_compact"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_B
                        if n["domain_root"] and len(n["domain_root"]) >= 4:
                            for s1_idx in self.query_index.get((CH_B, c, n["domain_root"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_B

                    # Channel C2: Candidate-DF Rare Token (Bit 4)
                    if self.ch_C2_enabled:
                        for tok in n["distinctive_tokens"]:
                            if len(tok) >= self.c2_min_len:
                                for s1_idx in self.query_index.get((CH_C2, c, tok), []):
                                    row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_C2

                    # Channel D2: True Address-Only Rescue (Bit 8)
                    if self.ch_D2_enabled and raw_addr and raw_addr.lower() not in {"", "nan", "<null>", "null", "none"}:
                        a = normalize_address_non_destructive(raw_addr)
                        # Exact & compact
                        if self.d2_exact and len(a["norm_unicode"]) >= 12:
                            for s1_idx in self.query_index.get((CH_D2, c, f"exact_{a['norm_unicode']}"), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_D2
                        if self.d2_compact and len(a["compact_norm"]) >= 15:
                            for s1_idx in self.query_index.get((CH_D2, c, f"compact_{a['compact_norm']}"), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_D2

                        # Building + Postal
                        if a["building_numeric"] and a["postal_code"]:
                            for s1_idx in self.query_index.get((CH_D2, c, f"bp_{a['building_numeric']}_{a['postal_code']}"), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_D2

                        # Building + street
                        if self.d2_bldg_street and a["building_numeric"]:
                            for st in a["street_tokens"][:2]:
                                for s1_idx in self.query_index.get((CH_D2, c, f"bs_{a['building_numeric']}_{st}"), []):
                                    row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_D2

                        # Postal + street
                        if self.d2_postal_street and a["postal_code"]:
                            for st in a["street_tokens"][:2]:
                                for s1_idx in self.query_index.get((CH_D2, c, f"ps_{a['postal_code']}_{st}"), []):
                                    row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_D2

                        # Address token pairs
                        if self.d2_addr_pair and len(a["distinctive_tokens"]) >= 2:
                            dtoks_a = a["distinctive_tokens"][:4]
                            for i in range(len(dtoks_a)):
                                for j in range(i + 1, len(dtoks_a)):
                                    pair_k = f"ap_{min(dtoks_a[i], dtoks_a[j])}_{max(dtoks_a[i], dtoks_a[j])}"
                                    for s1_idx in self.query_index.get((CH_D2, c, pair_k), []):
                                        row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_D2

                    # Channel E2: Symmetric Transliteration (Bit 16)
                    if self.ch_E2_enabled:
                        if n["trans_stripped"]:
                            for s1_idx in self.query_index.get((CH_E2, c, n["trans_stripped"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_E2
                        if len(n["trans_compact"]) >= 6:
                            for s1_idx in self.query_index.get((CH_E2, c, n["trans_compact"]), []):
                                row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_E2
                        for ttok in n["trans_distinctive_tokens"]:
                            if len(ttok) >= self.e2_min_len:
                                for s1_idx in self.query_index.get((CH_E2, c, f"ttok_{ttok}"), []):
                                    row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_E2

                    # Channel F: Order-Invariant Name Token Pairs (Bit 32)
                    if self.ch_F_enabled and len(n["distinctive_tokens"]) >= 2:
                        dtoks = n["distinctive_tokens"][:4]
                        for i in range(len(dtoks)):
                            for j in range(i + 1, len(dtoks)):
                                p_str = f"pair_{min(dtoks[i], dtoks[j])}_{max(dtoks[i], dtoks[j])}"
                                for s1_idx in self.query_index.get((CH_F, c, p_str), []):
                                    row_s1_matches[s1_idx] = row_s1_matches.get(s1_idx, 0) | CH_F

                    # Process all matches for this candidate row
                    if row_s1_matches:
                        for s1_idx, bitmask in row_s1_matches.items():
                            # 1. Uncapped volume accounting
                            uncapped_counts[s1_idx] += 1
                            for b in (CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F, CH_G_NAME, CH_G_ADDR, CH_G_TRANS):
                                if bitmask & b:
                                    uncapped_channel_counts[b][s1_idx] += 1

                            # 2. Mode A — Unbounded Ceiling Ground-Truth Hit Recording
                            if gt_links_set is not None and (s1_idx, cand_id) in gt_links_set:
                                gt_hits[(s1_idx, cand_id)] |= bitmask

                            # 3. Mode B — Evidence-Ranked Bounded Top-K Min-Heap
                            score = self.compute_candidate_evidence_score(bitmask)
                            item = CandidateHeapItem(score, cand_id, bitmask)
                            h = heaps[s1_idx]

                            if effective_k is None or len(h) < effective_k:
                                heapq.heappush(h, item)
                            elif item > h[0]:
                                heapq.heapreplace(h, item)

                    if rows_in_file % progress_interval == 0:
                        now = time.time()
                        rate = rows_in_file / (now - t_file)
                        print(f"  Scanned {rows_in_file:,} rows ({rate:.0f} rows/s)...", flush=True)

            print(f"Finished {path}: {rows_in_file:,} rows in {time.time()-t_file:.2f}s", flush=True)

        print(f"\nTotal candidate scanning complete: {total_scanned:,} rows in {time.time()-t_start:.2f}s", flush=True)

        # Convert heaps to candidate dicts for compatibility
        candidate_dicts = [
            {item.cand_id: item.bitmask for item in h}
            for h in heaps
        ]

        return StreamingCandidateResult(
            candidate_dicts=candidate_dicts,
            heaps=heaps,
            gt_hits=gt_hits,
            uncapped_counts=uncapped_counts,
            uncapped_channel_counts=uncapped_channel_counts,
            total_scanned=total_scanned
        )

    def compute_candidate_evidence_score(self, bitmask: int) -> float:
        """
        Computes deterministic retrieval evidence score for a candidate based on firing channels.
        """
        score = 0.0
        channels_active = 0

        if bitmask & CH_A:
            score += self.weights.get("CH_A", 10.0)
            channels_active += 1
        if bitmask & CH_B:
            score += self.weights.get("CH_B", 8.0)
            channels_active += 1
        if bitmask & CH_C2:
            score += self.weights.get("CH_C2", 5.0)
            channels_active += 1
        if bitmask & CH_D2:
            score += self.weights.get("CH_D2", 6.0)
            channels_active += 1
        if bitmask & CH_E2:
            score += self.weights.get("CH_E2", 7.0)
            channels_active += 1
        if bitmask & CH_F:
            score += self.weights.get("CH_F", 5.0)
            channels_active += 1
        if bitmask & CH_G_NAME:
            score += self.weights.get("CH_G_NAME", 4.0)
            channels_active += 1
        if bitmask & CH_G_ADDR:
            score += self.weights.get("CH_G_ADDR", 4.0)
            channels_active += 1
        if bitmask & CH_G_TRANS:
            score += self.weights.get("CH_G_TRANS", 4.0)
            channels_active += 1

        if channels_active > 1:
            score += (channels_active - 1) * self.multi_channel_bonus

        return score

    def rank_entity_candidates(
        self,
        cand_dict_or_heap: Any,
        cap: Optional[int] = None
    ) -> List[str]:
        """
        Deterministically ranks candidates for an S1 entity:
        1. evidence score descending
        2. candidate entity_id ascending (tie-break)
        Supports both candidate dictionary {cand_id: bitmask} and heap List[CandidateHeapItem].
        """
        if not cand_dict_or_heap:
            return []

        if isinstance(cand_dict_or_heap, list) and cand_dict_or_heap and isinstance(cand_dict_or_heap[0], CandidateHeapItem):
            # Already scored heap items: sort best to worst
            sorted_items = sorted(cand_dict_or_heap, key=lambda item: (-item.score, item.cand_id))
            if cap is not None:
                sorted_items = sorted_items[:cap]
            return [item.cand_id for item in sorted_items]
        elif isinstance(cand_dict_or_heap, dict):
            scored = [
                (cid, self.compute_candidate_evidence_score(mask))
                for cid, mask in cand_dict_or_heap.items()
            ]
            scored.sort(key=lambda x: (-x[1], x[0]))
            if cap is not None:
                return [cid for cid, _ in scored[:cap]]
            return [cid for cid, _ in scored]
        else:
            return []
