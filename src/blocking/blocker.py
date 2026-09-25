#!/usr/bin/env python3
"""
Multi-Channel Candidate Blocker for Business Entity Resolution — EXP_002.

Implements a union of independently measurable blocking channels:
- Channel A (bit 1): Exact normalized name (legal-stripped, punct-norm)
- Channel B (bit 2): Exact compact & domain-normalized name (compact alphanumeric, domain root)
- Channel C2 (bit 4): Candidate-population-aware rare token inverted index
- Channel D2 (bit 8): True address-only rescue (exact, compact, building+street, postal+street, address pairs)
- Channel E2 (bit 16): Symmetric cross-script transliteration and transliterated tokens
- Channel F (bit 32): Order-invariant distinctive token pairs

Strictly enforces open-set country equality (US -> US, India -> India, France -> France).
Supports authoritative YAML configuration, 2-pass candidate-DF streaming, and deterministic evidence ranking.
"""

import os
import sys
sys.path.insert(0, ".")
import re
import time
import yaml
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional

from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive,
    LEGAL_TERMS,
    COMMON_ADDR_STOP
)

# Channel bitmasks
CH_A = 1    # Exact normalized name
CH_B = 2    # Compact / domain name
CH_C2 = 4   # Candidate-DF-aware rare token
CH_D2 = 8   # True address-only rescue
CH_E2 = 16  # Symmetric cross-script transliteration
CH_F = 32   # Order-invariant distinctive token pairs

CHANNEL_NAMES = {
    CH_A: "A",
    CH_B: "B",
    CH_C2: "C2",
    CH_D2: "D2",
    CH_E2: "E2",
    CH_F: "F"
}


class MultiChannelBlocker:
    """
    Multi-channel candidate generator implementing Channels A, B, C2, D2, E2, F.
    Uses authoritative YAML configuration, streaming 2-pass candidate DF,
    and deterministic candidate evidence ranking.
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
                "name": "EXP_002_default",
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
                    "channel_E2_symmetric_transliteration": {"enabled": True, "token_min_len": 3},
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
                    "max_candidates_per_entity": None,
                    "diagnostic_caps": [None, 500, 250, 100, 50, 30]
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

        f_cfg = ch_cfg.get("channel_F_order_invariant_name", {})
        self.ch_F_enabled = f_cfg.get("enabled", True)
        self.f_max_df = f_cfg.get("max_candidate_df", 25)
        self.f_max_pairs = f_cfg.get("max_pairs_per_entity", 3)

        ranking_cfg = self.config.get("candidate_ranking", {})
        self.ranking_enabled = ranking_cfg.get("enabled", True)
        self.weights = ranking_cfg.get("channel_weights", {
            "CH_A": 10.0, "CH_B": 8.0, "CH_C2": 5.0, "CH_D2": 6.0, "CH_E2": 7.0, "CH_F": 5.0
        })
        self.multi_channel_bonus = ranking_cfg.get("multi_channel_bonus", 2.0)

        # Inverted index: (channel_bit, country, key_str) -> list of s1_int_indices
        self.query_index = defaultdict(list)
        self.s1_id_to_idx = {}
        self.idx_to_s1_id = []
        self.s1_records = []
        
        # Candidate-population DF counters
        self.candidate_name_df = defaultdict(int)
        self.candidate_addr_df = defaultdict(int)

    def scan_candidate_token_frequencies(
        self,
        candidate_file_paths: List[str],
        tracked_name_tokens: Set[str],
        tracked_addr_tokens: Set[str],
        progress_interval: int = 1000000
    ):
        """
        Pass 1: Streams through candidate source files (S2 and S3) and counts
        document frequencies strictly for S1 tracked name and address tokens.
        Memory-efficient: tracks only tokens appearing in the S1 query set.
        """
        print(f"\n[PASS 1] Scanning candidate token frequencies across {len(candidate_file_paths)} candidate files...", flush=True)
        print(f"  Tracking {len(tracked_name_tokens):,} S1 name tokens and {len(tracked_addr_tokens):,} S1 address tokens...", flush=True)
        t0 = time.time()
        total_rows = 0

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
                    name_toks = set(re.findall(r'[a-z0-9]{4,}', raw_name.lower()))
                    for tok in name_toks.intersection(tracked_name_tokens):
                        self.candidate_name_df[tok] += 1

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
        print(f"  Measured candidate DF for {len(self.candidate_name_df):,} name tokens and {len(self.candidate_addr_df):,} address tokens.", flush=True)

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
        tracked_addr_tokens = set()

        for r in s1_records:
            n = normalize_name_non_destructive(r.get("business_name", ""))
            a = normalize_address_non_destructive(r.get("business_address", ""))
            normalized_data.append((n, a, r.get("country", "")))

            for tok in n["distinctive_tokens"]:
                if len(tok) >= self.c2_min_len and tok not in LEGAL_TERMS:
                    tracked_name_tokens.add(tok)
            for tok in a["distinctive_tokens"]:
                if len(tok) >= self.d2_addr_min_len and tok not in COMMON_ADDR_STOP and tok not in LEGAL_TERMS:
                    tracked_addr_tokens.add(tok)

        # Step 2: Pass 1 — Stream candidate files to count frequencies if paths provided
        if candidate_file_paths and (self.ch_C2_enabled or self.ch_D2_enabled or self.ch_F_enabled):
            self.scan_candidate_token_frequencies(
                candidate_file_paths,
                tracked_name_tokens,
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
                c2_cand_tokens = [
                    t for t in n["distinctive_tokens"]
                    if len(t) >= self.c2_min_len and self.candidate_name_df[t] <= self.c2_max_df
                ]
                if c2_cand_tokens:
                    c2_cand_tokens.sort(key=lambda t: (self.candidate_name_df[t], -len(t)))
                    for rt in c2_cand_tokens[:self.c2_max_tokens]:
                        self.query_index[(CH_C2, c, rt)].append(s1_idx)
                        ch_counts[CH_C2] += 1

            # Channel D2: True Address-Only Rescue (Bit 8)
            if self.ch_D2_enabled and a["raw"]:
                # 1. Exact normalized address (if substantial)
                if self.d2_exact and len(a["norm_unicode"]) >= 10:
                    self.query_index[(CH_D2, c, f"exact_{a['norm_unicode']}")].append(s1_idx)
                    ch_counts[CH_D2] += 1

                # 2. Compact normalized address
                if self.d2_compact and len(a["compact_norm"]) >= 10:
                    self.query_index[(CH_D2, c, f"compact_{a['compact_norm']}")].append(s1_idx)
                    ch_counts[CH_D2] += 1

                # 3. Building numeric + street token
                if self.d2_bldg_street:
                    for bldg in a["all_building_numerics"][:2]:
                        for st in a["street_tokens"][:2]:
                            self.query_index[(CH_D2, c, f"bs_{bldg}_{st}")].append(s1_idx)
                            ch_counts[CH_D2] += 1

                # 4. Postal / PIN + street token
                if self.d2_postal_street and a["postal_code"]:
                    for st in a["street_tokens"][:2]:
                        self.query_index[(CH_D2, c, f"ps_{a['postal_code']}_{st}")].append(s1_idx)
                        ch_counts[CH_D2] += 1

                # 5. Distinctive address token pairs (sorted)
                if self.d2_addr_pair:
                    valid_addr_toks = [
                        t for t in a["distinctive_tokens"]
                        if len(t) >= self.d2_addr_min_len and self.candidate_addr_df[t] <= self.d2_addr_max_df
                    ]
                    if len(valid_addr_toks) >= 2:
                        valid_addr_toks.sort(key=lambda t: self.candidate_addr_df[t])
                        # Pair rarest token with up to 2 other distinctive tokens
                        t1 = valid_addr_toks[0]
                        for t2 in valid_addr_toks[1:3]:
                            pair_key = f"ap_{min(t1, t2)}_{max(t1, t2)}"
                            self.query_index[(CH_D2, c, pair_key)].append(s1_idx)
                            ch_counts[CH_D2] += 1

            # Channel E2: Symmetric Cross-Script Transliteration (Bit 16)
            if self.ch_E2_enabled:
                # Transliterated legal-stripped name (registers even if S1 is already ASCII!)
                if n["trans_stripped"]:
                    self.query_index[(CH_E2, c, n["trans_stripped"])].append(s1_idx)
                    ch_counts[CH_E2] += 1
                if len(n["trans_compact"]) >= 4:
                    self.query_index[(CH_E2, c, n["trans_compact"])].append(s1_idx)
                    ch_counts[CH_E2] += 1
                # Transliterated distinctive tokens
                for ttok in n["trans_distinctive_tokens"]:
                    if len(ttok) >= self.e2_min_len and self.candidate_name_df[ttok] <= self.c2_max_df:
                        self.query_index[(CH_E2, c, f"ttok_{ttok}")].append(s1_idx)
                        ch_counts[CH_E2] += 1

            # Channel F: Order-Invariant Name Token Pairs (Bit 32)
            if self.ch_F_enabled:
                dtoks = [
                    t for t in n["distinctive_tokens"]
                    if len(t) >= 4 and self.candidate_name_df[t] <= self.f_max_df
                ]
                if len(dtoks) >= 2:
                    dtoks.sort(key=lambda t: self.candidate_name_df[t])
                    # Take top rarest token and pair with other distinctive tokens
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
        progress_interval: int = 1000000
    ) -> List[Dict[str, int]]:
        """
        Pass 2: Streams through candidate files and queries the inverted index.
        Returns:
            candidates: list of length len(s1_records), where candidates[s1_idx] is dict:
                        cand_id -> bitmask (integer combining CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F)
        """
        n_s1 = len(self.idx_to_s1_id)
        candidates = [{} for _ in range(n_s1)]
        total_scanned = 0
        t_start = time.time()

        print(f"\n[PASS 2] Streaming candidates and querying multi-channel index...", flush=True)

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

                    # Channel A: Exact Normalized Name (Bit 1)
                    if self.ch_A_enabled:
                        if n["legal_stripped"]:
                            for s1_idx in self.query_index.get((CH_A, c, n["legal_stripped"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_A
                        if n["punct_norm"] and n["punct_norm"] != n["legal_stripped"]:
                            for s1_idx in self.query_index.get((CH_A, c, n["punct_norm"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_A

                    # Channel B: Compact & Domain (Bit 2)
                    if self.ch_B_enabled:
                        if len(n["compact_alnum"]) >= 4:
                            for s1_idx in self.query_index.get((CH_B, c, n["compact_alnum"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_B
                        if len(n["legal_stripped_compact"]) >= 4 and n["legal_stripped_compact"] != n["compact_alnum"]:
                            for s1_idx in self.query_index.get((CH_B, c, n["legal_stripped_compact"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_B
                        if n["domain_root"] and len(n["domain_root"]) >= 4:
                            for s1_idx in self.query_index.get((CH_B, c, n["domain_root"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_B

                    # Channel C2: Candidate-DF Rare Token (Bit 4)
                    if self.ch_C2_enabled:
                        for tok in n["distinctive_tokens"]:
                            if len(tok) >= self.c2_min_len:
                                for s1_idx in self.query_index.get((CH_C2, c, tok), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_C2

                    # Channel D2: True Address-Only Rescue (Bit 8)
                    if self.ch_D2_enabled and raw_addr and raw_addr.lower() not in {"", "nan", "<null>", "null", "none"}:
                        a = normalize_address_non_destructive(raw_addr)
                        # Exact & compact
                        if self.d2_exact and len(a["norm_unicode"]) >= 10:
                            for s1_idx in self.query_index.get((CH_D2, c, f"exact_{a['norm_unicode']}"), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D2
                        if self.d2_compact and len(a["compact_norm"]) >= 10:
                            for s1_idx in self.query_index.get((CH_D2, c, f"compact_{a['compact_norm']}"), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D2

                        # Building + street
                        if self.d2_bldg_street:
                            for bldg in a["all_building_numerics"][:2]:
                                for st in a["street_tokens"][:2]:
                                    for s1_idx in self.query_index.get((CH_D2, c, f"bs_{bldg}_{st}"), []):
                                        candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D2

                        # Postal + street
                        if self.d2_postal_street and a["postal_code"]:
                            for st in a["street_tokens"][:2]:
                                for s1_idx in self.query_index.get((CH_D2, c, f"ps_{a['postal_code']}_{st}"), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D2

                        # Address token pairs
                        if self.d2_addr_pair and len(a["distinctive_tokens"]) >= 2:
                            dtoks_a = a["distinctive_tokens"][:4]
                            for i in range(len(dtoks_a)):
                                for j in range(i + 1, len(dtoks_a)):
                                    pair_k = f"ap_{min(dtoks_a[i], dtoks_a[j])}_{max(dtoks_a[i], dtoks_a[j])}"
                                    for s1_idx in self.query_index.get((CH_D2, c, pair_k), []):
                                        candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D2

                    # Channel E2: Symmetric Transliteration (Bit 16)
                    if self.ch_E2_enabled:
                        if n["trans_stripped"]:
                            for s1_idx in self.query_index.get((CH_E2, c, n["trans_stripped"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_E2
                        if len(n["trans_compact"]) >= 4:
                            for s1_idx in self.query_index.get((CH_E2, c, n["trans_compact"]), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_E2
                        for ttok in n["trans_distinctive_tokens"]:
                            if len(ttok) >= self.e2_min_len:
                                for s1_idx in self.query_index.get((CH_E2, c, f"ttok_{ttok}"), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_E2

                    # Channel F: Order-Invariant Name Token Pairs (Bit 32)
                    if self.ch_F_enabled and len(n["distinctive_tokens"]) >= 2:
                        dtoks = n["distinctive_tokens"][:4]
                        for i in range(len(dtoks)):
                            for j in range(i + 1, len(dtoks)):
                                p_str = f"pair_{min(dtoks[i], dtoks[j])}_{max(dtoks[i], dtoks[j])}"
                                for s1_idx in self.query_index.get((CH_F, c, p_str), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_F

                    if rows_in_file % progress_interval == 0:
                        now = time.time()
                        rate = rows_in_file / (now - t_file)
                        print(f"  Scanned {rows_in_file:,} rows ({rate:.0f} rows/s)...", flush=True)

            print(f"Finished {path}: {rows_in_file:,} rows in {time.time()-t_file:.2f}s", flush=True)

        print(f"\nTotal candidate scanning complete: {total_scanned:,} rows in {time.time()-t_start:.2f}s", flush=True)
        return candidates

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

        if channels_active > 1:
            score += (channels_active - 1) * self.multi_channel_bonus

        return score

    def rank_entity_candidates(
        self,
        cand_dict: Dict[str, int],
        cap: Optional[int] = None
    ) -> List[str]:
        """
        Deterministically ranks candidates for an S1 entity:
        1. evidence score descending
        2. candidate entity_id ascending (tie-break)
        """
        if not cand_dict:
            return []

        # (cand_id, score)
        scored = [
            (cid, self.compute_candidate_evidence_score(mask))
            for cid, mask in cand_dict.items()
        ]

        # Deterministic sort: -score (highest score first), cid (alphabetical tie-break)
        scored.sort(key=lambda x: (-x[1], x[0]))

        if cap is not None:
            return [cid for cid, _ in scored[:cap]]
        return [cid for cid, _ in scored]
