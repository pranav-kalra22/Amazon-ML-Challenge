#!/usr/bin/env python3
"""
Multi-Channel Candidate Blocker for Business Entity Resolution.

Implements a union of independently measurable blocking channels:
- Channel A (bit 1): Exact normalized name (legal-stripped, punct-norm)
- Channel B (bit 2): Exact compact & domain-normalized name (compact alphanumeric, domain root)
- Channel C (bit 4): Document-frequency-aware rare token inverted index (min-DF brand tokens)
- Channel D (bit 8): Address anchors (building + rare token, building + street, postal + rare token)
- Channel E (bit 16): Transliteration-based lexical retrieval

Strictly enforces open-set country equality (US -> US, India -> India, France -> France).
Memory-conscious (bitmask representation) and scalable for multi-million candidate populations.
"""

import os
import sys
sys.path.insert(0, ".")
import re
import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional

from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive,
    LEGAL_TERMS
)

# Channel bitmasks
CH_A = 1   # Exact normalized name
CH_B = 2   # Compact / domain name
CH_C = 4   # Rare token
CH_D = 8   # Address anchors
CH_E = 16  # Transliteration


class MultiChannelBlocker:
    """
    Multi-channel candidate generator implementing Channels A, B, C, D, E.
    Uses integer indices and bitmasks for extreme memory efficiency.
    """
    def __init__(
        self,
        token_max_doc_freq: int = 3,
        token_min_len: int = 5,
        max_candidates_per_entity: Optional[int] = None
    ):
        self.token_max_doc_freq = token_max_doc_freq
        self.token_min_len = token_min_len
        self.max_candidates_per_entity = max_candidates_per_entity


        # Query index: (channel_bit, country, key_val) -> list of s1_int_indices
        self.query_index = defaultdict(list)
        self.s1_id_to_idx = {}
        self.idx_to_s1_id = []
        self.s1_records = []
        self.s1_token_df = defaultdict(int)

    def build_query_index(self, s1_records: List[Dict[str, Any]]):
        """
        Builds query inverted index from Source 1 entities (e.g., validation split).
        """
        print(f"Building query index for {len(s1_records):,} Source 1 entities...", flush=True)
        t0 = time.time()

        self.s1_records = s1_records
        self.idx_to_s1_id = [r["entity_id"] for r in s1_records]
        self.s1_id_to_idx = {r["entity_id"]: idx for idx, r in enumerate(s1_records)}

        # Step 1: Pre-normalize and compute token document frequencies
        normalized_data = []
        for r in s1_records:
            n = normalize_name_non_destructive(r.get("business_name", ""))
            a = normalize_address_non_destructive(r.get("business_address", ""))
            normalized_data.append((n, a, r.get("country", "")))

            seen_toks = set()
            for tok in n["distinctive_tokens"]:
                if len(tok) >= self.token_min_len and tok not in seen_toks and tok not in LEGAL_TERMS:
                    self.s1_token_df[tok] += 1
                    seen_toks.add(tok)

        eligible_rare_count = sum(1 for df in self.s1_token_df.values() if df <= self.token_max_doc_freq)
        print(f"  Distinctive tokens: {len(self.s1_token_df):,} (Eligible DF<={self.token_max_doc_freq}: {eligible_rare_count:,})", flush=True)

        # Step 2: Populate inverted index with bitmask channel keys
        ch_counts = {CH_A: 0, CH_B: 0, CH_C: 0, CH_D: 0, CH_E: 0}

        for s1_idx, (n, a, c) in enumerate(normalized_data):
            # Channel A: Exact Normalized Name (Bit 1)
            if n["legal_stripped"]:
                self.query_index[(CH_A, c, n["legal_stripped"])].append(s1_idx)
                ch_counts[CH_A] += 1
            if n["punct_norm"] and n["punct_norm"] != n["legal_stripped"]:
                self.query_index[(CH_A, c, n["punct_norm"])].append(s1_idx)
                ch_counts[CH_A] += 1

            # Channel B: Compact & Domain-Normalized Name (Bit 2)
            if len(n["compact_alnum"]) >= 4:
                self.query_index[(CH_B, c, n["compact_alnum"])].append(s1_idx)
                ch_counts[CH_B] += 1
            if len(n["legal_stripped_compact"]) >= 4 and n["legal_stripped_compact"] != n["compact_alnum"]:
                self.query_index[(CH_B, c, n["legal_stripped_compact"])].append(s1_idx)
                ch_counts[CH_B] += 1
            if n["domain_root"] and len(n["domain_root"]) >= 4:
                self.query_index[(CH_B, c, n["domain_root"])].append(s1_idx)
                ch_counts[CH_B] += 1

            # Channel C: Rare Token Inverted Index (Bit 4)
            # Pick the rarest 1-2 distinctive tokens for this entity with DF <= threshold
            cand_tokens = [t for t in n["distinctive_tokens"] if len(t) >= self.token_min_len and self.s1_token_df[t] <= self.token_max_doc_freq]
            if cand_tokens:
                cand_tokens.sort(key=lambda t: self.s1_token_df[t])
                for rt in cand_tokens[:2]:
                    self.query_index[(CH_C, c, rt)].append(s1_idx)
                    ch_counts[CH_C] += 1

            # Channel D: Address Anchors (Bit 8)
            bldg = a["building_numeric"]
            post = a["postal_code"]
            dist_toks = n["distinctive_tokens"]
            street_toks = a["street_tokens"]

            if bldg:
                for tok in dist_toks[:2]:
                    self.query_index[(CH_D, c, f"b_{bldg}_{tok}")].append(s1_idx)
                    ch_counts[CH_D] += 1
                if street_toks:
                    self.query_index[(CH_D, c, f"bs_{bldg}_{street_toks[0]}")].append(s1_idx)
                    ch_counts[CH_D] += 1
            if post:
                for tok in dist_toks[:2]:
                    self.query_index[(CH_D, c, f"p_{post}_{tok}")].append(s1_idx)
                    ch_counts[CH_D] += 1

            # Channel E: Transliteration-based lexical retrieval (Bit 16)
            if n["trans_stripped"] and n["trans_stripped"] != n["legal_stripped"]:
                self.query_index[(CH_E, c, n["trans_stripped"])].append(s1_idx)
                ch_counts[CH_E] += 1
            if len(n["trans_compact"]) >= 4 and n["trans_compact"] != n["compact_alnum"]:
                self.query_index[(CH_E, c, n["trans_compact"])].append(s1_idx)
                ch_counts[CH_E] += 1

        print(f"Query index constructed in {time.time()-t0:.2f}s across {len(self.query_index):,} keys:", flush=True)
        print(f"  Channel A (Exact):         {ch_counts[CH_A]:,} keys", flush=True)
        print(f"  Channel B (Compact/Domain): {ch_counts[CH_B]:,} keys", flush=True)
        print(f"  Channel C (Rare Tokens):   {ch_counts[CH_C]:,} keys", flush=True)
        print(f"  Channel D (Address Anchor): {ch_counts[CH_D]:,} keys", flush=True)
        print(f"  Channel E (Transliterate): {ch_counts[CH_E]:,} keys", flush=True)

    def generate_candidates_streaming(
        self,
        candidate_file_paths: List[str],
        progress_interval: int = 1000000
    ) -> List[Dict[str, int]]:
        """
        Streams through candidate source files (Source 2 and Source 3) and retrieves matches.
        
        Returns:
            candidates: list of length len(s1_records), where candidates[s1_idx] is dict:
                        cand_id -> bitmask (integer combining CH_A, CH_B, CH_C, CH_D, CH_E)
        """
        n_s1 = len(self.idx_to_s1_id)
        candidates = [{} for _ in range(n_s1)]
        total_scanned = 0
        t_start = time.time()

        for path in candidate_file_paths:
            print(f"\nStreaming candidate population from {path}...", flush=True)
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

                    # Channel A (Bit 1)
                    if n["legal_stripped"]:
                        for s1_idx in self.query_index.get((CH_A, c, n["legal_stripped"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_A
                    if n["punct_norm"] and n["punct_norm"] != n["legal_stripped"]:
                        for s1_idx in self.query_index.get((CH_A, c, n["punct_norm"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_A

                    # Channel B (Bit 2)
                    if len(n["compact_alnum"]) >= 4:
                        for s1_idx in self.query_index.get((CH_B, c, n["compact_alnum"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_B
                    if len(n["legal_stripped_compact"]) >= 4 and n["legal_stripped_compact"] != n["compact_alnum"]:
                        for s1_idx in self.query_index.get((CH_B, c, n["legal_stripped_compact"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_B
                    if n["domain_root"] and len(n["domain_root"]) >= 4:
                        for s1_idx in self.query_index.get((CH_B, c, n["domain_root"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_B

                    # Channel C (Bit 4): Check distinctive tokens len >= 4
                    for tok in n["distinctive_tokens"]:
                        if len(tok) >= self.token_min_len:
                            for s1_idx in self.query_index.get((CH_C, c, tok), []):
                                candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_C

                    # Channel D (Bit 8): Address Anchors (only if address non-empty)
                    if raw_addr and raw_addr.lower() not in {"", "nan", "<null>", "null", "none"}:
                        a = normalize_address_non_destructive(raw_addr)
                        bldg = a["building_numeric"]
                        post = a["postal_code"]
                        dist_toks = n["distinctive_tokens"]
                        street_toks = a["street_tokens"]

                        if bldg:
                            for tok in dist_toks[:2]:
                                for s1_idx in self.query_index.get((CH_D, c, f"b_{bldg}_{tok}"), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D
                            if street_toks:
                                for s1_idx in self.query_index.get((CH_D, c, f"bs_{bldg}_{street_toks[0]}"), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D
                        if post:
                            for tok in dist_toks[:2]:
                                for s1_idx in self.query_index.get((CH_D, c, f"p_{post}_{tok}"), []):
                                    candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_D

                    # Channel E (Bit 16): Transliteration
                    if n["trans_stripped"] and n["trans_stripped"] != n["legal_stripped"]:
                        for s1_idx in self.query_index.get((CH_E, c, n["trans_stripped"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_E
                    if len(n["trans_compact"]) >= 4 and n["trans_compact"] != n["compact_alnum"]:
                        for s1_idx in self.query_index.get((CH_E, c, n["trans_compact"]), []):
                            candidates[s1_idx][cand_id] = candidates[s1_idx].get(cand_id, 0) | CH_E

                    if rows_in_file % progress_interval == 0:
                        now = time.time()
                        rate = rows_in_file / (now - t_file)
                        print(f"  Scanned {rows_in_file:,} rows ({rate:.0f} rows/s)...", flush=True)

            print(f"Finished {path}: {rows_in_file:,} rows in {time.time()-t_file:.2f}s", flush=True)

        print(f"\nTotal candidate scanning complete: {total_scanned:,} rows in {time.time()-t_start:.2f}s", flush=True)
        return candidates
