#!/usr/bin/env python3
"""
Authoritative EXP_004 Cloud Candidate Generation Benchmark.

Unifies EXP_003 Multi-Channel Lexical Blocker (Channels A, B, C2, D2, E2, F)
with Scalable Character N-Gram TF-IDF Sparse Top-N Retrieval (Channels G_NAME, G_ADDR, G_TRANS)
partitioned strictly by Country x Candidate Source.

Evaluates against the competitive target: Oracle Entity-Level Macro F0.5 >= 0.995 (target > 0.988419).

Modes:
- 'smoke_test': Bounded local validation on small subset (e.g. 500 S1 queries, 20k candidates), labeled SMOKE_TEST_ONLY.
- 'pilot': Comprehensive single-partition profiling (e.g. India S2 or US S2) with the full development validation subset.
- 'full': Full AWS cloud execution across all partitions.
"""

import os
import sys
sys.path.insert(0, ".")
import gc
import time
import json
import yaml
import psutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional

from src.blocking.blocker import (
    MultiChannelBlocker,
    CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F,
    CH_G_NAME, CH_G_ADDR, CH_G_TRANS,
    CHANNEL_NAMES,
    StreamingCandidateResult
)
from src.blocking.tfidf_retriever import TfidfApproximateRetriever
from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive
)
from src.evaluation.macro_f05 import evaluate_predictions, compute_entity_f05


class MemoryTracker:
    """Continuous peak RAM tracking via psutil background thread."""
    def __init__(self, interval_sec: float = 0.5):
        import threading
        self.interval = interval_sec
        self.process = psutil.Process()
        self.initial_rss = self.process.memory_info().rss / (1024 * 1024)
        self.peak_rss = self.initial_rss
        self.final_rss = self.initial_rss
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        self._thread.start()

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                rss = self.process.memory_info().rss / (1024 * 1024)
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> Tuple[float, float, float]:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.final_rss = self.process.memory_info().rss / (1024 * 1024)
        if self.final_rss > self.peak_rss:
            self.peak_rss = self.final_rss
        return self.initial_rss, self.peak_rss, self.final_rss


def get_git_status() -> Tuple[str, bool]:
    """Returns (commit_sha, is_dirty) for reproducibility logging."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        is_dirty = len(status_out) > 0
        return sha, is_dirty
    except Exception:
        return "unknown", True


def compute_cardinality_breakdown(
    val_gt: Dict[str, Set[str]],
    oracle_preds: Dict[str, Set[str]],
    metadata_map: Dict[str, Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Computes blocking performance stratified by ground-truth match cardinality:
    0 (singletons), 1, 2, 3, 4, 5+ matches.
    """
    groups = {
        "0 matches (singletons)": [],
        "1 match": [],
        "2 matches": [],
        "3 matches": [],
        "4 matches": [],
        "5+ matches": []
    }

    for s1_id, truth in val_gt.items():
        k = len(truth)
        if k == 0:
            g = "0 matches (singletons)"
        elif k == 1:
            g = "1 match"
        elif k == 2:
            g = "2 matches"
        elif k == 3:
            g = "3 matches"
        elif k == 4:
            g = "4 matches"
        else:
            g = "5+ matches"
        groups[g].append(s1_id)

    breakdown = []
    for g_name, s1_ids in groups.items():
        if not s1_ids:
            continue
        g_gt = {sid: val_gt[sid] for sid in s1_ids}
        g_preds = {sid: oracle_preds[sid] for sid in s1_ids}
        g_meta = {sid: metadata_map.get(sid, {}) for sid in s1_ids}

        # Sub-evaluation
        sub_eval = evaluate_predictions(g_gt, g_preds, g_meta)

        total_true = sum(len(g_gt[sid]) for sid in s1_ids)
        total_hit = sum(len(g_gt[sid] & g_preds[sid]) for sid in s1_ids)
        link_rec = total_hit / total_true if total_true > 0 else 1.0

        non_singletons = [sid for sid in s1_ids if len(g_gt[sid]) > 0]
        full_cov_count = sum(1 for sid in non_singletons if g_gt[sid].issubset(g_preds[sid]))
        full_cov = full_cov_count / len(non_singletons) if non_singletons else 1.0

        breakdown.append({
            "cardinality_group": g_name,
            "entity_count": len(s1_ids),
            "true_links_count": total_true,
            "link_recall": round(link_rec, 6),
            "full_entity_coverage": round(full_cov, 6),
            "oracle_macro_f05": round(sub_eval.get("macro_f05", 0.0), 6),
            "macro_precision": round(sub_eval.get("macro_precision", 0.0), 6),
            "macro_recall": round(sub_eval.get("macro_recall", 0.0), 6)
        })

    return breakdown


def compute_zero_candidate_audit(
    val_gt: Dict[str, Set[str]],
    candidate_counts: np.ndarray,
    s1_id_list: List[str]
) -> Dict[str, Any]:
    """
    Audits zero-candidate populations across singletons and non-singletons.
    """
    total_entities = len(s1_id_list)
    zero_cand_s1_ids = [s1_id_list[i] for i in range(total_entities) if candidate_counts[i] == 0]
    non_singleton_zeros = [sid for sid in zero_cand_s1_ids if len(val_gt.get(sid, set())) > 0]
    singleton_zeros = [sid for sid in zero_cand_s1_ids if len(val_gt.get(sid, set())) == 0]
    singleton_with_cands = [
        s1_id_list[i] for i in range(total_entities)
        if len(val_gt.get(s1_id_list[i], set())) == 0 and candidate_counts[i] > 0
    ]

    return {
        "total_zero_candidate_entities": len(zero_cand_s1_ids),
        "non_singleton_zero_candidates": len(non_singleton_zeros),
        "singleton_zero_candidates": len(singleton_zeros),
        "singleton_with_candidates": len(singleton_with_cands),
        "non_singleton_zero_s1_ids_sample": non_singleton_zeros[:20]
    }


def run_cloud_benchmark(
    config_path: str = "configs/blocking/blocking_v04_cloud_tfidf.yaml",
    val_split_path: str = "artifacts/splits/val_s1_ids_50k_seed42.parquet",
    dataset_root: str = "dataset/train",
    output_dir: str = "reports/blocking",
    experiment_id: str = "EXP_004_cloud_ngram_tfidf_blocker",
    mode: str = "smoke_test",
    pilot_country: str = "India",
    pilot_source: str = "S2",
    smoke_s1_count: int = 500,
    smoke_cand_count: int = 20000,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Executes EXP_004 candidate generation benchmark.
    """
    mem_tracker = MemoryTracker(interval_sec=0.5)
    mem_tracker.start()
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    git_sha, git_dirty = get_git_status()
    is_smoke = (mode == "smoke_test")
    is_pilot = (mode == "pilot")
    tag_prefix = "SMOKE_TEST_ONLY" if is_smoke else ("PILOT_ONLY" if is_pilot else "FULL_EVALUATION")

    print(f"=== Starting Blocking Benchmark {experiment_id} [{tag_prefix}] ===", flush=True)
    print(f"Config:       {config_path}", flush=True)
    print(f"Mode:         {mode}", flush=True)
    print(f"Git Commit:   {git_sha} (dirty: {git_dirty})", flush=True)
    print(f"Initial RSS:  {mem_tracker.initial_rss:.2f} MB", flush=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    s1_path = os.path.join(dataset_root, "train_source1.tsv")
    s2_path = os.path.join(dataset_root, "train_source2.tsv")
    s3_path = os.path.join(dataset_root, "train_source3.tsv")
    gt_path = os.path.join(dataset_root, "train_ground_truth.tsv")

    # 1. Load validation S1 IDs
    val_meta_df = pd.read_parquet(val_split_path)
    all_val_s1_ids = list(val_meta_df["entity_id"])
    if is_smoke:
        selected_val_s1_ids = set(all_val_s1_ids[:smoke_s1_count])
        print(f"Smoke Test Mode: Bounding S1 queries to {len(selected_val_s1_ids):,} entities.", flush=True)
    else:
        selected_val_s1_ids = set(all_val_s1_ids)
        print(f"Loaded {len(selected_val_s1_ids):,} validation S1 IDs from {val_split_path}", flush=True)

    # 2. Extract S1 records
    val_s1_records = []
    val_s1_lookup = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_id = header.index("entity_id")
        col_name = header.index("business_name")
        col_addr = header.index("business_address")
        col_country = header.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[col_id] in selected_val_s1_ids:
                rec = {
                    "entity_id": parts[col_id],
                    "business_name": parts[col_name],
                    "business_address": parts[col_addr],
                    "country": parts[col_country]
                }
                val_s1_records.append(rec)
                val_s1_lookup[parts[col_id]] = rec

    print(f"Extracted {len(val_s1_records):,} S1 validation records.", flush=True)

    # 3. Load Ground Truth for selected S1 entities
    val_gt = {}
    val_gt_pairs = set()
    total_true_links = 0
    s2_true_links = 0
    s3_true_links = 0
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in selected_val_s1_ids:
                m_ids = set()
                if len(parts) >= 2 and parts[1].strip():
                    for x in parts[1].split(","):
                        cand = x.strip()
                        if cand:
                            m_ids.add(cand)
                            val_gt_pairs.add((s1_id, cand))
                            if cand.startswith("S2-"):
                                s2_true_links += 1
                            else:
                                s3_true_links += 1
                val_gt[s1_id] = m_ids
                total_true_links += len(m_ids)

    print(f"Ground Truth loaded: {len(val_gt):,} entities, {total_true_links:,} true links (S2: {s2_true_links:,}, S3: {s3_true_links:,})", flush=True)

    # 4. Read candidate records
    # For smoke test, read small subset from S2 and S3; for full, load by partitions
    candidate_partitions: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    cand_sources = [("S2", s2_path), ("S3", s3_path)]

    if is_pilot:
        cand_sources = [(pilot_source, s2_path if pilot_source == "S2" else s3_path)]

    for src_name, path in cand_sources:
        print(f"Loading candidate records from {path}...", flush=True)
        rows_read = 0
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            col_id = header.index("entity_id")
            col_name = header.index("business_name")
            col_addr = header.index("business_address")
            col_country = header.index("country")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                c = parts[col_country]
                if is_pilot and c != pilot_country:
                    continue
                candidate_partitions[(c, src_name)].append({
                    "entity_id": parts[col_id],
                    "business_name": parts[col_name],
                    "business_address": parts[col_addr],
                    "country": c
                })
                rows_read += 1
                if is_smoke and rows_read >= smoke_cand_count:
                    break

    total_cands = sum(len(cands) for cands in candidate_partitions.values())
    print(f"Candidate population loaded: {total_cands:,} records across {len(candidate_partitions)} partitions.", flush=True)

    # 5. Build Lexical Blocker & Query Index
    t_lex_start = time.time()
    blocker = MultiChannelBlocker(config=config)
    blocker.build_query_index(val_s1_records)

    val_gt_idx_pairs = {
        (blocker.s1_id_to_idx[s1_id], cid)
        for (s1_id, cid) in val_gt_pairs
        if s1_id in blocker.s1_id_to_idx
    }

    # Stream lexical candidates over loaded candidate partitions
    # To do so memory-efficiently, write candidate partitions to a temporary streaming format or iterate
    print("\nRunning EXP_003 lexical blocking baseline on loaded candidate records...", flush=True)
    lexical_matches: List[Tuple[int, str, int, float]] = []
    for (c, src_name), cands in candidate_partitions.items():
        for cand in cands:
            cand_id = cand["entity_id"]
            raw_name = cand["business_name"]
            raw_addr = cand["business_address"]

            n = normalize_name_non_destructive(raw_name)
            a = normalize_address_non_destructive(raw_addr)

            row_matches = {}
            # Channel A
            if blocker.ch_A_enabled:
                if n["norm_unicode"]:
                    for s1_idx in blocker.query_index.get((CH_A, c, n["norm_unicode"]), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_A
                if n["legal_stripped"]:
                    for s1_idx in blocker.query_index.get((CH_A, c, n["legal_stripped"]), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_A

            # Channel B
            if blocker.ch_B_enabled:
                if len(n["compact_alnum"]) >= 6:
                    for s1_idx in blocker.query_index.get((CH_B, c, n["compact_alnum"]), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_B
                if n["domain_root"] and len(n["domain_root"]) >= 4:
                    for s1_idx in blocker.query_index.get((CH_B, c, f"dom_{n['domain_root']}"), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_B

            # Channel C2
            if blocker.ch_C2_enabled:
                for tok in n["distinctive_tokens"]:
                    if len(tok) >= blocker.c2_min_len:
                        for s1_idx in blocker.query_index.get((CH_C2, c, f"tok_{tok}"), []):
                            row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_C2

            # Channel D2
            if blocker.ch_D2_enabled:
                if a["norm_unicode"]:
                    for s1_idx in blocker.query_index.get((CH_D2, c, a["norm_unicode"]), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_D2
                if a["building_numeric"] and a["street_tokens"]:
                    st = a["street_tokens"][0]
                    for s1_idx in blocker.query_index.get((CH_D2, c, f"bldg_{a['building_numeric']}_{st}"), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_D2

            # Channel E2
            if blocker.ch_E2_enabled:
                if n["trans_stripped"]:
                    for s1_idx in blocker.query_index.get((CH_E2, c, n["trans_stripped"]), []):
                        row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_E2
                for ttok in n["trans_distinctive_tokens"]:
                    if len(ttok) >= blocker.e2_min_len:
                        for s1_idx in blocker.query_index.get((CH_E2, c, f"ttok_{ttok}"), []):
                            row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_E2

            # Channel F
            if blocker.ch_F_enabled and len(n["distinctive_tokens"]) >= 2:
                dtoks = n["distinctive_tokens"][:4]
                for i in range(len(dtoks)):
                    for j in range(i + 1, len(dtoks)):
                        p_str = f"pair_{min(dtoks[i], dtoks[j])}_{max(dtoks[i], dtoks[j])}"
                        for s1_idx in blocker.query_index.get((CH_F, c, p_str), []):
                            row_matches[s1_idx] = row_matches.get(s1_idx, 0) | CH_F

            for s1_idx, mask in row_matches.items():
                lexical_matches.append((s1_idx, cand_id, mask, blocker.compute_candidate_evidence_score(mask)))

    n_s1 = len(val_s1_records)
    base_cand_dicts = [{} for _ in range(n_s1)]
    base_gt_hits = {}
    base_uncapped_counts = np.zeros(n_s1, dtype=np.int32)
    base_uncapped_channels = {
        b: np.zeros(n_s1, dtype=np.int32)
        for b in (CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F, CH_G_NAME, CH_G_ADDR, CH_G_TRANS)
    }

    for s1_idx, cand_id, mask, score in lexical_matches:
        base_uncapped_counts[s1_idx] += 1
        for b in (CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F):
            if mask & b:
                base_uncapped_channels[b][s1_idx] += 1
        if cand_id in base_cand_dicts[s1_idx]:
            base_cand_dicts[s1_idx][cand_id] |= mask
        else:
            base_cand_dicts[s1_idx][cand_id] = mask
        if (s1_idx, cand_id) in val_gt_idx_pairs:
            base_gt_hits[(s1_idx, cand_id)] = base_gt_hits.get((s1_idx, cand_id), 0) | mask

    lexical_result = StreamingCandidateResult(
        candidate_dicts=base_cand_dicts,
        gt_hits=base_gt_hits,
        uncapped_counts=base_uncapped_counts,
        uncapped_channel_counts=base_uncapped_channels,
        total_scanned=total_cands
    )

    lex_hit_pairs = set(lexical_result.gt_hits.keys())
    print(f"EXP_003 Lexical Baseline on subset: {len(lex_hit_pairs):,} / {len(val_gt_idx_pairs):,} true links hit ({len(lex_hit_pairs)/max(1, len(val_gt_idx_pairs)):.2%}).", flush=True)

    # 6. Initialize EXP_004 Approximate Retriever
    retriever = TfidfApproximateRetriever(config=config)
    tfidf_matches_by_channel = {
        CH_G_NAME: [],
        CH_G_ADDR: [],
        CH_G_TRANS: []
    }

    # Group S1 queries by country
    s1_by_country: Dict[str, Tuple[List[Dict[str, Any]], List[int]]] = defaultdict(lambda: ([], []))
    for s1_idx, rec in enumerate(val_s1_records):
        s1_by_country[rec["country"]][0].append(rec)
        s1_by_country[rec["country"]][1].append(s1_idx)

    # Execute partitioned TF-IDF retrieval
    for (country, src_name), cands in candidate_partitions.items():
        s1_recs, s1_idxs = s1_by_country[country]
        if not s1_recs or not cands:
            continue

        # Channel G_NAME
        if retriever.name_cfg.get("enabled", True):
            m = retriever.retrieve_partition_approximate_candidates(
                country=country,
                source=src_name,
                channel_bit=CH_G_NAME,
                channel_cfg=retriever.name_cfg,
                channel_type="name",
                s1_entities=s1_recs,
                s1_indices=s1_idxs,
                candidate_records=cands,
                use_cache=use_cache
            )
            tfidf_matches_by_channel[CH_G_NAME].extend(m)

        # Channel G_ADDR
        if retriever.addr_cfg.get("enabled", True):
            m = retriever.retrieve_partition_approximate_candidates(
                country=country,
                source=src_name,
                channel_bit=CH_G_ADDR,
                channel_cfg=retriever.addr_cfg,
                channel_type="address",
                s1_entities=s1_recs,
                s1_indices=s1_idxs,
                candidate_records=cands,
                use_cache=use_cache
            )
            tfidf_matches_by_channel[CH_G_ADDR].extend(m)

        # Channel G_TRANS
        if retriever.trans_cfg.get("enabled", True):
            m = retriever.retrieve_partition_approximate_candidates(
                country=country,
                source=src_name,
                channel_bit=CH_G_TRANS,
                channel_cfg=retriever.trans_cfg,
                channel_type="transliterated",
                s1_entities=s1_recs,
                s1_indices=s1_idxs,
                candidate_records=cands,
                use_cache=use_cache
            )
            tfidf_matches_by_channel[CH_G_TRANS].extend(m)

    all_tfidf_matches = (
        tfidf_matches_by_channel[CH_G_NAME] +
        tfidf_matches_by_channel[CH_G_ADDR] +
        tfidf_matches_by_channel[CH_G_TRANS]
    )

    # 7. Merge candidates into final UNION
    final_result = retriever.merge_tfidf_candidates_into_result(
        base_result=lexical_result,
        tfidf_matches=all_tfidf_matches,
        gt_links_set=val_gt_idx_pairs
    )

    # 8. Compute Authoritative Macro F0.5 & Primary Metrics
    t_eval_start = time.time()
    union_hit_pairs = set(final_result.gt_hits.keys())
    total_union_hits = len(union_hit_pairs)
    overall_link_recall = total_union_hits / total_true_links if total_true_links > 0 else 0.0

    metadata_map = {rec["entity_id"]: {"country": rec["country"]} for rec in val_s1_records}
    oracle_preds = {}
    full_cov_count = 0
    non_singleton_count = 0

    for s1_idx, rec in enumerate(val_s1_records):
        s1_id = rec["entity_id"]
        truth_set = val_gt.get(s1_id, set())
        if len(truth_set) > 0:
            non_singleton_count += 1

        cand_set = set(final_result[s1_idx].keys())
        hit_set = truth_set & cand_set
        oracle_preds[s1_id] = hit_set

        if len(truth_set) > 0 and truth_set.issubset(cand_set):
            full_cov_count += 1

    full_coverage_rate = full_cov_count / non_singleton_count if non_singleton_count > 0 else 1.0
    oracle_eval = evaluate_predictions(val_gt, oracle_preds, metadata_map)
    oracle_macro_f05 = oracle_eval.get("macro_f05", 0.0)

    # 9. Incremental True-Link Analysis & Miss Recovery
    # EXP_003 lexical misses in this subset
    lexical_misses = val_gt_idx_pairs - lex_hit_pairs
    total_lex_misses = len(lexical_misses)

    def analyze_channel_recovery(matches_list, ch_bit: int):
        ch_pairs = {(m[0], m[1]) for m in matches_list}
        ch_hits = ch_pairs & val_gt_idx_pairs
        recovered_misses = ch_hits & lexical_misses
        new_cands = len(ch_pairs)
        ratio = new_cands / max(1, len(recovered_misses))
        return {
            "standalone_hits": len(ch_hits),
            "standalone_recall": len(ch_hits) / max(1, total_true_links),
            "misses_recovered": len(recovered_misses),
            "pct_misses_recovered": len(recovered_misses) / max(1, total_lex_misses),
            "new_candidate_pairs": new_cands,
            "cands_per_recovered_link": round(ratio, 2)
        }

    incremental_table = [
        {
            "channel": "EXP_003_lexical",
            "hits": len(lex_hit_pairs),
            "recall": len(lex_hit_pairs) / max(1, total_true_links),
            "misses_recovered": 0,
            "pct_misses_recovered": 0.0,
            "new_candidate_pairs": sum(len(d) for d in base_cand_dicts),
            "cands_per_recovered_link": "-"
        }
    ]

    for ch_bit, name in (
        (CH_G_NAME, "Name_Char_TFIDF"),
        (CH_G_ADDR, "Address_Char_TFIDF"),
        (CH_G_TRANS, "Transliterated_Char_TFIDF")
    ):
        rec_data = analyze_channel_recovery(tfidf_matches_by_channel[ch_bit], ch_bit)
        incremental_table.append({
            "channel": name,
            "hits": rec_data["standalone_hits"],
            "recall": rec_data["standalone_recall"],
            "misses_recovered": rec_data["misses_recovered"],
            "pct_misses_recovered": rec_data["pct_misses_recovered"],
            "new_candidate_pairs": rec_data["new_candidate_pairs"],
            "cands_per_recovered_link": rec_data["cands_per_recovered_link"]
        })

    # Union row
    total_recovered_in_union = len(union_hit_pairs & lexical_misses)
    incremental_table.append({
        "channel": "FINAL_UNION",
        "hits": total_union_hits,
        "recall": overall_link_recall,
        "misses_recovered": total_recovered_in_union,
        "pct_misses_recovered": total_recovered_in_union / max(1, total_lex_misses),
        "new_candidate_pairs": sum(len(d) for d in final_result),
        "cands_per_recovered_link": round(sum(len(d) for d in final_result) / max(1, total_recovered_in_union), 2)
    })

    # 10. Entity Cardinality Breakdown
    cardinality_breakdown = compute_cardinality_breakdown(val_gt, oracle_preds, metadata_map)

    # 11. Zero-Candidate Audit
    cand_counts_arr = np.array([len(d) for d in final_result], dtype=np.int32)
    zero_cand_audit = compute_zero_candidate_audit(val_gt, cand_counts_arr, [r["entity_id"] for r in val_s1_records])

    # Candidate volume percentiles
    cand_stats = {
        "mean": float(np.mean(cand_counts_arr)),
        "median": float(np.median(cand_counts_arr)),
        "p90": float(np.percentile(cand_counts_arr, 90)),
        "p95": float(np.percentile(cand_counts_arr, 95)),
        "p99": float(np.percentile(cand_counts_arr, 99)),
        "max": int(np.max(cand_counts_arr)),
        "total_pairs": int(np.sum(cand_counts_arr))
    }

    init_rss, peak_rss, final_rss = mem_tracker.stop()
    total_runtime = time.time() - t_start

    # Success gate evaluation
    gate_cfg = config.get("success_gate", {})
    target_f05 = gate_cfg.get("min_oracle_macro_f05", 0.995)
    gate_passed = bool(oracle_macro_f05 >= target_f05)

    summary = {
        "experiment_id": experiment_id,
        "mode": mode,
        "tag": tag_prefix,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "s1_population_evaluated": len(val_s1_records),
        "total_true_links": total_true_links,
        "primary_metrics": {
            "oracle_macro_f05": round(oracle_macro_f05, 6),
            "true_link_recall": round(overall_link_recall, 6),
            "full_entity_coverage": round(full_coverage_rate, 6),
            "macro_precision": round(oracle_eval.get("macro_precision", 0.0), 6),
            "macro_recall": round(oracle_eval.get("macro_recall", 0.0), 6)
        },
        "success_gate": {
            "target_oracle_macro_f05": target_f05,
            "achieved_oracle_macro_f05": round(oracle_macro_f05, 6),
            "passed": gate_passed
        },
        "candidate_distribution": cand_stats,
        "incremental_analysis": incremental_table,
        "cardinality_breakdown": cardinality_breakdown,
        "zero_candidate_audit": zero_cand_audit,
        "system_resources": {
            "initial_rss_mb": round(init_rss, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "final_rss_mb": round(final_rss, 2),
            "total_runtime_sec": round(total_runtime, 2)
        }
    }

    # Save summary JSON
    summary_path = os.path.join(output_dir, f"{experiment_id}_{mode}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    report_path = os.path.join(output_dir, f"{experiment_id}_{mode}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Blocking Benchmark Report: {experiment_id} [{tag_prefix}]\n\n")
        f.write(f"- **Mode**: `{mode}`\n")
        f.write(f"- **Git Commit**: `{git_sha}` (dirty: `{git_dirty}`)\n")
        f.write(f"- **Evaluated S1 Entities**: `{len(val_s1_records):,}`\n")
        f.write(f"- **Evaluated Candidate Records**: `{total_cands:,}`\n")
        f.write(f"- **Total True Links**: `{total_true_links:,}`\n\n")

        f.write("## 1. Primary Blocking Ceiling Metrics\n\n")
        f.write("| Metric | Score | Target Gate |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Oracle Macro F0.5** | **{oracle_macro_f05:.6f}** | >= {target_f05:.4f} |\n")
        f.write(f"| True-Link Recall | {overall_link_recall:.4%} | >= 95.0% |\n")
        f.write(f"| Full-Entity Coverage | {full_coverage_rate:.4%} | High |\n")
        f.write(f"| Total Candidate Pairs | {cand_stats['total_pairs']:,} | Manageable |\n")
        f.write(f"| Mean Candidates / S1 | {cand_stats['mean']:.1f} | - |\n")
        f.write(f"| Peak RAM (RSS) | {peak_rss:.1f} MB | <= {config.get('execution', {}).get('memory_safety', {}).get('max_ram_gb', 55)*1024:.0f} MB |\n")
        f.write(f"| Runtime | {total_runtime:.2f}s | Scalable |\n\n")

        f.write("## 2. Incremental Miss Recovery Analysis\n\n")
        f.write("| Channel | Standalone Recall | EXP_003 Misses Recovered | % Misses Recovered | Candidate Pairs | Cands / Recovered Link |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in incremental_table:
            f.write(f"| `{row['channel']}` | {row['recall']:.2%} | {row['misses_recovered']:,} | {row['pct_misses_recovered']:.2%} | {row['new_candidate_pairs']:,} | {row['cands_per_recovered_link']} |\n")
        f.write("\n")

        f.write("## 3. Entity Cardinality Performance Breakdown\n\n")
        f.write("| Ground-Truth Match Cardinality | Entity Count | True Links | Link Recall | Full Coverage | Oracle Macro F0.5 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in cardinality_breakdown:
            f.write(f"| **{row['cardinality_group']}** | {row['entity_count']:,} | {row['true_links_count']:,} | {row['link_recall']:.2%} | {row['full_entity_coverage']:.2%} | {row['oracle_macro_f05']:.6f} |\n")
        f.write("\n")

        f.write("## 4. Zero-Candidate Entities Audit\n\n")
        f.write(f"- Total Zero-Candidate Entities: `{zero_cand_audit['total_zero_candidate_entities']:,}`\n")
        f.write(f"- Non-Singleton Zero-Candidate Entities (Severe Failures): `{zero_cand_audit['non_singleton_zero_candidates']:,}`\n")
        f.write(f"- Singleton Zero-Candidate Entities (Correct Empties): `{zero_cand_audit['singleton_zero_candidates']:,}`\n")
        f.write(f"- Singletons Receiving Candidates (False Positives): `{zero_cand_audit['singleton_with_candidates']:,}`\n\n")

    print(f"\nBenchmark completed successfully in {total_runtime:.2f}s.", flush=True)
    print(f"Summary JSON saved: {summary_path}", flush=True)
    print(f"Report Markdown saved: {report_path}", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP_004 Cloud Blocking Benchmark")
    parser.add_argument("--config", default="configs/blocking/blocking_v04_cloud_tfidf.yaml", help="Path to YAML config")
    parser.add_argument("--dataset-root", default="dataset/train", help="Dataset directory")
    parser.add_argument("--val-split-path", default="artifacts/splits/val_s1_ids_50k_seed42.parquet", help="Path to validation split parquet")
    parser.add_argument("--output-dir", default="reports/blocking", help="Output directory")
    parser.add_argument("--experiment-id", default="EXP_004_cloud_ngram_tfidf_blocker", help="Experiment ID")
    parser.add_argument("--mode", choices=["smoke_test", "pilot", "full"], default="smoke_test", help="Execution mode")
    parser.add_argument("--pilot-country", default="India", help="Country partition for pilot mode")
    parser.add_argument("--pilot-source", default="S2", help="Candidate source for pilot mode")
    parser.add_argument("--smoke-s1-count", type=int, default=500, help="S1 queries for smoke test")
    parser.add_argument("--smoke-cand-count", type=int, default=20000, help="Candidate rows per file for smoke test")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    run_cloud_benchmark(
        config_path=args.config,
        val_split_path=args.val_split_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        mode=args.mode,
        pilot_country=args.pilot_country,
        pilot_source=args.pilot_source,
        smoke_s1_count=args.smoke_s1_count,
        smoke_cand_count=args.smoke_cand_count,
        use_cache=not args.no_cache
    )
