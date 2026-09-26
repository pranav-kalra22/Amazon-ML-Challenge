#!/usr/bin/env python3
"""
Authoritative EXP_004 Cloud Candidate Generation Benchmark.

Unifies EXP_003 Multi-Channel Lexical Blocker (Channels A, B, C2, D2, E2, F)
with Scalable Character N-Gram TF-IDF Sparse Top-N Retrieval (Channels G_NAME, G_ADDR, G_TRANS)
partitioned strictly by Country x Candidate Source.

Evaluates against the competitive target: Oracle Entity-Level Macro F0.5 >= 0.995 (target > 0.988419).

Execution Modes:
- 'parity_check': Lexical regression gate reproducing EXP_003 Mode A baseline on val_50k_seed42 (TF-IDF disabled).
- 'plumbing_smoke': Bounded local verification on arbitrary candidate subset (labeled SMOKE_TEST_ONLY).
- 'positive_smoke': Ground-truth-inclusive smoke test with all true matches present (labeled SMOKE_TEST_ONLY).
- 'pilot': Bounded single-partition profiling (e.g. India x S2) reporting partition-only metrics (PILOT_PARTITION_MEASUREMENT).
- 'full': Full cloud execution across all partitions with partition-at-a-time memory release.
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
    CandidateHeapItem,
    StreamingCandidateResult
)
from src.blocking.tfidf_retriever import (
    TfidfApproximateRetriever,
    evaluate_parameter_sweep_in_memory
)
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
    mode: str = "plumbing_smoke",
    pilot_country: str = "India",
    pilot_source: str = "S2",
    smoke_s1_count: int = 500,
    smoke_cand_count: int = 20000,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Executes EXP_004 Candidate Generation Benchmark.
    """
    mem_tracker = MemoryTracker(interval_sec=0.5)
    mem_tracker.start()
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    git_sha, git_dirty = get_git_status()
    is_smoke = mode in ("plumbing_smoke", "positive_smoke", "smoke_test")
    is_pilot = (mode == "pilot")
    is_parity = (mode == "parity_check")

    if is_smoke:
        tag_prefix = "SMOKE_TEST_ONLY"
    elif is_pilot:
        tag_prefix = "PILOT_PARTITION_MEASUREMENT"
    elif is_parity:
        tag_prefix = "LEXICAL_PARITY_GATE"
    else:
        tag_prefix = "FULL_EVALUATION"

    print(f"=== Starting Blocking Benchmark {experiment_id} [{tag_prefix}] ===", flush=True)
    print(f"Config:       {config_path}", flush=True)
    print(f"Mode:         {mode}", flush=True)
    print(f"Git Commit:   {git_sha} (dirty: {git_dirty})", flush=True)
    print(f"Initial RSS:  {mem_tracker.initial_rss:.2f} MB", flush=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # If parity_check mode, disable TF-IDF channels unconditionally
    if is_parity:
        config.setdefault("tfidf_retrieval", {})["enabled"] = False
        print("[PARITY CHECK MODE] TF-IDF channels explicitly disabled. Verifying EXP_003 lexical baseline.", flush=True)

    s1_path = os.path.join(dataset_root, "train_source1.tsv")
    s2_path = os.path.join(dataset_root, "train_source2.tsv")
    s3_path = os.path.join(dataset_root, "train_source3.tsv")
    gt_path = os.path.join(dataset_root, "train_ground_truth.tsv")

    # 1. Load validation S1 IDs
    val_meta_df = pd.read_parquet(val_split_path)
    all_val_s1_ids = list(val_meta_df["entity_id"])

    if mode == "positive_smoke":
        # Positive-inclusive smoke test: 150 S1 entities
        smoke_count = min(150, len(all_val_s1_ids))
        selected_val_s1_ids = set(all_val_s1_ids[:smoke_count])
        print(f"Positive Smoke Mode: Evaluating {len(selected_val_s1_ids):,} entities with 100% of their true links included in corpus.", flush=True)
    elif mode in ("plumbing_smoke", "smoke_test"):
        selected_val_s1_ids = set(all_val_s1_ids[:smoke_s1_count])
        print(f"Plumbing Smoke Mode: Evaluating {len(selected_val_s1_ids):,} entities on bounded candidate subset.", flush=True)
    elif is_pilot:
        selected_val_s1_ids = set(all_val_s1_ids)
        print(f"Pilot Mode: Partition '{pilot_country} x {pilot_source}' across validation population.", flush=True)
    else:
        selected_val_s1_ids = set(all_val_s1_ids)
        print(f"Full Mode: Evaluating complete {len(selected_val_s1_ids):,} validation S1 entities.", flush=True)

    # 2. Extract S1 validation records
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
                c = parts[col_country]
                if is_pilot and c != pilot_country:
                    continue
                rec = {
                    "entity_id": parts[col_id],
                    "business_name": parts[col_name],
                    "business_address": parts[col_addr],
                    "country": c
                }
                val_s1_records.append(rec)
                val_s1_lookup[parts[col_id]] = rec

    print(f"Extracted {len(val_s1_records):,} S1 validation records (country filtered: {is_pilot}).", flush=True)

    # 3. Load Ground Truth for selected S1 entities
    val_gt: Dict[str, Set[str]] = {}
    val_gt_pairs: Set[Tuple[str, str]] = set()
    total_true_links = 0
    s2_true_links = 0
    s3_true_links = 0

    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_lookup:
                m_ids = set()
                if len(parts) >= 2 and parts[1].strip():
                    for x in parts[1].split(","):
                        cand = x.strip()
                        if cand:
                            # In pilot mode, strictly filter true links to requested candidate source
                            if is_pilot and not cand.startswith(f"{pilot_source}-"):
                                continue
                            m_ids.add(cand)
                            val_gt_pairs.add((s1_id, cand))
                            if cand.startswith("S2-"):
                                s2_true_links += 1
                            else:
                                s3_true_links += 1
                val_gt[s1_id] = m_ids
                total_true_links += len(m_ids)

    # Target true candidate IDs for positive-inclusive smoke test
    target_true_cand_ids = {cid for (_, cid) in val_gt_pairs} if mode == "positive_smoke" else set()

    print(f"Ground Truth loaded: {len(val_gt):,} entities, {total_true_links:,} true links (S2: {s2_true_links:,}, S3: {s3_true_links:,})", flush=True)

    # 4. Canonical Lexical Blocker: Pass 1 Candidate-DF Scanning & Query Index
    t_idx_start = time.time()
    blocker = MultiChannelBlocker(config=config)
    blocker.build_query_index(
        val_s1_records,
        candidate_file_paths=[s2_path, s3_path]
    )
    print(f"Authoritative query index built in {time.time()-t_idx_start:.2f}s. Peak RAM: {mem_tracker.peak_rss:.2f} MB", flush=True)

    val_gt_idx_pairs = {
        (blocker.s1_id_to_idx[s1_id], cid)
        for (s1_id, cid) in val_gt_pairs
        if s1_id in blocker.s1_id_to_idx
    }

    # 5. Initialize Candidate Structures
    n_s1 = len(val_s1_records)
    candidate_dicts: List[Dict[str, int]] = [{} for _ in range(n_s1)]
    similarities_list: List[Dict[str, Dict[str, float]]] = [{} for _ in range(n_s1)]
    gt_hits: Dict[Tuple[int, str], int] = {}
    uncapped_counts = np.zeros(n_s1, dtype=np.int32)
    uncapped_channel_counts = {
        b: np.zeros(n_s1, dtype=np.int32) for b in CHANNEL_NAMES
    }
    heaps: List[List[CandidateHeapItem]] = [[] for _ in range(n_s1)]
    max_k = config.get("constraints", {}).get("max_heap_k", 2000)

    # EXP_003 Lexical baseline hit set
    lexical_hit_pairs: Set[Tuple[int, str]] = set()

    # Determine partitions to execute
    all_countries = sorted(list({rec["country"] for rec in val_s1_records}))
    candidate_sources = [("S2", s2_path), ("S3", s3_path)]

    if is_pilot:
        all_countries = [pilot_country]
        candidate_sources = [(pilot_source, s2_path if pilot_source == "S2" else s3_path)]

    total_scanned_rows = 0
    all_tfidf_matches_with_similarity: List[Tuple[int, str, int, float]] = []

    retriever = TfidfApproximateRetriever(config=config)
    tfidf_active = retriever.tfidf_enabled and not is_parity

    # Group S1 records by country for rapid retrieval indexing
    s1_by_country: Dict[str, Tuple[List[Dict[str, Any]], List[int]]] = defaultdict(lambda: ([], []))
    for s1_idx, rec in enumerate(val_s1_records):
        s1_by_country[rec["country"]][0].append(rec)
        s1_by_country[rec["country"]][1].append(s1_idx)

    # =========================================================================
    # PARTITION-AT-A-TIME EXECUTION ARCHITECTURE (Item 5)
    # =========================================================================
    for country in all_countries:
        s1_country_recs, s1_country_idxs = s1_by_country[country]
        if not s1_country_recs:
            continue

        for src_name, src_path in candidate_sources:
            t_part_start = time.time()
            print(f"\n>>> Processing Partition: Country='{country}' x Source='{src_name}' from {src_path}...", flush=True)

            partition_candidate_records: List[Dict[str, Any]] = []
            rows_scanned_in_partition = 0

            with open(src_path, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\n").split("\t")
                col_id = header.index("entity_id") if "entity_id" in header else 0
                col_name = header.index("business_name") if "business_name" in header else 1
                col_addr = header.index("business_address") if "business_address" in header else 2
                col_country = header.index("country") if "country" in header else 3

                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= col_country:
                        continue
                    row_c = parts[col_country]

                    cand_id = parts[col_id]
                    raw_name = parts[col_name]
                    raw_addr = parts[col_addr]

                    # Filter for this partition
                    if row_c != country:
                        # In positive_smoke mode, allow true matches even if country has noise
                        if not (mode == "positive_smoke" and cand_id in target_true_cand_ids):
                            continue

                    # Filtering for smoke tests
                    if mode == "plumbing_smoke":
                        if rows_scanned_in_partition >= smoke_cand_count:
                            break
                    elif mode == "positive_smoke":
                        is_target = cand_id in target_true_cand_ids
                        if not is_target and rows_scanned_in_partition >= 5000:
                            continue

                    rows_scanned_in_partition += 1
                    total_scanned_rows += 1

                    # 1. Authoritative Lexical Matching (Single Canonical Implementation)
                    row_s1_matches = blocker.match_candidate_record(country, raw_name, raw_addr)
                    if row_s1_matches:
                        for s1_idx, bitmask in row_s1_matches.items():
                            uncapped_counts[s1_idx] += 1
                            for b in CHANNEL_NAMES:
                                if bitmask & b:
                                    uncapped_channel_counts[b][s1_idx] += 1

                            if cand_id in candidate_dicts[s1_idx]:
                                candidate_dicts[s1_idx][cand_id] |= bitmask
                            else:
                                candidate_dicts[s1_idx][cand_id] = bitmask

                            if (s1_idx, cand_id) in val_gt_idx_pairs:
                                gt_hits[(s1_idx, cand_id)] = gt_hits.get((s1_idx, cand_id), 0) | bitmask
                                lexical_hit_pairs.add((s1_idx, cand_id))

                            # Evidence-ranked heap
                            score = blocker.compute_candidate_evidence_score(bitmask)
                            item = CandidateHeapItem(score, cand_id, bitmask)
                            h = heaps[s1_idx]
                            if len(h) < max_k:
                                heapq.heappush(h, item)
                            elif item > h[0]:
                                heapq.heapreplace(h, item)

                    # Collect candidate record for TF-IDF if enabled
                    if tfidf_active:
                        partition_candidate_records.append({
                            "entity_id": cand_id,
                            "business_name": raw_name,
                            "business_address": raw_addr,
                            "country": country
                        })

            print(f"  Partition scanned {rows_scanned_in_partition:,} records. Lexical hits so far: {len(lexical_hit_pairs):,} / {len(val_gt_idx_pairs):,}", flush=True)

            # 2. Approximate TF-IDF Retrieval on Partition
            if tfidf_active and partition_candidate_records:
                for ch_bit, ch_cfg, ch_type in (
                    (CH_G_NAME, retriever.name_cfg, "name"),
                    (CH_G_ADDR, retriever.addr_cfg, "address"),
                    (CH_G_TRANS, retriever.trans_cfg, "transliterated")
                ):
                    if ch_cfg.get("enabled", True):
                        m = retriever.retrieve_partition_approximate_candidates(
                            country=country,
                            source=src_name,
                            channel_bit=ch_bit,
                            channel_cfg=ch_cfg,
                            channel_type=ch_type,
                            s1_entities=s1_country_recs,
                            s1_indices=s1_country_idxs,
                            candidate_records=partition_candidate_records,
                            use_cache=use_cache
                        )
                        all_tfidf_matches_with_similarity.extend(m)

                        # Merge into candidate structures immediately
                        for s1_idx, cand_id, bit, sim_sc in m:
                            sim_map = similarities_list[s1_idx].setdefault(cand_id, {})
                            if bit == CH_G_NAME:
                                sim_map["name"] = max(sim_map.get("name", 0.0), float(sim_sc))
                            elif bit == CH_G_ADDR:
                                sim_map["address"] = max(sim_map.get("address", 0.0), float(sim_sc))
                            elif bit == CH_G_TRANS:
                                sim_map["transliterated"] = max(sim_map.get("transliterated", 0.0), float(sim_sc))

                            if cand_id in candidate_dicts[s1_idx]:
                                candidate_dicts[s1_idx][cand_id] |= bit
                            else:
                                candidate_dicts[s1_idx][cand_id] = bit

                            uncapped_counts[s1_idx] += 1
                            uncapped_channel_counts[bit][s1_idx] += 1

                            if (s1_idx, cand_id) in val_gt_idx_pairs:
                                gt_hits[(s1_idx, cand_id)] = gt_hits.get((s1_idx, cand_id), 0) | bit

                            # Update evidence score in heap with continuous similarity
                            bitmask = candidate_dicts[s1_idx][cand_id]
                            score = blocker.compute_candidate_evidence_score(bitmask, similarities=sim_map)
                            item = CandidateHeapItem(score, cand_id, bitmask, similarities=dict(sim_map))
                            h = heaps[s1_idx]
                            found = False
                            for h_elem in h:
                                if h_elem.cand_id == cand_id:
                                    h_elem.score = score
                                    h_elem.bitmask = bitmask
                                    h_elem.similarities = dict(sim_map)
                                    found = True
                                    break
                            if found:
                                heapq.heapify(h)
                            else:
                                if len(h) < max_k:
                                    heapq.heappush(h, item)
                                elif item > h[0]:
                                    heapq.heapreplace(h, item)

            # Release partition candidate records to keep memory strictly bounded
            del partition_candidate_records
            gc.collect()
            print(f"  Partition complete in {time.time()-t_part_start:.2f}s. Current peak RSS: {mem_tracker.peak_rss:.2f} MB", flush=True)

    # Assemble StreamingCandidateResult
    final_result = StreamingCandidateResult(
        candidate_dicts=candidate_dicts,
        heaps=heaps,
        gt_hits=gt_hits,
        uncapped_counts=uncapped_counts,
        uncapped_channel_counts=uncapped_channel_counts,
        total_scanned=total_scanned_rows,
        similarities=similarities_list
    )

    total_union_hits = len(final_result.gt_hits)
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

    # Incremental miss recovery
    total_lex_misses = len(val_gt_idx_pairs - lexical_hit_pairs)
    total_recovered = len(set(final_result.gt_hits.keys()) - lexical_hit_pairs)

    # Sweeps (Item 8)
    sweep_table = []
    if tfidf_active and all_tfidf_matches_with_similarity:
        k_sweep = config.get("sweeps", {}).get("top_k_sweep", [10, 25, 50, 100])
        thresh_sweep = config.get("sweeps", {}).get("threshold_sweep", [0.30, 0.35, 0.40, 0.50, 0.60])
        base_lex_count = sum(len(d) for d in candidate_dicts) - len(all_tfidf_matches_with_similarity)

        sweep_table = evaluate_parameter_sweep_in_memory(
            matches=all_tfidf_matches_with_similarity,
            val_gt_pairs=val_gt_idx_pairs,
            lexical_hits=lexical_hit_pairs,
            total_true_links=total_true_links,
            k_list=k_sweep,
            thresh_list=thresh_sweep,
            val_gt=val_gt if not is_pilot else None,
            s1_id_list=[r["entity_id"] for r in val_s1_records] if not is_pilot else None,
            metadata_map=metadata_map if not is_pilot else None,
            base_lexical_pairs=max(0, base_lex_count)
        )

    # Cardinality & Zero-Candidate Audits
    cardinality_breakdown = compute_cardinality_breakdown(val_gt, oracle_preds, metadata_map)
    cand_counts_arr = np.array([len(d) for d in final_result], dtype=np.int32)
    zero_cand_audit = compute_zero_candidate_audit(val_gt, cand_counts_arr, [r["entity_id"] for r in val_s1_records])

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

    # Lexical Parity Verification against EXP_003
    parity_passed = False
    if is_parity:
        # Expected EXP_003 reference:
        # True links: 112,143 | Link recall: 0.649242 | Full cov: 0.304864 | Oracle F0.5: 0.831634
        exp003_links = 112143
        exp003_recall = 0.649242
        exp003_cov = 0.304864
        exp003_f05 = 0.831634

        parity_passed = (
            abs(total_union_hits - exp003_links) == 0 and
            abs(overall_link_recall - exp003_recall) < 1e-4 and
            abs(full_coverage_rate - exp003_cov) < 1e-4 and
            abs(oracle_macro_f05 - exp003_f05) < 1e-4
        )
        print(f"\n[LEXICAL PARITY VERDICT] True links: {total_union_hits:,} / {exp003_links:,} | Recall: {overall_link_recall:.6f} / {exp003_recall:.6f} | Coverage: {full_coverage_rate:.6f} / {exp003_cov:.6f} | Oracle F0.5: {oracle_macro_f05:.6f} / {exp003_f05:.6f}")
        print(f"Parity Gate Result: {'PASS' if parity_passed else 'FAIL'}", flush=True)

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
            "macro_recall": round(oracle_eval.get("macro_recall", 0.0), 6),
            "true_links_retrieved": total_union_hits,
            "lexical_baseline_hits": len(lexical_hit_pairs),
            "misses_recovered": total_recovered,
            "pct_misses_recovered": round(total_recovered / max(1, total_lex_misses), 6)
        },
        "candidate_distribution": cand_stats,
        "cardinality_breakdown": cardinality_breakdown,
        "zero_candidate_audit": zero_cand_audit,
        "sweeps": sweep_table,
        "system_resources": {
            "initial_rss_mb": round(init_rss, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "final_rss_mb": round(final_rss, 2),
            "total_runtime_sec": round(total_runtime, 2)
        }
    }

    if is_parity:
        summary["lexical_parity_gate"] = {
            "reference_true_links": 112143,
            "reference_link_recall": 0.649242,
            "reference_full_coverage": 0.304864,
            "reference_oracle_f05": 0.831634,
            "achieved_true_links": total_union_hits,
            "achieved_link_recall": round(overall_link_recall, 6),
            "achieved_full_coverage": round(full_coverage_rate, 6),
            "achieved_oracle_f05": round(oracle_macro_f05, 6),
            "parity_verdict": "PASS" if parity_passed else "FAIL"
        }

    # Save summary files
    mode_suffix = "lexical_parity" if is_parity else mode
    summary_path = os.path.join(output_dir, f"{experiment_id}_{mode_suffix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(output_dir, f"{experiment_id}_{mode_suffix}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Blocking Benchmark Report: {experiment_id} [{tag_prefix}]\n\n")
        f.write(f"- **Mode**: `{mode}`\n")
        f.write(f"- **Git Commit**: `{git_sha}` (dirty: `{git_dirty}`)\n")
        f.write(f"- **Evaluated S1 Entities**: `{len(val_s1_records):,}`\n")
        f.write(f"- **Evaluated Candidate Population**: `{total_scanned_rows:,}`\n")
        f.write(f"- **Total True Links**: `{total_true_links:,}`\n\n")

        if is_parity:
            f.write(f"## Lexical Parity Gate Verdict: **{'PASS' if parity_passed else 'FAIL'}**\n\n")
            f.write("| Metric | EXP_003 Authoritative Baseline | EXP_004 Lexical-Only Path | Match? |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **True Links Hit** | 112,143 / 172,729 | {total_union_hits:,} / {total_true_links:,} | {'YES' if total_union_hits == 112143 else 'NO'} |\n")
            f.write(f"| **True Link Recall** | 0.649242 (64.92%) | {overall_link_recall:.6f} ({overall_link_recall:.2%}) | {'YES' if abs(overall_link_recall-0.649242)<1e-4 else 'NO'} |\n")
            f.write(f"| **Full Entity Coverage** | 0.304864 (30.49%) | {full_coverage_rate:.6f} ({full_coverage_rate:.2%}) | {'YES' if abs(full_coverage_rate-0.304864)<1e-4 else 'NO'} |\n")
            f.write(f"| **Oracle Macro F0.5** | **0.831634** | **{oracle_macro_f05:.6f}** | {'YES' if abs(oracle_macro_f05-0.831634)<1e-4 else 'NO'} |\n\n")

        f.write("## 1. Primary Ceiling Metrics\n\n")
        f.write("| Metric | Score | Note |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Oracle Macro F0.5** | **{oracle_macro_f05:.6f}** | Entity-level macro F0.5 |\n")
        f.write(f"| True-Link Recall | {overall_link_recall:.4%} | {total_union_hits:,} / {total_true_links:,} |\n")
        f.write(f"| Full-Entity Coverage | {full_coverage_rate:.4%} | Non-singleton entities with 100% hits |\n")
        f.write(f"| Total Candidate Pairs | {cand_stats['total_pairs']:,} | Manageable volume |\n")
        f.write(f"| Mean Candidates / S1 | {cand_stats['mean']:.1f} | Median: {cand_stats['median']:.1f}, P99: {cand_stats['p99']:.1f} |\n")
        f.write(f"| Peak RAM (RSS) | {peak_rss:.1f} MB | Initial: {init_rss:.1f} MB |\n")
        f.write(f"| Total Runtime | {total_runtime:.2f}s | Finished cleanly |\n\n")

        if sweep_table:
            f.write("## 2. In-Memory Top-K & Similarity Threshold Sweep Results\n\n")
            f.write("| Top-K | Cosine Thresh | Links Hit | Recall | EXP_003 Misses Recovered | % Misses Recovered | Candidate Pairs | Cands / Rec Link |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for row in sweep_table:
                f.write(f"| {row['top_k']} | {row['similarity_threshold']:.2f} | {row['true_links_hit']:,} | {row['true_link_recall']:.2%} | {row['exp003_misses_recovered']:,} | {row['pct_misses_recovered']:.2%} | {row['total_candidate_pairs']:,} | {row['cands_per_recovered_link']} |\n")
            f.write("\n")

        f.write("## 3. Entity Cardinality Breakdown\n\n")
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
    parser.add_argument("--mode", choices=["plumbing_smoke", "positive_smoke", "pilot", "full", "parity_check"], default="plumbing_smoke", help="Execution mode")
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
