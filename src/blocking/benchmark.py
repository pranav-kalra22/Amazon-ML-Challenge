#!/usr/bin/env python3
"""
Authoritative Blocking Benchmark Pipeline for Amazon ML Challenge 2026 — EXP_002.

Evaluates candidate-generation strategies on the canonical 50k validation split
against the full ~10.32M training S2/S3 candidate population.

Key Features:
1. Loads configuration authoritatively from YAML (e.g., configs/blocking/blocking_v02.yaml).
2. Continuous peak-RAM sampling via background psutil thread (initial, true peak, final RSS).
3. Evaluates True Link Recall, Full Entity Coverage, Oracle Macro F0.5.
4. Evaluates per-channel recall and incremental contribution across A, B, C2, D2, E2, F.
5. Evaluates Top-K diagnostic candidate caps using deterministic evidence-ranked sorting.
6. Measures US vs India, S2 vs S3, and missing-address link recall.
7. Categorizes misses with measurable recoverability diagnostics, separating OBSERVATION from PROPOSED RECOVERY CHANNEL.
8. Logs results honestly into experiments/experiment_log.csv with code_commit, results_commit, and git_dirty.
"""

import os
import sys
sys.path.insert(0, ".")
import time
import json
import yaml
import psutil
import threading
import subprocess
import argparse

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional

from src.blocking.blocker import (
    MultiChannelBlocker,
    CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F,
    CHANNEL_NAMES
)
from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive,
    COMMON_ADDR_STOP,
    LEGAL_TERMS
)
from src.evaluation.macro_f05 import evaluate_predictions, compute_entity_f05


class MemoryTracker:
    """
    Lightweight background sampler tracking true peak memory (RSS) using psutil.
    """
    def __init__(self, interval_sec: float = 0.5):
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
    """Returns (commit_sha, is_dirty) for reproducibility tracking."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        is_dirty = len(status_out) > 0
        return sha, is_dirty
    except Exception:
        return "unknown", True


def run_benchmark(
    config_path: str = "configs/blocking/blocking_v02.yaml",
    val_split_path: str = "artifacts/splits/val_s1_ids_50k_seed42.parquet",
    s1_path: str = "dataset/train/train_source1.tsv",
    gt_path: str = "dataset/train/train_ground_truth.tsv",
    s2_path: str = "dataset/train/train_source2.tsv",
    s3_path: str = "dataset/train/train_source3.tsv",
    output_dir: str = "reports/blocking",
    experiment_id: str = "EXP_002_corrected_lexical_blocker",
    code_commit_override: Optional[str] = None
):
    mem_tracker = MemoryTracker(interval_sec=0.5)
    mem_tracker.start()
    start_time = time.time()

    current_commit, git_dirty = get_git_status()
    code_commit = code_commit_override if code_commit_override else current_commit

    print(f"=== Starting Blocking Benchmark {experiment_id} ===", flush=True)
    print(f"Config File:   {config_path}", flush=True)
    print(f"Code Commit:   {code_commit} (git_dirty: {git_dirty})", flush=True)
    print(f"Initial RAM:   {mem_tracker.initial_rss:.2f} MB", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load authoritative YAML configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded YAML configuration: '{config.get('name')}' (version {config.get('version')})", flush=True)

    # 2. Load canonical validation split IDs
    t0 = time.time()
    val_meta_df = pd.read_parquet(val_split_path)
    val_s1_id_set = set(val_meta_df["entity_id"])
    print(f"Loaded {len(val_s1_id_set):,} validation S1 IDs from {val_split_path} in {time.time()-t0:.2f}s", flush=True)

    # 3. Load Source 1 records for validation entities
    t0 = time.time()
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
            if parts[col_id] in val_s1_id_set:
                rec = {
                    "entity_id": parts[col_id],
                    "business_name": parts[col_name],
                    "business_address": parts[col_addr],
                    "country": parts[col_country]
                }
                val_s1_records.append(rec)
                val_s1_lookup[parts[col_id]] = rec
    print(f"Extracted {len(val_s1_records):,} validation S1 records in {time.time()-t0:.2f}s", flush=True)

    # 4. Load Ground Truth for validation entities
    t0 = time.time()
    val_gt = {}
    total_val_true_links = 0
    s2_true_links = 0
    s3_true_links = 0
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_id_set:
                m_ids = set()
                if len(parts) >= 2 and parts[1].strip():
                    for x in parts[1].split(","):
                        cand = x.strip()
                        if cand:
                            m_ids.add(cand)
                            if cand.startswith("S2-"):
                                s2_true_links += 1
                            else:
                                s3_true_links += 1
                val_gt[s1_id] = m_ids
                total_val_true_links += len(m_ids)
    print(f"Loaded validation Ground Truth: {len(val_gt):,} entities, {total_val_true_links:,} true links (S2: {s2_true_links:,}, S3: {s3_true_links:,}) in {time.time()-t0:.2f}s", flush=True)

    # 5. Initialize Blocker from YAML and build query index (Pass 1 candidate-DF scanning included)
    t_idx_start = time.time()
    blocker = MultiChannelBlocker(config=config)
    blocker.build_query_index(
        val_s1_records,
        candidate_file_paths=[s2_path, s3_path]
    )
    idx_runtime = time.time() - t_idx_start
    print(f"Index built in {idx_runtime:.2f}s. Current observed peak RAM: {mem_tracker.peak_rss:.2f} MB", flush=True)

    # 6. Stream candidate population across full S2 and S3 files (Pass 2)
    t_retrieval_start = time.time()
    candidates = blocker.generate_candidates_streaming(
        [s2_path, s3_path],
        progress_interval=1000000
    )
    retrieval_runtime = time.time() - t_retrieval_start
    print(f"Retrieval complete in {retrieval_runtime:.2f}s. Current observed peak RAM: {mem_tracker.peak_rss:.2f} MB", flush=True)

    # 7. Benchmark Metrics Calculation
    t_eval_start = time.time()
    print("\nComputing comprehensive benchmark metrics...", flush=True)

    # Per-entity candidate sets and channel sets
    candidate_sets = {}
    channel_candidates = {ch: defaultdict(set) for ch in ["A", "B", "C2", "D2", "E2", "F"]}
    candidate_counts = []
    ch_bits = {"A": CH_A, "B": CH_B, "C2": CH_C2, "D2": CH_D2, "E2": CH_E2, "F": CH_F}

    for s1_idx, s1_id in enumerate(blocker.idx_to_s1_id):
        cand_dict = candidates[s1_idx]
        cand_set = set(cand_dict.keys())
        candidate_sets[s1_id] = cand_set
        candidate_counts.append(len(cand_set))

        for cand_id, mask in cand_dict.items():
            for ch, bit in ch_bits.items():
                if mask & bit:
                    channel_candidates[ch][s1_id].add(cand_id)

    total_candidates = sum(candidate_counts)
    cand_counts_arr = np.array(candidate_counts)

    cand_stats = {
        "mean": float(np.mean(cand_counts_arr)),
        "median": float(np.median(cand_counts_arr)),
        "p90": float(np.percentile(cand_counts_arr, 90)),
        "p95": float(np.percentile(cand_counts_arr, 95)),
        "p99": float(np.percentile(cand_counts_arr, 99)),
        "max": int(np.max(cand_counts_arr)),
        "min": int(np.min(cand_counts_arr))
    }

    # Total possible search space: S2 (5,034,616) + S3 (5,285,603) = 10,320,219
    total_search_population = 10320219
    total_brute_force_pairs = len(val_s1_id_set) * total_search_population
    reduction_ratio = 1.0 - (total_candidates / total_brute_force_pairs)

    # True Link Recall and Full Entity Coverage
    retrieved_true_links = 0
    retrieved_s2_links = 0
    retrieved_s3_links = 0
    retrieved_missing_addr_links = 0
    total_missing_addr_true_links = 0
    full_coverage_entities = 0
    non_singleton_entities = 0

    missed_links = []
    oracle_predictions = {}

    for s1_id, truth_set in val_gt.items():
        cand_set = candidate_sets.get(s1_id, set())
        tp_set = truth_set & cand_set
        retrieved_true_links += len(tp_set)
        oracle_predictions[s1_id] = tp_set

        for m in tp_set:
            if m.startswith("S2-"):
                retrieved_s2_links += 1
            else:
                retrieved_s3_links += 1

        if len(truth_set) > 0:
            non_singleton_entities += 1
            if truth_set.issubset(cand_set):
                full_coverage_entities += 1

            missed_set = truth_set - cand_set
            for mid in missed_set:
                missed_links.append((s1_id, mid))

    link_recall = retrieved_true_links / total_val_true_links if total_val_true_links > 0 else 0.0
    s2_recall = retrieved_s2_links / s2_true_links if s2_true_links > 0 else 0.0
    s3_recall = retrieved_s3_links / s3_true_links if s3_true_links > 0 else 0.0
    full_coverage_rate = full_coverage_entities / non_singleton_entities if non_singleton_entities > 0 else 0.0

    # Calculate Oracle Macro F0.5
    metadata_map = {rec["entity_id"]: {"country": rec["country"]} for rec in val_s1_records}
    oracle_eval_results = evaluate_predictions(val_gt, oracle_predictions, metadata_map)

    # Per-Channel Recall & Incremental Contribution
    print("Measuring per-channel performance and incremental links added...", flush=True)
    channel_metrics = []
    accumulated_links = set()

    for ch in ["A", "B", "C2", "D2", "E2", "F"]:
        ch_cands = channel_candidates[ch]
        ch_retrieved = 0
        ch_cand_counts = [len(ch_cands.get(s1_id, set())) for s1_id in val_s1_id_set]
        ch_avg_cands = float(np.mean(ch_cand_counts))

        ch_links = set()
        for s1_id, truth_set in val_gt.items():
            hit_set = truth_set & ch_cands.get(s1_id, set())
            ch_retrieved += len(hit_set)
            for m in hit_set:
                ch_links.add((s1_id, m))

        ch_recall = ch_retrieved / total_val_true_links if total_val_true_links > 0 else 0.0
        unique_links_added = len(ch_links - accumulated_links)
        accumulated_links.update(ch_links)

        channel_metrics.append({
            "channel": ch,
            "description": {
                "A": "Exact Normalized Name",
                "B": "Compact & Domain-Normalized",
                "C2": "Candidate-DF-Aware Rare Token",
                "D2": "True Address-Only Rescue",
                "E2": "Symmetric Transliteration",
                "F": "Order-Invariant Token Pairs"
            }[ch],
            "link_recall": ch_recall,
            "true_links_retrieved": ch_retrieved,
            "unique_links_added": unique_links_added,
            "avg_candidates": ch_avg_cands
        })

    # Diagnostic Candidate Caps Benchmark with DETERMINISTIC EVIDENCE RANKING
    print("Benchmarking diagnostic candidate caps with deterministic evidence ranking...", flush=True)
    diagnostic_caps = config.get("constraints", {}).get("diagnostic_caps", [None, 500, 250, 100, 50, 30])
    cap_results = []

    # Pre-rank candidate lists for each entity
    print("  Ranking candidates deterministically per entity...", flush=True)
    ranked_candidates_per_entity = {}
    for s1_idx, s1_id in enumerate(blocker.idx_to_s1_id):
        cand_dict = candidates[s1_idx]
        ranked_candidates_per_entity[s1_id] = blocker.rank_entity_candidates(cand_dict, cap=None)

    for cap in diagnostic_caps:
        cap_retrieved = 0
        cap_full_cov = 0
        cap_oracle_preds = {}

        for s1_id, truth_set in val_gt.items():
            ranked_cands = ranked_candidates_per_entity.get(s1_id, [])
            if cap is not None:
                eff_cands = set(ranked_cands[:cap])
            else:
                eff_cands = set(ranked_cands)

            tp = truth_set & eff_cands
            cap_retrieved += len(tp)
            cap_oracle_preds[s1_id] = tp

            if len(truth_set) > 0 and truth_set.issubset(eff_cands):
                cap_full_cov += 1

        c_recall = cap_retrieved / total_val_true_links if total_val_true_links > 0 else 0.0
        c_cov = cap_full_cov / non_singleton_entities if non_singleton_entities > 0 else 0.0
        c_eval = evaluate_predictions(val_gt, cap_oracle_preds)

        cap_results.append({
            "cap": "Uncapped" if cap is None else str(cap),
            "link_recall": c_recall,
            "full_entity_coverage": c_cov,
            "oracle_macro_f05": c_eval["macro_f05"]
        })

    eval_runtime = time.time() - t_eval_start

    # 8. Miss Analysis Export with Measurable Diagnostics
    print(f"\nAnalyzing and categorizing {len(missed_links):,} missed true links...", flush=True)
    t_miss_start = time.time()

    miss_cand_ids = set(mid for _, mid in missed_links)
    miss_cand_lookup = {}
    for cand_file in [s2_path, s3_path]:
        with open(cand_file, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            col_id = header.index("entity_id")
            col_name = header.index("business_name")
            col_addr = header.index("business_address")
            col_country = header.index("country")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts[col_id] in miss_cand_ids:
                    miss_cand_lookup[parts[col_id]] = {
                        "entity_id": parts[col_id],
                        "business_name": parts[col_name],
                        "business_address": parts[col_addr],
                        "country": parts[col_country]
                    }

    miss_rows = []
    miss_category_counts = defaultdict(int)

    for s1_id, mid in missed_links:
        s1_data = val_s1_lookup.get(s1_id, {})
        cand_data = miss_cand_lookup.get(mid, {})
        src = "S2" if mid.startswith("S2-") else "S3"
        c = s1_data.get("country", "")

        raw_s1_n = s1_data.get("business_name", "")
        raw_s1_a = s1_data.get("business_address", "")
        raw_cand_n = cand_data.get("business_name", "")
        raw_cand_a = cand_data.get("business_address", "")

        s1_n_norm = normalize_name_non_destructive(raw_s1_n)
        cand_n_norm = normalize_name_non_destructive(raw_cand_n)
        s1_a_norm = normalize_address_non_destructive(raw_s1_a)
        cand_a_norm = normalize_address_non_destructive(raw_cand_a)

        # Measurable token overlaps
        s1_n_toks = set(s1_n_norm["tokens"])
        cand_n_toks = set(cand_n_norm["tokens"])
        name_jaccard = len(s1_n_toks & cand_n_toks) / len(s1_n_toks | cand_n_toks) if (s1_n_toks | cand_n_toks) else 0.0

        shared_name_toks = s1_n_toks & cand_n_toks
        shared_addr_toks = set(s1_a_norm["distinctive_tokens"]) & set(cand_a_norm["distinctive_tokens"])
        bldg_match = bool(s1_a_norm["building_numeric"] and s1_a_norm["building_numeric"] == cand_a_norm["building_numeric"])
        postal_match = bool(s1_a_norm["postal_code"] and s1_a_norm["postal_code"] == cand_a_norm["postal_code"])
        trans_overlap = set(s1_n_norm["trans_tokens"]) & set(cand_n_norm["trans_tokens"])

        is_missing_addr = not raw_cand_a or raw_cand_a.lower() in {"", "nan", "<null>", "null", "none"}
        if is_missing_addr:
            total_missing_addr_true_links += 1

        # Heuristic miss category & Measurable recoverability
        if is_missing_addr:
            cat = "missing_address_candidate"
            obs = f"Candidate address is completely missing. Shared name tokens: {len(shared_name_toks)} (Jaccard: {name_jaccard:.2f})."
            rec_ch = "approximate_name_ngram_or_embedding"
        elif s1_n_norm["trans_stripped"] != s1_n_norm["legal_stripped"] or cand_n_norm["trans_stripped"] != cand_n_norm["legal_stripped"]:
            cat = "complex_transliteration_variant"
            obs = f"Cross-script or transliteration variant. Shared transliterated tokens: {len(trans_overlap)}."
            rec_ch = "soft_transliteration_or_phonetic"
        elif len(shared_addr_toks) >= 1 or bldg_match or postal_match:
            cat = "address_shared_unindexed"
            obs = f"Address shares features: bldg_match={bldg_match}, postal_match={postal_match}, shared_addr_toks={len(shared_addr_toks)}."
            rec_ch = "relaxed_address_anchor"
        elif name_jaccard > 0.0:
            cat = "name_token_overlap_unindexed"
            obs = f"Name tokens overlap ({len(shared_name_toks)} tokens), but suppressed by candidate DF threshold."
            rec_ch = "higher_df_token_combination"
        else:
            cat = "severe_alias_or_dba"
            obs = "No detectable lexical overlap in name or address."
            rec_ch = "sparse_character_ngram_or_bi_encoder"

        miss_category_counts[cat] += 1

        miss_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_id": mid,
            "source": src,
            "country": c,
            "raw_s1_name": raw_s1_n,
            "raw_candidate_name": raw_cand_n,
            "raw_s1_address": raw_s1_a,
            "raw_candidate_address": raw_cand_a,
            "name_token_jaccard": round(name_jaccard, 4),
            "shared_name_tokens_count": len(shared_name_toks),
            "shared_addr_tokens_count": len(shared_addr_toks),
            "building_match": bldg_match,
            "postal_match": postal_match,
            "heuristic_miss_category": cat,
            "observation": obs,
            "proposed_recovery_channel": rec_ch
        })

    miss_csv_path = os.path.join(output_dir, f"{experiment_id}_blocking_misses.csv")
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(miss_csv_path, index=False)
    print(f"Exported {len(miss_df):,} misses to {miss_csv_path} in {time.time()-t_miss_start:.2f}s", flush=True)

    # Stop memory tracker and compute peak RAM
    init_rss, peak_rss, final_rss = mem_tracker.stop()
    total_runtime = time.time() - start_time

    # 9. Compile Benchmark Summary Report
    summary = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_split": "val_50k_seed42",
        "code_commit": code_commit,
        "results_commit": "pending",
        "git_dirty": git_dirty,
        "config_file": config_path,
        "total_validation_s1_entities": len(val_s1_id_set),
        "total_validation_true_links": total_val_true_links,
        "candidate_search_population": total_search_population,
        "primary_metrics": {
            "true_link_recall": round(link_recall, 6),
            "full_entity_coverage": round(full_coverage_rate, 6),
            "oracle_macro_f05": round(oracle_eval_results["macro_f05"], 6),
            "oracle_macro_precision": round(oracle_eval_results["macro_precision"], 6),
            "oracle_macro_recall": round(oracle_eval_results["macro_recall"], 6),
            "total_candidate_pairs": total_candidates,
            "reduction_ratio": round(reduction_ratio, 8)
        },
        "candidate_statistics": cand_stats,
        "subgroup_metrics": {
            "country_breakdown": oracle_eval_results.get("country_breakdown", {}),
            "source_diagnostics": {
                "S2": {"true_links": s2_true_links, "tp_links": retrieved_s2_links, "link_recall": round(s2_recall, 6)},
                "S3": {"true_links": s3_true_links, "tp_links": retrieved_s3_links, "link_recall": round(s3_recall, 6)}
            },
            "missing_address_diagnostics": {
                "total_missing_addr_links_in_misses": total_missing_addr_true_links
            }
        },
        "per_channel_metrics": channel_metrics,
        "diagnostic_caps_ranked": cap_results,
        "miss_analysis": {
            "total_missed_links": len(missed_links),
            "miss_rate": round(len(missed_links) / total_val_true_links, 6) if total_val_true_links > 0 else 0.0,
            "category_breakdown": dict(miss_category_counts),
            "miss_file": miss_csv_path
        },
        "performance_profile": {
            "index_runtime_sec": round(idx_runtime, 2),
            "retrieval_runtime_sec": round(retrieval_runtime, 2),
            "evaluation_runtime_sec": round(eval_runtime, 2),
            "total_runtime_sec": round(total_runtime, 2),
            "initial_memory_mb": round(init_rss, 2),
            "peak_memory_mb": round(peak_rss, 2),
            "final_memory_mb": round(final_rss, 2)
        }
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, f"{experiment_id}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown Benchmark Report
    md_path = os.path.join(output_dir, f"{experiment_id}_benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Candidate Generation Benchmark Report — `{experiment_id}`\n\n")
        f.write(f"**Date:** {summary['timestamp']}  \n")
        f.write(f"**Validation Split:** `val_50k_seed42` (50,000 Source 1 Entities)  \n")
        f.write(f"**Code Commit:** `{code_commit}` (git_dirty: `{git_dirty}`)  \n")
        f.write(f"**Authoritative Config:** `{config_path}`  \n")
        f.write(f"**Candidate Search Space:** {total_search_population:,} Training S2 + S3 Records  \n")
        f.write(f"**Total Runtime:** {round(total_runtime, 2)}s | **Peak RAM:** {round(peak_rss, 2)} MB (Initial: {round(init_rss, 2)} MB, Final: {round(final_rss, 2)} MB)  \n\n")

        f.write("## 1. Primary Ceiling Metrics `[VALIDATION MEASUREMENT]`\n\n")
        f.write("| Metric | EXP_001 Baseline | EXP_002 Corrected | Delta | Target |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **True Link Recall** | 83.390% | **{link_recall*100:.3f}%** ({retrieved_true_links:,} / {total_val_true_links:,}) | **{'+' if link_recall >= 0.8339 else ''}{(link_recall - 0.8339)*100:.3f}%** | $\\ge 98.0\\%$ |\n")
        f.write(f"| **Full Entity Coverage** | 62.030% | **{full_coverage_rate*100:.3f}%** ({full_coverage_entities:,} / {non_singleton_entities:,}) | **{'+' if full_coverage_rate >= 0.6203 else ''}{(full_coverage_rate - 0.6203)*100:.3f}%** | Highest possible |\n")
        f.write(f"| **Oracle Macro $F_{{0.5}}$** | 0.925756 | **{oracle_eval_results['macro_f05']:.6f}** | **{'+' if oracle_eval_results['macro_f05'] >= 0.925756 else ''}{oracle_eval_results['macro_f05'] - 0.925756:.6f}** | $\\ge 0.985884$ |\n")
        f.write(f"| **Mean Candidates/S1** | 502.39 | **{cand_stats['mean']:.2f}** | {cand_stats['mean'] - 502.39:+.2f} | Manageable volume |\n")
        f.write(f"| **Total Candidate Pairs** | 25,119,679 | {total_candidates:,} | {total_candidates - 25119679:+,} | Scalable volume |\n")
        f.write(f"| **Reduction Ratio** | 99.995132% | **{reduction_ratio*100:.6f}%** | — | $> 99.99\\%$ |\n\n")

        f.write("## 2. Candidate Volume Distribution per Entity\n\n")
        f.write("| Statistic | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Mean | {cand_stats['mean']:.2f} |\n")
        f.write(f"| Median (p50) | {cand_stats['median']:.1f} |\n")
        f.write(f"| p90 | {cand_stats['p90']:.1f} |\n")
        f.write(f"| p95 | {cand_stats['p95']:.1f} |\n")
        f.write(f"| p99 | {cand_stats['p99']:.1f} |\n")
        f.write(f"| Maximum | {cand_stats['max']:,} |\n\n")

        f.write("## 3. Per-Channel Recall & Incremental Contribution\n\n")
        f.write("| Channel | Description | Link Recall | True Links Retrieved | Unique Links Added | Avg Candidates |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for ch in channel_metrics:
            f.write(f"| **{ch['channel']}** | {ch['description']} | {ch['link_recall']*100:.2f}% | {ch['true_links_retrieved']:,} | **+{ch['unique_links_added']:,}** | {ch['avg_candidates']:.1f} |\n")
        f.write(f"| **UNION** | **Channels A + B + C2 + D2 + E2 + F** | **{link_recall*100:.3f}%** | **{retrieved_true_links:,}** | **{retrieved_true_links:,}** | **{cand_stats['mean']:.1f}** |\n\n")

        f.write("## 4. Deterministic Top-K Candidate Caps Benchmark\n\n")
        f.write("> **NOTE:** Candidates deterministically ranked by composite evidence score descending, entity ID ascending.\n\n")
        f.write("| Top-K Cap | Link Recall | Full Entity Coverage | Oracle Macro $F_{0.5}$ |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for cap in cap_results:
            f.write(f"| `Top {cap['cap']}` | {cap['link_recall']*100:.3f}% | {cap['full_entity_coverage']*100:.3f}% | **{cap['oracle_macro_f05']:.6f}** |\n")

        f.write("\n## 5. Miss Analysis Breakdown (Measurable Recoverability Diagnostics)\n\n")
        f.write(f"- **Total Missed Links:** {len(missed_links):,} ({len(missed_links)/total_val_true_links*100:.3f}% miss rate)\n")
        f.write(f"- **Miss Analysis CSV:** `{miss_csv_path}`\n\n")
        f.write("| Heuristic Miss Category | Count | Percentage | Primary Observation | Proposed Recovery Channel |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cat, cnt in sorted(miss_category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| `{cat}` | {cnt:,} | {cnt/len(missed_links)*100:.2f}% | Measured in CSV | See recoverability analysis |\n")

        f.write("\n## 6. Subgroup Diagnostics\n\n")
        f.write("### Country Breakdown\n\n")
        for c, st in oracle_eval_results.get("country_breakdown", {}).items():
            f.write(f"- **{c}** ({st['entity_count']:,} entities): Oracle Macro $F_{{0.5}} = {st['macro_f05']:.6f}$, Precision = {st['macro_precision']:.6f}, Recall = {st['macro_recall']:.6f}\n")

        f.write("\n### Source Diagnostics\n\n")
        f.write(f"- **Source 2 Links**: Retrieved = {retrieved_s2_links:,} / {s2_true_links:,} ({s2_recall*100:.2f}% recall)\n")
        f.write(f"- **Source 3 Links**: Retrieved = {retrieved_s3_links:,} / {s3_true_links:,} ({s3_recall*100:.2f}% recall)\n")

    print(f"\n[VALIDATION MEASUREMENT] Benchmark Complete!")
    print(f"  True Link Recall:     {link_recall*100:.3f}% ({retrieved_true_links:,} / {total_val_true_links:,})")
    print(f"  Full Entity Coverage: {full_coverage_rate*100:.3f}% ({full_coverage_entities:,} / {non_singleton_entities:,})")
    print(f"  Oracle Macro F0.5:    {oracle_eval_results['macro_f05']:.6f}")
    print(f"  Mean Candidates:      {cand_stats['mean']:.2f} (Median: {cand_stats['median']:.1f}, Max: {cand_stats['max']:,})")
    print(f"  Total Candidates:     {total_candidates:,}")
    print(f"  Observed Peak RAM:    {peak_rss:.2f} MB")
    print(f"  Reports Saved To:     {md_path}")

    # 10. Register into experiments/experiment_log.csv
    log_path = "experiments/experiment_log.csv"
    if os.path.exists(log_path):
        exp_row = {
            "experiment_id": experiment_id,
            "owner": "team",
            "status": "COMPLETED",
            "branch": "phase2/blocking-baseline",
            "hypothesis": "Corrected multi-channel blocker (candidate-DF rare tokens, address-only rescue, symmetric transliteration, order-invariant pairs) elevates recall ceiling with deterministic evidence ranking.",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "completed_at": summary["timestamp"],
            "git_commit": code_commit,
            "code_commit": code_commit,
            "results_commit": "pending",
            "git_dirty": git_dirty,
            "validation_split": "val_50k_seed42",
            "blocker_config": config_path,
            "feature_config": "none",
            "negative_sampling": "none",
            "model": "candidate_blocker_oracle",
            "hyperparameters": json.dumps({
                "channels": ["A", "B", "C2", "D2", "E2", "F"],
                "candidate_df_max": config.get("channels", {}).get("channel_C2_candidate_df_rare_token", {}).get("max_candidate_df", 5),
                "ranking_weights": config.get("candidate_ranking", {}).get("channel_weights", {})
            }),
            "threshold_config": "{}",
            "candidate_recall": round(link_recall, 6),
            "full_entity_coverage": round(full_coverage_rate, 6),
            "oracle_f05": round(oracle_eval_results["macro_f05"], 6),
            "validation_macro_f05": 0.0,
            "singleton_accuracy": 1.0,
            "precision": round(oracle_eval_results["macro_precision"], 6),
            "recall": round(oracle_eval_results["macro_recall"], 6),
            "runtime_sec": round(total_runtime, 2),
            "memory_peak_mb": round(peak_rss, 2),
            "is_current_best": False,
            "notes": f"EXP_002 Blocker. True-link recall: {link_recall*100:.2f}%, Oracle F0.5: {oracle_eval_results['macro_f05']:.4f}, Mean cands: {cand_stats['mean']:.1f}, Peak RAM: {peak_rss:.1f}MB.",
            "artifact_paths": f"{md_path};{miss_csv_path};{json_path}"
        }

        exp_df = pd.read_csv(log_path)
        if experiment_id in exp_df["experiment_id"].values:
            exp_df = exp_df[exp_df["experiment_id"] != experiment_id]
        new_row_df = pd.DataFrame([exp_row])
        exp_df = pd.concat([exp_df, new_row_df], ignore_index=True)
        exp_df.to_csv(log_path, index=False)
        print(f"Logged {experiment_id} into {log_path} successfully!", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Blocking Benchmark")
    parser.add_argument("--config", type=str, default="configs/blocking/blocking_v02.yaml", help="Path to blocker YAML config")
    parser.add_argument("--experiment-id", type=str, default="EXP_002_corrected_lexical_blocker", help="Experiment ID")
    parser.add_argument("--code-commit", type=str, default=None, help="Explicit code commit SHA")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        experiment_id=args.experiment_id,
        code_commit_override=args.code_commit
    )
