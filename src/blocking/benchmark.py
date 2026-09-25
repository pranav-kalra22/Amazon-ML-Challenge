#!/usr/bin/env python3
"""
Blocking Benchmark Pipeline for Amazon ML Challenge 2026.

Evaluates candidate-generation strategies on the canonical 50k validation split
against the full ~10.32M training S2/S3 candidate population.

Measures:
1. True Link Recall
2. Full Entity Coverage
3. Candidate Statistics (mean, median, p90, p95, p99, max)
4. Total Candidate Pairs
5. Reduction Ratio
6. Oracle Candidate Macro F0.5 (authoritative metric)
7. Per-Channel Recall & Incremental Contribution
8. Diagnostic Candidate Caps (uncapped, 200, 100, 50, 30)
9. Miss Analysis CSV: reports/blocking/EXP_001_blocking_misses.csv
10. Performance Profiling (runtimes, peak RAM)
"""

import os
import sys
sys.path.insert(0, ".")
import time
import json
import psutil

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any

from src.blocking.blocker import MultiChannelBlocker
from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive
)
from src.evaluation.macro_f05 import evaluate_predictions, compute_entity_f05


def run_benchmark(
    val_split_path: str = "artifacts/splits/val_s1_ids_50k_seed42.parquet",
    s1_path: str = "dataset/train/train_source1.tsv",
    gt_path: str = "dataset/train/train_ground_truth.tsv",
    s2_path: str = "dataset/train/train_source2.tsv",
    s3_path: str = "dataset/train/train_source3.tsv",
    output_dir: str = "reports/blocking",
    experiment_id: str = "EXP_001_blocking_baseline"
):
    start_time = time.time()
    process = psutil.Process()
    ram_initial = process.memory_info().rss / (1024 * 1024)
    print(f"=== Starting Blocking Benchmark {experiment_id} ===", flush=True)
    print(f"Initial Memory: {ram_initial:.2f} MB", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load canonical validation split IDs
    t0 = time.time()
    val_meta_df = pd.read_parquet(val_split_path)
    val_s1_id_set = set(val_meta_df["entity_id"])
    print(f"Loaded {len(val_s1_id_set):,} validation S1 IDs from {val_split_path} in {time.time()-t0:.2f}s", flush=True)

    # 2. Load Source 1 records for validation entities
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

    # 3. Load Ground Truth for validation entities
    t0 = time.time()
    val_gt = {}
    total_val_true_links = 0
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_id_set:
                m_ids = set()
                if len(parts) >= 2 and parts[1].strip():
                    m_ids = set(x.strip() for x in parts[1].split(",") if x.strip())
                val_gt[s1_id] = m_ids
                total_val_true_links += len(m_ids)
    print(f"Loaded validation Ground Truth: {len(val_gt):,} entities, {total_val_true_links:,} true links in {time.time()-t0:.2f}s", flush=True)

    # 4. Initialize Multi-Channel Blocker and Build Query Inverted Index
    t_idx_start = time.time()
    blocker = MultiChannelBlocker(token_max_doc_freq=3, token_min_len=5)
    blocker.build_query_index(val_s1_records)
    idx_runtime = time.time() - t_idx_start
    ram_after_index = process.memory_info().rss / (1024 * 1024)
    print(f"Index Memory: {ram_after_index:.2f} MB (Peak Delta: {ram_after_index - ram_initial:.2f} MB)", flush=True)


    # 5. Stream candidate population across full S2 and S3 files
    t_retrieval_start = time.time()
    candidates = blocker.generate_candidates_streaming(
        [s2_path, s3_path],
        progress_interval=1000000
    )
    retrieval_runtime = time.time() - t_retrieval_start
    ram_after_retrieval = process.memory_info().rss / (1024 * 1024)
    print(f"Retrieval Memory: {ram_after_retrieval:.2f} MB", flush=True)

    # 6. Benchmark Metrics Calculation
    t_eval_start = time.time()
    print("\nComputing comprehensive benchmark metrics...", flush=True)

    # Per-entity candidate sets and channel sets
    # candidate_sets: {s1_id: set(cand_ids)}
    # channel_candidates: {channel: {s1_id: set(cand_ids)}}
    candidate_sets = {}
    channel_candidates = {ch: defaultdict(set) for ch in ["A", "B", "C", "D", "E"]}
    candidate_counts = []
    ch_bits = {"A": 1, "B": 2, "C": 4, "D": 8, "E": 16}

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

    # Total possible search space
    # Total S2 + S3 = 5,034,616 + 5,285,603 = 10,320,219
    total_search_population = 10320219
    total_brute_force_pairs = len(val_s1_id_set) * total_search_population
    reduction_ratio = 1.0 - (total_candidates / total_brute_force_pairs)

    # Calculate True Link Recall and Full Entity Coverage
    retrieved_true_links = 0
    full_coverage_entities = 0
    non_singleton_entities = 0

    missed_links = []  # To export for miss analysis
    oracle_predictions = {}

    for s1_id, truth_set in val_gt.items():
        cand_set = candidate_sets.get(s1_id, set())
        tp_set = truth_set & cand_set
        retrieved_true_links += len(tp_set)
        oracle_predictions[s1_id] = tp_set

        if len(truth_set) > 0:
            non_singleton_entities += 1
            if truth_set.issubset(cand_set):
                full_coverage_entities += 1

            # Check missed true links
            missed_set = truth_set - cand_set
            for mid in missed_set:
                missed_links.append((s1_id, mid))

    link_recall = retrieved_true_links / total_val_true_links if total_val_true_links > 0 else 0.0
    full_coverage_rate = full_coverage_entities / non_singleton_entities if non_singleton_entities > 0 else 0.0

    # Calculate Oracle Macro F0.5 using the authoritative evaluator
    metadata_map = {rec["entity_id"]: {"country": rec["country"]} for rec in val_s1_records}
    oracle_eval_results = evaluate_predictions(val_gt, oracle_predictions, metadata_map)

    # Per-Channel Recall & Incremental Contribution
    print("Measuring per-channel performance and incremental links added...", flush=True)
    channel_metrics = []
    accumulated_links = set()

    for ch in ["A", "B", "C", "D", "E"]:
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
                "C": "Rare Token Inverted Index",
                "D": "Address Anchors",
                "E": "Transliteration Lexical"
            }[ch],
            "link_recall": ch_recall,
            "true_links_retrieved": ch_retrieved,
            "unique_links_added": unique_links_added,
            "avg_candidates": ch_avg_cands
        })

    # Diagnostic Candidate Caps Benchmark
    print("Benchmarking diagnostic candidate caps...", flush=True)
    diagnostic_caps = [None, 200, 100, 50, 30]
    cap_results = []

    for cap in diagnostic_caps:
        cap_retrieved = 0
        cap_full_cov = 0
        cap_oracle_preds = {}

        for s1_id, truth_set in val_gt.items():
            cand_set = candidate_sets.get(s1_id, set())
            if cap is not None and len(cand_set) > cap:
                # Truncate to cap
                cand_list = list(cand_set)[:cap]
                eff_cands = set(cand_list)
            else:
                eff_cands = cand_set

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
    total_runtime = time.time() - start_time
    peak_ram = process.memory_info().rss / (1024 * 1024)

    # 7. Miss Analysis Export
    print(f"\nAnalyzing and categorizing {len(missed_links):,} missed true links...", flush=True)
    t_miss_start = time.time()

    # Look up metadata for missed candidate records
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

        # Categorize Miss
        category = "other"
        if not raw_cand_a or raw_cand_a.lower() in {"", "nan", "<null>", "null"}:
            category = "missing_address"
        elif s1_n_norm["trans_stripped"] != s1_n_norm["legal_stripped"] or cand_n_norm["trans_stripped"] != cand_n_norm["legal_stripped"]:
            category = "transliteration"
        elif set(s1_n_norm["tokens"]) & set(cand_n_norm["tokens"]):
            category = "reordered_tokens_or_stopwords"
        elif len(s1_n_norm["tokens"]) > 0 and len(cand_n_norm["tokens"]) > 0:
            category = "severe_typo_or_dba"

        miss_category_counts[category] += 1

        miss_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_id": mid,
            "source": src,
            "country": c,
            "raw_s1_name": raw_s1_n,
            "raw_candidate_name": raw_cand_n,
            "raw_s1_address": raw_s1_a,
            "raw_candidate_address": raw_cand_a,
            "normalized_s1_name": s1_n_norm["legal_stripped"],
            "normalized_candidate_name": cand_n_norm["legal_stripped"],
            "blocker_channels_attempted": "A,B,C,D,E",
            "likely_miss_category": category
        })

    miss_csv_path = os.path.join(output_dir, f"{experiment_id}_blocking_misses.csv")
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(miss_csv_path, index=False)
    print(f"Exported {len(miss_df):,} misses to {miss_csv_path} in {time.time()-t_miss_start:.2f}s", flush=True)

    # 8. Compile Benchmark Summary Report
    summary = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_split": "val_50k_seed42",
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
            "source_diagnostics": oracle_eval_results.get("source_diagnostics", {})
        },
        "per_channel_metrics": channel_metrics,
        "diagnostic_caps": cap_results,
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
            "peak_memory_mb": round(peak_ram, 2)
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
        f.write(f"**Candidate Search Space:** {total_search_population:,} Training S2 + S3 Records  \n")
        f.write(f"**Total Runtime:** {round(total_runtime, 2)}s | **Peak RAM:** {round(peak_ram, 2)} MB  \n\n")

        f.write("## 1. Primary Ceiling Metrics `[VALIDATION MEASUREMENT]`\n\n")
        f.write("| Metric | Value | Target |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **True Link Recall** | **{link_recall*100:.3f}%** ({retrieved_true_links:,} / {total_val_true_links:,}) | $\\ge 99.8\\%$ |\n")
        f.write(f"| **Full Entity Coverage** | **{full_coverage_rate*100:.3f}%** ({full_coverage_entities:,} / {non_singleton_entities:,}) | Highest possible |\n")
        f.write(f"| **Oracle Macro $F_{{0.5}}$** | **{oracle_eval_results['macro_f05']:.6f}** | $\\ge 0.995$ |\n")
        f.write(f"| **Total Candidate Pairs** | {total_candidates:,} | Scalable volume |\n")
        f.write(f"| **Reduction Ratio** | **{reduction_ratio*100:.6f}%** | $> 99.99\\%$ |\n\n")

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
        f.write(f"| **UNION** | **Channels A + B + C + D + E** | **{link_recall*100:.3f}%** | **{retrieved_true_links:,}** | **{retrieved_true_links:,}** | **{cand_stats['mean']:.1f}** |\n\n")

        f.write("## 4. Diagnostic Candidate Caps Benchmark\n\n")
        f.write("| Candidate Cap | Link Recall | Full Entity Coverage | Oracle Macro $F_{0.5}$ |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for cap in cap_results:
            f.write(f"| `{cap['cap']}` | {cap['link_recall']*100:.3f}% | {cap['full_entity_coverage']*100:.3f}% | **{cap['oracle_macro_f05']:.6f}** |\n")

        f.write("\n## 5. Miss Analysis Breakdown\n\n")
        f.write(f"- **Total Missed Links:** {len(missed_links):,} ({len(missed_links)/total_val_true_links*100:.3f}% miss rate)\n")
        f.write(f"- **Miss Analysis CSV:** `{miss_csv_path}`\n\n")
        f.write("| Miss Category | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for cat, cnt in sorted(miss_category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| `{cat}` | {cnt:,} | {cnt/len(missed_links)*100:.2f}% |\n")

        f.write("\n## 6. Subgroup Diagnostics\n\n")
        f.write("### Country Breakdown\n\n")
        for c, st in oracle_eval_results.get("country_breakdown", {}).items():
            f.write(f"- **{c}** ({st['entity_count']:,} entities): Oracle Macro $F_{{0.5}} = {st['macro_f05']:.6f}$, Precision = {st['macro_precision']:.6f}, Recall = {st['macro_recall']:.6f}\n")

        f.write("\n### Source Diagnostics\n\n")
        s_diag = oracle_eval_results.get("source_diagnostics", {})
        f.write(f"- **Source 2 Links**: Retrieved = {s_diag['S2']['tp_links']:,} / {s_diag['S2']['true_links']:,} ({s_diag['S2']['link_recall']*100:.2f}% recall)\n")
        f.write(f"- **Source 3 Links**: Retrieved = {s_diag['S3']['tp_links']:,} / {s_diag['S3']['true_links']:,} ({s_diag['S3']['link_recall']*100:.2f}% recall)\n")

    print(f"\n[VALIDATION MEASUREMENT] Benchmark Complete!")
    print(f"  True Link Recall:     {link_recall*100:.3f}% ({retrieved_true_links:,} / {total_val_true_links:,})")
    print(f"  Full Entity Coverage: {full_coverage_rate*100:.3f}% ({full_coverage_entities:,} / {non_singleton_entities:,})")
    print(f"  Oracle Macro F0.5:    {oracle_eval_results['macro_f05']:.6f}")
    print(f"  Mean Candidates:      {cand_stats['mean']:.2f} (Median: {cand_stats['median']:.1f}, Max: {cand_stats['max']:,})")
    print(f"  Total Candidates:     {total_candidates:,}")
    print(f"  Reduction Ratio:      {reduction_ratio*100:.6f}%")
    print(f"  Reports Saved To:     {md_path}")

    # 9. Register into experiments/experiment_log.csv
    log_path = "experiments/experiment_log.csv"
    if os.path.exists(log_path):
        import subprocess
        try:
            commit_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            commit_sha = "unknown"

        exp_row = {
            "experiment_id": experiment_id,
            "owner": "team",
            "status": "COMPLETED",
            "branch": "phase2/blocking-baseline",
            "hypothesis": "Multi-channel blocker union (exact name, compact/domain, rare token DF, address anchors, transliteration) achieves high recall ceiling under country partitioning.",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "completed_at": summary["timestamp"],
            "git_commit": commit_sha,
            "validation_split": "val_50k_seed42",
            "blocker_config": "configs/blocking/blocking_v01.yaml",
            "feature_config": "none",
            "negative_sampling": "none",
            "model": "candidate_blocker_oracle",
            "hyperparameters": json.dumps({"token_max_doc_freq": 3, "token_min_len": 5, "channels": ["A", "B", "C", "D", "E"]}),
            "threshold_config": "{}",

            "candidate_recall": round(link_recall, 6),
            "full_entity_coverage": round(full_coverage_rate, 6),
            "oracle_f05": round(oracle_eval_results["macro_f05"], 6),
            "validation_macro_f05": 0.0,  # Explicitly 0.0 / not deployed model
            "singleton_accuracy": 1.0,
            "precision": round(oracle_eval_results["macro_precision"], 6),
            "recall": round(oracle_eval_results["macro_recall"], 6),
            "runtime_sec": round(total_runtime, 2),
            "memory_peak_mb": round(peak_ram, 2),
            "is_current_best": False,  # Blocker candidate ceiling, not deployed model
            "notes": f"EXP_001 Blocker baseline. True-link recall: {link_recall*100:.2f}%, Oracle F0.5: {oracle_eval_results['macro_f05']:.4f}, Mean cands: {cand_stats['mean']:.1f}. Misses exported.",
            "artifact_paths": f"{md_path};{miss_csv_path};{json_path}"
        }

        # Check if experiment_id already exists in log
        exp_df = pd.read_csv(log_path)
        if experiment_id in exp_df["experiment_id"].values:
            exp_df = exp_df[exp_df["experiment_id"] != experiment_id]
        new_row_df = pd.DataFrame([exp_row])
        exp_df = pd.concat([exp_df, new_row_df], ignore_index=True)
        exp_df.to_csv(log_path, index=False)
        print(f"Logged {experiment_id} into {log_path} successfully!", flush=True)

    return summary


if __name__ == "__main__":
    run_benchmark()
