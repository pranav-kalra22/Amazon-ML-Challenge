#!/usr/bin/env python3
"""
Creates a leakage-safe validation split partitioned strictly by Source 1 entity.
Preserves complete ground-truth match sets for every entity.
Stratifies by country and singleton status with fixed seed.
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_validation_split(
    s1_path: str = "dataset/train/train_source1.tsv",
    gt_path: str = "dataset/train/train_ground_truth.tsv",
    val_size: int = 50000,
    seed: int = 42,
    output_dir: str = "artifacts/splits"
):
    print(f"Loading Source 1 metadata from {s1_path}...")
    s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"])
    
    print(f"Loading Ground Truth from {gt_path}...")
    gt = pd.read_csv(gt_path, sep="\t")
    
    # Validate integrity between Source 1 and Ground Truth
    print("Validating Source 1 and Ground Truth 1-to-1 integrity...", flush=True)
    assert len(s1["entity_id"]) == s1["entity_id"].nunique(), "Integrity Error: Duplicate entity_id in Source 1"
    assert len(gt["source1_entity_id"]) == gt["source1_entity_id"].nunique(), "Integrity Error: Duplicate source1_entity_id in Ground Truth"
    
    s1_ids_set = set(s1["entity_id"])
    gt_s1_ids_set = set(gt["source1_entity_id"])
    
    missing_in_gt = s1_ids_set - gt_s1_ids_set
    assert len(missing_in_gt) == 0, f"Integrity Error: {len(missing_in_gt)} Source 1 IDs missing from Ground Truth"
    
    missing_in_s1 = gt_s1_ids_set - s1_ids_set
    assert len(missing_in_s1) == 0, f"Integrity Error: {len(missing_in_s1)} Ground Truth IDs missing from Source 1"
    print(f"Integrity verified: Exactly {len(s1):,} S1 IDs match 1-to-1 with Ground Truth with 0 duplicates.", flush=True)
    
    # Identify singletons
    gt["is_singleton"] = gt["matched_entity_ids"].isna() | (gt["matched_entity_ids"].astype(str).str.strip() == "")
    singleton_map = dict(zip(gt["source1_entity_id"], gt["is_singleton"]))
    
    # Explicit mapping without fallback; fail loudly if any ID is unmapped
    s1["is_singleton"] = s1["entity_id"].map(singleton_map)
    assert not s1["is_singleton"].isna().any(), "Integrity Error: Unmapped singleton status encountered"
    
    s1["strata"] = s1["country"].astype(str) + "_" + s1["is_singleton"].astype(str)
    
    print(f"Total Source 1 entities: {len(s1):,}")
    print("Strata distribution:")
    print(s1["strata"].value_counts(normalize=True))
    
    # Stratified split
    val_frac = val_size / len(s1)
    train_df, val_df = train_test_split(
        s1,
        test_size=val_frac,
        random_state=seed,
        stratify=s1["strata"]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    val_ids_path = os.path.join(output_dir, f"val_s1_ids_{val_size // 1000}k_seed{seed}.parquet")
    train_ids_path = os.path.join(output_dir, f"train_s1_ids_seed{seed}.parquet")
    meta_path = os.path.join(output_dir, "split_metadata.json")
    
    val_df[["entity_id", "country", "is_singleton"]].to_parquet(val_ids_path, index=False)
    train_df[["entity_id", "country", "is_singleton"]].to_parquet(train_ids_path, index=False)
    
    metadata = {
        "split_id": f"val_{val_size // 1000}k_seed{seed}",
        "seed": seed,
        "total_source1_entities": len(s1),
        "val_size": len(val_df),
        "train_size": len(train_df),
        "val_singletons": int(val_df["is_singleton"].sum()),
        "val_singleton_rate": float(val_df["is_singleton"].mean()),
        "val_country_distribution": val_df["country"].value_counts().to_dict(),
        "train_country_distribution": train_df["country"].value_counts().to_dict(),
        "val_ids_file": val_ids_path,
        "train_ids_file": train_ids_path
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n[VALIDATION MEASUREMENT] Created validation split:")
    print(f"  Validation entities: {len(val_df):,} (Singletons: {metadata['val_singletons']:,}, {metadata['val_singleton_rate']*100:.2f}%)")
    print(f"  Training entities:   {len(train_df):,}")
    print(f"  Saved metadata to:   {meta_path}")
    return metadata


def verify_canonical_split(
    s1_path: str = "dataset/train/train_source1.tsv",
    gt_path: str = "dataset/train/train_ground_truth.tsv",
    split_dir: str = "artifacts/splits"
):
    print("Verifying canonical validation split integrity...", flush=True)
    val_path = os.path.join(split_dir, "val_s1_ids_50k_seed42.parquet")
    train_path = os.path.join(split_dir, "train_s1_ids_seed42.parquet")
    meta_path = os.path.join(split_dir, "split_metadata.json")
    
    assert os.path.exists(val_path), f"Missing {val_path}"
    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"
    
    val_df = pd.read_parquet(val_path)
    train_df = pd.read_parquet(train_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    print(f"Loaded validation IDs: {len(val_df):,}")
    print(f"Loaded training IDs:   {len(train_df):,}")
    
    # Check disjointness
    val_ids = set(val_df["entity_id"])
    train_ids = set(train_df["entity_id"])
    overlap = val_ids.intersection(train_ids)
    assert len(overlap) == 0, f"Leakage Error: {len(overlap)} IDs appear in both train and validation!"
    assert len(val_ids) == len(val_df), "Integrity Error: Duplicate IDs in validation split!"
    assert len(train_ids) == len(train_df), "Integrity Error: Duplicate IDs in training split!"
    
    # Fast check S1 and GT matching
    print("Checking full Source 1 vs Ground Truth ID parity...", flush=True)
    s1_count = 0
    s1_id_set = set()
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            s1_id_set.add(parts[0])
            s1_count += 1
    assert s1_count == len(s1_id_set), "Duplicate entity_id in Source 1!"
    
    gt_count = 0
    gt_id_set = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            gt_id_set.add(parts[0])
            gt_count += 1
    assert gt_count == len(gt_id_set), "Duplicate source1_entity_id in Ground Truth!"
    assert s1_id_set == gt_id_set, "Mismatch between Source 1 IDs and Ground Truth IDs!"
    
    total_split_ids = val_ids.union(train_ids)
    assert total_split_ids == s1_id_set, "Train + Validation split does not partition entire Source 1 population!"
    
    print("\n[VALIDATION MEASUREMENT] Canonical split verified successfully:")
    print(f"  Total S1 Entities:     {s1_count:,} (100% accounted for)")
    print(f"  Validation Entities:   {len(val_df):,} (0 duplicates, 0 cross-split leakage)")
    print(f"  Training Entities:     {len(train_df):,}")
    print(f"  Validation Singletons: {val_df['is_singleton'].sum():,} ({val_df['is_singleton'].mean()*100:.2f}%)")
    print(f"  Split File Validated:  {val_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        verify_canonical_split()
    else:
        create_validation_split()

