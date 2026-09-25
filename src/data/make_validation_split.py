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
    
    # Identify singletons
    gt["is_singleton"] = gt["matched_entity_ids"].isna() | (gt["matched_entity_ids"].astype(str).str.strip() == "")
    singleton_map = dict(zip(gt["source1_entity_id"], gt["is_singleton"]))
    
    s1["is_singleton"] = s1["entity_id"].map(lambda x: singleton_map.get(x, True))
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


if __name__ == "__main__":
    create_validation_split()
