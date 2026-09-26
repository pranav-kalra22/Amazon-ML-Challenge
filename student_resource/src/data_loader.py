"""
data_loader.py — Load and validate TSV source files and ground-truth labels.

All source files are tab-separated with columns:
    entity_id, business_name, business_address, country

Ground-truth file columns:
    source1_entity_id, matched_entity_ids
"""

import os
import pandas as pd
import numpy as np


def load_source(path: str) -> pd.DataFrame:
    """Load a single source TSV file and perform basic validation."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    expected_cols = {"entity_id", "business_name", "business_address", "country"}
    actual_cols = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(
            f"Missing columns in {path}. "
            f"Expected {expected_cols}, got {actual_cols}"
        )

    # Strip whitespace from all string fields
    for col in df.columns:
        df[col] = df[col].str.strip()

    print(f"  Loaded {path}: {len(df):,} records")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load the ground-truth TSV and parse matched_entity_ids into lists."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    expected_cols = {"source1_entity_id", "matched_entity_ids"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in {path}. Expected {expected_cols}")

    # Parse comma-separated IDs into lists; empty string → empty list
    df["matched_list"] = df["matched_entity_ids"].apply(
        lambda x: [s.strip() for s in x.split(",") if s.strip()] if x.strip() else []
    )
    df["num_matches"] = df["matched_list"].apply(len)

    print(f"  Loaded {path}: {len(df):,} S1 entities")
    print(f"    - Singletons (0 matches): {(df['num_matches'] == 0).sum():,}")
    print(f"    - 1 match:  {(df['num_matches'] == 1).sum():,}")
    print(f"    - 2 matches: {(df['num_matches'] == 2).sum():,}")
    print(f"    - 3+ matches: {(df['num_matches'] >= 3).sum():,}")

    return df


def load_train_data(data_dir: str):
    """Load all training sources and ground truth.

    Returns:
        s1, s2, s3: DataFrames for Source 1/2/3
        gt: Ground truth DataFrame with parsed match lists
    """
    print("Loading training data...")
    train_dir = os.path.join(data_dir, "train")
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    return s1, s2, s3, gt


def load_test_data(data_dir: str):
    """Load all test sources.

    Returns:
        s1, s2, s3: DataFrames for Source 1/2/3
    """
    print("Loading test data...")
    test_dir = os.path.join(data_dir, "test")
    s1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(test_dir, "test_source3.tsv"))
    return s1, s2, s3


def build_lookup(s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """Build a dict mapping entity_id → row dict for fast lookups during
    candidate generation and feature engineering."""
    lookup = {}
    for _, row in s2.iterrows():
        lookup[row["entity_id"]] = row.to_dict()
    for _, row in s3.iterrows():
        lookup[row["entity_id"]] = row.to_dict()
    return lookup
