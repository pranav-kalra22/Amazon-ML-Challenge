# AWS Team Handoff Guide — EXP_004 Pilot Execution

## Target Repository State

- **Stable branch:** `main`
- **AWS baseline Git commit:** `81589fca6190b63ad93dbf751ab0074089fb71cc`
- **Safety Pre-cloud Tag:** `exp004-precloud-ready` (points to `da47cdd431487a69aa5062fc9fb0191d7d363db8`)
- **AWS Baseline Tag:** `exp004-aws-baseline` (points to `81589fca6190b63ad93dbf751ab0074089fb71cc`)
- **Experiment:** `EXP_004` (`EXP_004_cloud_ngram_tfidf_blocker`)

---

## Directives & Prohibitions

> [!CAUTION]
> **STRICT COMPETITION & OPERATIONAL DIRECTIVES:**
> - **First cloud action:** `India x S2` pilot ONLY.
> - **Do NOT** run full EXP_004 before pilot review and verification.
> - **Do NOT** inspect or touch the validation holdout split (`artifacts/splits/holdout_s1_ids_50k_seed2026.parquet`).
> - **Do NOT** train any matcher (LightGBM, XGBoost, CatBoost, embeddings, cross-encoders).
> - **Do NOT** modify thresholds or hyperparameter sweeps during the authoritative benchmark run.
> - **Do NOT** commit credentials, API keys, or PATs to GitHub or command logs.

---

## Step-by-Step Pilot Instructions

### 1. Clone & Pin Repository

In the SageMaker notebook terminal (`notebook-al2023-v1` on `ml.m5.4xlarge` or `ml.m6i.4xlarge`):

```bash
cd /home/ec2-user/SageMaker
git clone https://github.com/pranav-kalra22/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge

# Pin to the exact audited AWS baseline
git checkout exp004-aws-baseline
git status
```

### 2. Environment Setup

```bash
conda create -n amz_er python=3.12 -y
conda activate amz_er
pip install -r requirements-cloud.txt
pytest -v
```
*Verify that all 46 tests pass before proceeding.*

### 3. Stage Training Dataset (S3 or Git LFS)

**Option A: Sync from S3 (Recommended for Cloud)**
```bash
export S3_BUCKET="<YOUR_AUTHORIZED_BUCKET>"
mkdir -p dataset/train
aws s3 sync s3://${S3_BUCKET}/amazon-ml-challenge/dataset/ dataset/train/
ls -lh dataset/train/
```

**Option B: Unpack directly from Git LFS zip**
If `git lfs pull` was used during clone:
```bash
unzip -q 6ab10eb3b23ba_student_resource.zip
ls -lh dataset/train/
```

### 4. Execute Pilot Run (India × S2 Only)

Run the exact authoritative pilot command:

```bash
python src/blocking/cloud_benchmark.py \
  --config configs/blocking/blocking_v04_cloud_tfidf.yaml \
  --dataset-root dataset/train \
  --val-split-path artifacts/splits/val_s1_ids_50k_seed42.parquet \
  --mode pilot \
  --pilot-country India \
  --pilot-source S2 \
  --experiment-id EXP_004_pilot_india_s2
```

### 5. Pilot Review Gate

Before provisioning further compute, verify:
1. Peak RSS remains under 35 GiB.
2. Sparse retrieval throughput exceeds 2,000 queries/second.
3. Candidate matrix generation completes with zero dense conversion warnings.
4. Report output is generated in `reports/blocking/EXP_004_pilot_india_s2_report.md`.

Once reviewed and approved by the team, proceed to the full benchmark following [EXP_004_RUNBOOK.md](file:///c:/Users/acer/Desktop/Amazon%20ML%20Challenge/docs/aws/EXP_004_RUNBOOK.md).
