# Amazon ML Challenge 2026 — EXP_004 AWS SageMaker Runbook

> **CRITICAL COST SAFETY DIRECTIVE:**
> **STOP THE SAGEMAKER NOTEBOOK INSTANCE IMMEDIATELY AFTER THE EXPERIMENT COMPLETES.**
> Do NOT create persistent SageMaker endpoints, DynamoDB tables, continuous streaming jobs, or GPU instances for EXP_004.
> High-memory CPU compute is sufficient.

---

## 1. Cloud Architecture & Resource Sizing

| Parameter | Specification | Rationale |
| :--- | :--- | :--- |
| **Instance Type** | `ml.m5.4xlarge` or `m6i.4xlarge` | 16 vCPUs, 64 GiB RAM provides comfortable headroom for fitting character TF-IDF and multi-threaded sparse top-N multiplication. |
| **Attached Storage** | 150 GiB gp3 EBS | Holds training raw TSVs (~1.3 GB), validation splits, vectorizer matrix caches (~5–10 GB), and candidate diagnostics. |
| **Target Python Version**| Python 3.10, 3.11, or 3.12 | Fully supported by `sparse_dot_topn` 1.2.0 manylinux wheels. |
| **S3 Prefix Structure** | `s3://<BUCKET>/amazon-ml-challenge/` | Clean separation between dataset, experiment outputs, and caches. |

### Authoritative S3 Layout
```
s3://<BUCKET>/amazon-ml-challenge/
├── dataset/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
├── experiments/
│   └── EXP_004/
│       ├── reports/
│       │   ├── EXP_004_cloud_ngram_tfidf_blocker_pilot_summary.json
│       │   ├── EXP_004_cloud_ngram_tfidf_blocker_pilot_report.md
│       │   ├── EXP_004_cloud_ngram_tfidf_blocker_full_summary.json
│       │   └── EXP_004_cloud_ngram_tfidf_blocker_full_report.md
│       └── logs/
└── cache/
    └── tfidf/
        ├── us_s2_g_name_cand_mat.npz
        ├── india_s2_g_name_cand_mat.npz
        └── ...
```

---

## 2. Environment Setup in SageMaker Terminal

Open a terminal inside your SageMaker Notebook or Studio environment:

```bash
# 1. Update git and navigate to SageMaker home
cd /home/ec2-user/SageMaker

# 2. Clone repository
git clone https://github.com/pranav-kalra22/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge

# 3. Checkout active development branch
git checkout phase2/blocking-baseline
git status

# 4. Activate or create Python 3.12 / 3.10 virtual environment
conda create -n amz_er python=3.12 -y
conda activate amz_er

# 5. Install audited cloud dependencies
pip install -r requirements-cloud.txt

# 6. Verify automated test suite passes before running
pytest -v
```

---

## 3. Dataset Staging from S3

Stage the competition datasets onto the local NVMe/EBS disk (avoids network I/O during candidate scanning):

```bash
# Replace <YOUR_BUCKET> with your authorized bucket name
export S3_BUCKET="<YOUR_BUCKET>"

mkdir -p dataset/train

# Sync training datasets
aws s3 sync s3://${S3_BUCKET}/amazon-ml-challenge/dataset/ dataset/train/

# Confirm dataset presence and integrity
ls -lh dataset/train/
```

---

## 4. EXP_004 Cloud Phase 1: Pilot Run

Before running the full 10.32M candidate population, execute the **Pilot** on the `India x S2` partition to profile memory footprint, vocabulary dimensionality, vectorizer fit duration, and sparse cosine throughput:

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

### Pilot Decision Gate Checklist:
1. Did peak RSS stay under 35 GiB?
2. Did sparse top-N throughput exceed 2,000 queries/second?
3. Did the sparse candidate matrix load without any dense memory warnings?
4. If all checks pass, proceed immediately to Phase 2.

---

## 5. EXP_004 Cloud Phase 2: Full Validation Run

Execute the complete development-validation candidate generation across all country partitions (`US x S2`, `US x S3`, `India x S2`, `India x S3`):

```bash
python src/blocking/cloud_benchmark.py \
  --config configs/blocking/blocking_v04_cloud_tfidf.yaml \
  --dataset-root dataset/train \
  --val-split-path artifacts/splits/val_s1_ids_50k_seed42.parquet \
  --mode full \
  --experiment-id EXP_004_cloud_ngram_tfidf_blocker
```

---

## 6. Upload Results & Diagnostics to S3

Upload benchmark artifacts, summary metrics, and error analysis CSVs to permanent S3 storage:

```bash
aws s3 sync reports/blocking/ s3://${S3_BUCKET}/amazon-ml-challenge/experiments/EXP_004/reports/
aws s3 sync cache/tfidf/ s3://${S3_BUCKET}/amazon-ml-challenge/cache/tfidf/
```

---

## 7. Instance Shutdown (Mandatory)

After copying the summary JSON and markdown reports back to your local repository or pushing results commits:

1. In the AWS SageMaker Console, select the notebook instance.
2. Click **Actions** -> **Stop**.
3. Confirm status changes to **Stopped** to avoid incurring idle hourly compute charges.
