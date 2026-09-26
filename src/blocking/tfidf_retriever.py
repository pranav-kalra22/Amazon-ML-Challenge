#!/usr/bin/env python3
"""
Scalable Character N-Gram TF-IDF Sparse Top-N Approximate Retriever — EXP_004.

Implements character n-gram cosine retrieval over sparse CSR matrices using
`sparse_dot_topn.sp_matmul_topn` partitioned strictly by country x candidate source.

Retrieval Channels:
- G_NAME_CHAR_TFIDF (bit 64): Character TF-IDF over business name representations.
- G_ADDRESS_CHAR_TFIDF (bit 128): Character TF-IDF over normalized address representations (preserving digits/locality).
- G_TRANSLITERATED_NAME_CHAR_TFIDF (bit 256): Character TF-IDF over symmetric transliterated names.

Guarantees:
1. Strict open-set country isolation (no cross-country retrieval; works for US, India, France, etc.).
2. Zero dense matrix materialization (all operations in float32 sparse CSR).
3. Memory-safe batch query processing and proactive memory safety checks.
4. Channel bitmask attribution and seamless union with EXP_003 lexical candidates.
5. Deterministic cache fingerprinting, metadata validation, and row-to-entity-ID mapping assertion.
6. Similarity metadata retention for downstream ranking and feature engineering.
"""

import os
import sys
sys.path.insert(0, ".")
import gc
import time
import math
import json
import heapq
import hashlib
import subprocess
import joblib
import yaml
import psutil
import numpy as np
from scipy import sparse
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any, Optional, Iterator

from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn as sdt

from src.blocking.normalizer import (
    normalize_name_non_destructive,
    normalize_address_non_destructive
)
from src.blocking.blocker import (
    CH_A, CH_B, CH_C2, CH_D2, CH_E2, CH_F,
    CH_G_NAME, CH_G_ADDR, CH_G_TRANS,
    CHANNEL_NAMES,
    CandidateHeapItem,
    StreamingCandidateResult
)


def get_git_commit_sha() -> str:
    """Returns current git commit SHA or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def compute_dataset_fingerprint(candidate_records: List[Dict[str, Any]]) -> str:
    """
    Computes deterministic signature of the candidate dataset partition.
    """
    n = len(candidate_records)
    if n == 0:
        return "empty"
    sig = f"rows_{n}_first_{candidate_records[0].get('entity_id')}_mid_{candidate_records[n//2].get('entity_id')}_last_{candidate_records[-1].get('entity_id')}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def compute_config_fingerprint(
    country: str,
    source: str,
    channel_name: str,
    channel_cfg: Dict[str, Any],
    dataset_fingerprint: str,
    norm_version: str = "v1.0"
) -> str:
    """
    Computes a cryptographic fingerprint incorporating all vectorizer parameters,
    normalization version, and dataset identity.
    """
    canonical_dict = {
        "country": str(country),
        "source": str(source),
        "channel_name": str(channel_name),
        "representations": sorted(channel_cfg.get("representations", ["norm_unicode"])),
        "analyzer": channel_cfg.get("analyzer", "char_wb"),
        "ngram_min": channel_cfg.get("ngram_min", 3),
        "ngram_max": channel_cfg.get("ngram_max", 5),
        "min_df": channel_cfg.get("min_df", 2),
        "max_df": channel_cfg.get("max_df", 0.98),
        "max_features": channel_cfg.get("max_features", None),
        "sublinear_tf": channel_cfg.get("sublinear_tf", True),
        "norm": channel_cfg.get("norm", "l2"),
        "dtype": channel_cfg.get("dtype", "float32"),
        "norm_version": norm_version,
        "dataset_fingerprint": dataset_fingerprint
    }
    encoded = json.dumps(canonical_dict, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class TfidfApproximateRetriever:
    """
    Partitioned Character N-Gram TF-IDF Approximate Nearest-Neighbor Retriever.
    Uses sparse matrix multiplication with bounded Top-K retrieval via sparse_dot_topn.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            default_cfg_path = "configs/blocking/blocking_v04_cloud_tfidf.yaml"
            if os.path.exists(default_cfg_path):
                with open(default_cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}

        # Parsing execution settings
        exec_cfg = self.config.get("execution", {})
        self.batch_size = exec_cfg.get("batch_size", 5000)
        self.n_threads = exec_cfg.get("n_threads", 8)
        self.cache_dir = exec_cfg.get("cache_dir", "cache/tfidf")
        mem_cfg = exec_cfg.get("memory_safety", {})
        self.max_ram_gb = mem_cfg.get("max_ram_gb", 55.0)
        self.abort_on_limit = mem_cfg.get("abort_on_limit", True)

        # Parsing TF-IDF channel settings
        tfidf_cfg = self.config.get("tfidf_retrieval", {})
        self.tfidf_enabled = tfidf_cfg.get("enabled", True)

        self.name_cfg = tfidf_cfg.get("channel_G_name_char_tfidf", {})
        self.addr_cfg = tfidf_cfg.get("channel_G_address_char_tfidf", {})
        self.trans_cfg = tfidf_cfg.get("channel_G_transliterated_char_tfidf", {})

        # Ranking settings
        ranking_cfg = self.config.get("candidate_ranking", {})
        self.weights = ranking_cfg.get("channel_weights", {
            "CH_A": 10.0, "CH_B": 8.0, "CH_C2": 5.0, "CH_D2": 6.0, "CH_E2": 7.0, "CH_F": 5.0,
            "CH_G_NAME": 4.0, "CH_G_ADDR": 4.0, "CH_G_TRANS": 4.0
        })
        self.multi_channel_bonus = ranking_cfg.get("multi_channel_bonus", 2.0)

    def check_memory_headroom(self, required_mb: float = 500.0) -> bool:
        """
        Verifies available system memory and process RSS before large allocations.
        """
        proc = psutil.Process()
        proc_rss_gb = proc.memory_info().rss / (1024 ** 3)
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        required_gb = required_mb / 1024.0

        if available_gb < required_gb:
            msg = (
                f"[MEMORY SAFETY ABORT] Insufficient available RAM! Required: {required_gb:.2f} GB, "
                f"Available: {available_gb:.2f} GB."
            )
            if self.abort_on_limit:
                raise MemoryError(msg)
            else:
                print(f"WARNING: {msg}", flush=True)
                return False

        if proc_rss_gb + required_gb > self.max_ram_gb:
            msg = (
                f"[MEMORY SAFETY ABORT] Process RSS ({proc_rss_gb:.2f} GB) + Required ({required_gb:.2f} GB) "
                f"exceeds process safety threshold of {self.max_ram_gb:.2f} GB!"
            )
            if self.abort_on_limit:
                raise MemoryError(msg)
            else:
                print(f"WARNING: {msg}", flush=True)
                return False

        return True

    def build_vectorizer(self, channel_cfg: Dict[str, Any]) -> TfidfVectorizer:
        """
        Builds a scikit-learn TfidfVectorizer configured for float32 character n-grams.
        """
        analyzer = channel_cfg.get("analyzer", "char_wb")
        ngram_min = channel_cfg.get("ngram_min", 3)
        ngram_max = channel_cfg.get("ngram_max", 5)
        min_df = channel_cfg.get("min_df", 2)
        max_df = channel_cfg.get("max_df", 0.98)
        max_features = channel_cfg.get("max_features", None)
        sublinear_tf = channel_cfg.get("sublinear_tf", True)
        norm = channel_cfg.get("norm", "l2")

        return TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            norm=norm,
            dtype=np.float32
        )

    def extract_entity_texts(
        self,
        records: List[Dict[str, Any]],
        channel_type: str,
        representations: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extracts representative string corpus for entities depending on channel.
        - 'name': raw/casefolded, legal-stripped, compact
        - 'address': normalized address preserving digits, building numbers, locality
        - 'transliterated': symmetric transliterated ASCII name
        """
        texts = []
        representations = representations or ["norm_unicode"]

        for rec in records:
            raw_name = rec.get("business_name", "")
            raw_addr = rec.get("business_address", "")

            if channel_type == "name":
                n_norm = normalize_name_non_destructive(raw_name)
                parts = []
                for rep in representations:
                    val = n_norm.get(rep, "")
                    if val and val not in parts:
                        parts.append(val)
                text = " ".join(parts) if parts else (n_norm.get("norm_unicode") or "")
                texts.append(text)

            elif channel_type == "address":
                a_norm = normalize_address_non_destructive(raw_addr)
                val = a_norm.get("norm_unicode", "")
                texts.append(val)

            elif channel_type == "transliterated":
                n_norm = normalize_name_non_destructive(raw_name)
                val = n_norm.get("transliterated", "")
                texts.append(val)

            else:
                texts.append(str(raw_name))

        return texts

    def retrieve_partition_approximate_candidates(
        self,
        country: str,
        source: str,
        channel_bit: int,
        channel_cfg: Dict[str, Any],
        channel_type: str,
        s1_entities: List[Dict[str, Any]],
        s1_indices: List[int],
        candidate_records: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_cache: bool = True
    ) -> List[Tuple[int, str, int, float]]:
        """
        Fits/transforms candidate partition and performs sparse Top-K cosine retrieval
        for S1 entities within the specified country.

        Returns list of tuples: (s1_idx, cand_id, channel_bit, cosine_similarity)
        """
        if not s1_entities or not candidate_records:
            return []

        effective_k = top_k if top_k is not None else channel_cfg.get("retrieval_top_k", 50)
        effective_thresh = similarity_threshold if similarity_threshold is not None else channel_cfg.get("similarity_threshold", 0.35)
        reps = channel_cfg.get("representations", ["norm_unicode"])

        # Pre-allocation memory check
        n_cands = len(candidate_records)
        n_queries = len(s1_entities)
        est_mem_mb = (n_cands + n_queries) * 0.05
        self.check_memory_headroom(est_mem_mb)

        t_start = time.time()
        ch_name = CHANNEL_NAMES.get(channel_bit, f"CH_{channel_bit}")
        print(f"\n[TF-IDF RETRIEVAL] Partition: Country='{country}' x Source='{source}' | Channel={ch_name}", flush=True)
        print(f"  S1 Queries: {n_queries:,} | Candidate Corpus: {n_cands:,} | Top-K: {effective_k} | Thresh: {effective_thresh}", flush=True)

        input_cand_ids = [r["entity_id"] for r in candidate_records]

        # 1. Deterministic cache fingerprinting
        dataset_fp = compute_dataset_fingerprint(candidate_records)
        config_fp = compute_config_fingerprint(country, source, ch_name, channel_cfg, dataset_fp)

        cache_prefix = f"{country}_{source}_{ch_name}_{config_fp}".replace(" ", "_").lower()
        vec_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_vec.joblib")
        mat_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_cand_mat.npz")
        cand_ids_path = os.path.join(self.cache_dir, f"{cache_prefix}_cand_ids.joblib")
        meta_cache_path = os.path.join(self.cache_dir, f"{cache_prefix}_metadata.json")

        vectorizer = None
        cand_csr = None
        cand_ids = None

        if use_cache and os.path.exists(meta_cache_path) and os.path.exists(vec_cache_path) and os.path.exists(mat_cache_path) and os.path.exists(cand_ids_path):
            try:
                with open(meta_cache_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Validate fingerprints
                if meta.get("config_fingerprint") != config_fp or meta.get("dataset_fingerprint") != dataset_fp:
                    print(f"  [CACHE INVALIDATION] Metadata fingerprint mismatch! Rebuilding cache.", flush=True)
                else:
                    cached_cand_ids = joblib.load(cand_ids_path)
                    cached_mat = sparse.load_npz(mat_cache_path)

                    # Strict row count assertion
                    if len(cached_cand_ids) != cached_mat.shape[0]:
                        raise ValueError(f"Cache corruption: {len(cached_cand_ids)} candidate IDs != {cached_mat.shape[0]} matrix rows!")

                    # Compare candidate IDs set to verify dataset identity
                    if set(cached_cand_ids) != set(input_cand_ids):
                        print(f"  [CACHE INVALIDATION] Candidate ID set mismatch. Invalidation triggered.", flush=True)
                    else:
                        # Authoritative reuse: Row col_j maps strictly to cached_cand_ids[col_j]
                        cand_ids = cached_cand_ids
                        cand_csr = cached_mat
                        vectorizer = joblib.load(vec_cache_path)
                        print(f"  Loaded validated cache [{config_fp}]: matrix {cand_csr.shape} ({cand_csr.nnz:,} non-zeros)", flush=True)
            except Exception as e:
                print(f"  Notice: Failed loading cache ({e}). Rebuilding from scratch...", flush=True)
                vectorizer = None
                cand_csr = None
                cand_ids = None

        if vectorizer is None or cand_csr is None or cand_ids is None:
            # 2. Extract candidate texts and fit vectorizer
            cand_texts = self.extract_entity_texts(candidate_records, channel_type, reps)
            cand_ids = input_cand_ids
            vectorizer = self.build_vectorizer(channel_cfg)

            t_fit = time.time()
            cand_csr = vectorizer.fit_transform(cand_texts)
            if cand_csr.dtype != np.float32:
                cand_csr = cand_csr.astype(np.float32)
            if not isinstance(cand_csr, sparse.csr_matrix):
                cand_csr = cand_csr.tocsr()

            # Ensure row count match
            assert len(cand_ids) == cand_csr.shape[0], f"Candidate ID count {len(cand_ids)} != matrix rows {cand_csr.shape[0]}"

            fit_time = time.time() - t_fit
            vocab_size = len(vectorizer.vocabulary_)
            ram_mb = (cand_csr.data.nbytes + cand_csr.indices.nbytes + cand_csr.indptr.nbytes) / (1024 * 1024)
            print(f"  Fitted vectorizer in {fit_time:.2f}s | Vocab: {vocab_size:,} features | Shape: {cand_csr.shape} | NNZ: {cand_csr.nnz:,} | Matrix RAM: {ram_mb:.2f} MB", flush=True)

            # Save validated cache
            if use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                try:
                    joblib.dump(vectorizer, vec_cache_path)
                    sparse.save_npz(mat_cache_path, cand_csr)
                    joblib.dump(cand_ids, cand_ids_path)
                    meta = {
                        "config_fingerprint": config_fp,
                        "code_commit": get_git_commit_sha(),
                        "country": country,
                        "source": source,
                        "channel_name": ch_name,
                        "dataset_fingerprint": dataset_fp,
                        "row_count": len(cand_ids),
                        "candidate_id_count": len(cand_ids),
                        "matrix_shape": list(cand_csr.shape),
                        "nnz": int(cand_csr.nnz),
                        "dtype": str(cand_csr.dtype),
                        "vocabulary_size": vocab_size,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    with open(meta_cache_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    print(f"  Saved validated cache with fingerprint [{config_fp}].", flush=True)
                except Exception as e:
                    print(f"  Notice: Cache write skipped ({e}).", flush=True)

        # 3. Transpose candidate matrix for dot product: Q @ C.T
        cand_csr_T = cand_csr.T.tocsr()
        if cand_csr_T.dtype != np.float32:
            cand_csr_T = cand_csr_T.astype(np.float32)

        # 4. Extract S1 query texts
        s1_texts = self.extract_entity_texts(s1_entities, channel_type, reps)

        # 5. Query batching to keep memory tightly bounded
        matches: List[Tuple[int, str, int, float]] = []
        n_batches = math.ceil(n_queries / self.batch_size)
        t_query_start = time.time()

        for b_idx in range(n_batches):
            b_start = b_idx * self.batch_size
            b_end = min(b_start + self.batch_size, n_queries)
            batch_texts = s1_texts[b_start:b_end]
            batch_indices = s1_indices[b_start:b_end]

            Q_batch = vectorizer.transform(batch_texts)
            if Q_batch.dtype != np.float32:
                Q_batch = Q_batch.astype(np.float32)
            if not isinstance(Q_batch, sparse.csr_matrix):
                Q_batch = Q_batch.tocsr()

            # Sparse top-N multiplication
            sim_csr = sdt.sp_matmul_topn(
                Q_batch,
                cand_csr_T,
                top_n=effective_k,
                threshold=effective_thresh,
                sort=True,
                n_threads=self.n_threads
            )

            # Collect matches directly from sparse representation
            indptr = sim_csr.indptr
            indices = sim_csr.indices
            data = sim_csr.data

            for row_i in range(len(batch_texts)):
                s1_idx = batch_indices[row_i]
                start_p = indptr[row_i]
                end_p = indptr[row_i + 1]

                for p in range(start_p, end_p):
                    col_j = indices[p]
                    score = float(data[p])
                    cand_id = cand_ids[col_j]
                    matches.append((s1_idx, cand_id, channel_bit, score))

            del Q_batch, sim_csr
            if b_idx % 5 == 0 and b_idx > 0:
                gc.collect()

        del cand_csr_T, cand_csr, vectorizer
        gc.collect()

        total_retrieval_time = time.time() - t_start
        rate = n_queries / max(0.001, time.time() - t_query_start)
        print(f"  Retrieval finished: {len(matches):,} matches in {total_retrieval_time:.2f}s ({rate:.0f} queries/s)", flush=True)

        return matches

    def merge_tfidf_candidates_into_result(
        self,
        base_result: StreamingCandidateResult,
        tfidf_matches: List[Tuple[int, str, int, float]],
        gt_links_set: Optional[Set[Tuple[int, str]]] = None,
        effective_k: Optional[int] = 2000
    ) -> StreamingCandidateResult:
        """
        Merges approximate retrieval candidates into an existing StreamingCandidateResult:
        1. Bitwise OR channel attribution for duplicate candidates.
        2. Preserves continuous cosine similarity scores for downstream ML feature engineering.
        3. Mode A ground truth hit tracking.
        4. Mode B bounded Top-K evidence heaps.
        """
        print(f"\n[CANDIDATE UNION] Merging {len(tfidf_matches):,} TF-IDF matches into candidate result...", flush=True)
        t0 = time.time()

        candidate_dicts = list(base_result)
        heaps = base_result.heaps
        gt_hits = base_result.gt_hits
        uncapped_counts = base_result.uncapped_counts
        uncapped_channel_counts = base_result.uncapped_channel_counts
        similarities = base_result.similarities

        # Ensure channel count array exists for TF-IDF channels
        for ch_bit in (CH_G_NAME, CH_G_ADDR, CH_G_TRANS):
            if ch_bit not in uncapped_channel_counts:
                uncapped_channel_counts[ch_bit] = np.zeros(len(candidate_dicts), dtype=np.int32)

        new_pairs_added = 0
        existing_pairs_enriched = 0

        for s1_idx, cand_id, ch_bit, sim_score in tfidf_matches:
            cand_dict = candidate_dicts[s1_idx]
            uncapped_counts[s1_idx] += 1
            uncapped_channel_counts[ch_bit][s1_idx] += 1

            # Retain continuous cosine similarity values
            sim_map = similarities[s1_idx].setdefault(cand_id, {})
            if ch_bit == CH_G_NAME:
                sim_map["name"] = max(sim_map.get("name", 0.0), float(sim_score))
            elif ch_bit == CH_G_ADDR:
                sim_map["address"] = max(sim_map.get("address", 0.0), float(sim_score))
            elif ch_bit == CH_G_TRANS:
                sim_map["transliterated"] = max(sim_map.get("transliterated", 0.0), float(sim_score))

            if cand_id in cand_dict:
                cand_dict[cand_id] |= ch_bit
                existing_pairs_enriched += 1
            else:
                cand_dict[cand_id] = ch_bit
                new_pairs_added += 1

            # Mode A ground truth hit tracking
            if gt_links_set is not None and (s1_idx, cand_id) in gt_links_set:
                gt_hits[(s1_idx, cand_id)] = gt_hits.get((s1_idx, cand_id), 0) | ch_bit

            # Mode B bounded heap updating (incorporating continuous similarity)
            if heaps and s1_idx < len(heaps):
                bitmask = cand_dict[cand_id]
                score = self.compute_candidate_evidence_score(bitmask, similarities=sim_map)
                item = CandidateHeapItem(score, cand_id, bitmask, similarities=dict(sim_map))
                h = heaps[s1_idx]

                found = False
                for heap_elem in h:
                    if heap_elem.cand_id == cand_id:
                        heap_elem.bitmask = bitmask
                        heap_elem.score = score
                        heap_elem.similarities = dict(sim_map)
                        found = True
                        break

                if found:
                    heapq.heapify(h)
                else:
                    if effective_k is None or len(h) < effective_k:
                        heapq.heappush(h, item)
                    elif item > h[0]:
                        heapq.heapreplace(h, item)

        print(f"Candidate union complete in {time.time()-t0:.2f}s: {new_pairs_added:,} new pairs, {existing_pairs_enriched:,} enriched pairs.", flush=True)

        return StreamingCandidateResult(
            candidate_dicts=candidate_dicts,
            heaps=heaps,
            gt_hits=gt_hits,
            uncapped_counts=uncapped_counts,
            uncapped_channel_counts=uncapped_channel_counts,
            total_scanned=base_result.total_scanned,
            similarities=similarities
        )

    def compute_candidate_evidence_score(
        self,
        bitmask: int,
        similarities: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Evidence scoring with EXP_003 lexical channels and EXP_004 TF-IDF channels.
        Distinguishes high vs low cosine similarities (e.g. cosine .98 vs .35).
        """
        score = 0.0
        channels_active = 0
        sims = similarities or {}

        for b, name, key in (
            (CH_A, "CH_A", None), (CH_B, "CH_B", None), (CH_C2, "CH_C2", None),
            (CH_D2, "CH_D2", None), (CH_E2, "CH_E2", None), (CH_F, "CH_F", None),
            (CH_G_NAME, "CH_G_NAME", "name"), (CH_G_ADDR, "CH_G_ADDR", "address"),
            (CH_G_TRANS, "CH_G_TRANS", "transliterated")
        ):
            if bitmask & b:
                base_w = self.weights.get(name, 4.0)
                if key and key in sims:
                    score += base_w * sims[key]
                else:
                    score += base_w
                channels_active += 1

        if channels_active > 1:
            score += (channels_active - 1) * self.multi_channel_bonus

        return score


def evaluate_parameter_sweep_in_memory(
    matches: List[Tuple[int, str, int, float]],
    val_gt_pairs: Set[Tuple[int, str]],
    lexical_hits: Set[Tuple[int, str]],
    total_true_links: int,
    k_list: List[int],
    thresh_list: List[float],
    val_gt: Optional[Dict[str, Set[str]]] = None,
    s1_id_list: Optional[List[str]] = None,
    metadata_map: Optional[Dict[str, Dict[str, str]]] = None,
    base_lexical_pairs: int = 0
) -> List[Dict[str, Any]]:
    """
    Evaluates Top-K and Similarity Threshold sweeps entirely in-memory from a single
    maximum-retrieval run (K=100, threshold=0.30).
    """
    # Group matches by (s1_idx, channel_bit) sorted by score descending
    grouped: Dict[Tuple[int, int], List[Tuple[str, float]]] = defaultdict(list)
    for s1_idx, cid, ch_bit, sc in matches:
        grouped[(s1_idx, ch_bit)].append((cid, sc))

    for k_key in grouped:
        grouped[k_key].sort(key=lambda x: -x[1])

    sweep_results = []
    total_lex_misses = len(val_gt_pairs - lexical_hits)

    for k in k_list:
        for thresh in thresh_list:
            # Filter matches for (k, thresh)
            filtered_approx_pairs: Set[Tuple[int, str]] = set()
            for (s1_idx, ch_bit), cand_list in grouped.items():
                count = 0
                for cid, sc in cand_list:
                    if sc >= thresh:
                        filtered_approx_pairs.add((s1_idx, cid))
                        count += 1
                        if count >= k:
                            break

            # Combine with lexical hits
            approx_hits = filtered_approx_pairs & val_gt_pairs
            union_hits = lexical_hits | approx_hits
            total_hits_count = len(union_hits)
            rec = total_hits_count / max(1, total_true_links)

            recovered_misses = (union_hits - lexical_hits)
            n_recovered = len(recovered_misses)
            pct_recovered = n_recovered / max(1, total_lex_misses)

            n_pairs = base_lexical_pairs + len(filtered_approx_pairs)
            cands_per_rec = len(filtered_approx_pairs) / max(1, n_recovered)

            row = {
                "top_k": k,
                "similarity_threshold": thresh,
                "true_links_hit": total_hits_count,
                "true_link_recall": round(rec, 6),
                "exp003_misses_recovered": n_recovered,
                "pct_misses_recovered": round(pct_recovered, 6),
                "total_candidate_pairs": n_pairs,
                "cands_per_recovered_link": round(cands_per_rec, 2)
            }

            # If full ground truth provided, compute Macro F0.5
            if val_gt is not None and s1_id_list is not None:
                oracle_preds = {}
                full_cov_count = 0
                non_singleton_count = 0
                for s1_idx, s1_id in enumerate(s1_id_list):
                    truth = val_gt.get(s1_id, set())
                    if truth:
                        non_singleton_count += 1
                    hits = {cid for cid in truth if (s1_idx, cid) in union_hits}
                    oracle_preds[s1_id] = hits
                    if truth and truth.issubset(hits):
                        full_cov_count += 1

                from src.evaluation.macro_f05 import evaluate_predictions
                sub_eval = evaluate_predictions(val_gt, oracle_preds, metadata_map or {})
                row["oracle_macro_f05"] = round(sub_eval.get("macro_f05", 0.0), 6)
                row["full_entity_coverage"] = round(full_cov_count / max(1, non_singleton_count), 6)

            sweep_results.append(row)

    return sweep_results
