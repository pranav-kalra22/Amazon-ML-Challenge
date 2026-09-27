"""
embeddings.py — Sentence-transformer based semantic similarity for entity resolution.

Provides:
    1. Embedding engine: encode business names into dense vectors
    2. Semantic blocking: retrieve top-K similar candidates via cosine similarity
    3. Pairwise embedding cosine similarity as a feature for the ML model

Uses the 'all-MiniLM-L6-v2' model by default (fast, 384-dim, ~80MB).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class EmbeddingEngine:
    """Sentence-transformer embedding engine for business name similarity.

    Encodes normalized business names into dense vectors and provides:
        - Fast cosine similarity between any two entities
        - Semantic blocking: top-K retrieval for candidate generation
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256):
        """Initialize the embedding engine.

        Args:
            model_name: HuggingFace sentence-transformer model name.
            batch_size: Batch size for encoding.
        """
        from sentence_transformers import SentenceTransformer

        print(f"  Loading sentence-transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embeddings = {}       # entity_id → np.ndarray (normalized)
        self.entity_ids = []       # ordered list for matrix operations
        self.embedding_matrix = None  # stacked matrix for batch operations
        self._id_to_idx = {}       # entity_id → row index in embedding_matrix
        print(f"  Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def encode_entities(self, df: pd.DataFrame, text_column: str = "name_norm"):
        """Encode all entities' names into embeddings.

        Args:
            df: DataFrame with 'entity_id' and text_column columns.
            text_column: Column to encode (default: 'name_norm').
        """
        entity_ids = df["entity_id"].values
        texts = df[text_column].fillna("").values

        print(f"  Encoding {len(texts):,} entity names...")
        vectors = self.model.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine = dot product
        )

        for i, eid in enumerate(entity_ids):
            self.embeddings[eid] = vectors[i]

        # Build matrix for batch operations
        self.entity_ids = list(self.embeddings.keys())
        self._id_to_idx = {eid: i for i, eid in enumerate(self.entity_ids)}
        self.embedding_matrix = np.vstack(
            [self.embeddings[eid] for eid in self.entity_ids]
        )
        print(f"  Encoded {len(self.embeddings):,} entities → {self.embedding_matrix.shape}")

    def cosine_similarity(self, eid1: str, eid2: str) -> float:
        """Compute cosine similarity between two entities' name embeddings.

        Since embeddings are L2-normalized, cosine = dot product.
        """
        v1 = self.embeddings.get(eid1)
        v2 = self.embeddings.get(eid2)
        if v1 is None or v2 is None:
            return 0.0
        return float(np.dot(v1, v2))

    def batch_cosine_similarity(self, query_eid: str, candidate_eids: list) -> np.ndarray:
        """Compute cosine similarity between one query and multiple candidates.

        Returns array of similarities aligned with candidate_eids.
        """
        v_query = self.embeddings.get(query_eid)
        if v_query is None:
            return np.zeros(len(candidate_eids))

        candidate_vecs = []
        valid_mask = []
        for eid in candidate_eids:
            v = self.embeddings.get(eid)
            if v is not None:
                candidate_vecs.append(v)
                valid_mask.append(True)
            else:
                candidate_vecs.append(np.zeros_like(v_query))
                valid_mask.append(False)

        if not candidate_vecs:
            return np.zeros(len(candidate_eids))

        cand_matrix = np.vstack(candidate_vecs)  # (N, dim)
        sims = cand_matrix @ v_query  # dot product = cosine (normalized)

        # Zero out invalid entries
        for i, valid in enumerate(valid_mask):
            if not valid:
                sims[i] = 0.0

        return sims


def embedding_block(
    s1: pd.DataFrame,
    s_other: pd.DataFrame,
    embedding_engine: "EmbeddingEngine",
    top_k: int = 30,
    min_score: float = 0.3,
) -> dict:
    """Semantic blocking using sentence-transformer embeddings.

    For each S1 entity, find the top-K most semantically similar entities
    from s_other (within the same country) using cosine similarity of
    name embeddings.

    Args:
        s1: Source 1 DataFrame (normalized, with entity_id).
        s_other: Source 2 or 3 DataFrame (normalized, with entity_id).
        embedding_engine: Pre-fitted EmbeddingEngine with all entities encoded.
        top_k: Number of top candidates to retrieve per S1 entity.
        min_score: Minimum cosine similarity to include a candidate.

    Returns:
        dict: s1_entity_id → set of candidate entity_ids from s_other.
    """
    candidates = defaultdict(set)

    # Group s_other by country for country-constrained blocking
    country_groups = defaultdict(list)
    for _, row in s_other.iterrows():
        country_groups[row["country_norm"]].append(row["entity_id"])

    # Pre-build country → embedding matrix for s_other
    country_matrices = {}
    country_eids = {}
    for country, eids in country_groups.items():
        vecs = []
        valid_eids = []
        for eid in eids:
            v = embedding_engine.embeddings.get(eid)
            if v is not None:
                vecs.append(v)
                valid_eids.append(eid)
        if vecs:
            country_matrices[country] = np.vstack(vecs)  # (N_country, dim)
            country_eids[country] = valid_eids

    # For each S1 entity, find top-K in same country
    s1_iter = s1.iterrows()
    for _, row in tqdm(s1_iter, total=len(s1), desc="Embedding blocking"):
        s1_eid = row["entity_id"]
        country = row["country_norm"]

        v_query = embedding_engine.embeddings.get(s1_eid)
        if v_query is None or country not in country_matrices:
            continue

        # Cosine similarity = dot product (embeddings are normalized)
        sims = country_matrices[country] @ v_query  # (N_country,)
        eids_for_country = country_eids[country]

        # Get top-K above threshold
        if len(sims) <= top_k:
            # Take all above threshold
            for i, sim in enumerate(sims):
                if sim >= min_score:
                    candidates[s1_eid].add(eids_for_country[i])
        else:
            # Get top-K indices
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            for idx in top_indices:
                if sims[idx] >= min_score:
                    candidates[s1_eid].add(eids_for_country[idx])

    return dict(candidates)
