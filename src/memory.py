"""
memory.py
----------
Embedding-based memory retrieval for the Cognitive Twin system.

Core responsibilities:
1. Load SentenceTransformer model ONCE (module-level singleton).
2. Embed all email texts; cache embeddings to disk (numpy .npy files).
3. For a given target email, retrieve exactly TOP-20 most similar
   historical memories that are chronologically valid (timestamp < target).

DATA LEAKAGE PREVENTION — RETRIEVAL LAYER
==========================================
retrieve_memories() enforces:
  - Only records from the same user (user_id match).
  - Only records where timestamp STRICTLY < target timestamp.
  - The target record itself is never included.
  - The returned memory dicts NEVER expose 'user_response' of the target.
  - Returned memories include previous responses so the LLM can learn style —
    but these are past responses, not the target's response.
"""

import pathlib
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# -- Paths ---------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -- Model singleton -----------------------------------------------------------
# Loaded once; reused across all calls.  'all-MiniLM-L6-v2' is 22 MB,
# fast on CPU, and produces 384-dim embeddings.
_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the embedding model, loading it only on first call."""
    global _MODEL
    if _MODEL is None:
        print("[memory] Loading SentenceTransformer (all-MiniLM-L6-v2) …")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[memory] Model loaded.")
    return _MODEL


# -- Embedding helpers ---------------------------------------------------------

def _cache_key(texts: List[str]) -> str:
    """Deterministic cache key based on a hash of all input texts."""
    combined = "\n".join(texts)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings and return an (n, 384) float32 array.
    Results are cached to disk; re-loading avoids re-computation on reruns.
    """
    key = _cache_key(texts)
    cache_path = CACHE_DIR / f"embeddings_{key}.npy"

    if cache_path.exists():
        embeddings = np.load(cache_path)
        print(f"[memory] Loaded cached embeddings from {cache_path.name}")
        return embeddings

    model = get_model()
    print(f"[memory] Computing embeddings for {len(texts)} texts …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pre-normalise for cosine via dot product
    )
    np.save(cache_path, embeddings)
    print(f"[memory] Cached embeddings → {cache_path.name}")
    return embeddings.astype(np.float32)


def cosine_similarity_matrix(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a single query vector and a matrix of keys.
    Both query and keys are assumed to be L2-normalised (from embed_texts).
    Returns a 1-D array of shape (n,).
    """
    # Dot product of normalised vectors = cosine similarity
    return keys @ query


# -- Memory retrieval ----------------------------------------------------------

def build_memory_store(df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Build a per-user memory store from the full dataframe.
    Stores indices, timestamps, and a reference to the embeddings array.

    Parameters:
        df          : full records dataframe (all splits merged, sorted by timestamp)
        embeddings  : (n, 384) array aligned with df rows

    Returns:
        Dict with keys per user_id, each mapping to their row indices.
    """
    store: Dict[str, Any] = {
        "df": df.reset_index(drop=True),
        "embeddings": embeddings,
        "user_indices": {},
    }
    for user_id in df["user_id"].unique():
        idx = df.index[df["user_id"] == user_id].tolist()
        store["user_indices"][user_id] = idx
    return store


def retrieve_memories(
    target_record_id: str,
    target_user_id: str,
    target_timestamp: pd.Timestamp,
    target_embedding: np.ndarray,
    memory_store: Dict[str, Any],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retrieve the TOP-k most semantically similar historical memories for a
    target email, enforcing strict chronological validity.

    Leakage prevention (enforced here):
      [OK]  Only same-user records.
      [OK]  Only records with timestamp < target_timestamp.
      [OK]  Target record itself excluded by record_id check.
      ✗  Target's user_response is NOT included in returned dicts.
      ✗  Future records are NEVER returned.

    Parameters:
        target_record_id   : unique ID of the target email (excluded from pool)
        target_user_id     : user whose memories to search
        target_timestamp   : timestamp of the target email
        target_embedding   : (384,) embedding of the incoming email text
        memory_store       : built by build_memory_store()
        top_k              : number of memories to return (default=20)

    Returns:
        List of memory dicts, sorted descending by cosine similarity.
        Each dict contains: record_id, timestamp, email_subject,
        incoming_email, user_response, decision, similarity_score.
    """
    df: pd.DataFrame = memory_store["df"]
    embeddings: np.ndarray = memory_store["embeddings"]
    user_indices: List[int] = memory_store["user_indices"].get(target_user_id, [])

    if not user_indices:
        return []

    # Filter: same user, strictly before target timestamp, not the target itself
    candidate_indices = [
        i for i in user_indices
        if (
            df.iloc[i]["timestamp"] < target_timestamp
            and df.iloc[i]["record_id"] != target_record_id
        )
    ]

    if not candidate_indices:
        return []

    # Compute cosine similarities
    candidate_embeddings = embeddings[candidate_indices]          # (m, 384)
    similarities = cosine_similarity_matrix(target_embedding, candidate_embeddings)

    # Rank by similarity (descending); take top_k
    n_retrieve = min(top_k, len(candidate_indices))
    top_local_idx = np.argsort(similarities)[::-1][:n_retrieve]

    memories = []
    for local_idx in top_local_idx:
        global_idx = candidate_indices[local_idx]
        row = df.iloc[global_idx]
        memories.append({
            "record_id":     row["record_id"],
            "timestamp":     str(row["timestamp"]),
            "email_subject": row["email_subject"],
            "incoming_email": row["incoming_email"],
            "user_response": row["user_response"],   # PAST response (leakage-safe)
            "decision":      row["decision"],
            "similarity_score": float(similarities[local_idx]),
        })

    return memories
