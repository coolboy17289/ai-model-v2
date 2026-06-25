"""Embedding-based retrieval for the AI Model v2 brain.

Replaces the TF-IDF cosine pipeline in brain.py with semantic search using
sentence-transformers. Vectors are stored as float32 BLOBs in a new
`paragraph_embeddings` table keyed on (paragraph_id, model), so baseline
and fine-tuned variants can coexist side-by-side.

Public surface:
    init_db(conn)                -- additive schema migration
    load_active_model()          -- read active_model.txt, lazy-download if needed
    encode_paragraphs(model, texts)
    store_embeddings(model_name, ids, vectors)
    load_paragraph_matrix(model_name) -> (matrix[N, D] float32, ids[N])
    rebuild_embeddings()         -- delta-encode missing rows
    query_top_k(question, k=3)   -- cosine top-k over the matrix

Config:
    BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, MIT
"""

import os
import sys
import struct
import time

import numpy as np

# --- CONFIG -----------------------------------------------------------------

# BAAI/bge-small-en-v1.5: 120MB, 384-dim, MIT, strong on technical retrieval.
# Switch this single constant to swap the base model. Both the schema
# (paragraph_embeddings.dim) and downstream code are dim-agnostic.
BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# Filesystem layout
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DATABASE = os.path.join(DATA_DIR, "brain.db")
SBERT_CACHE = os.path.join(DATA_DIR, "sbert-cache")
ACTIVE_MODEL_FILE = os.path.join(DATA_DIR, "active_model.txt")
BASELINE_MODEL_DIR = os.path.join(DATA_DIR, "baseline-model")
FINETUNED_MODEL_DIR = os.path.join(DATA_DIR, "fine-tuned-model")

# BGE recommends an instruction prefix for queries (not for passages).
# This is a small but real quality bump on technical retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# In-memory matrix cache, keyed on (model_name, db_mtime).
# Re-built on every load_paragraph_matrix call whose key has drifted.
_MATRIX_CACHE: dict = {}


class EmbeddingsMissingError(Exception):
    """Raised when /ask is called before embeddings have been built."""


class ModelUnavailableError(Exception):
    """Raised when the embedding model can't be loaded or downloaded."""


# --- Schema migration -------------------------------------------------------

def init_db(conn) -> None:
    """Create the paragraph_embeddings table if it doesn't exist.

    Idempotent. Safe to call from brain.init_db().
    """
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS paragraph_embeddings (
            paragraph_id INTEGER NOT NULL,
            model        TEXT    NOT NULL,
            vector       BLOB    NOT NULL,
            dim          INTEGER NOT NULL,
            PRIMARY KEY (paragraph_id, model)
        )
    ''')
    conn.commit()


# --- Model loading ----------------------------------------------------------

def _read_active_model_path() -> str | None:
    if not os.path.exists(ACTIVE_MODEL_FILE):
        return None
    with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
        path = f.read().strip()
    return path or None


def _write_active_model_path(path: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(path)


def _is_valid_model_dir(path: str) -> bool:
    """Sentence-transformers saves a config.json + modules.json + weights."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "config.json"))


def load_active_model(verbose: bool = True):
    """Load the embedding model named in active_model.txt.

    Behavior:
      - If active_model.txt points to a valid local dir, load from there.
      - If active_model.txt is missing or stale:
          1. If a fine-tuned model exists in data/fine-tuned-model/, use it.
          2. Otherwise use the BASE_MODEL_NAME (download to sbert-cache).

    Lazy-downloads BASE_MODEL_NAME on first use. Prints progress to stderr
    so the Java spinner thread can display it.
    """
    # Import here so importing this module without torch installed still
    # works (e.g. for the EmbeddingsMissingError path).
    from sentence_transformers import SentenceTransformer

    active_path = _read_active_model_path()

    # 1. Explicit active path wins
    if active_path and _is_valid_model_dir(active_path):
        if verbose:
            print(f"Loading active model from {active_path}", file=sys.stderr)
        return SentenceTransformer(active_path)

    # 2. Fine-tuned model present and not yet activated -> use it
    if _is_valid_model_dir(FINETUNED_MODEL_DIR):
        _write_active_model_path(FINETUNED_MODEL_DIR)
        if verbose:
            print(f"Activating fine-tuned model at {FINETUNED_MODEL_DIR}", file=sys.stderr)
        return SentenceTransformer(FINETUNED_MODEL_DIR)

    # 3. Fall back to BASE_MODEL_NAME (download on first use)
    os.makedirs(SBERT_CACHE, exist_ok=True)
    if verbose:
        print(
            f"Downloading embedding model {BASE_MODEL_NAME} (~120MB) "
            f"to {SBERT_CACHE}...",
            file=sys.stderr,
            flush=True,
        )
    model = SentenceTransformer(BASE_MODEL_NAME, cache_folder=SBERT_CACHE)
    _write_active_model_path(BASE_MODEL_NAME)
    if verbose:
        print(f"Model ready: {BASE_MODEL_NAME}", file=sys.stderr, flush=True)
    return model


def active_model_name() -> str:
    """Return the name of the currently active model (for diagnostics).

    Reads active_model.txt; falls back to BASE_MODEL_NAME if unset.
    """
    path = _read_active_model_path()
    return path if path else BASE_MODEL_NAME


# --- Encoding + storage -----------------------------------------------------

def encode_paragraphs(model, texts: list[str]) -> np.ndarray:
    """Encode a batch of paragraph strings -> (N, EMBED_DIM) float32 matrix."""
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine sim == dot product on unit vectors
    )
    return vectors.astype(np.float32)


def _pack_vector(vec: np.ndarray) -> bytes:
    """Pack a 1-D float32 vector into raw bytes for SQLite BLOB storage."""
    assert vec.dtype == np.float32 and vec.ndim == 1
    return vec.tobytes()


def _unpack_vector(blob: bytes, dim: int = EMBED_DIM) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()


def store_embeddings(model_name: str, ids: list[int], vectors: np.ndarray) -> None:
    """Bulk upsert (id, model) -> vector into paragraph_embeddings.

    `vectors` must be (N, dim) float32, already L2-normalized.
    """
    import sqlite3
    assert vectors.shape[0] == len(ids), "ids and vectors length mismatch"
    assert vectors.shape[1] == EMBED_DIM, f"expected dim {EMBED_DIM}, got {vectors.shape[1]}"

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    rows = [
        (int(pid), model_name, _pack_vector(vectors[i].astype(np.float32)), EMBED_DIM)
        for i, pid in enumerate(ids)
    ]
    c.executemany(
        "INSERT OR REPLACE INTO paragraph_embeddings "
        "(paragraph_id, model, vector, dim) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_paragraph_matrix(model_name: str) -> tuple[np.ndarray, list[int]]:
    """Load all embeddings for `model_name` as a stacked matrix + ids list.

    Returns (matrix[N, EMBED_DIM] float32, ids[N]). Both are sorted by id.
    Cached in memory and invalidated by DB mtime drift.
    """
    import sqlite3

    db_mtime = os.path.getmtime(DATABASE)
    cache_key = (model_name, db_mtime)
    cached = _MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT paragraph_id, vector FROM paragraph_embeddings "
        "WHERE model = ? ORDER BY paragraph_id",
        (model_name,),
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        empty = (np.zeros((0, EMBED_DIM), dtype=np.float32), [])
        _MATRIX_CACHE[cache_key] = empty
        return empty

    ids = [int(r[0]) for r in rows]
    matrix = np.stack([_unpack_vector(r[1]) for r in rows]).astype(np.float32)
    _MATRIX_CACHE[cache_key] = (matrix, ids)
    return matrix, ids


def _load_paragraph_texts() -> list[tuple[int, str]]:
    """Return [(paragraph_id, text), ...] for all paragraphs."""
    import sqlite3
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, text FROM paragraphs ORDER BY id")
    rows = c.fetchall()
    conn.close()
    return [(int(r[0]), r[1]) for r in rows]


# --- Rebuild (delta-encode) ------------------------------------------------

def rebuild_embeddings(verbose: bool = True) -> dict:
    """Encode any paragraphs missing from paragraph_embeddings for the active model.

    Delta-encode: if a paragraph already has a row for the active model, skip it.
    Returns a stats dict: {"new": N, "skipped": M, "model": name, "dim": dim}.
    Called from end of train/clear hooks in brain.py.
    """
    model = load_active_model(verbose=verbose)
    model_name = active_model_name()

    # Build set of already-encoded paragraph ids for this model
    import sqlite3
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT paragraph_id FROM paragraph_embeddings WHERE model = ?",
        (model_name,),
    )
    already = {row[0] for row in c.fetchall()}
    conn.close()

    all_paragraphs = _load_paragraph_texts()
    to_encode = [(pid, text) for (pid, text) in all_paragraphs if pid not in already]

    if not to_encode:
        if verbose:
            print(f"Embeddings up to date for model '{model_name}'.", file=sys.stderr)
        return {"new": 0, "skipped": len(all_paragraphs), "model": model_name, "dim": EMBED_DIM}

    if verbose:
        print(
            f"Encoding {len(to_encode)} new paragraph(s) with {model_name}...",
            file=sys.stderr,
            flush=True,
        )

    ids = [pid for pid, _ in to_encode]
    texts = [text for _, text in to_encode]
    vectors = encode_paragraphs(model, texts)
    store_embeddings(model_name, ids, vectors)

    if verbose:
        print(
            f"Done. Total encoded: {len(already) + len(to_encode)} / {len(all_paragraphs)}",
            file=sys.stderr,
        )

    return {
        "new": len(to_encode),
        "skipped": len(already),
        "model": model_name,
        "dim": EMBED_DIM,
    }


# --- Query ------------------------------------------------------------------

def query_top_k(question: str, k: int = 3) -> list[tuple[int, str, float]]:
    """Semantic search: encode `question`, return top-k (id, text, cosine) rows.

    Output shape matches the legacy TF-IDF query_question so brain.py's
    print formatting stays byte-identical: [(para_id, text, score), ...].
    """
    model_name = active_model_name()
    matrix, ids = load_paragraph_matrix(model_name)

    if matrix.shape[0] == 0:
        raise EmbeddingsMissingError(
            f"No embeddings found for model '{model_name}'. "
            "Run `python scripts/brain.py embed_rebuild` first, "
            "or just /train a new topic — embeddings are built automatically."
        )

    # BGE recommends a query prefix; passages don't get one.
    query_text = BGE_QUERY_PREFIX + question if "bge" in model_name.lower() else question

    model = load_active_model(verbose=False)
    q_vec = model.encode(
        [query_text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    # matrix is already L2-normalized by encode_paragraphs; cosine == dot product.
    scores = matrix @ q_vec[0]  # shape (N,)

    top_idx = np.argsort(-scores)[:k]

    # Fetch texts for the top-k ids
    id_set = {ids[i] for i in top_idx}
    import sqlite3
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        f"SELECT id, text FROM paragraphs WHERE id IN ({','.join('?' * len(id_set))})",
        list(id_set),
    )
    text_by_id = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    return [(ids[i], text_by_id.get(ids[i], ""), float(scores[i])) for i in top_idx]


# --- Diagnostics ------------------------------------------------------------

def stats() -> dict:
    """Return embedding-side stats: active model, dim, encoded row count."""
    import sqlite3
    model_name = active_model_name()
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM paragraph_embeddings WHERE model = ?",
        (model_name,),
    )
    encoded_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM paragraphs")
    para_count = c.fetchone()[0]
    conn.close()
    return {
        "model": model_name,
        "dim": EMBED_DIM,
        "encoded": encoded_count,
        "paragraphs": para_count,
    }