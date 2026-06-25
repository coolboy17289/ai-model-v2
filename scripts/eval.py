"""Held-out evaluation: recall@1 / recall@3 for the active model vs baseline.

Compares the currently-active model (whatever's in data/active_model.txt) to a
snapshot of the pristine baseline model. Writes the winning path back to
data/active_model.txt.

Public surface:
    recall_at_k(model, eval_pairs, k) -> float
    compare() -> str
"""

import os
import shutil
import sys

import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import (
    BASE_MODEL_NAME,
    BASELINE_MODEL_DIR,
    DATA_DIR,
    FINETUNED_MODEL_DIR,
    load_active_model,
    encode_paragraphs,
)
from synthetic_data import load_pairs


# --- Recall@k --------------------------------------------------------------

def _all_paragraphs() -> tuple[np.ndarray, list[int]]:
    """Return (matrix[N, D] float32, ids[N]) using the BASE model only.

    We deliberately use the baseline model for both compared runs' matrix, so
    the only difference being measured is the query encoder. If we used the
    model's own embeddings for its own scoring, that would conflate query-
    encoder quality with passage-encoder quality.
    """
    import sqlite3
    from embeddings import _unpack_vector
    DATABASE = os.path.join(DATA_DIR, "brain.db")
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT paragraph_id, vector FROM paragraph_embeddings "
        "WHERE model = ? ORDER BY paragraph_id",
        (BASE_MODEL_NAME,),
    )
    rows = c.fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(
            f"No baseline embeddings found for model '{BASE_MODEL_NAME}'. "
            "Run `python scripts/brain.py embed_rebuild` first."
        )
    ids = [int(r[0]) for r in rows]
    matrix = np.stack([_unpack_vector(r[1]) for r in rows]).astype(np.float32)
    return matrix, ids


def recall_at_k(query_encoder, eval_pairs: list[tuple[str, int]], k: int,
                matrix: np.ndarray, ids: list[int]) -> float:
    """Fraction of (query, true_paragraph_id) pairs where true is in top-k."""
    if not eval_pairs:
        return 0.0
    queries = [q for q, _ in eval_pairs]
    truths = [pid for _, pid in eval_pairs]

    # BGE query prefix for bge-* models
    from embeddings import active_model_name, BGE_QUERY_PREFIX
    model_name = active_model_name()
    if "bge" in model_name.lower():
        queries = [BGE_QUERY_PREFIX + q for q in queries]

    q_vecs = query_encoder.encode(
        queries,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    # matrix is L2-normalized; cosine == dot product.
    scores = q_vecs @ matrix.T  # shape (Q, N)

    top_k_idx = np.argsort(-scores, axis=1)[:, :k]  # (Q, k)
    top_k_ids = np.array(ids)[top_k_idx]  # (Q, k)
    truths_arr = np.array(truths)  # (Q,)

    hits = (top_k_ids == truths_arr[:, None]).any(axis=1).sum()
    return float(hits) / float(len(eval_pairs))


# --- Snapshot baseline for comparison --------------------------------------

def _ensure_baseline_snapshot(verbose: bool = True) -> str:
    """Copy BASE_MODEL_NAME to data/baseline-model/ on first run. Return path."""
    if os.path.isdir(BASELINE_MODEL_DIR) and os.path.exists(
        os.path.join(BASELINE_MODEL_DIR, "config.json")
    ):
        return BASELINE_MODEL_DIR
    if verbose:
        print(f"Snapshotting baseline {BASE_MODEL_NAME} to {BASELINE_MODEL_DIR}...", file=sys.stderr)
    os.makedirs(BASELINE_MODEL_DIR, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BASE_MODEL_NAME)
    model.save(BASELINE_MODEL_DIR)
    return BASELINE_MODEL_DIR


# --- Head-to-head compare --------------------------------------------------

def compare() -> str:
    """Run recall@1 / recall@3 for baseline and fine-tuned, pick winner, return formatted string."""
    from sentence_transformers import SentenceTransformer

    eval_pairs = load_pairs("eval")
    if not eval_pairs:
        return "No eval pairs in DB. Run `python scripts/synthetic_data.py` first to populate synthetic_pairs."

    matrix, ids = _all_paragraphs()
    print(f"Eval set: {len(eval_pairs)} pair(s) over {len(ids)} paragraph(s).", file=sys.stderr)

    # Sanity-warning when the eval set is small or dominated by verbatim positives.
    # Recall saturates at 1.0 on near-identical (query, paragraph) pairs regardless
    # of encoder quality, so the eval signal is degenerate. The 2pt rollback rule
    # still catches catastrophic degradation; small improvements won't be visible.
    n = len(eval_pairs)
    if n < 50:
        print(
            f"NOTE: eval set is small (n={n}). Recall@k is a noisy estimator here; "
            "small improvements may not be visible.",
            file=sys.stderr,
        )

    # Load both models
    baseline_path = _ensure_baseline_snapshot(verbose=True)
    baseline_model = SentenceTransformer(baseline_path)

    has_finetuned = os.path.isdir(FINETUNED_MODEL_DIR) and os.path.exists(
        os.path.join(FINETUNED_MODEL_DIR, "config.json")
    )

    # Score baseline
    b_r1 = recall_at_k(baseline_model, eval_pairs, k=1, matrix=matrix, ids=ids)
    b_r3 = recall_at_k(baseline_model, eval_pairs, k=3, matrix=matrix, ids=ids)

    # Score fine-tuned if present
    ft_r1 = ft_r3 = None
    if has_finetuned:
        ft_model = SentenceTransformer(FINETUNED_MODEL_DIR)
        ft_r1 = recall_at_k(ft_model, eval_pairs, k=1, matrix=matrix, ids=ids)
        ft_r3 = recall_at_k(ft_model, eval_pairs, k=3, matrix=matrix, ids=ids)

    # Decide what to ship. Rollback rule (2pt tolerance).
    ROLLBACK_TOLERANCE = 0.02
    CATASTROPHIC_DROP = 0.05

    if not has_finetuned:
        ship = "baseline"
        ship_path = BASE_MODEL_NAME
        delta = 0.0
        reason = "no fine-tuned model present"
    else:
        ship_ft = (
            ft_r1 >= b_r1 - ROLLBACK_TOLERANCE
            and ft_r3 >= b_r3 - ROLLBACK_TOLERANCE
        )
        catastrophic = (b_r1 - ft_r1) > CATASTROPHIC_DROP
        if ship_ft:
            ship = "finetuned"
            ship_path = FINETUNED_MODEL_DIR
            delta = ft_r1 - b_r1
            reason = f"r@1 {'up' if delta >= 0 else 'within tolerance'} ({delta:+.3f})"
        else:
            ship = "baseline"
            ship_path = baseline_path
            delta = ft_r1 - b_r1
            reason = (
                f"regression > {ROLLBACK_TOLERANCE:.2f} (r@1 {delta:+.3f})"
                + (" — CATASTROPHIC" if catastrophic else "")
            )

    # Write active_model.txt
    active_model_file = os.path.join(DATA_DIR, "active_model.txt")
    with open(active_model_file, "w", encoding="utf-8") as f:
        f.write(ship_path)

    # Format output as a small ASCII table (line-based stdout for Java bridge)
    lines = []
    lines.append(f"Eval set:       {len(eval_pairs)} pair(s) over {len(ids)} paragraph(s)")
    lines.append("")
    lines.append(f"{'Model':<14}{'Recall@1':>10}{'Recall@3':>10}")
    lines.append(f"{'baseline':<14}{b_r1:>10.3f}{b_r3:>10.3f}")
    if has_finetuned:
        lines.append(f"{'finetuned':<14}{ft_r1:>10.3f}{ft_r3:>10.3f}")
    lines.append("")
    lines.append(f"→ ship: {ship}  (r@1 delta {delta:+.3f}, {reason})")
    if has_finetuned and (b_r1 - ft_r1) > CATASTROPHIC_DROP:
        lines.append(f"WARN: catastrophic regression — consider deleting {FINETUNED_MODEL_DIR}")
    lines.append(f"Active model file: {active_model_file}")

    output = "\n".join(lines)

    # Persist a copy for human inspection
    eval_log = os.path.join(DATA_DIR, "eval-log.txt")
    with open(eval_log, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    return output


if __name__ == "__main__":
    print(compare())