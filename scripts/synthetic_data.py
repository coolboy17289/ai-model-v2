"""Fine-tune the embedding model on synthetic (query, paragraph) pairs.

Uses sentence-transformers' MultipleNegativesRankingLoss with in-batch
negatives. After training, runs eval.compare() to decide whether to ship
the fine-tuned model or roll back to baseline.

Public surface:
    finetune(epochs=3, batch_size=32, lr=2e-5) -> dict
"""

import os
import sqlite3
import sys
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import (
    BASE_MODEL_NAME,
    BASELINE_MODEL_DIR,
    DATA_DIR,
    FINETUNED_MODEL_DIR,
    load_active_model,
)
from synthetic_data import load_pairs
from eval import _ensure_baseline_snapshot


DATABASE = os.path.join(DATA_DIR, "brain.db")


# --- Helpers ---------------------------------------------------------------

def _fetch_paragraphs_by_ids(ids: Iterable[int]) -> dict[int, str]:
    """Return {paragraph_id: text} for the given ids."""
    ids = list(set(ids))
    if not ids:
        return {}
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    c.execute(
        f"SELECT id, text FROM paragraphs WHERE id IN ({placeholders})",
        ids,
    )
    rows = c.fetchall()
    conn.close()
    return {int(r[0]): r[1] for r in rows}


# --- Main entry point ------------------------------------------------------

def finetune(epochs: int = 3, batch_size: int = 32, lr: float = 2e-5,
             warmup_steps: int = 100) -> dict:
    """Fine-tune the base model on synthetic train pairs.

    Returns a stats dict with paths and metrics.
    """
    # Lazy imports — sentence-transformers is heavy and not always needed.
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    train_pairs = load_pairs("train")
    eval_pairs = load_pairs("eval")
    if not train_pairs:
        return {"error": "No train pairs in DB. Run `python scripts/synthetic_data.py` first."}

    # Ensure baseline snapshot exists (eval.compare() needs it)
    _ensure_baseline_snapshot(verbose=True)

    print(
        f"Fine-tuning {BASE_MODEL_NAME} on {len(train_pairs)} train pair(s) "
        f"({len(eval_pairs)} held out for eval)...",
        file=sys.stderr,
        flush=True,
    )

    # Build (query, positive_paragraph_text) examples
    para_by_id = _fetch_paragraphs_by_ids(pid for _, pid in train_pairs)
    examples = []
    skipped = 0
    for query, pid in train_pairs:
        text = para_by_id.get(pid)
        if not text:
            skipped += 1
            continue
        examples.append(InputExample(texts=[query, text]))

    if not examples:
        return {"error": "No usable train pairs after resolving paragraph ids."}

    print(f"Built {len(examples)} training examples ({skipped} skipped).", file=sys.stderr)

    # Load the base model fresh from the snapshot (so we train from a known
    # starting point every time, not from a possibly-already-finetuned state).
    base = SentenceTransformer(BASELINE_MODEL_DIR)

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(base)

    # Use a temp output dir; we'll move the result to FINETUNED_MODEL_DIR
    # once eval.compare() tells us whether to ship it.
    tmp_out = os.path.join(DATA_DIR, "fine-tuned-tmp")
    if os.path.isdir(tmp_out):
        import shutil
        shutil.rmtree(tmp_out)

    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, warmup={warmup_steps}",
          file=sys.stderr)
    base.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=tmp_out,
        show_progress_bar=False,  # tqdm writes to stderr and clashes with the Java spinner
        save_best_model=False,
    )
    print("Training complete.", file=sys.stderr)

    # Move to final location so eval.compare() can find it.
    if os.path.isdir(FINETUNED_MODEL_DIR):
        import shutil
        shutil.rmtree(FINETUNED_MODEL_DIR)
    import shutil
    shutil.move(tmp_out, FINETUNED_MODEL_DIR)

    # Run eval — this writes active_model.txt to the winning model path.
    from eval import compare
    eval_output = compare()
    print(eval_output, file=sys.stderr)

    # Read which model was shipped
    active_model_file = os.path.join(DATA_DIR, "active_model.txt")
    shipped_path = ""
    if os.path.exists(active_model_file):
        with open(active_model_file, "r", encoding="utf-8") as f:
            shipped_path = f.read().strip()

    shipped = "finetuned" if shipped_path == FINETUNED_MODEL_DIR else "baseline"

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_pairs": len(examples),
        "eval_pairs": len(eval_pairs),
        "shipped": shipped,
        "fine_tuned_path": FINETUNED_MODEL_DIR,
        "active_model_path": shipped_path,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(finetune(), indent=2))
