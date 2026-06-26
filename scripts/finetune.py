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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import (
    BASE_MODEL_NAME,
    BASELINE_MODEL_DIR,
    DATA_DIR,
    FINETUNED_MODEL_DIR,
)
from synthetic_data import load_pairs
from eval import _ensure_baseline_snapshot


DATABASE = os.path.join(DATA_DIR, "brain.db")


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


def finetune(epochs: int = 3, batch_size: int = 8, lr: float = 2e-5,
             warmup_steps: int = 50) -> dict:
    """Fine-tune the base model on synthetic train pairs.

    Defaults are tuned for ~25-50 train pairs (small custom dataset):
      - batch_size=8 (small enough to fit the dataset; drop_last=True)
      - warmup_steps=50 (proportional to ~1 epoch's worth of steps)

    For larger train sets, raise batch_size to 32 and warmup_steps to 100.
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        train_pairs = load_pairs("train")
        eval_pairs = load_pairs("eval")
        if not train_pairs:
            return {"error": "No train pairs in DB. Run `python scripts/synthetic_data.py` first."}

        _ensure_baseline_snapshot(verbose=True)

        print(
            f"Fine-tuning {BASE_MODEL_NAME} on {len(train_pairs)} train pair(s) "
            f"({len(eval_pairs)} held out for eval)...",
            file=sys.stderr,
            flush=True,
        )

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

        # Train from a known starting point: the baseline snapshot.
        base = SentenceTransformer(BASELINE_MODEL_DIR)

        loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
        loss = losses.MultipleNegativesRankingLoss(base)

        tmp_out = os.path.join(DATA_DIR, "fine-tuned-tmp")
        if os.path.isdir(tmp_out):
            import shutil
            shutil.rmtree(tmp_out)

        print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, warmup={warmup_steps}",
              file=sys.stderr)
        # Use old_fit to stay on the v2 DataLoader+InputExample API. The new
        # `fit()` in sentence-transformers 3.x wraps a Trainer that requires
        # the `datasets` package, which we deliberately don't depend on.
        base.old_fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            output_path=tmp_out,
            show_progress_bar=False,
            save_best_model=False,
        )
        print("Training complete.", file=sys.stderr)

        if os.path.isdir(FINETUNED_MODEL_DIR):
            import shutil
            shutil.rmtree(FINETUNED_MODEL_DIR)
        import shutil
        shutil.move(tmp_out, FINETUNED_MODEL_DIR)

        # Eval decides which model ships. Writes active_model.txt.
        from eval import compare
        eval_output = compare()
        print(eval_output, file=sys.stderr)

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
    except Exception as e:
        import traceback
        print(f"Error during fine-tune: {e}")
        print(traceback.format_exc(), file=sys.stderr)
        raise


if __name__ == "__main__":
    import json
    print(json.dumps(finetune(), indent=2))
