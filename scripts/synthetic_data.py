"""Generate synthetic (query, paragraph) pairs for fine-tuning + eval.

The pipeline:
  1. For each paragraph in `paragraphs`, generate up to 3 (query, paragraph)
     positive pairs using cheap heuristics (no LLM).
  2. Persist into the `synthetic_pairs` table (query, paragraph_id, split).
  3. Deterministically split 80/20 into 'train' and 'eval' via seed=42.

Heuristics:
  - Verbatim positive (paragraph[:200]). Trivial; tests the pipeline.
  - "What is X" template on definition-style openings.
  - Verb-fronting transform on copula-first sentences.

Public surface:
    populate()              -- one-shot generation + persistence
    generate_pairs(texts)   -- generator function (returns [(query, paragraph_idx)])
    load_pairs(split)       -- read from DB by split
    clear()                 -- wipe table (called by brain.clear_database)
"""

import os
import random
import re
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(_HERE, "..", "data", "brain.db")

# Deterministic split so re-runs of /finetune and /eval are comparable.
SEED = 42
EVAL_FRAC = 0.2

# Length filters. Too short: no signal. Too long: probably a code block or
# table that won't tokenize cleanly into a useful "question".
MIN_PARA_LEN = 30
MAX_PARA_LEN = 1500
MAX_QUERY_LEN = 120

# Hard cap per paragraph. Three pairs is enough for MultipleNegativesRankingLoss
# with batch_size=32 (in-batch negatives provide the real signal).
MAX_PAIRS_PER_PARA = 3


# --- Heuristic pair generation ---------------------------------------------

# Definition openings: "X is a ...", "X, also known as ...", "X refers to ...",
# "X, a type of ...", "X — a ...", "X: a ..."
_DEFINITION_RE = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9 \-_/&]{2,60}?)"
    r"(?:\s*,)?\s+"
    r"(?:is|are|refers to|was|were|denotes?|describes?)\s+",
    re.IGNORECASE,
)

# Copula-first sentence: "X is Y."  Used for verb-fronting transform.
_COPULA_RE = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9 \-_/&]{1,40}?)\s+"
    r"(?P<copula>is|are|was|were|has|have|can|refers to|denotes?)\s+"
    r"(?P<predicate>.+?)\.\s*$",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Crude sentence splitter on `. ! ?` followed by space or end."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clean_query(q: str) -> str:
    """Strip whitespace and trailing punctuation, capitalize first letter."""
    q = q.strip().rstrip(".?!")
    if not q:
        return ""
    return q[0].upper() + q[1:]


# Heading-like opening. These are Wikipedia collapsible section labels, table
# headers, navigation elements. Tokenized as paragraphs but useless for QA.
_HEADING_PREFIXES = (
    "Toggle ", "Jump to ", "edit ", "See also ", "References ",
    "External links ", "Further reading ", "Notes ", "Citations ",
    "Bibliography ", "Source: ", "Sources: ",
)
# Paragraphs that look like titles: short, no terminating punctuation,
# contain digits/colons/dashes typical of section headers.
_HEADING_RE = re.compile(r"^[\dA-Za-z][\w \-:/&()',.]{0,80}$")


def _looks_like_heading(text: str) -> bool:
    if any(text.startswith(p) for p in _HEADING_PREFIXES):
        return True
    stripped = text.strip()
    if len(stripped) > 80:
        return False
    if stripped.endswith((".", "?", "!")):
        return False
    # Short, no sentence punctuation, contains year or colon -> heading-like
    if re.search(r"\d{4}", stripped) and (":" in stripped or "–" in stripped or "—" in stripped):
        return True
    if _HEADING_RE.match(stripped) and not stripped.endswith("."):
        # Only flag if it doesn't look like a regular sentence (no lowercase
        # verb forms etc.) - cheap proxy: < 12 words and no "the/a/an".
        if len(stripped.split()) < 12 and not re.search(r"\b(the|a|an|of|in|for|to|and)\b", stripped, re.IGNORECASE):
            return True
    return False


# Subject extraction for "What is X": prefer a clean capitalized noun phrase.
# Avoids grabbing "The first set of rules" when the real subject follows.
def _extract_subject(text: str) -> str | None:
    """Return a clean subject noun-phrase from the first sentence of `text`."""
    sentences = _split_sentences(text)
    if not sentences:
        return None
    first = sentences[0]

    # Pattern A: "X is a/an/the ..."  -> subject is X
    m = re.match(
        r"^(?P<subject>[A-Z][A-Za-z0-9\-_/&]+(?:\s+[A-Z][A-Za-z0-9\-_/&]+){0,4})"
        r"\s+(?:is|are|was|were|refers to|denotes?|describes?)\s+",
        first,
    )
    if m:
        subj = m.group("subject").strip()
        words = subj.split()
        if 2 <= len(words) <= 6 and not any(w.lower() in {"the", "a", "an", "this", "that"} for w in words[:1]):
            return subj

    # Pattern B: "X, also known as ..." -> subject is X
    m = re.match(r"^(?P<subject>[A-Z][\w\-_/& ]+?),\s+(?:also known as|formerly|sometimes)", first, re.IGNORECASE)
    if m:
        subj = m.group("subject").strip()
        if 2 <= len(subj.split()) <= 6:
            return subj

    # Pattern C: fallback - first capitalized run of 2-4 words.
    m = re.match(r"^(?P<subject>[A-Z][\w\-_/&]+(?:\s+[A-Z][\w\-_/&]+){0,3})", first)
    if m:
        subj = m.group("subject").strip()
        if 2 <= len(subj.split()) <= 5:
            return subj
    return None


def _is_clean_subject(s: str) -> bool:
    """Reject subjects that contain stopwords or year prefixes."""
    if not s:
        return False
    if re.match(r"^\d", s):
        return False
    bad = {"the", "a", "an", "this", "that", "these", "those", "after", "before", "in", "on", "at"}
    words = s.lower().split()
    if words[0] in bad:
        return False
    if any(w in bad for w in words[:2]):
        return False
    return True


# Wikipedia boilerplate that ends up in `paragraphs` but isn't useful for QA.
# If a paragraph's first 200 chars contain any of these, skip it entirely.
_BOILERPLATE_MARKERS = (
    "From Wikipedia, the free encyclopedia",
    "This article needs additional citations",
    "The printable version is no longer supported",
    "Find sources:",
    "(Redirected from",
    "Jump to navigation",
    "Jump to content",
    "Toggle ",
    "&#160;",  # non-breaking space, common in nav boxes
)


def _is_boilerplate(text: str) -> bool:
    head = text[:200]
    return any(marker in head for marker in _BOILERPLATE_MARKERS)


def _generate_for_paragraph(text: str) -> list[str]:
    """Return up to MAX_PAIRS_PER_PARA queries for a single paragraph."""
    if not (MIN_PARA_LEN <= len(text) <= MAX_PARA_LEN):
        return []
    if _looks_like_heading(text):
        return []
    if _is_boilerplate(text):
        return []

    queries: list[str] = []

    # 1. Verbatim positive (truncated)
    verbatim = text[:200].strip()
    if verbatim:
        queries.append(verbatim)

    # 2. "What is X" template on definition openings
    subject = _extract_subject(text)
    if subject and _is_clean_subject(subject):
        q = f"What is {subject}?"
        if len(q) <= MAX_QUERY_LEN and q not in queries:
            queries.append(q)

    # 3. Verb-fronting transform: "X is Y." -> "What is Y (X)?"
    # Skip: low yield on Wikipedia and produces nonsense more often than not.
    # Empirically <10% useful yield on the test corpus; not worth the noise.

    # Dedupe and cap
    seen = set()
    out = []
    for q in queries:
        qc = _clean_query(q)
        if qc and qc.lower() not in seen:
            seen.add(qc.lower())
            out.append(qc)
        if len(out) >= MAX_PAIRS_PER_PARA:
            break
    return out


def generate_pairs(paragraphs: list[tuple[int, str]]) -> list[tuple[str, int]]:
    """Run heuristics over [(paragraph_id, text), ...]. Return [(query, para_id), ...]."""
    out: list[tuple[str, int]] = []
    for pid, text in paragraphs:
        for q in _generate_for_paragraph(text):
            out.append((q, pid))
    return out


# --- Persistence + split ----------------------------------------------------

def clear() -> None:
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("DELETE FROM synthetic_pairs")
    conn.commit()
    conn.close()


def _fetch_all_paragraphs() -> list[tuple[int, str]]:
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, text FROM paragraphs ORDER BY id")
    rows = c.fetchall()
    conn.close()
    return [(int(r[0]), r[1]) for r in rows]


def populate(verbose: bool = True) -> dict:
    """Generate pairs, deterministic split, persist. Idempotent (wipes first)."""
    paragraphs = _fetch_all_paragraphs()
    if not paragraphs:
        if verbose:
            print("No paragraphs in DB; nothing to generate.")
        return {"total": 0, "train": 0, "eval": 0}

    pairs = generate_pairs(paragraphs)
    if not pairs:
        if verbose:
            print("No pairs generated (paragraphs too short or no heuristic matches).")
        return {"total": 0, "train": 0, "eval": 0}

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * (1 - EVAL_FRAC)))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    clear()
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.executemany(
        "INSERT INTO synthetic_pairs (query, paragraph_id, split) VALUES (?, ?, ?)",
        [(q, pid, "train") for (q, pid) in train_pairs]
        + [(q, pid, "eval") for (q, pid) in eval_pairs],
    )
    conn.commit()
    conn.close()

    if verbose:
        print(f"Generated {len(pairs)} pair(s) from {len(paragraphs)} paragraph(s).")
        print(f"  train: {len(train_pairs)}")
        print(f"  eval:  {len(eval_pairs)}")
    return {"total": len(pairs), "train": len(train_pairs), "eval": len(eval_pairs)}


def load_pairs(split: str) -> list[tuple[str, int]]:
    """Read all pairs from synthetic_pairs where split = split."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT query, paragraph_id FROM synthetic_pairs WHERE split = ? ORDER BY id",
        (split,),
    )
    rows = c.fetchall()
    conn.close()
    return [(r[0], int(r[1])) for r in rows]


if __name__ == "__main__":
    populate(verbose=True)