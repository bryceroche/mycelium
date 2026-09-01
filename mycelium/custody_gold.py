"""custody_gold.py — the one lawful way to read gold off a corpus row.

THE LAW (ledger 2026-07-28, ruled after the book-8 certification's
custody correction caught a would-be lie at quorum): **solution vectors
are pen-side scratch, never custody-side gold.** A delegated book row's
`solution` field descends from the pen's own graph; grading against it
lets the pen grade itself. Mint rows (generator-written, solution-first
construction under the uniqueness gate) carry lawful solution vectors.

Provenance discriminator (verified 2026-07-28 across books 1-8 + mint
fixtures): every pen row's `gen` carries `src_idx`; mint rows never do.

For pen rows, gold comes from the HARVEST answer by TEXT IDENTITY —
src_idx is book-local (candidate-relative in book 2, a third space in
book 3, harvest-global only from book 4 on); the row's `text` field is
the source problem verbatim and is the one cross-book key. Hard-error
on any miss — no silent fallback, ever.
"""
import json
import re

HARVEST_PATH = ".cache/math_harvest_v0.jsonl"
_harvest_by_text = None


GSM8K_PARQUET = (".cache/gsm8k/datasets--openai--gsm8k/snapshots/"
                 "740312add88f781978c0658806c59bc2815b9866/main/"
                 "train-00000-of-00001.parquet")


def _load_harvest():
    global _harvest_by_text
    if _harvest_by_text is None:
        _harvest_by_text = {}
        for line in open(HARVEST_PATH):
            r = json.loads(line)
            _harvest_by_text[r["problem"].strip()] = r["answer"]
        # THE GSM8K KEY TABLE (2026-09-01, wild-lane extension): the
        # dataset's own #### answers, same custody standing as the MATH
        # harvest key — an answer source no pen can touch. Text identity
        # is the key, exactly as for books. Lazy + optional: absent
        # parquet just means GSM8K rows miss and hard-error as always.
        import os
        if os.path.exists(GSM8K_PARQUET):
            import pyarrow.parquet as pq
            t = pq.read_table(GSM8K_PARQUET).to_pydict()
            for q, a in zip(t["question"], t["answer"]):
                _harvest_by_text.setdefault(
                    q.strip(), a.split("####")[-1].strip())
    return _harvest_by_text


def is_pen_row(row):
    """True iff this row was written by a delegated/hand annotation pen
    (a book row), as opposed to the generator mint."""
    return "src_idx" in row.get("gen", {})


def row_gold(row):
    """Custody-side gold for a corpus/fixture row.

    Mint row  -> solution[query_var] (the mint wrote it, solution-first).
    Pen row   -> the harvest answer, looked up by source-text identity.
    Hard-errors on a pen row whose source is missing from the harvest or
    whose harvest answer is not an integer — the caller must decide, not
    inherit a silent wrong gold.
    """
    if not is_pen_row(row):
        return row["solution"][row["query_var"]]
    key = row["text"].strip()
    hb = _load_harvest()
    if key not in hb:
        raise KeyError(
            f"custody_gold: pen row (gen.src_idx={row['gen'].get('src_idx')}) "
            f"whose text is not in the harvest — cannot establish "
            f"independent gold; refusing to fall back to the pen's own "
            f"solution field")
    s = str(hb[key]).strip().replace("$", "").replace(",", "")
    if not re.fullmatch(r"-?\d+", s):
        raise ValueError(
            f"custody_gold: harvest answer for pen row "
            f"(gen.src_idx={row['gen'].get('src_idx')}) is non-integer "
            f"({hb[key]!r}) — not gradable in this pipeline")
    return int(s)
