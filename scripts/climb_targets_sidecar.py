"""climb_targets_sidecar.py -- THE GOVERNOR CLIMB's targets for ANY jsonl fixture (2026-10-14).

scripts/climb_targets.py built the diet pool's sidecar (.cache/climb_targets_formpm35c.npz) and the
DIAGNOSTIC-ONLY wild pass; it hard-codes those paths. The CPU gate fixture (form_tiny64.jsonl) is not
the pool, so the trainer needs a sidecar keyed by ITS OWN row order. This script runs climb_targets.py's
own functions UNMODIFIED (process_pool: the same spaCy teacher, the same governor walk, the same
tokenization + head-piece alignment) over a given jsonl and writes the same npz layout, plus two stamps
the trainer checks: `mix_sha` (mycelium.era.mix_sha16 of the jsonl it was built from) and `n_rows`.
It never touches the pool's outputs (refuses to write any of climb_targets.py's own OUT_* paths).

FENCE: the wild holdout / any measure-only fixture must be written to a path containing
'DIAGNOSTIC' (the trainer refuses any target path containing that word); a jsonl whose name contains
'wild' or 'holdout' or 'math500' is refused unless --out contains 'DIAGNOSTIC'.

USAGE: .venv/bin/python3 scripts/climb_targets_sidecar.py --jsonl .cache/form_tiny64.jsonl \
           --out .cache/climb_targets_tiny64.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import climb_targets as CT  # noqa: E402  (its own functions, unmodified)
from mycelium.era import mix_sha16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    for banned in (CT.OUT_NPZ, CT.OUT_SUMMARY, CT.OUT_SAMPLE, CT.OUT_WILD_DIAG_NPZ, CT.OUT_WILD_DIAG_SUMMARY):
        assert os.path.realpath(out) != os.path.realpath(banned), f"refusing to overwrite climb_targets.py's own output {banned}"
    low = os.path.basename(args.jsonl).lower()
    if any(w in low for w in ("wild", "holdout", "math500")):
        assert "DIAGNOSTIC" in os.path.basename(out), \
            f"{args.jsonl} is a measure-only fixture: its targets must be stamped DIAGNOSTIC (no trainer may read them)"
    assert not os.path.exists(out), f"{out} exists -- banked artifacts are never overwritten; pick a new name"

    import spacy
    from tokenizers import Tokenizer
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    hf_tok = Tokenizer.from_file(CT.TOKENIZER_JSON)
    rows = [json.loads(l) for l in open(args.jsonl)]
    for i, r in enumerate(rows):
        r["_orig_idx"] = i
    arrays, stats, _ = CT.process_pool(rows, nlp, hf_tok, label=os.path.basename(args.jsonl))
    assert stats["n_rows_processed"] == len(rows), stats
    arrays["mix_sha"] = np.array(mix_sha16(args.jsonl))
    arrays["n_rows"] = np.array(len(rows))
    arrays["source_jsonl"] = np.array(os.path.abspath(args.jsonl))
    np.savez_compressed(out, **arrays)
    summ = dict(jsonl=os.path.abspath(args.jsonl), out=out, mix_sha=str(arrays["mix_sha"]), stats=stats)
    with open(out.replace(".npz", "_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
