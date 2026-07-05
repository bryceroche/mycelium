"""kenken_nl_gen.py — KenKen-in-words: the Phase-1 template generator + gold labeling.

THE OBJECT (docs/phase1_skeleton_spec.md §4): render existing KenKen records into
templated NL with a span<->factor GOLD ALIGNMENT that supervises THREE consumers with
one scheme: (1) the delta head (which factors, per cycle), (2) input-side
remove-at-read band masking (mask a factor's spans once committed), (3) the parse-side
BirdNet segmentation re-run (token spans ARE the calls).

GOLD FORMAT COMMITMENTS (the expensive-to-change choices, settled 2026-07-05):
  * A factor's alignment is a SET of char spans (list of [start,end)), NOT one span —
    contiguity is NOT assumed, from day one. Templated factors mostly have a singleton
    set; the SPLIT-REF template family deliberately produces >=2 disjoint spans
    ("One cage consists of cells (1,1) and (1,2). ... That cage multiplies to 12.")
    so downstream code cannot silently inherit a contiguity assumption. (At most ONE
    split-ref factor per problem in v0 — keeps the "that cage" anaphora unambiguous.)
  * Targets are EXACT INTEGERS (the symbolic verifier consumes them; the inlet's
    log-buckets are derived downstream via target_to_bucket — never emitted here).
  * Canonical factor order (the slot-supervision order): (type_rank row<col<cage,
    first member flat id on the N_MAX=7 grid) — deterministic, generator-controlled.
  * Cells are 0-indexed in DATA (matching the corpus records), 1-indexed in NL.
  * Row/col factors all align to the ONE preamble sentence (the deliberate
    one-sentence -> many-factors case). Distractor sentences align to NO factor
    (spans recorded separately so masking/segmentation can score them).

RECORD OUT (jsonl, one per problem):
  { "N", "text",
    "factors": [ {"ftype": "row"|"col"|"cage", "members_rc": [[r,c],..] (0-idx),
                  "members_flat": [r*7+c,..] (N_MAX grid, deducer layout),
                  "op": str|None (cage only; 'given' included), "target": int|None,
                  "spans": [[s,e],..] } ... ]  (canonical order),
    "distractor_spans": [[s,e],..],
    "source": {"path", "idx"}, "gen": {"seed", "shuffle", "split_ref", "distractors"} }

THE ROUND-TRIP SELFTEST (zero model in the path — any labeling bug dies HERE, not in
training curves): reconstruct (n, cages, clues) from the GOLD FACTORS ALONE (N = #row
factors; cages+clues = the cage factors incl. givens), feed the search tier
(problem_from_kenken -> solve_symbolic), and assert the solved grid equals the record's
solution exactly (corpus records are unique-solution). Proves the gold labels carry
everything needed to rebuild + solve the puzzle.

USAGE:
  Selftest (CPU; includes the search-tier round trip on real corpus records):
    .venv/bin/python3 scripts/kenken_nl_gen.py --selftest
  Generate:
    .venv/bin/python3 scripts/kenken_nl_gen.py --src .cache/kenken_train.jsonl \
        --n 200 --seed 0 --out .cache/kenken_nl_train.jsonl \
        --split-ref-prob 0.3 --distractor-prob 0.3
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))

N_MAX = 7  # the deducer grid (mycelium.kenken_data.N_MAX) — flat ids are r*N_MAX+c

TYPE_RANK = {"row": 0, "col": 1, "cage": 2}

# ---------------------------------------------------------------------------
# Template banks (paraphrase variation is the curriculum knob)
# ---------------------------------------------------------------------------

PREAMBLES = [
    "Solve the {n}x{n} KenKen: every row and every column contains the digits 1 through {n} exactly once.",
    "This is a {n} by {n} KenKen puzzle, so each row and each column must use each of 1..{n} exactly once.",
    "In the {n}x{n} grid below, no row or column repeats a digit from 1 to {n}.",
]

CELL_FORMS = [
    lambda r, c: f"({r},{c})",
    lambda r, c: f"row {r}, column {c}",
]

OP_PHRASES = {
    "add": ["add up to {t}", "sum to {t}", "have a total of {t}"],
    "mul": ["multiply to {t}", "have a product of {t}"],
    "sub": ["differ by {t}", "have a difference of {t}"],
    "div": ["divide to give {t}", "have a quotient of {t}"],
}

CAGE_LEADS = ["The cage containing cells {cells} ", "Cells {cells} ", "Together, cells {cells} "]

GIVEN_FORMS = [
    "Row {r}, column {c} is a {v}.",
    "The cell at ({r},{c}) contains {v}.",
    "You are told that ({r},{c}) is {v}.",
]

SPLIT_INTRO = ["One cage consists of cells {cells}.", "There is a cage made of cells {cells}."]
SPLIT_CONSTR = ["That cage must {phrase}.", "The values in that cage {phrase}."]

DISTRACTORS = [
    "KenKen puzzles were invented by a Japanese math teacher.",
    "Take your time and double-check each deduction.",
    "A pencil is recommended for puzzles of this size.",
]


def _cells_nl(members_rc, rng, n: int = 0) -> str:
    """Render a 0-indexed cell list as 1-indexed NL ('(1,1), (1,2) and (2,1)').

    Big-N puzzles (n>=6) use ONLY the compact '(r,c)' form: T=512 is a JIT
    graph-shape parameter every downstream script inherits, so the token budget is
    protected by terser phrasing, not a bigger window (decision 2026-07-06)."""
    form = CELL_FORMS[0] if n >= 6 else rng.choice(CELL_FORMS)
    parts = [form(r + 1, c + 1) for (r, c) in members_rc]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def render_record(rec: dict, rng: random.Random, split_ref_prob: float = 0.0,
                  distractor_prob: float = 0.0, shuffle: bool = True) -> dict:
    """Render ONE corpus record into (text, gold factors with span sets)."""
    n = int(rec["N"])
    cages = rec["cages"]
    clues = rec["clues"]

    # --- sentence units: (text, factor_keys) ; factor key = cage index or ('rowcol',) ---
    units = []
    preamble = rng.choice(PREAMBLES).format(n=n)
    units.append((preamble, ["rowcol"]))

    # pick at most ONE multi-cell non-given cage for the split-ref family
    split_ci = -1
    if split_ref_prob > 0 and rng.random() < split_ref_prob:
        cands = [i for i, (cage, clue) in enumerate(zip(cages, clues))
                 if clue[0] != "given" and len(cage) >= 2]
        if cands:
            split_ci = rng.choice(cands)

    split_units = []  # [(text, ci)] — intro + constraint, order preserved under shuffle
    for ci, (cage, clue) in enumerate(zip(cages, clues)):
        op, target = clue[0], int(clue[1])
        members_rc = [(int(r), int(c)) for (r, c) in cage]
        terse = n >= 6  # big-N: shortest forms only (the T=512 token-budget decision)
        if op == "given":
            (r, c) = members_rc[0]
            gform = "Cell ({r},{c}) is {v}." if terse else rng.choice(GIVEN_FORMS)
            units.append((gform.format(r=r + 1, c=c + 1, v=target), [ci]))
        elif ci == split_ci:
            intro = rng.choice(SPLIT_INTRO).format(cells=_cells_nl(members_rc, rng, n))
            phrase = rng.choice(OP_PHRASES[op]).format(t=target)
            constr = rng.choice(SPLIT_CONSTR).format(phrase=phrase)
            split_units = [(intro, [ci]), (constr, [ci])]
        else:
            lead = ("Cells {cells} " if terse
                    else rng.choice(CAGE_LEADS)).format(cells=_cells_nl(members_rc, rng, n))
            phrase = (OP_PHRASES[op][0] if terse
                      else rng.choice(OP_PHRASES[op])).format(t=target)
            units.append((lead + phrase + ".", [ci]))

    distract = []
    while distractor_prob > 0 and rng.random() < distractor_prob and len(distract) < 2:
        distract.append((rng.choice([d for d in DISTRACTORS
                                     if d not in [t for t, _ in distract]]), []))
    body = units[1:] + distract
    if shuffle:
        rng.shuffle(body)
    # splice the split pair in, intro strictly before constraint, gap when possible
    if split_units:
        i1 = rng.randrange(0, len(body) + 1)
        body.insert(i1, split_units[0])
        i2 = rng.randrange(min(i1 + 1, len(body)), len(body) + 1)
        body.insert(i2, split_units[1])
    ordered = [units[0]] + body

    # --- assemble text + char spans per unit ---
    text_parts, spans_by_key, distractor_spans = [], {}, []
    pos = 0
    for i, (s, keys) in enumerate(ordered):
        if i > 0:
            text_parts.append(" ")
            pos += 1
        start = pos
        text_parts.append(s)
        pos += len(s)
        span = [start, pos]
        if not keys:
            distractor_spans.append(span)
        for k in keys:
            spans_by_key.setdefault(k, []).append(span)
    text = "".join(text_parts)

    # --- gold factors, canonical order ---
    factors = []
    for r in range(n):  # row all-different
        members = [(r, c) for c in range(n)]
        factors.append(_factor("row", members, None, None, spans_by_key["rowcol"]))
    for c in range(n):  # col all-different
        members = [(r, c) for r in range(n)]
        factors.append(_factor("col", members, None, None, spans_by_key["rowcol"]))
    for ci, (cage, clue) in enumerate(zip(cages, clues)):
        members_rc = [(int(r), int(c)) for (r, c) in cage]
        factors.append(_factor("cage", members_rc, clue[0], int(clue[1]),
                               spans_by_key[ci]))
    factors.sort(key=lambda f: (TYPE_RANK[f["ftype"]], f["members_flat"][0]))

    return {"N": n, "text": text, "factors": factors,
            "distractor_spans": distractor_spans}


def _factor(ftype, members_rc, op, target, spans):
    members_rc = sorted((int(r), int(c)) for (r, c) in members_rc)
    return {"ftype": ftype,
            "members_rc": [list(m) for m in members_rc],
            "members_flat": [r * N_MAX + c for (r, c) in members_rc],
            "op": op, "target": target,
            "spans": [list(s) for s in spans]}


# ---------------------------------------------------------------------------
# The round trip: gold factors -> (n, cages, clues) -> search tier -> solution
# ---------------------------------------------------------------------------

def reconstruct_from_gold(factors) -> tuple:
    """Rebuild (n, cages, clues) from the GOLD FACTORS ALONE (no source record)."""
    n = sum(1 for f in factors if f["ftype"] == "row")
    cages, clues = [], []
    for f in factors:
        if f["ftype"] != "cage":
            continue
        cages.append([list(m) for m in f["members_rc"]])
        clues.append([f["op"], int(f["target"])])
    return n, cages, clues


def roundtrip_solve_matches(sample: dict, rec: dict, budget: int = 200_000) -> bool:
    """Gold factors -> search tier -> exact grid match vs the record solution."""
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    n, cages, clues = reconstruct_from_gold(sample["factors"])
    if n != int(rec["N"]):
        return False
    prob = problem_from_kenken(n, cages, clues)
    res = solve_symbolic(prob, budget=budget, seed=0)
    if res["status"] != "solved":
        return False
    asg = res["assignment"]
    get = (lambda v: asg[v]) if not isinstance(asg, dict) else (lambda v: asg[v])
    grid = [[int(get(r * n + c)) for c in range(n)] for r in range(n)]
    return grid == [[int(v) for v in row] for row in rec["solution"]]


# ---------------------------------------------------------------------------
# Span-integrity checks (pure text-side; run on every generated sample)
# ---------------------------------------------------------------------------

def span_integrity(sample: dict) -> list:
    """Return a list of violation strings (empty = clean)."""
    errs = []
    text = sample["text"]
    L = len(text)
    cage_spans = []
    for f in sample["factors"]:
        if not f["spans"]:
            errs.append(f"factor {f['ftype']}/{f['members_flat'][0]} has NO spans")
        for (s, e) in f["spans"]:
            if not (0 <= s < e <= L):
                errs.append(f"span [{s},{e}) out of bounds (len {L})")
        if f["ftype"] == "cage":
            cage_spans.append((f, [tuple(s) for s in f["spans"]]))
            joined = " ".join(text[s:e] for (s, e) in f["spans"])
            if str(f["target"]) not in joined:
                errs.append(f"target {f['target']} not in its span text: {joined!r}")
    # distinct cage factors must not share spans
    seen = {}
    for f, spans in cage_spans:
        for sp in spans:
            if sp in seen:
                errs.append(f"cage span {sp} shared by two factors")
            seen[sp] = True
    # distractor spans must not collide with any factor span
    fact_spans = {tuple(s) for f in sample["factors"] for s in f["spans"]}
    for sp in sample["distractor_spans"]:
        if tuple(sp) in fact_spans:
            errs.append(f"distractor span {sp} collides with a factor span")
    return errs


# ---------------------------------------------------------------------------
# Selftest (CPU; real corpus records; search-tier round trip included)
# ---------------------------------------------------------------------------

def selftest(src: str, n_records: int = 12) -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    with open(_THIS_FILE) as f:
        ast.parse(f.read())
    print("  [OK] ast.parse")

    recs = [json.loads(l) for l in open(src)][:n_records]
    check(f"loaded {n_records} records from {os.path.basename(src)}", len(recs) == n_records)

    # determinism
    a = render_record(recs[0], random.Random(7), 0.5, 0.5)
    b = render_record(recs[0], random.Random(7), 0.5, 0.5)
    check("determinism (same seed -> identical sample)", a == b)

    # force a split-ref sample and verify >=2 disjoint spans on that factor
    sr = None
    for seed in range(50):
        s = render_record(recs[0], random.Random(seed), split_ref_prob=1.0)
        multi = [f for f in s["factors"] if len(f["spans"]) >= 2 and f["ftype"] == "cage"]
        if multi:
            sr = (s, multi[0])
            break
    check("split-ref family yields a cage factor with >=2 spans", sr is not None)
    if sr is not None:
        (s0, e0), (s1, e1) = sorted(tuple(x) for x in sr[1]["spans"])[:2]
        check("split-ref spans are disjoint", e0 <= s1)

    # span integrity + round trip over records x seeds
    n_int = n_rt = 0
    for i, rec in enumerate(recs):
        for seed in (0, 1):
            smp = render_record(rec, random.Random(seed), 0.3, 0.3)
            errs = span_integrity(smp)
            if errs:
                print(f"    integrity errs rec{i} seed{seed}: {errs[:2]}")
            n_int += not errs
            n_rt += roundtrip_solve_matches(smp, rec)
    total = len(recs) * 2
    check(f"span integrity {n_int}/{total}", n_int == total)
    check(f"ROUND TRIP gold->search-tier->exact solution {n_rt}/{total}", n_rt == total)

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src", default=".cache/kenken_test.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-ref-prob", type=float, default=0.3)
    ap.add_argument("--distractor-prob", type=float, default=0.3)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--out", default=".cache/kenken_nl.jsonl")
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(0 if selftest(args.src) else 1)

    recs = [json.loads(l) for l in open(args.src)][: args.n]
    rng = random.Random(args.seed)
    n_bad = 0
    with open(args.out, "w") as f:
        for i, rec in enumerate(recs):
            smp = render_record(rec, random.Random(rng.randrange(2**31)),
                                args.split_ref_prob, args.distractor_prob,
                                shuffle=not args.no_shuffle)
            errs = span_integrity(smp)
            if errs or not roundtrip_solve_matches(smp, rec):
                n_bad += 1
                continue
            smp["source"] = {"path": args.src, "idx": i}
            smp["gen"] = {"seed": args.seed, "shuffle": not args.no_shuffle,
                          "split_ref": args.split_ref_prob,
                          "distractors": args.distractor_prob}
            f.write(json.dumps(smp) + "\n")
    print(f"[gen] wrote {len(recs) - n_bad}/{len(recs)} samples to {args.out} "
          f"({n_bad} rejected by integrity/round-trip gates)")


if __name__ == "__main__":
    main()
