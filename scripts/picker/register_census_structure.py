"""register_census_structure.py -- ZERO-GPU CENSUS (ledger question 'OR_241', 2026-10-13): which
register does the MATH-500 TARGET (L1-3) resemble -- wild (one-read single-step) or mint/silver
(formulaic chains)? Measured from GOLD GRAPHS ONLY (each row's `factors` field -- the fixture
format phase1_algebra_head.load_alg's own samples jsonl, not the custody `solution` text, per the
custody-gold law). No model, no GPU, no MATH-500 answers read.

POOLS:
  wild    .cache/wild_admitted_holdout.jsonl       (311, the wild holdout)
  mint    .cache/algebra_nl_test.jsonl             (300, fixture test23)
  silver  .cache/silver460_test.jsonl              (460)
  census  .cache/book1_prose_pairs.jsonl           (14; the EXACT first-100-pool rows a book
          annotated -- text-identity-matched against .cache/census_mouth_join.json's 100 problems,
          ledger 2026-07-xx "book 1 already trained 14 pool items")
  books   .cache/book{1..8}_prose_pairs*.jsonl combined (874 rows, gen.src_idx-stamped harvest
          annotations) -- the broader MATH-register proxy when n=14 is too thin to trust alone;
          book8_candidates.py's own census characterizes this harvest draw as "in-reach L1-3"
          (903 eligible before the fixture-100 exclusion), so `books` is treated as additional
          L1-3-register signal, not independently level-filtered row-by-row here (stated, not
          hidden -- a limitation of this cheap read).

STRUCTURE (per row, from factors alone):
  n_vars   = count of distinct variable ids touched by any factor
  n_givens = count of ftype=='given' factors
  n_rels   = count of ftype in ('rel','fdiv') factors (fdiv = derived-value-by-constant, a unary
             edge: result = var / k; folded into the same dependency graph as rel's binary edges)
  depth(v) = 0 if v is a given (or never produced -- a free/unconstrained var); else
             1 + max(depth(input) for input in producing factor's inputs) -- the derived-value
             CHAIN DEPTH, replayed from the factors list itself (THE POSITIONAL LAW's dependency
             graph, not an assumption that slot==var)
  row chain depth = max(depth(f.result) for f in rel/fdiv factors), 0 if the row has no relations
  derived-arg share = of all ('rel' factor) args, the fraction whose producing factor is NOT
             'given' (i.e. the arg is itself a relation's output)
  inverse share = scripts/polarity_census.classify_row's own fwd/inv label (THE POSITIONAL LAW's
             replay of variable-introduction order; "inv" = the relation introduces one of its
             ARGS rather than its result -- the sub/div-as-add/mul encoding), reused VERBATIM
             (no reimplementation)
  sentences = len(args_census.sentence_spans(text)) (the project's own ". "-split convention)
  entities  = distinct capitalized words not at their sentence's first word (cheap proxy; mint's
             single-lowercase-letter variable names never trigger this, so mint reads ~0 by
             construction -- the intended contrast, not a bug)

OR-vs-PMS8 BY DEPTH (if the banked dumps exist): scripts/polarity_census.load_dump +
head_fields on .cache/dump_wild_{OR_241,PMS8_241}.pkl, restricted to gft==0 (relation) slots,
bucketed by that slot's OWN row-derived chain depth (==1 vs >=2) via the SAME depth(v) replay
above (ground truth from wild_admitted_holdout.jsonl's factors, joined by (row i, slot j) -- slot
j was verified to align positionally with factors[j], ledger-adjacent sanity check run inline).

OUTPUTS (new files only; never overwrites a banked artifact):
  .cache/register_census_structure.json  -- the full per-pool table + the depth split, machine-
                                             readable
  (this script itself, scripts/picker/register_census_structure.py, committed on gen-weights)

usage: .venv/bin/python3 scripts/picker/register_census_structure.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from args_census import sentence_spans            # noqa: E402
from polarity_census import classify_row           # noqa: E402

POOLS = {
    "wild":   [".cache/wild_admitted_holdout.jsonl"],
    "mint":   [".cache/algebra_nl_test.jsonl"],
    "silver": [".cache/silver460_test.jsonl"],
    "census": [".cache/book1_prose_pairs.jsonl"],
}
BOOKS_GLOB = ".cache/book*_prose_pairs*.jsonl"
CENSUS_JOIN = ".cache/census_mouth_join.json"
OUT = ".cache/register_census_structure.json"

_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


# ===========================================================================
# gold-graph structure (factors only -- never `solution`/`decisions`)
# ===========================================================================

def vars_touched(factors):
    vs = set()
    for f in factors:
        if f["ftype"] == "given":
            vs.add(f["var"])
        elif f["ftype"] == "rel":
            vs.update(f.get("args", []))
            vs.add(f.get("result"))
        elif f["ftype"] == "fdiv":
            vs.add(f["var"]); vs.add(f["result"])
        elif "var" in f:
            vs.add(f["var"])
    return vs


def build_depth_fn(factors):
    """var -> (kind, factor) that produces it; depth(v) memoized replay."""
    produced_by = {}
    for f in factors:
        if f["ftype"] == "given":
            produced_by.setdefault(f["var"], ("given", f))
        elif f["ftype"] == "rel":
            produced_by.setdefault(f["result"], ("rel", f))
        elif f["ftype"] == "fdiv":
            produced_by.setdefault(f["result"], ("fdiv", f))
    cache = {}

    def depth(v):
        if v in cache:
            return cache[v]
        cache[v] = 0  # cycle guard (shouldn't occur in a DAG factor graph)
        if v not in produced_by:
            d = 0
        else:
            kind, f = produced_by[v]
            if kind == "given":
                d = 0
            elif kind == "fdiv":
                d = 1 + depth(f["var"])
            else:
                d = 1 + max((depth(a) for a in f.get("args", [])), default=0)
        cache[v] = d
        return d

    return depth, produced_by


def entity_count(text):
    spans = sentence_spans(text)
    starts = {s for s, e in spans}
    cnt = set()
    for m in _CAP_RE.finditer(text):
        # sentence-initial occurrence of this exact span start is excluded
        if m.start() in starts:
            continue
        cnt.add(m.group(0))
    return len(cnt)


def row_stats(row):
    factors = row["factors"]
    text = row.get("text", "")
    depth, produced_by = build_depth_fn(factors)
    rels = [f for f in factors if f["ftype"] in ("rel", "fdiv")]
    n_rel_depths = []
    derived_args = 0
    total_args = 0
    for f in rels:
        rv = f.get("result")
        n_rel_depths.append(depth(rv))
        if f["ftype"] == "rel":
            for a in f.get("args", []):
                total_args += 1
                kind = produced_by.get(a, ("given", None))[0]
                if kind != "given":
                    derived_args += 1
    cls = classify_row(factors)
    n_fwd = sum(1 for c in cls if c == "fwd")
    n_inv = sum(1 for c in cls if c == "inv")
    n_other = sum(1 for c in cls if c == "other")
    row_depth = max(n_rel_depths) if n_rel_depths else 0
    return dict(
        n_vars=len(vars_touched(factors)),
        n_givens=sum(1 for f in factors if f["ftype"] == "given"),
        n_rels=len(rels),
        row_depth=row_depth,
        rel_depths=n_rel_depths,
        derived_args=derived_args,
        total_args=total_args,
        n_fwd=n_fwd, n_inv=n_inv, n_other=n_other,
        sentences=len(sentence_spans(text)) or 1,
        entities=entity_count(text),
    )


def load_rows(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [skip] {p} not found", file=sys.stderr)
            continue
        rows.extend(json.loads(l) for l in open(p) if l.strip())
    return rows


def pool_summary(name, rows):
    stats = [row_stats(r) for r in rows]
    n = len(stats)
    if n == 0:
        return None

    def mean(key):
        return float(np.mean([s[key] for s in stats]))

    depth_bucket = np.array([min(s["row_depth"], 3) for s in stats])  # cap display at 3+
    # share of rows whose MAX relation depth is 1 / 2 / >=3 (rows with 0 relations excluded
    # from the depth-share denominator -- they have no chain to speak of)
    with_rel = [s for s in stats if s["n_rels"] > 0]
    nr = len(with_rel)
    d1 = sum(1 for s in with_rel if s["row_depth"] == 1) / nr if nr else float("nan")
    d2 = sum(1 for s in with_rel if s["row_depth"] == 2) / nr if nr else float("nan")
    d3 = sum(1 for s in with_rel if s["row_depth"] >= 3) / nr if nr else float("nan")
    tot_derived = sum(s["derived_args"] for s in stats)
    tot_args = sum(s["total_args"] for s in stats)
    tot_fwd = sum(s["n_fwd"] for s in stats)
    tot_inv = sum(s["n_inv"] for s in stats)
    tot_other = sum(s["n_other"] for s in stats)
    tot_cls = tot_fwd + tot_inv + tot_other
    return dict(
        n=n,
        n_vars_mean=mean("n_vars"),
        n_givens_mean=mean("n_givens"),
        n_rels_mean=mean("n_rels"),
        row_depth_mean=mean("row_depth"),
        depth1_share=d1, depth2_share=d2, depth3plus_share=d3,
        derived_arg_share=(tot_derived / tot_args if tot_args else float("nan")),
        inverse_share=(tot_inv / (tot_fwd + tot_inv) if (tot_fwd + tot_inv) else float("nan")),
        # coverage of the DAG single-new-variable assumption classify_row relies on -- LOW
        # coverage (mint) means inverse_share above is measuring a degenerate residual, not the
        # register, and must be read with this number attached (stated, not hidden)
        fwd_inv_coverage=(1.0 - tot_other / tot_cls if tot_cls else float("nan")),
        sentences_mean=mean("sentences"),
        entities_mean=mean("entities"),
    )


def census_pool_check():
    """Confirm book1_prose_pairs.jsonl's text identity against the 100-item census join."""
    if not os.path.exists(CENSUS_JOIN):
        return None
    join = json.load(open(CENSUS_JOIN))
    census_texts = {c["problem"].strip() for c in join}
    rows = load_rows(POOLS["census"])
    hit = sum(1 for r in rows if r.get("text", "").strip() in census_texts)
    return dict(pool_n=len(census_texts), book1_pp_n=len(rows), text_match=hit)


# ===========================================================================
# OR vs PMS8 by depth (wild only, if the banked dumps exist)
# ===========================================================================

def depth_split():
    import pickle
    wild_path = POOLS["wild"][0]
    rows = load_rows([wild_path])
    depth_of_slot = {}   # (i, j) -> chain depth of that relation slot's result var
    for i, row in enumerate(rows):
        depth, _ = build_depth_fn(row["factors"])
        for j, f in enumerate(row["factors"]):
            if f["ftype"] == "rel":
                depth_of_slot[(i, j)] = depth(f["result"])

    out = {}
    for tag in ("OR_241", "PMS8_241"):
        pkl = f".cache/dump_wild_{tag}.pkl"
        if not os.path.exists(pkl):
            print(f"  [skip] {pkl} not found", file=sys.stderr)
            continue
        D = pickle.load(open(pkl, "rb"))
        buckets = {"depth1": [], "depth_ge2": []}
        for t in D:
            i, j = t[0], t[1]
            if (i, j) not in depth_of_slot:
                continue   # not a relation slot (given/fdiv) -- excluded, this split is rel-only
            rest = t[2:]
            gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = rest
            gset = set(gargs)
            if len(gset) == 1:
                f_args = bool(pdup) and (pargs[0] in gset)
            else:
                f_args = set(pargs) == gset
            ok = bool(ppres) and (pft == gft) and (pres_ == gres) and (pop == gop) and f_args
            d = depth_of_slot[(i, j)]
            buckets["depth1" if d <= 1 else "depth_ge2"].append(ok)
        out[tag] = {k: dict(n=len(v), acc=(float(np.mean(v)) if v else float("nan")))
                    for k, v in buckets.items()}
    return out


def main():
    result = {"pools": {}, "files_used": {}}

    for name, paths in POOLS.items():
        rows = load_rows(paths)
        result["files_used"][name] = paths
        s = pool_summary(name, rows)
        if s:
            result["pools"][name] = s

    books_files = sorted(glob.glob(BOOKS_GLOB))
    books_rows = load_rows(books_files)
    result["files_used"]["books"] = books_files
    s = pool_summary("books", books_rows)
    if s:
        result["pools"]["books"] = s

    result["census_pool_check"] = census_pool_check()
    result["depth_split_wild_OR_vs_PMS8"] = depth_split()

    os.makedirs(".cache", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    # ---- human-readable table ----
    cols = ["n", "n_vars_mean", "n_givens_mean", "n_rels_mean", "row_depth_mean",
            "depth1_share", "depth2_share", "depth3plus_share",
            "derived_arg_share", "inverse_share", "fwd_inv_coverage",
            "sentences_mean", "entities_mean"]
    order = ["wild", "mint", "silver", "census", "books"]
    print(f"{'pool':8s} " + " ".join(f"{c:>17s}" for c in cols))
    for name in order:
        p = result["pools"].get(name)
        if not p:
            continue
        print(f"{name:8s} " + " ".join(f"{p[c]:17.4f}" if isinstance(p[c], float) else f"{p[c]:17d}"
                                        for c in cols))

    print("\ncensus-pool text-identity check:", result["census_pool_check"])
    print("\nOR vs PMS8 by depth (wild, gold-relation slots only):")
    for tag, buckets in result["depth_split_wild_OR_vs_PMS8"].items():
        print(f"  {tag}: {buckets}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
