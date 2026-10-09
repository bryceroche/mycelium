"""direction_cue_census.py -- THE DIRECTION ROLE's data-side census (2026-10-09, zero-GPU,
CPU-only). Word given: before the arm, measure whether the closed direction-word lexicon
(DIRROLE_LEXICON in scripts/phase1_algebra_head.py -- fewer/less/more/than/left/remaining/
gave/away/lost/spent/half/twice/per/each/total/altogether/together/combined/difference, env-
overridable via ALG_DIRROLE_LEXICON) actually sits in the relations it is meant to inform, and
in particular whether it reaches the INVERSE-form relations THE POLARITY CENSUS (2026-10-08)
found blind (0.250 res accuracy vs 0.873 forward) -- the road must reach >= 60% of wild's
inverse-form relations or the lexicon widens before the arm fires.

Reuses, never reimplements: polarity_census.classify_row/vars_of (fwd/inv/other classification,
replaying THE POSITIONAL LAW's own variable-introduction order) and polarity_census.clause_of
(a relation's own grounded clause -- mentions[result] span if present, else the sentence
containing an arg's gold numeral); phase1_algebra_head.dirrole_cue_spans (THE DIRECTION ROLE's
own lexicon matcher -- the SAME function the model calls at train and at read, so this census
measures exactly what the organ sees, not a parallel reimplementation that could drift).

Three numbers, per fixture (diet prose, wild):
  (1) RELATION COVERAGE: fraction of classified (fwd+inv, "other" excluded) relations whose own
      grounded clause contains >=1 lexicon cue.
  (2) THE CUE -> DIRECTION CONTINGENCY TABLE: per lexicon word, how many classified relations it
      co-occurs with, forward vs inverse, and the inverse share (which cues actually predict
      inverse -- the thing the embedding table has to learn from).
  (3) INVERSE COVERAGE (the gate number): fraction of INVERSE-form relations alone whose clause
      carries >=1 cue -- the >= 0.60 bar.

usage: .venv/bin/python3 scripts/direction_cue_census.py
outputs: .cache/direction_cue_census.txt
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from args_census import sentence_bounds, sentence_spans
from polarity_census import classify_row, clause_of
from phase1_algebra_head import DIRROLE_LEXICON, dirrole_cue_spans

DIET_PATH = ".cache/form_mix_pm35c.jsonl"
WILD_PATH = ".cache/wild_admitted_holdout.jsonl"
DIET_PROSE_CAP = 2000   # matches twin_split_census.DIET_CAP's convention -- same fixture slice
OUT = ".cache/direction_cue_census.txt"
INVERSE_COVERAGE_BAR = 0.60


def load_diet_prose(path, cap):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "src" in r.get("gen", {}):
                out.append(r)
            if len(out) >= cap:
                break
    return out


def cues_in_span(cue_spans, a, b):
    """Which of `cue_spans` ([(cs, ce, cue_id), ...], global over the row's text) fall inside
    the clause [a, b) -- any overlap counts, matching _spans_to_tokmask's convention."""
    return [cid for (cs, ce, cid) in cue_spans if cs < b and ce > a]


def census_rows(rows, label, lines):
    cue_n = collections.Counter()          # cue_id -> relations it co-occurs with (fwd+inv)
    cue_inv = collections.Counter()        # cue_id -> of those, how many are inverse
    n_fwd = n_inv = n_other = 0
    n_fwd_cue = n_inv_cue = 0
    for row in rows:
        factors = row["factors"]
        text = row["text"]
        cls = classify_row(factors)
        if not any(c in ("fwd", "inv") for c in cls):
            continue
        bounds = sentence_bounds(text)
        spans = sentence_spans(text, bounds)
        row_cue_spans = dirrole_cue_spans(text)   # global over the row, THE ORGAN's own matcher
        for k, f in enumerate(factors):
            c = cls[k]
            if c == "other" or c is None:
                if c == "other":
                    n_other += 1
                continue
            a, b, _src = clause_of(row, k, f, bounds, spans)
            cids = cues_in_span(row_cue_spans, a, b) if a is not None else []
            has = bool(cids)
            if c == "fwd":
                n_fwd += 1
                n_fwd_cue += int(has)
            else:
                n_inv += 1
                n_inv_cue += int(has)
            for cid in set(cids):
                cue_n[cid] += 1
                if c == "inv":
                    cue_inv[cid] += 1
    n_class = n_fwd + n_inv
    lines.append(f"\n[{label}] classified relations: fwd={n_fwd} inv={n_inv} other={n_other}")
    if n_class:
        lines.append(f"  RELATION COVERAGE (fwd+inv, >=1 cue in own clause): "
                      f"{(n_fwd_cue + n_inv_cue) / n_class:.4f} "
                      f"({n_fwd_cue + n_inv_cue}/{n_class})")
    if n_fwd:
        lines.append(f"  forward coverage: {n_fwd_cue / n_fwd:.4f} ({n_fwd_cue}/{n_fwd})")
    inv_cov = (n_inv_cue / n_inv) if n_inv else None
    if n_inv:
        lines.append(f"  INVERSE coverage: {inv_cov:.4f} ({n_inv_cue}/{n_inv})  "
                      f"[bar >= {INVERSE_COVERAGE_BAR:.2f}: "
                      f"{'PASS' if inv_cov >= INVERSE_COVERAGE_BAR else 'MISS -- widen the lexicon'}]")
    lines.append("  cue -> direction contingency (word: n_total n_inv inverse_share):")
    for cid in sorted(cue_n, key=lambda c: -cue_n[c]):
        w = DIRROLE_LEXICON[cid - 1]
        nt = cue_n[cid]; ni = cue_inv[cid]
        lines.append(f"    {w:14s} n={nt:5d} n_inv={ni:5d} inverse_share={ni / nt:.3f}")
    return dict(n_fwd=n_fwd, n_inv=n_inv, n_other=n_other,
                n_fwd_cue=n_fwd_cue, n_inv_cue=n_inv_cue, inv_coverage=inv_cov)


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE DIRECTION ROLE -- CUE CENSUS (2026-10-09, zero-GPU)")
    P("=" * 78)
    P(f"lexicon ({len(DIRROLE_LEXICON)} words, ALG_DIRROLE_LEXICON-overridable): "
      f"{', '.join(DIRROLE_LEXICON)}")

    diet_rows = load_diet_prose(DIET_PATH, DIET_PROSE_CAP)
    wild_rows = [json.loads(l) for l in open(WILD_PATH)]
    P(f"\ndiet prose rows read: {len(diet_rows)} (cap {DIET_PROSE_CAP}, {DIET_PATH})")
    P(f"wild rows read: {len(wild_rows)} ({WILD_PATH})")

    diet_stats = census_rows(diet_rows, "diet prose", lines)
    wild_stats = census_rows(wild_rows, "wild", lines)

    P("\n" + "=" * 78)
    P("THE GATE NUMBER (wild inverse coverage, the road must reach this before the arm)")
    P("=" * 78)
    if wild_stats["inv_coverage"] is not None:
        verdict = "PASS" if wild_stats["inv_coverage"] >= INVERSE_COVERAGE_BAR else "MISS"
        P(f"  wild inverse-relation coverage = {wild_stats['inv_coverage']:.4f} "
          f"({wild_stats['n_inv_cue']}/{wild_stats['n_inv']}) vs bar "
          f"{INVERSE_COVERAGE_BAR:.2f} -- {verdict}")
        if verdict == "MISS":
            P("  RECOMMENDATION: widen DIRROLE_LEXICON (ALG_DIRROLE_LEXICON=<comma list>) before "
              "firing DIRROLE_241 -- the cues above with the highest inverse_share and nonzero n "
              "on wild are the ones already carrying signal; words absent from both tables never "
              "fired on this fixture and are free to add or drop.")
    else:
        P("  no inverse-form relations classified on wild -- cannot score the bar (check WILD_PATH)")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[direction_cue_census] wrote {OUT}")


if __name__ == "__main__":
    main()
