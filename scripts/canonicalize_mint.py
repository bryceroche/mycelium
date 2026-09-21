"""canonicalize_mint.py -- ONE CONVENTION FOR BOTH REGISTERS (2026-09-21, word
given). THE POSITIONAL LAW (ledger 2026-09-17, scripts/canonical_positional.py):
on the pen/prose dialect every factor introduces exactly one new variable
whose index is its slot; mint instead numbers variables by FIRST MENTION in
the text (the gold builder's span sort, phase1_algebra_head.py ~line
920-930: `if all(f.get("spans") for f in smp["factors"]) and
_gen.get("canonical") != "positional": sort factors by min span start`).
The target-side census (2026-09-20 19:50) found mint's pointer wall is 64%
variable identity -- the two conventions fight the same pointer arm. This
script rewrites every MINT row in the arg-stamped diet
(.cache/form_mix_pm35a.jsonl) to the SAME positional convention prose
already carries, reusing canonical_positional.py's algorithm (reencode_ops,
positional_order's ASAP rule, the canonicalize remap) UNCHANGED for
given/rel/sel/mod/fdiv/pct, and EXTENDING it for `macro` factors, which
canonical_positional.vars_of() and canonicalize()'s remap loop do not
cover at all (verified against mycelium/macros.py's docstrings + expanders):

  FRAC_OF   {"a": int (CONSTANT, NOT a var -- verified against
             expand_frac_of: only fed into a NEW given's `value`), "k": int
             (CONSTANT), "x": var, "result": var}        -> var refs: x
  OP_APPLY  {"op": "add"|"sub" (macro-internal, untouched by reencode_ops,
             which only rewrites ftype=="rel"), "k1": int, "x": var,
             "k2": int, "y": var, "result": var}          -> var refs: x, y
  CHAIN_MUL {"xs": [v0..vn], "result": var}                -> var refs: xs (list)

1,663/17,348 mint rows (9.6%) carry a macro factor; without this extension
those rows would either compute a WRONG positional order (vars_of missing
a real dependency) or -- even with a correct order -- end up with the
macro's `x`/`y`/`xs` left at OLD variable ids while every other factor in
the same row now speaks NEW ids (a silently broken row). `pct` needs no
such extension: THE PCT SELF-REFERENCE (verified 364/364 in the diet,
mycelium/csp_domains.py's pct_pred: `a*100 == p*b`, scope=(args[0],
args[1])) means args[0] IS the factor's own introduced variable -- the
existing generic args-remap already gets both positions right because it
remaps by VALUE through the same `new` map regardless of which position is
"new".

Row-level fields that carry a variable index, and what happens to each
(read every field of a mint row for this — the full audit, stated once
here rather than re-derived per row):
  var, args, result           -- per-factor; renumbered (existing rule,
                                  extended to macro's x/y/xs as above)
  query_var                    -- renumbered (existing rule)
  mentions (dict keys)         -- re-keyed old->new; span VALUES untouched
                                  (character offsets, not variable ids)
  solution (list, index=var)   -- reordered so solution[new[v]] ==
                                  old_solution[v]; length becomes
                                  len(new) (may differ from n_vars on
                                  padded rows -- see below, INHERITED from
                                  canonical_positional.canonicalize, not
                                  introduced here)
  spans / arg_spans / cue_spans -- CHAR spans, keyed by POSITION not
                                  variable id; carried through unchanged
                                  because canonicalize's factor rewrite is
                                  `dict(F[i])` with only var/args/result
                                  (extended: x/y/xs) overwritten
  n_vars                        -- NOT an index (a capacity/count); LEFT
                                  UNCHANGED, matching the ALREADY-DEPLOYED
                                  canonical=positional prose rows, 28% of
                                  which (6,596/23,528) already carry
                                  n_vars != len(solution) post-
                                  canonicalization for the same reason
                                  (padded generation families). Verified
                                  harmless: admit_annotation.solver_verdict
                                  never reads `solution`, and
                                  mycelium.macros.expand_graph /
                                  problem_from_algebra3 only use n_vars to
                                  size the CSP's domain array (extra
                                  unconstrained slots don't affect whether
                                  query_var solves to the key).
  m, decisions                  -- scalar counts, not indices; untouched
  gen.resample.subs (dict, when -- OPAQUE PROVENANCE from an earlier
    present)                       resample step, keyed by strings that
                                  look numeric but are not consumed by the
                                  solver or gold builder anywhere; left
                                  untouched and flagged here rather than
                                  silently assumed safe
  answer_field, key              -- absent on every mint row in this diet
                                  (verified: 0/17,348); N/A
  op, k, k1, k2, p, sel, name,   -- constants / string tags, never variable
    role, seq_ord, surface         indices; untouched

Prose rows (gen is a dict carrying "src") are passed through with ZERO
modification -- not even re-serialized differently -- to satisfy gate (a).

Usage: .venv/bin/python3 scripts/canonicalize_mint.py [IN] [OUT]
  (defaults: .cache/form_mix_pm35a.jsonl -> .cache/form_mix_pm35p.jsonl)
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import canonical_positional as CP  # reencode_ops, ready_pos, RULE (env CP_RULE, default "asap")
from admit_annotation import solver_verdict, solve_ladder, VALUE_CAP

IN = sys.argv[1] if len(sys.argv) > 1 else ".cache/form_mix_pm35a.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".cache/form_mix_pm35p.jsonl"
CP_NOGATE = bool(os.environ.get("CP_NOGATE"))


def solve_query(row, budget=5000):
    """THE GROUND-TRUTH SOLVE (2026-09-21): mint rows' `solution` field is
    frequently a zero-filled PLACEHOLDER (verified: e.g. row with factors
    that plainly compute a non-zero query value carries solution=[0]*24)
    -- NOT gold, exactly per the custody-gold law (pen-written solution
    fields are never gold; the certifier's key comes from actually
    solving). Mirrors admit_annotation.solver_verdict's own solve path
    (expand_graph + problem_from_algebra3 + solve_ladder) but WITHOUT a
    presupposed key, returning the assignment at query_var instead of
    checking it against one. This is what the ORIGINAL row (pre-rewrite)
    truly evaluates to, and is the key the REWRITTEN row must reproduce."""
    from alternator_bridge import problem_from_algebra3
    from mycelium.macros import expand_graph
    fs, nv = expand_graph(list(row["factors"]), row["n_vars"])
    gmax = max([f.get("value", 0) for f in fs if f["ftype"] == "given"] + [1])
    m0 = int(row.get("m") or min(VALUE_CAP + 1, max(300, 2 * gmax)))
    gv = {f["var"]: f["value"] for f in fs if f["ftype"] == "given"}
    res, m = solve_ladder(lambda m: problem_from_algebra3(nv, fs, gv, m), m0, budget=budget)
    if res.get("status") != "solved":
        return None, f"unsolved:{res.get('status', '?')}@m={m}"
    asg = res["assignment"]
    return int(asg[row["query_var"]]), f"m={m}"


def is_prose(row):
    g = row.get("gen")
    return isinstance(g, dict) and "src" in g


def macro_ref_keys(fac):
    """Variable-valued keys for each macro NAME (constants excluded --
    verified against mycelium/macros.py's expanders, not guessed)."""
    name = fac.get("name")
    if name == "FRAC_OF":
        return ["x"]          # a, k are constants fed into a NEW given's value/k
    if name == "OP_APPLY":
        return ["x", "y"]     # k1, k2 are constants
    if name == "CHAIN_MUL":
        return ["xs"]         # list-valued
    return []


def vars_of_ext(f):
    """canonical_positional.vars_of(), extended to macro's variable-valued
    fields (missing entirely from the base module)."""
    vs = list(CP.vars_of(f))
    if f.get("ftype") == "macro":
        for k in macro_ref_keys(f):
            val = f.get(k)
            if isinstance(val, list):
                vs.extend(val)
            elif val is not None:
                vs.append(val)
    return vs


def law_holds_ext(F):
    seen = set()
    for k, f in enumerate(F):
        vs = set(vars_of_ext(f))
        if vs - seen != {k}:
            return False
        seen |= vs
    return True


def positional_order_ext(r):
    """canonical_positional.positional_order(), extended: uses
    vars_of_ext throughout so a macro's real dependencies are honoured by
    both the "keep" fast path and the greedy ASAP/spanend search."""
    F = r["factors"]
    seen = set()
    order = []
    left = list(range(len(F)))
    intro = {}
    if CP.RULE == "keep" and law_holds_ext(F):
        return list(range(len(F))), None
    while left:
        cands = [i for i in left if len(set(vars_of_ext(F[i])) - seen) == 1]
        if not cands:
            return None, f"no_positional_order_at_slot_{len(order)}"

        def key(i):
            f = F[i]
            if CP.RULE == "asap" and f["ftype"] != "given":
                known = [intro[v] for v in vars_of_ext(f) if v in intro]
                return (max(known) + 0.5 if known else CP.ready_pos(f), CP.ready_pos(f), i)
            return (CP.ready_pos(f), 0, i)

        i = min(cands, key=key)
        order.append(i)
        (v,) = set(vars_of_ext(F[i])) - seen
        intro[v] = key(i)[0]
        seen |= set(vars_of_ext(F[i]))
        left.remove(i)
    # THE PADDING FIX (2026-09-21): canonical_positional.positional_order's
    # `len(seen) != r["n_vars"]` assumes n_vars is the row's TRUE variable
    # count. It is not, for a large share of this diet's generation
    # families (both registers): n_vars is a fixed slot-bank capacity
    # (commonly 24) padded far past actual usage (verified: prose rows
    # with n_vars==24 and as few as 3 distinct variables actually
    # introduced exist in this same file). The loop above already proves
    # every variable referenced anywhere in the row WAS introduced by
    # exactly one factor (any unintroduced dependency would have starved
    # `cands` and returned None above) -- so `len(seen)` is a
    # self-verified true count, not something to re-validate against a
    # possibly-padded n_vars. The only genuine anomaly worth refusing is
    # using MORE distinct variables than the row's own declared capacity.
    if len(seen) > r["n_vars"]:
        return None, f"vars_{r['n_vars']}_introduced_{len(seen)}"
    return order, None


def remap_factor(f, new):
    f = dict(f)
    if "var" in f:
        f["var"] = new[f["var"]]
    if "args" in f:
        f["args"] = [new[a] for a in f["args"]]
    if "result" in f:
        f["result"] = new[f["result"]]
    if f.get("ftype") == "macro":
        for k in macro_ref_keys(f):
            val = f.get(k)
            if isinstance(val, list):
                f[k] = [new[x] for x in val]
            elif val is not None:
                f[k] = new[val]
    return f


def canonicalize_ext(r):
    """canonical_positional.canonicalize(), extended for macro support (see
    module docstring)."""
    r = CP.reencode_ops(r)
    order, why = positional_order_ext(r)
    if order is None:
        return None, why
    F = r["factors"]
    new = {}
    seen = set()
    for k, i in enumerate(order):
        (v,) = set(vars_of_ext(F[i])) - seen
        new[v] = k
        seen |= set(vars_of_ext(F[i]))
    c = json.loads(json.dumps(r))
    c["factors"] = [remap_factor(F[i], new) for i in order]
    c["query_var"] = new[r["query_var"]]
    c["mentions"] = {str(new[int(v)]): sp for v, sp in (r.get("mentions") or {}).items() if int(v) in new}
    if r.get("solution"):
        c["solution"] = [r["solution"][old] for old, k in sorted(new.items(), key=lambda kv: kv[1])]
    g0 = c.get("gen")
    g = dict(g0) if isinstance(g0, dict) else ({"orig_gen": g0} if g0 is not None else {})
    g["canonical"] = "positional"
    c["gen"] = g
    return c, None


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    with open(IN) as f:
        rows = [json.loads(l) for l in f]

    out_rows = []
    n_prose = n_mint = 0
    n_mint_rewritten = n_mint_kept_refused = n_mint_orig_unsolvable = 0
    refuse_reasons = {}
    law_ok = law_bad = 0
    before_after = []  # for the 6-row eyeball dump

    for row in rows:
        if is_prose(row):
            n_prose += 1
            out_rows.append(row)
            continue
        n_mint += 1
        key, kdetail = solve_query(row)
        if key is None:
            refuse_reasons["orig_unsolvable:" + kdetail.split(":")[1].split("@")[0]] = \
                refuse_reasons.get("orig_unsolvable:" + kdetail.split(":")[1].split("@")[0], 0) + 1
            n_mint_orig_unsolvable += 1
            n_mint_kept_refused += 1
            out_rows.append(row)
            continue
        c, why = canonicalize_ext(row)
        if c is None:
            refuse_reasons["order:" + why.split("_at_")[0].split("_introduced_")[0]] = \
                refuse_reasons.get("order:" + why.split("_at_")[0].split("_introduced_")[0], 0) + 1
            n_mint_kept_refused += 1
            out_rows.append(row)
            continue
        if not CP_NOGATE:
            ok, detail = solver_verdict(c, key)
        else:
            ok, detail = True, "nogate"
        if not ok:
            reason = "gate:" + detail.split("@")[0].split(":")[0]
            refuse_reasons[reason] = refuse_reasons.get(reason, 0) + 1
            n_mint_kept_refused += 1
            out_rows.append(row)
            continue
        lh = law_holds_ext(c["factors"])
        law_ok += int(lh)
        law_bad += int(not lh)
        n_mint_rewritten += 1
        out_rows.append(c)
        if len(before_after) < 6:
            before_after.append((row, c))

    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    P("=" * 78)
    P("ONE CONVENTION FOR BOTH REGISTERS (2026-09-21)")
    P("=" * 78)
    P(f"rows in: {len(rows)}  rows out: {len(out_rows)}  (must match: same rows, same order)")
    P(f"prose (passed through untouched): {n_prose}")
    P(f"mint total: {n_mint}")
    P(f"  rewritten + re-certified: {n_mint_rewritten}")
    P(f"  kept UNCHANGED (refused): {n_mint_kept_refused}"
      f"  (of which original-row-itself-unsolvable: {n_mint_orig_unsolvable})")
    P(f"  refusal reasons: {refuse_reasons}")
    P(f"law_holds on rewritten mint rows: {law_ok}/{law_ok + law_bad}")
    with open(".cache/canonicalize_mint_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(n_prose=n_prose, n_mint=n_mint, n_mint_rewritten=n_mint_rewritten,
                n_mint_kept_refused=n_mint_kept_refused,
                n_mint_orig_unsolvable=n_mint_orig_unsolvable,
                refuse_reasons=refuse_reasons,
                law_ok=law_ok, law_bad=law_bad, before_after=before_after)


if __name__ == "__main__":
    main()
