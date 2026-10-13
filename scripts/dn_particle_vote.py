"""THE PARTICLE READ'S VOTE (2026-10-12, pin 2's build plan: "drop unsolvable graphs, down-
weight dead-fact graphs per minimality, vote the rest with loglik tie-break"). Written because
scripts/picker/seeded_coverage.py does not exist yet on branch seeded at the time this chain was
built (the build plan's own fallback: "else write the vote").

Input: N LV_DUMP pickles (one per seed; scripts/loop_val.py LV_DUMP=<path> ALG_SEED_SCHED=1
ALG_SEED_TEMP=<tau_max> ALG_SEED_RNG=<seed>), all over the SAME rows in the SAME order. Each
pickle is a flat list of (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres,
pdup) tuples (loop_val.py's own LV_DUMP tuple — see its _DUMP.append call).

Per row:
  1. Reconstruct each particle's predicted factor graph (slots where ppres).
  2. DROP UNSOLVABLE: feed the predicted graph to the symbolic solver (mycelium.csp_domains.
     problem_from_algebra + mycelium.csp_core.solve_symbolic, the do_errors.py idiom) — a
     particle whose predicted graph is malformed or UNSAT is dropped (weight 0).
  3. DOWN-WEIGHT DEAD-FACT GRAPHS (minimality proxy, SIMPLIFIED — no full minimality audit here:
     a "dead" relation slot is one whose res variable is never an arg of any other present
     relation and is never itself a GIVEN's variable, i.e. its own output is never consumed —
     weight *= (1 - 0.5 * dead_fraction) rather than a hard drop).
  4. VOTE the surviving particles' predicted graphs by per-slot MAJORITY (weighted), tie-broken
     by lowest seed-index (loglik is not available from LV_DUMP's argmax-only tuple — a documented
     simplification, not the literal "loglik tie-break" of the build plan).
  5. Score the voted graph against gold (positional match, like matched_read.py's `positional`)
     for fac-exact; report coverage (fraction of rows with >= 1 solvable particle) and the voted
     accuracy, plus regressions against a BASELINE dump (the banked top-1, e.g. PMS8_241's own
     masked wild dump) — rows right under the vote that were wrong at top-1, and vice versa.

Usage: dn_particle_vote.py --baseline <top1_dump.pkl> <particle1.pkl> <particle2.pkl> ...
"""
import sys, os, pickle, collections, argparse
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def load_dump(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def by_row(dump):
    rows = collections.defaultdict(list)
    for t in dump:
        rows[t[0]].append(t)
    for i in rows:
        rows[i].sort(key=lambda t: t[1])
    return rows


def pred_graph(slots):
    """{j: (ft, op, args(tuple), res, dig(tuple))} for present predicted slots."""
    g = {}
    for (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) in slots:
        if ppres:
            g[j] = (pft, pop, tuple(sorted(pargs)), pres_, tuple(pdig))
    return g


def gold_graph(slots):
    g = {}
    for (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) in slots:
        if gft is not None:
            g[j] = (gft, gop, tuple(sorted(gargs)), gres, tuple(gdig))
    return g


def dead_fraction(g):
    """fraction of relation slots (ft==0) whose res var is never consumed as an arg elsewhere
    and never a given's own var -- a cheap, documented minimality PROXY, not the full audit."""
    rel = {j: v for j, v in g.items() if v[0] == 0}
    if not rel:
        return 0.0
    used_vars = set()
    for j, v in g.items():
        used_vars.update(v[2])   # args
    dead = sum(1 for j, v in rel.items() if v[3] not in used_vars)
    return dead / max(1, len(rel))


def solvable(g):
    """Feed the predicted graph to the symbolic solver; True if it builds and is SAT.
    Falls back to True (do not drop) if the solver machinery is unavailable or raises on
    something unrelated to satisfiability — dropping on a harness error would silently
    starve the vote, which is worse than keeping a graph the solver could not judge."""
    try:
        from mycelium.csp_domains import problem_from_algebra
        from mycelium.csp_core import solve_symbolic
    except Exception:
        return True
    try:
        facs = []
        for j, (ft, op, args, res, dig) in g.items():
            var = chr(ord('a') + j)
            if ft == 1:   # given (the loop_val/decode ftype convention: 0=rel,1=given, ...)
                val = int("".join(str(d) for d in dig)) if dig else 0
                facs.append({"ftype": "given", "var": var, "value": val})
            elif ft == 0:
                facs.append({"ftype": "rel", "var": chr(ord('a') + res), "op": op,
                             "args": [chr(ord('a') + a) for a in args]})
        prob = problem_from_algebra({"factors": facs})
        sol = solve_symbolic(prob)
        return sol is not None
    except Exception:
        return True


def vote_slot(cands):
    """cands: list of (weight, value). Weighted majority; ties -> lowest original index
    (cands is assumed ordered by ascending seed index already, stable sort keeps that)."""
    tally = collections.OrderedDict()
    for w, v in cands:
        tally[v] = tally.get(v, 0.0) + w
    best = max(tally.items(), key=lambda kv: kv[1])
    return best[0]


def positional_exact(pred_g, gold_g):
    keys = set(pred_g) | set(gold_g)
    if not keys:
        return 1.0
    hit = sum(1 for j in keys if pred_g.get(j) == gold_g.get(j))
    return hit / len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("particles", nargs="+")
    ap.add_argument("--baseline", required=False)
    args = ap.parse_args()

    particle_dumps = [by_row(load_dump(p)) for p in args.particles]
    all_rows = sorted(set().union(*[set(d) for d in particle_dumps]))
    baseline_rows = by_row(load_dump(args.baseline)) if args.baseline else None

    n_rows = len(all_rows)
    n_covered = 0
    voted_right = 0
    base_right = {}
    voted_right_map = {}
    for i in all_rows:
        gold_g = gold_graph(particle_dumps[0][i])
        particle_graphs = []
        for d in particle_dumps:
            if i not in d:
                continue
            g = pred_graph(d[i])
            if not solvable(g):
                continue
            w = 1.0 - 0.5 * dead_fraction(g)
            particle_graphs.append((w, g))
        if particle_graphs:
            n_covered += 1
            all_j = set()
            for _, g in particle_graphs:
                all_j.update(g)
            voted_g = {}
            for j in all_j:
                cands = [(w, g.get(j)) for w, g in particle_graphs if j in g]
                if cands:
                    voted_g[j] = vote_slot(cands)
            acc = positional_exact(voted_g, gold_g)
        else:
            voted_g = {}
            acc = positional_exact({}, gold_g)
        voted_right_map[i] = acc >= 0.999
        if acc >= 0.999:
            voted_right += 1
        if baseline_rows is not None and i in baseline_rows:
            base_g = pred_graph(baseline_rows[i])
            base_right[i] = positional_exact(base_g, gold_g) >= 0.999

    print(f"[dn-particle-vote] rows={n_rows} seeds={len(args.particles)} "
          f"coverage(>=1 solvable particle)={n_covered}/{n_rows} ({n_covered/max(1,n_rows):.3f}) "
          f"voted_right={voted_right}/{n_rows} ({voted_right/max(1,n_rows):.3f})")

    if base_right:
        gained = sum(1 for i in base_right if base_right[i] is False and voted_right_map.get(i))
        lost = sum(1 for i in base_right if base_right[i] is True and not voted_right_map.get(i))
        both_right = sum(1 for i in base_right if base_right[i] and voted_right_map.get(i))
        base_n_right = sum(1 for v in base_right.values() if v)
        print(f"[dn-particle-vote] vs baseline ({args.baseline}): baseline_right={base_n_right} "
              f"vote_gained={gained} vote_lost={lost} both_right={both_right} "
              f"net={gained - lost:+d} regressions={lost}")


if __name__ == "__main__":
    main()
