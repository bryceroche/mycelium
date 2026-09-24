"""hier_targets_mint.py -- THE HIERARCHICAL TARGETS (2026-09-24, word given; the nested ladder of THE
ALTERNATION SPEC and the 09-24 blog "One job, six resolutions"). Every breath does the WHOLE job graded
at its own resolution; the targets must NEST (a correct coarse answer is never contradicted by the fine
one) so the rungs' gradients agree. From the gold rows, per factor slot:

  ARGS   c1 (coarse):  the multi-hot over variables introduced by PRESENT slots whose inherited entity
                       key set intersects the gold argument's introducer's (the same-entity group — the
                       mixed-bucket census's KEY-INH: a relation is about what its args are about); a
                       gold arg with no key keeps only itself.  fine ⊂ c1 by construction.
         c2 (medium):  c1 restricted to introducers at the SAME CHAIN DEPTH as the gold arg's (given
                       depth 0; a relation 1 + max over its args' introducers) — "the same entity, the
                       same moment"; fine ⊂ c2 ⊂ c1.
         fine:         the staged g_args, untouched.
  DIGITS c1: only the value's most-significant position is graded (a per-position mask);
         c2: the two leading positions;  fine: all N_DIG positions (mask of ones).
  FTYPE  c1: the 3-way collapse {relation, given, other} of the 9-way head (a class map).
Output: .cache/phase1_alg_hier_<split>.npz with g_args_c1, g_args_c2 (n, L_FAC, K_VARS) float32,
g_dig_m1, g_dig_m2 (n, L_FAC, N_DIG) float32, g_ftype_c1 (n, L_FAC) int32, and the row count for the
sha-fence's sibling check. Slots follow the row's factor order (the staged gold's convention).
usage: hier_targets_mint.py <rows.jsonl> <split-name> [N_DIG=7]
"""
import sys
import json

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
import stamp_arg_mentions as SAM
import lexical_identity_census as LIC
import mixed_bucket_census as MBC

L_FAC = K_VARS = 24
FT_INDEX = {"rel": 0, "given": 1}   # build_gold's; everything else is "other" (2) at the coarse level


def own_var(f):
    return SAM.own_var(f)


def depth_of(factors, intro_map):
    memo = {}

    def dep(i, seen):
        if i in memo:
            return memo[i]
        f = factors[i]
        if f["ftype"] == "given":
            memo[i] = 0
            return 0
        ds = []
        for a in SAM.recursion_args(f):
            k = intro_map.get(a)
            if k is not None and k != i and k not in seen:
                ds.append(dep(k, seen | {k}))
        memo[i] = 1 + max(ds, default=0)
        return memo[i]
    return {i: dep(i, {i}) for i in range(len(factors))}


def row_targets(row, n_dig):
    factors = row["factors"][:L_FAC]
    (text, _f, solution, bounds, sspans, intro_map, memo, cand_key, cand_sent) = LIC.build_candidate_tables(dict(row, factors=factors))
    cmemo = {}
    closure = {i: MBC.arg_closure(i, factors, intro_map, cmemo) for i in range(len(factors))}
    inh = {i: MBC.inherited_keys(i, factors, cand_key, closure) for i in range(len(factors))}
    depth = depth_of(factors, intro_map)
    var_of = {i: own_var(f) for i, f in enumerate(factors)}
    a1 = np.zeros((L_FAC, K_VARS), np.float32); a2 = np.zeros((L_FAC, K_VARS), np.float32)
    m1 = np.zeros((L_FAC, n_dig), np.float32); m2 = np.zeros((L_FAC, n_dig), np.float32)
    ft = np.full(L_FAC, 2, np.int32)
    for j, f in enumerate(factors):
        ft[j] = FT_INDEX.get(f["ftype"], 2)
        if f["ftype"] == "given" and f.get("value") is not None:
            v = int(abs(f["value"]))
            s = str(v).zfill(n_dig)[-n_dig:]
            msd = next((i for i, ch in enumerate(s) if ch != "0"), n_dig - 1)
            m1[j, msd] = 1.0
            m2[j, msd] = 1.0
            if msd + 1 < n_dig:
                m2[j, msd + 1] = 1.0
        args = f.get("args") if f["ftype"] in ("rel", "sel") else None
        if not args:
            continue
        for v in args:
            k = intro_map.get(v)
            if k is None or k == j or v is None or v >= K_VARS:
                continue
            a1[j, v] = 1.0; a2[j, v] = 1.0
            K = inh.get(k, set())
            if not K:
                continue
            for c in range(len(factors)):
                if c == j or c == k:
                    continue
                vc = var_of.get(c)
                if vc is None or vc >= K_VARS:
                    continue
                if inh.get(c, set()) & K:
                    a1[j, vc] = 1.0
                    if depth.get(c) == depth.get(k):
                        a2[j, vc] = 1.0
    return a1, a2, m1, m2, ft


def main():
    path, split = sys.argv[1], sys.argv[2]
    n_dig = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    A1 = np.zeros((n, L_FAC, K_VARS), np.float32); A2 = np.zeros_like(A1)
    M1 = np.zeros((n, L_FAC, n_dig), np.float32); M2 = np.zeros_like(M1)
    FT = np.full((n, L_FAC), 2, np.int32)
    for i, r in enumerate(rows):
        try:
            a1, a2, m1, m2, ft = row_targets(r, n_dig)
        except Exception as e:   # a row the census helpers cannot resolve keeps only its fine targets (stated)
            a1 = a2 = None
            ft = np.full(L_FAC, 2, np.int32); m1 = np.zeros((L_FAC, n_dig), np.float32); m2 = m1.copy()
            for j, f in enumerate(r["factors"][:L_FAC]):
                ft[j] = FT_INDEX.get(f["ftype"], 2)
            print(f"[hier] row {i}: {type(e).__name__}: {e} — fine-only", flush=True)
        if a1 is not None:
            A1[i], A2[i] = a1, a2
        M1[i], M2[i], FT[i] = m1, m2, ft
        if i % 5000 == 0 and i:
            print(f"[hier] {i}/{n}", flush=True)
    out = f".cache/phase1_alg_hier_{split}.npz"
    np.savez(out, g_args_c1=A1, g_args_c2=A2, g_dig_m1=M1, g_dig_m2=M2, g_ftype_c1=FT, n=np.int64(n))
    rel = A1.sum(-1) > 0
    print(f"[hier] {out}: rows {n} | relation slots {int(rel.sum())} | mean group size c1 {A1.sum(-1)[rel].mean():.2f}, "
          f"c2 {A2.sum(-1)[rel].mean():.2f} (fine = the gold's ~1.9 args) | slots with c1 > fine {float((A1.sum(-1)[rel] > 2).mean()):.3f} "
          f"| ftype coarse classes {np.bincount(FT.reshape(-1), minlength=3).tolist()}")


if __name__ == "__main__":
    main()
