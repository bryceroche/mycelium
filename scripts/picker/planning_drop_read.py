"""planning_drop_read.py -- WHY DID PLANNING FALL? (2026-10-16, after the reading-order verdict; word given; zero GPU.)
Three reads on banked artifacts (wild gold jsonl + the LV_DUMP pickles of SRK3_241 and ROK3_241):
A  THE BACKWARD-UNKNOWN CENSUS (gold only, gold order): every relation argument is one of
     GIVEN        a given's var
     PRODUCED     the RESULT of an earlier relation (addressable by its producing step)
     SEEN-ARG     first appeared as an ARGUMENT of an earlier relation, never produced before this one (an unknown
                  the solver works out backward, already introduced)
     FIRST-HERE   first appearance at this relation (a backward unknown born here)
   SEEN-ARG + FIRST-HERE = what a (reading pointer | derivation register) address CANNOT reach: the third kind.
   Also each relation's RESULT: NEW (a fresh var) vs CLOSES (an already-known var: a constraint, not a production).
B  THE LAYOUT COST: per relation slot, its args' mean slot distance (j - the slot where the arg's var first appears)
   in the OLD frame (SRK3's dump) and the RO frame (ROK3's dump); relations pair k-th to k-th (RO keeps the relations'
   order; the gold op is asserted equal). Matched per-slot verdicts (matched_read's rule, per field) bucketed by the
   CHANGE in distance. A drop concentrated where the distance grew = a layout cost; flat = not positional.
C  MINT BY INTERLEAVING (same frame, positional, paired): per row, depth = givens placed AFTER the first relation slot
   (0 = the row is already blocked: both layouts agree). Relation-slot exact/args/res for SRK3 vs ROK3 by depth.
   A drop at depth 0 as large as at depth >= 1 = capacity, not the two-layout confound.
Out: .cache/planning_drop_read.txt"""
import collections
import json
import pickle
import sys

sys.path.insert(0, "scripts")

OUT = []


def say(s=""):
    print(s); OUT.append(s)


def rows_of(path):
    by = collections.defaultdict(list)
    for t in pickle.load(open(path, "rb")):
        by[t[0]].append(t[1:])
    return {i: sorted(s, key=lambda s: s[0]) for i, s in by.items()}


def match_fields(slots):
    """matched_read.match_row's rule, returning per relation slot (op, args, res) instead of the aggregate."""
    gold_g = [s for s in slots if s[1] != 0]; gold_r = [s for s in slots if s[1] == 0]
    pred_g = [s for s in slots if s[6] != 0 and s[11]]; pred_r = [s for s in slots if s[6] == 0 and s[11]]
    vmap, used, per = {}, set(), {}
    for g in gold_g:
        c = [p for p in pred_g if p[0] not in used and p[10] == g[5]] or [p for p in pred_g if p[0] not in used]
        if not c:
            continue
        p = min(c, key=lambda p: abs(p[0] - g[0])); used.add(p[0]); vmap[g[0]] = p[0]
    for g in gold_r:
        k, gop, gargs, gres = g[0], g[2], g[3], g[4]; best = None
        for p in pred_r:
            if p[0] in used:
                continue
            vm = dict(vmap); vm[k] = p[0]
            m_args = all(a in vm for a in gargs) and ((len(gargs) == 1 and p[12] and p[8][0] == vm[gargs[0]])
                                                       or (len(gargs) == 2 and set(p[8]) == {vm[a] for a in gargs}))
            m_res = gres in vm and p[9] == vm[gres]; m_op = p[7] == gop
            sc = (m_op + m_args + m_res, -abs(p[0] - k))
            if best is None or sc > best[0]:
                best = (sc, p, m_op, m_args, m_res)
        if best is None:
            per[k] = (False, False, False); continue
        _, p, m_op, m_args, m_res = best; used.add(p[0]); vmap[k] = p[0]
        per[k] = (m_op, m_args, m_res)
    return per


def intro_slot(slots):
    """var -> the slot where it first appears (given var / relation args then result), in this frame's gold."""
    first = {}
    for s in slots:
        vs = [s[4]] if s[1] != 0 else list(s[3]) + [s[4]]
        for v in vs:
            first.setdefault(v, s[0])
    return first


def census():
    say("== A. THE BACKWARD-UNKNOWN CENSUS (wild gold, derivation order; .cache/wild_admitted_holdout.jsonl)")
    C = collections.Counter(); rows_third = 0
    rows = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
    for row in rows:
        given, produced, seen_arg = set(), set(), set(); third = False
        for f in row["factors"]:
            if f["ftype"] == "given":
                given.add(f["var"]); continue
            args = list(f.get("args", [])); res = f.get("result")
            C["rel"] += 1
            for a in args:
                kind = ("GIVEN" if a in given else "PRODUCED" if a in produced else
                        "SEEN-ARG" if a in seen_arg else "FIRST-HERE")
                C[kind] += 1; C["arg"] += 1; C[f"{f['ftype']}:{kind}"] += 1
                third |= kind in ("SEEN-ARG", "FIRST-HERE")
            if res is not None:
                C["res_" + ("CLOSES" if (res in given or res in produced or res in seen_arg) else "NEW")] += 1
                produced.add(res)
            seen_arg |= {a for a in args if a not in given and a not in produced}
        rows_third += third
    n = C["arg"]
    say(f"  {C['rel']} relations, {n} argument lookups, {len(rows)} rows")
    for k in ("GIVEN", "PRODUCED", "SEEN-ARG", "FIRST-HERE"):
        say(f"    {k:11s} {C[k]:5d}  {C[k] / n:.3f}")
    say(f"  THE THIRD KIND (SEEN-ARG + FIRST-HERE): {(C['SEEN-ARG'] + C['FIRST-HERE']) / n:.3f} of lookups; "
        f"rows needing it {rows_third}/{len(rows)} = {rows_third / len(rows):.3f}")
    say(f"  relation RESULTS: NEW {C['res_NEW']} | CLOSES (constraint on a known var) {C['res_CLOSES']} "
        f"({C['res_CLOSES'] / max(C['res_NEW'] + C['res_CLOSES'], 1):.3f})")
    say("  by ftype: " + " | ".join(f"{k} {v}" for k, v in sorted(C.items()) if ":" in k))


def layout():
    say("\n== B. THE LAYOUT COST (wild; matched per-field verdicts by the change in arg slot distance, old frame -> RO)")
    S, R = rows_of(".cache/dump_wild_SRK3_241.pkl"), rows_of(".cache/dump_wild_ROK3_241.pkl")
    B = collections.defaultdict(collections.Counter); mism = 0
    for i in S:
        s, r = S[i], R[i]
        sr = [x for x in s if x[1] == 0]; rr = [x for x in r if x[1] == 0]
        if len(sr) != len(rr) or any(a[2] != b[2] for a, b in zip(sr, rr)):
            mism += 1; continue
        fs, fr = intro_slot(s), intro_slot(r); ps, pr = match_fields(s), match_fields(r)
        for a, b in zip(sr, rr):
            if not a[3]:
                continue
            ds = sum(a[0] - fs[v] for v in a[3]) / len(a[3]); dr = sum(b[0] - fr[v] for v in b[3]) / len(b[3])
            d = dr - ds
            bk = "same" if abs(d) < 0.5 else ("closer" if d < 0 else ("+0.5..2" if d <= 2 else "+>2"))
            for tag, per, slot in (("S", ps, a[0]), ("R", pr, b[0])):
                op, ar, rs = per.get(slot, (False, False, False))
                B[bk][tag + "ex"] += op and ar and rs; B[bk][tag + "args"] += ar; B[bk][tag + "res"] += rs
            B[bk]["n"] += 1
    say(f"  rows skipped (relation lists differ between frames): {mism}")
    say(f"  {'dist change':12s} {'n':>5s} | {'exact S->R':>15s} | {'args S->R':>15s} | {'res S->R':>15s}")
    for bk in ("closer", "same", "+0.5..2", "+>2"):
        c = B[bk]; n = max(c["n"], 1)
        say(f"  {bk:12s} {c['n']:5d} | " + " | ".join(
            f"{c['S' + f] / n:.3f}->{c['R' + f] / n:.3f} ({(c['R' + f] - c['S' + f]) / n:+.3f})" for f in ("ex", "args", "res")))


def mint():
    say("\n== C. MINT BY INTERLEAVING DEPTH (positional, same frame, paired SRK3_241 -> ROK3_241, relation slots)")
    S, R = rows_of(".cache/dump_mint_SRK3_241.pkl"), rows_of(".cache/dump_mint_ROK3_241.pkl")
    B = collections.defaultdict(collections.Counter)
    for i in S:
        s, r = S[i], R[i]
        rel = [x[0] for x in s if x[1] == 0]
        depth = sum(1 for x in s if x[1] == 1 and rel and x[0] > rel[0])
        bk = "0" if depth == 0 else ("1" if depth == 1 else "2+")
        B[bk]["rows"] += 1
        for a, b in zip(s, r):
            assert a[:6] == b[:6]
            B[bk]["slots"] += 1
            for tag, x in (("S", a), ("R", b)):
                hit = x[11] and x[6] == x[1] and x[9] == x[4]
                hit = hit and (x[7] == x[2] and ((len(x[3]) == 1 and x[12] and x[8][0] == x[3][0]) or (len(x[3]) == 2 and set(x[8]) == set(x[3])))
                               if x[1] == 0 else x[10] == x[5])
                B[bk][tag + "all"] += hit
            if a[1] != 0:
                continue
            B[bk]["n"] += 1
            for tag, x in (("S", a), ("R", b)):
                ar = (len(x[3]) == 1 and x[12] and x[8][0] == x[3][0]) or (len(x[3]) == 2 and set(x[8]) == set(x[3]))
                rs = x[9] == x[4]
                B[bk][tag + "args"] += ar; B[bk][tag + "res"] += rs; B[bk][tag + "ex"] += ar and rs and x[7] == x[2] and x[11]
    say(f"  {'depth':5s} {'rows':>5s} {'rel':>5s} | {'all slots S->R':>22s} | {'rel exact':>22s} | {'args':>22s} | {'res':>22s}")
    for bk in ("0", "1", "2+"):
        c = B[bk]; n = max(c["n"], 1); m = max(c["slots"], 1)
        say(f"  {bk:5s} {c['rows']:5d} {c['n']:5d} | {c['Sall'] / m:.3f}->{c['Rall'] / m:.3f} ({(c['Rall'] - c['Sall']) / m:+.3f}) | "
            + " | ".join(f"{c['S' + f] / n:.3f}->{c['R' + f] / n:.3f} ({(c['R' + f] - c['S' + f]) / n:+.3f})" for f in ("ex", "args", "res")))


if __name__ == "__main__":
    census(); layout(); mint()
    open(".cache/planning_drop_read.txt", "w").write("\n".join(OUT) + "\n")
