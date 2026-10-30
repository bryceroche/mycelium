"""gsm_macro_miner.py -- STEP D (ledger 2026-10-29, Bryce "word given for the whole queue"): mine recurring subgraphs of the GSM8K
pointer programs (gsmb_data.json gold, teacher picks) as macro CANDIDATES for the tree's upper floor. Zero-training, CPU.
Unit: a root step + any subset of its step-operands expanded into their producers, recursively (a tree; a shared step is duplicated),
2..4 ops. Leaves: a question numeral -> 'n', a constant -> 'c<value>', a cut step operand -> 'x'. Canonical form sorts the children
of + and x (commutative). A schema's count = DISTINCT TRAIN problems carrying it. RANK-NEVER-ADMIT: this script ranks and reads; a
macro enters the tree only by hand admission (the quota). Reads: the top-N library's step COVERAGE (a gold step is covered when it
lies inside an instance of a library schema) on TRAIN and TEST. Writes .cache/gsm_macro_candidates.json + .cache/gsm_macro_cov.txt."""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD

TOPN, MAXOPS = int(os.environ.get("MM_TOPN", "20")), 4
OPN = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


def first(S):
    return ("t", S["t"][0]) if S["t"] else (("s", S["s"][0]) if S["s"] else ("c", S["c"][0]))


def shapes(prog, r):
    """-> [(canonical string, n ops, set of step ids)] of every unit rooted at step r."""
    op, A, B = prog[r]
    def leaf(o):
        k, i = o
        if k == "t": return [("n", 0, frozenset())]
        if k == "c": return [(f"c{GD.CONSTS[i]:g}", 0, frozenset())]
        return [("x", 0, frozenset())] + shapes(prog, i)
    out = []
    for la, na, sa in leaf(first(A)):
        for lb, nb, sb in leaf(first(B)):
            n = 1 + na + nb
            if n > MAXOPS: continue
            ch = sorted((la, lb)) if op in "+*" else [la, lb]
            out.append((f"{OPN[op]}({ch[0]},{ch[1]})", n, sa | sb | {r}))
    return out


def instances(it):
    prog = [(o, A, B) for o, A, B in it["prog"]]
    return [(s, ids) for r in range(len(prog)) for s, n, ids in shapes(prog, r) if n >= 2]


if __name__ == "__main__":
    Dt = json.load(open(GD.OUT)); cnt = collections.Counter()
    for it in Dt["TRAIN"]:
        cnt.update({s for s, _ in instances(it)})
    lib = [s for s, _ in cnt.most_common(TOPN)]; L = []
    for sp in ("TRAIN", "TEST"):
        cov = tot = 0; multi = 0
        for it in Dt[sp]:
            hit = set().union(*([ids for s, ids in instances(it) if s in lib] or [set()]))
            cov += len(hit); tot += len(it["prog"]); multi += len(it["prog"]) if len(it["prog"]) > 1 else 0
        L.append(f"{sp}: top-{TOPN} library covers {cov / tot:.3f} of gold steps (n {tot}; {cov / max(multi, 1):.3f} of steps in multi-step programs)")
    L.append(f"distinct schemas (2..{MAXOPS} ops) in TRAIN: {len(cnt)}; with >= 20 problems: {sum(1 for c in cnt.values() if c >= 20)}")
    L.append("library (TRAIN problems): " + "; ".join(f"{s} {cnt[s]}" for s in lib))
    json.dump([[s, c] for s, c in cnt.most_common(200)], open(GD.V1.C("gsm_macro_candidates.json"), "w"))
    print("\n".join(L)); open(GD.V1.C("gsm_macro_cov.txt"), "w").write("\n".join(L) + "\n")
