"""gsm_type_forest.py -- STEP B1-2 (ledger 2026-10-29 17:07, Bryce "word given for the whole queue"): THE TYPE FOREST from the answer
key, and its CEILING as a solver predicate. Zero-training, CPU.
NOUN of a numeral word: '$'-prefixed -> money; followed by '%' / 'percent', or a scalar word (half, twice, ...) -> DIM (dimensionless);
else the spaCy head of a nummod/quantmod numeral, else the first NOUN within 3 tokens (lemma; dollar / cent / buck -> money); else
untyped. Constants are DIM. FOREST: over TRAIN gold programs (teacher picks = the first member of each valid set), a + / - step whose
operands are both typed (not DIM) UNIONS their classes (result = that class); a x / / step with one DIM side keeps the other's class,
else its result is untyped. PREDICATE: a step is REJECTED iff its op is + / - and both operand classes are CONCRETE (the noun took part
in >= 1 TRAIN union) and differ; unseen nouns are wildcards.
Writes .cache/gsm_type_forest.json (classes, merges, concrete nouns). With .cache/gsmb_picks_VANILLA_s0.json present (gsm_pred_dump.py),
reads B-CEIL on TEST and writes .cache/gsm_type_ceiling.txt."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_trunk_states as GT

MINPROB = int(os.environ.get("TF_MINPROB", "1"))     # RESCUE (ledger 17:20): an edge unions only when >= MINPROB distinct TRAIN problems carry it
SUF = "" if MINPROB == 1 else f"_k{MINPROB}"
OUT, PICKS, VERDICT = GD.V1.C(f"gsm_type_forest{SUF}.json"), GD.V1.C("gsmb_picks_VANILLA_s0.json"), GD.V1.C(f"gsm_type_ceiling{SUF}.txt")
SCALAR = {"twice", "double", "doubles", "doubled", "triple", "tripled", "triples", "thrice", "half", "halves", "quarter", "quarters",
          "third", "thirds"}
MONEY = {"dollar", "cent", "buck", "$"}
DIM = "DIM"


def nouns(q, nlp):
    """-> per bench word: a noun lemma, 'money', DIM, or None (untyped / not a numeral)."""
    doc = nlp(q); spans = [m.span() for m in GD.TOK.finditer(q)]; words = [q[a:b] for a, b in spans]; out = []
    for i, ((a, b), w) in enumerate(zip(spans, words)):
        lw = w.lower(); nxt = words[i + 1].lower() if i + 1 < len(words) else ""
        if not (GD.NUMRX.fullmatch(w) or lw in GD.V1.WORDS): out.append(None); continue
        if w.startswith("$"): out.append("money"); continue
        if lw in SCALAR or nxt in ("%", "percent"): out.append(DIM); continue
        tk = [t for t in doc if t.idx < b and t.idx + len(t) > a][-1]
        hd = tk.head if tk.dep_ in ("nummod", "quantmod") and tk.head.pos_ in ("NOUN", "PROPN") else None
        if hd is None: hd = next((t for t in doc[tk.i + 1:tk.i + 4] if t.pos_ == "NOUN"), None)
        lem = None if hd is None else hd.lemma_.lower()
        out.append("money" if lem in MONEY else lem)
    return out


class UF:
    def __init__(s): s.p = {}; s.merges = []
    def find(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[rb] = ra; s.merges.append((ra, rb))


def first(S):
    return ("t", S["t"][0]) if S["t"] else (("s", S["s"][0]) if S["s"] else ("c", S["c"][0]))


def walk(steps, nn, n, step_fn):
    """propagate types along a program [(op, (kind, idx), (kind, idx))]; step_fn(op, ta, tb) is called on each step; -> step types."""
    ty = []
    def opt(o):
        k, i = o
        if k == "c": return DIM
        if k == "t": return nn[i] if i < n else None
        return ty[i] if i < len(ty) else None
    for op, a, b in steps:
        ta, tb = opt(a), opt(b); step_fn(op, ta, tb)
        if op in "+-": ty.append(ta if ta not in (None, DIM) else (tb if tb not in (None, DIM) else (DIM if DIM in (ta, tb) else None)))
        else: ty.append(tb if ta == DIM else (ta if tb == DIM else None))
    return ty


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    Dt, Q = GT.texts(); NN = {sp: [nouns(q, nlp) for q in Q[sp]] for sp in ("TRAIN", "TEST")}
    uf = UF(); occ = collections.Counter(); concrete = set()
    edges = collections.defaultdict(set)
    for pid, (it, nn) in enumerate(zip(Dt["TRAIN"], NN["TRAIN"])):
        def grow(op, ta, tb):
            for x in (ta, tb):
                if x not in (None, DIM): occ[x] += 1
            if op in "+-" and ta not in (None, DIM) and tb not in (None, DIM) and ta != tb: edges[tuple(sorted((ta, tb)))].add(pid)
            elif op in "+-" and ta not in (None, DIM) and ta == tb: edges[(ta, ta)].add(pid)
        walk([(o, first(A), first(B)) for o, A, B in it["prog"]], nn, len(it["toks"]), grow)
    for (a, b), ps in sorted(edges.items()):
        if len(ps) >= MINPROB: uf.union(a, b); concrete.update((a, b))
    rej = lambda op, ta, tb: op in "+-" and ta in concrete and tb in concrete and uf.find(ta) != uf.find(tb)
    bad = []
    for it, nn in zip(Dt["TRAIN"], NN["TRAIN"]):
        walk([(o, first(A), first(B)) for o, A, B in it["prog"]], nn, len(it["toks"]), lambda op, a, b: bad.append(rej(op, a, b)))
    if MINPROB == 1: assert not any(bad), "TRAIN gold rejected: the forest is inconsistent with its own source"
    cls = collections.Counter(); [cls.update({uf.find(x): c}) for x, c in occ.items() if x in concrete]
    typed = sum(occ.values()); big_root, big = cls.most_common(1)[0]
    L = [f"FOREST: {len(concrete)} concrete nouns in {len(cls)} classes ({len(uf.merges)} merges); typed operand occurrences {typed}",
         f"  largest class '{big_root}' holds {big} = {big / typed:.3f} of typed occurrences (VOID if > 0.50)",
         "  top classes: " + "; ".join(f"{r} ({c}: {', '.join(sorted([x for x in concrete if uf.find(x) == r], key=lambda x: -occ[x])[:6])})"
                                       for r, c in cls.most_common(8))]
    json.dump(dict(parent={x: uf.find(x) for x in concrete}, merges=uf.merges, occ=dict(occ), concrete=sorted(concrete)), open(OUT, "w"))
    tg = []
    for it, nn in zip(Dt["TEST"], NN["TEST"]):
        walk([(o, first(A), first(B)) for o, A, B in it["prog"]], nn, len(it["toks"]), lambda op, a, b: tg.append(rej(op, a, b)))
    gold_rej = float(np.mean(tg)); L.append(f"TRAIN GOLD steps rejected: {np.mean(bad):.4f} (min distinct problems per edge {MINPROB})")
    L.append(f"TEST GOLD steps rejected: {gold_rej:.4f} (n {len(tg)}; bar (c) <= 0.05)")
    if os.path.exists(PICKS):
        P = json.load(open(PICKS)); wrong, right, by_op, wrong_pm = [], [], collections.defaultdict(list), []
        for it, nn, pk in zip(Dt["TEST"], NN["TEST"], P["TEST"]):
            n = len(it["toks"]); steps = []
            for op, a, b in pk["slots"]:
                if op == ".": break
                steps.append((op, tuple(a), tuple(b)))
            r = []; walk(steps, nn, n, lambda op, a, b: r.append(rej(op, a, b)))
            for k, (gop, A, B) in enumerate(it["prog"]):
                if k >= len(pk["slots"]): break
                op, a, b = pk["slots"][k]
                inset = lambda o, S: (o[0] == "t" and o[1] < n and o[1] in S["t"]) or (o[0] == "s" and o[1] in S["s"]) or (o[0] == "c" and o[1] in S["c"])
                ok = op == gop and inset(a, A) and inset(b, B); rk = r[k] if k < len(r) else False
                (right if ok else wrong).append(rk)
                if not ok: by_op[op].append(rk); wrong_pm.append(op in "+-")
        a_, b_ = float(np.mean(wrong)), float(np.mean(right)); sa = len(right) / (len(right) + len(wrong))
        L.append(f"picks: VANILLA_s0 re-run TEST step_all {P['TEST_step_all']:.4f} (recomputed {sa:.4f}; grid 0.1083, |diff| "
                 f"{abs(P['TEST_step_all'] - 0.1083):.4f}{' > 0.01 DISCLOSED' if abs(P['TEST_step_all'] - 0.1083) > 0.01 else ''})")
        L.append(f"(a) WRONG steps rejected: {a_:.4f} (n {len(wrong)}; bar >= 0.20)")
        L.append(f"(b) CORRECT steps rejected: {b_:.4f} (n {len(right)}; bar <= 0.05)")
        L.append(f"read: share of wrong steps whose predicted op is + / -: {np.mean(wrong_pm):.3f}")
        L.append("read: rejected share of wrong steps by predicted op: " + "; ".join(f"{o} {np.mean(v):.3f} (n {len(v)})" for o, v in sorted(by_op.items())))
        void = big / typed > 0.50; ok = a_ >= 0.20 and b_ <= 0.05 and gold_rej <= 0.05
        L.append(f"B-CEIL: {'VOID (over-merged forest)' if void else ('PASS' if ok else 'FAIL')}")
    print("\n".join(L)); open(VERDICT, "w").write("\n".join(L) + "\n")
