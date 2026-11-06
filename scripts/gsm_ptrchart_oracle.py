"""gsm_ptrchart_oracle.py -- THE POINTER-CHART ORACLE (ledger 2026-11-05 18:36, Bryce's word; zero GPU, no model).

Can a Tree chart ADDRESS a step's operand, and does an abstract chart generalize where a lexical one does not? Gold type, gold history,
per pointer (a ceiling). Every gold operand whose teacher operand is a token; candidates = the problem's numeral tokens not consumed by an
earlier gold step (the teacher's first operand); correct = the choice is in the valid set. Tables fit on TRAIN gold; the query context
ctx = (op, kind A, kind B, role), backoff to (role,) when its TRAIN count < 5. Score = log P(chart | ctx) + log P(rank | ctx), ties by text order.
Arms: P0 (rank only) / LEX1 (previous word) / LEX2 (previous + next) / UNITQ (the unit's relation to the question unit) / UNIT (the unit word) /
ST8 / ST64 / ST512 (a nested Tree of RB8 reader states at numeral tokens: k-means 8 per node, three levels, fit on TRAIN numerals, seed 0) /
TREE (the descent: log P(n0 | ctx) + sum_l log P(n_l | n_{l-1}, ctx), stopping at the first node whose (ctx, node) TRAIN count < 5) /
TREE+U (PRIMARY: TREE + UNITQ). Bars PC-C (ST512 gold-chart unique >= 0.50), PC-1 (TREE+U - P0 >= +0.020), PC-L (TREE+U - LEX2 >= +0.020),
PC-G ((LEX2 TRAIN - TEST) - (TREE+U TRAIN - TEST) >= +0.020).  -> .cache/gsm_ptrchart_oracle.json + _verdict.txt
"""
import collections, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import gsm_dto as DTO
from gsm_units import unit_of
from sklearn.cluster import KMeans

Dt, states = BR.TR.load()
SPL = ("TRAIN", "TEST"); NB, A = 8, 0.5


def unitq(u, qu):
    if u is None: return "unk"
    if u == "1": return "dim"
    if u == "%": return "pct"
    if "/" in u:
        x, y = u.split("/", 1)
        return "rate_qnum" if x == qu else "rate_qden" if y == qu else "rate"
    return "plain_noq" if qu is None else "eq" if u == qu else "other"


def rows(it):
    """-> per token-operand: ctx, valid set, candidates (token idx in text order)"""
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, out = set(), []
    for op, Aa, Bb in it["prog"]:
        ks = "".join(BR.TR.first(X)[0] for X in (Aa, Bb))
        for r, X in enumerate((Aa, Bb)):
            if BR.TR.first(X)[0] == "t":
                out.append(dict(ctx=(op, ks, r), valid=set(X["t"]), cand=[i for i in num if i not in cons]))
        for X in (Aa, Bb):
            f = BR.TR.first(X)
            if f[0] == "t": cons.add(f[1])
    return out


# ---- per-token charts
def lex(it, i):
    t = it["toks"]; return (t[i - 1] if i > 0 else "^"), (t[i + 1] if i + 1 < len(t) else "$")


feat = {}
for sp in SPL:
    for g, it in enumerate(Dt[sp]):
        qu = DTO.question_unit(it)
        for i, v in enumerate(it["vals"]):
            if v is None: continue
            p, n = lex(it, i); u = unit_of(it, i)
            feat[sp, g, i] = dict(LEX1=p, LEX2=(p, n), UNIT=u, UNITQ=unitq(u, qu))

# the nested Tree over RB8 states (L2-normalized), fit on TRAIN numerals
keys = {sp: [(g, i) for g, it in enumerate(Dt[sp]) for i, v in enumerate(it["vals"]) if v is not None] for sp in SPL}
X = {sp: np.stack([states[Dt[sp][g]["st0"] + i] for g, i in keys[sp]]).astype(np.float32) for sp in SPL}
for sp in SPL: X[sp] /= np.linalg.norm(X[sp], axis=1, keepdims=True) + 1e-6
path = {sp: np.zeros((len(keys[sp]), 3), np.int64) for sp in SPL}
node = {(): None}


def grow(idx, pre, lvl):
    if lvl == 3: return
    km = KMeans(NB, n_init=1, random_state=0).fit(X["TRAIN"][idx]) if len(idx) >= NB else None
    node[pre] = km
    lab = km.labels_ if km else np.zeros(len(idx), np.int64)
    path["TRAIN"][idx, lvl] = lab
    for c in range(NB): grow(idx[lab == c], pre + (c,), lvl + 1)


grow(np.arange(len(keys["TRAIN"])), (), 0)
for j in range(len(keys["TEST"])):
    pre = ()
    for lvl in range(3):
        km = node.get(pre); c = 0 if km is None else int(km.predict(X["TEST"][j:j + 1])[0]); path["TEST"][j, lvl] = c; pre += (c,)
for sp in SPL:
    for j, (g, i) in enumerate(keys[sp]):
        p = tuple(path[sp][j]); f = feat[sp, g, i]; f["ST8"], f["ST64"], f["ST512"] = p[:1], p[:2], p[:3]

# ---- tables (TRAIN gold)
R = {sp: [(g, r) for g, it in enumerate(Dt[sp]) for r in rows(it)] for sp in SPL}
FLAT = ("LEX1", "LEX2", "UNIT", "UNITQ", "ST8", "ST64", "ST512")
cnt = {a: collections.defaultdict(collections.Counter) for a in FLAT + ("RANK",)}
tree = collections.defaultdict(collections.Counter)            # (ctx, parent node) -> child counts
nctx = collections.Counter()
for g, r in R["TRAIN"]:
    gold = [i for i in r["cand"] if i in r["valid"]]
    if not gold: continue
    i = gold[0]; f = feat["TRAIN", g, i]; rk = min(r["cand"].index(i), 5)
    for c in (r["ctx"], (r["ctx"][2],)):
        nctx[c] += 1; cnt["RANK"][c][rk] += 1
        for a in FLAT: cnt[a][c][f[a]] += 1
        for l in range(3): tree[c, f["ST512"][:l]][f["ST512"][l]] += 1
V = {a: max(1, len({k for C in cnt[a].values() for k in C})) for a in cnt}


def ctxof(c): return c if nctx[c] >= 5 else (c[2],)


def lp(a, c, k):
    C = cnt[a][c]; return math.log((C[k] + A) / (sum(C.values()) + A * V[a]))


def ltree(c, p):
    s = 0.0
    for l in range(3):
        C = tree[c, p[:l]]; n = sum(C.values())
        if n < 5: break
        s += math.log((C[p[l]] + A) / (n + A * NB))
    return s


ARMS = ("P0",) + FLAT + ("TREE", "TREE+U")


def score(a, sp, g, r, i):
    c = ctxof(r["ctx"]); f = feat[sp, g, i]; s = lp("RANK", c, min(r["cand"].index(i), 5))
    if a in FLAT: s += lp(a, c, f[a])
    if a in ("TREE", "TREE+U"): s += ltree(c, f["ST512"])
    if a == "TREE+U": s += lp("UNITQ", c, f["UNITQ"])
    return s


o = {}
for sp in SPL:
    o[f"n_{sp}"] = len(R[sp]); o[f"no_valid_cand_{sp}"] = sum(not (set(r["cand"]) & r["valid"]) for _, r in R[sp]) / len(R[sp])
    o[f"mean_cand_{sp}"] = float(np.mean([len(r["cand"]) for _, r in R[sp]]))
    for a in ARMS:
        ok = 0
        for g, r in R[sp]:
            if not r["cand"]: continue
            best = max(r["cand"], key=lambda i: (score(a, sp, g, r, i), -i)); ok += best in r["valid"]
        o[f"{a}_{sp}"] = ok / len(R[sp])
    for a in FLAT:   # the ceiling: some valid candidate's chart is shared by no non-valid candidate
        u = 0
        for g, r in R[sp]:
            ch = {i: feat[sp, g, i][a] for i in r["cand"]}; bad = {ch[i] for i in r["cand"] if i not in r["valid"]}
            u += any(ch[i] not in bad for i in r["cand"] if i in r["valid"])
        o[f"uniq_{a}_{sp}"] = u / len(R[sp])
o["tree_leaves_train"] = int(len({tuple(p) for p in path["TRAIN"]}))
json.dump(o, open(".cache/gsm_ptrchart_oracle.json", "w"), indent=1)

L = [f"token operands TRAIN {o['n_TRAIN']} TEST {o['n_TEST']} | mean candidates TEST {o['mean_cand_TEST']:.2f} | no valid candidate "
     f"TEST {o['no_valid_cand_TEST']:.4f} | Tree leaves (TRAIN) {o['tree_leaves_train']}"]
L += [f"  {a:6s} TRAIN {o[a + '_TRAIN']:.4f} TEST {o[a + '_TEST']:.4f}" + (f" | gold chart unique TEST {o['uniq_' + a + '_TEST']:.4f}"
      if a in FLAT else "") for a in ARMS]
t = lambda a: o[a + "_TEST"]; gap = lambda a: o[a + "_TRAIN"] - o[a + "_TEST"]
L.append(f"PC-C: ST512 gold chart unique TEST {o['uniq_ST512_TEST']:.4f} (bar >= 0.50) -> {'PASS' if o['uniq_ST512_TEST'] >= 0.50 else 'FAIL'}")
L.append(f"PC-1: TREE+U - P0 TEST {t('TREE+U') - t('P0'):+.4f} (bar +0.020) -> {'PASS' if t('TREE+U') - t('P0') >= 0.020 else 'FAIL'}")
L.append(f"PC-L: TREE+U - LEX2 TEST {t('TREE+U') - t('LEX2'):+.4f} (bar +0.020) -> {'PASS' if t('TREE+U') - t('LEX2') >= 0.020 else 'FAIL'}")
d = gap("LEX2") - gap("TREE+U")
L.append(f"PC-G: LEX2 gap {gap('LEX2'):+.4f} - TREE+U gap {gap('TREE+U'):+.4f} = {d:+.4f} (bar +0.020) -> {'PASS' if d >= 0.020 else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_ptrchart_oracle_verdict.txt", "w").write("\n".join(L) + "\n")
