"""gsm_chartquery_oracle.py -- THE CHART QUERY (ledger 2026-11-06 06:13, Bryce's word; zero GPU, no model).

The POINTER-CHART ORACLE's frame (rows, candidates, consumed rule, the nested Tree of RB8 states, tables on TRAIN gold, backoff < 5; its setup
copied verbatim below, since importing it would rerun its read) with base = ST512. Each query arm adds the pointwise term
log P(c64 | qkey) - log P(c64 | role) for the candidate's ST64 chart (0 when the qkey's TRAIN count < 5):
  Q-PREV  qkey = (role, ST64 chart of the most recently consumed token; "none" before any)       -- the chart transition
  Q-QN    qkey = (role, the question's chart: k-means 16 on TRAIN states of the question noun)   -- what the problem asks for
  Q-SIB   qkey = (op, ST64 chart of operand a's gold token), role b only                         -- the sibling operand
  Q-ALL   (PRIMARY) base + all three.
Bars QC-1 (Q-ALL - ST512 >= +0.020), QC-G (Q-ALL TRAIN - TEST <= 0.05), QC-H (Q-ALL >= 0.6876).  -> .cache/gsm_chartquery_oracle.json + _verdict.txt
"""
import collections, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import gsm_dto as DTO
from gsm_units import content, unit_of
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
    """-> per token-operand: ctx, valid set, candidates (text order), prev (the most recently consumed token), sib (operand a's token)"""
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, order, out = set(), [], []
    for op, Aa, Bb in it["prog"]:
        ks = "".join(BR.TR.first(X)[0] for X in (Aa, Bb)); fa = BR.TR.first(Aa)
        for r, X in enumerate((Aa, Bb)):
            if BR.TR.first(X)[0] == "t":
                out.append(dict(ctx=(op, ks, r), valid=set(X["t"]), cand=[i for i in num if i not in cons],
                                prev=order[-1] if order else None, sib=fa[1] if r == 1 and fa[0] == "t" else None))
        for X in (Aa, Bb):
            f = BR.TR.first(X)
            if f[0] == "t": cons.add(f[1]); order.append(f[1])
    return out


# ---- per-token charts (as the POINTER-CHART ORACLE)
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


# ---- the question's chart: the question noun's state (content word after "how many / much", else the token before the last "?")
def qnoun(it):
    toks, sid = it["toks"], it["sid"]; q0 = sid.index(max(sid)); q = toks[q0:]
    for i in range(len(q) - 1):
        if q[i] == "how" and q[i + 1] in ("many", "much"):
            j, _ = content(q, i + 2)
            if j is not None: return q0 + j
    last = max(i for i, t in enumerate(toks) if t == "?") if "?" in toks else len(toks) - 1
    return max(0, last - 1)


QX = {sp: np.stack([states[it["st0"] + qnoun(it)] for it in Dt[sp]]).astype(np.float32) for sp in SPL}
for sp in SPL: QX[sp] /= np.linalg.norm(QX[sp], axis=1, keepdims=True) + 1e-6
qkm = KMeans(16, n_init=1, random_state=0).fit(QX["TRAIN"])
QC = {sp: qkm.predict(QX[sp]) for sp in SPL}

# ---- tables (TRAIN gold)
R = {sp: [(g, r) for g, it in enumerate(Dt[sp]) for r in rows(it)] for sp in SPL}
c64 = lambda sp, g, i: feat[sp, g, i]["ST64"] if i is not None else "none"


def qkeys(sp, g, r):
    k = {"Q-PREV": (r["ctx"][2], c64(sp, g, r["prev"])), "Q-QN": (r["ctx"][2], int(QC[sp][g]))}
    if r["ctx"][2] == 1 and r["sib"] is not None: k["Q-SIB"] = (r["ctx"][0], c64(sp, g, r["sib"]))
    return k


QA = ("Q-PREV", "Q-QN", "Q-SIB")
cnt = {a: collections.defaultdict(collections.Counter) for a in ("RANK", "ST512")}
qt = {a: collections.defaultdict(collections.Counter) for a in QA + ("ROLE64",)}
nctx = collections.Counter()
for g, r in R["TRAIN"]:
    gold = [i for i in r["cand"] if i in r["valid"]]
    if not gold: continue
    i = gold[0]; f = feat["TRAIN", g, i]; rk = min(r["cand"].index(i), 5)
    for c in (r["ctx"], (r["ctx"][2],)):
        nctx[c] += 1; cnt["RANK"][c][rk] += 1; cnt["ST512"][c][f["ST512"]] += 1
    qt["ROLE64"][r["ctx"][2]][f["ST64"]] += 1
    for a, k in qkeys("TRAIN", g, r).items(): qt[a][k][f["ST64"]] += 1
V = {a: max(1, len({k for C in cnt[a].values() for k in C})) for a in cnt}
V64 = max(1, len({k for C in qt["ROLE64"].values() for k in C}))


def ctxof(c): return c if nctx[c] >= 5 else (c[2],)


def lp(a, c, k):
    C = cnt[a][c]; return math.log((C[k] + A) / (sum(C.values()) + A * V[a]))


def lq(a, k, ch):
    C = qt[a][k]; n = sum(C.values())
    return math.log((C[ch] + A) / (n + A * V64)) if n >= 5 else None


ARMS = ("ST512",) + QA + ("Q-ALL",)


def score(a, sp, g, r, i, qk):
    c = ctxof(r["ctx"]); f = feat[sp, g, i]
    s = lp("RANK", c, min(r["cand"].index(i), 5)) + lp("ST512", c, f["ST512"])
    base = math.log((qt["ROLE64"][r["ctx"][2]][f["ST64"]] + A) / (sum(qt["ROLE64"][r["ctx"][2]].values()) + A * V64))
    for q in (QA if a == "Q-ALL" else (a,) if a in QA else ()):
        if q in qk:
            t = lq(q, qk[q], f["ST64"])
            if t is not None: s += t - base
    return s


o = {}
for sp in SPL:
    o[f"n_{sp}"] = len(R[sp])
    for a in ARMS:
        ok = 0
        for g, r in R[sp]:
            if not r["cand"]: continue
            qk = qkeys(sp, g, r)
            best = max(r["cand"], key=lambda i: (score(a, sp, g, r, i, qk), -i)); ok += best in r["valid"]
        o[f"{a}_{sp}"] = ok / len(R[sp])
    o[f"qkey_live_{sp}"] = {q: sum(q in qkeys(sp, g, r) and lq(q, qkeys(sp, g, r)[q], ()) is not None for g, r in R[sp]) / len(R[sp])
                            for q in QA}
json.dump(o, open(".cache/gsm_chartquery_oracle.json", "w"), indent=1)

t = lambda a: o[a + "_TEST"]; gap = o["Q-ALL_TRAIN"] - o["Q-ALL_TEST"]
L = [f"token operands TRAIN {o['n_TRAIN']} TEST {o['n_TEST']} | query keys live on TEST {o['qkey_live_TEST']}"]
L += [f"  {a:6s} TRAIN {o[a + '_TRAIN']:.4f} TEST {o[a + '_TEST']:.4f}" + ("" if a == "ST512" else f" | vs ST512 {t(a) - t('ST512'):+.4f}") for a in ARMS]
L.append(f"QC-1: Q-ALL - ST512 TEST {t('Q-ALL') - t('ST512'):+.4f} (bar +0.020) -> {'PASS' if t('Q-ALL') - t('ST512') >= 0.020 else 'FAIL'}")
L.append(f"QC-G: Q-ALL TRAIN - TEST {gap:+.4f} (bar <= 0.05) -> {'PASS' if gap <= 0.05 else 'FAIL'}")
L.append(f"QC-H: Q-ALL TEST {t('Q-ALL'):.4f} (bar >= 0.6876) -> {'PASS' if t('Q-ALL') >= 0.6876 else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_chartquery_oracle_verdict.txt", "w").write("\n".join(L) + "\n")
