"""gsm_relquery_oracle.py -- THE RELATIONAL QUERY (ledger 2026-11-06 06:18, Bryce's word; zero GPU, no model).

The CHART QUERY's frame (its setup copied verbatim below: rows, candidates, consumed rule, the nested Tree, the question noun) with base =
ST512 (log P(rank | ctx) + log P(leaf | ctx), tables on TRAIN gold). Relations = the cosine of the candidate's RB8 state to a context
token's state in the SAME problem (glue on the fly, nothing stored per chart): R-QN the question noun / R-PREV the most recently consumed
token / R-SIB (role b) operand a's gold token; R-ALL (PRIMARY) all three. A conditional logit over each row's candidates (features: the base
score + relation x role), fit on TRAIN gold by L-BFGS (<= 7 parameters), applied to TEST.
Bars RQ-1 (R-ALL - ST512 >= +0.020), RQ-G (R-ALL TRAIN - TEST <= 0.05), RQ-H (R-ALL >= 0.6876).  -> .cache/gsm_relquery_oracle.json + _verdict.txt
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

# ---- the base (tables on TRAIN gold, as the CHART QUERY's ST512)
R = {sp: [(g, r) for g, it in enumerate(Dt[sp]) for r in rows(it)] for sp in SPL}
cnt = {a: collections.defaultdict(collections.Counter) for a in ("RANK", "ST512")}
nctx = collections.Counter()
for g, r in R["TRAIN"]:
    gold = [i for i in r["cand"] if i in r["valid"]]
    if not gold: continue
    i = gold[0]; rk = min(r["cand"].index(i), 5)
    for c in (r["ctx"], (r["ctx"][2],)):
        nctx[c] += 1; cnt["RANK"][c][rk] += 1; cnt["ST512"][c][feat["TRAIN", g, i]["ST512"]] += 1
V = {a: max(1, len({k for C in cnt[a].values() for k in C})) for a in cnt}


def ctxof(c): return c if nctx[c] >= 5 else (c[2],)


def lp(a, c, k):
    C = cnt[a][c]; return math.log((C[k] + A) / (sum(C.values()) + A * V[a]))


# ---- per-row candidate features: [base, qn_a, qn_b, prev_a, prev_b, sib_b]
pos = {sp: {k: j for j, k in enumerate(keys[sp])} for sp in SPL}
vec = lambda sp, g, i: X[sp][pos[sp][g, i]]
NAMES = ("base", "qn_a", "qn_b", "prev_a", "prev_b", "sib_b")


def feats(sp, g, r):
    c = ctxof(r["ctx"]); role = r["ctx"][2]; F = np.zeros((len(r["cand"]), 6), np.float32)
    for j, i in enumerate(r["cand"]):
        v = vec(sp, g, i)
        F[j, 0] = lp("RANK", c, min(j, 5)) + lp("ST512", c, feat[sp, g, i]["ST512"])
        F[j, 1 + role] = float(v @ QX[sp][g])
        if r["prev"] is not None: F[j, 3 + role] = float(v @ vec(sp, g, r["prev"]))
        if role == 1 and r["sib"] is not None: F[j, 5] = float(v @ vec(sp, g, r["sib"]))
    return F


D = {sp: [(feats(sp, g, r), np.array([i in r["valid"] for i in r["cand"]])) for g, r in R[sp] if r["cand"]] for sp in SPL}
N = {sp: len(R[sp]) for sp in SPL}
ARMS = {"ST512": [0], "R-QN": [0, 1, 2], "R-PREV": [0, 3, 4], "R-SIB": [0, 5], "R-ALL": [0, 1, 2, 3, 4, 5]}


def fit(cols):
    import torch
    rows_ = [(torch.as_tensor(F[:, cols]), torch.as_tensor(y)) for F, y in D["TRAIN"] if y.any()]
    w = torch.zeros(len(cols), requires_grad=True); w.data[0] = 1.0
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        l = -sum(torch.logsumexp((F @ w)[y], 0) - torch.logsumexp(F @ w, 0) for F, y in rows_) / len(rows_)
        l.backward(); return l
    opt.step(closure); return w.detach().numpy()


def acc(sp, cols, w):
    ok = 0
    for F, y in D[sp]:
        s = F[:, cols] @ w; ok += bool(y[int(np.argmax(s))])           # ties -> the first (text order)
    return ok / N[sp]


o = {}
for a, cols in ARMS.items():
    w = np.array([1.0]) if a == "ST512" else fit(cols)
    o[a] = {sp: acc(sp, cols, w) for sp in SPL}; o[a]["w"] = {NAMES[c]: round(float(x), 4) for c, x in zip(cols, w)}
json.dump(o, open(".cache/gsm_relquery_oracle.json", "w"), indent=1)

t = lambda a: o[a]["TEST"]; gap = o["R-ALL"]["TRAIN"] - o["R-ALL"]["TEST"]
L = [f"token operands TRAIN {N['TRAIN']} TEST {N['TEST']}"]
L += [f"  {a:6s} TRAIN {o[a]['TRAIN']:.4f} TEST {o[a]['TEST']:.4f}" + ("" if a == "ST512" else f" | vs ST512 {t(a) - t('ST512'):+.4f}")
      + f" | w {o[a]['w']}" for a in ARMS]
L.append(f"RQ-1: R-ALL - ST512 TEST {t('R-ALL') - t('ST512'):+.4f} (bar +0.020) -> {'PASS' if t('R-ALL') - t('ST512') >= 0.020 else 'FAIL'}")
L.append(f"RQ-G: R-ALL TRAIN - TEST {gap:+.4f} (bar <= 0.05) -> {'PASS' if gap <= 0.05 else 'FAIL'}")
L.append(f"RQ-H: R-ALL TEST {t('R-ALL'):.4f} (bar >= 0.6876) -> {'PASS' if t('R-ALL') >= 0.6876 else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_relquery_oracle_verdict.txt", "w").write("\n".join(L) + "\n")
