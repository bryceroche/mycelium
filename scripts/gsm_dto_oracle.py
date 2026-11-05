"""gsm_dto_oracle.py -- THE DTO ORACLE (ledger 2026-11-05 13:43, Bryce's word; zero GPU, no model).

Can the op of each GOLD step be decided from a typed request object (DTO) alone -- operand units, kinds, the question's unit (the return type),
operand values -- with no access to the text? Gold pointers (an oracle ceiling). A lookup table fit on TRAIN gold: exact key, backoff to the
coarser key when the TRAIN count < 5, the majority op last. Units: scripts/gsm_units.py (UNIT-BIND's exponent algebra), slots by the same algebra
over the gold program; the question unit by the OP-DIAG heuristic. Arms D0 (the OP-DIAG class: sanity, must give 0.4598) / D1 / D1R / D2.
Bars DO-1 (D2 TEST op >= 0.4965), DO-2 (D1R - D1 >= +0.020), DO-G (D2 TRAIN - TEST <= 0.05). -> .cache/gsm_dto_oracle.json + _verdict.txt
"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import gsm_units as GU
from gsm_units import content, norm, unit_of

# ---- verbatim from scripts/gsm_opdiag_read.py (the banked OP-DIAG instrument; importing it would run its read): D0's class + the question unit
MONEYQ = {"money", "dollar", "dollars", "$", "cost", "costs", "spend", "spent", "pay", "paid", "earn", "earned", "earns", "make", "makes",
          "made", "save", "saved", "charge", "charges", "owe", "price", "profit", "worth"}


def question_unit(it):
    toks, sid = it["toks"], it["sid"]
    q = [t for t, s in zip(toks, sid) if s == max(sid)]
    for i in range(len(q) - 1):
        if q[i] == "how" and q[i + 1] == "many":
            _, w = content(q, i + 2); return None if w is None else norm(w)
        if q[i] == "how" and q[i + 1] == "much":
            if MONEYQ & set(q): return "dollar"
            _, w = content(q, i + 2); return None if w is None else norm(w)
    return None


def mul(a, b):
    if a is None or b is None: return None
    if a == "1": return b
    if b == "1": return a
    if "/" in a and a.split("/", 1)[1] == b: return a.split("/", 1)[0]
    if "/" in b and b.split("/", 1)[1] == a: return b.split("/", 1)[0]
    return f"{a}*{b}"


def div(a, b):
    if a is None or b is None: return None
    if a == b: return "1"
    if b == "1": return a
    if "/" in b and b.split("/", 1)[0] == a: return b.split("/", 1)[1]
    return f"{a}/{b}"


def add(a, b):
    if a is None or b is None: return a if b is None else b if a is None else None
    if a == b or b == "1": return a
    if a == "1": return b
    return a


def cls(a, b):
    if a is None or b is None: return "unk"
    if a == "1" or b == "1": return "dim"
    if "/" in a and b in a.split("/", 1) or "/" in b and a in b.split("/", 1): return "rate"
    return "same" if a == b else "diff"


def derive(it):
    """-> per gold step (op, unit class, operand sentence id or None); the final unit"""
    out, su = [], []
    for op, A, B in it["prog"]:
        us, ss = [], []
        for X in (A, B):
            if X["t"]: us.append(unit_of(it, X["t"][0])); ss.append(it["sid"][X["t"][0]])
            elif X["s"]: us.append(su[X["s"][0]] if X["s"][0] < len(su) else None)
            else: us.append("1")
        a, b = us
        su.append({"+": add, "-": add, "*": mul, "/": div}[op](a, b))
        out.append((op, cls(a, b), ss[0] if ss else None))
    return out, (su[-1] if su else None)
# ---- end verbatim

OPS = "+-*/"
Dt, _ = BR.TR.load()


def frac(x):
    return {} if x is None else x


def rate_dir(a, b):
    """how A's and B's units relate as rates: A per B / B per A / A over B's numerator / ... ('-' when neither is a rate or unknown)"""
    if a is None or b is None: return "unk"
    neg = lambda u: {w for w, e in u.items() if e < 0}; pos = lambda u: {w for w, e in u.items() if e > 0}
    if neg(a) and neg(a) == pos(b): return "a_per_b"        # (X/Y, Y): x cancels
    if neg(b) and neg(b) == pos(a): return "b_per_a"
    if neg(b) and pos(b) == pos(a): return "a_over_rate"    # (X, X/Y): / gives Y
    if neg(a) and pos(a) == pos(b): return "rate_over_b"
    return "-"


def steps(it):
    """-> per gold step: the DTO fields and the gold op"""
    U, _ = GU.token_units(it); qu = question_unit(it); qe = GU.exps(qu) if qu else None
    sv, su, out = [], [], []
    od = derive(it)[0]
    for k, (op, A, B) in enumerate(it["prog"]):
        us, vs, ks = [], [], []
        for X in (A, B):
            if X["t"]: us.append(U[X["t"][0]]); vs.append(it["vals"][X["t"][0]]); ks.append("t")
            elif X["s"]:
                i = X["s"][0]; us.append(su[i] if i < len(su) else None); vs.append(sv[i] if i < len(sv) else None); ks.append("s")
            else: us.append({}); vs.append(X["c"][0] if X["c"] else None); ks.append("c")
        a, b = us; va, vb = vs
        ret = tuple("?" if qe is None or a is None or b is None else int(GU.combine(o, a, b) == qe) for o in OPS)
        if va is None or vb is None: val = "unk"
        else: val = (">" if va > vb else "<" if va < vb else "="), (vb != 0 and abs(va / vb - round(va / vb)) < 1e-9)
        out.append(dict(op=op, d0=od[k][1], rd=rate_dir(a, b), kinds="".join(ks), last=k == len(it["prog"]) - 1, ret=ret, val=val))
        r = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y if y else None}[op]
        sv.append(None if va is None or vb is None else r(va, vb)); su.append(GU.combine(op, a, b))
    return out


KEYS = {"D0": lambda s: (s["d0"],),
        "D1": lambda s: (s["d0"], s["rd"], s["kinds"], s["last"]),
        "D1R": lambda s: (s["d0"], s["rd"], s["kinds"], s["last"], s["ret"]),
        "D2": lambda s: (s["d0"], s["rd"], s["kinds"], s["last"], s["ret"], s["val"])}
CHAIN = {"D0": ["D0"], "D1": ["D1", "D0"], "D1R": ["D1R", "D1", "D0"], "D2": ["D2", "D1R", "D1", "D0"]}

S = {sp: [s for it in Dt[sp] for s in steps(it)] for sp in ("TRAIN", "TEST")}
tab = {a: collections.defaultdict(collections.Counter) for a in KEYS}
for s in S["TRAIN"]:
    for a, f in KEYS.items(): tab[a][f(s)][s["op"]] += 1
maj = collections.Counter(s["op"] for s in S["TRAIN"]).most_common(1)[0][0]


def pred(arm, s):
    for a in CHAIN[arm]:
        c = tab[a].get(KEYS[a](s))
        if c and sum(c.values()) >= 5: return c.most_common(1)[0][0]
    return maj


FAM = lambda o: "add" if o in "+-" else "mul"
o = dict(n_train=len(S["TRAIN"]), n_test=len(S["TEST"]), majority=maj,
         majority_test=sum(s["op"] == maj for s in S["TEST"]) / len(S["TEST"]))
for arm in KEYS:
    for sp in ("TRAIN", "TEST"):
        p = [(pred(arm, s), s["op"]) for s in S[sp]]
        o[f"{arm}_{sp}"] = sum(x == y for x, y in p) / len(p); o[f"{arm}_{sp}_fam"] = sum(FAM(x) == FAM(y) for x, y in p) / len(p)
    o[f"{arm}_keys"] = len(tab[arm])
o["D2_confusion"] = {f"{y}->{x}": n for (x, y), n in sorted(collections.Counter((pred("D2", s), s["op"]) for s in S["TEST"]).items())}
o["ret_known_test"] = sum(s["ret"][0] != "?" for s in S["TEST"]) / len(S["TEST"])
o["ret_any_match_test"] = sum(1 in s["ret"] for s in S["TEST"]) / len(S["TEST"])
o["val_known_test"] = sum(s["val"] != "unk" for s in S["TEST"]) / len(S["TEST"])
o["rd_share_test"] = dict(collections.Counter(s["rd"] for s in S["TEST"]))
json.dump(o, open(".cache/gsm_dto_oracle.json", "w"), indent=1, default=str)

R = 0.4765
L = [f"steps TRAIN {o['n_train']} TEST {o['n_test']} | majority '{maj}' TEST {o['majority_test']:.4f} | reader (BT4RB8) TEST op 0.4683 / 0.4847"]
L += [f"  {a:4s} op TRAIN {o[a + '_TRAIN']:.4f} TEST {o[a + '_TEST']:.4f} | family TEST {o[a + '_TEST_fam']:.4f} | keys {o[a + '_keys']}" for a in KEYS]
L.append(f"D0 sanity: {o['D0_TEST']:.4f} vs OP-DIAG 0.4598 -> {'OK' if abs(o['D0_TEST'] - 0.4598) < 0.002 else 'MISMATCH'}")
L.append(f"DO-1: D2 TEST op {o['D2_TEST']:.4f} (bar >= 0.4965 = reader mean {R} + 0.020) -> {'PASS' if o['D2_TEST'] >= 0.4965 else 'FAIL'}")
g = o["D1R_TEST"] - o["D1_TEST"]
L.append(f"DO-2: D1R - D1 TEST op {g:+.4f} (bar +0.020) -> {'PASS' if g >= 0.020 else 'FAIL'}")
gap = o["D2_TRAIN"] - o["D2_TEST"]
L.append(f"DO-G: D2 TRAIN - TEST {gap:+.4f} (bar <= 0.05) -> {'PASS' if gap <= 0.05 else 'FAIL'}")
L.append(f"  beside: return type known {o['ret_known_test']:.3f}, some op matches it {o['ret_any_match_test']:.3f}; values known "
         f"{o['val_known_test']:.3f}; rate relations {o['rd_share_test']}")
L.append(f"  D2 confusion gold->pred: {o['D2_confusion']}")
print("\n".join(L)); open(".cache/gsm_dto_oracle_verdict.txt", "w").write("\n".join(L) + "\n")
