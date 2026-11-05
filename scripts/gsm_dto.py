"""gsm_dto.py -- THE DTO (ledger 2026-11-05, the SEALED OP HEAD): the typed request object a slot's op is decided from. No text.

Per slot, from its two operands (token / earlier slot / constant) -> 10 field ids in one shared vocabulary (NV), field i drawn from its own
range: kind A, kind B, unit class (unk / dim / rate / same / diff), rate direction (unk / - / a_per_b / b_per_a / a_over_rate / rate_over_b),
the return type per op x4 (does op's result unit equal the question unit: ? / no / yes), value order (unk / > / < / =), divisibility
(unk / b divides a / not). Units: gsm_units (UNIT-BIND's exponent algebra), constants dimensionless; slots' values and units by the
executor over the COMMITTED slots only (an uncommitted operand slot is unknown, at training as at inference). The question unit: the
OP-DIAG heuristic (copied verbatim from scripts/gsm_opdiag_read.py, whose import would run its read). The DTO table this mirrors:
scripts/gsm_dto_oracle.py (D2 without the last-step flag: TEST op 0.5845).
"""
import numpy as np

import gsm_bench_data as GD
import gsm_units as GU
from gsm_units import content, norm

OPS = "+-*/"
SIZES = (3, 3, 5, 6, 3, 3, 3, 3, 4, 3)                      # per field; the ids of field i start at OFF[i]
OFF = np.concatenate([[0], np.cumsum(SIZES)[:-1]]).astype(np.int64); NV = int(sum(SIZES)); NF = len(SIZES)
KINDS = "tsc"; CLS = ("unk", "dim", "rate", "same", "diff"); RD = ("unk", "-", "a_per_b", "b_per_a", "a_over_rate", "rate_over_b")

# ---- verbatim from scripts/gsm_opdiag_read.py: the question unit
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
# ---- end verbatim


def rate_dir(a, b):
    neg = lambda u: {w for w, e in u.items() if e < 0}; pos = lambda u: {w for w, e in u.items() if e > 0}
    if neg(a) and neg(a) == pos(b): return "a_per_b"
    if neg(b) and neg(b) == pos(a): return "b_per_a"
    if neg(b) and pos(b) == pos(a): return "a_over_rate"
    if neg(a) and pos(a) == pos(b): return "rate_over_b"
    return "-"


def _q(it):
    if "_qe" not in it: qu = question_unit(it); it["_qe"] = (GU.exps(qu) if qu else None,)
    return it["_qe"][0]


def known(it, prog, committed):
    """the executor over the committed slots, in slot order -> per slot (value, unit); None = unknown"""
    U, _ = GU.token_units(it); vals = it["vals"]; n = len(vals); K = len(prog)
    sv, su = [None] * K, [None] * K
    for k in range(K):
        op, A, B = prog[k]
        if not committed[k] or op == ".": continue
        (va, ua), (vb, ub) = (operand(it, U, sv, su, committed, X) for X in (A, B))
        su[k] = GU.combine(op, ua, ub)
        if va is not None and vb is not None and not (op == "/" and vb == 0):
            try: sv[k] = va + vb if op == "+" else va - vb if op == "-" else va * vb if op == "*" else va / vb
            except OverflowError: sv[k] = None
    return sv, su


def operand(it, U, sv, su, committed, X):
    kind, i = X; vals = it["vals"]
    if kind == "t": return (vals[i] if i < len(vals) else None), (U[i] if i < len(U) else None)
    if kind == "s": return (sv[i], su[i]) if i < len(sv) and committed[i] else (None, None)
    return GD.CONSTS[i], {}


def fields(it, U, sv, su, committed, A, B):
    """one slot's 10 field ids (global vocabulary)"""
    (va, a), (vb, b) = (operand(it, U, sv, su, committed, X) for X in (A, B))
    qe = _q(it)
    if a is None or b is None: c, rd = "unk", "unk"
    else:
        rd = rate_dir(a, b)
        c = "dim" if (not a or not b) else "rate" if rd != "-" else "same" if a == b else "diff"
    ret = [0 if qe is None or a is None or b is None else 1 + int(GU.combine(o, a, b) == qe) for o in OPS]
    if va is None or vb is None: od, dv = 0, 0
    else:
        od = 1 if va > vb else 2 if va < vb else 3
        dv = 0 if vb == 0 else 1 if abs(va / vb - round(va / vb)) < 1e-9 else 2
    loc = [KINDS.index(A[0]), KINDS.index(B[0]), CLS.index(c), RD.index(rd)] + ret + [od, dv]
    return OFF + np.asarray(loc, np.int64)


def dto_ids(its, progs, committed, cand=None):
    """(B, K, NF) field ids. progs[b][k] = (op, A, B) (the committed program; stops ".");
    cand[b][k] = (A, B) the operands to describe for slot k (default: progs' own); a slot with no operands gets the unknown row."""
    B, K = len(its), len(progs[0]); out = np.tile(OFF, (B, K, 1))
    for b, it in enumerate(its):
        U, _ = GU.token_units(it); sv, su = known(it, progs[b], committed[b])
        for k in range(K):
            A, Bo = cand[b][k] if cand is not None else progs[b][k][1:]
            if A is None or Bo is None: continue
            out[b, k] = fields(it, U, sv, su, committed[b], A, Bo)
    return out
