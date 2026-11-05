"""OP-DIAG HOW DOES THE READER RECOGNIZE THE OPERATION? (ledger 2026-11-04 20:31, word given). Zero training.

Units are a HEURISTIC (never gold): each text number's unit = the first content token after it (stop list skipped; a comma / question word ends
the scan; "$" is stripped by the tokenizer, so a bare amount after a money word -> dollar; dollar(s) -> dollar; "<num> times" -> "1";
% / percent -> %; word multipliers half/twice/double/triple/... and constants -> "1"; "<unit> per|an|a|every <noun>" -> a rate unit/noun).
Units propagate through the GOLD program (+/- keep, x cancels a rate or carries "1", / of the same unit -> "1", X / (X/Y) -> Y, else X/Y).
Each gold step gets a unit CLASS from its operands (same / dim / rate / diff / unk); P(op | class) is a 5-row table fit on TRAIN gold.
The BT's op = the committed type at each gold step's slot in the BT6-gate loop (BR.LOAD; rho0 = the arm's FB NT2 fit). TEST.
OD_SAMPLE=1: print 30 random TRAIN (number -> unit) pairs with their token windows and exit (the instrument check; no op or BT numbers).
-> .cache/gsmb_opdiag_{TAG}.json
"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule): it sets RD_ARM=TICK, which gsm_trunk_reader reads at import

STOPW = set("of a an the more less fewer each per extra additional other different new old big small large full whole total than as many much "
            "times and or to in on for with at by from his her their its my your our this that these those every only about around over under "
            "just all some few several is are was were be been has have had will would can could - 's \" ( ) : ; x".split())
MONEYC = {"cost", "costs", "pay", "paid", "pays", "earn", "earns", "earned", "save", "saves", "saved", "spend", "spends", "spent", "price",
          "charge", "charges", "charged", "sell", "sells", "sold", "buy", "buys", "bought", "worth", "for", "at", "makes", "made", "salary", "wage"}
MULT = {"half", "twice", "double", "doubled", "triple", "tripled", "thrice", "quarter", "quadruple"}
RATEW = {"per", "an", "a", "every"}
MONEY = {"dollar", "buck"}
MONEYQ = {"money", "dollar", "dollars", "$", "cost", "costs", "spend", "spent", "pay", "paid", "earn", "earned", "earns", "make", "makes",
          "made", "save", "saved", "charge", "charges", "owe", "price", "profit", "worth"}


def sing(w):
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if w.endswith("es") and w[:-2].endswith(("s", "x", "ch", "sh")): return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w


def content(toks, j, lim=3):
    """the first content token in toks[j : j + lim] (stops at a number or sentence punctuation) -> (index, word) or (None, None)"""
    for i in range(j, min(j + lim, len(toks))):
        w = toks[i]
        if w in ("<num>", ".", "?", "!", ",", "how", "what", "if", "then", "so", "when") or w in MULT: return None, None
        if w in STOPW or not any(ch.isalpha() for ch in w): continue
        return i, w
    return None, None


def norm(w):
    w = sing(w)
    if w in MONEY or w == "dollars": return "dollar"
    if w == "cents": return "cent"
    return w


def unit_of(it, i):
    toks = it["toks"]
    if toks[i] in MULT: return "1"
    nx = toks[i + 1] if i + 1 < len(toks) else ""
    if nx in ("%", "percent"): return "%"
    if nx == "times": return "1"
    if (i > 0 and toks[i - 1] == "$") or nx in ("dollar", "dollars"):
        u, j = "dollar", i + (2 if nx in ("dollar", "dollars") else 1)
    else:
        j0, w = content(toks, i + 1)
        if w is None:                                    # "$" is stripped by the tokenizer: a bare amount after a money word is dollars
            return "dollar" if MONEYC & set(toks[max(0, i - 3):i]) else None
        u, j = norm(w), j0 + 1
    if j < len(toks) and toks[j] in RATEW:
        _, y = content(toks, j + 1, 2)
        if y is not None: u = f"{u}/{norm(y)}"
    return u


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


Dt, states = BR.TR.load()
FAM = lambda o: "add" if o in "+-" else "mul"

if os.environ.get("OD_SAMPLE"):
    rng = np.random.default_rng(0); n = 0
    for k in rng.permutation(len(Dt["TRAIN"])):
        it = Dt["TRAIN"][int(k)]
        pos = [i for i, v in enumerate(it["vals"]) if v is not None]
        if not pos: continue
        i = int(rng.choice(pos))
        print(f"  {' '.join(it['toks'][max(0, i - 2):i + 5])!r:70s} -> {unit_of(it, i)}")
        n += 1
        if n == 30: break
    sys.exit(0)

# ---- the unit table (TRAIN gold) and the TEST unit reads
tab = collections.defaultdict(collections.Counter); opc = collections.Counter()
for it in Dt["TRAIN"]:
    for op, c, _ in derive(it)[0]: tab[c][op] += 1; opc[op] += 1
maj = opc.most_common(1)[0][0]
upred = {c: t.most_common(1)[0][0] for c, t in tab.items()}
bigr = set()
for it in Dt["TRAIN"]: bigr.update(zip(it["toks"], it["toks"][1:]))

# ---- the BT's committed op per gold step
import gsm_beam_read as GB
assert BR.TR.ARM == "TICK" and BR.TR.TICK and BR.TR.NOTE and BR.CYCLOCK and BR.T6OFF
rho = float(json.load(open(f".cache/gsmb_fbeam_{GB.TAG}_nt2.json"))["rho0"])
_rec, _sc = {}, GB.slot_cands
GB.slot_cands = lambda b, k, dec, tv, ti, alt: (_rec.__setitem__((b, k), int(ti[b, k, 0])), _sc(b, k, dec, tv, ti, alt))[1]
its = Dt["TEST"]; rows = []
for s0 in range(0, len(its), 128):
    part = its[s0:s0 + 128]; _rec.clear()
    GB.gate_batch(part, rho, s0)
    for b, it in enumerate(part):
        steps, fu = derive(it); qu = question_unit(it)
        for k, (op, c, sd) in enumerate(steps):
            ty = _rec.get((b, k), GB.STOP); bo = BR.TYPES[ty][0] if ty != GB.STOP else "."
            nov = None
            if sd is not None:
                sent = [t for t, s in zip(it["toks"], it["sid"]) if s == sd]; bg = list(zip(sent, sent[1:]))
                nov = float(np.mean([g not in bigr for g in bg])) if bg else None
            rows.append(dict(op=op, cls=c, bt=bo, nov=nov, last=k == len(steps) - 1, fu=fu, qu=qu))
op = np.array([r["op"] for r in rows]); bt = np.array([r["bt"] for r in rows]); cl = np.array([r["cls"] for r in rows])
un = np.array([upred.get(c, maj) for c in cl])
fam = lambda a: np.array([FAM(x) if x in "+-*/" else "none" for x in a])
o = dict(tag=GB.TAG, steps=len(rows), bt_op=float((bt == op).mean()), majority_op=maj, majority_acc=float((op == maj).mean()),
         unit_op=float((un == op).mean()), bt_fam=float((fam(bt) == fam(op)).mean()), unit_fam=float((fam(un) == fam(op)).mean()),
         coverage=float((cl != "unk").mean()), table={c: dict(t) for c, t in tab.items()}, class_share=dict(collections.Counter(cl.tolist())))
w = (fam(bt) != fam(op)) & (cl != "unk")
o["complement"] = float((fam(un)[w] == fam(op)[w]).mean()) if w.any() else float("nan"); o["n_complement"] = int(w.sum())
L = [r for r in rows if r["last"] and r["fu"] is not None and r["qu"] is not None]
o["q_match"] = float(np.mean([r["fu"] == r["qu"] for r in L])) if L else float("nan"); o["n_q"] = len(L)
o["q_known"] = float(np.mean([r["qu"] is not None for r in rows if r["last"]]))
nv = [(r["nov"], r["bt"] == r["op"]) for r in rows if r["nov"] is not None]
nv.sort(key=lambda x: x[0]); th = len(nv) // 3
o["nov_tertiles"] = [float(np.mean([x[1] for x in part])) for part in (nv[:th], nv[th:2 * th], nv[2 * th:])]
o["nov_means"] = [float(np.mean([x[0] for x in part])) for part in (nv[:th], nv[th:2 * th], nv[2 * th:])]
conf = collections.Counter(zip(op.tolist(), bt.tolist())); o["confusion"] = {f"{a}->{b}": n for (a, b), n in sorted(conf.items())}
print(f"[od] {GB.TAG} TEST steps {o['steps']} | BT op {o['bt_op']:.4f} | majority ('{maj}') {o['majority_acc']:.4f} | unit-class op {o['unit_op']:.4f} "
      f"| family: BT {o['bt_fam']:.4f} units {o['unit_fam']:.4f} | unit coverage {o['coverage']:.3f}", flush=True)
print(f"[od] {GB.TAG} complement: units right on {o['complement']:.3f} of the {o['n_complement']} BT-family-wrong steps with a known class | "
      f"final unit == question unit {o['q_match']:.3f} (n {o['n_q']}; question unit found {o['q_known']:.3f})", flush=True)
print(f"[od] {GB.TAG} novelty tertiles (mean bigram novelty {', '.join(f'{x:.3f}' for x in o['nov_means'])}): BT op "
      f"{', '.join(f'{x:.4f}' for x in o['nov_tertiles'])}", flush=True)
print(f"[od] table P(op | class) on TRAIN: " + " | ".join(f"{c}: {dict(t)}" for c, t in sorted(tab.items())), flush=True)
print(f"[od] confusion gold->BT: {o['confusion']}", flush=True)
json.dump(o, open(f".cache/gsmb_opdiag_{GB.TAG}.json", "w"), indent=1, default=float)
