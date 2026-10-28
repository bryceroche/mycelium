"""rel_dsl_census.py -- THE RELATION-DSL TASK, feasibility census (ledger 2026-10-28, word given; zero GPU, zero training).
The breathing transformer's real-data task moved BEHIND the bottleneck: it reads the GOLD factor graph as generic symbols (vars =
anonymous indices, given values withheld, factor order shuffled) and proposes, for every derivable var, WHICH FACTOR determines
it. The executor (exact Fractions, the fixed algebra) runs that plan with the gold values; the row scores iff the plan is acyclic,
every step is solvable when it fires, and the query value equals the CUSTODY key (custody_gold.row_gold). Any valid plan counts
(set-valued: no arbitrary single order). Neural proposes, symbolic disposes.
FIXTURE: rows whose factors are all given / rel(add|mul|sub) and whose forced propagation (one unknown per factor) reaches the
query with the key. Everything else is COUNTED, never silently dropped. STRUCTURE = the canonical WL digest of the graph with given
values zeroed (mycelium.canonicalizer.canonical_digest): resampled variants of one problem share it.
This census: survival, depth (propagation rounds to the query), structure repetition (holdout structures present in train), and
the baselines a learner must beat: RESULT (each var takes the factor it is the result of, else the first factor containing it),
FIRST-SOLVABLE GREEDY is the symbolic oracle (scores 1.0 by construction), LOOKUP (train plan for the same structure).
Writes .cache/rel_dsl_rows.jsonl (task rows) + .cache/rel_dsl_census.txt."""
import collections, json, os, sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
from mycelium.custody_gold import row_gold
from mycelium.canonicalizer import canonical_digest
import eq_slot_breath as EB
import eq_slot_grade as EG

OPS = ("add", "mul", "sub")


def fvars(f):
    return [f["var"]] if f["ftype"] == "given" else list(f["args"]) + [f["result"]]


def solve_for(f, v, val):
    """Value of var v from factor f given known values val (all of f's other vars), or None if f cannot determine v."""
    if f["ftype"] == "given": return Fraction(f["value"]) if f["var"] == v else None
    a, r = list(f["args"]), f["result"]; op = f["op"]
    if v == r and v not in a:
        x = [val[u] for u in a]
        return sum(x, Fraction(0)) if op == "add" else (x[0] - sum(x[1:], Fraction(0))) if op == "sub" else _prod(x)
    if v in a and v != r and r in val:
        k = a.count(v); rest = [val[u] for u in a if u != v]; R = val[r]
        if op == "add": return (R - sum(rest, Fraction(0))) / k
        if op == "mul":
            p = _prod(rest)
            if p == 0: return None
            q = R / p
            if k == 1: return q
            if k == 2 and q >= 0:
                s = _isqrt(q); return s
            return None
        if op == "sub" and k == 1:   # a0 - sum(a1..) = R
            if a[0] == v: return R + sum(rest, Fraction(0))
            return val[a[0]] - R - sum((val[u] for u in a[1:] if u != v), Fraction(0))
    return None


def _prod(x):
    p = Fraction(1)
    for t in x: p *= t
    return p


def _isqrt(q):
    import math
    if q.denominator != 1: return None
    s = math.isqrt(q.numerator); return Fraction(s) if s * s == q.numerator else None


def propagate(facs):
    """Forced propagation in parallel rounds. Returns (val, rnd, ok) where ok[v] = set of factor ids that determine v at its
    optimal round (the set-valued target); rnd[v] = round (givens 0)."""
    val, rnd, ok = {}, {}, {}
    for i, f in enumerate(facs):
        if f["ftype"] == "given": val[f["var"]] = Fraction(f["value"]); rnd[f["var"]] = 0; ok.setdefault(f["var"], set()).add(i)
    t = 0
    while True:
        t += 1; new = {}
        for i, f in enumerate(facs):
            if f["ftype"] == "given": continue
            unk = {u for u in fvars(f) if u not in val}
            if len(unk) != 1: continue
            v = unk.pop(); x = solve_for(f, v, val)
            if x is None: continue
            if v in new and new[v][0] != x: return val, rnd, ok, "conflict"
            new.setdefault(v, (x, set()))[1].add(i)
        if not new: return val, rnd, ok, None
        for v, (x, fs) in new.items(): val[v] = x; rnd[v] = t; ok[v] = fs


def execute(facs, plan, query):
    """plan: {var: factor id}. Fires factors as their inputs become known; returns the query value or None."""
    val = {f["var"]: Fraction(f["value"]) for f in facs if f["ftype"] == "given"}; pending = {v: i for v, i in plan.items() if v not in val}
    while pending:
        fired = False
        for v, i in list(pending.items()):
            f = facs[i]
            if v not in fvars(f): return None
            if any(u not in val for u in fvars(f) if u != v): continue
            x = solve_for(f, v, val)
            if x is None: return None
            val[v] = x; del pending[v]; fired = True
        if not fired: return None   # a cycle or a step whose inputs never arrive
    return val.get(query)


def task_row(smp, split, row):
    facs = [{k: f[k] for k in ("ftype", "var", "value", "op", "args", "result") if k in f} for f in smp["factors"]]
    if any(f["ftype"] not in ("given", "rel") or (f["ftype"] == "rel" and f["op"] not in OPS) for f in facs): return None, "ftype"
    try: gold = row_gold(smp)
    except (KeyError, ValueError, IndexError): return None, "nogold"
    q = smp["query_var"]; val, rnd, ok, err = propagate(facs)
    if err: return None, err
    if q not in val: return None, "not_forced"
    if val[q] != gold: return None, "off_key"
    sdig = canonical_digest([dict(f, value=0) if f["ftype"] == "given" else f for f in facs], q, smp.get("n_vars", 24))
    der = sorted(v for v in val if rnd[v] > 0)
    return dict(split=split, row=row, n_vars=smp["n_vars"], query=q, gold=gold, facs=facs, sdig=sdig, pkey=EB.problem_key(smp),
                src=(smp.get("gen") or {}).get("src") if isinstance(smp.get("gen"), dict) else None,
                der=der, ok={v: sorted(ok[v]) for v in der}, rnd={v: rnd[v] for v in val}, depth=rnd[q]), None


def result_plan(t):
    plan = {}
    for v in t["der"]:
        rs = [i for i, f in enumerate(t["facs"]) if f["ftype"] == "rel" and f["result"] == v]
        an = [i for i, f in enumerate(t["facs"]) if f["ftype"] == "rel" and v in fvars(f)]
        plan[v] = (rs or an)[0]
    return plan


if __name__ == "__main__":
    OUT = ".cache/rel_dsl_census.txt"; ROWS = ".cache/rel_dsl_rows.jsonl"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    T = {}
    for split in ("train", "holdout"):
        drop = collections.Counter(); T[split] = []
        for i, l in enumerate(open(EG.MIX[split])):
            t, why = task_row(json.loads(l), split, i)
            if t: T[split].append(t)
            else: drop[why] += 1
        n = len(T[split]) + sum(drop.values())
        say(f"{split}: rows {n} | FIXTURE {len(T[split])} ({len(T[split]) / n:.3f}) | dropped {dict(drop)}")
    with open(ROWS, "w") as f:
        for s in ("train", "holdout"):
            for t in T[s]: f.write(json.dumps(dict(t, ok={str(k): v for k, v in t["ok"].items()}, rnd={str(k): v for k, v in t["rnd"].items()},
                                                  gold=str(t["gold"]))) + "\n")
    tr, ho = T["train"], T["holdout"]
    for s, R in (("train", tr), ("holdout", ho)):
        dep = collections.Counter(min(t["depth"], 8) for t in R)
        amb = sum(any(len(t["ok"][v]) > 1 for v in t["der"]) for t in R) / len(R)
        inv = sum(any(all(t["facs"][i]["result"] != v for i in t["ok"][v]) for v in t["der"]) for t in R) / len(R)
        rp = sum(execute(t["facs"], result_plan(t), t["query"]) == t["gold"] for t in R) / len(R)
        say(f"{s}: depth to query {dict(sorted(dep.items()))} (8 = 8+) | mean derived vars {sum(len(t['der']) for t in R) / len(R):.2f} | "
            f"rows with a var several factors can determine {amb:.3f} | rows needing an INVERSE use {inv:.3f} | RESULT baseline {rp:.3f}")
    gs = [t for t in tr if t["src"] == "gsm8k"]
    say(f"train structures: {len({t['sdig'] for t in tr})} distinct over {len(tr)} rows; GSM8K {len({t['sdig'] for t in gs})} over "
        f"{len(gs)} rows / {len({t['pkey'] for t in gs})} problems")
    trs = collections.defaultdict(list)
    for t in tr: trs[t["sdig"]].append(t)
    seen = [t for t in ho if t["sdig"] in trs]
    say(f"holdout structures seen in train: {len(seen)}/{len(ho)} = {len(seen) / len(ho):.3f} | holdout depth on UNSEEN structures "
        f"{dict(sorted(collections.Counter(min(t['depth'], 8) for t in ho if t['sdig'] not in trs).items()))}")
    rp_u = [execute(t["facs"], result_plan(t), t["query"]) == t["gold"] for t in ho if t["sdig"] not in trs]
    say(f"RESULT baseline on UNSEEN holdout structures: {sum(rp_u) / max(len(rp_u), 1):.3f} (n {len(rp_u)})")
