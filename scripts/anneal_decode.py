"""anneal_decode.py — THE ANNEALED DECODE (2026-09-18, word given). The argmax decode gives one graph; the
median wild row has four wrong slots and the fixes must agree with each other — a coupled energy. Per row,
from the machine's raw heads (CA_RAWDUMP), search over graphs near the argmax:
    E(graph) = -log p_head(graph) + L_UNSAT * [the solver finds no assignment] + L_MULTI * [the assignment is not unique]
with Metropolis moves (a slot's ftype / op / res / one arg / value-in-the-legal-set / presence) under a temperature
annealed T0 -> T1 over N steps; the lowest-energy graph goes to the solver and is graded end to end against the
key. Reported beside the argmax decode: CORRECT / REFUSED / WRONG. THE PINNED BAR: correct up AND wrong not up —
a search that rewards consistency can turn refusals (safe) into consistent wrong answers (dangerous).
Modes: AD_MODE=anneal (default) | sample (K independent draws from the marginals, filtered, max-likelihood).
usage: anneal_decode.py rawdump.pkl [workers]"""
import os, sys, math, random, pickle, collections
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from mycelium.rulebook import legal_values, digits_of

L_UNSAT = float(os.environ.get("AD_L_UNSAT", "8.0")); L_MULTI = float(os.environ.get("AD_L_MULTI", "3.0"))
STEPS = int(os.environ.get("AD_STEPS", "120")); T0 = float(os.environ.get("AD_T0", "1.0")); T1 = float(os.environ.get("AD_T1", "0.05"))
K_SAMPLES = int(os.environ.get("AD_K", "48")); MODE = os.environ.get("AD_MODE", "anneal"); WALL = float(os.environ.get("AD_WALL", "2"))

def lsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
def lsig(x): return -np.logaddexp(0.0, -x)
def lsig_neg(x): return -np.logaddexp(0.0, x)

class Row:
    def __init__(self, r):
        self.i = r["i"]; self.text = r["text"]; self.key = r["key"]; self.q = int(np.argmax(r["q"]))
        self.L = r["ftype"].shape[0]; self.K = r["res"].shape[1]; self.ND = r["dig"].shape[1]
        self.lp_pres = lsig(r["pres"].reshape(self.L)); self.lp_abs = lsig_neg(r["pres"].reshape(self.L))
        self.lp_ft = lsm(r["ftype"]); self.lp_op = lsm(r["op"]); self.lp_res = lsm(r["res"]); self.lp_dig = lsm(r["dig"])
        self.lp_arg = lsig(r["args"]); self.lp_noarg = lsig_neg(r["args"])
        self.lp_dup = lsig(r["dup"].reshape(self.L)) if "dup" in r else np.full(self.L, -0.7); self.lp_nodup = lsig_neg(r["dup"].reshape(self.L)) if "dup" in r else np.full(self.L, -0.7)
        self.legal = legal_values(self.text, self.ND); self.legal_lp = {}
        for j in range(self.L):
            self.legal_lp[j] = {v: sum(self.lp_dig[j, d, dd] for d, dd in enumerate(digits_of(v, self.ND))) for v in self.legal}
    def argmax_state(self):
        st = []
        for j in range(self.L):
            pres = self.lp_pres[j] > self.lp_abs[j]; ft = int(np.argmax(self.lp_ft[j])); res = int(np.argmax(self.lp_res[j]))
            a = np.argsort(-self.lp_arg[j])[:2].tolist(); dup = self.lp_dup[j] > self.lp_nodup[j]
            val = max(self.legal_lp[j], key=self.legal_lp[j].get) if self.legal_lp[j] else 0
            st.append({"pres": bool(pres), "ft": ft, "op": int(np.argmax(self.lp_op[j])), "res": res, "args": [a[0], a[0]] if dup else a, "dup": bool(dup), "val": val})
        return st
    def loglik(self, st):
        s = 0.0
        for j, x in enumerate(st):
            if not x["pres"]: s += self.lp_abs[j]; continue
            s += self.lp_pres[j] + self.lp_ft[j, x["ft"]] + self.lp_res[j, x["res"]]
            if x["ft"] == 0:
                s += self.lp_op[j, x["op"]]
                if x["dup"]: s += self.lp_dup[j] + self.lp_arg[j, x["args"][0]]
                else: s += self.lp_nodup[j] + self.lp_arg[j, x["args"][0]] + self.lp_arg[j, x["args"][1]]
            else:
                s += self.legal_lp[j].get(x["val"], -50.0)
        return s
    def parse(self, st):
        facs = []
        for j, x in enumerate(st):
            if not x["pres"]: continue
            if x["ft"] == 0: facs.append({"ftype": "rel", "op": "add" if x["op"] == 0 else "mul", "args": list(x["args"]), "result": x["res"]})
            else: facs.append({"ftype": "given", "var": x["res"], "value": int(x["val"])})
        return facs
    def move(self, st, rng):
        st = [dict(x) for x in st]; j = rng.randrange(self.L); x = st[j]; kind = rng.choice(["pres", "ft", "op", "res", "arg", "val", "arg"])
        if kind == "pres": x["pres"] = not x["pres"]
        elif kind == "ft": x["ft"] = int(rng.choices(range(self.lp_ft.shape[1]), weights=np.exp(self.lp_ft[j]))[0])
        elif kind == "op": x["op"] = 1 - x["op"]
        elif kind == "res": x["res"] = int(rng.choices(range(self.K), weights=np.exp(self.lp_res[j]))[0])
        elif kind == "arg":
            a = list(x["args"]); a[rng.randrange(2)] = int(rng.choices(range(self.K), weights=np.exp(self.lp_arg[j]))[0]); x["args"] = a; x["dup"] = a[0] == a[1]
        elif kind == "val" and self.legal:
            x["val"] = rng.choices(self.legal, weights=[math.exp(self.legal_lp[j][v]) for v in self.legal])[0]
        return st

def solve(parse, q, key):
    """(status, value, unique) via the June core at the row-sized domain"""
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    from mycelium.doors import certify_unique
    if not parse: return "empty", None, False
    used = [f.get("var") for f in parse if f["ftype"] == "given"] + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
    nvv = max([q + 1] + [v + 1 for v in used if v is not None]); gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
    gmax = max([int(v) for v in gv.values()] + [1]); m = int(min(10001, max(300, 2 * gmax, 2 * (key or 1))))
    try:
        prob = problem_from_algebra3(nvv, parse, gv, m); res = solve_walled(prob, budget=5000, wall=WALL)
    except Exception: return "unbuildable", None, False
    if res.get("status") != "solved": return res.get("status", "?"), None, False
    val = int(res["assignment"][q])
    try: uniq = bool(certify_unique(problem_from_algebra3(nvv, parse, gv, m), q, val, 5000))
    except Exception: uniq = False
    return "solved", val, uniq

def energy(row, st, cache):
    key = tuple((x["pres"], x["ft"], x["op"], x["res"], tuple(x["args"]), x["val"]) for x in st)
    if key not in cache: cache[key] = solve(row.parse(st), row.q, row.key)
    status, val, uniq = cache[key]
    e = -row.loglik(st) + (L_UNSAT if status != "solved" else 0.0) + (L_MULTI if (status == "solved" and not uniq) else 0.0)
    return e, status, val

def run_row(r):
    row = Row(r); rng = random.Random(row.i); cache = {}
    st0 = row.argmax_state(); e0, s0, v0 = energy(row, st0, cache)
    best = (e0, st0, s0, v0)
    if MODE == "sample":
        for _ in range(K_SAMPLES):
            st = st0
            for _ in range(3): st = row.move(st, rng)
            e, s, v = energy(row, st, cache)
            if e < best[0]: best = (e, st, s, v)
    else:
        cur = (e0, st0)
        for t in range(STEPS):
            T = T0 * (T1 / T0) ** (t / max(STEPS - 1, 1)); st = row.move(cur[1], rng); e, s, v = energy(row, st, cache)
            if e < cur[0] or rng.random() < math.exp(-(e - cur[0]) / T):
                cur = (e, st)
                if e < best[0]: best = (e, st, s, v)
    def grade(s, v): return "correct" if (s == "solved" and v == row.key) else ("refused" if s != "solved" else "wrong")
    return row.i, grade(s0, v0), grade(best[2], best[3]), len(cache)

if __name__ == "__main__":
    import multiprocessing as mp
    D = pickle.load(open(sys.argv[1], "rb")); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    base = collections.Counter(); ann = collections.Counter(); moved = collections.Counter(); n_solves = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        it = pool.imap_unordered(run_row, D, chunksize=1); got = 0
        try:
            for _ in range(len(D)):
                i, g0, g1, ns = it.next(timeout=STEPS * WALL * 2 + 60); base[g0] += 1; ann[g1] += 1; moved[(g0, g1)] += 1; n_solves += ns; got += 1
        except mp.TimeoutError:
            print(f"[anneal] {len(D) - got} rows never returned", flush=True)
    n = len(D); tag = os.path.basename(sys.argv[1])
    print(f"[anneal] {tag} mode={MODE} steps={STEPS} L_unsat={L_UNSAT} L_multi={L_MULTI}: rows {n} | ARGMAX correct {base['correct']} refused {base['refused']} wrong {base['wrong']} | ANNEALED correct {ann['correct']} refused {ann['refused']} wrong {ann['wrong']} | transitions {dict(moved)} | solver calls {n_solves}", flush=True)
