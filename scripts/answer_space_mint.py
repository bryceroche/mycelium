"""answer_space_mint.py — THE E1+E3 MINT (2026-08-03; the corpus stage
of the answer-space rollout). Solution-first signed/wide chains through
the STANDING render machinery (render2 — spans/mentions correct by
construction), gated by: solver certify (signed, m=1e6, budget 5000,
exhaustion rejects), tokenizer pin (sign+digits present per literal),
canon knot dedup, 24-slot geometry, and the ALG_WIDE gold builder.
Per-terminal counters printed per the dose law's per-terminal
extension: rows and literals exercising the SIGN head vs DIGITS 4-7.
Class mix: SIGN / WIDE / BOTH round-robin. Seed base 103900."""
import os, sys, json, random, argparse, hashlib
os.environ["ALG_WIDE"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from algebra2_nl_gen import render2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from mycelium.doors import certify_unique
from hash_audit_iso import canon
from tokenizers import Tokenizer
from phase1_algebra_head import TOKENIZER_JSON
tok = Tokenizer.from_file(TOKENIZER_JSON)

M = 10**6


def gen_row(rng, cls):
    """Solution-first: 2-3 givens + 1-3 add/mul chain, class-typed values."""
    n_giv = rng.randint(2, 3)
    vals = []
    for i in range(n_giv):
        if cls == "sign" or (cls == "both" and i == 0):
            vals.append(-rng.randint(1, 300))
        elif cls == "wide" or (cls == "both" and i == 1):
            vals.append(rng.randint(1000, 999_999))
        else:
            vals.append(rng.randint(1, 300))
    facs = [{"ftype": "given", "var": i, "value": vals[i], "surface": "given"}
            for i in range(n_giv)]
    sol = list(vals)
    nxt = n_giv
    for _ in range(rng.randint(1, 3)):
        a, b = rng.sample(range(nxt), 2)
        op = rng.choice(["add", "mul"])
        v = sol[a] + sol[b] if op == "add" else sol[a] * sol[b]
        if abs(v) > M:
            op, v = "add", sol[a] + sol[b]
            if abs(v) > M:
                continue
        facs.append({"ftype": "rel", "op": op, "surface": op,
                     "args": [a, b], "result": nxt})
        sol.append(v)
        nxt += 1
    if nxt == n_giv:
        return None
    return nxt, facs, sol, nxt - 1  # query = last derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=103900)
    ap.add_argument("--out", default=".cache/answer_space_smoke.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen = set()
    rows, rej = [], {"solve": 0, "unique": 0, "tok": 0, "dup": 0, "geom": 0}
    counts = {"sign": 0, "wide": 0, "both": 0}
    lit = {"neg": 0, "wide": 0}
    classes = ["sign", "wide", "both"]
    while len(rows) < a.n:
        cls = classes[len(rows) % 3]
        g = gen_row(rng, cls)
        if g is None:
            continue
        n_vars, facs, sol, query = g
        if n_vars > 24 or len(facs) > 24:
            rej["geom"] += 1; continue
        text, gfactors, mentions, _ = render2(rng, n_vars, facs, query)
        # tokenizer pin: every literal's digits (and sign) present
        dec = tok.decode(tok.encode(text).ids)
        pin = all(str(abs(f["value"])) in dec and
                  (("-" in dec) if f["value"] < 0 else True)
                  for f in gfactors if f["ftype"] == "given")
        if not pin:
            rej["tok"] += 1; continue
        givens = {f["var"]: f["value"] for f in gfactors if f["ftype"] == "given"}
        pr = problem_from_algebra3(n_vars, gfactors, givens, M, signed=True)
        res = solve_symbolic(pr, budget=5000, seed=0)
        if res["status"] != "solved" or \
                [int(res["assignment"][v]) for v in range(n_vars)] != sol:
            rej["solve"] += 1; continue
        pr2 = problem_from_algebra3(n_vars, gfactors, givens, M, signed=True)
        if not certify_unique(pr2, query, sol[query], budget=5000):
            rej["unique"] += 1; continue
        digest, _ = canon({"factors": gfactors, "query_var": query,
                           "n_vars": n_vars})
        if digest in seen:
            rej["dup"] += 1; continue
        seen.add(digest)
        counts[cls] += 1
        lit["neg"] += sum(1 for f in gfactors
                          if f["ftype"] == "given" and f["value"] < 0)
        lit["wide"] += sum(1 for f in gfactors
                           if f["ftype"] == "given" and abs(f["value"]) >= 1000)
        rows.append({"n_vars": n_vars, "m": M, "text": text,
                     "factors": gfactors, "mentions": mentions,
                     "query_var": query, "solution": sol,
                     "decisions": int(res["counters"]["decisions"])
                     if "counters" in res else 0,
                     "gen": {"seed": a.seed, "shape": "answer-space-v0",
                             "cls": cls, "generation": 23}})
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[mint] {len(rows)} rows -> {a.out}  rejects: {rej}")
    print(f"[per-terminal] rows: {counts}  literals: sign-head {lit['neg']}, "
          f"digits-4-7 {lit['wide']}")


if __name__ == "__main__":
    main()
