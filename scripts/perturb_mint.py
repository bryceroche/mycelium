"""perturb_mint.py — THE PERTURBATION MINT (2026-08-22, word given).
The anchor law is the mintability condition: every gold given appears
verbatim in its text, so value substitution is mechanical. Each of the
123 diet rows (wild-val 20 EXCLUDED — their texts are eval surface)
becomes a template: perturb given values, swap the same numbers in the
REAL text (one-pass word-boundary substitution), re-solve, key-gate.
Guards: values colliding with macro/pct params stay fixed; word-anchored
values stay fixed; ambiguous rows emit nothing rather than lie.
"""
import sys, json, re, random, glob
sys.path.insert(0, '.')
from mycelium.macros import expand_graph
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from mycelium.anchor_law import unanchored_givens

def load_seeds():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f):
            r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    skips = set(json.load(open('.cache/book12_anchor_skips.json')))
    wv = set(json.loads(l)["src_idx"] for l in open('.cache/g55_wildval.jsonl'))
    return [v for k, v in sorted(byid.items()) if k not in skips and k not in wv]

def param_values(facs):
    out = set()
    for f in facs:
        if f["ftype"] == "macro":
            for k in ("k1", "k2", "a", "k"):
                if isinstance(f.get(k), int): out.add(f[k])
        if f["ftype"] == "pct": out.add(f["p"])
        if f["ftype"] == "fdiv": out.add(f["k"])
    return out

def variants(row, rng, want=30):
    text = row["original"]; facs = row["factors"]
    givens = sorted(set(f["value"] for f in facs if f["ftype"] == "given"))
    pv = param_values(facs)
    movable = [v for v in givens
               if v not in pv and re.search(rf"\b{v}\b", text)]
    if not movable: return []
    out = []; seen = {text}
    for _ in range(want * 6):
        if len(out) >= want: break
        mapping = {}
        for v in movable:
            if rng.random() < 0.25 and len(movable) > 1:
                mapping[v] = v          # sometimes hold a value fixed
            else:
                lo = max(1, v - max(3, v))
                hi = min(280, v + max(6, v))
                mapping[v] = rng.randint(lo, hi)
        if all(mapping[v] == v for v in movable): continue
        newvals = list(mapping.values())
        if len(set(newvals)) != len(newvals): continue
        if any(nv in pv for nv in newvals): continue
        t2 = re.sub(r"\b(\d+)\b",
                    lambda m: str(mapping.get(int(m.group(1)), m.group(1))),
                    text)
        if t2 in seen: continue
        f2 = json.loads(json.dumps(facs))
        for f in f2:
            if f["ftype"] == "given" and f["value"] in mapping:
                f["value"] = mapping[f["value"]]
        r2 = {"src_idx": None, "construction": f'perturb-of-{row["src_idx"]}',
              "factors": f2, "query": row["query"], "answer": None,
              "lane": "L0-mint-perturb", "original": t2}
        if unanchored_givens(r2): continue
        try:
            prim, nv = expand_graph(f2, 24)
            if sum(1 for f in prim if f["ftype"] == "fdiv") > 1: continue
            gv = {f["var"]: f["value"] for f in prim if f["ftype"] == "given"}
            res = solve_symbolic(problem_from_algebra3(max(nv, 24), prim, gv, 300),
                                 budget=300_000, seed=0)
            if res["status"] != "solved": continue
            ans = int(res["assignment"][row["query"]])
            if not (0 <= ans <= 300): continue
        except Exception:
            continue
        r2["answer"] = ans; seen.add(t2); out.append(r2)
    return out

def main():
    rng = random.Random(6262)
    seeds = load_seeds()
    out = []; barren = 0
    for row in seeds:
        vs = variants(row, rng)
        if not vs: barren += 1
        out.extend(vs)
    for i, r in enumerate(out): r["src_idx"] = 200000 + i
    with open('.cache/mint_perturb_v2.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[perturb] {len(seeds)} seeds -> {len(out)} variants "
          f"({barren} barren seeds) -> .cache/mint_perturb_v2.jsonl")

if __name__ == "__main__":
    main()
