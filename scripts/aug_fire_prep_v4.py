"""aug_fire_prep.py — THE AUGMENTATION FIRE'S PREP (2026-08-01, the
word given; docs/AUG_FIRE_DESIGN.md is the design). Re-render 10% of
gen22_mix through the licensed table — graph fixed, gold preserved,
text replaced IN PLACE (same row order → base states patch at changed
indices only). V-LOW = first half of entries per construction;
V-FULL = all. Rows with sub/macro factors are skipped (not in the
table's constructions). Staged: assemble → verify → one atomic write
per arm."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np, re
from collections import Counter
L = "abcdefghijklmnopqrstuvwx"
T = json.load(open('.cache/aug_table_v4.json'))["licensed"]
BY = {}
for e in T:
    BY.setdefault(e["construction"], []).append(e["fmt"])

def pools(half):
    return {c: (fmts[:max(1,len(fmts)//2)] if half else fmts) for c, fmts in BY.items()}

def render_row(r, pool, rng):
    facs = r["factors"]
    if any(f.get("ftype") in ("macro",) or f.get("op") == "sub" for f in facs):
        return None
    sents = []
    mx = 0
    for f in facs:
        vs = [f.get("var",0)] + list(f.get("args",[])) + [f.get("result",0)]
        mx = max([mx]+vs)
    for f in facs:
        ft = f["ftype"]
        try:
            if ft == "given":
                fmt = pool["given"][rng.randint(len(pool["given"]))]
                sents.append(fmt.format(x=L[f["var"]], v=f["value"]))
            elif ft == "rel":
                a,b = f["args"]; c = f["result"]
                if a == b:
                    key = "dup"
                    cands = [m for m in pool.get("dup",[]) if any(w in m for w in ("times","*","square","multiplied","x ","roduct")) == (f["op"]=="mul")]
                    if not cands: return None
                    fmt = cands[rng.randint(len(cands))]
                    sents.append(fmt.format(a=L[a], c=L[c]))
                else:
                    key = "mul" if f["op"]=="mul" else "add"
                    fmt = pool[key][rng.randint(len(pool[key]))]
                    sents.append(fmt.format(a=L[a], b=L[b], c=L[c]))
            elif ft in ("fdiv","mod"):
                cands = pool.get(ft,[])
                cands = [m for m in cands if not any(w in m for w in ("Half","third","quarter","fifth"))] or cands
                fmt = cands[rng.randint(len(cands))]
                sents.append(fmt.format(a=L[f["var"]], k=f["k"], b=L[f["result"]]))
            elif ft == "pct":
                fmt = pool["pct"][rng.randint(len(pool["pct"]))]
                sents.append(fmt.format(p=f["p"], p2=L[f["args"][0]], b2=L[f["args"][1]]))
            elif ft == "sel":
                cands = [m for m in pool["sel"] if any(w in m for w in ("smaller","inimum")) == (f.get("sel")=="smaller")]
                if not cands: return None
                fmt = cands[rng.randint(len(cands))]
                sents.append(fmt.format(a=L[f["args"][0]], b=L[f["args"][1]], c=L[f["result"]]))
            else:
                return None
        except (IndexError, KeyError):
            return None
    letters = ", ".join(L[:mx+1])
    q = r["query_var"]
    return f"Consider the numbers {letters}. " + " ".join(sents) + f" What is {L[q]}?"

base = [json.loads(l) for l in open('.cache/gen22_mix.jsonl')]
rng0 = np.random.RandomState(84000)
cand = [i for i,r in enumerate(base)
        if not any(f.get("ftype")=="macro" or f.get("op")=="sub" for f in r["factors"])]
picks = sorted(rng0.choice(cand, size=8240, replace=False).tolist())
import json as _j
picks = sorted(set(picks) & set(_j.load(open('.cache/augfire_vfull_changed.json'))))  # EQUALIZATION: exactly vfull's rows
print(f"[aug] base {len(base)}  renderable {len(cand)}  picked {len(picks)}")
for arm, half in (("vf14", False),):
    pool = pools(half)
    rng = np.random.RandomState(89000)
    n_ok = 0
    rows = [dict(r) for r in base]
    changed = []
    for i in picks:
        t = render_row(rows[i], pool, rng)
        if t is None: continue
        vals = [f["value"] for f in rows[i]["factors"] if f.get("ftype")=="given"]
        tc = Counter(int(m) for m in re.findall(r"\d+", t))
        if any(tc[v] < 1 for v in vals): continue
        rows[i]["text"] = t; changed.append(i); n_ok += 1
    with open(f'.cache/augfire_{arm}_mix.jsonl','w') as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    json.dump(changed, open(f'.cache/augfire_{arm}_changed.json','w'))
    print(f"[{arm}] re-rendered {n_ok} rows (pool sizes: " +
          ", ".join(f"{c}:{len(v)}" for c,v in sorted(pool.items())) + ")")
print("[prep] both mixes staged — assembly patches changed indices only")
