"""coverage_sweep.py — THE MINT-VS-HARVEST COVERAGE SWEEP (2026-07-31,
registered before firing; #105's lead). The read that found the
fdiv-on-derived hole by accident, run systematically across every
construction: mint distribution (gen22_mix — what the gate has seen) vs
book demand (the certified rows — the harvest expressed in-register).

Axes: ftype share; OPERAND PROVENANCE per ftype (given vs derived
operand — the hole's own axis); ordered pairs (A feeds B — doubles as
the composition matrix's training-side co-occurrence column); chain
depth to query; arg multiplicity. Pins in the ledger: fdiv-on-derived
must read COVERED; hole = demand >= 2% AND mix < demand/10."""
import json, glob, sys
from collections import Counter, defaultdict

def feats(row):
    """Feature set for one factor graph."""
    facs = row["factors"]; q = row["query_var"]
    # deep clean 2026-08-01: pct produces args[0] (no "result" key) — the
    # old code made pct->X pairs structurally unmeasurable
    derived = {f.get("result") for f in facs if "result" in f}
    derived |= {f["args"][0] for f in facs if f.get("ftype") == "pct"}
    derived.discard(None)
    out = Counter()
    kinds = []
    producer = {}
    for f in facs:
        ft = f["ftype"]
        k = ft if ft not in ("rel",) else f"rel-{f.get('op','?')}"
        kinds.append(k)
        out[f"ftype:{k}"] += 1
        ops = (list(f.get("args", [])) + ([f["var"]] if "var" in f and ft != "given" else []))
        on_derived = any(v in derived for v in ops)
        if ft != "given":
            out[f"{k}:on-{'derived' if on_derived else 'given'}"] += 1
        if "result" in f:
            producer[f["result"]] = k
        elif f.get("ftype") == "pct":
            producer[f["args"][0]] = k
        if "args" in f and len(f["args"]) == 2 and f["args"][0] == f["args"][1]:
            out["dup-args"] += 1
    # ordered pairs: producer kind -> consumer kind
    for f in facs:
        ft = f["ftype"]
        k = ft if ft != "rel" else f"rel-{f.get('op','?')}"
        if ft == "given": continue
        ops = (list(f.get("args", [])) + ([f["var"]] if "var" in f else []))
        for v in ops:
            if v in producer:
                out[f"pair:{producer[v]}->{k}"] += 1
    # chain depth to query (longest producer chain ending at q)
    def depth(v, seen=()):
        if v not in producer or v in seen: return 0
        f = next(f for f in facs if f.get("result") == v)
        ops = list(f.get("args", [])) + ([f["var"]] if "var" in f else [])
        return 1 + max([depth(o, seen + (v,)) for o in ops] or [0])
    out[f"depth:{min(depth(q), 4)}"] += 1
    return out

def population(rows, name):
    tot = Counter(); n = 0
    for r in rows:
        try:
            tot += feats(r)
        except Exception:
            continue
        n += 1
    return {"name": name, "n": n, "counts": tot}

mix = [json.loads(l) for l in open(".cache/gen22_mix.jsonl")]
books = []
for draft in sorted(glob.glob(".cache/book8_*prose_pairs_draft.jsonl")):
    certf = draft.replace("prose_pairs_draft.jsonl", "certification.json")
    import os
    if not os.path.exists(certf): certf = ".cache/book8_certification.json"
    rows = [json.loads(l) for l in open(draft)]
    cert = json.load(open(certf))["certified"]
    books += [rows[e["i"]] for e in cert]
print(f"[sweep] mix n={len(mix)}  certified book rows n={len(books)}")
P_mix = population(mix, "mix")
P_book = population(books, "books")

def share(P, key):
    return P["counts"].get(key, 0) / max(P["n"], 1)

keys = sorted(set(P_mix["counts"]) | set(P_book["counts"]))
holes, near, covered = [], [], []
print(f"\n{'feature':34s} {'book-demand':>11s} {'mix-share':>10s}  verdict")
for k in keys:
    d, m = share(P_book, k), share(P_mix, k)
    if d < 0.02:
        continue   # below demand floor; not a hole candidate by pin
    if m < d / 10:
        holes.append((k, d, m)); v = "HOLE"
    elif m < d / 3:
        near.append((k, d, m)); v = "near"
    else:
        covered.append((k, d, m)); v = "ok"
    print(f"{k:34s} {d:10.1%} {m:9.1%}   {v}")
print(f"\n[pin check] fdiv:on-derived — book {share(P_book,'fdiv:on-derived'):.1%} "
      f"mix {share(P_mix,'fdiv:on-derived'):.1%} -> "
      f"{'COVERED (pin holds)' if share(P_mix,'fdiv:on-derived') >= share(P_book,'fdiv:on-derived')/10 else 'NOT COVERED — SWEEP SUSPECT (the pin)'}")
print(f"\n=== VERDICT: {len(holes)} hole(s), {len(near)} near-hole(s), {len(covered)} covered (of {len(holes)+len(near)+len(covered)} demand-bearing features) ===")
for k, d, m in holes: print(f"  HOLE: {k}  demand {d:.1%} vs mix {m:.2%}")
for k, d, m in near: print(f"  near: {k}  demand {d:.1%} vs mix {m:.2%}")

# the composition matrix's training-side column: pair co-occurrence
pairs_mix = {k for k in P_mix["counts"] if k.startswith("pair:") and P_mix["counts"][k] >= 20}
pairs_book = {k for k in P_book["counts"] if k.startswith("pair:")}
print(f"\n[matrix column] ordered pairs in TRAINING (n>=20): {len(pairs_mix)}")
print(f"[matrix column] pairs demanded by books but ABSENT/rare in training:")
for k in sorted(pairs_book - pairs_mix):
    print(f"  NOVEL-to-training: {k}  (book n={P_book['counts'][k]}, mix n={P_mix['counts'].get(k,0)})")
json.dump({"mix_n": P_mix["n"], "book_n": P_book["n"],
           "mix_counts": dict(P_mix["counts"]), "book_counts": dict(P_book["counts"]),
           "holes": holes, "near": near},
          open(".cache/coverage_sweep.json", "w"), indent=1)
print("[saved] .cache/coverage_sweep.json")
