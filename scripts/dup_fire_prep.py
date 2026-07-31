"""dup_fire_prep.py — THE DUP FIRE'S PREPARATION (2026-07-31, the word
given; docs/DUP_DIET_DESIGN.md is the design). Stage 1 of the fire:
(a) mint the dup pool (1,200 uniques, full mix-row schema, knot-dedup,
solver-verified under the door); (b) build the book-8 WET block with
the sub→rearranged-add fold; (c) precompute states for BOTH small
blocks (the only fresh trunk work); (d) build six mixes (nested doses
2/5/12% × WET/DRY) and ASSEMBLE their states from banked verified
blocks + fresh blocks; (e) build gold npz per mix via build_gold (the
real function, CPU); (f) sentinel-verify each arm (live forward vs
assembled memmap). Nothing trains here — the orchestrator fires arms
only after every sentinel passes."""
import sys, os, json, glob
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from mycelium.doors import certify_unique
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from tta_alg2_dials import solve2
from hash_audit_iso import canon

L = "abcdefghij"
BASE_N = 82400

# ---------- (a) the dup pool ----------
def mint_pool(n_target, seed):
    rng = np.random.RandomState(seed)
    rows, seen, tries = [], set(), 0
    while len(rows) < n_target and tries < n_target * 12:
        tries += 1
        op = "add" if rng.rand() < 0.5 else "mul"
        x = int(rng.randint(2, 90)) if op == "add" else int(rng.randint(2, 13))
        n_dist = int(rng.randint(2, 5))
        gv = [int(rng.randint(2, 90)) for _ in range(n_dist)]
        dv = n_dist; res = n_dist + 1
        gold = x + x if op == "add" else x * x
        if gold > 300: continue
        facs = [{"ftype": "given", "var": i, "value": gv[i]} for i in range(n_dist)]
        facs.append({"ftype": "given", "var": dv, "value": x})
        facs.append({"ftype": "rel", "op": op, "args": [dv, dv], "result": res})
        word = "plus" if op == "add" else "times"
        order = list(range(n_dist)); rng.shuffle(order)
        sents = [f"{L[i]} is {gv[i]}." for i in order] + \
                [f"{L[dv]} is {x}.", f"{L[dv]} {word} {L[dv]} equals {L[res]}."]
        letters = ", ".join(L[:res+1])
        text = f"Consider the numbers {letters}. " + " ".join(sents) + f" What is {L[res]}?"
        row = {"text": text, "factors": facs, "query_var": res, "n_vars": 24, "m": 300}
        dig = canon(row)[0]
        if dig in seen: continue
        if solve2(facs, res, {"n_vars": 24, "m": 300}) != gold: continue
        gv_map = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
        prob = problem_from_algebra3(24, facs, gv_map, 300)
        r = solve_symbolic(prob, budget=200_000, seed=0)
        assert r["status"] == "solved"
        sol = [int(r["assignment"][v]) for v in range(24)]
        row["decisions"] = int(r.get("decisions", 0))
        row["mentions"] = {}
        row["solution"] = sol
        row["gen"] = {"diet": "dup-args", "minted": "2026-07-31"}
        seen.add(dig)
        rows.append(row)
    return rows

if not os.path.exists(".cache/dup_pool.jsonl"):
    pool = mint_pool(1200, 51000)
    with open(".cache/dup_pool.jsonl", "w") as f:
        for r in pool: f.write(json.dumps(r) + "\n")
    print(f"[pool] minted {len(pool)} unique dup rows (knot-deduped, door-verified)")
pool = [json.loads(l) for l in open(".cache/dup_pool.jsonl")]
assert len(pool) >= 1160, len(pool)

# ---------- (b) the WET block: book-8 certified rows, sub folded ----------
def fold_sub(f):
    """c = a - b  ==  a = c + b (the canonical rearranged-add; sub is
    unrepresentable in the op head — training raw sub as add would
    mis-supervise; the fold trains what the gate emits)."""
    if f.get("ftype") == "rel" and f.get("op") == "sub":
        a, b = f["args"]; c = f["result"]
        return {"ftype": "rel", "op": "add", "args": sorted([b, c]), "result": a}
    return f

if not os.path.exists(".cache/book8_wet_block.jsonl"):
    wet = []
    for draft in sorted(glob.glob(".cache/book8_*prose_pairs_draft.jsonl")):
        certf = draft.replace("prose_pairs_draft.jsonl", "certification.json")
        if not os.path.exists(certf): certf = ".cache/book8_certification.json"
        rows = [json.loads(l) for l in open(draft)]
        for e in json.load(open(certf))["certified"]:
            r = rows[e["i"]]
            facs = [fold_sub(dict(f)) for f in r["factors"]]
            wet.append({"text": r["gen"]["dialect"], "factors": facs,
                        "query_var": r["query_var"], "n_vars": r["n_vars"],
                        "m": r["m"], "decisions": 1, "mentions": {},
                        "solution": r["solution"]})
    with open(".cache/book8_wet_block.jsonl", "w") as f:
        for r in wet: f.write(json.dumps(r) + "\n")
    print(f"[wet] book-8 block: {len(wet)} rows (sub folded to rearranged-add)")
wet = [json.loads(l) for l in open(".cache/book8_wet_block.jsonl")]

# ---------- (c) fresh states for the two small blocks ----------
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_JSON)

def states_of(rows, tag):
    npy = f".cache/dupfire_{tag}_states.npy"
    if os.path.exists(npy):
        return np.load(npy, mmap_mode="r")
    n = len(rows)
    out = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16,
                                    shape=(n, T_ALG, 2048))
    for s0 in range(0, n, 8):
        chunk = rows[s0:s0+8]
        ids = np.zeros((8, T_ALG), np.int32)
        for i, r in enumerate(chunk):
            e = tok.encode(r["text"]); Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]
        st = recompute_states(ids).astype(np.float16)
        out[s0:s0+len(chunk)] = st[:len(chunk)]
        if (s0 // 8) % 20 == 0: print(f"  [{tag}] {s0}/{n}", flush=True)
    out.flush()
    return np.load(npy, mmap_mode="r")

pool_states = states_of(pool, "pool")
wet_states = states_of(wet, "wet")
print(f"[states] pool {pool_states.shape}  wet {wet_states.shape}")

# ---------- (d)+(e) six mixes with assembled states + real gold ----------
from phase1_algebra_head import build_gold
import phase1_algebra_head as PH

def uniques_for(share, base_n):
    rows_needed = share / (1.0 - share) * base_n
    return int(round(rows_needed / 10))

ARMS = []
for wetness in ("dry", "wet"):
    bn = BASE_N + (len(wet) * 10 if wetness == "wet" else 0)
    for share, nm in ((0.02, "d02"), (0.05, "d05"), (0.12, "d12")):
        ARMS.append({"name": f"{wetness}_{nm}", "wet": wetness == "wet",
                     "uniq": uniques_for(share, bn), "share": share})
print("[arms]", [(a["name"], a["uniq"]) for a in ARMS])

base_states = np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
assert base_states.shape[0] == BASE_N
base_lines = open(".cache/gen22_mix.jsonl").read().splitlines()
assert len(base_lines) == BASE_N

for arm in ARMS:
    mixp = f".cache/dupfire_{arm['name']}_mix.jsonl"
    npyp = f".cache/phase1_alg_states_g23{arm['name']}_states.npy"
    npzp = f".cache/phase1_alg_states_g23{arm['name']}.npz"
    if os.path.exists(npzp):
        print(f"[{arm['name']}] already assembled"); continue
    block_rows, block_srcs = [], []            # (source, index) per appended row
    if arm["wet"]:
        for i, r in enumerate(wet):
            for _ in range(10):
                block_rows.append(r); block_srcs.append(("wet", i))
    for i in range(arm["uniq"]):               # nested: pool[:U]
        for _ in range(10):
            block_rows.append(pool[i]); block_srcs.append(("pool", i))
    n_tot = BASE_N + len(block_rows)
    with open(mixp, "w") as f:
        f.write("\n".join(base_lines) + "\n")
        for r in block_rows: f.write(json.dumps(r) + "\n")
    # states assembly: banked base + indexed fresh blocks
    out = np.lib.format.open_memmap(npyp, mode="w+", dtype=np.float16,
                                    shape=(n_tot, T_ALG, 2048))
    CH = 4096
    for s0 in range(0, BASE_N, CH):
        out[s0:min(s0+CH, BASE_N)] = base_states[s0:min(s0+CH, BASE_N)]
    for j, (src, i) in enumerate(block_srcs):
        out[BASE_N + j] = (wet_states if src == "wet" else pool_states)[i]
    out.flush(); del out
    # gold npz via the REAL pipeline: tokenize + build_gold + sent
    samples, ids, mask, offsets = PH.tokenize(mixp)
    gold = build_gold(samples, offsets)
    sent = np.stack([sent_indices(s["text"], o, mask[i])
                     for i, (s, o) in enumerate(zip(samples, offsets))])
    np.savez(npzp, tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
             **{f"g_{k}": v for k, v in gold.items()})
    print(f"[{arm['name']}] mix {n_tot} rows; states assembled; gold built", flush=True)

# ---------- (f) sentinel verification per arm ----------
for arm in ARMS:
    mixp = f".cache/dupfire_{arm['name']}_mix.jsonl"
    npyp = f".cache/phase1_alg_states_g23{arm['name']}_states.npy"
    rows = [json.loads(l) for l in open(mixp)]
    st = np.load(npyp, mmap_mode="r")
    assert st.shape[0] == len(rows)
    picks = [0, BASE_N - 1, BASE_N, len(rows) - 1,
             BASE_N + (len(rows) - BASE_N) // 2, 40000]
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    for i, ridx in enumerate(picks):
        e = tok.encode(rows[ridx]["text"]); Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
    live = recompute_states(ids).astype(np.float32)
    for i, ridx in enumerate(picks):
        m_ = msk[i] > 0
        a = live[i][m_]; b = np.asarray(st[ridx], np.float32)[m_]
        cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos > 0.9999, f"SENTINEL FAIL {arm['name']} row {ridx} cos {cos}"
    print(f"[sentinel] {arm['name']}: 6/6 rows verified (assembly TRUSTED)")
print("[prep] ALL SIX ARMS ASSEMBLED AND VERIFIED — the orchestrator may fire")
