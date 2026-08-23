"""retrieval_bridge.py — THE RETRIEVAL BRIDGE, A0 (2026-08-23, word
given; the fuzzy-lookup gut made explicit). The reader IS a retrieval
system — so retrieve explicitly: pooled pure-trunk states index the
trained human corpus; a wild query kNNs into it; the neighbor's GRAPH
SHAPE is rebound with the query's surface numbers (the perturbation
mint's substitution INVERTED: full surface-number sequence alignment,
order-greedy, count-gated); solve_forced (uniqueness certificate)
disposes. DETERMINISTIC single candidate per row — neighbors tried in
similarity order, the FIRST that solves is graded, no key-peeking.
Fixtures: wild-val 20 + held-out 20. Baselines: G64 head reads
(2/20 + 1/10). Report: retrieval rights, head-union, neighbor cosines.
Zero training.
"""
import os, sys, json, glob, re
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, load_alg
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states
from mycelium.anchor_law import atomic_numbers
from mycelium.macros import expand_graph
from tokenizers import Tokenizer

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")

def pooled(texts):
    n = len(texts); out = np.zeros((n, 2048), np.float32)
    for s0 in range(0, n, 8):
        sl = texts[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        for li, t in enumerate(sl):
            e = tok.encode(t)
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        for li in range(len(sl)):
            m = msk[li][:, None]
            out[s0 + li] = (sts[li] * m).sum(0) / max(m.sum(), 1)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

def num_seq(text):
    return [int(m.group(1)) for m in NUM.finditer(text)]

def graph_value_slots(facs):
    """(kind, index, field) for every numeric surface-slot in graph order."""
    slots = []
    for i, f in enumerate(facs):
        if f["ftype"] == "given":
            slots.append(("given", i, "value"))
        elif f["ftype"] == "macro":
            for k in ("k1", "k2", "a", "k"):
                if isinstance(f.get(k), int): slots.append(("param", i, k))
        elif f["ftype"] == "pct":
            slots.append(("param", i, "p"))
        elif f["ftype"] == "fdiv":
            slots.append(("param", i, "k"))
    return slots

def rebind(neigh, qtext):
    """Transplant neigh's graph shape with qtext's numbers via order-greedy
    full-sequence alignment. Returns rebound factors or None."""
    seq_n = num_seq(neigh["original"]); seq_q = num_seq(qtext)
    if len(seq_n) != len(seq_q) or not seq_n: return None
    facs = json.loads(json.dumps(neigh["factors"]))
    used = [False] * len(seq_n)
    for kind, i, field in graph_value_slots(facs):
        v = facs[i][field]
        pos = next((j for j, (sv, u) in enumerate(zip(seq_n, used))
                    if sv == v and not u), None)
        if pos is None:
            pos = next((j for j, sv in enumerate(seq_n) if sv == v), None)
            if pos is None: return None       # value not surface (shouldn't
        else:                                  # happen under the anchor law)
            used[pos] = True
        nv = seq_q[pos]
        if kind == "param" and nv < (2 if field == "k" else 1): return None
        if kind == "given" and not (0 <= nv <= 300): return None
        facs[i][field] = nv
    return facs

def main():
    # ---- the index: trained human diet rows (fixed wild-val excluded) ----
    _all = [json.loads(l) for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl'))
            for l in open(f) if l.strip()]
    byid = {r["src_idx"]: r for r in _all}
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    wvs = set(json.loads(l)["src_idx"] for l in open('.cache/g55_wildval.jsonl'))
    index_rows = [v for k, v in sorted(byid.items())
                  if k not in sk and k not in wvs]
    print(f"[rb] index: {len(index_rows)} trained rows", flush=True)
    # ---- the fixtures ----
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    fixtures = [{"original": r["original"], "answer": r["answer"]} for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(r["src_idx"] for r in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        fixtures += [{"original": h[i]["problem"],
                      "answer": int(str(h[i]["answer"]).strip())}
                     for i in rg.permutation(len(h)) if i not in drafted
                     and str(h[i]["answer"]).strip().isdigit()][:10]
    # ---- embed ----
    E_idx = pooled([r["original"] for r in index_rows])
    E_q = pooled([r["original"] for r in fixtures])
    sims = E_q @ E_idx.T
    right = 0; attempted = 0; used_cos = []
    per_row = []
    for qi, q in enumerate(fixtures):
        order = np.argsort(-sims[qi])[:8]
        got = None; cos = None
        for ni in order:
            facs = rebind(index_rows[int(ni)], q["original"])
            if facs is None: continue
            try:
                a = solve_forced(facs, index_rows[int(ni)]["query"],
                                 {"n_vars": 24, "m": 300})
            except Exception:
                continue
            if a is not None:
                got = a; cos = float(sims[qi][int(ni)]); break
        if got is not None:
            attempted += 1; used_cos.append(cos)
            if got == q["answer"]: right += 1
        per_row.append((got, q["answer"], cos))
    print(f"[rb A0] fixtures 40: SOLVED {attempted} RIGHT {right} "
          f"(mean used-cos {np.mean(used_cos):.3f})" if used_cos else
          f"[rb A0] fixtures 40: SOLVED {attempted} RIGHT {right}", flush=True)
    for i, (got, key, cos) in enumerate(per_row[:20]):
        if got is not None:
            print(f"  wv[{i}] got={got} key={key} "
                  f"{'RIGHT' if got == key else 'wrong'} cos={cos:.3f}", flush=True)

if __name__ == "__main__":
    main()
