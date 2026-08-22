"""form_assemble22.py — THE JOINT SCALE FIRE's diet (2026-08-22, word
given): 25k dialect base + symbolic-mint-at-scale + perturbation mint +
human anchors x20, all key-lawful. Under ALG_TRUNK_LORA the trainer
computes states IN-GRAPH, so the states memmap is written SPARSE (only
existence + length are checked) — assembly cost collapses to
tokenize+gold. The fixed wild-val 20 stay excluded.
"""
import sys, os, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, build_gold
import phase1_algebra_head as PH
from mycelium.era import mix_sha16
from mycelium.anchor_law import unanchored_givens
import glob as _g

files = sorted(_g.glob('.cache/book*_t*_batch*.jsonl'))
assert not any('t7self' in f for f in files)
assert os.path.exists('.cache/base_t7self_deeds.jsonl')

def load_rows(fs):
    return [json.loads(l) for f in fs for l in open(f) if l.strip()]

def to_train(rows, gen):
    out = []
    for r in rows:
        out.append({"text": r["original"], "factors": r["factors"],
                    "query_var": r["query"], "n_vars": 24, "m": 300,
                    "solution": None, "decisions": 0, "mentions": {},
                    "gen": gen})
    from mycelium.csp_domains import problem_from_algebra3
    from mycelium.csp_core import solve_symbolic
    from mycelium.macros import expand_graph
    for t, src in zip(out, rows):
        prim, nv = expand_graph(t["factors"], 24)
        gv = {f["var"]: f["value"] for f in prim if f["ftype"] == "given"}
        res = solve_symbolic(problem_from_algebra3(max(nv, 24), prim, gv, 300),
                             budget=500_000, seed=0)
        assert res["status"] == "solved"
        t["solution"] = [int(res["assignment"][v]) for v in range(24)]
        assert t["solution"][t["query_var"]] == src["answer"]
    return out

# human anchors (anchor-law corpus minus skips minus fixed wild-val)
_byid = {r["src_idx"]: r for r in load_rows(files)}
for l in open('.cache/book12_anchor_batch1.jsonl'):
    r = json.loads(l); _byid[r["src_idx"]] = r
_skips = set(json.load(open('.cache/book12_anchor_skips.json')))
_wv = set(json.loads(l)["src_idx"] for l in open('.cache/g55_wildval.jsonl'))
_byid = {k: v for k, v in _byid.items() if k not in _skips and k not in _wv}
_bad = [k for k, v in _byid.items() if unanchored_givens(v)]
assert not _bad, f"ANCHOR LAW: {sorted(_bad)[:8]}"
human = sorted(_byid.values(), key=lambda r: r["src_idx"])
perturb = load_rows(['.cache/mint_perturb_v2.jsonl'])
symb = load_rows(['.cache/mint_symbolic_v1.jsonl'])
assert len(symb) > 20000, f"POSITIVE PRESENCE: scale mint short ({len(symb)})"
HUMAN_REPS = 20
print(f"[assemble22] JOINT SCALE: human {len(human)}x{HUMAN_REPS} + perturb "
      f"{len(perturb)} + symbolic {len(symb)} + dialect base 25000", flush=True)
wild = to_train(human, "b22human") * HUMAN_REPS + \
       to_train(perturb, "b22perturb") + to_train(symb, "b22symb")
rows8 = [json.loads(l) for l in open('.cache/form_mix8.jsonl')]
rng = np.random.RandomState(41)
keep = np.sort(rng.choice(96100, 25000, replace=False))
mix = [rows8[i] for i in keep] + wild
with open('.cache/form_mix22.jsonl', 'w') as f:
    for r in mix: f.write(json.dumps(r) + "\n")
n = len(mix)
print(f"[assemble22] mix {n} rows (dialect 25000 / wild {len(wild)})", flush=True)
assert int(os.environ.get("ALG_TRUNK_LORA", "1")), \
    "assemble22 is TRUNK_LORA-only (sparse states)"
out = np.lib.format.open_memmap('.cache/phase1_alg_states_form22_states.npy',
                                mode='w+', dtype=np.float16,
                                shape=(n, T_ALG, 2048))
del out   # SPARSE: never written — the in-graph trunk computes states
samples, ids2, mask, offsets = PH.tokenize('.cache/form_mix22.jsonl')
gold = build_gold(samples, offsets)
sent = np.stack([sent_indices(s["text"], o, mask[i])
                 for i, (s, o) in enumerate(zip(samples, offsets))])
np.savez('.cache/phase1_alg_states_form22.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16('.cache/form_mix22.jsonl'),
         **{f"g_{k}": v for k, v in gold.items()})
print(f"[assemble22] STITCHED sparse-states + gold + sha ({n} rows)", flush=True)
