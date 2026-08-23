"""form_assemble23.py — THE JOINT SCALE FIRE's diet (2026-08-22, word
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
comp = load_rows(['.cache/mint_comp_v3.jsonl'])
assert len(comp) > 20000, f"POSITIVE PRESENCE: comp mint short ({len(comp)})"
_shapes = len(set(r.get("shape") for r in comp))
assert _shapes > 800, f"EFFECTIVE-N FENCE: only {_shapes} shapes (nominal-n is not n)"
symb_all = load_rows(['.cache/mint_symbolic_v1.jsonl'])
_r = np.random.RandomState(77)
symb = [symb_all[i] for i in sorted(_r.choice(len(symb_all),
        min(10000, len(symb_all)), replace=False))]
HUMAN_REPS = 20
print(f"[assemble23] JOINT SCALE: human {len(human)}x{HUMAN_REPS} + perturb "
      f"{len(perturb)} + COMP {len(comp)} ({_shapes} shapes) + template "
      f"{len(symb)} + dialect base 25000", flush=True)
SKEW = os.environ.get("DIET_SKEW", "bal")
skel = load_rows(['.cache/mint_skel_v1.jsonl']) if SKEW == "skel" else []
if os.environ.get("IN_CANON", "0") == "1":
    from mycelium.clause_grains import canonicalize
    for _grp in (human, perturb, comp, symb, skel):
        for _r in _grp: _r["original"] = canonicalize(_r["original"])
    print("[assemble23] IN=canon: wild surfaces canonicalized", flush=True)
_comp_sys = [r for r in comp if str(r.get("shape", "")).startswith("sys")]
_comp_ev = [r for r in comp if not str(r.get("shape", "")).startswith("sys")]
wild = to_train(human, "b22human") * HUMAN_REPS + \
       to_train(perturb, "b22perturb") * (5 if SKEW == "pert" else 1) + \
       to_train(_comp_sys, "b22comp") * (3 if SKEW == "sys" else 1) + \
       to_train(_comp_ev, "b22comp") * (3 if SKEW == "chain" else 1) + \
       to_train(symb, "b22symb") + \
       (to_train(skel, "b22skel") * 3 if SKEW == "skel" else [])
print(f"[assemble23] DIET_SKEW={SKEW} wild={len(wild)}", flush=True)
rows8 = [json.loads(l) for l in open('.cache/form_mix8.jsonl')]
rng = np.random.RandomState(41)
keep = np.sort(rng.choice(96100, 25000, replace=False))
mix = [rows8[i] for i in keep] + wild
with open(os.environ.get('MIX_OUT', '.cache/form_mix23.jsonl'), 'w') as f:
    for r in mix: f.write(json.dumps(r) + "\n")
n = len(mix)
print(f"[assemble23] mix {n} rows (dialect 25000 / wild {len(wild)})", flush=True)
assert int(os.environ.get("ALG_TRUNK_LORA", "1")), \
    "assemble23 is TRUNK_LORA-only (sparse states)"
out = np.lib.format.open_memmap(os.environ.get('MIX_OUT', '.cache/form_mix23.jsonl').replace('form_mix','phase1_alg_states_form').replace('.jsonl','_states.npy'),
                                mode='w+', dtype=np.float16,
                                shape=(n, T_ALG, 2048))
del out   # SPARSE: never written — the in-graph trunk computes states
samples, ids2, mask, offsets = PH.tokenize(os.environ.get('MIX_OUT', '.cache/form_mix23.jsonl'))
gold = build_gold(samples, offsets)
sent = np.stack([sent_indices(s["text"], o, mask[i])
                 for i, (s, o) in enumerate(zip(samples, offsets))])
np.savez(os.environ.get('MIX_OUT', '.cache/form_mix23.jsonl').replace('form_mix','phase1_alg_states_form').replace('.jsonl','.npz'), tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16(os.environ.get('MIX_OUT', '.cache/form_mix23.jsonl')),
         **{f"g_{k}": v for k, v in gold.items()})
print(f"[assemble23] STITCHED sparse-states + gold + sha ({n} rows)", flush=True)
