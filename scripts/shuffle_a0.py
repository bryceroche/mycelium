"""shuffle_a0.py — THE SHUFFLE ORGAN, RUNG A0 (SHUFFLE_SPEC 2026-08-15;
fired 2026-08-22 under the word, wild-register form). Two passes, one
discrete matching between them: pass-1 fat attention proposes slot->
segment affinities; a greedy 1:1 matching (the jig — wiring, not knobs)
assigns each present factor slot one segment; pass-2 re-reads with the
factor bank masked per slot to its segment via the pmask socket.
Segments = math-run + punctuation cuts (inference-side symbolic
structure; the dead snt-grains reused as the jig's TEETH, not as
embeddings). Zero training. Env: A0_CKPT (LoRA ckpt).
Fixtures: wild-val 20 + held-out 20 (conversion read) + 60-row dialect
spot-hold (bar >=0.98). Honest floor: rights delta >=3 = the organ
touches the wall; else report, no claim.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest", "ALG_TRUNK_LORA": "1"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 L_FAC)
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states, _trunk_host
from mycelium.llama_loader import _rms_norm
from mycelium.clause_grains import math_cut_positions
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = os.environ.get("A0_CKPT", ".cache/g63_bridge.safetensors")
samples_d, _, _, _, _ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0); sd = safe_load(CKPT)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
HAS_LORA = any(k.startswith("lora") for k in sd.keys())
host = _trunk_host()
LD = [{f"{nm}_{ab}": (p[f"lora{li}_{nm}_{ab}"] * (8.0 if ab == "B" else 1.0))
       for nm in ("wq", "wo", "wdown") for ab in ("A", "B")}
      for li in range(4)] if HAS_LORA else None
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")

def states_of(ids):
    x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
    for li, layer in enumerate(host.llama_layers):
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin,
                  lora=(LD[li] if LD else None))
    x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
    return x.cast(dtypes.float).realize().numpy()

def segments_of(text, offs, ntk):
    cuts = math_cut_positions(text)
    if not cuts:
        return np.zeros(ntk, np.int32), 1
    arr = np.asarray([o[0] for o in offs[:ntk]], dtype=np.int64)
    seg = np.searchsorted(np.asarray(cuts, dtype=np.int64), arr, "right")
    nseg = int(seg.max()) + 1 if len(seg) else 1
    return seg.astype(np.int32), nseg

def read(rows, textf, ansf, tag):
    m = len(rows); f1 = r1 = f2 = r2 = 0; jig_used = 0
    for s0 in range(0, m, 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32)
        msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        segs = []; nsegs = []
        for li, r in enumerate(sl):
            t = textf(r); e = tok.encode(t)
            if len(e.ids) > T_ALG:
                segs.append(None); nsegs.append(1); continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(t, list(e.offsets), msk[li])
            sg, ns = segments_of(t, list(e.offsets), len(e.ids))
            segs.append(sg); nsegs.append(ns)
        sts = states_of(ids)
        tt = Tensor(sts.astype(np.float32), dtype=dtypes.float)
        tk = Tensor(msk.astype(np.float32), dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o = forward(p, tt, tk, se)
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o)
        o1 = {k2: o[k2].realize().numpy() for k2 in K + ex + ("fat", "pres")}
        # ---- the jig: greedy 1:1 slot->segment matching on fat mass ----
        pm = np.zeros((8, L_FAC, T_ALG), np.float32)
        for li in range(len(sl)):
            if segs[li] is None or nsegs[li] < 2: continue
            fat = o1["fat"][li]                       # (L_FAC, T)
            pres = o1["pres"][li]
            present = [j for j in range(L_FAC) if pres[j] > 0.5]
            ntk = int(msk[li].sum())
            aff = np.zeros((len(present), nsegs[li]))
            for a, j in enumerate(present):
                for sgi in range(nsegs[li]):
                    aff[a, sgi] = fat[j, :ntk][segs[li] == sgi].sum()
            taken = set(); assign = {}
            order = sorted(((aff[a, sgi], a, sgi) for a in range(len(present))
                            for sgi in range(nsegs[li])), reverse=True)
            for score, a, sgi in order:
                if a in assign or sgi in taken: continue
                assign[a] = sgi; taken.add(sgi)
                if len(assign) == min(len(present), nsegs[li]): break
            if not assign: continue
            jig_used += 1
            for a, j in enumerate(present):
                if a not in assign: continue
                allowed = (segs[li] == assign[a])
                row = np.full(T_ALG, -1e9, np.float32)
                row[:ntk][allowed] = 0.0
                pm[li, j] = row
        o = forward(p, tt, tk, se, pmask=Tensor(pm, dtype=dtypes.float))
        o2 = {k2: o[k2].realize().numpy() for k2 in K + ex}
        for bi, r in enumerate(sl):
            facs, q = decode({k2: o1[k2][bi] for k2 in o1
                              if k2 not in ("fat", "pres")} |
                             {"pres": o1["pres"][bi]})
            a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
            if a is not None: f1 += 1
            if a == ansf(r): r1 += 1
            facs, q = decode({k2: o2[k2][bi] for k2 in o2})
            a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
            if a is not None: f2 += 1
            if a == ansf(r): r2 += 1
    print(f"[A0 {tag}] pass1 forced {f1}/{m} right {r1}  |  "
          f"JIG forced {f2}/{m} right {r2}  (jig engaged {jig_used})",
          flush=True)
    return r1, r2

wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
rows = [{"original": r["original"], "answer": r["answer"]} for r in wv]
_all = [json.loads(l) for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl'))
        for l in open(f) if l.strip()]
_byid = {r["src_idx"]: r for r in _all}
for l in open('.cache/book12_anchor_batch1.jsonl'):
    r = json.loads(l); _byid[r["src_idx"]] = r
_sk = set(json.load(open('.cache/book12_anchor_skips.json')))
_dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
drafted = set(_byid) | _sk | set(r["src_idx"] for r in _dd)
for seed in (99, 299):
    rg = np.random.default_rng(seed)
    rows += [{"original": h[i]["problem"],
              "answer": int(str(h[i]["answer"]).strip())}
             for i in rg.permutation(len(h)) if i not in drafted
             and str(h[i]["answer"]).strip().isdigit()][:10]
read(rows, lambda r: r["original"], lambda r: r["answer"], "wild-40")
dial = [samples_d[i] for i in np.random.default_rng(41)
        .choice(len(samples_d), 60, replace=False)]
read(dial, lambda r: r["text"],
     lambda r: r["solution"][r["query_var"]], "dialect-hold-60")
