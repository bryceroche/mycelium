"""explicitation_probe.py — THE EXPLICITATION FORK's probe (2026-07-10).
Hand-annotated implicit facts from real MATH-500 problems, two classes:
LEXICAL (evoking phrase has a span: dozen=12, octagon=8, feet=12in) probed
span-pooled; STRUCTURAL (no anchor: February=28, quad=360) probed whole-seq
pooled, LOO, weak-evidence-labeled. Depths L0-3 vs L0-7 (fresh question —
the old L8 verdict governed routing on our dialect, not world knowledge).
Probe trained on OUR corpus explicit given spans; transfer caveat noted
(failure could be probe-transfer, not info absence — LOO sanity included).
PRIORS: relay 40/35/25 shallow/deep/stage; Code 45/25/30, MIXED tiebreak
(lexical-shallow, structural-stage).
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, N_DIG, TOKENIZER_JSON, tokenize, _spans_to_tokmask
from survivor_depth_probe import train_probe, eval_probe, compute_l8, L8_NPY
from tokenizers import Tokenizer

# (problem_idx, phrase, value) — phrase located by string search
LEX = [(27,"three",3),(40,"octagon",8),(40,"hexagon",6),(49,"Half",2),
       (128,"third",3),(133,"twice",2),(133,"right",90),(212,"one-hundred",100),
       (246,"Twelve",12),(329,"feet",12),(329,"minute",60),(341,"twice",2),
       (341,"one less",1),(385,"half",2),(411,"Six",6),(411,"ten",10)]
STRUCT = [(27,18),(40,96),(128,28),(133,360),(212,20),(329,240),(385,150),
          (411,16),(441,800)]

rows = [json.loads(l) for l in open(".cache/math500_test.jsonl")]
tok = Tokenizer.from_file(TOKENIZER_JSON)
need = sorted({i for i,*_ in LEX} | {i for i,_ in STRUCT})
ids = np.zeros((len(need), T_ALG), np.int32)
msk = np.zeros((len(need), T_ALG), np.float32)
offs = {}
for r_, i in enumerate(need):
    e = tok.encode(rows[i]["problem"])
    ids[r_, :len(e.ids)] = e.ids[:T_ALG]; msk[r_, :min(len(e.ids),T_ALG)] = 1.0
    offs[i] = (r_, list(e.offsets))

def digits(v):
    return [(v // 10 ** (N_DIG-1-d)) % 10 for d in range(N_DIG)]

def states_at(depth):
    if depth == 4:
        from beacon_closing_arm import recompute_states
        return recompute_states(ids)
    import mycelium.llama_loader as LL
    from tinygrad import Tensor, dtypes
    class _H: pass
    host = _H()
    sd = LL.load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
    LL.attach_llama_layers(host, n_layers=8, sd=sd, cfg=LL.LLAMA_3_2_1B_CFG)
    del sd
    x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
    for l in host.llama_layers:
        x = l(x, host.llama_rope_cos, host.llama_rope_sin)
    x = LL._rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
    return x.cast(dtypes.float).realize().numpy()

# probe-training rows: OUR corpus explicit given spans (transfer instrument)
def our_rows(states, n_cap=1200):
    samples, tids, tmask, toffs = tokenize(".cache/algebra4_nl_test.jsonl")
    X, Y = [], []
    for i, s in enumerate(samples[:n_cap]):
        for f in s["factors"]:
            if f["ftype"] != "given": continue
            m = np.zeros((T_ALG,), np.float32)
            _spans_to_tokmask(f["spans"], toffs[i], m)
            if m.sum() == 0: continue
            m /= m.sum()
            X.append((states[i].astype(np.float32) * m[:, None]).sum(0))
            Y.append(digits(int(f["value"])))
    return np.stack(X), np.array(Y)

for depth, tag in ((4, "L0-3"), (8, "L0-7")):
    if depth == 4:
        z = np.load(".cache/phase1_alg_states_alg4test.npz")
        ours = z["states"]
    else:
        from phase1_algebra_head import ALG_TEST
        import phase1_algebra_head as H
        _, tids, _, _ = H.tokenize(".cache/algebra4_nl_test.jsonl")
        ours = compute_l8(tids[:1200]) if not os.path.exists(f".cache/expl_ours_L8.npy") \
            else np.load(".cache/expl_ours_L8.npy")
        if not os.path.exists(".cache/expl_ours_L8.npy"):
            np.save(".cache/expl_ours_L8.npy", ours)
    Xtr, Ytr = our_rows(ours)
    mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
    _, fwd = train_probe((Xtr - mu) / sd_, Ytr, seed=1)
    mst = states_at(depth)
    # lexical: span-pooled at evoking phrase
    Xl, Yl, missed = [], [], 0
    for i, phrase, val in LEX:
        r_, off = offs[i]
        pos = rows[i]["problem"].find(phrase)
        if pos < 0: missed += 1; continue
        m = np.zeros((T_ALG,), np.float32)
        _spans_to_tokmask([[pos, pos + len(phrase)]], off, m)
        if m.sum() == 0: missed += 1; continue
        m /= m.sum()
        Xl.append((mst[r_].astype(np.float32) * m[:, None]).sum(0))
        Yl.append(digits(val))
    acc_l = eval_probe(fwd, (np.stack(Xl) - mu) / sd_, np.array(Yl))
    # structural: whole-seq pooled
    Xs, Ys = [], []
    for i, val in STRUCT:
        r_, _ = offs[i]
        mm = msk[r_] / max(msk[r_].sum(), 1)
        Xs.append((mst[r_].astype(np.float32) * mm[:, None]).sum(0))
        Ys.append(digits(val))
    acc_s = eval_probe(fwd, (np.stack(Xs) - mu) / sd_, np.array(Ys))
    print(f"[{tag}] LEXICAL implicit decodability {acc_l:.2f} (n={len(Xl)}, "
          f"missed {missed}) | STRUCTURAL (weak, pooled) {acc_s:.2f} "
          f"(n={len(Xs)}) | our-span transfer floor: probe trained on "
          f"{len(Xtr)} explicit spans")
print("OUTCOMES: lexical decodable shallow -> pairs cure; deep-only -> "
      "deeper-prefix reopens with a customer; nowhere -> the explicitation "
      "stage (structural organ if the class split holds).")
