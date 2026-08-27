"""telegraph_read.py — THE TELEGRAPH READ (2026-08-27, gut-audit fire):
does the breath trajectory carry half-duplex ALTERNATION (read/check
ping-pong — the model conversing with an internal checker through an
internal interface) or is it pure diffusive lowering (the ladder's
population-grain verdict)? Two channels, shuffled-order nulls:
  (a) DISCRETE: per-slot 7-cycle RAW nearest-centroid sequences (no
      Viterbi — the transition prior would smooth away ping-pong);
      A->B->A return-triplet rate vs order-shuffled null (100 perms).
  (b) CONTINUOUS (atlas-free): direction-reversal rate of consecutive
      cycle-velocity vectors (cos < 0) vs order-shuffled null.
BARS (pinned): telegraph confirmed per channel at z > 3 AND discrete
absolute rate > 0.10. The ladder predicts null; the gut predicts
returns. One of them is wrong on the record today.
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.setdefault("ATLAS_TABLE", "waist_patterns_op")
os.environ.setdefault("ATLAS_TRANS", "op_transitions")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, T_ALG, TOKENIZER_JSON,
                                 sent_indices, load_alg, build_slot_masks,
                                 L_FAC)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from chain_decode import load_atlas

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
rng = np.random.default_rng(0)
NPERM = 100

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    banks = {}
    for cyc in cycles:
        bids = list(cents[cyc].keys())
        M = np.stack([cents[cyc][b] for b in bids])
        banks[cyc] = (bids, M)
    p = build_params(0)
    sd = safe_load('.cache/gsb227_real.safetensors')
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    seqs = []            # per slot: list of cluster ids across cycles
    vecs = []            # per slot: (K, H) state vectors
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        ts = Tensor(sts, dtype=dtypes.float)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k2: o0[k2].realize().numpy() for k2 in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        Bst = [b.realize().numpy() for b in o["breaths_all"]]
        pres = o["pres"].realize().numpy()
        for li in range(len(sl)):
            for j in range(L_FAC):
                if pres[li, j] <= 0.5: continue
                V = np.stack([Bst[min(ci, len(Bst) - 1)][li, j]
                              for ci in range(len(cycles))])
                Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
                seq = []
                for ci, cyc in enumerate(cycles):
                    bids, M = banks[cyc]
                    seq.append(bids[int((M @ Vn[ci]).argmax())])
                seqs.append(seq)
                vecs.append(V)
    print(f"[tg] slots: {len(seqs)} across {len(rows)} golds", flush=True)

    def ret_rate(s):
        n = 0; d = 0
        for t in range(2, len(s)):
            d += 1
            if s[t] == s[t - 2] and s[t] != s[t - 1]: n += 1
        return n, d
    obs_n = obs_d = 0
    for s in seqs:
        n, d = ret_rate(s); obs_n += n; obs_d += d
    obs = obs_n / max(obs_d, 1)
    null = []
    for _ in range(NPERM):
        tn = td = 0
        for s in seqs:
            sp = list(s); rng.shuffle(sp)
            n, d = ret_rate(sp); tn += n; td += d
        null.append(tn / max(td, 1))
    mu, sdv = np.mean(null), np.std(null) + 1e-9
    z_a = (obs - mu) / sdv
    print(f"[tg] DISCRETE A->B->A: rate {obs:.4f} vs null {mu:.4f}+-{sdv:.4f}"
          f"  z={z_a:+.2f}  (bars: z>3 AND rate>0.10)", flush=True)

    def rev_rate(V):
        dv = np.diff(V, axis=0)
        dn = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
        c = (dn[1:] * dn[:-1]).sum(1)
        return (c < 0).sum(), len(c)
    on = od = 0
    for V in vecs:
        n, d = rev_rate(V); on += n; od += d
    obs_v = on / max(od, 1)
    nullv = []
    for _ in range(NPERM):
        tn = td = 0
        for V in vecs:
            idx = rng.permutation(len(V))
            n, d = rev_rate(V[idx]); tn += n; td += d
        nullv.append(tn / max(td, 1))
    muv, sdvv = np.mean(nullv), np.std(nullv) + 1e-9
    z_b = (obs_v - muv) / sdvv
    print(f"[tg] CONTINUOUS reversals: rate {obs_v:.4f} vs null {muv:.4f}"
          f"+-{sdvv:.4f}  z={z_b:+.2f}", flush=True)
    tele = (z_a > 3 and obs > 0.10)
    print("[tg] VERDICT: " + ("TELEGRAPH — alternation beyond null; the "
          "half-duplex exchange is real" if tele else
          "no telegraph at slot grain — diffusive lowering stands (the "
          "ladder's verdict holds; the gut's drawer read honestly)"),
          flush=True)

if __name__ == "__main__":
    main()
