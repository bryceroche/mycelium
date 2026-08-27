"""depth_telegraph.py — THE DEPTH-TELEGRAPH READ (2026-08-27; Bryce's
bet: full-depth Llama carries layer-wise ping-pong with ~4-layer
half-period — our L0..L3 window sliced the telegraph in half, and the
head's laminarity is one speaker with no interlocutor). Instrument:
yesterday's exact reversal meter applied ACROSS DEPTH — full 16-layer
trunk on the 143 golds, per-layer pooled states, direction-reversal
rate of consecutive layer-deltas vs shuffled-layer null + reversal
autocorrelation (period hunt at lag ~4).
BARS (pinned): depth reversals z > 3 ABOVE null (breaths read z=-85
BELOW) = the telegraph lives across depth; autocorr peak at lag 4+-1 =
the four-layer-half bet confirmed as stated.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, load_alg
from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                   LLAMA_3_2_1B_CFG)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
rng = np.random.default_rng(0)
NPERM = 200
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    class _H: pass
    host = _H()
    sd = load_llama_weights(os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=16, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    P = []       # per row: (n_layers+1, H) pooled trajectory incl. embed
    for s0 in range(0, len(rows), 4):
        sl = rows[s0:s0 + 4]
        ids = np.zeros((4, T_ALG), np.int32); msk = np.zeros((4, T_ALG), np.float32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        m = msk[:, :, None]
        traj = [ (x.cast(dtypes.float).numpy() * m).sum(1) / np.maximum(m.sum(1), 1) ]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            traj.append((x.cast(dtypes.float).numpy() * m).sum(1)
                        / np.maximum(m.sum(1), 1))
        T2 = np.stack(traj, axis=1)          # (4, L+1, H)
        for li in range(len(sl)):
            P.append(T2[li])
    P = np.stack(P)
    print(f"[dtg] rows {len(P)}, layer trajectory {P.shape[1]} points",
          flush=True)

    def rev_pattern(V):
        dv = np.diff(V, axis=0)
        dn = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
        c = (dn[1:] * dn[:-1]).sum(1)
        return (c < 0).astype(np.float32)     # per-boundary reversal flags
    pats = np.stack([rev_pattern(V) for V in P])   # (n, L-1)
    obs = pats.mean()
    per_boundary = pats.mean(0)
    null = []
    for _ in range(NPERM):
        tot = []
        for V in P[:200]:
            idx = rng.permutation(len(V))
            tot.append(rev_pattern(V[idx]).mean())
        null.append(np.mean(tot))
    mu, sdv = np.mean(null), np.std(null) + 1e-9
    z = (obs - mu) / sdv
    print(f"[dtg] DEPTH reversal rate {obs:.4f} vs null {mu:.4f}+-{sdv:.4f} "
          f" z={z:+.2f} (breath-grain was 0.0000, z=-85)", flush=True)
    print("[dtg] per-boundary reversal profile (L1..L15): "
          + " ".join(f"{v:.2f}" for v in per_boundary), flush=True)
    # period hunt: autocorrelation of the mean-centered boundary profile
    x = per_boundary - per_boundary.mean()
    ac = [1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1])
                  for k in range(1, 8)]
    print("[dtg] reversal-profile autocorr lags 0..7: "
          + " ".join(f"{v:+.2f}" for v in ac), flush=True)
    peak = int(np.argmax(ac[2:])) + 2
    print(f"[dtg] VERDICT: depth-telegraph {'CONFIRMED' if z > 3 else 'NOT above null'}"
          f"; strongest period lag {peak} (bet: 4+-1"
          f" -> {'BET CONFIRMED' if z > 3 and 3 <= peak <= 5 else 'bet not as stated'})",
          flush=True)

if __name__ == "__main__":
    main()
