"""pythia_depth.py — CROSS-LINEAGE DEPTH ANATOMY (2026-08-27, word
given): the depth-telegraph instrument on Pythia-410M (24 layers,
GPT-NeoX lineage — a different family from Llama). Question: is the
five-phase anatomy (turn/zigzag/mixed/glide/turn) Llama's fingerprint
or a universal grammar of transformer depth? Same meter: per-layer
pooled trajectories on the 143 golds, reversal profile vs shuffled
null, autocorrelation period hunt. A replicated anatomy = a Shape of
Thought cross-lineage datum; a different one = family fingerprints.
"""
import os, sys, json, glob
os.environ.setdefault("DEV", "AMD")
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from mycelium.config import Config
from mycelium.loader import load_pythia_baseline
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes

rng = np.random.default_rng(0)
NPERM = 200
TMAX = 384

def main():
    cfg = Config()
    stack = load_pythia_baseline(cfg, n_layers=24)
    ptok = Tokenizer.from_file('.cache/pythia-410m/tokenizer.json')
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    P = []
    for s0 in range(0, len(rows), 4):
        sl = rows[s0:s0 + 4]
        ids = np.zeros((4, TMAX), np.int32); msk = np.zeros((4, TMAX), np.float32)
        for li, r in enumerate(sl):
            e = ptok.encode(r["original"])
            tid = e.ids[:TMAX]
            ids[li, :len(tid)] = tid; msk[li, :len(tid)] = 1.0
        sts = stack.hidden_states(Tensor(ids, dtype=dtypes.int))
        m = msk[:, :, None]
        traj = [(s.cast(dtypes.float).numpy() * m).sum(1)
                / np.maximum(m.sum(1), 1) for s in sts]
        T2 = np.stack(traj, axis=1)          # (4, 25, H)
        for li in range(len(sl)):
            P.append(T2[li])
    P = np.stack(P)
    print(f"[pd] rows {len(P)}, trajectory {P.shape[1]} points (embed + 24)",
          flush=True)

    def rev_pattern(V):
        dv = np.diff(V, axis=0)
        dn = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
        c = (dn[1:] * dn[:-1]).sum(1)
        return (c < 0).astype(np.float32)
    pats = np.stack([rev_pattern(V) for V in P])
    obs = pats.mean(); prof = pats.mean(0)
    null = []
    for _ in range(NPERM):
        tot = []
        for V in P[:100]:
            idx = rng.permutation(len(V))
            tot.append(rev_pattern(V[idx]).mean())
        null.append(np.mean(tot))
    mu, sdv = np.mean(null), np.std(null) + 1e-9
    z = (obs - mu) / sdv
    print(f"[pd] DEPTH reversal rate {obs:.4f} vs null {mu:.4f}+-{sdv:.4f} "
          f"z={z:+.2f}  (llama-1B was 0.657, z=-24)", flush=True)
    print("[pd] per-boundary profile (L1..L23): "
          + " ".join(f"{v:.2f}" for v in prof), flush=True)
    x = prof - prof.mean()
    ac = [1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 12)]
    print("[pd] autocorr lags 0..11: " + " ".join(f"{v:+.2f}" for v in ac),
          flush=True)

if __name__ == "__main__":
    main()
