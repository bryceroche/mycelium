"""granite_depth.py — THIRD-FAMILY DEPTH ANATOMY (2026-08-27, word
given): the reversal meter on IBM Granite-4.2-3B — 40-layer dense
transformer, REASONING-POST-TRAINED (the first specimen whose training
targeted the thing the telegraph gut listens for). Granite-isms:
attention_multiplier 1/64 replaces 1/sqrt(d); embedding/residual
multipliers are 1.0 (no-ops); weights Llama-convention, sharded x2.
Same meter, same 143 golds, same nulls. Shelf so far: Llama-1B base
(early zigzag + mixed zone), Pythia-410M base (punctuated glide).
"""
import os, sys, json, glob
os.environ.setdefault("DEV", "AMD")
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from mycelium.llama_loader import (attach_llama_layers, LlamaConfig)
from tinygrad.nn.state import safe_load
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes

rng = np.random.default_rng(0)
NPERM = 200
TMAX = 384
GDIR = '.cache/granite-4.2-3b'

def main():
    cfg = LlamaConfig(hidden_size=2560, intermediate_size=8192,
                      num_hidden_layers=40, num_attention_heads=40,
                      num_key_value_heads=8, vocab_size=100352,
                      rms_norm_eps=1e-5, rope_theta=10000000.0)
    sd = {}
    for shard in sorted(glob.glob(f'{GDIR}/model-*.safetensors')):
        sd.update(safe_load(shard))
    print(f"[gd] merged {len(sd)} weight keys", flush=True)
    class _H: pass
    host = _H()
    attach_llama_layers(host, n_layers=40, sd=sd, cfg=cfg)
    del sd
    for layer in host.llama_layers:
        layer._scale = 0.015625          # granite attention_multiplier
    gtok = Tokenizer.from_file(f'{GDIR}/tokenizer.json')
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    P = []
    for s0 in range(0, len(rows), 2):
        sl = rows[s0:s0 + 2]
        ids = np.zeros((2, TMAX), np.int32); msk = np.zeros((2, TMAX), np.float32)
        for li, r in enumerate(sl):
            tid = gtok.encode(r["original"]).ids[:TMAX]
            ids[li, :len(tid)] = tid; msk[li, :len(tid)] = 1.0
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        m = msk[:, :, None]
        traj = [(x.cast(dtypes.float).numpy() * m).sum(1) / np.maximum(m.sum(1), 1)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            traj.append((x.cast(dtypes.float).numpy() * m).sum(1)
                        / np.maximum(m.sum(1), 1))
        T2 = np.stack(traj, axis=1)
        for li in range(len(sl)):
            P.append(T2[li])
    P = np.stack(P)
    print(f"[gd] rows {len(P)}, trajectory {P.shape[1]} points (embed + 40)",
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
        for V in P[:80]:
            idx = rng.permutation(len(V))
            tot.append(rev_pattern(V[idx]).mean())
        null.append(np.mean(tot))
    mu, sdv = np.mean(null), np.std(null) + 1e-9
    print(f"[gd] DEPTH reversal rate {obs:.4f} vs null {mu:.4f}+-{sdv:.4f} "
          f"z={(obs-mu)/sdv:+.2f}  (llama-1B 0.657 | pythia 0.195)", flush=True)
    print("[gd] per-boundary profile (L1..L39): "
          + " ".join(f"{v:.2f}" for v in prof), flush=True)
    x = prof - prof.mean()
    ac = [1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 16)]
    print("[gd] autocorr lags 0..15: " + " ".join(f"{v:+.2f}" for v in ac),
          flush=True)

if __name__ == "__main__":
    main()
