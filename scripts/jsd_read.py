"""jsd_read.py — THE JSD AUDITION (2026-08-24, word given): does some
waist subspace SEPARATE what the bank reads from what it cannot?
Phase A (GPU, ~1 min): g65-adapter states for the 180 fixture rows ->
waist activations (gelu(states@W+b) + sent_emb), mask-mean pooled.
Phase B (CPU): per-dim Jensen-Shannon divergence readable-vs-residue
(readable = union-right rows from recruit_round1.json), top-subspace
AUC on a held-out split, residue clustering (k-means, silhouette).
BARS (pinned pre-read): AUC >= 0.75 on the held split = a separating
subspace EXISTS (the segmenter earns its build); below = the residue is
diet-shaped (round 2 answers with candidates). Preview grain: 17-reader
residue; the verdict of record re-runs post-R24.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest", "ALG_TRUNK_LORA": "1"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, T_ALG, TOKENIZER_JSON,
                                 sent_indices, load_alg)
from beacon_closing_arm import recompute_states, _trunk_host
from mycelium.llama_loader import _rms_norm
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0); sd = safe_load('.cache/g65_bridge.safetensors')
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
host = _trunk_host()
LD = [{f"{nm}_{ab}": (p[f"lora{li}_{nm}_{ab}"] * (8.0 if ab == "B" else 1.0))
       for nm in ("wq", "wo", "wdown") for ab in ("A", "B")} for li in range(4)]

# fixture identical to audition_read_one's (order matters — indices align
# with the recruiter's row indexing)
sys.argv = []
import importlib
aro = open('scripts/audition_read_one.py').read()
# reuse its fixtures() verbatim by exec-ing just that function's deps
_ns = {"json": json, "glob": glob, "np": np}
exec(aro[aro.index("def fixtures():"):aro.index("rows = fixtures()")], _ns)
rows = _ns["fixtures"]()
print(f"[jsd] fixture rows: {len(rows)}", flush=True)

W_w = p["waist_w"].numpy(); W_b = p["waist_b"].numpy()
SE = p["sent_emb"].numpy()
feats = np.zeros((len(rows), W_w.shape[1]), np.float32)
for s0 in range(0, len(rows), 16):
    sl = rows[s0:s0 + 16]
    ids = np.zeros((16, T_ALG), np.int32); msk = np.zeros((16, T_ALG), np.float32)
    snt = np.zeros((16, T_ALG), np.int32)
    for li, r in enumerate(sl):
        e = tok.encode(r["original"])
        if len(e.ids) > T_ALG: continue
        ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
        snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
    x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
    for li2, layer in enumerate(host.llama_layers):
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin, lora=LD[li2])
    x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
    sts = x.cast(dtypes.float).realize().numpy()
    for li in range(len(sl)):
        h = sts[li] @ W_w + W_b
        h = 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
        h = h + SE[snt[li]]
        m = msk[li][:, None]
        feats[s0 + li] = (h * m).sum(0) / max(m.sum(), 1)
np.save('.cache/jsd_feats.npy', feats)
print("[jsd] waist features banked (.cache/jsd_feats.npy) — GPU phase done",
      flush=True)
