"""padleak_counterfactual.py (2026-10-14) -- zero-GPU (DEV=CPU). Opus's side test, content-controlled:
the SAME text through the deployed trunk regime (L0-L3, no mask, right-padded with id 0 '!' + the final RMSNorm, exactly
do_precompute) at frame lengths T in {192, 256, 320}: only the PAD COUNT changes. Reads: (a) per-token cosine of the
D4 state between frames; (b) the pooled-state shift d = pool(T=320) - pool(T=192) per row, and whether that shift is a
SHARED direction across rows (mean pairwise cosine of the d's; random 2048-d ~ 0); (c) the size of the shift relative to
the between-row spread of pooled states (|d| / RMS distance between different rows at T=256)."""
import os, sys, json
os.environ["DEV"] = "CPU"
sys.path.insert(0, ".")
import numpy as np
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
N = int(os.environ.get("N_ROWS", 24))
rng = np.random.default_rng(0)
lines = open(".cache/form_mix_pm35c.jsonl").readlines()
rows = [t for t in (json.loads(lines[i])["text"] for i in rng.choice(len(lines), 4 * N, replace=False)) if len(tok.encode(t).ids) <= 160][:N]
class _H: pass
h = _H(); sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(h, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG); del sd
def deployed(ids, T):
    x = np.zeros((1, T), np.int32); x[0, :len(ids)] = ids
    x = h.llama_embed[Tensor(x, dtype=dtypes.int)]
    for L in h.llama_layers: x = L(x, h.llama_rope_cos, h.llama_rope_sin)
    return _rms_norm(x, h.llama_layers[-1].ffn_norm, h.llama_cfg.rms_norm_eps).numpy()[0, :len(ids)]
FR = [192, 256, 320]
pools = {T: [] for T in FR}; tokcos = {T: [] for T in FR}; lens = []
for t in rows:
    ids = np.array(tok.encode(t).ids, np.int32); n = len(ids); lens.append(n)
    s = {T: deployed(ids, T) for T in FR}
    for T in FR:
        a, b = s[T], s[256]
        tokcos[T].append(np.mean(np.sum(a*b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)))
        pools[T].append(s[T].mean(0))
P = {T: np.stack(v) for T, v in pools.items()}
d = P[320] - P[192]
dn = d / np.linalg.norm(d, axis=1, keepdims=True)
C = dn @ dn.T; off = C[~np.eye(N, dtype=bool)]
between = np.sqrt(np.mean([np.sum((P[256][i]-P[256][j])**2) for i in range(N) for j in range(i+1, N)]))
print(f"rows {N}, real lengths {min(lens)}..{max(lens)} (mean {np.mean(lens):.0f})")
for T in FR: print(f"(a) per-token cos(D4 at T={T}, D4 at T=256) mean {np.mean(tokcos[T]):.4f}  min {np.min(tokcos[T]):.4f}")
print(f"(b) pad-shift direction shared across rows: mean pairwise cos {off.mean():.3f} (min {off.min():.3f})")
print(f"(c) |pool shift 192->320| mean {np.linalg.norm(d,axis=1).mean():.3f}  vs between-row RMS distance at T=256 {between:.3f}  ratio {np.linalg.norm(d,axis=1).mean()/between:.2f}")
mu = dn.mean(0); mu /= np.linalg.norm(mu)
proj = P[256] @ mu
print(f"(d) projection of T=256 pooled states on the shared pad axis vs real length: corr {np.corrcoef(proj, lens)[0,1]:.3f}")
