"""enriched_capture.py — the decisive fixture (2026-07-26): ALL 78
boundary-crossers + 78 controls matched BY CONSTRUCTION on the four floor
features (length/n_vars/digits/sentences). Per-layer trunk states banked;
discriminant read runs offline. Bars at 16c15b0."""
import sys, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
from beacon_closing_arm import _trunk_host, T_ALG, H_TRUNK
from phase1_algebra_head import TOKENIZER_JSON
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
def out(lat, i):
    nn = [a for a in lat[i] if a is not None]
    c = Counter(nn).most_common(1); p, n = c[0] if c else (None, 0)
    return "right" if n >= 3 and p == gold[i] else ("lie" if n >= 3 else "abstain")
g21 = json.load(open(".cache/lattice_gen21_H.json"))["bigtest"]
g16 = json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"]
o21 = [out(g21, i) for i in range(1500)]; o16 = [out(g16, i) for i in range(1500)]
crossers = [i for i in range(1500) if o21[i] != o16[i] and "lie" not in (o21[i], o16[i])]
noncross = [i for i in range(1500) if o21[i] == o16[i] and o21[i] != "lie"]
feats = np.stack([
    np.array([len(rows[i]["text"]) for i in range(1500)]),
    np.array([rows[i].get("n_vars", len(rows[i]["solution"])) for i in range(1500)]),
    np.array([len(re.findall(r"\d+", rows[i]["text"])) for i in range(1500)]),
    np.array([rows[i]["text"].count(".") for i in range(1500)])], 1).astype(np.float32)
Fz = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
used = set(); controls = []
for i in crossers:
    d = np.linalg.norm(Fz[noncross] - Fz[i], axis=1)
    for j in np.argsort(d):
        if noncross[j] not in used:
            controls.append(noncross[j]); used.add(noncross[j]); break
pick = np.array(crossers + controls, dtype=np.int32)
print(f"[enriched] crossers {len(crossers)} | controls {len(controls)} | control outcome mix {Counter(o21[i] for i in controls)}", flush=True)
match_d = np.mean([np.linalg.norm(Fz[c]-Fz[k]) for c, k in zip(crossers, controls)])
print(f"[enriched] mean matched feature distance {match_d:.3f} (z-units)", flush=True)

tok = Tokenizer.from_file(TOKENIZER_JSON)
host = _trunk_host()
N = len(pick)
states = np.zeros((5, N, T_ALG, H_TRUNK), np.float16)
ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
for j, i in enumerate(pick):
    e = tok.encode(rows[i]["text"]); L = min(len(e.ids), T_ALG)
    ids[j, :L] = e.ids[:L]; msk[j, :L] = 1.0
from mycelium.llama_loader import _rms_norm
for s0 in range(0, N, 8):
    sl = slice(s0, min(s0+8, N))
    x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
    states[0, sl] = x.cast(dtypes.float).realize().numpy().astype(np.float16)
    for li, layer in enumerate(host.llama_layers):
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        states[li+1, sl] = x.cast(dtypes.float).realize().numpy().astype(np.float16)
    print(f"[enriched] {min(s0+8,N)}/{N}", flush=True)
np.savez_compressed(".cache/enriched_capture.npz", states=states, mask=msk,
    items=pick, is_crosser=np.array([1]*len(crossers)+[0]*len(controls), np.int8),
    y21=np.array([1 if o21[i]=="right" else 0 for i in pick], np.int8),
    y16=np.array([1 if o16[i]=="right" else 0 for i in pick], np.int8))
print("[enriched] banked -> .cache/enriched_capture.npz", flush=True)
