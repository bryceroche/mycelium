"""parser_layer_capture.py — the parser-side capture (2026-07-26): per-layer
trunk states (embed + L0..L3 post-layer) on a zone-stratified bigtest
subset; labels joinable offline (bank-don't-read). Registration f38fcbf."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
from beacon_closing_arm import _trunk_host, T_ALG, H_TRUNK
from phase1_algebra_head import TOKENIZER_JSON, sent_indices
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from mycelium.llama_loader import _rms_norm

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
lat = json.load(open(".cache/lattice_gen21_H.json"))["bigtest"]
def zone(i):
    right = sum(1 for a in lat[i] if a == gold[i])
    return "umbra" if right == 5 else ("dark" if right == 0 else "penumbra")
def outcome(i):
    nn = [a for a in lat[i] if a is not None]
    c = Counter(nn).most_common(1)
    plur, cnt = c[0] if c else (None, 0)
    return "right" if cnt >= 3 and plur == gold[i] else ("lie" if cnt >= 3 else "abstain")
rng = np.random.default_rng(0)
by_z = {}
for i in range(len(rows)): by_z.setdefault(zone(i), []).append(i)
pick = np.concatenate([rng.choice(by_z[z], min(80, len(by_z[z])), replace=False)
                       for z in ("umbra", "penumbra", "dark")])
print(f"[capture] {len(pick)} items: " + str(Counter(zone(i) for i in pick)), flush=True)

tok = Tokenizer.from_file(TOKENIZER_JSON)
host = _trunk_host()
N = len(pick)
states = np.zeros((5, N, T_ALG, H_TRUNK), np.float16)
msk = np.zeros((N, T_ALG), np.float32)
ids = np.zeros((N, T_ALG), np.int32)
for j, i in enumerate(pick):
    e = tok.encode(rows[i]["text"]); L = min(len(e.ids), T_ALG)
    ids[j, :L] = e.ids[:L]; msk[j, :L] = 1.0
for s0 in range(0, N, 8):
    sl = slice(s0, min(s0+8, N))
    x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
    states[0, sl] = x.cast(dtypes.float).realize().numpy().astype(np.float16)
    for li, layer in enumerate(host.llama_layers):
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        c = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(c).all()
        states[li+1, sl] = c.astype(np.float16)
    print(f"[capture] {min(s0+8,N)}/{N}", flush=True)
np.savez_compressed(".cache/parser_layer_capture.npz",
    states=states, mask=msk, items=pick.astype(np.int32),
    zone=np.array([zone(i) for i in pick]),
    outcome=np.array([outcome(i) for i in pick]))
print("[capture] banked -> .cache/parser_layer_capture.npz", flush=True)
