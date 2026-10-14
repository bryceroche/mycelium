"""depthprobe_collect.py -- THE DEPTH PROBE's trunk pass at scale (2026-10-14; replaces the dead agent's
16-row smoke, ledger 10-14 ~16:00).

Three attention regimes over the frozen Llama-3.2-1B trunk, all on the T_ALG=256 frame (right-padded with id 0 = '!'):
  native  : causal + pad keys masked   (what Llama's deep layers were trained for; THE DECISION TABLE reads here)
  deployed: no mask, pads visible      (do_precompute today, scripts/phase1_algebra_head.py:3090)
  bidir   : bidirectional, pad keys masked (the cheapest stack fix; isolates the pad leak from the direction change)
Depth naming: D0 = embedding, Dk = after k Llama layers (D4 = the deployed tap, 'L0-L3', the July 'L4');
EV1..EV4 = the every-fourth stack (embed -> layer 0 -> 4 -> 8 -> 12), stage outputs. 21 taps total.

Rows: the climb targets' pool (form_mix_pm35c), not truncated, >= 2 numerals with an owner link; seeded sample.
Storage (ragged, fp16 memmaps, one per regime, layout (21, Npos, 2048)):
  FULL rows (the first N_FULL of the sample): every real token (the tree probe);
  the rest: the union of the numeral / noun / verb / owner piece positions (owner-pair, which-later, value probes).
Index: .cache/depthprobe2_index.npz (row, pos per stored position; per-numeral spans + surfaces + value; token ids
and char offsets for the FULL rows).

--selftest (DEV=CPU): native-padded == causal-unpadded and bidir-padded == bidir-unpadded on 2 rows, every tap.
Run on the card under .cache/gpu.lock, DEV=PCI+AMD.
"""
import os, sys, json, time, argparse
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n_rows", type=int, default=1500)
ap.add_argument("--n_full", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--out", default=".cache/depthprobe2")
args = ap.parse_args()
if args.selftest:
    assert os.environ.get("DEV") == "CPU", "selftest runs DEV=CPU"
else:
    assert os.environ.get("DEV") == "PCI+AMD", "DEV=PCI+AMD (never bare AMD)"

from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG

T = 256
B = 8
H = 2048
REGIMES = ["native", "deployed", "bidir"]
TAPS = [f"D{k}" for k in range(17)] + ["EV1", "EV2", "EV3", "EV4"]
EV_LAYERS = [0, 4, 8, 12]
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
assert tok.id_to_token(0) == "!"


def load_host(n_layers):
    class _H: pass
    h = _H()
    sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
    attach_llama_layers(h, n_layers=n_layers, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    return h


def make_mask(regime, msk):
    """msk (b, S) 1=real. Returns an additive (b,1,S,S) Tensor or None."""
    if regime == "deployed":
        return None
    b, S = msk.shape
    m = np.where(msk[:, None, None, :] > 0, 0.0, -1e4).astype(np.float32)       # pad keys masked
    m = np.broadcast_to(m, (b, 1, S, S)).copy()
    if regime == "native":
        m = m + np.triu(np.full((S, S), -1e4, np.float32), 1)[None, None]
    return Tensor(m)


def taps(h, ids, regime, msk):
    """Returns list of 21 arrays (b, S, H) float32."""
    mask = make_mask(regime, msk)
    rc, rs = h.llama_rope_cos, h.llama_rope_sin
    x0 = h.llama_embed[Tensor(ids, dtype=dtypes.int)]
    out = [x0.numpy()]
    x = x0
    for L in h.llama_layers:
        x = L(x, rc, rs, attn_mask=mask).realize()
        out.append(x.numpy())
    x = x0
    for li in EV_LAYERS:
        x = h.llama_layers[li](x, rc, rs, attn_mask=mask).realize()
        out.append(x.numpy())
    return out


if args.selftest:
    h = load_host(16)
    rows = [json.loads(l) for _, l in zip(range(2), open(".cache/form_mix_pm35c.jsonl"))]
    worst = 0.0
    for r in rows:
        ids = np.array(tok.encode(r["text"]).ids, np.int32); n = len(ids)
        pad = np.zeros((1, T), np.int32); pad[0, :n] = ids
        msk = np.zeros((1, T), np.float32); msk[0, :n] = 1
        for regime, ref_mask in [("native", Tensor(np.triu(np.full((n, n), -1e4, np.float32), 1)[None, None])),
                                 ("bidir", None)]:
            got = taps(h, pad, regime, msk)
            # reference: unpadded, same direction
            rc, rs = h.llama_rope_cos, h.llama_rope_sin
            x0 = h.llama_embed[Tensor(ids[None], dtype=dtypes.int)]
            ref = [x0.numpy()]; x = x0
            for L in h.llama_layers:
                x = L(x, rc, rs, attn_mask=ref_mask).realize(); ref.append(x.numpy())
            x = x0
            for li in EV_LAYERS:
                x = h.llama_layers[li](x, rc, rs, attn_mask=ref_mask).realize(); ref.append(x.numpy())
            for ti, (g, f) in enumerate(zip(got, ref)):
                rel = float(np.abs(g[0, :n] - f[0]).max() / (np.abs(f[0]).max() + 1e-9))
                worst = max(worst, rel)
                assert rel < 1e-3, f"selftest FAIL {regime} {TAPS[ti]} rel {rel:.2e}"
    print(f"[selftest] PASS: padded+masked == unpadded on 2 rows x 2 regimes x 21 taps (worst rel {worst:.1e})")
    sys.exit(0)

# ---------------------------------------------------------------- row selection
ct = np.load(".cache/climb_targets_formpm35c.npz", allow_pickle=True)
nn = ct["n_num"]; ov = ct["owner_valid"]
eligible = np.where((ct["truncated"] == 0) & (nn >= 2) & (ov.sum(1) >= 2))[0]
rng = np.random.default_rng(args.seed)
pick = rng.choice(eligible, size=args.n_rows, replace=False)
pool_idx = ct["row_idx"][pick]
need = set(int(i) for i in pool_idx)
texts = {}
for i, l in enumerate(open(".cache/form_mix_pm35c.jsonl")):
    if i in need:
        texts[i] = json.loads(l)["text"]
assert len(texts) == len(need)
print(f"[collect] eligible {len(eligible)} rows; picked {len(pick)} (seed {args.seed}); full {args.n_full}", flush=True)

# per-row positions to store + numeral table
row_ids, row_off, row_pos = [], [], []
num_tab = {k: [] for k in ["row", "k", "num_s", "num_e", "noun", "verb", "owner", "owner_pron",
                          "owner_same_sent", "num_text", "noun_text", "owner_text", "value"]}
for ri, (ci, pi) in enumerate(zip(pick, pool_idx)):
    e = tok.encode(texts[int(pi)])
    ids = np.array(e.ids, np.int32); n = len(ids)
    assert n <= T
    pos = set()
    for k in range(int(nn[ci])):
        s, en = ct["numeral_span"][ci, k]
        if s < 0:
            continue
        nv, vv, ow = ct["noun_valid"][ci, k], ct["verb_valid"][ci, k], ct["owner_valid"][ci, k]
        noun = int(ct["noun_pos"][ci, k]) if nv else -1
        verb = int(ct["verb_pos"][ci, k]) if vv else -1
        own = int(ct["owner_pos"][ci, k]) if ow else -1
        pos.update(range(int(s), int(en)))
        for p in (noun, verb, own):
            if p >= 0: pos.add(p)
        ntext = tok.decode(ids[s:en].tolist()).strip()
        val = int(ntext) if ntext.isdigit() and int(ntext) <= 999 else -1
        dec = lambda p: tok.decode([int(ids[p])]).strip().lower() if p >= 0 else ""
        for kk, vvv in [("row", ri), ("k", k), ("num_s", int(s)), ("num_e", int(en)), ("noun", noun),
                        ("verb", verb), ("owner", own), ("owner_pron", int(ct["owner_pronoun"][ci, k])),
                        ("owner_same_sent", int(ct["owner_same_sent"][ci, k])), ("num_text", ntext),
                        ("noun_text", dec(noun)), ("owner_text", dec(own)), ("value", val)]:
            num_tab[kk].append(vvv)
    if ri < args.n_full:
        pos = set(range(n))
    row_ids.append(ids); row_off.append(np.array([o for o in e.offsets], np.int32)); row_pos.append(sorted(pos))

idx_row = np.concatenate([np.full(len(p), ri, np.int32) for ri, p in enumerate(row_pos)])
idx_pos = np.concatenate([np.array(p, np.int32) for p in row_pos])
NP = len(idx_pos)
gb = len(TAPS) * NP * H * 2 / 1e9
print(f"[collect] stored positions {NP} -> {gb:.2f} GB per regime, {3*gb:.1f} GB total", flush=True)
np.savez(args.out + "_index.npz", idx_row=idx_row, idx_pos=idx_pos, pool_idx=pool_idx, n_full=args.n_full,
         taps=np.array(TAPS), regimes=np.array(REGIMES),
         row_ids=np.array(row_ids, dtype=object), row_off=np.array(row_off, dtype=object),
         texts=np.array([texts[int(p)] for p in pool_idx], dtype=object),
         **{f"num_{k}": np.array(v, dtype=object if k.endswith("text") else np.int32) for k, v in num_tab.items()})

# ---------------------------------------------------------------- the trunk pass
h = load_host(16)
mm = {r: np.lib.format.open_memmap(f"{args.out}_{r}.npy", mode="w+", dtype=np.float16,
                                   shape=(len(TAPS), NP, H)) for r in REGIMES}
starts = np.concatenate([[0], np.cumsum([len(p) for p in row_pos])])
t0 = time.time()
for b0 in range(0, len(row_ids), B):
    sl = range(b0, min(b0 + B, len(row_ids)))
    ids = np.zeros((B, T), np.int32); msk = np.zeros((B, T), np.float32)
    for j, ri in enumerate(sl):
        ids[j, :len(row_ids[ri])] = row_ids[ri]; msk[j, :len(row_ids[ri])] = 1
    for regime in REGIMES:
        out = taps(h, ids, regime, msk)
        for ti, a in enumerate(out):
            assert np.isfinite(a[:len(sl)]).all(), f"non-finite {regime} {TAPS[ti]}"
            for j, ri in enumerate(sl):
                mm[regime][ti, starts[ri]:starts[ri + 1]] = a[j, row_pos[ri]].astype(np.float16)
    if (b0 // B) % 20 == 0:
        print(f"[collect] batch {b0//B}/{(len(row_ids)+B-1)//B}  {time.time()-t0:.0f}s", flush=True)
for r in REGIMES:
    mm[r].flush()
print(f"[collect] DONE {len(row_ids)} rows x 3 regimes x {len(TAPS)} taps in {time.time()-t0:.0f}s -> {args.out}_*.npy",
      flush=True)
