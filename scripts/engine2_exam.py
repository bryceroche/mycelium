"""engine2_exam.py — THE EXAM (authorized 2026-07-29): one exam, three
arms, decided by the manifold criterion. Measurement, not fire — no
training, eager math only.

ARM A — BINARY PAWL on token-native (the measured amendment's test):
  per station-pass, a seed-fixed random 50% of live positions FULL-STEP
  (two local iterations of the station's layer), the rest HARD-FREEZE.
  Manifold read with PER-POSITION-APPROPRIATE references: an advanced
  position compares against the straight-pass image at its OWN
  advancement count (counts >4 hold at the L3 image — stated
  approximation); frozen positions are identity (trivially on their own
  last state) and are reported separately, not averaged in.
ARM B — LATENT BANK (option 2, init-lawful form): the state is a
  32-position strided subsample of the token stream (aligned-by-
  construction — carved from the pretrained embedding, no random
  latents); stations recirculate the short sequence; manifold read vs
  the straight-pass images of the same subsampled sequence.
ARM C — RESIDUAL SHARED (option 3): stations write an additive residual
  r over the anchored embedding — state x = x0 + r, r accumulates
  station deltas with Seam-1 on r; manifold read of x vs the straight
  L-images (the anchor keeps x0's contribution persistent).

Shared: 4 stations = Llama-3.2-1B L0..L3, causal mask, B=8 bigtest
texts, R=4 rounds, LOCAL=2, band 0.885. Baseline to beat (2026-07-29
cold read): option-1 ungated = OUT everywhere by round 1; zero pawls.
"""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from mycelium.llama_loader import attach_llama_layers, LLAMA_3_2_1B_CFG
from tokenizers import Tokenizer

ARM = os.environ.get("E2_ARM", "A")
B, T, R, LOCAL, BAND, SEED = 8, 96, 4, 2, 0.885, 7

class M: pass
model = M()
sd = safe_load(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(model, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")][:B]
ids = np.zeros((B, T), np.int32); msk = np.zeros((B, T), np.float32)
for i, r in enumerate(rows):
    e = tok.encode(r["text"]); L = min(len(e.ids), T)
    ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0

def prep(T_):
    m = np.triu(np.full((T_, T_), -1e9, np.float32), k=1)
    return (Tensor(m, dtype=dtypes.float).reshape(1, 1, T_, T_).realize(),
            model.llama_rope_cos[:T_].realize(), model.llama_rope_sin[:T_].realize())

def npv(t): return t.detach().cast(dtypes.float).realize().numpy()

def seam1(x):
    s = (x.pow(2).mean(axis=(1, 2), keepdim=True) + 1e-6).sqrt().detach()
    return (x / s).realize()

def cos_np(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return num / den

rng = np.random.RandomState(SEED)
print(f"=== ARM {ARM} ===")

if ARM == "A":
    am, rc, rs = prep(T)
    x0 = model.llama_embed[Tensor(ids, dtype=dtypes.int)].realize()
    # straight-pass images 1..4 (count 0 = x0)
    refs = [npv(x0)]
    h = x0
    for k in range(4):
        h = model.llama_layers[k](h, rc, rs, attn_mask=am).realize()
        refs.append(npv(h))
    x = x0; counts = np.zeros((B, T), np.int32)
    for rnd in range(1, R + 1):
        for k in range(4):
            h = x
            for _ in range(LOCAL):
                h = model.llama_layers[k](h, rc, rs, attn_mask=am).realize()
            gate = (rng.rand(B, T) < 0.5) & (msk > 0)          # binary pawl, 50% duty
            gt = Tensor(gate.astype(np.float32)[:, :, None], dtype=dtypes.float)
            x = (x * (1 - gt) + h * gt).realize()
            counts[gate] += 1
            # pawl-set counts per station per pass (bench 2026-07-29): with a
            # binary pawl, engagement is a DISCRETE EVENT with a timestamp —
            # the meter reports set-positions exactly, mechanically checkable.
            print(f"    [pawl] round {rnd} station {k}: advanced {int(gate.sum())} / frozen {int(((~gate)&(msk>0)).sum())}")
        x = seam1(x)
    xn = npv(x)
    ref_per_pos = np.zeros_like(xn)
    for b in range(B):
        for t in range(T):
            ref_per_pos[b, t] = refs[min(int(counts[b, t]), 4)][b, t]
    c = cos_np(xn, ref_per_pos)
    adv = (counts > 0) & (msk > 0); frz = (counts == 0) & (msk > 0)
    adv_cos = float(c[adv].mean()) if adv.any() else float("nan")
    print(f"advanced positions ({adv.sum()}): cos vs own-count image = {adv_cos:.4f} "
          f"{'IN' if adv_cos >= BAND else 'OUT'} (band {BAND})")
    print(f"frozen positions ({frz.sum()}): identity by construction (reported separately)")
    print(f"advancement-count histogram: {np.bincount(counts[msk>0], minlength=17)[:17].tolist()}")
    verdict = adv_cos >= BAND
    out = {"arm": "A-binary-pawl", "advanced_cos": adv_cos, "verdict_in_band": bool(verdict),
           "note": "counts>4 held at L3 image (stated approximation); Seam-1 scale folded into cos (scale-invariant metric)"}

elif ARM == "B":
    stride = max(1, T // 32)
    sel = np.arange(0, T, stride)[:32]
    Ts = len(sel)
    am, rc, rs = prep(Ts)
    ids_s = ids[:, sel]; msk_s = msk[:, sel]
    x0 = model.llama_embed[Tensor(ids_s, dtype=dtypes.int)].realize()
    refs = []
    h = x0
    for k in range(4):
        h = model.llama_layers[k](h, rc, rs, attn_mask=am).realize()
        refs.append(h)
    x = x0
    mani_final = {}
    for rnd in range(1, R + 1):
        for k in range(4):
            for _ in range(LOCAL):
                x = model.llama_layers[k](x, rc, rs, attn_mask=am).realize()
            cm = cos_np(npv(x), npv(refs[k]))
            mani_final[k] = float(cm[msk_s > 0].mean())
        x = seam1(x)
    print("final cos per station: " + " ".join(f"L{k}:{v:.3f}" for k, v in mani_final.items()))
    verdict = all(v >= BAND for v in mani_final.values())
    print(f"-> {'ALL IN BAND' if verdict else 'OUT OF BAND'}")
    out = {"arm": "B-latent-bank-32", "manifold_final": {str(k): v for k, v in mani_final.items()},
           "verdict_in_band": bool(verdict)}

else:  # ARM C
    am, rc, rs = prep(T)
    x0 = model.llama_embed[Tensor(ids, dtype=dtypes.int)].realize()
    refs = []
    h = x0
    for k in range(4):
        h = model.llama_layers[k](h, rc, rs, attn_mask=am).realize()
        refs.append(h)
    r_state = (x0 * 0).realize()
    mani_final = {}
    for rnd in range(1, R + 1):
        for k in range(4):
            xin = (x0 + r_state).realize()
            h = xin
            for _ in range(LOCAL):
                h = model.llama_layers[k](h, rc, rs, attn_mask=am).realize()
            r_state = (r_state + (h - xin)).realize()
            cm = cos_np(npv((x0 + r_state).realize()), npv(refs[k]))
            mani_final[k] = float(cm[msk > 0].mean())
        r_state = seam1(r_state)
    print("final cos per station (x0+r vs L-image): " + " ".join(f"L{k}:{v:.3f}" for k, v in mani_final.items()))
    verdict = all(v >= BAND for v in mani_final.values())
    print(f"-> {'ALL IN BAND' if verdict else 'OUT OF BAND'}")
    out = {"arm": "C-residual-anchored", "manifold_final": {str(k): v for k, v in mani_final.items()},
           "verdict_in_band": bool(verdict)}

json.dump(out, open(f".cache/engine2_exam_arm{ARM}.json", "w"), indent=1)
print(f"[done] wrote engine2_exam_arm{ARM}.json")
