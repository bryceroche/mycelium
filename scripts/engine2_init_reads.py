"""engine2_init_reads.py — BUILD-TO-MEASUREMENT (2026-07-29, the word:
build yes, fire no). Engine 2 wired to init and measured cold; #72
prerequisite (a): the two staged reads (clock, manifold) + engagement +
thermometer, on the built stations. NO TRAINING. Eager math only (no
TinyJit — the quirks file's constraints don't bind eager reads).

Wiring per spec §2: 4 untied stations ← Llama-3.2-1B L0..L3 (distinct
pretrained layers — v200's actual trunk, so phase-A's 2.28σ ramp and the
Fire-2 manifold band both apply to THESE layers); token-native substrate
(dialect text through the pretrained embed); station-local loops L=2;
R=4 recirculation rounds; Seam-1 per-item scalar RMS between rounds
(the proven contraction). Causal mask = pretrained-native attention.

Reads:
  1. MANIFOLD RESIDENCY (condition 2): cos(station k's state, straight-
     pass layer-k image), per round. Fire-2 band: >=0.885 IN.
  2. THERMOMETER: per-station-pass ||dx||/||x|| (per-item scalar).
  3. ENGAGEMENT: per-position settle round (last round with relative
     delta > 1e-3) — the pawl-gate's future g-trace, read passively.
  4. PROPAGATION CLOCK (condition 1, init-measurable form): rounds to
     settle (t95 of state-motion decay). SCOPE STATED HONESTLY: this is
     the PROPAGATION clock — the deduction clock completes at first
     training smoke; the capture banked here is its substrate.
  5. CAPTURE: states per station-pass -> fp16 npz for the future
     deduction-clock probe.
"""
import sys, os, json, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from mycelium.llama_loader import attach_llama_layers, LLAMA_3_2_1B_CFG
from tokenizers import Tokenizer

TOKENIZER_JSON = ".cache/llama-3.2-1b-weights/tokenizer.json"
B, T, R, LOCAL = 8, 96, 4, 2
ALPHA = float(os.environ.get("E2_ALPHA", "1.0"))   # pawl gate, degenerate init form (uniform); driver activates at fire
BAND = 0.885   # Fire-2's pinned manifold band

class M: pass
model = M()
sd = safe_load(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(model, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
tok = Tokenizer.from_file(TOKENIZER_JSON)

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")][:B]
ids = np.zeros((B, T), np.int32); msk = np.zeros((B, T), np.float32)
for i, r in enumerate(rows):
    e = tok.encode(r["text"]); L = min(len(e.ids), T)
    ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0

x0 = model.llama_embed[Tensor(ids, dtype=dtypes.int)].realize()   # (B,T,H)
mask_np = np.triu(np.full((T, T), -1e9, np.float32), k=1)
attn_mask = Tensor(mask_np, dtype=dtypes.float).reshape(1, 1, T, T).realize()
rc = model.llama_rope_cos[:T].realize(); rs = model.llama_rope_sin[:T].realize()
msk_t = Tensor(msk[:, :, None], dtype=dtypes.float).realize()

def npv(t): return t.detach().cast(dtypes.float).realize().numpy()

def masked_cos(a, b):
    num = (a * b).sum(-1); den = (a.pow(2).sum(-1).sqrt() * b.pow(2).sum(-1).sqrt() + 1e-8)
    c = (num / den) * Tensor(msk, dtype=dtypes.float)
    return float(npv(c.sum())) / float(msk.sum())

def seam1_scalar(x):
    s = (x.pow(2).mean(axis=(1, 2), keepdim=True) + 1e-6).sqrt().detach()
    return (x / s).realize()

# ---- reference: straight pretrained pass, layer images banked
ref = []
h = x0
for k in range(4):
    h = model.llama_layers[k](h, rc, rs, attn_mask=attn_mask).realize()
    ref.append(h)
print("[ref] straight-pass layer images banked (L0..L3)")

# ---- the engine at init
x = x0
prev_np = npv(x)
capture = []
therm_log, mani_log = [], []
settle_round = np.zeros((B, T), np.int32)
for rnd in range(1, R + 1):
    for k in range(4):
        h = x
        for li in range(LOCAL):
            h = model.llama_layers[k](h, rc, rs, attn_mask=attn_mask).realize()
        x = (x + (h - x) * ALPHA).realize() if ALPHA < 1.0 else h
        cur = npv(x)
        d = np.linalg.norm((cur - prev_np) * msk[:, :, None], axis=-1)
        nrm = np.linalg.norm(cur * msk[:, :, None], axis=-1) + 1e-8
        rel = (d / nrm)
        therm = float((rel * msk).sum() / msk.sum())
        mani = masked_cos(x, ref[k])
        therm_log.append((rnd, k, therm)); mani_log.append((rnd, k, mani))
        moved = (rel > 1e-3) & (msk > 0)
        settle_round[moved] = (rnd - 1) * 4 + k + 1
        capture.append(cur.astype(np.float16))
        prev_np = cur
        print(f"  [round {rnd} station {k}] thermometer {therm:.4f}  manifold-cos vs L{k}-image {mani:.4f} {'IN' if mani>=BAND else 'OUT'}")
    x = seam1_scalar(x)
    prev_np = npv(x)

# ---- propagation clock: t95 of state-motion decay (station-pass grain)
t_series = [t for (_, _, t) in therm_log]
final = t_series[-1]; first = t_series[0]
target = final + 0.05 * (first - final)
t95 = next((i + 1 for i, v in enumerate(t_series) if v <= target), len(t_series))
print(f"\n[clock/propagation] station-passes to settle (t95 of motion decay): {t95} of {len(t_series)}"
      f"  (first {first:.4f} -> final {final:.4f})")
print(f"[engagement] settle-pass histogram (per-position): "
      f"{np.bincount(settle_round[msk>0].ravel(), minlength=17)[:17].tolist()}")
mani_final = {k: [m for (r_, k_, m) in mani_log if k_ == k][-1] for k in range(4)}
verdict = all(m >= BAND for m in mani_final.values())
print(f"[manifold] final cos per station: " +
      " ".join(f"L{k}:{v:.3f}" for k, v in mani_final.items()) +
      f"  -> {'ALL IN BAND (>=0.885)' if verdict else 'OUT OF BAND'}")

np.savez_compressed(f".cache/engine2_init_capture_a{ALPHA}.npz",
                    states=np.stack(capture), ids=ids, mask=msk,
                    therm=np.array(therm_log), mani=np.array(mani_log),
                    settle=settle_round)
json.dump({"t95_station_passes": int(t95), "n_passes": len(t_series),
           "therm_first": first, "therm_final": final,
           "manifold_final": {str(k): v for k, v in mani_final.items()},
           "manifold_band": BAND, "manifold_verdict": bool(verdict),
           "scope": "PROPAGATION clock at init; deduction clock completes at first training smoke (capture banked)"},
          open(f".cache/engine2_init_reads_a{ALPHA}.json", "w"), indent=1)
print("[done] wrote engine2_init_capture.npz + engine2_init_reads.json")
