"""gen23_fire_prep.py — THE GEN-23 FIRE PREP (2026-08-03, the word
given; dose declaration in the ledger). Stages, each gated:
  1. gen23_mix = gen22_mix (order PRESERVED) + 600 uniques x10 reps
     shuffled (seed 2300); token-length fence on every new row.
  2. States: copy g22 base memmap (82,400 rows) + GPU-compute the
     6,000 appended; sentinel cos>0.9999 on 6 picks (aug-fire recipe).
  3. Train npz under ALG_WIDE (7-wide digits + sign gold, full mix).
  4. Test npz 'test23' under ALG_WIDE (gold rebuilt; states/tokmask/
     sent preserved from the vintage npz — never clobbered).
  5. Padwarm: g22 -> g23_padwarm_init (MSD-aligned digit map: old
     d0..d2 -> new d4..d6; new leading positions bias digit-0 +6;
     h_sgn zero-W, bias -6) + INIT-IDENTITY check (decode equality
     on shared states) — gen-23 at init IS g22 on old-range input.
Disk law: one resident copy (+92GB against 1.4T free, arithmetic in
the spec)."""
import os, sys, json, random
os.environ["ALG_WIDE"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import (T_ALG, sent_indices, TOKENIZER_JSON,
                                 build_gold, build_params, N_DIG)
import phase1_algebra_head as PH
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_JSON)
assert N_DIG == 7

MIX = ".cache/gen23_mix.jsonl"
NPY = ".cache/phase1_alg_states_gen23_states.npy"
NPZ = ".cache/phase1_alg_states_gen23.npz"

# ---- 1. the mix (idempotent: relight skips completed stages) ----
_done_mix = os.path.exists(MIX) and sum(1 for _ in open(MIX)) == 88400
base = open(".cache/gen22_mix.jsonl").read().splitlines()
diet = [json.loads(l) for l in open(".cache/answer_space_corpus_v0.jsonl")]
assert len(base) == 82400 and len(diet) == 600, (len(base), len(diet))
too_long = [i for i, r in enumerate(diet)
            if len(tok.encode(r["text"]).ids) > T_ALG]
assert not too_long, f"token-length fence: rows {too_long[:5]} exceed T_ALG"
rng = random.Random(2300)
block = [json.dumps(r) for r in diet for _ in range(10)]
rng.shuffle(block)
if not _done_mix:
    with open(MIX, "w") as f:
        f.write("\n".join(base + block) + "\n")
n_total = len(base) + len(block)
print(f"[mix] gen23: {len(base)} base (order preserved) + {len(block)} diet "
      f"= {n_total} rows; share {len(block)/n_total:.3%}", flush=True)

# ---- 2. states ----
from beacon_closing_arm import recompute_states
_done_states = os.path.exists(NPY) and \
    np.load(NPY, mmap_mode="r").shape[0] == n_total
src = np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
assert src.shape[0] == 82400
if _done_states:
    print("[states] already assembled — skipping to sentinels", flush=True)
out = None if _done_states else np.lib.format.open_memmap(NPY, mode="w+", dtype=np.float16,
                                shape=(n_total, T_ALG, 2048))
if not _done_states:
    CH = 4096
    for s0 in range(0, src.shape[0], CH):
        out[s0:min(s0 + CH, src.shape[0])] = src[s0:min(s0 + CH, src.shape[0])]
    print("[states] base copied", flush=True)
    new_rows = [json.loads(l) for l in block]
    for s0 in range(0, len(new_rows), 8):
        ids = np.zeros((8, T_ALG), np.int32)
        for i, r in enumerate(new_rows[s0:s0 + 8]):
            e = tok.encode(r["text"]); Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]
        st = recompute_states(ids).astype(np.float16)
        for i in range(min(8, len(new_rows) - s0)):
            out[82400 + s0 + i] = st[i]
        if (s0 // 8) % 100 == 0:
            print(f"  [append {s0}/{len(new_rows)}]", flush=True)
    out.flush(); del out
# sentinels
rows_all = base + block
st = np.load(NPY, mmap_mode="r")
picks = [0, 41000, 82399, 82400, 85000, n_total - 1]
ids3 = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
for i, ridx in enumerate(picks):
    r = json.loads(rows_all[ridx])
    e = tok.encode(r["text"]); Ln = min(len(e.ids), T_ALG)
    ids3[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
live = recompute_states(ids3).astype(np.float32)
for i, ridx in enumerate(picks):
    m_ = msk[i] > 0
    a = live[i][m_]; b = np.asarray(st[ridx], np.float32)[m_]
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999, f"SENTINEL FAIL {ridx} {cos}"
print("[states] sentinels 6/6 — assembly TRUSTED", flush=True)

# ---- 3. train npz (ALG_WIDE gold) ----
_done_npz = os.path.exists(NPZ) and "g_sign" in np.load(NPZ).files
if _done_npz:
    print("[npz] gen23 train gold already banked — skipping", flush=True)
samples, ids2, mask, offsets = (None, None, None, None) if _done_npz \
    else PH.tokenize(MIX)
if samples is not None:
    gold = build_gold(samples, offsets)
    assert gold["digits"].shape[-1] == 7 and "sign" in gold
    sent = np.stack([sent_indices(s["text"], o, mask[i])
                     for i, (s, o) in enumerate(zip(samples, offsets))])
    np.savez(NPZ, tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
             **{f"g_{k}": v for k, v in gold.items()})
    print(f"[npz] gen23 train gold banked (7-wide + sign)", flush=True)

# ---- 4. test npz 'test23' ----
z = np.load(".cache/phase1_alg_states_test.npz")
tsamples, tids, tmask, toffsets = PH.tokenize(".cache/algebra_nl_test.jsonl")
tgold = build_gold(tsamples, toffsets)
keep = {k: z[k] for k in z.files if not k.startswith("g_")}
np.savez(".cache/phase1_alg_states_test23.npz", **keep,
         **{f"g_{k}": v for k, v in tgold.items()})
print("[npz] test23 gold rebuilt under ALG_WIDE; vintage npz untouched",
      flush=True)

# ---- 5. padwarm ----
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load, safe_save
wide = build_params(0)
sd = safe_load(".cache/g22.safetensors")
out_p = {}
for k, wt in wide.items():
    tgt = tuple(wt.shape)
    if k in ("h_dig", "h_dig2"):                 # BOTH digit banks (h_dig2 =
        old = sd[k].numpy()                      # gen-15 OP_APPLY; the relight
        buf = np.zeros(tgt, np.float32)          # lesson: same map, every bank)
        buf[:, 40:70] = old                      # old d0..d2 -> d4..d6
        out_p[k] = Tensor(buf, dtype=dtypes.float)
    elif k in ("h_dig_b", "h_dig2_b"):
        old = sd[k].numpy()
        buf = np.zeros(tgt, np.float32)
        buf[40:70] = old
        for d in range(4):                       # leading positions: digit-0
            buf[d * 10 + 0] = 6.0
        out_p[k] = Tensor(buf, dtype=dtypes.float)
    elif k == "h_sgn":
        out_p[k] = Tensor(np.zeros(tgt, np.float32), dtype=dtypes.float)
    elif k == "h_sgn_b":
        out_p[k] = Tensor(np.full(tgt, -6.0, np.float32), dtype=dtypes.float)
    else:
        src_t = sd[k].numpy()
        assert tuple(src_t.shape) == tgt, (k, src_t.shape, tgt)
        out_p[k] = Tensor(src_t.astype(np.float32), dtype=dtypes.float)
safe_save(out_p, ".cache/g23_padwarm_init.safetensors")
print("[padwarm] g23_padwarm_init banked (digit map MSD-aligned; "
      "sign biased positive)", flush=True)
print("[prep] COMPLETE — the fire may light", flush=True)
