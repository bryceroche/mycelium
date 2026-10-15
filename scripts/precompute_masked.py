"""precompute_masked.py -- the trunk precompute under a chosen ATTENTION REGIME (2026-10-14; the pad-leak arm,
ledger 10-14 ~16:00 / ~16:30 / this entry). do_precompute (phase1_algebra_head.py:3064) runs L0-L3 with NO mask:
bidirectional, every real token attending to the ~190 id-0 ('!') pads of the 256 frame. This writes the SAME arrays
(same tokenize(), same 4 layers, same final RMSNorm, same fp16 memmap layout) under:
  deployed : no mask (must reproduce the banked states BIT-IDENTICALLY -- the --verify gate)
  bidirpad : bidirectional, pad KEYS masked (-1e4 additive; exp underflows to exact 0)
  native   : causal + pad keys masked
Output: .cache/phase1_alg_states_{OUT}_states.npy + a COPY of the source split's npz (gold/tokmask/sent are
mask-independent), so ALG_TRAIN_NAME/ALG_TEST_NAME={OUT} reads the new states with every other array unchanged.
Usage: DEV=PCI+AMD ... --split formpm35c --jsonl .cache/form_mix_pm35c.jsonl --mask bidirpad --out formpm35cBP
       --verify N : deployed mode only, compare the first N rows to the banked memmap and exit.
"""
import os, sys, shutil, time, argparse
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True); ap.add_argument("--jsonl", required=True)
ap.add_argument("--mask", required=True, choices=["deployed", "bidirpad", "native"])
ap.add_argument("--out", default=""); ap.add_argument("--verify", type=int, default=0)
args = ap.parse_args()
assert os.environ.get("DEV") in ("PCI+AMD", "CPU"), "DEV=PCI+AMD (never bare AMD) or CPU"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "9"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_WIDE", "1")
import numpy as np
from tinygrad import Tensor, dtypes
from phase1_algebra_head import tokenize, T_ALG, H_TRUNK, STATES_NPZ, STATES_NPY
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm

samples, ids, mask, offsets = tokenize(args.jsonl)
n = len(samples)
src_npy = STATES_NPY.format(split=args.split); src_npz = STATES_NPZ.format(split=args.split)
banked = np.load(src_npy, mmap_mode="r")
assert banked.shape == (n, T_ALG, H_TRUNK), f"banked {banked.shape} vs tokenized {n}"
ztm = np.load(src_npz)["tokmask"]
assert (ztm.astype(np.float32) == mask.astype(np.float32)).all(), "tokmask mismatch vs the banked npz"


class _H: pass
host = _H()
sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
del sd


def run(sl):
    x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
    am = None
    if args.mask != "deployed":
        m = mask[sl].astype(np.float32)
        a = np.where(m[:, None, None, :] > 0, 0.0, -1e4).astype(np.float32)
        a = np.broadcast_to(a, (len(m), 1, T_ALG, T_ALG)).copy()
        if args.mask == "native":
            a = a + np.triu(np.full((T_ALG, T_ALG), -1e4, np.float32), 1)[None, None]
        am = Tensor(a)
    for layer in host.llama_layers:
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin, attn_mask=am)
    x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
    c = x.cast(dtypes.float).realize().numpy()
    assert np.isfinite(c).all()
    return c.astype(np.float16)


if args.verify:
    assert args.mask == "deployed", "--verify reproduces the banked (deployed) states"
    worst = 0.0; neq = 0
    for s0 in range(0, args.verify, 8):
        sl = slice(s0, min(s0 + 8, args.verify))
        got = run(sl); ref = np.asarray(banked[sl])
        neq += int((got != ref).sum()); worst = max(worst, float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max()))
    print(f"[verify] {args.split}: first {args.verify} rows: elements differing {neq}, max abs diff {worst:.3e} "
          f"-> {'BIT-IDENTICAL' if neq == 0 else 'NOT bit-identical'}", flush=True)
    sys.exit(0)

assert args.out and args.out != args.split
dst_npy = STATES_NPY.format(split=args.out); dst_npz = STATES_NPZ.format(split=args.out)
assert not os.path.exists(dst_npy), f"{dst_npy} exists -- refusing to overwrite"
shutil.copyfile(src_npz, dst_npz)
st = np.lib.format.open_memmap(dst_npy, mode="w+", dtype=np.float16, shape=(n, T_ALG, H_TRUNK))
t0 = time.time()
for s0 in range(0, n, 8):
    sl = slice(s0, min(s0 + 8, n))
    st[sl] = run(sl)
    if (s0 // 8) % 500 == 0:
        print(f"[precompute-{args.mask}] {args.out}: batch {s0//8}/{(n+7)//8} {time.time()-t0:.0f}s", flush=True)
st.flush()
# the pad-leak check on the product: real-token states of the first 64 rows vs the banked deployed states
r = slice(0, min(64, n)); a = np.asarray(st[r]).astype(np.float32); b = np.asarray(banked[r]).astype(np.float32)
mm = mask[r] > 0
cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
print(f"[precompute-{args.mask}] DONE {args.out}: {n} rows in {time.time()-t0:.0f}s; real-token cos vs banked deployed "
      f"mean {cos[mm].mean():.4f} (min {cos[mm].min():.4f})", flush=True)
