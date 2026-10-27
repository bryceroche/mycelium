"""distill_reader.py -- (B) THE DISTILLED READER (ledger 2026-10-26 19:27, word given). The vanilla reading half of the
system map: a small BIDIRECTIONAL transformer trained to reproduce the frozen Llama-3.2-1B L0-L3 BP states (bidirectional,
pad keys masked, final RMSNorm) from the banked teacher memmap -- no Llama forward at train time.
  embed   the FROZEN Llama embedding table (a lookup, not a transformer) -> learned 2048 -> D projection
  body    N pre-norm blocks: RMSNorm, MHA (RoPE, theta 500000, pad KEYS masked), RMSNorm, SwiGLU
  out     RMSNorm -> D -> 2048 (+ bias initialized to the teacher's per-dim mean)
  loss    on REAL tokens only: MSE / teacher variance + (1 - cos)
Modes:
  train : DR_STEPS, DR_BATCH, DR_LR, DR_D, DR_LAYERS, DR_TAG -> .cache/distill_reader_{TAG}.safetensors (+ .json, .log lines)
  --export JSONL --src SPLIT --out SPLIT2 : run the student over JSONL and write the states memmap under SPLIT2 + a copy of
          SPLIT's npz (tokmask/sent/gold are state-independent; tokmask asserted equal), so ALG_TEST_NAME=SPLIT2 reads it.
          Pad positions are written as ZEROS (the student is never trained on them).
          DR_VOCAB=fold: EMB @ w_in folded into one fp16 V x D table; DR_VOCAB=prune: + reader_vocab.py's kept set, dropped
          tokens = the mean of their fallback pieces' rows (ledger 10-27, the shrunk reader)
  --padzero SRC --out SPLIT2 : the teacher's own states with pads zeroed (the feasibility control: do pads matter?)"""
import argparse, json, math, os, queue, shutil, sys, threading, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "9"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_WIDE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--export", default=""); ap.add_argument("--src", default=""); ap.add_argument("--out", default="")
ap.add_argument("--padzero", default="")
args = ap.parse_args()
assert os.environ.get("DEV") in ("PCI+AMD", "CPU"), "DEV=PCI+AMD (never bare AMD) or CPU"
import numpy as np
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load
from phase1_algebra_head import tokenize, T_ALG, H_TRUNK, STATES_NPZ, STATES_NPY
from mycelium.llama_loader import load_llama_weights, _build_rope_freqs, _apply_rope, _rms_norm

TAG = os.environ.get("DR_TAG", "s6x512")
D, NL = int(os.environ.get("DR_D", 512)), int(os.environ.get("DR_LAYERS", 6))
NH, HD = D // 64, 64
FF = int(os.environ.get("DR_FF", 1408))
STEPS, BATCH, LR = int(os.environ.get("DR_STEPS", 30000)), int(os.environ.get("DR_BATCH", 32)), float(os.environ.get("DR_LR", 3e-4))
TRAIN_JSONL, TRAIN_SPLIT = ".cache/form_mix_pm35c.jsonl", "formpm35cBP"
CKPT, META = f".cache/distill_reader_{TAG}.safetensors", f".cache/distill_reader_{TAG}.json"
EPS = 1e-5
PROF = bool(int(os.environ.get("DR_PROF", "0")))


def padzero(src, out):
    s_npy, d_npy = STATES_NPY.format(split=src), STATES_NPY.format(split=out)
    assert not os.path.exists(d_npy), f"{d_npy} exists -- refusing to overwrite"
    a = np.load(s_npy, mmap_mode="r"); tm = np.load(STATES_NPZ.format(split=src))["tokmask"]
    shutil.copyfile(STATES_NPZ.format(split=src), STATES_NPZ.format(split=out))
    st = np.lib.format.open_memmap(d_npy, mode="w+", dtype=np.float16, shape=a.shape)
    for s0 in range(0, len(a), 256):
        st[s0:s0 + 256] = np.asarray(a[s0:s0 + 256]) * tm[s0:s0 + 256, :, None].astype(np.float16)
    st.flush(); print(f"[padzero] {src} -> {out}: {len(a)} rows, pads zeroed", flush=True)


if args.padzero:
    assert args.out; padzero(args.padzero, args.out); sys.exit(0)

sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
EMB = sd["model.embed_tokens.weight"].to(Device.DEFAULT).cast(dtypes.half).realize().numpy()   # frozen lookup, gathered on the HOST
# (a device-side EMB[ids] lowers to a compare-and-reduce over the whole 128k-row table: ~2e12 ops per batch, 16 s/step)
del sd
COS, SIN = _build_rope_freqs(HD, T_ALG, 500000.0)


def init_params(mean_out):
    rng = np.random.default_rng(0)
    nrm = lambda *s: Tensor(rng.normal(0, 1 / math.sqrt(s[0]), s).astype(np.float32), dtype=dtypes.float).contiguous()
    p = {"w_in": nrm(H_TRUNK, D), "fnorm": Tensor.ones(D, dtype=dtypes.float).contiguous(),
         "w_out": nrm(D, H_TRUNK), "b_out": Tensor(mean_out.astype(np.float32), dtype=dtypes.float).contiguous()}
    for i in range(NL):
        p[f"an{i}"] = Tensor.ones(D, dtype=dtypes.float).contiguous(); p[f"fn{i}"] = Tensor.ones(D, dtype=dtypes.float).contiguous()
        for k in ("wq", "wk", "wv"):
            p[f"{k}{i}"] = nrm(D, D)
        p[f"wo{i}"] = Tensor.zeros(D, D, dtype=dtypes.float).contiguous()         # ReZero-style: blocks start as identity
        p[f"wg{i}"], p[f"wu{i}"] = nrm(D, FF), nrm(D, FF)
        p[f"wd{i}"] = Tensor.zeros(FF, D, dtype=dtypes.float).contiguous()
    return p


def forward(p, x0, am):
    B = x0.shape[0]
    x = x0.cast(dtypes.float) @ p["w_in"]
    for i in range(NL):
        h = _rms_norm(x, p[f"an{i}"], EPS)
        q = (h @ p[f"wq{i}"]).reshape(B, T_ALG, NH, HD).transpose(1, 2)
        k = (h @ p[f"wk{i}"]).reshape(B, T_ALG, NH, HD).transpose(1, 2)
        v = (h @ p[f"wv{i}"]).reshape(B, T_ALG, NH, HD).transpose(1, 2)
        q, k = _apply_rope(q, k, COS, SIN, T_ALG)
        sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(HD)) + am
        a = sc.clip(-1e4, 1e4).softmax(-1)
        x = x + (a @ v).transpose(1, 2).reshape(B, T_ALG, D) @ p[f"wo{i}"]
        h2 = _rms_norm(x, p[f"fn{i}"], EPS)
        x = x + ((h2 @ p[f"wg{i}"]).silu() * (h2 @ p[f"wu{i}"])) @ p[f"wd{i}"]
    return _rms_norm(x, p["fnorm"], EPS) @ p["w_out"] + p["b_out"]


def addmask(tm):
    return Tensor(np.where(tm[:, None, None, :] > 0, 0.0, -1e4).astype(np.float32), dtype=dtypes.float)


if args.export:
    assert args.src and args.out and os.path.exists(CKPT), "need --src, --out and a trained checkpoint"
    d_npy = STATES_NPY.format(split=args.out); assert not os.path.exists(d_npy), f"{d_npy} exists -- refusing to overwrite"
    samples, ids, mask, _ = tokenize(args.export)
    ztm = np.load(STATES_NPZ.format(split=args.src))["tokmask"]
    assert ztm.shape == mask.shape and (ztm.astype(np.float32) == mask).all(), "tokmask mismatch vs the source split"
    ld = safe_load(CKPT); p = {k: ld[k].to(Device.DEFAULT).cast(dtypes.float).contiguous().realize() for k in ld}
    VOCAB = os.environ.get("DR_VOCAB", "full")
    if VOCAB != "full":   # THE SHRUNK READER (ledger 10-27): fold EMB @ w_in into one fp16 V x D table (exact up to fp16 storage);
        # "prune" keeps reader_vocab.py's K and gives every dropped token the MEAN of its fallback pieces' rows
        EMB = (EMB.astype(np.float32) @ p["w_in"].numpy()).astype(np.float16)
        if VOCAB == "prune":
            z = np.load(".cache/reader_vocab_k.npz"); ptr, pc = z["piece_ptr"], z["pieces"]
            keep = np.zeros(len(EMB), bool); keep[z["keep"]] = True; assert keep[pc].all(), "a fallback piece was dropped"
            for j, i in enumerate(z["oov_ids"]):
                EMB[i] = EMB[pc[ptr[j]:ptr[j + 1]]].astype(np.float32).mean(0).astype(np.float16)
            print(f"[export] PRUNED: {int(keep.sum())} kept rows x {EMB.shape[1]}, {len(z['oov_ids'])} tokens composed from pieces", flush=True)
        else:
            print(f"[export] FOLDED: {EMB.shape[0]} x {EMB.shape[1]} fp16", flush=True)
        p["w_in"] = Tensor.eye(D, dtype=dtypes.float).contiguous().realize()
    shutil.copyfile(STATES_NPZ.format(split=args.src), STATES_NPZ.format(split=args.out))
    st = np.lib.format.open_memmap(d_npy, mode="w+", dtype=np.float16, shape=(len(ids), T_ALG, H_TRUNK))
    for s0 in range(0, len(ids), 8):
        sl = slice(s0, min(s0 + 8, len(ids)))
        o = forward(p, Tensor(EMB[ids[sl]], dtype=dtypes.half), addmask(mask[sl])).realize().numpy()
        assert np.isfinite(o).all()
        st[sl] = (o * mask[sl][:, :, None]).astype(np.float16)
    st.flush(); print(f"[export] {args.export} -> {args.out}: {len(ids)} rows (pads zero)", flush=True); sys.exit(0)

# ---- train ------------------------------------------------------------------------------------------------------
assert not os.path.exists(CKPT), f"{CKPT} exists -- refusing to overwrite"
samples, ids, mask, _ = tokenize(TRAIN_JSONL)
teach = np.load(STATES_NPY.format(split=TRAIN_SPLIT), mmap_mode="r")
assert teach.shape == (len(ids), T_ALG, H_TRUNK)
ztm = np.load(STATES_NPZ.format(split=TRAIN_SPLIT))["tokmask"]; assert (ztm.astype(np.float32) == mask).all(), "tokmask mismatch"
import hashlib
val = np.array([int(hashlib.sha1(s["text"].encode()).hexdigest(), 16) % 100 == 0 for s in samples])   # 1% by TEXT identity
tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0][:256]
smp = np.sort(np.random.default_rng(1).choice(tr_idx, 512, replace=False))
tsm = np.asarray(teach[smp]).astype(np.float32); msm = mask[smp] > 0
mean_out = tsm[msm].mean(0); var_t = float(tsm[msm].var(0).mean())
print(f"[distill] rows {len(ids)} train {len(tr_idx)} val {len(va_idx)} | teacher per-dim var {var_t:.4f} | D {D} layers {NL} FF {FF} "
      f"steps {STEPS} batch {BATCH} lr {LR}", flush=True)
p = init_params(mean_out); params = list(p.values())
for t in params:
    t.requires_grad = True
n_par = sum(int(np.prod(t.shape)) for t in params); print(f"[distill] trainable params {n_par:,} (+ frozen embedding)", flush=True)
opt = AdamW(params, lr=LR, weight_decay=0.01)
INV_VAR = 1.0 / var_t


def loss_fn(out, tgt, tm):
    # .contiguous() = schedule boundaries: without them tinygrad fuses the w_out matmul's reduction under the per-row
    # mean and recomputes per-row/per-batch reductions inside the backward's elementwise kernels (13.7 s/step -> 0.28)
    out = out.contiguous(); tgt = tgt.cast(dtypes.float); n = tm.sum().contiguous()
    mse = (((out - tgt) ** 2).mean(-1) * tm).sum() / n * INV_VAR
    cos = (out * tgt).sum(-1).contiguous() / ((out * out).sum(-1).sqrt().contiguous() * (tgt * tgt).sum(-1).sqrt().contiguous() + 1e-6)
    return mse + ((1 - cos) * tm).sum() / n, mse


@TinyJit
def train_step(x0, am, tm, tgt):
    with Tensor.train():
        out = forward(p, x0, am); loss, mse = loss_fn(out, tgt, tm)
        opt.zero_grad(); loss.backward()
        ok = loss.isfinite().cast(dtypes.float).contiguous()                # single-kernel guard (a boundary, or the loss is recomputed per grad element)
        for t in params:
            t.grad = t.grad * ok
        opt.step()
        return loss.realize(), mse.realize()


@TinyJit
def eval_step(x0, am, tm, tgt):
    out = forward(p, x0, am); loss, mse = loss_fn(out, tgt, tm)
    return loss.realize(), mse.realize()


def batch_of(idx):
    idx = np.sort(idx)
    return (Tensor(EMB[ids[idx]], dtype=dtypes.half), addmask(mask[idx]), Tensor(mask[idx], dtype=dtypes.float),
            Tensor(np.asarray(teach[idx]), dtype=dtypes.half))


Q = queue.Queue(maxsize=6)


def loader():
    rng = np.random.default_rng(2)
    while True:
        idx = rng.choice(tr_idx, BATCH, replace=False); idx = np.sort(idx)
        Q.put((EMB[ids[idx]], mask[idx], np.asarray(teach[idx])))


threading.Thread(target=loader, daemon=True).start()
VB = [va_idx[i:i + BATCH] for i in range(0, len(va_idx) - BATCH + 1, BATCH)]
t0 = time.time(); hist = []
for step in range(STEPS):
    ta = time.time(); bi, bm, bt = Q.get(); tb = time.time()
    xs = [Tensor(bi, dtype=dtypes.half), addmask(bm), Tensor(bm, dtype=dtypes.float), Tensor(bt, dtype=dtypes.half)]
    xs = [x.contiguous().realize() for x in xs]; tc = time.time()
    loss, mse = train_step(*xs)
    if PROF:
        loss.numpy(); print(f"[prof] step {step}: queue wait {tb - ta:.3f}s, inputs {tc - tb:.3f}s, train_step {time.time() - tc:.3f}s", flush=True)
    if step % 200 == 0 or step == STEPS - 1:
        lv = float(loss.numpy()); assert math.isfinite(lv), f"non-finite loss at step {step}"
        msg = f"[distill] step {step} (queue {Q.qsize()}) loss {lv:.4f} mse {float(mse.numpy()):.4f} | {(time.time() - t0) / (step + 1):.3f}s/step"
        if step % 2000 == 0 or step == STEPS - 1:
            vl = [eval_step(*batch_of(b)) for b in VB]
            vloss = float(np.mean([float(a.numpy()) for a, _ in vl])); vmse = float(np.mean([float(b.numpy()) for _, b in vl]))
            msg += f" | VAL loss {vloss:.4f} mse {vmse:.4f}"; hist.append(dict(step=step, val_loss=vloss, val_mse=vmse))
        print(msg, flush=True)
safe_save({k: v.realize() for k, v in p.items()}, CKPT)
json.dump(dict(tag=TAG, D=D, layers=NL, ff=FF, steps=STEPS, batch=BATCH, lr=LR, params=n_par, teacher=TRAIN_SPLIT,
               var_t=var_t, hist=hist, seconds=time.time() - t0), open(META, "w"), indent=1)
print(f"[distill] saved {CKPT} in {time.time() - t0:.0f}s", flush=True)
