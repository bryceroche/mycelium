"""scripts/unet/train.py — THE U-NET PICTURE BAKE-OFF, the training loop.
A NEW script; no existing script touched. TinyJit'd step over FIXED
buffers (memory/reference_tinygrad_am_quirks.md's rules: no
`.cast(dtypes.float32)` literal inside JIT -- use `dtypes.float`; clip
logits before any log/softmax [done in model.py's `_unet_body`]; a
single-kernel isfinite NaN guard; every `.assign()`-targeted buffer
allocated once outside the JIT and fed every step).

Trains ONLY on the diet (UN_SPLIT/UN_JSONL, default formpm35c /
.cache/form_mix_pm35c.jsonl) -- wild and MATH-500 are MEASURED, never
trained on (CLAUDE.md); eval_wild.py is the only thing that ever touches
wild, and it never backprops.

ENV
  UN_PICTURE   Aprime | Adouble | Adouble_shuf   (default Adouble)
  UN_STEPS     training steps                    (default 20000)
  UN_BATCH     rows per step                      (default 4)
  UN_T         picture side T (<= cached T_full=256)   (default 256)
  UN_LAYERS    trunk layer selector -- only "default" is implemented;
               anything else raises (see pictures.PictureSource)
  UN_CKPT      checkpoint path (default .cache/unet_<UN_PICTURE>.safetensors)
  UN_LR        AdamW lr                            (default 3e-4)
  UN_BASE      U-Net base channel width             (default 16)
  UN_BREATHS   breath count K_B                     (default 7)
  UN_SEED      row-sampler + model-init seed        (default 0)
  UN_SPLIT     states split name                    (default formpm35c)
  UN_JSONL     the split's jsonl                     (default .cache/form_mix_pm35c.jsonl)
  UN_CLASS_W   4 comma-separated class weights (none,given,arg,result_op)
               (default "1,50,50,50" -- class 0 is ~99.9% of pixels,
               see this file's own census print at startup)
  UN_LOG_EVERY print every N steps                   (default 200)
  UN_SNAP_EVERY  save ckpt every N steps              (default 5000)
  UN_SMOKE=1   run the PINNED CPU smoke test instead of a real training
               run (ignores the knobs above except UN_PICTURE; see
               `smoke_test()`): 2 steps, batch 2, 8 diet rows, T=64.
"""
import json
import os
import sys
import time

import numpy as np
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_state_dict, safe_save, safe_load, load_state_dict

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/unet")

from pictures import PictureSource, fit_pca, build_picture, n_channels   # noqa: E402
from model import UNet   # noqa: E402
from targets import build_target, N_CLASSES, CLASS_NAMES   # noqa: E402


def class_census(src, tok, n_rows, T, seed=0):
    """Informational: the per-class pixel share over a sample of rows,
    printed at startup so UN_CLASS_W's defaults are legible (class 0 is
    ~99.9% of pixels at T=256 -- measured on 300 diet rows during this
    build: none 19,642,027 / given_value 5,097 / arg 8,606 /
    result_op 5,070 px)."""
    rng = np.random.RandomState(seed)
    rows = rng.choice(src.n, size=min(n_rows, src.n), replace=False)
    tot = np.zeros(N_CLASSES, np.int64)
    for i in rows.tolist():
        label, _ = build_target(src.text(i), tok, src.gold, i, T)
        for c in range(N_CLASSES):
            tot[c] += int((label == c).sum())
    return tot


def fixed_class_weights(spec):
    w = np.array([float(x) for x in spec.split(",")], np.float32)
    assert len(w) == N_CLASSES, f"UN_CLASS_W needs {N_CLASSES} values, got {spec!r}"
    return w


def build_batch(src, idxs, mode, mean, comps, tok, T, rng):
    B = len(idxs)
    Cin = n_channels(mode)
    pics = np.zeros((B, Cin, T, T), np.float32)
    labels = np.zeros((B, T, T), np.int32)
    reals = np.zeros((B, T), np.float32)
    for b, i in enumerate(idxs):
        state, tm, se = src.row(i, T=T)
        text = src.text(i)
        pics[b] = build_picture(state, tm, se, text, mean, comps, mode, rng=rng)
        label, real = build_target(text, tok, src.gold, i, T)
        labels[b] = label
        reals[b] = real.astype(np.float32)
    return pics, labels, reals


def make_step(model, opt, b_pic, b_label, b_real, class_w):
    K = model.K_B
    cw = Tensor(class_w.reshape(1, 1, N_CLASSES, 1, 1), dtype=dtypes.float)
    cls_ids = Tensor(np.arange(N_CLASSES, dtype=np.int32).reshape(1, 1, N_CLASSES, 1, 1),
                      dtype=dtypes.int)

    def step():
        Tensor.training = True
        B, T = b_label.shape[0], b_label.shape[1]
        logits = model.forward_breaths(b_pic)                       # (B,K,4,T,T)
        logp = logits.log_softmax(axis=2)
        label_exp = b_label.reshape(B, 1, T, T).expand(B, K, T, T)
        onehot = (label_exp.unsqueeze(2) == cls_ids).cast(dtypes.float)   # (B,K,4,T,T)
        w_pp = (onehot * cw).sum(axis=2)                              # (B,K,T,T)
        ce_pp = -(onehot * logp).sum(axis=2)                          # (B,K,T,T)
        pix = (b_real.reshape(B, 1, T, 1) * b_real.reshape(B, 1, 1, T)).expand(B, K, T, T)
        num = (ce_pp * w_pp * pix).sum()
        den = (w_pp * pix).sum() + 1e-6
        loss = num / den
        opt.zero_grad()
        loss.backward()
        # the no-grad fence (phase1_algebra_head.py's convention): name
        # starved params instead of AdamW's bare assert.
        nog = [k for k, t in get_state_dict(model).items() if t.requires_grad and t.grad is None]
        assert not nog, f"params with NO gradient: {nog}"
        # single-kernel isfinite NaN guard (quirks doc: multiply-gating a
        # NaN never zeroes it -- use where() on the loss itself, pre-step)
        healthy = loss.isfinite().cast(dtypes.float)
        opt.step()
        return loss.realize(), healthy.realize()
    return step


def run(steps, batch, T, un_picture, lr, base, K_B, seed, split, jsonl,
        class_w_spec, ckpt, log_every, snap_every, layers="default"):
    from tokenizers import Tokenizer
    import phase1_algebra_head as H   # FAM env must already be set (tree_row_ids/TOKENIZER_JSON)
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    src = PictureSource(split, jsonl, layer=layers)
    mean, comps = fit_pca()
    census = class_census(src, tok, min(300, src.n), T, seed=seed)
    print(f"[unet/train] class census ({min(300, src.n)} rows, T={T}): "
          + ", ".join(f"{CLASS_NAMES[c]}={census[c]}" for c in range(N_CLASSES)), flush=True)
    class_w = fixed_class_weights(class_w_spec)
    Cin = n_channels(un_picture)
    n_where = 0 if un_picture == "Aprime" else 3
    model = UNet(c_content=Cin - n_where, n_where=n_where, K_B=K_B, base=base, seed=seed)
    opt = AdamW(model.parameters(), lr=lr)

    def fix(shape, dt, npdt):
        return Tensor(np.zeros(shape, npdt), dtype=dt).contiguous().realize()
    b_pic = fix((batch, Cin, T, T), dtypes.float, np.float32)
    b_label = fix((batch, T, T), dtypes.int, np.int32)
    b_real = fix((batch, T), dtypes.float, np.float32)
    step = make_step(model, opt, b_pic, b_label, b_real, class_w)
    step = TinyJit(step)

    rng = np.random.default_rng(seed)
    row_rng = np.random.RandomState(seed)
    t0 = time.time()
    for s in range(steps):
        idxs = row_rng.choice(src.n, size=batch, replace=False)
        pics, labels, reals = build_batch(src, idxs, un_picture, mean, comps, tok, T, rng)
        b_pic.assign(Tensor(np.ascontiguousarray(pics), dtype=b_pic.dtype)).realize()
        b_label.assign(Tensor(np.ascontiguousarray(labels), dtype=b_label.dtype)).realize()
        b_real.assign(Tensor(np.ascontiguousarray(reals), dtype=b_real.dtype)).realize()
        t_step0 = time.time()
        loss, healthy = step()
        dt_step = time.time() - t_step0
        if s % log_every == 0 or s == steps - 1:
            print(f"[unet/train] step {s:7d} loss={loss.item():.5f} "
                  f"healthy={healthy.item():.0f} step_time={dt_step:.3f}s "
                  f"wall={time.time()-t0:.1f}s", flush=True)
        if snap_every and (s + 1) % snap_every == 0:
            safe_save(get_state_dict(model), ckpt)
            print(f"[unet/train] snapshot -> {ckpt}", flush=True)
    safe_save(get_state_dict(model), ckpt)
    print(f"[unet/train] DONE -> {ckpt}", flush=True)


def smoke_test(un_picture):
    """THE PINNED CPU SMOKE TEST (per the word: build-only, never the
    GPU): 2 steps, batch 2, 8 diet rows, T=64 pictures. Verifies (a) the
    loss is finite at both steps and prints whether it decreased, and (b)
    the per-breath mask changes the loss -- by comparing the trained
    step's own loss against the SAME batch/weights run through
    `forward_breaths` with the where-schedule forced to all-zero (a
    non-jitted side call, done BEFORE any optimizer step so weights are
    identical across the comparison)."""
    from tokenizers import Tokenizer
    import phase1_algebra_head as H
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    split, jsonl = "formpm35c", ".cache/form_mix_pm35c.jsonl"
    src = PictureSource(split, jsonl)
    mean, comps = fit_pca()
    T, batch, steps = 64, 2, 2
    class_w = fixed_class_weights("1,50,50,50")
    Cin = n_channels(un_picture)
    n_where = 0 if un_picture == "Aprime" else 3
    model = UNet(c_content=Cin - n_where, n_where=n_where, K_B=4, base=8, seed=0)
    opt = AdamW(model.parameters(), lr=1e-3)

    def fix(shape, dt, npdt):
        return Tensor(np.zeros(shape, npdt), dtype=dt).contiguous().realize()
    b_pic = fix((batch, Cin, T, T), dtypes.float, np.float32)
    b_label = fix((batch, T, T), dtypes.int, np.int32)
    b_real = fix((batch, T), dtypes.float, np.float32)
    step = make_step(model, opt, b_pic, b_label, b_real, class_w)
    step = TinyJit(step)

    rng = np.random.default_rng(0)
    row_rng = np.random.RandomState(0)
    row_idxs = row_rng.choice(src.n, size=8, replace=False)
    losses = []
    for s in range(steps):
        idxs = row_rng.choice(row_idxs, size=batch, replace=False)
        pics, labels, reals = build_batch(src, idxs, un_picture, mean, comps, tok, T, rng)
        if s == 0 and n_where:
            # (b) the ablation check, BEFORE the first opt.step() touches
            # the weights -- same pics/model, where-schedule forced off.
            x_t = Tensor(np.ascontiguousarray(pics), dtype=dtypes.float)
            out_default = model.forward_breaths(x_t).numpy()
            out_ablated = model.forward_breaths(x_t, where_sched=np.zeros((4, 3), np.float32)).numpy()
            changed = not np.allclose(out_default, out_ablated)
            print(f"[unet/train smoke] per-breath mask changes output when a "
                  f"where-channel is switched off: {changed}")
            assert changed, "the where-schedule has NO effect on forward_breaths's output"
        b_pic.assign(Tensor(np.ascontiguousarray(pics), dtype=b_pic.dtype)).realize()
        b_label.assign(Tensor(np.ascontiguousarray(labels), dtype=b_label.dtype)).realize()
        b_real.assign(Tensor(np.ascontiguousarray(reals), dtype=b_real.dtype)).realize()
        t0 = time.time()
        loss, healthy = step()
        dt = time.time() - t0
        lv = loss.item()
        losses.append(lv)
        print(f"[unet/train smoke] step {s} loss={lv:.5f} healthy={healthy.item():.0f} "
              f"step_time={dt:.3f}s")
        assert np.isfinite(lv), f"non-finite loss at smoke step {s}"
    print(f"[unet/train smoke] losses={losses} "
          f"{'DECREASED' if losses[-1] < losses[0] else 'did not decrease'} "
          f"(2 steps is not a convergence claim either way)")
    print("[unet/train smoke] PASS")


if __name__ == "__main__":
    un_picture = os.environ.get("UN_PICTURE", "Adouble")
    assert un_picture in ("Aprime", "Adouble", "Adouble_shuf"), un_picture
    if os.environ.get("UN_SMOKE"):
        smoke_test(un_picture)
    else:
        run(
            steps=int(os.environ.get("UN_STEPS", "20000")),
            batch=int(os.environ.get("UN_BATCH", "4")),
            T=int(os.environ.get("UN_T", "256")),
            un_picture=un_picture,
            lr=float(os.environ.get("UN_LR", "3e-4")),
            base=int(os.environ.get("UN_BASE", "16")),
            K_B=int(os.environ.get("UN_BREATHS", "7")),
            seed=int(os.environ.get("UN_SEED", "0")),
            split=os.environ.get("UN_SPLIT", "formpm35c"),
            jsonl=os.environ.get("UN_JSONL", ".cache/form_mix_pm35c.jsonl"),
            class_w_spec=os.environ.get("UN_CLASS_W", "1,50,50,50"),
            ckpt=os.environ.get("UN_CKPT", f".cache/unet_{un_picture}.safetensors"),
            log_every=int(os.environ.get("UN_LOG_EVERY", "200")),
            snap_every=int(os.environ.get("UN_SNAP_EVERY", "5000")),
            layers=os.environ.get("UN_LAYERS", "default"),
        )
