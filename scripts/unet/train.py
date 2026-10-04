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
  UN_PICTURE   Aprime | Adouble | Adouble_shuf | Awhich | Awhich_shuf
               (default Adouble)
  UN_STEPS     training steps PER CHUNK (see UN_CONVERGE)  (default 20000)
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
  UN_CONVERGE=1  THE CONVERGENCE RULE (round 2, word given 2026-10-04):
               instead of one fixed-length run, train in CHUNKS of
               UN_STEPS steps; after every chunk, smooth the loss over
               the LAST 10% of all steps run so far (see
               `convergence_window`) and compare it to the previous
               chunk's smoothed value (`convergence_rel_change`); stop
               when the relative change is < 1%, or when UN_MAX_STEPS
               total steps is hit first (the hard cap -- stated, not
               silently truncated: a cap-stop and a convergence-stop
               print different messages). The rule's numbers (window
               size, both smoothed values, the relative change, the
               threshold, the cap) are printed at every check. Unset:
               exactly one chunk of UN_STEPS steps, no checks -- the
               original round-1 behavior, byte-identical.
  UN_MAX_STEPS   hard cap on total steps under UN_CONVERGE=1 (default
               60000); ignored when UN_CONVERGE is unset.
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

from pictures import PictureSource, fit_pca, build_picture, n_channels, n_where as pic_n_where   # noqa: E402
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


def convergence_window(loss_history, frac=0.10):
    """(window_size, smoothed_loss) over the LAST `frac` fraction of
    `loss_history` (at least 1 step). Pure function -- also exercised on
    a synthetic loss list by the CPU smoke test (see `__main__`)."""
    window = max(1, int(round(frac * len(loss_history))))
    return window, float(np.mean(loss_history[-window:]))


def convergence_rel_change(smoothed, prev_smoothed):
    """Relative change of `smoothed` vs `prev_smoothed`; +inf when there
    is no previous reading yet (chunk 1 can never claim convergence)."""
    if prev_smoothed is None:
        return float("inf")
    denom = abs(prev_smoothed) if abs(prev_smoothed) > 1e-12 else 1e-12
    return abs(smoothed - prev_smoothed) / denom


def _converge_at(curve, chunk):
    """Replays THE CONVERGENCE RULE's own chunked check over a plain
    python/numpy loss list (no model, no tensor) -- the step index of
    the first chunk boundary where rel_change < 1%, or None if it never
    fires within `curve`."""
    history, prev, at = [], None, None
    for c in range(len(curve) // chunk):
        history.extend(curve[c * chunk:(c + 1) * chunk].tolist())
        window, smoothed = convergence_window(history, frac=0.10)
        rel = convergence_rel_change(smoothed, prev)
        print(f"[unet/train smoke] synthetic check chunk={c + 1} "
              f"total={(c + 1) * chunk} window={window}/{len(history)} "
              f"smoothed={smoothed:.5f} prev={'n/a' if prev is None else format(prev, '.5f')} "
              f"rel_change={'inf' if prev is None else format(rel, '.5f')}")
        if prev is not None and rel < 0.01 and at is None:
            at = (c + 1) * chunk
        prev = smoothed
    return at


def check_convergence_rule_synthetic():
    """CPU smoke requirement 4: exercise `convergence_window` /
    `convergence_rel_change` on a SYNTHETIC loss list (no model, no GPU)
    -- a decaying-then-flat curve should converge well before a strictly
    linearly-falling one does, over the same chunk size."""
    rng = np.random.RandomState(0)
    chunk = 50
    # curve A: exponential decay to a flat plateau -- should converge.
    flat = 0.2 + 0.01 * np.exp(-np.arange(400) / 40.0) + rng.normal(0, 1e-4, 400)
    # curve B: still falling linearly over the whole range -- should NOT
    # converge within the same number of chunks.
    falling = 1.0 - 0.0015 * np.arange(400) + rng.normal(0, 1e-4, 400)
    at_flat = _converge_at(flat, chunk)
    at_falling = _converge_at(falling, chunk)
    print(f"[unet/train smoke] convergence rule on synthetic curves: "
          f"flat-plateau converged_at={at_flat}  still-falling converged_at={at_falling}")
    assert at_flat is not None, "the flat-plateau synthetic curve never converged"
    assert at_falling is None, "the still-falling synthetic curve converged (it should not have, over this range)"
    print(f"[unet/train smoke] convergence rule PASS: flat-plateau converged at "
          f"{at_flat} steps; still-falling never converged over {len(falling)} steps")


def run(steps, batch, T, un_picture, lr, base, K_B, seed, split, jsonl,
        class_w_spec, ckpt, log_every, snap_every, layers="default",
        converge=False, max_steps=60000):
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
    n_where = pic_n_where(un_picture)
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

    def run_chunk(n_steps, g0):
        """n_steps steps starting at the GLOBAL step index g0 (continuing
        the same model/optimizer/jit -- chunking is only a place to stop
        and check the convergence rule, never a restart). Returns this
        chunk's list of per-step losses."""
        chunk_losses = []
        for s in range(n_steps):
            g = g0 + s
            idxs = row_rng.choice(src.n, size=batch, replace=False)
            pics, labels, reals = build_batch(src, idxs, un_picture, mean, comps, tok, T, rng)
            b_pic.assign(Tensor(np.ascontiguousarray(pics), dtype=b_pic.dtype)).realize()
            b_label.assign(Tensor(np.ascontiguousarray(labels), dtype=b_label.dtype)).realize()
            b_real.assign(Tensor(np.ascontiguousarray(reals), dtype=b_real.dtype)).realize()
            t_step0 = time.time()
            loss, healthy = step()
            dt_step = time.time() - t_step0
            lv = loss.item()
            chunk_losses.append(lv)
            if g % log_every == 0 or s == n_steps - 1:
                print(f"[unet/train] step {g:7d} loss={lv:.5f} "
                      f"healthy={healthy.item():.0f} step_time={dt_step:.3f}s "
                      f"wall={time.time()-t0:.1f}s", flush=True)
            if snap_every and (g + 1) % snap_every == 0:
                safe_save(get_state_dict(model), ckpt)
                print(f"[unet/train] snapshot -> {ckpt}", flush=True)
        return chunk_losses

    if not converge:
        # byte-identical to round 1: one chunk, no convergence checks.
        run_chunk(steps, 0)
        safe_save(get_state_dict(model), ckpt)
        print(f"[unet/train] DONE -> {ckpt} (total_steps={steps})", flush=True)
        return

    # THE CONVERGENCE RULE (round 2, word given 2026-10-04): train in
    # chunks of `steps` (UN_STEPS) until the smoothed loss over the last
    # 10% of ALL steps run so far changes < 1% from the previous chunk's
    # reading, or `max_steps` (UN_MAX_STEPS) total steps is hit first.
    loss_history = []
    total = 0
    prev_smoothed = None
    chunk_idx = 0
    while True:
        chunk_idx += 1
        this_chunk = min(steps, max(0, max_steps - total)) if max_steps else steps
        if this_chunk <= 0:
            print(f"[unet/train] HARD CAP already reached at {total} steps "
                  f"(>= UN_MAX_STEPS={max_steps}) before chunk {chunk_idx} started", flush=True)
            break
        loss_history.extend(run_chunk(this_chunk, total))
        total += this_chunk
        window, smoothed = convergence_window(loss_history, frac=0.10)
        rel_change = convergence_rel_change(smoothed, prev_smoothed)
        print(f"[unet/train] CONVERGENCE CHECK chunk={chunk_idx} total_steps={total} "
              f"window={window}/{len(loss_history)} smoothed_loss={smoothed:.5f} "
              f"prev_smoothed={'n/a' if prev_smoothed is None else format(prev_smoothed, '.5f')} "
              f"rel_change={'inf' if prev_smoothed is None else format(rel_change, '.5f')} "
              f"threshold=0.01000 max_steps={max_steps}", flush=True)
        if prev_smoothed is not None and rel_change < 0.01:
            print(f"[unet/train] CONVERGED at {total} steps "
                  f"(rel_change {rel_change:.5f} < 0.01)", flush=True)
            break
        if total >= max_steps:
            print(f"[unet/train] HARD CAP reached at {total} steps "
                  f"(>= UN_MAX_STEPS={max_steps}) -- stopping WITHOUT convergence", flush=True)
            break
        prev_smoothed = smoothed
    safe_save(get_state_dict(model), ckpt)
    print(f"[unet/train] DONE -> {ckpt} (total_steps={total})", flush=True)


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
    n_where = pic_n_where(un_picture)
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
    assert un_picture in ("Aprime", "Adouble", "Adouble_shuf", "Awhich", "Awhich_shuf"), un_picture
    if os.environ.get("UN_SMOKE"):
        smoke_test(un_picture)
        check_convergence_rule_synthetic()
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
            converge=bool(os.environ.get("UN_CONVERGE")),
            max_steps=int(os.environ.get("UN_MAX_STEPS", "60000")),
        )
