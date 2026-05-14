"""Profile one training step in detail. Run with the SAME hyperparameters as a real
training run to identify hot spots.

Outputs:
- Forward pass time (with internal breakdown if possible)
- Backward pass time
- Optimizer step time
- Per-breath time (for n_loops=8)
- Per-layer time within a breath (RISE/PEAK/FALL/TROUGH)
- Memory snapshot

Usage:
  DEV='PCI+AMD' CKPT=/path/to/ckpt.safetensors \\
    .venv/bin/python scripts/profile_train_step.py

Caveats:
- Inserts Device.synchronize() between blocks for timing — disables JIT fusion across
  those boundaries, so absolute numbers will be SLIGHTLY pessimistic. Relative
  proportions still informative.
- Run AFTER any training run finishes (GPU lock is held during training).
"""
import os
import sys
import time

os.environ.setdefault("DEV", "PCI+AMD")
os.environ.setdefault("SINE_TEMP", "1")
os.environ.setdefault("SINE_TEMP_MAX", "2.0")
os.environ.setdefault("SINE_TEMP_MIN", "0.7")
os.environ.setdefault("ABLATE_BREATH_ROTATION", "1")
os.environ.setdefault("BREATH_TIME_EMBED", "1")
os.environ.setdefault("BREATH_TIME_INIT_SCALE", "0.0")
os.environ.setdefault("CROSS_BREATH_HANDOFF", "1")
os.environ.setdefault("LEARN_PITCH", "1")

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, encode_cycles, collate, split_train_eval
from mycelium.lookup_table import eq_token_ids_for
from tinygrad.nn.state import safe_load


# Match l3_train.py's collect_params + cast pattern
def cast_model_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    _cast(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def collect_params(model):
    nps = [model.embed.weight, model.embed_out, model.ln_f_g, model.ln_f_b]
    sw = model.block.shared
    nps += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
            sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        nps += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    nps += [model.block.breath_embed]
    nps += [model.block.handoff_w, model.block.handoff_b]
    nps += [model.block.rope.pitch]
    nps += model.confidence_head.parameters()
    return nps


def sync():
    Device[Device.DEFAULT].synchronize()


def section(label, fn, *args, warmup=2, iters=5, **kwargs):
    """Time fn over iters runs (after warmup). Returns mean ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
        sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        sync()
        ts.append(time.perf_counter() - t0)
    avg_ms = sum(ts) / len(ts) * 1000
    print(f"  {label:40s} {avg_ms:7.1f} ms  (min {min(ts)*1000:.1f}, max {max(ts)*1000:.1f}, n={iters})")
    return avg_ms, result


def main():
    CKPT = os.environ.get("CKPT", "/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v12_cross_breath_handoff_step1500.safetensors")
    BATCH = int(os.environ.get("BATCH", "16"))
    N_LOOPS = int(os.environ.get("N_LOOPS", "8"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "96"))

    print(f"=== Profiling one training step ===")
    print(f"  ckpt:      {CKPT}")
    print(f"  BATCH:     {BATCH}")
    print(f"  N_LOOPS:   {N_LOOPS}")
    print(f"  FIXED_LEN: {FIXED_LEN}")
    print()

    # Build model
    cfg = Config()
    print("Loading Pythia base...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    sync()
    print(f"Model built. Loading ckpt: {CKPT}")
    state = safe_load(CKPT)
    info = model.load_state_dict(state, strict=False)
    if info.get("missing"):
        print(f"  ({len(info['missing'])} missing keys, kept default init: {info['missing'][:3]}...)")
    sync()

    tok = load_tokenizer()
    examples = generate_math("L4_MIXED", num_problems=200, seed=42, digit_spacing=True)
    train, _ = split_train_eval(examples, n_eval=20, seed=42)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(train), size=BATCH)
    batch = [train[int(i)] for i in idx]

    # Encode + collate (single cycle for simplicity; we time the per-cycle path)
    cycles_per_ex = [encode_cycles(tok, ex) for ex in batch]
    encoded = [ex_cycles[0] for ex_cycles in cycles_per_ex]
    tokens_np, labels_np = collate(encoded, fixed_len=FIXED_LEN)
    tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
    labels = Tensor(labels_np, dtype=dtypes.int).realize()

    params = collect_params(model)
    opt = AdamW(params, lr=3e-5)
    Tensor.training = True
    sync()

    print()
    print("=== Timing breakdown (after warmup) ===")

    # --- Full forward pass ---
    def fwd():
        return model(tokens, N_LOOPS)
    fwd_ms, _ = section("Forward pass (n_loops=8)", fwd)

    # --- Forward + loss ---
    def fwd_loss():
        out = model(tokens, N_LOOPS)
        logits = (out @ model.embed_out).cast(dtypes.float)
        pred = logits[:, :-1, :]
        loss = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")
        return loss
    fwdloss_ms, loss = section("Forward + loss (CE)", fwd_loss)

    # --- Forward + loss + backward ---
    def fwd_bwd():
        opt.zero_grad()
        out = model(tokens, N_LOOPS)
        logits = (out @ model.embed_out).cast(dtypes.float)
        pred = logits[:, :-1, :]
        loss = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")
        loss.backward()
        return loss
    fwdbwd_ms, _ = section("Forward + loss + backward", fwd_bwd)

    # --- Full step ---
    def full_step():
        opt.zero_grad()
        out = model(tokens, N_LOOPS)
        logits = (out @ model.embed_out).cast(dtypes.float)
        pred = logits[:, :-1, :]
        loss = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")
        loss.backward()
        opt.step()
        return loss
    full_ms, _ = section("Full step (fwd + loss + bwd + opt.step)", full_step)

    print()
    print("=== Derived ===")
    print(f"  Forward:   {fwd_ms:.1f} ms")
    print(f"  Loss only: {fwdloss_ms - fwd_ms:.1f} ms")
    print(f"  Backward:  {fwdbwd_ms - fwdloss_ms:.1f} ms")
    print(f"  Opt.step:  {full_ms - fwdbwd_ms:.1f} ms")
    print()
    print(f"  Forward ratio:  {fwd_ms/full_ms*100:5.1f}%")
    print(f"  Backward ratio: {(fwdbwd_ms - fwdloss_ms)/full_ms*100:5.1f}%")
    print(f"  Opt.step ratio: {(full_ms - fwdbwd_ms)/full_ms*100:5.1f}%")

    # --- Per-breath timing ---
    print()
    print("=== Per-breath timing (forward only) ===")

    def one_breath(x, l):
        return model.block.breathe_once(x, l, temp_mult=1.0)

    x = model.embed(tokens).cast(dtypes.half)
    sync()
    breath_times = []
    for l in range(N_LOOPS):
        for _ in range(2):  # warmup
            _ = one_breath(x, l)
            sync()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            x_out = one_breath(x, l)
            sync()
            ts.append(time.perf_counter() - t0)
        breath_times.append(min(ts) * 1000)
        x = x_out  # propagate
    for l, t in enumerate(breath_times):
        print(f"  breath {l}: {t:6.1f} ms")
    print(f"  total: {sum(breath_times):.1f} ms (vs full forward {fwd_ms:.1f} ms — note: this adds sync overhead)")


if __name__ == "__main__":
    main()
