"""First training step on the breathing transformer.

Pythia-initialized weights, synthetic next-token CE loss, AdamW. Verifies that:
  1. Forward + loss is finite
  2. Backward populates gradients on EVERY parameter family
  3. Optimizer step actually changes the parameters
  4. A second forward step produces a different (lower, ideally) loss
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW

from mycelium import Config
from mycelium.loader import _load_state, load_breathing


def grad_norm(t: Tensor) -> float:
    if t.grad is None:
        return float("nan")
    g = t.grad.cast(dtypes.float).square().sum().realize()
    return float(g.numpy()) ** 0.5


def param_diff_norm(before: np.ndarray, after_t: Tensor) -> float:
    after = after_t.cast(dtypes.float).realize().numpy()
    return float(np.linalg.norm(after - before))


def named_params(model):
    """Returns [(name, tensor)] for every trainable parameter family."""
    nps = []
    nps.append(("embed", model.embed.weight))
    nps.append(("embed_out", model.embed_out))
    nps.append(("ln_f.g", model.ln_f_g))
    nps.append(("ln_f.b", model.ln_f_b))

    sw = model.block.shared
    nps.append(("shared.wv", sw.wv))
    nps.append(("shared.bv", sw.bv))
    nps.append(("shared.wo", sw.wo))
    nps.append(("shared.bo", sw.bo))
    nps.append(("shared.w_out", sw.w_out))
    nps.append(("shared.b_out", sw.b_out))
    nps.append(("shared.in_ln.g", sw.in_ln_g))
    nps.append(("shared.in_ln.b", sw.in_ln_b))
    nps.append(("shared.post_ln.g", sw.post_ln_g))
    nps.append(("shared.post_ln.b", sw.post_ln_b))

    for i, layer in enumerate(model.block.layers):
        nps.append((f"phase{i}.wq", layer.wq))
        nps.append((f"phase{i}.bq", layer.bq))
        nps.append((f"phase{i}.wk", layer.wk))
        nps.append((f"phase{i}.bk", layer.bk))
        nps.append((f"phase{i}.w_in", layer.w_in))
        nps.append((f"phase{i}.b_in", layer.b_in))
    return nps


def forward_loss(model, tokens, n_loops):
    """Compute next-token CE loss using the breathed integral as the representation."""
    # Hidden states post-final-LN, shape (B, S, hidden), dtype half
    h = model(tokens, n_loops)
    # Output head: logits (B, S, vocab). Cast to float32 for stable CE.
    logits = (h @ model.embed_out).cast(dtypes.float)
    # Shift: predict tokens[1:] from logits[:-1]
    pred = logits[:, :-1, :]
    targ = tokens[:, 1:]
    return pred.sparse_categorical_crossentropy(targ, reduction="mean")


def main():
    cfg = Config()
    B = getenv("B", 4)
    SEQ = getenv("SEQ", 64)
    N_LOOPS = getenv("LOOPS", 2)
    LR = float(getenv("LR", "1e-4"))

    print(f"device={Device.DEFAULT}  B={B}  seq={SEQ}  loops={N_LOOPS}  lr={LR}")

    print("\nLoading Pythia weights into BreathingTransformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)

    # Cast all trainable weights to FP32 for stable AdamW. (FP16 v=g^2 underflows
    # to zero in tinygrad's optimizer state, blowing up the update.) Mixed-
    # precision training in FP16 is a future optimization that needs a proper
    # loss scaler — out of scope for this milestone.
    def _cast_attr_fp32(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            new_t = t.cast(dtypes.float).contiguous().realize()
            setattr(obj, attr, new_t)
    for attr in ("weight",):
        _cast_attr_fp32(model.embed, attr)
    for attr in ("embed_out",):
        _cast_attr_fp32(model, attr)
    sw = model.block.shared
    for attr in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast_attr_fp32(sw, attr)
    for layer in model.block.layers:
        for attr in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast_attr_fp32(layer, attr)
    Device[Device.DEFAULT].synchronize()

    # Synthetic input tokens
    Tensor.manual_seed(42)
    tokens = Tensor.randint(B, SEQ, low=0, high=cfg.vocab_size).realize()

    # Collect parameters and snapshot pre-step values
    nps = named_params(model)
    params = [t for _, t in nps]
    before_snap = {n: t.cast(dtypes.float).realize().numpy().copy() for n, t in nps}

    # tinygrad's AdamW factory already wraps LAMB with adam=True (no trust-ratio).
    opt = AdamW(params, lr=LR)
    Device[Device.DEFAULT].synchronize()

    # ---- Step 1: forward + backward + step ----
    # NOTE: tinygrad consumes the lazy graph on .realize()/.numpy(); backward must
    # run BEFORE we extract the loss value, otherwise grads come back as None.
    print("\n--- Step 1 (Pythia init -> next-token CE) ---")
    t0 = time.perf_counter()
    Tensor.training = True
    opt.zero_grad()
    loss = forward_loss(model, tokens, N_LOOPS)
    loss.backward()
    Device[Device.DEFAULT].synchronize()
    fwd_bwd_t = time.perf_counter() - t0
    loss_val = float(loss.numpy())
    print(f"forward+backward loss: {loss_val:.4f}  (time {fwd_bwd_t:.2f}s)")

    if not (loss_val == loss_val):
        print("LOSS IS NaN — aborting")
        return

    # Gradient flow check — every named parameter family must have finite, non-trivial grad
    print(f"\n{'parameter':<22} {'shape':<22} {'grad_norm':>14}  flag")
    fail = []
    for name, t in nps:
        gn = grad_norm(t)
        flag = "OK"
        if t.grad is None:
            flag = "MISSING"
            fail.append(name)
        elif not (gn == gn):
            flag = "NaN"
            fail.append(name)
        elif gn == 0.0:
            flag = "ZERO"
            fail.append(name)
        print(f"{name:<22} {str(tuple(t.shape)):<22} {gn:>14.4e}  {flag}")

    if fail:
        print(f"\nGRADIENT FLOW FAIL on: {fail}")
    else:
        print(f"\n[OK] gradients reached all {len(nps)} parameter families")

    t0 = time.perf_counter()
    opt.step()
    Device[Device.DEFAULT].synchronize()
    step_t = time.perf_counter() - t0
    print(f"\noptimizer step time: {step_t:.2f}s")

    # Param-change check
    print(f"\n{'parameter':<22} {'param_change_norm':>20}")
    no_change = []
    for name, t in nps:
        d = param_diff_norm(before_snap[name], t)
        if d == 0.0:
            no_change.append(name)
        print(f"{name:<22} {d:>20.4e}")
    if no_change:
        print(f"\nPARAMS UNCHANGED on: {no_change}")
    else:
        print(f"\n[OK] all {len(nps)} parameter families updated by AdamW")

    # ---- Step 2: confirm loss can decrease ----
    print("\n--- Step 2 (verify loss can move) ---")
    opt.zero_grad()
    loss2 = forward_loss(model, tokens, N_LOOPS)
    loss2_val = float(loss2.numpy())
    print(f"step 2 loss: {loss2_val:.4f}  (delta {loss2_val - loss_val:+.4f})")
    direction = "DECREASED" if loss2_val < loss_val else "INCREASED"
    print(f"[{direction}]")


if __name__ == "__main__":
    main()
