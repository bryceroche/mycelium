"""GATE 1 — byte-identity validation of PERCEIVER_DEFUSE_BREATH.

Runs ONE forward+backward (NO opt.step — clean at init weights) with the DEFUSE
toggle OFF then ON, on the SAME seed / SAME batch / SAME fresh weights, and
compares: total loss, cell_ce, the per-breath CE ladder, grad_norm, cell_logits
(final breath), and the t=0 membership_match. .contiguous() is a pure
fusion-grouping realize barrier -> EXACT (to fp round-off) match expected.

Env (set by the caller):
  SEED=42 BATCH=8 PERCEIVER_K_MAX=8 PERCEIVER_BALL_PATH=per_constraint
  KENKEN_TRAIN/KENKEN_TEST = the curriculum corpora
  PERCEIVER_CONSTRAINT_WEIGHT=0.3 (the energy term; identical both toggles)

This script flips the module-global PERCEIVER_DEFUSE_BREATH in-process and
rebuilds the model from the same seed for each toggle, so the comparison is
apples-to-apples (identical RNG draws, identical weights, identical batch).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken_data import N_CELLS, N_MAX, load_jsonl
from mycelium.kenken import kenken_constraint_energy
import mycelium.perceiver_poincare as pp
from mycelium.perceiver_poincare import (
    attach_perceiver_params, perceiver_breathing_forward, t0_anchor_check,
)
from mycelium.perceiver_poincare_data import PerceiverLoader, latent_capacity

# ---- mirror the trainer's fp32 cast + collect_params ----
from scripts.perceiver_train import cast_layers_fp32, collect_params

K     = int(getenv("PERCEIVER_K_MAX", 8))
BATCH = int(getenv("BATCH", 8))
SEED  = int(getenv("SEED", 42))
BALL_PATH = getenv("PERCEIVER_BALL_PATH", "per_constraint").strip().lower()
CW    = float(getenv("PERCEIVER_CONSTRAINT_WEIGHT", "0.3"))
KENKEN_TRAIN = getenv("KENKEN_TRAIN", ".cache/kenken_train_curriculum.jsonl")
KENKEN_TEST  = getenv("KENKEN_TEST",  ".cache/kenken_test_curriculum.jsonl")

# n_cages_max over BOTH corpora (stable latent shape) — same as the trainer.
train_recs = load_jsonl(KENKEN_TRAIN)
test_recs  = load_jsonl(KENKEN_TEST)
N_CAGES_MAX = max(max(len(r["cages"]) for r in train_recs),
                  max(len(r["cages"]) for r in test_recs))
L_MAX = latent_capacity(N_CAGES_MAX, pp.PERCEIVER_N_GLOBAL)
print(f"K={K} B={BATCH} L_MAX={L_MAX} ball_path={BALL_PATH} cw={CW} seed={SEED}", flush=True)

cfg = Config()
sd_disk = _load_state()


def build_model():
    """Fresh model from the SAME seed -> identical weights every call."""
    np.random.seed(SEED)
    Tensor.manual_seed(SEED)
    model = load_breathing(cfg, sd=sd_disk)
    cast_layers_fp32(model)
    attach_perceiver_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                            L_max=L_MAX, k_max=K)
    Device[Device.DEFAULT].synchronize()
    return model


def make_batch():
    """Same seed -> the loader emits the SAME first batch."""
    loader = PerceiverLoader(KENKEN_TRAIN, batch_size=BATCH, seed=SEED,
                             n_cages_max=N_CAGES_MAX)
    return loader.sample_batch()


def run_one(defuse: bool):
    pp.PERCEIVER_DEFUSE_BREATH = bool(defuse)
    model = build_model()
    batch = make_batch()

    # t=0 anchor check (membership_match) — pure read, no training.
    Tensor.training = False
    chk = t0_anchor_check(model, batch, BALL_PATH)
    mm = float(chk["membership_match"])
    Tensor.training = True

    params = collect_params(model, ball_path=BALL_PATH)
    opt = AdamW(params, lr=0.0, weight_decay=0.0)  # lr=0: step would be a no-op anyway; we DON'T step
    opt.zero_grad()

    cell_logits_history, _ = perceiver_breathing_forward(
        model, batch, K=K, ball_path=BALL_PATH, collect_engagement=False)

    input_cells = batch.input_cells
    gold = batch.gold
    cell_valid = batch.cell_valid
    observed = (input_cells > 0).cast(dtypes.float)
    supervise = cell_valid * (1.0 - observed)
    sup_sum = supervise.sum() + 1e-6
    gold_idx = (gold - 1).clip(0, N_MAX - 1).reshape(BATCH * N_CELLS)
    supervise_flat = supervise.reshape(BATCH * N_CELLS)

    cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    weight_sum = 0.0
    per_breath_ce = []
    for k, logits in enumerate(cell_logits_history):
        weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
        ce_elems = logits.reshape(BATCH * N_CELLS, N_MAX).sparse_categorical_crossentropy(
            gold_idx, reduction="none")
        ce_k = (ce_elems * supervise_flat).sum() / sup_sum
        per_breath_ce.append(ce_k)
        cell_loss_sum = cell_loss_sum + ce_k * weight_k
        weight_sum += weight_k
    cell_loss = cell_loss_sum / float(weight_sum)

    final_probs = cell_logits_history[-1].softmax(axis=-1)
    energy = kenken_constraint_energy(final_probs, batch).mean()
    total = cell_loss + CW * energy
    total.backward()

    sq = Tensor.zeros((), dtype=dtypes.float).contiguous()
    for p in params:
        if p.grad is not None:
            sq = sq + p.grad.cast(dtypes.float).square().sum()
    grad_norm = (sq + 1e-12).sqrt()

    final_logits = cell_logits_history[-1]

    out = {
        "total": float(total.numpy()),
        "cell_ce": float(cell_loss.numpy()),
        "grad_norm": float(grad_norm.numpy()),
        "per_breath_ce": [float(c.numpy()) for c in per_breath_ce],
        "membership_match": mm,
        "cell_logits": final_logits.numpy().astype(np.float64),  # (B,49,N_MAX)
    }
    del model, opt, params, cell_logits_history
    import gc as _gc; _gc.collect()
    return out


print("\n=== RUN A: PERCEIVER_DEFUSE_BREATH = 0 (baseline / HEAD) ===", flush=True)
a = run_one(False)
print("\n=== RUN B: PERCEIVER_DEFUSE_BREATH = 1 (DEFUSE on) ===", flush=True)
b = run_one(True)

print("\n=== GATE 1 COMPARISON (DEFUSE off vs on) ===", flush=True)


def cmp(name, x, y):
    d = abs(x - y)
    rel = d / (abs(x) + 1e-12)
    print(f"  {name:>18}: off={x:.10g}  on={y:.10g}  |abs_diff|={d:.3e}  rel={rel:.3e}")
    return d, rel


dt, _ = cmp("total", a["total"], b["total"])
dc, _ = cmp("cell_ce", a["cell_ce"], b["cell_ce"])
dg, rg = cmp("grad_norm", a["grad_norm"], b["grad_norm"])
dm, _ = cmp("membership_match", a["membership_match"], b["membership_match"])

ladder_diffs = [abs(x - y) for x, y in zip(a["per_breath_ce"], b["per_breath_ce"])]
print(f"  per_breath_ce off: {['%.6f'%x for x in a['per_breath_ce']]}")
print(f"  per_breath_ce on : {['%.6f'%x for x in b['per_breath_ce']]}")
print(f"  per_breath_ce max|abs_diff| = {max(ladder_diffs):.3e}")

logit_absdiff = np.abs(a["cell_logits"] - b["cell_logits"])
print(f"  cell_logits shape={a['cell_logits'].shape}  "
      f"max|abs_diff|={logit_absdiff.max():.3e}  mean|abs_diff|={logit_absdiff.mean():.3e}")

# verdict: byte-identical to ~1e-6 on scalars; logits to a small absolute tol
TOL = 1e-6
scalar_ok = (dt < TOL and dc < TOL and rg < 1e-5 and dm < TOL
             and max(ladder_diffs) < TOL)
# logits can carry slightly larger absolute error from reduction reorder; allow 1e-4 abs
logit_ok = logit_absdiff.max() < 1e-4
PASS = scalar_ok and logit_ok
print(f"\n  GATE1_BYTE_IDENTICAL = {PASS}  (scalar_ok={scalar_ok} logit_ok={logit_ok})")

summary = {
    "gate1_pass": bool(PASS),
    "total_off": a["total"], "total_on": b["total"], "total_absdiff": dt,
    "cell_ce_off": a["cell_ce"], "cell_ce_on": b["cell_ce"], "cell_ce_absdiff": dc,
    "grad_norm_off": a["grad_norm"], "grad_norm_on": b["grad_norm"],
    "grad_norm_absdiff": dg, "grad_norm_reldiff": rg,
    "membership_match_off": a["membership_match"],
    "membership_match_on": b["membership_match"],
    "per_breath_ce_off": a["per_breath_ce"], "per_breath_ce_on": b["per_breath_ce"],
    "per_breath_ce_maxdiff": max(ladder_diffs),
    "cell_logits_maxabsdiff": float(logit_absdiff.max()),
    "cell_logits_meanabsdiff": float(logit_absdiff.mean()),
}
print("GATE1_JSON " + json.dumps(summary))
