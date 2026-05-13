"""Verify the JIT'd calibration_train_step matches the eager path.

Loads the same warm-start ckpt twice. Runs ONE calibration step with use_jit=False,
records the loss + diagnostics. Re-loads. Runs ONE step with use_jit=True from
the identical starting state. Compares.

Float32 numerical noise is acceptable (diff < 1e-3). Larger deltas mean JIT
miscompilation or graph-structure differences.
"""
import os
import sys

# Set platform before importing tinygrad
os.environ.setdefault("DEV", "PCI+AMD")
os.environ.setdefault("SINE_TEMP", "1")
os.environ.setdefault("SINE_TEMP_MAX", "2.0")
os.environ.setdefault("SINE_TEMP_MIN", "0.7")

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.optim import AdamW

# Mycelium imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval
from mycelium.l3_training import calibration_train_step
from mycelium.lookup_table import eq_token_ids_for
from mycelium.calibration import digit_token_ids_for

# Same collect_params and cast_model_fp32 as l3_train.py
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
    nps += model.confidence_head.parameters()
    return nps


def build_model_and_load(ckpt_path: str):
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    # Load warm-start
    from tinygrad.nn.state import safe_load
    state = safe_load(ckpt_path)
    info = model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    return model, cfg


def run_one_step(model, batch_examples, tok, digit_ids, eq_ids,
                  n_loops, fixed_len, calibration_weight, use_jit):
    params = collect_params(model)
    opt = AdamW(params, lr=3e-5)
    Tensor.training = True
    info = calibration_train_step(
        model, opt, batch_examples, tok,
        digit_token_ids=digit_ids,
        eq_token_ids=eq_ids,
        n_loops=n_loops,
        fixed_len=fixed_len,
        calibration_weight=calibration_weight,
        use_jit=use_jit,
    )
    Device[Device.DEFAULT].synchronize()
    return info


def main():
    CKPT = os.environ.get("CKPT", "/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v1_step1500.safetensors")
    BATCH = int(os.environ.get("BATCH", "4"))
    N_LOOPS = int(os.environ.get("N_LOOPS", "4"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "96"))
    CALIB_W = float(os.environ.get("CALIB_W", "0.1"))

    print(f"=== JIT vs eager equivalence check ===")
    print(f"  ckpt: {CKPT}")
    print(f"  BATCH={BATCH}  N_LOOPS={N_LOOPS}  FIXED_LEN={FIXED_LEN}  CALIB_W={CALIB_W}")
    print()

    tok = load_tokenizer()
    digit_ids = digit_token_ids_for(tok)
    eq_ids = eq_token_ids_for(tok)

    # Generate the same batch deterministically
    examples = generate_math("L4_MIXED", num_problems=200, seed=42, digit_spacing=True)
    train, _ = split_train_eval(examples, n_eval=20, seed=42)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(train), size=BATCH)
    batch = [train[int(i)] for i in idx]

    # === Eager run ===
    print(">>> Eager (use_jit=False)")
    model_e, _ = build_model_and_load(CKPT)
    info_e = run_one_step(model_e, batch, tok, digit_ids, eq_ids,
                           N_LOOPS, FIXED_LEN, CALIB_W, use_jit=False)
    print(f"  loss            = {info_e['loss']:.6f}")
    print(f"  answer_ce       = {info_e['answer_ce']:.6f}")
    print(f"  calib_bce       = {info_e['calib_bce']:.6f}")
    print(f"  correct_per_breath = {info_e['correct_per_breath']}")
    print(f"  n_correct       = {info_e['n_correct']}, n_wrong = {info_e['n_wrong']}")
    print(f"  mean_conf[+]    = {info_e['mean_conf_correct']:.6f}")
    print(f"  mean_conf[-]    = {info_e['mean_conf_wrong']:.6f}")
    del model_e

    print()
    print(">>> JIT (use_jit=True)")
    model_j, _ = build_model_and_load(CKPT)
    info_j = run_one_step(model_j, batch, tok, digit_ids, eq_ids,
                           N_LOOPS, FIXED_LEN, CALIB_W, use_jit=True)
    print(f"  loss            = {info_j['loss']:.6f}")
    print(f"  answer_ce       = {info_j['answer_ce']:.6f}")
    print(f"  calib_bce       = {info_j['calib_bce']:.6f}")
    print(f"  correct_per_breath = {info_j['correct_per_breath']}")
    print(f"  n_correct       = {info_j['n_correct']}, n_wrong = {info_j['n_wrong']}")
    print(f"  mean_conf[+]    = {info_j['mean_conf_correct']:.6f}")
    print(f"  mean_conf[-]    = {info_j['mean_conf_wrong']:.6f}")
    del model_j

    # Compare
    print()
    print("=== Deltas (eager - JIT) ===")
    def _delta(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b)
        return f"{a} vs {b}"
    print(f"  loss      delta = {_delta(info_e['loss'], info_j['loss']):.2e}")
    print(f"  ans_ce    delta = {_delta(info_e['answer_ce'], info_j['answer_ce']):.2e}")
    print(f"  calib_bce delta = {_delta(info_e['calib_bce'], info_j['calib_bce']):.2e}")
    print(f"  n_correct delta = {info_e['n_correct'] - info_j['n_correct']}")
    print(f"  cpb eager = {info_e['correct_per_breath']}")
    print(f"  cpb JIT   = {info_j['correct_per_breath']}")

    # Verdict
    max_loss_delta = max(_delta(info_e[k], info_j[k]) for k in ("loss", "answer_ce", "calib_bce"))
    if max_loss_delta < 1e-3:
        print()
        print(f"  ✓ MATCH (max loss delta {max_loss_delta:.2e} < 1e-3)")
    elif max_loss_delta < 1e-2:
        print()
        print(f"  ~ CLOSE (max loss delta {max_loss_delta:.2e}, likely fp32 precision)")
    else:
        print()
        print(f"  ✗ MISMATCH (max loss delta {max_loss_delta:.2e})")


if __name__ == "__main__":
    main()
