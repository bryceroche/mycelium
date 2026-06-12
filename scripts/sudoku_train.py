"""v98 sudoku training driver.

Iterative-prefill breathing transformer on 9x9 sudoku. Standalone — does NOT
share the train loop with l3_train.py (different task, different loss). Reuses:
  - BreathingTransformer's L0-L3 attention/FFN weights (Pythia-410M init)
  - The model's state_dict / load_state_dict / AdamW pipeline

Env vars (the smoke launcher sets these — see scripts/v98_sudoku_smoke.sh):
  SUDOKU_TASK=1                 enable sudoku-specific params + forward
  SUDOKU_K_MAX=30               number of iterative-prefill breaths
  SUDOKU_CONSTRAINT_WEIGHT=0.3
  SUDOKU_CALIB_WEIGHT=0.1
  SUDOKU_DIFFICULTY_FILTER=easy filter training data to one band (smoke mode)
  SUDOKU_TRAIN=.cache/sudoku_train.jsonl
  SUDOKU_VAL=.cache/sudoku_val.jsonl
  BATCH=8
  STEPS=1000
  LR=3e-5
  CKPT_EVERY=200
  CKPT_LABEL=v98_smoke
  RESUME_FROM=...               warm-start from a saved sudoku ckpt
  PYTHIA_INIT=1                 use Pythia weights for L0-L3 (default 1); set
                                to 0 to train from random.
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.sudoku import (
    SUDOKU_K_MAX, SUDOKU_CONSTRAINT_WEIGHT, SUDOKU_CALIB_WEIGHT,
    attach_sudoku_params, sudoku_parameters, sudoku_state_dict,
    sudoku_breathing_forward, sudoku_loss, per_breath_ce, sudoku_accuracy,
    _compile_jit_sudoku_step, _compile_jit_sudoku_eval,
)
from mycelium.sudoku_data import SudokuLoader


def cast_layers_fp32(model):
    """Cast the L0-L3 layer + shared weights from fp16 to fp32 for stable training.
    Mirrors the cast_model_fp32 logic in scripts/l3_train.py but only for the
    components sudoku actually uses.
    """
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    # Embedding (we don't use it but the loader assigned to it; harmless to cast)
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def collect_sudoku_params(model) -> list[Tensor]:
    """Trainable parameters for sudoku training: shared L0-L3 attn/FFN +
    sudoku-specific embeddings/codebook/calib.

    Skips: token embedding (model.embed.weight), embed_out, lookup_table,
    controllers, notebooks — none of those are touched by the sudoku forward.
    """
    params: list[Tensor] = []
    # Shared L0-L3 weights (Pythia-init transformer that does the breathing).
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    # Final LN (used by the sudoku readout)
    params += [model.ln_f_g, model.ln_f_b]
    # Sudoku-specific (state_embed, position_embed, digit_codebook, calib_head)
    params += sudoku_parameters(model)
    return params


def model_state_dict_sudoku(model) -> dict:
    """Save only what sudoku needs: shared + L0-L3 phase weights + final LN +
    sudoku params. Excludes embed/embed_out/lookup_table/controllers/etc.
    Compact ckpt — round-trips through this script only.
    """
    sd = {
        "ln_f.g": model.ln_f_g,
        "ln_f.b": model.ln_f_b,
    }
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(sudoku_state_dict(model))
    return sd


def load_ckpt(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_sudoku(model)
    missing = []
    partial = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            # Partial-row copy for the per-breath tensors when K_new > K_old:
            # copy rows 0..K_old-1 from src into dst, leave the trailing rows at init.
            # This preserves the trained breath markers and lets new breaths (K_new-1)
            # explore from a fresh orthogonal init.
            if (name in ("sudoku.breath_embed", "sudoku.delta_gate")
                    and src.ndim == dst.ndim
                    and (src.ndim == 1 or src.shape[1:] == dst.shape[1:])
                    and src.shape[0] <= dst.shape[0]):
                k_old = int(src.shape[0])
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                # Build new tensor: rows 0..k_old-1 from src, rest from dst's current init.
                cur = dst.numpy()
                src_np = src.numpy()
                cur[:k_old] = src_np[:k_old]
                from tinygrad import Tensor as _T
                dst.assign(_T(cur, dtype=dst.dtype, device=dst.device).contiguous()).realize()
                partial.append(f"{name} ({k_old}/{dst.shape[0]} rows)")
                continue
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} sudoku keys (kept init): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if partial:
        print(f"  ckpt partial load for {len(partial)} keys (trailing rows kept at init): {partial}")


def evaluate(model, loader: SudokuLoader, K: int, max_batches: int = 50,
             label: str = "val", eval_fn=None) -> dict:
    """Run forward on `max_batches` × batch_size puzzles, return per-difficulty
    cell + puzzle accuracy + constraint violations.

    If `eval_fn` is provided (TinyJit'd eval step), use it for forward+eq fusion.
    Otherwise fall back to the eager sudoku_breathing_forward path (slower; only
    used the first call before the eval JIT is warmed up).
    """
    Tensor.training = False
    n_seen = 0
    # Aggregate per-difficulty: { diff: {cell_eq, n_cells, puzzle_eq, n_puzzles} }
    agg = {}
    n_batches = 0
    for input_cells, gold, picks in loader.iter_eval(batch_size=loader.batch_size):
        if eval_fn is not None:
            eq, _cell_acc, _puzzle_acc = eval_fn(input_cells, gold)
            eq_np = eq.numpy()
        else:
            cell_logits_history, _ = sudoku_breathing_forward(model, input_cells, K=K)
            final_logits = cell_logits_history[-1]
            pred = final_logits.argmax(axis=-1) + 1  # (B, 81)
            eq = (pred == gold).cast(dtypes.float)   # (B, 81)
            eq_np = eq.realize().numpy()
        # Per-puzzle
        for b, rec in enumerate(picks):
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"cell_eq": 0.0, "n_cells": 0,
                             "puzzle_eq": 0, "n_puzzles": 0,
                             "n_givens_sum": 0}
            agg[diff]["cell_eq"] += float(eq_np[b].sum())
            agg[diff]["n_cells"] += 81
            agg[diff]["puzzle_eq"] += int(eq_np[b].prod())
            agg[diff]["n_puzzles"] += 1
            agg[diff]["n_givens_sum"] += int(rec.get("n_givens", 0))
        n_seen += len(picks)
        n_batches += 1
        if n_batches >= max_batches:
            break

    out = {}
    for d, v in agg.items():
        if v["n_puzzles"] == 0:
            continue
        out[d] = {
            "cell_acc": v["cell_eq"] / v["n_cells"],
            "puzzle_acc": v["puzzle_eq"] / v["n_puzzles"],
            "n_puzzles": v["n_puzzles"],
            "avg_n_givens": v["n_givens_sum"] / v["n_puzzles"],
        }
    Tensor.training = True
    return out


def main():
    SUDOKU_TASK_LOCAL = int(getenv("SUDOKU_TASK", 0)) > 0
    assert SUDOKU_TASK_LOCAL, "SUDOKU_TASK=1 must be set"

    K = int(getenv("SUDOKU_K_MAX", str(SUDOKU_K_MAX)))
    BATCH = int(getenv("BATCH", 8))
    STEPS = int(getenv("STEPS", 1000))
    LR = float(getenv("LR", "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY", 200))
    EVAL_EVERY = int(getenv("EVAL_EVERY", 100))
    LOG_EVERY = int(getenv("LOG_EVERY", 10))
    PER_BREATH_CE_EVERY = int(getenv("PER_BREATH_CE_EVERY", 50))
    GC_EVERY = int(getenv("GC_EVERY", 50))
    CKPT_LABEL = getenv("CKPT_LABEL", "v98_smoke")
    RESUME_FROM = getenv("RESUME_FROM", "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT", 1)) > 0
    SEED = int(getenv("SEED", 42))
    DIFFICULTY_FILTER = os.environ.get("SUDOKU_DIFFICULTY_FILTER", "easy").strip() or None
    CURRICULUM = int(getenv("SUDOKU_CURRICULUM", 0)) > 0
    CURRICULUM_ANNEAL = int(getenv("SUDOKU_CURRICULUM_ANNEAL_STEPS", 500))

    SUDOKU_TRAIN = getenv("SUDOKU_TRAIN", ".cache/sudoku_train.jsonl")
    SUDOKU_VAL = getenv("SUDOKU_VAL", ".cache/sudoku_val.jsonl")
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", 20))
    EVAL_BATCH = int(getenv("EVAL_BATCH", BATCH))

    CONSTRAINT_WEIGHT = float(getenv("SUDOKU_CONSTRAINT_WEIGHT",
                                     str(SUDOKU_CONSTRAINT_WEIGHT)))
    CALIB_WEIGHT = float(getenv("SUDOKU_CALIB_WEIGHT", str(SUDOKU_CALIB_WEIGHT)))

    print(f"=== v98 sudoku training ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"constraint_weight={CONSTRAINT_WEIGHT}  calib_weight={CALIB_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train_path={SUDOKU_TRAIN}  val_path={SUDOKU_VAL}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- model
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)
    # Attach sudoku-specific params (state_embed, position_embed, codebook, calib, attn_bias)
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
    Device[Device.DEFAULT].synchronize()
    params = collect_sudoku_params(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"resuming from sudoku ckpt: {RESUME_FROM}")
        load_ckpt(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    # ---- data
    train_loader = SudokuLoader(SUDOKU_TRAIN,
                                 batch_size=BATCH,
                                 difficulty_filter=DIFFICULTY_FILTER,
                                 curriculum=CURRICULUM,
                                 curriculum_anneal_steps=CURRICULUM_ANNEAL,
                                 seed=SEED)
    val_loader = SudokuLoader(SUDOKU_VAL,
                               batch_size=EVAL_BATCH,
                               difficulty_filter=None,
                               curriculum=False,
                               seed=SEED + 1)

    # ---- ckpt dir
    ckpt_dir = ".cache/sudoku_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Photon mechanism (env-driven, see mycelium/sudoku.py for design)
    PHOTON_ENABLE   = int(os.environ.get("SUDOKU_PHOTON_ENABLE", "0")) > 0
    PHOTON_ALPHA    = float(os.environ.get("SUDOKU_PHOTON_ALPHA", "0.0"))
    PHOTON_FREQ     = float(os.environ.get("SUDOKU_PHOTON_FREQ_MULT", "1.0"))
    PHOTON_ROT_RAMP = int(os.environ.get("SUDOKU_PHOTON_ROT_RAMP_STEPS", "100"))
    if PHOTON_ENABLE:
        print(f"  photon: ENABLE alpha={PHOTON_ALPHA} freq_mult={PHOTON_FREQ} "
              f"rot_ramp_steps={PHOTON_ROT_RAMP}")
    else:
        print(f"  photon: disabled (alpha forced to 0)")
        PHOTON_ALPHA = 0.0

    # ---- JIT compile the train + eval steps (first call inside the loop will
    # block ~60-90s for compile, then ~1-2s/step steady state).
    Tensor.training = True
    step_fn = _compile_jit_sudoku_step(
        model, opt, K=K, B=BATCH,
        constraint_weight=CONSTRAINT_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        grad_clip=0.0,
        photon_alpha=PHOTON_ALPHA,
        photon_freq_mult=PHOTON_FREQ,
    )
    # Eval JIT (separate cache key) — built lazily on first eval if EVAL_BATCH==BATCH
    # so we share the warm KV / weight layouts. If EVAL_BATCH != BATCH, the eval
    # JIT compiles its own graph.
    eval_fn = _compile_jit_sudoku_eval(
        model, K=K, B=EVAL_BATCH,
        photon_alpha=PHOTON_ALPHA, photon_freq_mult=PHOTON_FREQ,
    )

    # ---- pre-step-0 val: measures the freq-dependent "yank" from full
    # rotation perturbation at warm-start, BEFORE any training. Diagnoses
    # the rotation-confound: if yank scales with freq, lower-freq results
    # may "win" the sweep just by being a gentler perturbation. Sets
    # rot_scale=1.0 (full rotation) for the diagnostic, then training
    # restarts the ramp from 0 → 1 over RAMP_STEPS steps.
    if PHOTON_ENABLE and PHOTON_ALPHA > 0:
        if hasattr(model, "sudoku_photon_rot_scale"):
            model.sudoku_photon_rot_scale.assign(
                Tensor(np.array(1.0, dtype=np.float32), dtype=dtypes.float)
            ).realize()
        print(f"  pre-step-0 val (rot_scale=1.0 full rotation; "
              f"measures freq-dependent yank size)...", flush=True)
        Tensor.training = False
        pre_results = evaluate(model, val_loader, K=K, max_batches=EVAL_BATCHES,
                                label="pre-step-0", eval_fn=eval_fn)
        Tensor.training = True
        for d in ["easy", "medium", "hard", "expert"]:
            if d not in pre_results:
                continue
            v = pre_results[d]
            print(f"  pre[{d:7s}]: cell_acc={v['cell_acc']:.3f} "
                  f"puzzle_acc={v['puzzle_acc']:.3f} n={v['n_puzzles']}",
                  flush=True)

    # ---- train loop
    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss_sum = 0.0
    log_cell_ce_sum = 0.0
    log_energy_sum = 0.0
    log_calib_sum = 0.0
    log_n = 0

    for step in range(1, STEPS + 1):
        # Per-step rotation alpha ramp: 0 → 1 over PHOTON_ROT_RAMP steps.
        # Disabled (no-op) when PHOTON_ALPHA=0 or no rot_scale attribute.
        if (PHOTON_ENABLE and PHOTON_ALPHA > 0
                and hasattr(model, "sudoku_photon_rot_scale")):
            if PHOTON_ROT_RAMP > 0:
                ramp_val = min(float(step) / float(PHOTON_ROT_RAMP), 1.0)
            else:
                ramp_val = 1.0
            model.sudoku_photon_rot_scale.assign(
                Tensor(np.array(ramp_val, dtype=np.float32), dtype=dtypes.float)
            ).realize()

        input_cells, gold, _picks = train_loader.sample_batch(step=step)

        # JIT'd train step — single fused graph; replays at ~1-2s/step after warm-up.
        # Returns: (total, healthy, cell_ce, energy, calib, cell_acc, puzzle_acc, *pb_ce, *pb_calib)
        outs = step_fn(input_cells, gold)
        total_t   = outs[0]
        healthy_t = outs[1]
        cell_ce_t = outs[2]
        energy_t  = outs[3]
        calib_t   = outs[4]
        cell_acc_t   = outs[5]
        puzzle_acc_t = outs[6]
        pb_ce_ts     = outs[7:7 + K]
        pb_calib_ts  = outs[7 + K:7 + 2 * K]

        # Cheap readback: tinygrad scalars realized inside JIT — .numpy() is sync-only.
        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: NaN grad detected — step skipped", flush=True)

        # Log accumulators
        log_loss_sum += float(total_t.numpy())
        log_cell_ce_sum += float(cell_ce_t.numpy())
        log_energy_sum += float(energy_t.numpy())
        log_calib_sum += float(calib_t.numpy())
        log_n += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            avg_loss = log_loss_sum / log_n
            avg_ce = log_cell_ce_sum / log_n
            avg_energy = log_energy_sum / log_n
            avg_calib = log_calib_sum / log_n
            print(f"[step {step:5d}] loss={avg_loss:.4f} cell_ce={avg_ce:.4f} "
                  f"energy={avg_energy:.4f} calib={avg_calib:.4f}  "
                  f"({dt:.1f}s, {dt/step:.2f}s/step)", flush=True)
            log_loss_sum = log_cell_ce_sum = log_energy_sum = log_calib_sum = 0.0
            log_n = 0

        if step % PER_BREATH_CE_EVERY == 0:
            # Per-breath CE is already computed inside the JIT step — just read back.
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_calib = [float(t.numpy()) for t in pb_calib_ts]
            # Compact: show first 4 and last 4 of K (or all if K ≤ 8)
            if K <= 8:
                pb_ce_str    = " ".join(f"{v:.2f}" for v in pb_ce)
                pb_calib_str = " ".join(f"{v:.2f}" for v in pb_calib)
            else:
                pb_ce_str = (
                    " ".join(f"{v:.2f}" for v in pb_ce[:4]) + " ... "
                    + " ".join(f"{v:.2f}" for v in pb_ce[-4:])
                )
                pb_calib_str = (
                    " ".join(f"{v:.2f}" for v in pb_calib[:4]) + " ... "
                    + " ".join(f"{v:.2f}" for v in pb_calib[-4:])
                )
            # Also report train batch accuracy from the JIT'd scalars (no extra forward).
            cell_acc = float(cell_acc_t.numpy())
            puzzle_acc = float(puzzle_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]:    {pb_ce_str}  "
                  f"(train cell_acc={cell_acc:.3f} puzzle_acc={puzzle_acc:.3f})",
                  flush=True)
            print(f"  per_breath_calib[B0..B{K-1}]: {pb_calib_str}",
                  flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating on val ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate(model, val_loader, K=K, max_batches=EVAL_BATCHES,
                               label="val", eval_fn=eval_fn)
            for d in ["easy", "medium", "hard", "expert"]:
                if d not in results:
                    continue
                v = results[d]
                print(f"  val[{d:7s}]: cell_acc={v['cell_acc']:.3f} "
                      f"puzzle_acc={v['puzzle_acc']:.3f} "
                      f"n={v['n_puzzles']} avg_G={v['avg_n_givens']:.1f}", flush=True)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_sudoku(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final ckpt
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_sudoku(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
