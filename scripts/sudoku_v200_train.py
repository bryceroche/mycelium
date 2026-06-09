"""v200-Sudoku training driver.

Iterative-prefill breathing transformer on 9×9 sudoku. Supports two backbones:
  V200_SUDOKU_BASE=smollm2_1_7b (default) — SmolLM2-1.7B L0-L3, H=2048, 32 heads
  V200_SUDOKU_BASE=pythia_410m            — Pythia-410M L0-L3, H=1024, 16 heads

Same v98 paradigm: K=20 shared passes, per-breath weighted CE + constraint
energy + calibration.

Does NOT use BreathingTransformer. Uses SudokuV200Model (a minimal model shell)
that branches on V200_SUDOKU_BASE to load the appropriate backbone.

Env vars:
  V200_SUDOKU_BASE=smollm2_1_7b     backbone selection (default)
  V200_SUDOKU_TASK=1                enable this script
  V200_SUDOKU_K_MAX=20              number of iterative-prefill breaths
  V200_SUDOKU_CONSTRAINT_WEIGHT=0.3
  V200_SUDOKU_CALIB_WEIGHT=0.1
  SUDOKU_TRAIN=.cache/sudoku_train.jsonl
  SUDOKU_VAL=.cache/sudoku_val.jsonl
  BATCH=8                           training batch size
  STEPS=500                         training steps (smoke: 500; prod: 5000+)
  LR=3e-5
  CKPT_EVERY=200
  EVAL_EVERY=100
  LOG_EVERY=10
  PER_BREATH_CE_EVERY=50
  GC_EVERY=50
  EVAL_BATCHES=20
  EVAL_BATCH=8
  CKPT_LABEL=sudoku_v200_smoke
  RESUME_FROM=                      warm-start from a saved v200-sudoku ckpt
  LLAMA_WEIGHTS=                    override SmolLM2 weights path
  DIFFICULTY_FILTER=easy            filter to one difficulty band (smoke mode)
  SUDOKU_CURRICULUM=0               enable easy→all difficulty curriculum
  SUDOKU_CURRICULUM_ANNEAL_STEPS=500
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
from tinygrad.nn.state import safe_save

from mycelium.sudoku_data import SudokuLoader
from mycelium.sudoku_v200 import (
    V200_SUDOKU_K_MAX,
    V200_SUDOKU_CONSTRAINT_WEIGHT,
    V200_SUDOKU_CALIB_WEIGHT,
    V200_SUDOKU_BASE,
    SudokuV200Model,
    sudoku_breathing_forward_v200_dispatch,
    sudoku_accuracy_v200,
    _compile_jit_sudoku_step_v200,
    _compile_jit_sudoku_eval_v200,
)


def collect_params(model: SudokuV200Model) -> list[Tensor]:
    """All trainable params: SmolLM2 L0-L3 backbone + sudoku-specific.

    Excludes:
      - llama_embed (token embeddings — not used in the sudoku path)
      - llama_rope_cos / llama_rope_sin (precomputed, not trained)
      - sudoku_v200_attn_bias (frozen structural constraint)
    """
    return model.parameters()


def evaluate(model: SudokuV200Model,
             loader: SudokuLoader,
             K: int,
             max_batches: int = 20,
             eval_fn=None) -> dict:
    """Evaluate on val set; return per-difficulty cell + puzzle accuracy."""
    Tensor.training = False
    agg: dict = {}
    n_batches = 0

    for input_cells, gold, picks in loader.iter_eval(batch_size=loader.batch_size):
        if eval_fn is not None:
            eq, _ca, _pa = eval_fn(input_cells, gold)
            eq_np = eq.numpy()
        else:
            cell_logits_history, _ = sudoku_breathing_forward_v200_dispatch(
                model, input_cells, K=K
            )
            final_logits = cell_logits_history[-1]
            pred = final_logits.argmax(axis=-1) + 1
            eq   = (pred == gold).cast(dtypes.float)
            eq_np = eq.realize().numpy()

        for b, rec in enumerate(picks):
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"cell_eq": 0.0, "n_cells": 0,
                             "puzzle_eq": 0, "n_puzzles": 0,
                             "n_givens_sum": 0}
            agg[diff]["cell_eq"]      += float(eq_np[b].sum())
            agg[diff]["n_cells"]      += 81
            agg[diff]["puzzle_eq"]    += int(eq_np[b].prod())
            agg[diff]["n_puzzles"]    += 1
            agg[diff]["n_givens_sum"] += int(rec.get("n_givens", 0))

        n_batches += 1
        if n_batches >= max_batches:
            break

    Tensor.training = True
    out = {}
    for d, v in agg.items():
        if v["n_puzzles"] == 0:
            continue
        out[d] = {
            "cell_acc":    v["cell_eq"] / v["n_cells"],
            "puzzle_acc":  v["puzzle_eq"] / v["n_puzzles"],
            "n_puzzles":   v["n_puzzles"],
            "avg_n_givens": v["n_givens_sum"] / v["n_puzzles"],
        }
    return out


def main():
    assert int(getenv("V200_SUDOKU_TASK", 0)) > 0, "V200_SUDOKU_TASK=1 must be set"

    K               = int(getenv("V200_SUDOKU_K_MAX", str(V200_SUDOKU_K_MAX)))
    BATCH           = int(getenv("BATCH", 8))
    STEPS           = int(getenv("STEPS", 500))
    LR              = float(getenv("LR", "3e-5"))
    CKPT_EVERY      = int(getenv("CKPT_EVERY", 200))
    EVAL_EVERY      = int(getenv("EVAL_EVERY", 100))
    LOG_EVERY       = int(getenv("LOG_EVERY", 10))
    PB_CE_EVERY     = int(getenv("PER_BREATH_CE_EVERY", 50))
    GC_EVERY        = int(getenv("GC_EVERY", 50))
    CKPT_LABEL      = getenv("CKPT_LABEL", "sudoku_v200_smoke")
    RESUME_FROM     = getenv("RESUME_FROM", "")
    SEED            = int(getenv("SEED", 42))

    DIFFICULTY_FILTER = os.environ.get("DIFFICULTY_FILTER", "easy").strip() or None
    CURRICULUM        = int(getenv("SUDOKU_CURRICULUM", 0)) > 0
    CURRICULUM_ANNEAL = int(getenv("SUDOKU_CURRICULUM_ANNEAL_STEPS", 500))

    SUDOKU_TRAIN  = getenv("SUDOKU_TRAIN",  ".cache/sudoku_train.jsonl")
    SUDOKU_VAL    = getenv("SUDOKU_VAL",    ".cache/sudoku_val.jsonl")
    EVAL_BATCHES  = int(getenv("EVAL_BATCHES", 20))
    EVAL_BATCH    = int(getenv("EVAL_BATCH", BATCH))

    CONSTRAINT_WEIGHT = float(getenv(
        "V200_SUDOKU_CONSTRAINT_WEIGHT", str(V200_SUDOKU_CONSTRAINT_WEIGHT)
    ))
    CALIB_WEIGHT = float(getenv(
        "V200_SUDOKU_CALIB_WEIGHT", str(V200_SUDOKU_CALIB_WEIGHT)
    ))

    BASE_MODEL = os.environ.get("V200_SUDOKU_BASE", V200_SUDOKU_BASE).strip()

    print(f"=== v200-Sudoku training ({BASE_MODEL} backbone) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"constraint_weight={CONSTRAINT_WEIGHT}  calib_weight={CALIB_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train_path={SUDOKU_TRAIN}  val_path={SUDOKU_VAL}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- model (backbone L0-L3 + sudoku params) ----
    print(f"loading {BASE_MODEL} L0-L3 weights…")
    model = SudokuV200Model(k_max=K, base_model=BASE_MODEL)
    model.load()
    Device[Device.DEFAULT].synchronize()

    params = collect_params(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  total trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"resuming from ckpt: {RESUME_FROM}")
        model.load_ckpt(RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    # ---- data ----
    train_loader = SudokuLoader(
        SUDOKU_TRAIN, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        seed=SEED,
    )
    val_loader = SudokuLoader(
        SUDOKU_VAL, batch_size=EVAL_BATCH,
        difficulty_filter=None,
        curriculum=False,
        seed=SEED + 1,
    )

    ckpt_dir = ".cache/sudoku_v200_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- JIT compile ----
    Tensor.training = True
    step_fn = _compile_jit_sudoku_step_v200(
        model, opt, K=K, B=BATCH,
        constraint_weight=CONSTRAINT_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        grad_clip=0.0,
    )
    eval_fn = _compile_jit_sudoku_eval_v200(model, K=K, B=EVAL_BATCH)

    # ---- train loop ----
    print("\ntraining…\n")
    t0 = time.time()
    log_loss_sum = log_ce_sum = log_energy_sum = log_calib_sum = 0.0
    log_n = 0

    for step in range(1, STEPS + 1):
        input_cells, gold, _picks = train_loader.sample_batch(step=step)

        outs = step_fn(input_cells, gold)
        total_t      = outs[0]
        healthy_t    = outs[1]
        cell_ce_t    = outs[2]
        energy_t     = outs[3]
        calib_t      = outs[4]
        cell_acc_t   = outs[5]
        puzzle_acc_t = outs[6]
        pb_ce_ts     = outs[7:7 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: NaN detected — step skipped", flush=True)

        log_loss_sum   += float(total_t.numpy())
        log_ce_sum     += float(cell_ce_t.numpy())
        log_energy_sum += float(energy_t.numpy())
        log_calib_sum  += float(calib_t.numpy())
        log_n          += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            avg_loss   = log_loss_sum   / log_n
            avg_ce     = log_ce_sum     / log_n
            avg_energy = log_energy_sum / log_n
            avg_calib  = log_calib_sum  / log_n
            print(
                f"[step {step:5d}] loss={avg_loss:.4f} cell_ce={avg_ce:.4f} "
                f"energy={avg_energy:.4f} calib={avg_calib:.4f}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss_sum = log_ce_sum = log_energy_sum = log_calib_sum = 0.0
            log_n = 0

        if step % PB_CE_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            if K <= 8:
                pb_str = " ".join(f"{v:.2f}" for v in pb_ce)
            else:
                head = " ".join(f"{v:.2f}" for v in pb_ce[:4])
                tail = " ".join(f"{v:.2f}" for v in pb_ce[-4:])
                pb_str = f"{head} ... {tail}"
            cell_acc   = float(cell_acc_t.numpy())
            puzzle_acc = float(puzzle_acc_t.numpy())
            print(
                f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                f"(train cell_acc={cell_acc:.3f} puzzle_acc={puzzle_acc:.3f})",
                flush=True,
            )

        if step % EVAL_EVERY == 0:
            print(
                f"  evaluating on val ({EVAL_BATCHES} batches × B={EVAL_BATCH})…",
                flush=True,
            )
            results = evaluate(model, val_loader, K=K,
                               max_batches=EVAL_BATCHES, eval_fn=eval_fn)
            for d in ["easy", "medium", "hard", "expert"]:
                if d not in results:
                    continue
                v = results[d]
                print(
                    f"  val[{d:7s}]: cell_acc={v['cell_acc']:.3f} "
                    f"puzzle_acc={v['puzzle_acc']:.3f} "
                    f"n={v['n_puzzles']} avg_G={v['avg_n_givens']:.1f}",
                    flush=True,
                )

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final ckpt
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model.state_dict(), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
