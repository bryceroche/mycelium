"""Math curriculum training (L3 / L4 / L4.5) for the breathing transformer.

Standard mixed-loops training (random {1,2,4}), masked-loss CE on the answer
span only. Periodic accuracy evaluation at fixed loop counts {1,2,4,8} on a
held-out set.

Select the curriculum level with the LEVEL env var (default L3). FIXED_LEN
defaults to a level-appropriate value (L3=64, L4=96, L4.5=160). Checkpoints
land in .cache/{level_lower}_ckpts/.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval
from mycelium.l3_training import (
    multi_cycle_train_step, multi_cycle_eval_loss, accuracy_at_loops_multi,
)
from mycelium.lookup_eval import lookup_eval


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
    return nps


def load_checkpoint(model, path: str):
    """Load a safetensors checkpoint produced by named_state(). Tensor identity is
    preserved via .assign() so the optimizer + autograd wiring on `model` stays
    valid (we keep the same parameter Tensors, just overwrite their values).
    """
    sd = safe_load(path)
    targets = named_state(model)
    missing = set(targets) - set(sd)
    extra = set(sd) - set(targets)
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {sorted(missing)[:5]}")
    if extra:
        print(f"  (ignoring extra ckpt keys: {sorted(extra)[:5]})")
    for name, dst in targets.items():
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            src = src.reshape(dst.shape)
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    Device[Device.DEFAULT].synchronize()


def named_state(model):
    sd = {
        "embed.weight": model.embed.weight,
        "embed_out": model.embed_out,
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
    return sd


DEFAULT_FIXED_LEN = {"ARITH": 32, "L3": 64, "L4": 96, "L4.5": 160}


def main():
    cfg = Config()
    LEVEL = getenv("LEVEL", "L3")
    BATCH = getenv("BATCH", 16)
    FIXED_LEN = getenv("FIXED_LEN", DEFAULT_FIXED_LEN.get(LEVEL, 96))
    STEPS = getenv("STEPS", 500)
    LR = float(getenv("LR", "3e-5"))
    LOSS_EVAL_EVERY = getenv("LOSS_EVAL_EVERY", 100)
    ACC_EVAL_EVERY = getenv("ACC_EVAL_EVERY", 250)
    CKPT_EVERY = getenv("CKPT_EVERY", 250)
    NUM_PROBLEMS = getenv("NUM_PROBLEMS", 20000)
    NUM_EVAL = getenv("NUM_EVAL", 100)
    NUM_VAL_BATCHES = getenv("NUM_VAL_BATCHES", 4)
    TRAIN_LOOPS = [int(x) for x in getenv("TRAIN_LOOPS", "1,2,4").split(",")]   # Phase A choices
    EVAL_LOOPS = [int(x) for x in getenv("EVAL_LOOPS", "1,2,4,8").split(",")]   # Phase A test points
    PHASE_C_LOOPS = getenv("PHASE_C_LOOPS", 1)                                  # light breathing for execution cycles
    SEED = getenv("SEED", 42)
    RESUME_FROM = getenv("RESUME_FROM", "")
    SPACE_DIGITS = bool(getenv("SPACE_DIGITS", 0))   # digit-by-digit tokenization for arithmetic
    EVAL_BATCH = getenv("EVAL_BATCH", 64)            # batched accuracy eval (kept fixed → JIT reuse)
    # K/V cache length for eval. Defaults to FIXED_LEN (matches the level's training sequence length).
    # Override only if you know your prompts + max_new are smaller (e.g., 32 for pure ARITH).
    EVAL_CACHE_LEN = getenv("EVAL_CACHE_LEN", 0) or FIXED_LEN
    LOOKUP_EVAL = getenv("LOOKUP_EVAL", 1)           # 1 = run per-checkpoint lookup-table classification eval
    LOOKUP_EVAL_LOOPS = getenv("LOOKUP_EVAL_LOOPS", 8)  # n_loops for the lookup eval (single value)

    print(f"=== Math training — level {LEVEL} (three-phase: heavy A, light C) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  seq_len={FIXED_LEN}  steps={STEPS}  lr={LR}")
    print(f"corpus={NUM_PROBLEMS}, eval set={NUM_EVAL}, space_digits={SPACE_DIGITS}")
    print(f"phase_A_train_loops={TRAIN_LOOPS}  phase_A_eval_loops={EVAL_LOOPS}  phase_C_loops={PHASE_C_LOOPS}")
    print(f"eval batch={EVAL_BATCH}  cache_len={EVAL_CACHE_LEN}  lookup_eval={'on' if LOOKUP_EVAL else 'off'}@A={LOOKUP_EVAL_LOOPS}")
    print()

    print(f"generating {LEVEL} problems...")
    t0 = time.perf_counter()
    all_examples = generate_math(LEVEL, NUM_PROBLEMS, seed=SEED, digit_spacing=SPACE_DIGITS)
    train_examples, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=SEED)
    print(f"  train={len(train_examples)}  eval={len(eval_examples)}  ({time.perf_counter()-t0:.1f}s)")
    if SPACE_DIGITS:
        ex0 = train_examples[0]
        print(f"  sample (digit-spaced): {ex0.problem!r} -> {ex0.gen!r}")

    tok = load_tokenizer()

    print("\nloading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    n_params = sum(int(np.prod(t.shape)) for t in collect_params(model))
    print(f"  trainable params: {n_params/1e6:.1f}M")
    del sd

    if RESUME_FROM:
        print(f"\nresuming from checkpoint: {RESUME_FROM}")
        load_checkpoint(model, RESUME_FROM)
        print("  loaded.")

    params = collect_params(model)
    opt = AdamW(params, lr=LR)
    Tensor.training = True

    rng = np.random.default_rng(SEED)
    py_rng = np.random.default_rng(SEED + 1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_label_default = LEVEL.lower().replace(".", "_") + ("_spaced" if SPACE_DIGITS else "_abs")
    ckpt_label = getenv("CKPT_LABEL", ckpt_label_default)
    # ckpt_dir uses just the level prefix so spaced + abs share a directory
    ckpt_dir = os.path.join(project_root, ".cache", f"{LEVEL.lower().replace('.', '_')}_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # We don't pre-tokenize for multi-cycle — the encoder is called per-batch.
    # For a small corpus (L3 ~20K examples) this is fine.
    print()

    t_start = time.perf_counter()
    for step in range(STEPS):
        # Three-phase scheduling: cycle 0 (Phase A) gets heavy breathing,
        # subsequent cycles (Phase C) get light. The list is padded to actual cycle count
        # inside multi_cycle_train_step.
        phase_a_loops = int(py_rng.choice(TRAIN_LOOPS))
        loops_per_cycle = [phase_a_loops, PHASE_C_LOOPS]
        idx = rng.integers(0, len(train_examples), size=BATCH)
        batch_examples = [train_examples[i] for i in idx]

        t0 = time.perf_counter()
        loss = multi_cycle_train_step(model, opt, batch_examples, tok, loops_per_cycle, FIXED_LEN)
        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start

        if step % 10 == 0 or step + 1 == STEPS:
            print(f"step {step:4d}  A={phase_a_loops} C={PHASE_C_LOOPS}  loss={loss:.4f}  ({dt:.2f}s, total {elapsed:.0f}s)", flush=True)

        # Cheap loss eval — Phase A loops vary, Phase C fixed
        if (step + 1) % LOSS_EVAL_EVERY == 0 or step + 1 == STEPS:
            print(f"  --- loss eval at step {step+1} ---")
            for nl in EVAL_LOOPS:
                losses = []
                for _ in range(NUM_VAL_BATCHES):
                    eidx = rng.integers(0, len(eval_examples), size=BATCH)
                    eb = [eval_examples[i] for i in eidx]
                    losses.append(multi_cycle_eval_loss(model, eb, tok,
                                                       [nl, PHASE_C_LOOPS], FIXED_LEN))
                print(f"    val loss @ A={nl} C={PHASE_C_LOOPS}: {np.mean(losses):.4f}  (+-{np.std(losses):.3f})")
            Tensor.training = True

        # Expensive accuracy eval — same scheduling: Phase A varies, Phase C fixed
        if (step + 1) % ACC_EVAL_EVERY == 0 or step + 1 == STEPS:
            print(f"  --- accuracy at step {step+1} (multi-cycle, {NUM_EVAL} held-out) ---")
            for nl in EVAL_LOOPS:
                t0 = time.perf_counter()
                acc, rows = accuracy_at_loops_multi(model, tok, eval_examples,
                                                    n_loops=[nl, PHASE_C_LOOPS],
                                                    batch_size=EVAL_BATCH,
                                                    cache_max_len=EVAL_CACHE_LEN)
                gt = time.perf_counter() - t0
                print(f"    acc @ A={nl} C={PHASE_C_LOOPS}: {acc*100:.1f}%  ({gt:.1f}s)")
                if step + 1 == STEPS:
                    for ex, parsed, gen in rows[:2]:
                        print(f"      Q: {ex.problem}")
                        print(f"      gen: {gen.strip()!r}")
                        print(f"      parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
            # Per-checkpoint lookup-table eval — second axis of training signal.
            # Trains a fresh 16x1024 cosine-similarity table on op classification
            # from the integrated rep at "=" position. Reports held-out classification
            # accuracy + on-target count (out of 4) + per-op purity.
            if LOOKUP_EVAL:
                m = lookup_eval(model, tok, n_loops=int(LOOKUP_EVAL_LOOPS), verbose=False)
                pur = " ".join(f"{o}={m['purity_per_op'].get(o, 0):.2f}" for o in ["+","-","*","/"])
                print(f"    lookup-eval @ A={m['n_loops']}: trained={m['trained_acc']*100:.1f}%  "
                      f"ncm={m['ncm_acc']*100:.1f}%  on-target={m['on_target_count']}/4  "
                      f"purity[{pur}]  ({m['elapsed_s']:.1f}s)")
            Tensor.training = True

        # Periodic checkpoint
        if (step + 1) % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_label}_step{step+1}.safetensors")
            safe_save(named_state(model), ckpt_path)
            print(f"  saved: {ckpt_path}")
            print()

    total = time.perf_counter() - t_start
    print(f"\n=== done. {STEPS} steps in {total:.0f}s ===")

    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_label}_step{STEPS}.safetensors")
    safe_save(named_state(model), ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
