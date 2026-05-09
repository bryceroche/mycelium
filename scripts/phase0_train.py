"""Phase 0: Loop Consistency Training.

Teaches the breathing transformer to produce coherent generation across multiple
loop depths. Pythia-init layers can do single-pass generation but degrade when
looped (we proved this with frozen weights -> "had had had"). Fine-tuning teaches
the layers to be productive at any depth.

Training: random loop count from {1, 2, 4} per batch, next-token CE on the post-
final-LN integrated representation. No tokens generated mid-breath.

Eval: loss at fixed loop counts {1, 2, 4, 8}, plus greedy generation samples.
Success: coherent input-dependent text at 4 loops.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer, load_wikitext, sample_batch
from mycelium.training import train_step, eval_loss, sample_text


def cast_model_fp32(model):
    """Cast all trainable weights from FP16 to FP32 (avoids AdamW v_hat
    underflow, see yesterday's debugging). Mixed precision is a future project."""
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


def named_state(model):
    """Returns dict of weight name -> Tensor for safe_save."""
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


def main():
    cfg = Config()
    BATCH = getenv("BATCH", 8)
    SEQ = getenv("SEQ", 256)
    STEPS = getenv("STEPS", 50)
    LR = float(getenv("LR", "3e-5"))
    EVAL_EVERY = getenv("EVAL_EVERY", 25)
    SAMPLE_EVERY = getenv("SAMPLE_EVERY", 50)
    TRAIN_LOOPS = [int(x) for x in getenv("TRAIN_LOOPS", "1,2,4").split(",")]
    EVAL_LOOPS = [int(x) for x in getenv("EVAL_LOOPS", "1,2,4,8").split(",")]
    SEED = getenv("SEED", 42)

    print(f"=== Phase 0 ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  seq={SEQ}  steps={STEPS}  lr={LR}")
    print(f"train_loops={TRAIN_LOOPS}  eval_loops={EVAL_LOOPS}")
    print()

    tok = load_tokenizer()
    print("tokenizing wikitext-2...")
    train_ids = load_wikitext(tok, "train")
    val_ids = load_wikitext(tok, "validation")
    print(f"  train tokens: {len(train_ids):,}")
    print(f"  val tokens:   {len(val_ids):,}")
    print()

    print("loading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    n_params = sum(int(np.prod(t.shape)) for t in collect_params(model))
    print(f"  trainable params: {n_params/1e6:.1f}M ({n_params:,})")
    del sd  # release the safetensors mmap
    print()

    params = collect_params(model)
    opt = AdamW(params, lr=LR)
    Tensor.training = True

    rng = np.random.default_rng(SEED)
    py_rng = np.random.default_rng(SEED + 1)  # for loop count selection + sampling

    # Pre-build eval/sample fixtures
    eval_batch = sample_batch(val_ids, BATCH, SEQ, np.random.default_rng(SEED + 100))
    # Two prompts: an OOD pangram (stress test) and an in-distribution wikitext heading
    sample_prompts = {
        "ood": " The quick brown fox",
        "wiki": " = History = \n\n In",
    }
    sample_prompt_ids = {k: tok.encode(p).ids for k, p in sample_prompts.items()}
    for k, ids in sample_prompt_ids.items():
        print(f"sample prompt[{k}]: {sample_prompts[k]!r}  ({len(ids)} tokens)")
    print()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, ".cache", "phase0_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    t_start = time.perf_counter()
    losses = []
    for step in range(STEPS):
        n_loops = int(py_rng.choice(TRAIN_LOOPS))
        batch = sample_batch(train_ids, BATCH, SEQ, rng)

        t0 = time.perf_counter()
        loss = train_step(model, opt, batch, n_loops)
        dt = time.perf_counter() - t0
        losses.append(loss)
        elapsed = time.perf_counter() - t_start
        print(f"step {step:4d}  loops={n_loops}  loss={loss:.4f}  ({dt:.2f}s, total {elapsed:.0f}s)", flush=True)

        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            print(f"  --- eval at step {step+1} ---")
            for nl in EVAL_LOOPS:
                el = eval_loss(model, eval_batch, nl)
                print(f"    val loss @ {nl} loops: {el:.4f}")
            Tensor.training = True

        if (step + 1) % SAMPLE_EVERY == 0 or step + 1 == STEPS:
            print(f"  --- generation samples at step {step+1} ---")
            for prompt_name, prompt_ids in sample_prompt_ids.items():
                prompt_text = sample_prompts[prompt_name]
                # greedy
                for nl in EVAL_LOOPS:
                    gen = sample_text(model, prompt_ids, n_new_tokens=20,
                                      n_loops=nl, temperature=0.0)
                    print(f"    [{prompt_name} greedy @ {nl}]: {prompt_text + tok.decode(gen)!r}")
                # temperature=0.7 for the OOD prompt only (cheaper)
                if prompt_name == "ood":
                    gen = sample_text(model, prompt_ids, n_new_tokens=20,
                                      n_loops=4, temperature=0.7,
                                      rng=np.random.default_rng(SEED + step))
                    print(f"    [{prompt_name} T=0.7 @ 4]:  {prompt_text + tok.decode(gen)!r}")
            Tensor.training = True
            print()

    total = time.perf_counter() - t_start
    print(f"\n=== done. {STEPS} steps in {total:.0f}s ({total/STEPS:.1f}s/step). final loss: {losses[-1]:.4f} ===")

    ckpt_path = os.path.join(ckpt_dir, f"phase0_step{STEPS}.safetensors")
    safe_save(named_state(model), ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
