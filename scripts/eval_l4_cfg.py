"""Full-answer L4 accuracy eval with per-step classifier-free guidance.

The probe_cfg.py probe found α=3.0 gives +10.8% first-digit accuracy on v6 ARITH_MIXED.
But: does the +10.8% on the first digit translate to full-answer accuracy on L4 v3?

This eval applies CFG at EVERY decode step (slow path — no cached K/V; re-breathes
full context per token). For each problem:
  1. Tokenize "problem ="
  2. Build unconditional context: same length, all PAD
  3. For each decode step:
       conditional logits  = model(cond_ctx, n_loops)[:, -1, :] @ embed_out
       unconditional logits = model(uncond_ctx, n_loops)[:, -1, :] @ embed_out
       guided = cond_logits + α * (cond_logits - uncond_logits)
       next_tok = argmax(guided)
       append to BOTH contexts
       stop on EOS or SEP-sequence
  4. Parse answer, compare to gold.

Slow: ~10× the cached path, since we re-breathe per token. Run on a small N for
fast iteration; scale up if positive.

Usage:
    DEV=PCI+AMD SINE_TEMP=1 SINE_TEMP_MAX=2.0 SINE_TEMP_MIN=0.7 \\
        CKPT=.cache/l4_ckpts/l4_v3_step500.safetensors \\
        N=20 ALPHAS=0,1,3 LOOPS=8 \\
        .venv/bin/python scripts/eval_l4_cfg.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, parse_int_answer, SEP


def cast_fp32(model):
    def _c(o, a):
        t = getattr(o, a)
        if t.dtype == dtypes.half:
            setattr(o, a, t.cast(dtypes.float).contiguous().realize())
    _c(model.embed, "weight"); _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv","bv","wo","bo","w_out","b_out"): _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq","bq","wk","bk","w_in","b_in"): _c(layer, a)


def cfg_generate(model, prompt_ids: list, n_loops: int, alpha: float,
                 sep_ids: list, max_new: int = 80, eos: int = 0) -> list:
    """Per-step CFG greedy generation. Re-breathes full context per token (slow)."""
    ctx_c = list(prompt_ids)
    L0 = len(prompt_ids)
    # Unconditional context: same shape as conditional, all PAD initially
    ctx_u = [0] * L0
    generated = []
    sep_len = len(sep_ids)
    cycle_count = 0  # cycles complete; L4 has 2 cycles; stop after 2.

    for step in range(max_new):
        # Forward conditional
        toks_c = Tensor([ctx_c], dtype=dtypes.int).realize()
        hidden_c = model(toks_c, n_loops)
        last_c = hidden_c[:, -1, :]
        logits_c = (last_c @ model.embed_out).cast(dtypes.float)

        if alpha != 0.0:
            # Forward unconditional
            toks_u = Tensor([ctx_u], dtype=dtypes.int).realize()
            hidden_u = model(toks_u, n_loops)
            last_u = hidden_u[:, -1, :]
            logits_u = (last_u @ model.embed_out).cast(dtypes.float)
            guided = logits_c + alpha * (logits_c - logits_u)
        else:
            guided = logits_c

        next_tok = int(guided.argmax(axis=-1).realize().numpy()[0])
        generated.append(next_tok)
        ctx_c.append(next_tok)
        ctx_u.append(next_tok)

        if next_tok == eos:
            break
        # SEP marks end of a cycle; for L4 we want to continue past one SEP
        if sep_len > 0 and generated[-sep_len:] == sep_ids:
            cycle_count += 1
            if cycle_count >= 2:
                break

    return generated


def main():
    ckpt = getenv("CKPT", ".cache/l4_ckpts/l4_v3_step500.safetensors")
    LEVEL = getenv("LEVEL", "L4")
    N = getenv("N", 20)
    LOOPS = [int(s) for s in getenv("LOOPS", "1,8").split(",")]
    ALPHAS = [float(s) for s in getenv("ALPHAS", "0,1,3").split(",")]
    SEED = getenv("SEED", 42)
    MAX_NEW = getenv("MAX_NEW", 80)

    cfg = Config()
    print(f"=== L4 CFG eval on {os.path.basename(ckpt)} ===")
    print(f"LEVEL={LEVEL} N={N} LOOPS={LOOPS} ALPHAS={ALPHAS}")
    print(f"SINE_TEMP={os.environ.get('SINE_TEMP', '0')}\n")

    examples = generate_math(LEVEL, N + 50, seed=SEED, digit_spacing=True)[:N]

    print(f"loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    info = model.load_state_dict(safe_load(ckpt), strict=False)
    print(f"  loaded.\n")
    tok = load_tokenizer()
    Tensor.training = False
    sep_ids = tok.encode(SEP).ids
    print(f"  SEP token ids: {sep_ids}\n")

    # Tokenize prompts (problem text)
    prompt_ids_all = [tok.encode(ex.problem).ids for ex in examples]
    print(f"avg prompt length: {sum(len(p) for p in prompt_ids_all)/len(prompt_ids_all):.0f} tokens\n")

    results = {}  # {(n_loops, alpha): accuracy}
    for n_loops in LOOPS:
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            print(f"\n--- n_loops={n_loops}  α={alpha} ---")
            correct = 0
            samples = []
            for i, ex in enumerate(examples):
                gen = cfg_generate(model, prompt_ids_all[i], n_loops, alpha,
                                   sep_ids, max_new=MAX_NEW, eos=0)
                gen_text = tok.decode(gen)
                # Take the last cycle's answer
                if "####" in gen_text:
                    last_chunk = gen_text.rsplit("####", 2)[-2] if gen_text.count("####") >= 2 else gen_text.rsplit("####", 1)[0]
                else:
                    last_chunk = gen_text
                parsed = parse_int_answer(last_chunk)
                ok = (parsed == ex.answer)
                if ok:
                    correct += 1
                if i < 3:
                    samples.append((ex, parsed, gen_text))
            dt = time.perf_counter() - t0
            acc = correct / len(examples)
            results[(n_loops, alpha)] = acc
            print(f"  acc: {correct}/{len(examples)} = {acc*100:.1f}%   ({dt:.1f}s)")
            for ex, parsed, gen in samples:
                ok = "OK" if parsed == ex.answer else "WRONG"
                print(f"    Q: {ex.problem!r}")
                print(f"    gen: {gen[:160].strip()!r}{'...' if len(gen) > 160 else ''}")
                print(f"    parsed={parsed} gold={ex.answer} [{ok}]")

    print()
    print("--- summary ---")
    print(f"{'n_loops':>8s}  " + "  ".join(f"α={a:>4.1f}" for a in ALPHAS))
    for n_loops in LOOPS:
        row = f"{n_loops:>8d}"
        for alpha in ALPHAS:
            acc = results.get((n_loops, alpha), float('nan'))
            row += f"  {acc*100:>5.1f}%"
        print(row)


if __name__ == "__main__":
    main()
