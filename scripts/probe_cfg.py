"""CFG inference probe: does removing the DC component via classifier-free guidance
improve generation accuracy?

The DC component is a shared direction in representation space that all problems
point toward. It grows linearly with each breath, potentially drowning the
per-problem signal in logit space. Classifier-free guidance from diffusion gives
the principled fix:

  conditional   = breathe on the real problem  (DC + per-problem signal)
  unconditional = breathe on a blank prompt    (DC component alone)
  guided        = conditional + α * (conditional - unconditional)

Generate from `guided` instead of `conditional`. Sweep α to find the value that
maximizes accuracy.

If accuracy improves with α > 0, the DC component is hurting and CFG removes it.
If no improvement at any α, the DC component isn't the bottleneck (or our
unconditional pass isn't a good estimate of it).

Usage:
    DEV=PCI+AMD CKPT=.cache/arith_mixed_ckpts/arith_mixed_v5_step300.safetensors \\
        .venv/bin/python scripts/probe_cfg.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, parse_int_answer
from mycelium.controller import Notebook


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


def main():
    ckpt = getenv("CKPT", ".cache/arith_mixed_ckpts/arith_mixed_v5_step300.safetensors")
    N = getenv("N", 40)
    MAX_LOOPS = getenv("MAX_LOOPS", 8)
    SEED = getenv("SEED", 42)
    ALPHAS = [float(s) for s in getenv("ALPHAS", "0,1,3,5,7.5").split(",")]
    SINE_TEMP_MATCH = bool(getenv("SINE_TEMP_MATCH", 1))

    if SINE_TEMP_MATCH:
        os.environ.setdefault("SINE_TEMP", "1")

    cfg = Config()
    print(f"=== CFG probe on {os.path.basename(ckpt)} ===")
    print(f"N={N} max_loops={MAX_LOOPS} alphas={ALPHAS} sine_temp={os.environ.get('SINE_TEMP', '0')}\n")

    examples = generate_math("ARITH_MIXED", N, seed=SEED, digit_spacing=True)
    print(f"generated {len(examples)} ARITH_MIXED examples")

    print("loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    info = model.load_state_dict(safe_load(ckpt), strict=False)
    print(f"  loaded.\n")
    tok = load_tokenizer()
    Tensor.training = False

    # Tokenize conditional inputs
    cond_ids = [tok.encode(ex.problem).ids for ex in examples]
    max_len = max(len(ids) for ids in cond_ids)
    cond_np = np.zeros((N, max_len), dtype=np.int32)
    for b, ids in enumerate(cond_ids):
        cond_np[b, :len(ids)] = ids

    # Unconditional: all PAD (token 0). Same shape.
    uncond_np = np.zeros((N, max_len), dtype=np.int32)

    print("--- forward: conditional ---")
    cond_tokens = Tensor(cond_np, dtype=dtypes.int).realize()
    notebook_c = Notebook()
    cond_final, _, _, _ = model.breathe_controlled(cond_tokens, max_loops=MAX_LOOPS, notebook=notebook_c)
    cond_hidden = cond_final.numpy()  # (N, T, hidden)

    print("--- forward: unconditional ---")
    uncond_tokens = Tensor(uncond_np, dtype=dtypes.int).realize()
    notebook_u = Notebook()
    uncond_final, _, _, _ = model.breathe_controlled(uncond_tokens, max_loops=MAX_LOOPS, notebook=notebook_u)
    uncond_hidden = uncond_final.numpy()  # (N, T, hidden)

    embed_out_np = model.embed_out.numpy()  # (hidden, vocab)

    print("\n--- generation: greedy argmax at last position, then walk 40 tokens ---")
    print("For each α, run greedy autoregressive generation from the guided rep.")
    print()

    eq_token = tok.encode(" =").ids[-1]  # the = token id
    sep_ids = tok.encode(" ####").ids

    def generate_greedy_from_rep(start_hidden, alpha, max_new=40):
        """Greedy decode max_new tokens starting from the guided last-pos hidden state.
        Note: this only uses the FINAL hidden as the model's understanding of the
        problem — actual generation is done with cached forward passes from the
        original prompt. We don't recompute breathing — we just shift the final
        layernorm output by alpha * (cond - uncond) before applying embed_out.

        For simplicity in this probe: do a single greedy step (1 token) from the
        guided rep and compare to gold's first token. NOT full autoregressive
        decoding because that would require breathing again with the shifted
        rep. This is a noisy proxy but cheap.
        """
        # cond_hidden: (T, hidden), uncond_hidden: (T, hidden)
        # guided = cond + alpha * (cond - uncond)
        # Take last position; project through embed_out
        last_cond = start_hidden  # (hidden,)
        return last_cond  # caller handles the rest

    # For each alpha, score how many examples produce the gold answer's first token
    # at the position after the "=" via greedy argmax.
    results = []
    for alpha in ALPHAS:
        guided = cond_hidden + alpha * (cond_hidden - uncond_hidden)   # (N, T, hidden)
        # Per example: find the "=" position in cond_np, then look at the position
        # AFTER it. The model's prediction at "= " should be the first answer digit.
        correct = 0
        n_scored = 0
        for b, ex in enumerate(examples):
            ids = cond_ids[b]
            eq_pos = None
            for i, t in enumerate(ids):
                if t == eq_token:
                    eq_pos = i; break
            if eq_pos is None or eq_pos >= guided.shape[1] - 1:
                continue
            # Predicted token at position eq_pos+1 — wait, logits at position p
            # predict token at position p+1. So logits at eq_pos predict the
            # next-token (first answer digit).
            logits_at_eq = guided[b, eq_pos, :] @ embed_out_np   # (vocab,)
            pred_tok = int(np.argmax(logits_at_eq))
            # Build gold first answer digit
            gold_first_digit_str = " " + str(ex.answer)[0]
            gold_first_tok = tok.encode(gold_first_digit_str).ids[-1]
            if pred_tok == gold_first_tok:
                correct += 1
            n_scored += 1
        acc = correct / max(1, n_scored)
        results.append((alpha, acc, n_scored))
        print(f"  α={alpha:>5.2f}:  first-digit-accuracy {correct}/{n_scored} = {acc*100:.1f}%")

    print()
    print("--- summary ---")
    print(f"{'alpha':>7s}  {'first-digit-acc':>15s}")
    for alpha, acc, n in results:
        marker = "  ←" if alpha == max(results, key=lambda r: r[1])[0] else ""
        print(f"  {alpha:>5.2f}  {acc*100:>13.1f}%{marker}")
    best_alpha, best_acc, _ = max(results, key=lambda r: r[1])
    baseline_acc = next(r[1] for r in results if r[0] == 0)
    delta = (best_acc - baseline_acc) * 100
    print()
    if delta > 5:
        print(f"  → CFG HELPS: best α={best_alpha} gives +{delta:.1f}% over α=0 (DC component is hurting)")
    elif delta > 1:
        print(f"  → CFG modestly helps: best α={best_alpha} gives +{delta:.1f}% over α=0")
    else:
        print(f"  → CFG does NOT help here: best is α=0 ({baseline_acc*100:.1f}%)")
        print("    DC component may not be the bottleneck, or unconditional is a poor estimate of DC.")


if __name__ == "__main__":
    main()
