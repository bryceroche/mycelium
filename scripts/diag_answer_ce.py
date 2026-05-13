"""Diagnostic: does per-example answer-CE actually differentiate easy from hard problems?

This is the key question for the v4 calibration result. v4 supervised stop_logit
against per-example answer-CE but per-problem stop_logit std plateaued at the
same level as v2 (which used op-CE). Two possibilities:

  (A) The supervisory signal (per-example answer-CE) doesn't actually vary much
      across easy/hard problems → the bottleneck is the SIGNAL, not the head.
      The rep at breath 1-2 might be too early to differentiate problems even
      though the OUTPUT eventually does at breath 4-8.
  (B) The signal varies but the controller's linear stop head can't extract it.

This script computes per-example per-breath answer-CE on a mixed ARITH_MIXED
batch, classifies problems by difficulty (easy vs hard via problem inspection),
and prints the distribution per breath. If (A), easy and hard trajectories
overlap. If (B), easy and hard separate but the controller still can't differentiate.

Usage:
    DEV=PCI+AMD CKPT=.cache/arith_mixed_ckpts/arith_mixed_v4_step300.safetensors \\
        .venv/bin/python scripts/diag_answer_ce.py
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
from mycelium.l3_data import generate_math, collate, encode_cycles
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


def classify_difficulty(problem_text: str) -> str:
    """Heuristic: easy = 2-digit no-carry/no-borrow; hard = otherwise."""
    # Strip digit spaces (problem text is digit-spaced in our dataset)
    flat = re.sub(r'(\d) (?=\d)', r'\1', problem_text)
    nums = [int(s) for s in re.findall(r'\d+', flat)]
    if not nums: return "?"
    has_3digit = any(n >= 100 for n in nums)
    if has_3digit: return "HARD-3d"
    if "+" in flat:
        a, b = nums[0], nums[1]
        return "HARD-carry" if (a % 10 + b % 10) >= 10 else "EASY-add"
    if "-" in flat and "*" not in flat and "/" not in flat:
        a, b = nums[0], nums[1]
        if (a % 10) < (b % 10) or ((a // 10) % 10) < ((b // 10) % 10):
            return "HARD-borrow"
        return "EASY-sub"
    return "?"


def main():
    ckpt = getenv("CKPT", ".cache/arith_mixed_ckpts/arith_mixed_v4_step300.safetensors")
    B = getenv("B", 32)
    MAX_LOOPS = getenv("MAX_LOOPS", 8)
    SEED = getenv("SEED", 42)
    FIXED_LEN = 64 + 40

    cfg = Config()
    print(f"=== diag answer-CE on {os.path.basename(ckpt)} ===")
    print(f"B={B} max_loops={MAX_LOOPS} fixed_len={FIXED_LEN}\n")

    examples = generate_math("ARITH_MIXED", B + 50, seed=SEED, digit_spacing=True)[:B]
    diffs = [classify_difficulty(ex.problem) for ex in examples]
    print(f"distribution: {dict((d, diffs.count(d)) for d in set(diffs))}\n")

    print("loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    info = model.load_state_dict(safe_load(ckpt), strict=False)
    print(f"  loaded.\n")

    tok = load_tokenizer()
    Tensor.training = False

    # Encode + collate
    cycles_per_ex = [encode_cycles(tok, ex) for ex in examples]
    encoded = [ex_cycles[0] for ex_cycles in cycles_per_ex]
    tokens_np, labels_np = collate(encoded, fixed_len=FIXED_LEN)
    tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
    labels = Tensor(labels_np, dtype=dtypes.int).realize()
    answer_mask_np = (labels_np != -100).astype(np.float32)
    answer_mask = Tensor(answer_mask_np, dtype=dtypes.float).realize()

    # Forward
    notebook = Notebook()
    _, decisions, n_breaths, mw, integrated_per_breath = model.breathe_controlled(
        tokens, max_loops=MAX_LOOPS, notebook=notebook, return_per_breath_reps=True,
    )

    # Per-breath per-example answer CE
    per_breath_per_ex_ans = []
    for rep in integrated_per_breath:
        full_logits = (rep @ model.embed_out).cast(dtypes.float)
        pred = full_logits[:, :-1, :]
        per_tok_ce = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="none")
        per_ex_ans = (per_tok_ce * answer_mask).sum(axis=1) / answer_mask.sum(axis=1).maximum(1.0)
        per_breath_per_ex_ans.append(per_ex_ans.numpy())   # (B,)

    # Layout: arr[breath, example]
    arr = np.stack(per_breath_per_ex_ans, axis=0)  # (n_breaths, B)

    # Group by difficulty and show stats per breath
    groups = {}
    for b, d in enumerate(diffs):
        groups.setdefault(d, []).append(b)

    print("--- per-breath per-example answer-CE ---")
    print(f"{'breath':>7s} " + " ".join(f"{g:>11s}" for g in sorted(groups.keys())))
    for l in range(arr.shape[0]):
        row = [f"{l:>7d} "]
        for g in sorted(groups.keys()):
            idxs = groups[g]
            vals = arr[l, idxs]
            row.append(f"{vals.mean():5.2f}±{vals.std():4.2f}")
        print("  ".join(row))

    print()
    print("--- per-example trajectories (sample of 8) ---")
    sample_idxs = np.linspace(0, B-1, 8, dtype=int)
    for b in sample_idxs:
        traj = arr[:, b]
        print(f"  [{diffs[b]:11s}] {examples[b].problem:25s} → " +
              "  ".join(f"{v:5.2f}" for v in traj))

    # The KEY question: at which breath does easy vs hard separation become measurable?
    print()
    print("--- easy vs hard separation per breath ---")
    print(f"{'breath':>7s}  {'easy_mean':>10s}  {'hard_mean':>10s}  {'separation':>11s}  ({'easy_n':>6s} {'hard_n':>6s})")
    easy_groups = [g for g in groups.keys() if g.startswith("EASY")]
    hard_groups = [g for g in groups.keys() if g.startswith("HARD")]
    easy_idxs = sum((groups[g] for g in easy_groups), [])
    hard_idxs = sum((groups[g] for g in hard_groups), [])
    for l in range(arr.shape[0]):
        easy_m = arr[l, easy_idxs].mean() if easy_idxs else np.nan
        hard_m = arr[l, hard_idxs].mean() if hard_idxs else np.nan
        sep = hard_m - easy_m
        print(f"{l:>7d}  {easy_m:>10.3f}  {hard_m:>10.3f}  {sep:>+11.3f}  ({len(easy_idxs):>6d} {len(hard_idxs):>6d})")


if __name__ == "__main__":
    main()
