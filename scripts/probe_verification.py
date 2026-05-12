"""Verification probe: does the trained model produce different representational
trajectories when given correct vs wrong candidate answers?

For each problem in the eval set, construct two inputs:
  - CORRECT: tokenize "X op Y = GOLD_ANSWER" (full equation, correct)
  - WRONG:   tokenize "X op Y = CORRUPTED_GOLD" (same prefix, one digit corrupted)

Breathe through both. Capture per-breath integrated reps. Compute two metrics per
breath:
  (a) Per-example loss at the answer-token positions — measures the model's
      surprise at the candidate. CORRECT should be lower than WRONG (just from
      the model's arithmetic prior).
  (b) L2 norm of representational delta between consecutive breaths at the
      answer positions — measures whether the representation STABILIZES vs
      keeps shifting. If verification is naturally available via rotation,
      CORRECT should show small delta (stable) while WRONG shows large delta
      (the model's later breaths "object" to the wrong answer).

If (b) differentiates correct vs wrong AND the gap grows across breaths,
verification is naturally available — controller just needs to learn to read
the energy signal. If trajectories are indistinguishable, verification needs
to be trained explicitly.

Usage:
    DEV=PCI+AMD CKPT=.cache/arith_mixed_ckpts/arith_mixed_v5_step300.safetensors \\
        .venv/bin/python scripts/probe_verification.py
"""
import sys, os, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, encode_cycles, collate, space_digits
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


def corrupt_answer_digit(answer_int: int, rng: random.Random) -> int:
    """Corrupt one digit of the answer to produce a plausibly-wrong candidate."""
    s = str(answer_int)
    if len(s) == 1:
        # 1-digit: pick a different 1-digit value
        choices = [d for d in range(10) if d != answer_int]
        return rng.choice(choices)
    # Pick a digit position and bump it by ±1, ±2, or ±3 (mod 10)
    pos = rng.randrange(len(s))
    orig = int(s[pos])
    delta_choices = [d for d in (-3, -2, -1, 1, 2, 3) if 0 <= orig + d <= 9]
    delta = rng.choice(delta_choices)
    new_digit = orig + delta
    new_s = s[:pos] + str(new_digit) + s[pos+1:]
    # Avoid leading zero accidents
    if new_s[0] == '0' and len(new_s) > 1:
        new_s = '1' + new_s[1:]
    return int(new_s)


def main():
    ckpt = getenv("CKPT", ".cache/arith_mixed_ckpts/arith_mixed_v5_step300.safetensors")
    B = getenv("B", 24)
    MAX_LOOPS = getenv("MAX_LOOPS", 8)
    SEED = getenv("SEED", 42)
    SINE_TEMP_MATCH = bool(getenv("SINE_TEMP_MATCH", 1))   # match training (SINE_TEMP=1 if ckpt was trained that way)

    # NB: if the ckpt was trained with SINE_TEMP=1, the env var must be set
    # before importing breathing.py. Re-run with SINE_TEMP=1 in the command.
    if SINE_TEMP_MATCH:
        os.environ.setdefault("SINE_TEMP", "1")

    cfg = Config()
    print(f"=== verification probe on {os.path.basename(ckpt)} ===")
    print(f"B={B} max_loops={MAX_LOOPS} sine_temp_env={os.environ.get('SINE_TEMP', '0')}\n")

    # Generate ARITH_MIXED problems
    raw = generate_math("ARITH_MIXED", B + 50, seed=SEED, digit_spacing=False)[:B]
    rng = random.Random(SEED + 13)

    # For each problem, build (correct_full, wrong_full) inputs
    examples_correct = []
    examples_wrong = []
    diffs = []
    for ex in raw:
        # Build "X op Y = GOLD" and "X op Y = WRONG" (both digit-spaced)
        wrong_ans = corrupt_answer_digit(ex.answer, rng)
        correct_full = space_digits(f"{ex.problem} {ex.answer}.")
        wrong_full = space_digits(f"{ex.problem} {wrong_ans}.")
        examples_correct.append(correct_full)
        examples_wrong.append(wrong_full)
        diffs.append(("HARD-3d" if max(int(s) for s in re.findall(r'\d+', ex.problem)) >= 100
                      else "HARD-carry" if "+" in ex.problem and sum(int(s) for s in re.findall(r'\d+', ex.problem)[:2][:1] + re.findall(r'\d+', ex.problem)[:2][1:2]) % 10 + sum(int(s) for s in re.findall(r'\d+', ex.problem)[:2][:1] + re.findall(r'\d+', ex.problem)[:2][1:2]) // 10 >= 10
                      else "EASY"))

    print("Sample of corrupted vs gold answers:")
    for i in range(3):
        print(f"  [{diffs[i]}] CORRECT: {examples_correct[i]!r}")
        print(f"             WRONG:   {examples_wrong[i]!r}")
    print()

    print("loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    info = model.load_state_dict(safe_load(ckpt), strict=False)
    print(f"  loaded. (missing={len(info['missing'])} unexpected={len(info['unexpected'])})\n")

    tok = load_tokenizer()
    Tensor.training = False

    def tokenize_batch(strs, max_len=64):
        """Tokenize each string, right-pad to max_len with 0, return tensor (B, max_len) + valid_lens."""
        ids_list = [tok.encode(s).ids for s in strs]
        max_len = max(max_len, max(len(ids) for ids in ids_list))
        out = np.zeros((len(strs), max_len), dtype=np.int32)
        valid_lens = []
        for b, ids in enumerate(ids_list):
            L = min(len(ids), max_len)
            out[b, :L] = ids[:L]
            valid_lens.append(L)
        return Tensor(out, dtype=dtypes.int).realize(), valid_lens

    def run_forward(strs):
        """Run breathe_controlled, return list of per-breath integrated reps (numpy)."""
        tokens, valid_lens = tokenize_batch(strs)
        notebook = Notebook()
        _, _, _, _, integrated = model.breathe_controlled(
            tokens, max_loops=MAX_LOOPS, notebook=notebook, return_per_breath_reps=True
        )
        # integrated: list of (B, T, hidden) tensors
        return [r.numpy() for r in integrated], valid_lens

    print("Forward on CORRECT inputs...")
    reps_correct, lens_correct = run_forward(examples_correct)
    print(f"  done. {len(reps_correct)} breaths × shape {reps_correct[0].shape}")
    print("Forward on WRONG inputs...")
    reps_wrong, lens_wrong = run_forward(examples_wrong)
    print(f"  done.\n")

    # Metric (b): per-breath delta norm at the equation's last 5 positions
    # (where the answer tokens live for digit-spaced answers).
    def delta_norms_per_breath(reps, valid_lens):
        """Per-example per-breath ||rep[l] - rep[l-1]|| at the last few positions of each example."""
        T = reps[0].shape[1]
        B = reps[0].shape[0]
        out = np.zeros((len(reps) - 1, B), dtype=np.float32)
        for l in range(1, len(reps)):
            for b in range(B):
                end = valid_lens[b]
                start = max(0, end - 5)
                d = reps[l][b, start:end, :] - reps[l-1][b, start:end, :]
                out[l-1, b] = np.linalg.norm(d.reshape(-1))
        return out

    delta_corr = delta_norms_per_breath(reps_correct, lens_correct)
    delta_wrong = delta_norms_per_breath(reps_wrong, lens_wrong)

    # Metric (a): rep norm at last position (per breath)
    def rep_norms_per_breath(reps, valid_lens):
        out = np.zeros((len(reps), reps[0].shape[0]), dtype=np.float32)
        for l in range(len(reps)):
            for b in range(reps[0].shape[0]):
                end = valid_lens[b]
                start = max(0, end - 5)
                out[l, b] = np.linalg.norm(reps[l][b, start:end, :].reshape(-1))
        return out

    norm_corr = rep_norms_per_breath(reps_correct, lens_correct)
    norm_wrong = rep_norms_per_breath(reps_wrong, lens_wrong)

    print("--- per-breath delta norm (consecutive change at answer positions) ---")
    print(f"{'breath_change':>14s}  {'correct_mean':>13s}  {'wrong_mean':>11s}  {'ratio_wrong/correct':>20s}")
    for l in range(delta_corr.shape[0]):
        cm = delta_corr[l].mean()
        wm = delta_wrong[l].mean()
        ratio = wm / cm if cm > 0 else float('inf')
        print(f"  {l}→{l+1:>12d}  {cm:>13.3f}  {wm:>11.3f}  {ratio:>20.3f}")

    print()
    print("--- per-breath rep norm at answer positions ---")
    print(f"{'breath':>7s}  {'correct_mean':>13s}  {'wrong_mean':>11s}  {'wrong - correct':>17s}")
    for l in range(norm_corr.shape[0]):
        cm = norm_corr[l].mean()
        wm = norm_wrong[l].mean()
        print(f"  {l:>6d}  {cm:>13.3f}  {wm:>11.3f}  {wm-cm:>+17.3f}")

    print()
    print("--- verdict ---")
    early_ratio = (delta_wrong[0].mean() / max(delta_corr[0].mean(), 1e-6))
    late_ratio = (delta_wrong[-1].mean() / max(delta_corr[-1].mean(), 1e-6))
    print(f"  early breath delta ratio (wrong/correct): {early_ratio:.3f}")
    print(f"  late breath delta ratio  (wrong/correct): {late_ratio:.3f}")
    if late_ratio > 1.5 * early_ratio:
        print("  → VERIFICATION SIGNAL EMERGING: wrong-answer instability grows across breaths")
    elif late_ratio > 1.2:
        print("  → mild verification signal: wrong answers more unstable but flat across breaths")
    else:
        print("  → NO verification signal: correct vs wrong indistinguishable in delta norm")


if __name__ == "__main__":
    main()
