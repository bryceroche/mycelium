"""Math dataset: word problems with answers, parameterized by curriculum level.

L3 = 1-step, L4 = 2-step, L4.5 = 3-step. Reuses the existing
scripts/generate_per_cycle_data.py problem generators.

For multi-step problems, the per-cycle gen targets are concatenated into a
single answer text (the model is trained to produce all steps in sequence).
"""
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Make scripts/ importable so we can pull in the math generators
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
from generate_per_cycle_data import L3_GENERATORS, L4_GENERATORS, L4_BORROW_GENERATORS, L4_MIXED_GENERATORS  # type: ignore
try:
    from generate_per_cycle_data import L45_GENERATORS  # type: ignore
except ImportError:
    L45_GENERATORS = None


_INT_RE = re.compile(r"-?\d+")
_INT_OR_SPACED_RE = re.compile(r"-?\d(?:\s\d)*")  # matches "38" OR "3 8" OR "1 2 3"


def space_digits(text: str) -> str:
    """Insert spaces between digits in any integer substring.

    "Sarah had 170 cookies" -> "Sarah had 1 7 0 cookies"
    The space-prefix BPE absorbs into each digit token, giving the model digit-by-
    digit prediction granularity that breathing can refine individually.
    """
    def _split(m):
        s = m.group(0)
        sign = ""
        if s[0] == "-":
            sign = "-"
            s = s[1:]
        return sign + " ".join(s)
    return _INT_RE.sub(_split, text)


def _collapse_spaced(s: str) -> str:
    """Inverse of space_digits for a single matched integer string."""
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]
    return sign + s.replace(" ", "")


_L4_BORROW_REGISTERED = True
_LEVEL_GENERATORS = {
    "L3": L3_GENERATORS,
    "L4": L4_GENERATORS,
}
if L45_GENERATORS is not None:
    _LEVEL_GENERATORS["L4.5"] = L45_GENERATORS
_LEVEL_GENERATORS["L4_BORROW"] = L4_BORROW_GENERATORS
_LEVEL_GENERATORS["L4_MIXED"] = L4_MIXED_GENERATORS


# ---------------------------------------------------------------------------
# ARITH: pure equations, no language. Used for an arithmetic capacity diagnostic
# and for arithmetic pretraining before language-rich curriculum levels.
# ---------------------------------------------------------------------------

# ARITH generators return the L3/L4-compatible 4-tuple
# (problem, cycle_targets, final_answer, gens) so they slot into generate_math().
def _arith_add(rng):
    a = rng.randint(10, 200)
    b = rng.randint(5, 200)
    r = a + b
    return f"{a} + {b} =", [r], r, [f"{a + b}."]


def _arith_sub(rng):
    a = rng.randint(20, 300)
    b = rng.randint(1, a - 1)
    r = a - b
    return f"{a} - {b} =", [r], r, [f"{r}."]


def _arith_mul(rng):
    a = rng.randint(2, 30)
    b = rng.randint(2, 15)
    r = a * b
    return f"{a} * {b} =", [r], r, [f"{r}."]


def _arith_double(rng):
    a = rng.randint(10, 150)
    r = 2 * a
    return f"{a} * 2 =", [r], r, [f"{r}."]


def _arith_triple(rng):
    a = rng.randint(10, 100)
    r = 3 * a
    return f"{a} * 3 =", [r], r, [f"{r}."]


def _arith_half(rng):
    a = rng.randint(10, 250) * 2
    r = a // 2
    return f"{a} / 2 =", [r], r, [f"{r}."]


ARITH_GENERATORS = [_arith_add, _arith_sub, _arith_mul, _arith_double, _arith_triple, _arith_half]
_LEVEL_GENERATORS["ARITH"] = ARITH_GENERATORS


# ARITH_HARD: targeted at the specific failure modes from the L4 7/7 v2 run —
# 3-digit subtraction with borrows (170 - 132 = 38) and 2-digit addition with
# carries that cross 100 (68 + 63 = 131). The pure-random ARITH distribution
# only forces these ~50% of the time; ARITH_HARD adds three generators that
# *always* require carry/borrow so the model is forced to learn the cross-digit
# mechanics, not just digit-local lookup.
def _arith_add_carry(rng):
    """2-digit + 2-digit, ones-digit carry guaranteed."""
    a_ones = rng.randint(2, 9)
    b_ones = rng.randint(10 - a_ones, 9)
    a_tens = rng.randint(2, 9)
    b_tens = rng.randint(2, 9)
    a = 10 * a_tens + a_ones
    b = 10 * b_tens + b_ones
    r = a + b
    return f"{a} + {b} =", [r], r, [f"{r}."]


def _arith_sub_borrow_2d(rng):
    """2-digit - 2-digit, ones-digit borrow guaranteed."""
    while True:
        a_tens = rng.randint(2, 9)
        b_tens = rng.randint(1, a_tens)
        a_ones = rng.randint(0, 8)
        b_ones = rng.randint(a_ones + 1, 9)
        a = 10 * a_tens + a_ones
        b = 10 * b_tens + b_ones
        if a > b:
            break
    r = a - b
    return f"{a} - {b} =", [r], r, [f"{r}."]


def _arith_sub_borrow_3d(rng):
    """3-digit minuend in [100, 300] - 2-digit subtrahend, borrow guaranteed.
    Mimics the exact operation L4 intermediates produce (e.g., 170 - 132 = 38)."""
    while True:
        a = rng.randint(100, 300)
        b = rng.randint(20, min(a - 1, 199))
        a_ones, a_tens = a % 10, (a // 10) % 10
        b_ones, b_tens = b % 10, (b // 10) % 10
        if a_ones < b_ones or a_tens < b_tens:
            break
    r = a - b
    return f"{a} - {b} =", [r], r, [f"{r}."]


ARITH_HARD_GENERATORS = ARITH_GENERATORS + [_arith_add_carry, _arith_sub_borrow_2d, _arith_sub_borrow_3d]
_LEVEL_GENERATORS["ARITH_HARD"] = ARITH_HARD_GENERATORS


# ARITH_MIXED: bimodal difficulty. Half the problems are trivially easy
# (1-digit add/sub with no carries/borrows — the model should solve in 1 breath).
# Half are hard (3-digit subtraction with borrows or 2-digit addition with
# guaranteed carry). Designed for the option-1 experiment: the controller
# needs a reason to differentiate problems. With bimodal difficulty, an
# intelligent controller would output stop_logit > 0 early on easy problems
# and stop_logit < 0 on hard ones — i.e., LEARN to use observation to
# orchestrate compute. The ARITH_HARD trained controller emits open-loop
# schedules (f(breath_idx) only); this dataset gives a clear gradient on
# observation-conditional stopping.
def _arith_easy_add_no_carry(rng):
    """2-digit + 2-digit, NO carry guaranteed. Same format as the hard add_carry
    generator but the per-digit sums all stay < 10 — model can do this in 1 breath
    because there's no cross-digit dependency to track."""
    a_tens = rng.randint(1, 4)
    b_tens = rng.randint(1, 4)
    a_ones = rng.randint(0, 4)
    b_ones = rng.randint(0, 4)
    a = 10 * a_tens + a_ones
    b = 10 * b_tens + b_ones
    r = a + b                                     # all column sums < 10
    return f"{a} + {b} =", [r], r, [f"{r}."]


def _arith_easy_sub_no_borrow(rng):
    """2-digit - 2-digit, NO borrow guaranteed. Same format as hard sub_borrow
    but each digit of a >= each digit of b — model can subtract digit-by-digit
    with no carry tracking, doable in 1 breath."""
    a_tens = rng.randint(2, 9)
    b_tens = rng.randint(1, a_tens)               # b_tens <= a_tens
    a_ones = rng.randint(0, 9)
    b_ones = rng.randint(0, a_ones)               # b_ones <= a_ones
    a = 10 * a_tens + a_ones
    b = 10 * b_tens + b_ones
    if a == b:                                    # avoid trivial 0
        b_ones = max(0, b_ones - 1)
        b = 10 * b_tens + b_ones
    r = a - b
    return f"{a} - {b} =", [r], r, [f"{r}."]


ARITH_MIXED_GENERATORS = [_arith_easy_add_no_carry, _arith_easy_sub_no_borrow,
                          _arith_sub_borrow_3d, _arith_add_carry]
_LEVEL_GENERATORS["ARITH_MIXED"] = ARITH_MIXED_GENERATORS


# ARITH_BORROW: targeted at the cascading-borrow failure mode revealed by
# L4 v4 error classification (58% of L4 errors are off-by-10, caused by
# multi-level borrows that cascade from ones → tens → hundreds).
# The cascade generator forces (a_ones < b_ones) AND (a_tens <= b_tens):
# the ones-borrow decrements tens, which then ALSO needs to borrow from
# hundreds. This is the specific pattern that ARITH_HARD's borrow generator
# under-sampled.
def _arith_sub_borrow_cascade(rng):
    """3-digit subtraction with forced CASCADING borrow.

    Conditions: a_ones < b_ones (forces ones-borrow) AND a_tens <= b_tens
    (after ones-borrow, the tens position has a_tens - 1 < b_tens which
    forces a second borrow from hundreds). The cascade is the specific
    pattern that produces off-by-10 errors in L4 v4."""
    while True:
        a_hundreds = rng.randint(1, 9)
        a_tens = rng.randint(0, 8)            # leave room for b_tens >= a_tens
        a_ones = rng.randint(0, 8)            # leave room for b_ones > a_ones
        a = 100 * a_hundreds + 10 * a_tens + a_ones
        b_ones = rng.randint(a_ones + 1, 9)
        b_tens = rng.randint(a_tens, 9)       # CASCADE: tens forced into borrow
        # 50/50 split: 2-digit vs 3-digit subtrahend
        if rng.random() < 0.5:
            b = 10 * b_tens + b_ones
        else:
            b_hundreds = rng.randint(0, a_hundreds - 1)
            b = 100 * b_hundreds + 10 * b_tens + b_ones
        if a > b:
            break
    r = a - b
    return f"{a} - {b} =", [r], r, [f"{r}."]


# ARITH_BORROW level: ~50% cascading borrow (the targeted hard case),
# 50% mix of other arithmetic to keep other operations from regressing.
ARITH_BORROW_GENERATORS = (
    [_arith_sub_borrow_cascade] * 4            # 4× weight on the targeted case
    + [_arith_easy_add_no_carry,
       _arith_easy_sub_no_borrow,
       _arith_add_carry,
       _arith_sub_borrow_3d]                   # maintenance: keep other ops trained
)
_LEVEL_GENERATORS["ARITH_BORROW"] = ARITH_BORROW_GENERATORS


SEP = " ####"  # marker between outer cycles; tokenizes consistently


@dataclass
class MathExample:
    problem: str
    gen_targets: List[str]   # per-cycle gen text (no separator)
    answer: int
    level: str

    @property
    def gen(self) -> str:
        """Concatenated gen with separators between cycles. Appended SEP also at end."""
        return SEP.join(self.gen_targets) + SEP


# Backwards-compat alias
L3Example = MathExample


def generate_math(level: str, num_problems: int, seed: int = 42,
                  digit_spacing: bool = False) -> List[MathExample]:
    """Generate `num_problems` examples at the given curriculum level.

    digit_spacing=True applies space_digits() to problem + gen_targets so each
    decimal digit becomes its own token. Required for proper iterative arithmetic
    via breathing — without it Pythia BPE merges multi-digit numbers into single
    tokens and answer prediction becomes a one-shot 50K-way classification.
    """
    if level not in _LEVEL_GENERATORS:
        raise ValueError(f"unknown level {level!r}; available: {sorted(_LEVEL_GENERATORS)}")
    generators = _LEVEL_GENERATORS[level]
    rng = random.Random(seed)
    out: List[MathExample] = []
    for _ in range(num_problems):
        gen_fn = rng.choice(generators)
        problem, _cycle_targets, final_answer, gens = gen_fn(rng)
        if not isinstance(gens, list):
            gens = [gens]
        if digit_spacing:
            problem = space_digits(problem)
            gens = [space_digits(g) for g in gens]
        out.append(MathExample(
            problem=problem,
            gen_targets=list(gens),
            answer=int(final_answer),
            level=level,
        ))
    return out


def generate_l3(num_problems: int, seed: int = 42) -> List[MathExample]:
    """Backwards-compat wrapper."""
    return generate_math("L3", num_problems, seed)


def parse_int_answer(text: str) -> int | None:
    """Pull the last integer from generated text. Handles both unspaced ('38')
    and digit-spaced ('3 8') answers — collapses spaced digit runs back to ints.
    """
    matches = _INT_OR_SPACED_RE.findall(text)
    if not matches:
        return None
    try:
        return int(_collapse_spaced(matches[-1]))
    except ValueError:
        return None


def encode_example(tok, ex: MathExample, eos_id: int = 0) -> Tuple[List[int], int, int]:
    """Tokenize problem + ' ' + gen + EOS. Returns (ids, problem_len, total_len).

    Single-cycle / backwards-compatible encoding. For multi-cycle training use
    encode_cycles() instead.
    """
    p_ids = tok.encode(ex.problem).ids
    g_ids = tok.encode(" " + ex.gen).ids
    ids = p_ids + g_ids + [eos_id]
    return ids, len(p_ids), len(ids)


def encode_cycles(tok, ex: MathExample, eos_id: int = 0) -> List[Tuple[List[int], int, int]]:
    """Per-cycle encoding for multi-cycle training.

    Returns a list with one (ids, prefix_len, total_len) tuple per outer cycle.
    For each cycle:
      - prefix = problem + previously-generated cycles' targets (with separators)
      - target = " " + gen_targets[i] + SEP   (first cycle: leading space)
                 OR   gen_targets[i] + SEP    (later cycles: SEP already provides space)
      - For the LAST cycle we also append EOS so the model learns to terminate.
    The training mask runs from prefix_len-1 onwards (predict from last prefix token).
    """
    p_ids = tok.encode(ex.problem).ids
    cycles: List[Tuple[List[int], int, int]] = []
    prefix = list(p_ids)
    for i, gt in enumerate(ex.gen_targets):
        target_text = (" " if i == 0 else "") + gt + SEP
        target_ids = tok.encode(target_text).ids
        ids = list(prefix) + list(target_ids)
        if i == len(ex.gen_targets) - 1:
            ids = ids + [eos_id]
        cycles.append((ids, len(prefix), len(ids)))
        prefix = ids if i == len(ex.gen_targets) - 1 else (prefix + list(target_ids))
        # NOTE: we don't carry the EOS into prefix for cycle iteration — last cycle is
        # the last so this only matters here for clarity.
    return cycles


def collate(examples: List[Tuple[List[int], int, int]], pad_id: int = 0,
            fixed_len: int | None = None):
    """Right-pad a batch to a length. Returns:
      tokens:  (B, T) int32 — input ids
      labels:  (B, T-1) int32 — next-token targets, with -100 in the masked positions
                (problem span and post-EOS pads).

    If fixed_len is given, pad/truncate to exactly that length. Otherwise pad to
    the max in batch (which causes JIT recompiles per unique shape).
    """
    max_T = fixed_len if fixed_len is not None else max(total for _, _, total in examples)
    B = len(examples)
    tokens = np.full((B, max_T), pad_id, dtype=np.int32)
    labels = np.full((B, max_T - 1), -100, dtype=np.int32)
    for b, (ids, p_len, t_len) in enumerate(examples):
        eff_t = min(t_len, max_T)
        tokens[b, :eff_t] = ids[:eff_t]
        # next-token targets: targets[i] = ids[i+1]; keep them only for i >= p_len - 1
        # AND i < eff_t - 1 (so we don't supervise predicting pad / past truncation).
        for i in range(max(p_len - 1, 0), eff_t - 1):
            labels[b, i] = ids[i + 1]
    return tokens, labels


def split_train_eval(examples: List[L3Example], n_eval: int = 200, seed: int = 0):
    rng = random.Random(seed)
    shuf = examples[:]
    rng.shuffle(shuf)
    return shuf[n_eval:], shuf[:n_eval]
