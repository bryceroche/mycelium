"""
Stepping stones curriculum for GSM8K generalization.

Levels bridge pure arithmetic (proven) → full word problems (target).
Each level adds ONE new challenge. Train sequentially, warm-start next level
from previous checkpoint. Find exactly where the model breaks.

Level 0: "48 / 2 ="                               (pure arithmetic)
Level 1: "(48 / 2) + 48 ="                        (chained arithmetic)
Level 2: "half of 48 plus 48"                     (words for operations)
Level 3: "48 clips in April, half as many in May, total?"  (named quantities)
Level 4: Easy 2-step GSM8K-style (small numbers, clean format)
Level 5: Full GSM8K                               (target)

ALL levels constrained to positive integer answers in [1, 200].
log10(1)=0, log10(200)≈2.3 — a tight, bounded target range for the log-head.
"""

# Hard constraint: every generated sample has answer in [_ANS_MIN, _ANS_MAX].
_ANS_MIN = 1
_ANS_MAX = 200


def _in_range(x) -> bool:
    return isinstance(x, int) and _ANS_MIN <= x <= _ANS_MAX

import random
from typing import List, Dict, Optional


# ---------- Level 0: pure arithmetic "A op B =" ----------

_L0_OPS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
]


def _level0_sample(rng: random.Random) -> Dict:
    while True:
        sym, fn = rng.choice(_L0_OPS)
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        ans = fn(a, b)
        if _in_range(ans):
            return {"question": f"{a} {sym} {b} =", "answer": ans, "level": 0}


# ---------- Level 1: chained arithmetic "(A op B) op C =" ----------

def _level1_sample(rng: random.Random) -> Dict:
    while True:
        s1, f1 = rng.choice(_L0_OPS)
        s2, f2 = rng.choice(_L0_OPS)
        a, b, c = rng.randint(1, 30), rng.randint(1, 30), rng.randint(1, 30)
        mid = f1(a, b)
        if not _in_range(mid):
            continue
        ans = f2(mid, c)
        if _in_range(ans):
            return {"question": f"({a} {s1} {b}) {s2} {c} =", "answer": ans, "level": 1}


# ---------- Level 2: operation words ----------

_L2_BIN = [
    ("plus", lambda a, b: a + b),
    ("minus", lambda a, b: a - b),
    ("times", lambda a, b: a * b),
]

_L2_UNARY = [
    ("half of", lambda a: a // 2, lambda rng: rng.randrange(2, 50) * 2),
    ("double", lambda a: a * 2, lambda rng: rng.randrange(1, 40)),
    ("triple", lambda a: a * 3, lambda rng: rng.randrange(1, 25)),
    ("a quarter of", lambda a: a // 4, lambda rng: rng.randrange(1, 25) * 4),
]


def _level2_sample(rng: random.Random) -> Dict:
    while True:
        word, unary_fn, draw = rng.choice(_L2_UNARY)
        a = draw(rng)
        op_word, bin_fn = rng.choice(_L2_BIN)
        b = rng.randint(1, 50)
        ans = bin_fn(unary_fn(a), b)
        if _in_range(ans):
            return {"question": f"{word} {a} {op_word} {b}", "answer": ans, "level": 2}


# ---------- Level 3: minimal named-quantity word problems ----------

_L3_TEMPLATES = [
    # (template, solver(a,b) → answer)
    # "X items in period1, half as many in period2, total?"
    ("{a} {item} in {p1}, half as many in {p2}. How many in total?",
     lambda a, b: a + a // 2),
    ("{a} {item} in {p1}, twice as many in {p2}. How many in total?",
     lambda a, b: a + a * 2),
    ("{name} has {a} {item}. {name2} has {b} more. How many do they have together?",
     lambda a, b: a + (a + b)),
    ("{name} had {a} {item} and gave {b} away. How many are left?",
     lambda a, b: a - b),
    ("{name} buys {a} {item} at {b} dollars each. How much did {name} spend?",
     lambda a, b: a * b),
]

_ITEMS = ["clips", "apples", "cookies", "marbles", "stickers", "books", "pencils"]
_NAMES = ["Natalia", "Weng", "Betty", "Alex", "Jamie", "Sam", "Kim"]
_PERIODS = [("April", "May"), ("Monday", "Tuesday"), ("morning", "afternoon")]


def _level3_sample(rng: random.Random) -> Dict:
    while True:
        tpl, solver = rng.choice(_L3_TEMPLATES)
        a = rng.randrange(2, 40) * 2  # even so halves are clean
        b = rng.randrange(1, 20)
        ans = solver(a, b)
        if not _in_range(ans):
            continue
        item = rng.choice(_ITEMS)
        name = rng.choice(_NAMES)
        name2 = rng.choice([n for n in _NAMES if n != name])
        p1, p2 = rng.choice(_PERIODS)
        q = tpl.format(a=a, b=b, item=item, name=name, name2=name2, p1=p1, p2=p2)
        return {"question": q, "answer": ans, "level": 3}


# ---------- Level 4: easy 2-step GSM8K-style (small numbers) ----------

def _level4_sample(rng: random.Random) -> Dict:
    """2-step word problem with small numbers, answer in [1, 200]."""
    patterns = [
        lambda a, b: (
            f"A store has {a} {rng.choice(_ITEMS)}. They sell {b} of them. "
            f"Then they receive a shipment of {b} more. How many do they have now?",
            a - b + b,
        ),
        lambda a, b: (
            f"{rng.choice(_NAMES)} has {a} dollars. She spends {b} dollars on "
            f"lunch and half of what's left on a book. How much does she have now?",
            (a - b) - (a - b) // 2,
        ),
        lambda a, b: (
            f"There are {a} students in a class. {b} are boys and the rest are "
            f"girls. How many more girls than boys are there?",
            (a - b) - b,
        ),
        lambda a, b: (
            f"{rng.choice(_NAMES)} reads {a} pages on Monday and {b} pages on "
            f"Tuesday. How many pages did she read in total?",
            a + b,
        ),
    ]
    while True:
        a = rng.randint(10, 80)
        b = rng.randint(1, min(a, 40))
        build = rng.choice(patterns)
        q, ans = build(a, b)
        if _in_range(ans):
            return {"question": q, "answer": ans, "level": 4}


# ---------- Public API ----------

_LEVEL_FNS = {
    0: _level0_sample,
    1: _level1_sample,
    2: _level2_sample,
    3: _level3_sample,
    4: _level4_sample,
}


def generate(level: int, n: int, seed: Optional[int] = None) -> List[Dict]:
    assert level in _LEVEL_FNS, f"Level {level} not in this module (L5 uses real GSM8K)"
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(_LEVEL_FNS[level](rng))
    return out


if __name__ == "__main__":
    for lvl in (0, 1, 2, 3, 4):
        print(f"--- Level {lvl} ---")
        for s in generate(lvl, 3, seed=0):
            print(f"  Q: {s['question']}")
            print(f"  A: {s['answer']}")
