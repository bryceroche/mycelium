"""
Procedural dataset generators for L4.5 and L4.7 difficulty levels.

L4.5: 2-step word problems with BIGGER numbers (answers [1, 2000]).
      Same 6 generator types as L4 but numbers drawn from [10, 500].

L4.7: 3-step word problems (answers [1, 5000]).
      Numbers drawn from [10, 500], three chained operations with
      narrative context.

Both datasets return {'problem': str, 'answer': str, 'final': int}.
"""

import random
from torch.utils.data import Dataset


# ---------- Shared vocabulary (same as L4) ----------

NAMES = [
    'Jamie', 'Sarah', 'Mike', 'Emma', 'Alex', 'Lisa', 'Tom', 'Anna',
    'Ben', 'Kate', 'Sam', 'Mia', 'Jack', 'Zoe', 'Noah', 'Lily',
    'Ryan', 'Ella', 'Dan', 'Sophia',
]

PLACES = [
    'a store', 'a bakery', 'a farm', 'a school', 'a library',
    'a garden', 'a shop', 'a market', 'a cafe', 'a toy store',
]

OBJECTS = [
    'cookies', 'apples', 'books', 'flowers', 'cupcakes', 'oranges',
    'pencils', 'stickers', 'balloons', 'muffins', 'cards', 'toys',
    'donuts', 'sandwiches', 'tickets', 'bottles', 'candles', 'stamps',
]

GAIN_VERBS = [
    ('received', 'from a supplier'),
    ('baked', 'more'),
    ('bought', 'more'),
    ('found', 'more'),
    ('got', 'as a delivery'),
    ('made', 'more'),
]

LOSE_VERBS = [
    ('sold', ''),
    ('gave away', ''),
    ('used', ''),
    ('donated', ''),
    ('threw out', ''),
    ('lost', ''),
]

TIME_PAIRS = [
    ('on Monday', 'on Tuesday'),
    ('in the morning', 'in the afternoon'),
    ('on Saturday', 'on Sunday'),
    ('in the first hour', 'in the second hour'),
    ('before lunch', 'after lunch'),
    ('yesterday', 'today'),
    ('in January', 'in February'),
    ('on the first day', 'on the second day'),
]

# Extended time triples for 3-step problems
TIME_TRIPLES = [
    ('on Monday', 'on Tuesday', 'on Wednesday'),
    ('in the morning', 'in the afternoon', 'in the evening'),
    ('on Friday', 'on Saturday', 'on Sunday'),
    ('in the first hour', 'in the second hour', 'in the third hour'),
    ('in January', 'in February', 'in March'),
    ('on the first day', 'on the second day', 'on the third day'),
    ('before lunch', 'during lunch', 'after lunch'),
    ('in week one', 'in week two', 'in week three'),
]


# =====================================================================
# L4.5 generators — 2-step, bigger numbers [10, 500]
# =====================================================================

def _l45_sub_sub(rng):
    """Sub then sub: start - a - b."""
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v1, _ = rng.choice(LOSE_VERBS)
    v2, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(50, 500)
    a = rng.randint(10, start // 2)
    mid = start - a
    b = rng.randint(10, mid - 1)
    result = mid - b

    problem = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1} and {v2} {b} {t2}. "
        f"How many {obj} are left?"
    )
    cot = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1}. {start} - {a} = {mid}. "
        f"They {v2} {b} {t2}. {mid} - {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l45_add_add(rng):
    """Add then add: start + a + b."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v1, s1 = rng.choice(GAIN_VERBS)
    v2, s2 = rng.choice(GAIN_VERBS)

    start = rng.randint(10, 400)
    a = rng.randint(10, 400)
    mid = start + a
    b = rng.randint(10, 400)
    result = mid + b

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v1} {a} {s1} {t1} and {v2} {b} {s2} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v1} {a} {s1} {t1}. {start} + {a} = {mid}. "
        f"{name} {v2} {b} {s2} {t2}. {mid} + {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l45_sub_add(rng):
    """Sub then add: start - a + b."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v_lose, _ = rng.choice(LOSE_VERBS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)

    start = rng.randint(50, 500)
    a = rng.randint(10, start - 1)
    mid = start - a
    b = rng.randint(10, 400)
    result = mid + b

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose} {a} {t1}, then {v_gain} {b} {s_gain} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose} {a} {t1}. {start} - {a} = {mid}. "
        f"{name} {v_gain} {b} {s_gain} {t2}. {mid} + {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l45_add_sub(rng):
    """Add then sub: start + a - b."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    v_lose, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(10, 400)
    a = rng.randint(10, 400)
    mid = start + a
    b = rng.randint(10, mid - 1)
    result = mid - b

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}, then {v_lose} {b} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}. {start} + {a} = {mid}. "
        f"{name} {v_lose} {b} {t2}. {mid} - {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l45_person_transfer(rng):
    """Person-to-person: A gives to B, then B gives to C."""
    names = rng.sample(NAMES, 3)
    obj = rng.choice(OBJECTS)

    start = rng.randint(50, 500)
    a = rng.randint(10, start // 2)
    mid = start - a
    b = rng.randint(10, mid - 1)
    result = mid - b

    problem = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]} and then gave {b} to {names[2]}. "
        f"How many {obj} does {names[0]} have now?"
    )
    cot = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]}. {start} - {a} = {mid}. "
        f"{names[0]} gave {b} to {names[2]}. {mid} - {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l45_group_event(rng):
    """Group event: people arrive and leave (bigger numbers)."""
    place = rng.choice(PLACES)
    t1, t2 = rng.choice(TIME_PAIRS)

    pattern = rng.choice(['arrive_arrive', 'arrive_leave', 'leave_leave'])

    start = rng.randint(20, 500)

    if pattern == 'arrive_arrive':
        a = rng.randint(10, 300)
        mid = start + a
        b = rng.randint(10, 300)
        result = mid + b
        problem = (
            f"There were {start} people at {place}. "
            f"{a} more people arrived {t1} and {b} more arrived {t2}. "
            f"How many people are at {place} now?"
        )
        cot = (
            f"There were {start} people at {place}. "
            f"{a} more arrived {t1}. {start} + {a} = {mid}. "
            f"{b} more arrived {t2}. {mid} + {b} = {result}. "
            f"The answer is {result}."
        )
    elif pattern == 'arrive_leave':
        a = rng.randint(10, 400)
        mid = start + a
        b = rng.randint(10, mid - 1)
        result = mid - b
        problem = (
            f"There were {start} people at {place}. "
            f"{a} more people arrived {t1}, but {b} left {t2}. "
            f"How many people are at {place} now?"
        )
        cot = (
            f"There were {start} people at {place}. "
            f"{a} arrived {t1}. {start} + {a} = {mid}. "
            f"{b} left {t2}. {mid} - {b} = {result}. "
            f"The answer is {result}."
        )
    else:  # leave_leave
        a = rng.randint(10, start // 2)
        mid = start - a
        b = rng.randint(10, max(11, mid - 1))
        if b >= mid:
            b = mid - 1
        result = mid - b
        problem = (
            f"There were {start} people at {place}. "
            f"{a} people left {t1} and {b} more left {t2}. "
            f"How many people are at {place} now?"
        )
        cot = (
            f"There were {start} people at {place}. "
            f"{a} left {t1}. {start} - {a} = {mid}. "
            f"{b} left {t2}. {mid} - {b} = {result}. "
            f"The answer is {result}."
        )

    return problem, cot, result


L45_GENERATORS = [
    _l45_sub_sub,
    _l45_add_add,
    _l45_sub_add,
    _l45_add_sub,
    _l45_person_transfer,
    _l45_group_event,
]


class L45TwoStepWordDataset(Dataset):
    """
    L4.5: two-step word problems with bigger numbers.
    Same 6 generator types as L4 but numbers drawn from [10, 500].
    Answers in [1, 2000].
    """
    def __init__(self, num_samples=20000, seed=42):
        rng = random.Random(seed)
        self.samples = []
        attempts = 0
        while len(self.samples) < num_samples:
            attempts += 1
            if attempts > num_samples * 50:
                raise RuntimeError(
                    f"Could not generate {num_samples} samples after {attempts} attempts"
                )
            gen = rng.choice(L45_GENERATORS)
            problem, cot, result = gen(rng)
            if 1 <= result <= 2000:
                self.samples.append({
                    'problem': problem,
                    'answer': cot,
                    'final': result,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =====================================================================
# L4.7 generators — 3-step word problems, numbers [10, 500]
# =====================================================================

def _l47_sub_sub_sub(rng):
    """Sub-sub-sub: start - a - b - c."""
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v1, _ = rng.choice(LOSE_VERBS)
    v2, _ = rng.choice(LOSE_VERBS)
    v3, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(100, 500)
    a = rng.randint(10, start // 3)
    mid1 = start - a
    b = rng.randint(10, mid1 // 2)
    mid2 = mid1 - b
    c = rng.randint(10, mid2 - 1)
    result = mid2 - c

    problem = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1}, {v2} {b} {t2}, and {v3} {c} {t3}. "
        f"How many {obj} are left?"
    )
    cot = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1}. {start} - {a} = {mid1}. "
        f"They {v2} {b} {t2}. {mid1} - {b} = {mid2}. "
        f"They {v3} {c} {t3}. {mid2} - {c} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l47_add_sub_sub(rng):
    """Add-sub-sub: start + a - b - c."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    v_lose1, _ = rng.choice(LOSE_VERBS)
    v_lose2, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(10, 300)
    a = rng.randint(10, 300)
    mid1 = start + a
    b = rng.randint(10, mid1 // 2)
    mid2 = mid1 - b
    c = rng.randint(10, mid2 - 1)
    result = mid2 - c

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}, "
        f"{v_lose1} {b} {t2}, and {v_lose2} {c} {t3}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}. {start} + {a} = {mid1}. "
        f"{name} {v_lose1} {b} {t2}. {mid1} - {b} = {mid2}. "
        f"{name} {v_lose2} {c} {t3}. {mid2} - {c} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l47_sub_add_sub(rng):
    """Sub-add-sub: start - a + b - c."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v_lose1, _ = rng.choice(LOSE_VERBS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    v_lose2, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(50, 500)
    a = rng.randint(10, start - 10)
    mid1 = start - a
    b = rng.randint(10, 400)
    mid2 = mid1 + b
    c = rng.randint(10, mid2 - 1)
    result = mid2 - c

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose1} {a} {t1}, "
        f"then {v_gain} {b} {s_gain} {t2}, "
        f"and {v_lose2} {c} {t3}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose1} {a} {t1}. {start} - {a} = {mid1}. "
        f"{name} {v_gain} {b} {s_gain} {t2}. {mid1} + {b} = {mid2}. "
        f"{name} {v_lose2} {c} {t3}. {mid2} - {c} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l47_add_add_sub(rng):
    """Add-add-sub: start + a + b - c."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v_gain1, s_gain1 = rng.choice(GAIN_VERBS)
    v_gain2, s_gain2 = rng.choice(GAIN_VERBS)
    v_lose, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(10, 200)
    a = rng.randint(10, 300)
    mid1 = start + a
    b = rng.randint(10, 300)
    mid2 = mid1 + b
    c = rng.randint(10, mid2 - 1)
    result = mid2 - c

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain1} {a} {s_gain1} {t1}, "
        f"{v_gain2} {b} {s_gain2} {t2}, "
        f"and then {v_lose} {c} {t3}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain1} {a} {s_gain1} {t1}. {start} + {a} = {mid1}. "
        f"{name} {v_gain2} {b} {s_gain2} {t2}. {mid1} + {b} = {mid2}. "
        f"{name} {v_lose} {c} {t3}. {mid2} - {c} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l47_sub_sub_add(rng):
    """Sub-sub-add: start - a - b + c."""
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v1, _ = rng.choice(LOSE_VERBS)
    v2, _ = rng.choice(LOSE_VERBS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)

    start = rng.randint(100, 500)
    a = rng.randint(10, start // 3)
    mid1 = start - a
    b = rng.randint(10, mid1 // 2)
    mid2 = mid1 - b
    c = rng.randint(10, 400)
    result = mid2 + c

    problem = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1}, {v2} {b} {t2}, "
        f"and then {v_gain} {c} {s_gain} {t3}. "
        f"How many {obj} does {place} have now?"
    )
    cot = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1}. {start} - {a} = {mid1}. "
        f"They {v2} {b} {t2}. {mid1} - {b} = {mid2}. "
        f"They {v_gain} {c} {s_gain} {t3}. {mid2} + {c} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _l47_person_chain(rng):
    """Person chain: A gives to B, B gives to C, A also gives to C."""
    names = rng.sample(NAMES, 3)
    obj = rng.choice(OBJECTS)

    start = rng.randint(100, 500)
    a = rng.randint(10, start // 3)
    mid1 = start - a
    b = rng.randint(10, mid1 // 2)
    mid2 = mid1 - b
    c = rng.randint(10, mid2 - 1)
    result = mid2 - c

    problem = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]}, "
        f"then gave {b} to {names[2]}, "
        f"and then donated {c} to charity. "
        f"How many {obj} does {names[0]} have now?"
    )
    cot = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]}. {start} - {a} = {mid1}. "
        f"{names[0]} gave {b} to {names[2]}. {mid1} - {b} = {mid2}. "
        f"{names[0]} donated {c} to charity. {mid2} - {c} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


L47_GENERATORS = [
    _l47_sub_sub_sub,
    _l47_add_sub_sub,
    _l47_sub_add_sub,
    _l47_add_add_sub,
    _l47_sub_sub_add,
    _l47_person_chain,
]


class L47ThreeStepWordDataset(Dataset):
    """
    L4.7: three-step word problems with bigger numbers.
    Numbers drawn from [10, 500], answers in [1, 5000].
    6 generator types chaining 3 sequential operations.
    """
    def __init__(self, num_samples=20000, seed=42):
        rng = random.Random(seed)
        self.samples = []
        attempts = 0
        while len(self.samples) < num_samples:
            attempts += 1
            if attempts > num_samples * 50:
                raise RuntimeError(
                    f"Could not generate {num_samples} samples after {attempts} attempts"
                )
            gen = rng.choice(L47_GENERATORS)
            problem, cot, result = gen(rng)
            if 1 <= result <= 5000:
                self.samples.append({
                    'problem': problem,
                    'answer': cot,
                    'final': result,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =====================================================================
# Main: generate samples and print stats
# =====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("L4.5 — Two-Step Word Problems (bigger numbers)")
    print("=" * 70)

    ds45 = L45TwoStepWordDataset(num_samples=100, seed=99)
    finals_45 = [s['final'] for s in ds45.samples]

    print(f"\nGenerated {len(ds45)} samples")
    print(f"Answer range: [{min(finals_45)}, {max(finals_45)}]")
    print(f"Answer mean:  {sum(finals_45) / len(finals_45):.1f}")

    print("\n--- 5 examples ---\n")
    for i in range(5):
        s = ds45[i]
        print(f"  Q: {s['problem']}")
        print(f"  A: {s['answer']}")
        print(f"  Final: {s['final']}")
        print()

    print("=" * 70)
    print("L4.7 — Three-Step Word Problems")
    print("=" * 70)

    ds47 = L47ThreeStepWordDataset(num_samples=100, seed=99)
    finals_47 = [s['final'] for s in ds47.samples]

    print(f"\nGenerated {len(ds47)} samples")
    print(f"Answer range: [{min(finals_47)}, {max(finals_47)}]")
    print(f"Answer mean:  {sum(finals_47) / len(finals_47):.1f}")

    print("\n--- 5 examples ---\n")
    for i in range(5):
        s = ds47[i]
        print(f"  Q: {s['problem']}")
        print(f"  A: {s['answer']}")
        print(f"  Final: {s['final']}")
        print()
