"""
Generate per-cycle intermediate target data for L3 through L4.9.

Each problem includes a `cycle_targets` list where each entry is the
intermediate numeric result the model should produce at that thinking cycle.

Output format (JSONL):
  {"problem": "...", "cycle_targets": [54, 64], "final_answer": 64, "num_steps": 2, "level": "L4"}

Usage:
  python scripts/generate_per_cycle_data.py --level L3 --num_problems 2000
  python scripts/generate_per_cycle_data.py --level all --num_problems 2000
"""
import argparse
import json
import os
import random


# ---------------------------------------------------------------------------
# Shared vocabulary (matches existing L3/L4/L4.5/L4.7 generators)
# ---------------------------------------------------------------------------

NAMES = [
    'Jamie', 'Sarah', 'Mike', 'Emma', 'Alex', 'Lisa', 'Tom', 'Anna',
    'Ben', 'Kate', 'Sam', 'Mia', 'Jack', 'Zoe', 'Noah', 'Lily',
    'Ryan', 'Ella', 'Dan', 'Sophia',
]

OBJECTS = [
    'cookies', 'apples', 'marbles', 'stickers', 'pencils', 'books',
    'cards', 'coins', 'shells', 'flowers', 'balloons', 'crayons',
    'rocks', 'stamps', 'buttons', 'beads', 'toy cars', 'ribbons',
]

PLACES = [
    'a store', 'a bakery', 'a farm', 'a school', 'a library',
    'a garden', 'a shop', 'a market', 'a cafe', 'a toy store',
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

TIME_QUADS = [
    ('on Monday', 'on Tuesday', 'on Wednesday', 'on Thursday'),
    ('in the morning', 'at noon', 'in the afternoon', 'in the evening'),
    ('in January', 'in February', 'in March', 'in April'),
    ('in week one', 'in week two', 'in week three', 'in week four'),
    ('on the first day', 'on the second day', 'on the third day', 'on the fourth day'),
]

TIME_QUINTS = [
    ('on Monday', 'on Tuesday', 'on Wednesday', 'on Thursday', 'on Friday'),
    ('in January', 'in February', 'in March', 'in April', 'in May'),
    ('in week one', 'in week two', 'in week three', 'in week four', 'in week five'),
    ('on day one', 'on day two', 'on day three', 'on day four', 'on day five'),
]

# Additional scenario variety for L4.9
STORE_ITEMS = [
    ('shirts', 'dollars'),
    ('notebooks', 'dollars'),
    ('pens', 'dollars'),
    ('bags of flour', 'dollars'),
    ('boxes of crayons', 'dollars'),
    ('packets of seeds', 'dollars'),
    ('jars of honey', 'dollars'),
    ('bottles of juice', 'dollars'),
]

RATE_UNITS = [
    ('miles', 'hours'),
    ('pages', 'hours'),
    ('problems', 'minutes'),
    ('laps', 'minutes'),
]


# =====================================================================
# L3 generators — 1 step, single arithmetic on named quantities
# =====================================================================

def _l3_add(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    a = rng.randint(10, 200)
    b = rng.randint(5, 200)
    result = a + b
    templates = [
        f"{name} had {a} {obj} and found {b} more. How many {obj} does {name} have now?",
        f"{name} collected {a} {obj} in the morning and {b} in the afternoon. How many {obj} did {name} collect in total?",
        f"{name} has {a} {obj}. {name} buys {b} more. How many {obj} does {name} have now?",
    ]
    problem = rng.choice(templates)
    return problem, [result], result


def _l3_sub(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    a = rng.randint(20, 300)
    b = rng.randint(1, a - 1)
    result = a - b
    templates = [
        f"{name} had {a} {obj} and gave {b} away. How many {obj} does {name} have now?",
        f"{name} had {a} {obj}. {name} lost {b}. How many {obj} does {name} have now?",
        f"{name} started with {a} {obj} and used {b}. How many {obj} are left?",
        f"{name} had {a} {obj}. {name} ate {b} of them. How many {obj} are left?",
    ]
    problem = rng.choice(templates)
    return problem, [result], result


def _l3_double(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    a = rng.randint(10, 150)
    result = a * 2
    templates = [
        f"{name} had {a} {obj}. {name} doubled the collection. How many {obj} does {name} have now?",
        f"{name} has {a} {obj}. {name} gets the same amount again. How many {obj} does {name} have now?",
    ]
    problem = rng.choice(templates)
    return problem, [result], result


def _l3_half(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    a = rng.randint(10, 250) * 2  # ensure even
    result = a // 2
    templates = [
        f"{name} had {a} {obj} and gave half away. How many {obj} does {name} have now?",
        f"{name} has {a} {obj}. {name} splits them evenly with a friend. How many does {name} keep?",
    ]
    problem = rng.choice(templates)
    return problem, [result], result


def _l3_triple(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    a = rng.randint(10, 100)
    result = a * 3
    problem = f"{name} had {a} {obj}. {name} tripled the collection. How many {obj} does {name} have now?"
    return problem, [result], result


def _l3_multiply(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    a = rng.randint(2, 30)
    b = rng.randint(2, 15)
    result = a * b
    templates = [
        f"{name} bought {a} packs of {obj} with {b} in each pack. How many {obj} does {name} have in total?",
        f"There are {a} boxes with {b} {obj} each. How many {obj} are there in total?",
    ]
    problem = rng.choice(templates)
    return problem, [result], result


L3_GENERATORS = [_l3_add, _l3_sub, _l3_double, _l3_half, _l3_triple, _l3_multiply]


# =====================================================================
# L4 generators — 2 steps
# =====================================================================

def _l4_sub_sub(rng):
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v1, _ = rng.choice(LOSE_VERBS)
    v2, _ = rng.choice(LOSE_VERBS)
    start = rng.randint(30, 300)
    a = rng.randint(5, start // 2)
    mid = start - a
    b = rng.randint(1, mid - 1)
    result = mid - b
    problem = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1} and {v2} {b} {t2}. "
        f"How many {obj} are left?"
    )
    return problem, [mid, result], result


def _l4_add_add(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v1, s1 = rng.choice(GAIN_VERBS)
    v2, s2 = rng.choice(GAIN_VERBS)
    start = rng.randint(10, 150)
    a = rng.randint(5, 150)
    mid = start + a
    b = rng.randint(5, 150)
    result = mid + b
    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v1} {a} {s1} {t1} and {v2} {b} {s2} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    return problem, [mid, result], result


def _l4_sub_add(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v_lose, _ = rng.choice(LOSE_VERBS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    start = rng.randint(30, 300)
    a = rng.randint(5, start - 5)
    mid = start - a
    b = rng.randint(5, 200)
    result = mid + b
    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose} {a} {t1}, then {v_gain} {b} {s_gain} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    return problem, [mid, result], result


def _l4_add_sub(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    v_lose, _ = rng.choice(LOSE_VERBS)
    start = rng.randint(10, 200)
    a = rng.randint(5, 200)
    mid = start + a
    b = rng.randint(1, mid - 1)
    result = mid - b
    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}, then {v_lose} {b} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    return problem, [mid, result], result


def _l4_half_add(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    start = rng.randint(20, 300) * 2  # ensure even
    mid = start // 2
    b = rng.randint(5, 200)
    result = mid + b
    problem = (
        f"{name} had {start} {obj}. {name} gave half to a friend, "
        f"then found {b} more. How many {obj} does {name} have now?"
    )
    return problem, [mid, result], result


def _l4_double_sub(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    start = rng.randint(10, 150)
    mid = start * 2
    b = rng.randint(1, mid - 1)
    result = mid - b
    problem = (
        f"{name} had {start} {obj}. {name} doubled the collection, "
        f"then gave {b} away. How many {obj} does {name} have now?"
    )
    return problem, [mid, result], result


L4_GENERATORS = [_l4_sub_sub, _l4_add_add, _l4_sub_add, _l4_add_sub,
                 _l4_half_add, _l4_double_sub]


# =====================================================================
# L4.5 generators — 3 steps (first cycle can be extraction)
# =====================================================================

def _l45_sub_sub_sub(rng):
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
    return problem, [mid1, mid2, result], result


def _l45_add_sub_add(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v_gain1, s1 = rng.choice(GAIN_VERBS)
    v_lose, _ = rng.choice(LOSE_VERBS)
    v_gain2, s2 = rng.choice(GAIN_VERBS)
    start = rng.randint(20, 200)
    a = rng.randint(10, 200)
    mid1 = start + a
    b = rng.randint(10, mid1 - 10)
    mid2 = mid1 - b
    c = rng.randint(10, 200)
    result = mid2 + c
    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain1} {a} {s1} {t1}, "
        f"{v_lose} {b} {t2}, "
        f"and {v_gain2} {c} {s2} {t3}. "
        f"How many {obj} does {name} have now?"
    )
    return problem, [mid1, mid2, result], result


def _l45_sub_add_sub(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3 = rng.choice(TIME_TRIPLES)
    v_lose1, _ = rng.choice(LOSE_VERBS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    v_lose2, _ = rng.choice(LOSE_VERBS)
    start = rng.randint(50, 400)
    a = rng.randint(10, start - 10)
    mid1 = start - a
    b = rng.randint(10, 300)
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
    return problem, [mid1, mid2, result], result


def _l45_double_sub_add(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    start = rng.randint(20, 150)
    mid1 = start * 2
    b = rng.randint(10, mid1 - 10)
    mid2 = mid1 - b
    c = rng.randint(10, 200)
    result = mid2 + c
    problem = (
        f"{name} had {start} {obj}. {name} doubled the collection, "
        f"then gave {b} away, and later found {c} more. "
        f"How many {obj} does {name} have now?"
    )
    return problem, [mid1, mid2, result], result


def _l45_extract_sub_sub(rng):
    """First cycle is extraction (reading a number), then two ops."""
    name = rng.choice(NAMES)
    friend = rng.choice([n for n in NAMES if n != name])
    obj = rng.choice(OBJECTS)
    start = rng.randint(50, 300)
    a = rng.randint(10, start // 2)
    mid1 = start  # extraction: just read the starting number
    mid2 = start - a
    c = rng.randint(10, mid2 - 1)
    result = mid2 - c
    problem = (
        f"{name} has a collection of {start} {obj}. "
        f"{name} gives {a} to {friend} and then loses {c}. "
        f"How many {obj} does {name} have left?"
    )
    return problem, [mid1, mid2, result], result


def _l45_half_add_sub(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    start = rng.randint(40, 300) * 2  # ensure even
    mid1 = start // 2
    b = rng.randint(10, 200)
    mid2 = mid1 + b
    c = rng.randint(1, mid2 - 1)
    result = mid2 - c
    problem = (
        f"{name} had {start} {obj}. {name} gave half away, "
        f"then found {b} more, and later used {c}. "
        f"How many {obj} does {name} have now?"
    )
    return problem, [mid1, mid2, result], result


L45_GENERATORS = [_l45_sub_sub_sub, _l45_add_sub_add, _l45_sub_add_sub,
                  _l45_double_sub_add, _l45_extract_sub_sub, _l45_half_add_sub]


# =====================================================================
# L4.7 generators — 4 steps
# =====================================================================

def _l47_four_ops_linear(rng):
    """4 sequential +/- operations on a single quantity."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2, t3, t4 = rng.choice(TIME_QUADS)
    start = rng.randint(50, 300)
    vals = [start]
    ops_text = []
    for i in range(4):
        cur = vals[-1]
        if rng.random() < 0.5 and cur > 20:
            amt = rng.randint(5, max(6, cur // 2))
            vals.append(cur - amt)
            v, _ = rng.choice(LOSE_VERBS)
            ops_text.append(f"{v} {amt}")
        else:
            amt = rng.randint(5, 150)
            vals.append(cur + amt)
            v, s = rng.choice(GAIN_VERBS)
            ops_text.append(f"{v} {amt} {s}")
    times = [t1, t2, t3, t4]
    actions = ", ".join(f"{ops_text[i]} {times[i]}" for i in range(4))
    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {actions}. "
        f"How many {obj} does {name} have now?"
    )
    cycle_targets = vals[1:]  # 4 intermediate results
    return problem, cycle_targets, cycle_targets[-1]


def _l47_buy_sell_profit(rng):
    """Buy items at cost, sell some at price, compute remainder value."""
    name = rng.choice(NAMES)
    item, unit = rng.choice(STORE_ITEMS)
    qty_bought = rng.randint(10, 80)
    cost_each = rng.randint(2, 15)
    total_cost = qty_bought * cost_each  # step 1
    qty_sold = rng.randint(5, qty_bought - 1)
    sell_price = cost_each + rng.randint(1, 5)
    revenue = qty_sold * sell_price  # step 2
    remaining = qty_bought - qty_sold  # step 3
    result = revenue - total_cost  # step 4: profit (can be negative = loss)
    problem = (
        f"{name} bought {qty_bought} {item} at {cost_each} {unit} each. "
        f"{name} sold {qty_sold} of them at {sell_price} {unit} each. "
        f"How much profit or loss did {name} make?"
    )
    return problem, [total_cost, revenue, remaining, result], result


def _l47_multi_person(rng):
    """Track items across multiple people: A gives to B, B gives to C, etc."""
    names = rng.sample(NAMES, 3)
    obj = rng.choice(OBJECTS)
    start = rng.randint(60, 300)
    a = rng.randint(10, start // 3)
    mid1 = start - a  # after giving to B
    b = rng.randint(10, mid1 // 2)
    mid2 = mid1 - b  # after giving to C
    c = rng.randint(10, 200)
    mid3 = mid2 + c  # found more
    d = rng.randint(1, mid3 - 1)
    result = mid3 - d  # gave some away
    problem = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]}, gave {b} to {names[2]}, "
        f"found {c} more, and then donated {d} to charity. "
        f"How many {obj} does {names[0]} have now?"
    )
    return problem, [mid1, mid2, mid3, result], result


def _l47_rate_problem(rng):
    """Rate * time, then adjust: e.g. pages/hour for N hours, minus pages already read."""
    name = rng.choice(NAMES)
    rate = rng.randint(5, 30)
    time1 = rng.randint(2, 8)
    done1 = rate * time1  # step 1
    time2 = rng.randint(1, 5)
    done2 = rate * time2  # step 2
    total_done = done1 + done2  # step 3
    total_pages = total_done + rng.randint(10, 100)
    remaining = total_pages - total_done  # step 4
    problem = (
        f"{name} reads {rate} pages per hour. "
        f"{name} read for {time1} hours in the morning and {time2} hours in the afternoon. "
        f"The book has {total_pages} pages. How many pages are left to read?"
    )
    return problem, [done1, done2, total_done, remaining], remaining


L47_GENERATORS = [_l47_four_ops_linear, _l47_buy_sell_profit,
                  _l47_multi_person, _l47_rate_problem]


# =====================================================================
# L4.9 generators — 5 steps, GSM8K-style simple word problems
# =====================================================================

def _l49_weekly_earnings(rng):
    """Compute weekly earnings from different jobs."""
    name = rng.choice(NAMES)
    hours1 = rng.randint(3, 10)
    rate1 = rng.randint(8, 25)
    earn1 = hours1 * rate1  # step 1
    hours2 = rng.randint(2, 8)
    rate2 = rng.randint(8, 20)
    earn2 = hours2 * rate2  # step 2
    total_earn = earn1 + earn2  # step 3
    rent = rng.randint(20, min(total_earn - 10, 200))
    after_rent = total_earn - rent  # step 4
    food = rng.randint(10, min(after_rent - 5, 80))
    savings = after_rent - food  # step 5
    problem = (
        f"{name} works two jobs. At the first job, {name} works {hours1} hours "
        f"at {rate1} dollars per hour. At the second job, {name} works {hours2} hours "
        f"at {rate2} dollars per hour. {name} pays {rent} dollars for rent and "
        f"{food} dollars for food each week. How much does {name} save each week?"
    )
    return problem, [earn1, earn2, total_earn, after_rent, savings], savings


def _l49_shopping_trip(rng):
    """Multi-item shopping with discount or tax."""
    name = rng.choice(NAMES)
    item1_qty = rng.randint(2, 8)
    item1_price = rng.randint(3, 20)
    cost1 = item1_qty * item1_price  # step 1
    item2_qty = rng.randint(1, 6)
    item2_price = rng.randint(5, 25)
    cost2 = item2_qty * item2_price  # step 2
    subtotal = cost1 + cost2  # step 3
    # Use a simple discount (flat amount to keep integers)
    discount = rng.randint(5, min(subtotal // 3, 30))
    after_discount = subtotal - discount  # step 4
    budget = after_discount + rng.randint(10, 100)
    change = budget - after_discount  # step 5
    problem = (
        f"{name} goes shopping with {budget} dollars. "
        f"{name} buys {item1_qty} notebooks at {item1_price} dollars each "
        f"and {item2_qty} pens at {item2_price} dollars each. "
        f"{name} has a coupon for {discount} dollars off. "
        f"How much change does {name} get?"
    )
    return problem, [cost1, cost2, subtotal, after_discount, change], change


def _l49_garden_harvest(rng):
    """Planting and harvesting across multiple days."""
    name = rng.choice(NAMES)
    rows = rng.randint(3, 10)
    per_row = rng.randint(5, 20)
    total_plants = rows * per_row  # step 1
    died = rng.randint(2, total_plants // 4)
    surviving = total_plants - died  # step 2
    fruit_per = rng.randint(2, 8)
    total_fruit = surviving * fruit_per  # step 3
    ate = rng.randint(5, total_fruit // 3)
    remaining = total_fruit - ate  # step 4
    gave_away = rng.randint(5, remaining - 5) if remaining > 10 else 1
    final = remaining - gave_away  # step 5
    problem = (
        f"{name} planted {rows} rows of tomatoes with {per_row} plants in each row. "
        f"{died} plants died. Each surviving plant produced {fruit_per} tomatoes. "
        f"{name} ate {ate} tomatoes and gave {gave_away} to neighbors. "
        f"How many tomatoes does {name} have left?"
    )
    return problem, [total_plants, surviving, total_fruit, remaining, final], final


def _l49_classroom(rng):
    """Classroom/school supply problem."""
    name = rng.choice(NAMES)
    num_students = rng.randint(15, 35)
    pencils_each = rng.randint(2, 6)
    total_pencils = num_students * pencils_each  # step 1
    erasers_each = rng.randint(1, 3)
    total_erasers = num_students * erasers_each  # step 2
    total_supplies = total_pencils + total_erasers  # step 3
    boxes = rng.randint(3, 8)
    per_box = total_supplies // boxes  # step 4, integer division
    leftover = total_supplies - (per_box * boxes)  # step 5
    problem = (
        f"{name} is a teacher with {num_students} students. "
        f"Each student needs {pencils_each} pencils and {erasers_each} erasers. "
        f"{name} packs the supplies into {boxes} boxes with {per_box} items each. "
        f"How many items are left over?"
    )
    return problem, [total_pencils, total_erasers, total_supplies, per_box, leftover], leftover


def _l49_travel(rng):
    """Travel problem with different speeds and distances."""
    name = rng.choice(NAMES)
    speed1 = rng.randint(30, 70)
    time1 = rng.randint(1, 5)
    dist1 = speed1 * time1  # step 1
    speed2 = rng.randint(20, 60)
    time2 = rng.randint(1, 4)
    dist2 = speed2 * time2  # step 2
    total_dist = dist1 + dist2  # step 3
    total_trip = total_dist + rng.randint(20, 200)
    remaining = total_trip - total_dist  # step 4
    gas_per_mile = rng.choice([2, 3, 4, 5])
    gas_cost = remaining * gas_per_mile  # step 5 (cost for remaining)
    problem = (
        f"{name} is driving {total_trip} miles. "
        f"{name} drove {speed1} miles per hour for {time1} hours, "
        f"then {speed2} miles per hour for {time2} hours. "
        f"Gas costs {gas_per_mile} dollars per mile for the remaining distance. "
        f"How much will gas cost for the rest of the trip?"
    )
    return problem, [dist1, dist2, total_dist, remaining, gas_cost], gas_cost


L49_GENERATORS = [_l49_weekly_earnings, _l49_shopping_trip, _l49_garden_harvest,
                  _l49_classroom, _l49_travel]


# =====================================================================
# Generation engine
# =====================================================================

LEVEL_CONFIG = {
    'L3':   {'generators': L3_GENERATORS,  'min_result': 1,  'max_result': 500},
    'L4':   {'generators': L4_GENERATORS,  'min_result': 1,  'max_result': 1000},
    'L4.5': {'generators': L45_GENERATORS, 'min_result': 1,  'max_result': 2000},
    'L4.7': {'generators': L47_GENERATORS, 'min_result': None, 'max_result': None},  # allow negatives for profit/loss
    'L4.9': {'generators': L49_GENERATORS, 'min_result': 0,  'max_result': None},
}


def generate_level(level, num_problems, seed):
    """Generate problems for a given level."""
    config = LEVEL_CONFIG[level]
    generators = config['generators']
    min_result = config['min_result']
    max_result = config['max_result']

    rng = random.Random(seed)
    samples = []
    attempts = 0
    max_attempts = num_problems * 100

    while len(samples) < num_problems and attempts < max_attempts:
        attempts += 1
        gen = rng.choice(generators)
        try:
            problem, cycle_targets, final_answer = gen(rng)
        except (ValueError, ZeroDivisionError):
            continue

        # Validate
        if cycle_targets[-1] != final_answer:
            continue
        if min_result is not None and final_answer < min_result:
            continue
        if max_result is not None and final_answer > max_result:
            continue

        samples.append({
            'problem': problem,
            'cycle_targets': cycle_targets,
            'final_answer': final_answer,
            'num_steps': len(cycle_targets),
            'level': level,
        })

    if len(samples) < num_problems:
        print(f"WARNING: Only generated {len(samples)}/{num_problems} for {level} "
              f"after {attempts} attempts")

    return samples


def write_jsonl(samples, path):
    """Write samples to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(f"Wrote {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-cycle intermediate target data'
    )
    parser.add_argument('--level', type=str, default='all',
                        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'all'],
                        help='Level to generate (default: all)')
    parser.add_argument('--num_problems', type=int, default=2000,
                        help='Number of problems per level (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='data/per_cycle',
                        help='Output directory (default: data/per_cycle)')
    args = parser.parse_args()

    levels = list(LEVEL_CONFIG.keys()) if args.level == 'all' else [args.level]

    for level in levels:
        print(f"\n--- Generating {level} ({args.num_problems} problems, seed={args.seed}) ---")
        samples = generate_level(level, args.num_problems, args.seed)

        # Split 90/10 train/eval
        split = int(len(samples) * 0.9)
        train_samples = samples[:split]
        eval_samples = samples[split:]

        # Filename format matches train_per_cycle.py expectations: {level}_train.jsonl
        train_path = os.path.join(args.output_dir, f'{level}_train.jsonl')
        eval_path = os.path.join(args.output_dir, f'{level}_eval.jsonl')
        write_jsonl(train_samples, train_path)
        write_jsonl(eval_samples, eval_path)
        print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

        # Print a few examples
        for i, s in enumerate(samples[:3]):
            print(f"  Example {i+1}: {s['problem'][:80]}...")
            print(f"    cycle_targets={s['cycle_targets']}, final={s['final_answer']}")


if __name__ == '__main__':
    main()
