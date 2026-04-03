#!/usr/bin/env python3
"""
Generate curriculum learning data for Mycelium v18.

Difficulty levels (VERY gentle ramp):
  Level 0: Single-step arithmetic (2 + 3 = ?)
  Level 1: Two-step arithmetic ((2 + 3) * 4 = ?)
  Level 2: Three-step arithmetic (((2 + 3) * 4) - 5 = ?)
  Level 3: Four-step arithmetic
  Level 4: Simple word problems (templated, single operation)
  Level 5: Two-step word problems
  Level 6: GSM8K easy (filtered by solution length)
  Level 7: Full GSM8K
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def generate_single_step(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Level 0: Single operation arithmetic."""
    random.seed(seed)
    problems = []

    for i in range(n):
        op = random.choice(['+', '-', '*'])
        if op == '+':
            a, b = random.randint(1, 50), random.randint(1, 50)
            answer = a + b
            question = f"What is {a} + {b}?"
        elif op == '-':
            a = random.randint(10, 100)
            b = random.randint(1, a)  # Ensure positive result
            answer = a - b
            question = f"What is {a} - {b}?"
        else:  # multiply
            a, b = random.randint(2, 12), random.randint(2, 12)
            answer = a * b
            question = f"What is {a} × {b}?"

        solution = f"The answer is {answer}.\n\\boxed{{{answer}}}"
        problems.append({
            "question": question,
            "answer": str(answer),
            "solution": solution,
            "level": 0,
            "num_steps": 1,
        })

    return problems


def generate_two_step(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Level 1: Two-step arithmetic with explicit intermediate."""
    random.seed(seed)
    problems = []

    templates = [
        # (format_str, ops, compute_func)
        ("What is ({a} + {b}) × {c}?", lambda a,b,c: (a+b)*c, "First, {a} + {b} = {ab}. Then {ab} × {c} = {ans}."),
        ("What is ({a} × {b}) + {c}?", lambda a,b,c: (a*b)+c, "First, {a} × {b} = {ab}. Then {ab} + {c} = {ans}."),
        ("What is ({a} + {b}) - {c}?", lambda a,b,c: (a+b)-c, "First, {a} + {b} = {ab}. Then {ab} - {c} = {ans}."),
        ("What is ({a} × {b}) - {c}?", lambda a,b,c: (a*b)-c, "First, {a} × {b} = {ab}. Then {ab} - {c} = {ans}."),
    ]

    for i in range(n):
        template_idx = i % len(templates)
        q_template, compute, s_template = templates[template_idx]

        a = random.randint(2, 15)
        b = random.randint(2, 15)
        c = random.randint(2, 15)

        # Compute intermediate and final
        if template_idx == 0:
            ab = a + b
        elif template_idx == 1:
            ab = a * b
        elif template_idx == 2:
            ab = a + b
            c = random.randint(1, ab)  # Ensure positive
        else:
            ab = a * b
            c = random.randint(1, ab)  # Ensure positive

        answer = compute(a, b, c)
        question = q_template.format(a=a, b=b, c=c)
        solution = s_template.format(a=a, b=b, c=c, ab=ab, ans=answer) + f"\n\\boxed{{{answer}}}"

        problems.append({
            "question": question,
            "answer": str(answer),
            "solution": solution,
            "level": 1,
            "num_steps": 2,
        })

    return problems


def generate_three_step(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Level 2: Three-step arithmetic."""
    random.seed(seed)
    problems = []

    for i in range(n):
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        c = random.randint(2, 10)
        d = random.randint(2, 10)

        # ((a + b) × c) - d pattern (most common)
        ab = a + b
        abc = ab * c
        d = random.randint(1, min(abc, 50))  # Ensure positive, reasonable
        answer = abc - d

        question = f"What is (({a} + {b}) × {c}) - {d}?"
        solution = (
            f"Step 1: {a} + {b} = {ab}\n"
            f"Step 2: {ab} × {c} = {abc}\n"
            f"Step 3: {abc} - {d} = {answer}\n"
            f"\\boxed{{{answer}}}"
        )

        problems.append({
            "question": question,
            "answer": str(answer),
            "solution": solution,
            "level": 2,
            "num_steps": 3,
        })

    return problems


def generate_four_step(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Level 3: Four-step arithmetic."""
    random.seed(seed)
    problems = []

    for i in range(n):
        a = random.randint(2, 8)
        b = random.randint(2, 8)
        c = random.randint(2, 8)
        d = random.randint(2, 8)
        e = random.randint(2, 8)

        # (((a + b) × c) - d) + e pattern
        ab = a + b
        abc = ab * c
        d = random.randint(1, min(abc, 30))
        abcd = abc - d
        answer = abcd + e

        question = f"What is ((({a} + {b}) × {c}) - {d}) + {e}?"
        solution = (
            f"Step 1: {a} + {b} = {ab}\n"
            f"Step 2: {ab} × {c} = {abc}\n"
            f"Step 3: {abc} - {d} = {abcd}\n"
            f"Step 4: {abcd} + {e} = {answer}\n"
            f"\\boxed{{{answer}}}"
        )

        problems.append({
            "question": question,
            "answer": str(answer),
            "solution": solution,
            "level": 3,
            "num_steps": 4,
        })

    return problems


def generate_simple_word_problems(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Level 4: Simple templated word problems (single operation)."""
    random.seed(seed)
    problems = []

    # Templates: (question_template, solution_template, answer_func)
    templates = [
        # Addition
        (
            "{name} has {a} {item}. {name2} gives {name} {b} more {item}. How many {item} does {name} have now?",
            "{name} started with {a} {item}. {name2} gave {b} more. {a} + {b} = {ans}.",
            lambda a, b: a + b
        ),
        (
            "There are {a} {item} in one box and {b} {item} in another box. How many {item} are there in total?",
            "Total {item}: {a} + {b} = {ans}.",
            lambda a, b: a + b
        ),
        # Subtraction
        (
            "{name} has {a} {item}. {name} gives {b} {item} to {name2}. How many {item} does {name} have left?",
            "{name} had {a} {item}. Gave away {b}. {a} - {b} = {ans}.",
            lambda a, b: a - b
        ),
        # Multiplication
        (
            "There are {a} boxes with {b} {item} in each box. How many {item} are there in total?",
            "Total {item}: {a} × {b} = {ans}.",
            lambda a, b: a * b
        ),
        (
            "{name} buys {a} packs of {item}. Each pack has {b} {item}. How many {item} does {name} have?",
            "{name} has {a} × {b} = {ans} {item}.",
            lambda a, b: a * b
        ),
    ]

    names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
    items = ["apples", "oranges", "books", "pencils", "stickers", "marbles", "cookies", "cards"]

    for i in range(n):
        template_idx = i % len(templates)
        q_template, s_template, answer_func = templates[template_idx]

        name = random.choice(names)
        name2 = random.choice([n for n in names if n != name])
        item = random.choice(items)

        if template_idx in [0, 1]:  # Addition
            a = random.randint(5, 30)
            b = random.randint(3, 20)
        elif template_idx == 2:  # Subtraction
            a = random.randint(15, 50)
            b = random.randint(3, a - 1)
        else:  # Multiplication
            a = random.randint(2, 10)
            b = random.randint(3, 12)

        answer = answer_func(a, b)
        question = q_template.format(name=name, name2=name2, a=a, b=b, item=item)
        solution = s_template.format(name=name, name2=name2, a=a, b=b, item=item, ans=answer)
        solution += f"\n\\boxed{{{answer}}}"

        problems.append({
            "question": question,
            "answer": str(answer),
            "solution": solution,
            "level": 4,
            "num_steps": 1,
        })

    return problems


def generate_two_step_word_problems(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Level 5: Two-step word problems."""
    random.seed(seed)
    problems = []

    templates = [
        # Buy and give away
        (
            "{name} has {a} {item}. {name} buys {b} more {item}, then gives {c} to {name2}. How many {item} does {name} have now?",
            lambda a, b, c: (a + b, a + b - c),
            "First, {name} has {a} + {b} = {step1} {item}. Then gives away {c}: {step1} - {c} = {ans}."
        ),
        # Groups and extras
        (
            "There are {a} groups of students with {b} students in each group. {c} more students join. How many students are there in total?",
            lambda a, b, c: (a * b, a * b + c),
            "First, {a} × {b} = {step1} students. Then {c} more join: {step1} + {c} = {ans}."
        ),
        # Split and add
        (
            "{name} has {a} {item}. {name} gets {b} times as many more. Then {name2} gives {name} {c} extra. How many does {name} have?",
            lambda a, b, c: (a + a * b, a + a * b + c),
            "First, {name} gets {a} × {b} = {ab} more, total = {a} + {ab} = {step1}. Then +{c}: {step1} + {c} = {ans}."
        ),
    ]

    names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    items = ["apples", "books", "pencils", "stickers", "marbles", "cookies"]

    for i in range(n):
        template_idx = i % len(templates)
        q_template, compute_func, s_template = templates[template_idx]

        name = random.choice(names)
        name2 = random.choice([n for n in names if n != name])
        item = random.choice(items)

        if template_idx == 0:
            a = random.randint(10, 30)
            b = random.randint(5, 15)
            c = random.randint(3, a + b - 1)
            step1, ans = compute_func(a, b, c)
        elif template_idx == 1:
            a = random.randint(3, 8)
            b = random.randint(4, 10)
            c = random.randint(5, 20)
            step1, ans = compute_func(a, b, c)
        else:
            a = random.randint(5, 15)
            b = random.randint(2, 4)
            c = random.randint(5, 15)
            step1, ans = compute_func(a, b, c)

        question = q_template.format(name=name, name2=name2, a=a, b=b, c=c, item=item)

        if template_idx == 2:
            ab = a * b
            solution = s_template.format(name=name, a=a, b=b, c=c, ab=ab, step1=step1, ans=ans)
        else:
            solution = s_template.format(name=name, name2=name2, a=a, b=b, c=c, step1=step1, ans=ans)
        solution += f"\n\\boxed{{{ans}}}"

        problems.append({
            "question": question,
            "answer": str(ans),
            "solution": solution,
            "level": 5,
            "num_steps": 2,
        })

    return problems


def generate_all_levels(
    samples_per_level: int = 1000,
    output_dir: str = "data/curriculum",
    seed: int = 42
) -> Dict[int, List[Dict]]:
    """Generate all curriculum levels."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generators = {
        0: ("single_step", generate_single_step),
        1: ("two_step", generate_two_step),
        2: ("three_step", generate_three_step),
        3: ("four_step", generate_four_step),
        4: ("simple_word", generate_simple_word_problems),
        5: ("two_step_word", generate_two_step_word_problems),
    }

    all_data = {}

    for level, (name, generator) in generators.items():
        print(f"Generating level {level}: {name}...")
        data = generator(n=samples_per_level, seed=seed + level)
        all_data[level] = data

        # Save individual level
        level_file = output_path / f"level_{level}_{name}.jsonl"
        with open(level_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"  Saved {len(data)} problems to {level_file}")

    # Save combined file
    combined_file = output_path / "curriculum_all.jsonl"
    with open(combined_file, 'w') as f:
        for level in sorted(all_data.keys()):
            for item in all_data[level]:
                f.write(json.dumps(item) + '\n')
    print(f"\nSaved combined curriculum to {combined_file}")

    # Print summary
    print("\n" + "="*50)
    print("CURRICULUM SUMMARY")
    print("="*50)
    for level, data in sorted(all_data.items()):
        name = generators[level][0]
        print(f"Level {level} ({name}): {len(data)} problems")
        # Show example
        ex = data[0]
        print(f"  Example: {ex['question'][:60]}...")
        print(f"  Answer: {ex['answer']}")
    print("="*50)

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate curriculum learning data")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per level")
    parser.add_argument("--output", type=str, default="data/curriculum", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_all_levels(
        samples_per_level=args.samples,
        output_dir=args.output,
        seed=args.seed,
    )
