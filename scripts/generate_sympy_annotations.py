#!/usr/bin/env python3
"""
Generate SymPy annotations for GSM8K training data.

Converts GSM8K CoT traces into per-step SymPy expressions that can be used
for teacher forcing during training. Each step becomes a SymPy assignment
that can be executed to get verified intermediate values.

Usage:
    python scripts/generate_sympy_annotations.py
    python scripts/generate_sympy_annotations.py --max_samples 100  # for testing

Output:
    data/gsm8k_sympy_annotations.jsonl

Format:
    {"problem": "...", "gold_answer": 72, "sympy_steps": ["n_april = 48", "n_may = n_april / 2", ...]}
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers for number cleaning
# ---------------------------------------------------------------------------

def clean_number(s: str) -> str:
    """Clean number string: remove $, commas, whitespace."""
    s = s.strip()
    s = s.replace('$', '').replace(',', '').strip()
    return s


def parse_number(s: str) -> Optional[float]:
    """Parse a cleaned number string to float."""
    s = clean_number(s)
    try:
        return float(s)
    except ValueError:
        return None


def format_number(val: float) -> str:
    """Format number for SymPy: int if whole, else float."""
    if val == int(val):
        return str(int(val))
    return str(val)


# ---------------------------------------------------------------------------
# Extract final answer from GSM8K format
# ---------------------------------------------------------------------------

def parse_final_answer(answer_text: str) -> Optional[float]:
    """Extract the final numeric answer after ####."""
    m = re.search(r'####\s*(.+)', answer_text)
    if not m:
        return None
    try:
        return float(clean_number(m.group(1)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Clean CoT for processing
# ---------------------------------------------------------------------------

def clean_cot_text(answer_text: str) -> str:
    """Strip <<calc=result>> annotations and #### line for clean CoT text."""
    # Remove <<...>> calculator annotations
    cleaned = re.sub(r'<<.*?>>', '', answer_text)
    # Remove the #### final answer line
    cleaned = re.sub(r'\n####.*$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Equation extraction patterns
# ---------------------------------------------------------------------------

# Pattern for simple binary equations: A op B = C
# Handles: 48 / 2 = 24, $100 + $50 = $150, 1,000 - 500 = 500
BINARY_EQ_PATTERN = re.compile(
    r'(-?\$?\d[\d,]*\.?\d*)\s*([+\-*/x])\s*(-?\$?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)'
)

# Pattern for equations with text/units between numbers:
# "5 snakes/jaguar * 6 jaguars = 30 snakes"
# "12 beetles/bird * 3 birds = 36 beetles"
UNIT_EQ_PATTERN = re.compile(
    r'(\d[\d,]*\.?\d*)\s*[a-zA-Z/\s]+\s*([*x])\s*(\d[\d,]*\.?\d*)\s*[a-zA-Z/\s]*\s*=\s*(\d[\d,]*\.?\d*)'
)

# Pattern for percentage equations: X% * Y = Z  or X% of Y = Z
# Handles: 20% * 2 = .4, 50% of 100 = 50
PERCENT_EQ_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*%\s*(?:[*x]|of)\s*\$?(\d[\d,]*\.?\d*)\s*=\s*\.?(\d[\d,]*\.?\d*)'
)

# Pattern for chained equations: A op B op C = R
# Handles: 48 + 24 + 12 = 84
CHAINED_EQ_PATTERN = re.compile(
    r'(\d[\d,]*\.?\d*(?:\s*[+\-*/x]\s*\d[\d,]*\.?\d*)+)\s*=\s*\$?(-?\d[\d,]*\.?\d*)'
)

# Pattern for decimal results: .4, .25, etc (with leading dot)
DECIMAL_PATTERN = re.compile(r'=\s*(\.\d+)')

# Pattern for fraction: X/Y of Z  or  "half of X"
FRACTION_WORDS = {
    'half': 0.5,
    'third': 1/3,
    'quarter': 0.25,
    'fifth': 0.2,
    'tenth': 0.1,
    'twice': 2,
    'double': 2,
    'triple': 3,
}


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def parse_chained_equation(lhs: str, result: str) -> List[Tuple[str, str, str, str]]:
    """
    Parse a chained equation like "100 - 50 - 30 - 15 = 5" into binary operations.

    Returns a list of (left, op, right, intermediate_result) tuples.
    Evaluates left-to-right (no precedence) as GSM8K CoT typically does.
    """
    # Tokenize: extract numbers and operators
    tokens = re.findall(r'(-?\d[\d,]*\.?\d*|[+\-*/x])', lhs)
    if len(tokens) < 3:
        return []

    equations = []
    try:
        # Start with first number
        current_value = float(clean_number(tokens[0]))
        i = 1
        while i < len(tokens) - 1:
            op = tokens[i]
            if op == 'x':
                op = '*'
            operand = float(clean_number(tokens[i + 1]))

            # Compute intermediate result
            if op == '+':
                new_value = current_value + operand
            elif op == '-':
                new_value = current_value - operand
            elif op == '*':
                new_value = current_value * operand
            elif op == '/':
                if operand == 0:
                    return []
                new_value = current_value / operand
            else:
                i += 2
                continue

            equations.append((
                format_number(current_value),
                op,
                format_number(operand),
                format_number(new_value)
            ))
            current_value = new_value
            i += 2

    except (ValueError, IndexError):
        return []

    return equations


def extract_equations_from_cot(cot_text: str) -> List[Tuple[str, str, str, str]]:
    """
    Extract all equations from CoT text.

    Returns list of (left_operand, operator, right_operand, result) tuples.
    Handles multiple equation formats found in GSM8K CoT traces.
    """
    equations = []
    seen_positions = set()  # Track matched positions to avoid duplicates

    # 0. First, find chained equations: A op B op C ... = R
    # These need special handling to break into binary operations
    for match in CHAINED_EQ_PATTERN.finditer(cot_text):
        lhs = match.group(1)
        result = clean_number(match.group(2))

        # Check if this is actually a multi-operation chain (not just binary)
        ops_count = len(re.findall(r'[+\-*/x]', lhs))
        if ops_count >= 2:
            pos = (match.start(), match.end())
            if pos not in seen_positions:
                seen_positions.add(pos)
                chain_eqs = parse_chained_equation(lhs, result)
                equations.extend(chain_eqs)

    # 1. Find all simple binary equations: A op B = C
    for match in BINARY_EQ_PATTERN.finditer(cot_text):
        pos = (match.start(), match.end())
        # Check for overlap with already processed chained equations
        overlaps = any(
            not (pos[1] <= existing[0] or pos[0] >= existing[1])
            for existing in seen_positions
        )
        if overlaps:
            continue
        seen_positions.add(pos)

        left = clean_number(match.group(1))
        op = match.group(2)
        if op == 'x':
            op = '*'
        right = clean_number(match.group(3))
        result = clean_number(match.group(4))
        equations.append((left, op, right, result))

    # 2. Find equations with units/text: "5 snakes * 6 jaguars = 30"
    for match in UNIT_EQ_PATTERN.finditer(cot_text):
        pos = (match.start(), match.end())
        overlaps = any(
            not (pos[1] <= existing[0] or pos[0] >= existing[1])
            for existing in seen_positions
        )
        if overlaps:
            continue
        seen_positions.add(pos)

        left = clean_number(match.group(1))
        op = match.group(2)
        if op == 'x':
            op = '*'
        right = clean_number(match.group(3))
        result = clean_number(match.group(4))
        equations.append((left, op, right, result))

    # 3. Find percentage equations: 20% * 2 = .4
    for match in PERCENT_EQ_PATTERN.finditer(cot_text):
        pos = (match.start(), match.end())
        overlaps = any(
            not (pos[1] <= existing[0] or pos[0] >= existing[1])
            for existing in seen_positions
        )
        if overlaps:
            continue
        seen_positions.add(pos)

        percent = clean_number(match.group(1))
        base = clean_number(match.group(2))
        result = clean_number(match.group(3))

        # Convert: X% * Y = Z  becomes  (X/100) * Y
        percent_decimal = str(float(percent) / 100)
        equations.append((percent_decimal, '*', base, result))

    return equations


def fuzzy_lookup(value_to_var: Dict[float, str], target: float, tolerance: float = 0.01) -> Optional[str]:
    """
    Look up a variable by value with fuzzy matching.
    Returns the variable name if found within tolerance, else None.
    """
    # Try exact match first
    if target in value_to_var:
        return value_to_var[target]

    # Try integer conversion
    if target == int(target):
        int_target = float(int(target))
        if int_target in value_to_var:
            return value_to_var[int_target]

    # Try fuzzy matching
    for val, var_name in value_to_var.items():
        if abs(val - target) < tolerance:
            return var_name
        # Also try relative tolerance for larger numbers
        if target != 0 and abs(val - target) / abs(target) < tolerance:
            return var_name

    return None


def cot_to_sympy_steps(cot_trace: str, gold_answer: float) -> List[str]:
    """
    Convert a CoT trace into per-step SymPy expressions.

    Strategy:
    1. Parse equations from CoT (regex for "X op Y = Z" patterns)
    2. Assign variable names (v1, v2, ...)
    3. Build dependency chain (if operand matches previous result, use variable)
    4. Add final "answer = ..." step

    Returns:
        List of SymPy assignment strings, e.g.:
        ["v1 = 48", "v2 = v1 / 2", "v3 = v1 + v2", "answer = v3"]
    """
    # Clean the CoT text
    clean_cot = clean_cot_text(cot_trace)

    # Extract all equations
    equations = extract_equations_from_cot(clean_cot)

    if not equations:
        # No equations found - return empty (will be handled as fallback)
        return []

    steps = []
    # Map from numeric value to variable name
    # Use OrderedDict to maintain insertion order for reproducibility
    value_to_var: Dict[float, str] = OrderedDict()
    var_counter = 0

    for left_str, op, right_str, result_str in equations:
        left_val = parse_number(left_str)
        right_val = parse_number(right_str)
        result_val = parse_number(result_str)

        if left_val is None or right_val is None or result_val is None:
            continue

        # Check if operands match previous results (with fuzzy matching)
        left_var = fuzzy_lookup(value_to_var, left_val)
        right_var = fuzzy_lookup(value_to_var, right_val)

        left_ref = left_var if left_var else format_number(left_val)
        right_ref = right_var if right_var else format_number(right_val)

        # Create new variable for result
        var_counter += 1
        var_name = f"v{var_counter}"

        # Build the SymPy expression
        sympy_code = f"{var_name} = {left_ref} {op} {right_ref}"
        steps.append(sympy_code)

        # Register the result value -> variable mapping
        value_to_var[result_val] = var_name

    # Add answer step
    # Check if gold_answer matches any computed variable
    gold_answer_float = float(gold_answer)

    # Use fuzzy lookup to find matching variable
    answer_var = fuzzy_lookup(value_to_var, gold_answer_float)

    if answer_var:
        steps.append(f"answer = {answer_var}")
    elif var_counter > 0:
        # Use the last computed variable (most likely the answer)
        steps.append(f"answer = v{var_counter}")
    else:
        # No variables computed, use literal answer
        steps.append(f"answer = {format_number(gold_answer_float)}")

    return steps


def validate_sympy_steps(steps: List[str], expected_answer: float) -> bool:
    """
    Validate that executing the SymPy steps produces the expected answer.

    Returns True if the steps execute correctly and produce the right answer.
    """
    if not steps:
        return False

    try:
        namespace = {}
        for step in steps:
            exec(step, {"__builtins__": {}}, namespace)

        if 'answer' not in namespace:
            return False

        computed = float(namespace['answer'])

        # Allow small absolute tolerance for small numbers
        if abs(computed - expected_answer) < 0.01:
            return True

        # Allow relative tolerance for larger numbers (1%)
        if expected_answer != 0:
            rel_error = abs(computed - expected_answer) / abs(expected_answer)
            if rel_error < 0.01:
                return True

        # Check integer comparison (handles floating point issues like 72.0 vs 72)
        if abs(computed - expected_answer) < 1:
            if int(round(computed)) == int(round(expected_answer)):
                return True

        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Process full dataset
# ---------------------------------------------------------------------------

def process_gsm8k(max_samples: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    """
    Process GSM8K dataset and generate SymPy annotations.

    Returns:
        (annotations, stats)
        - annotations: list of dicts with problem, gold_answer, sympy_steps
        - stats: dict with conversion statistics
    """
    print("Loading GSM8K dataset from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    annotations = []
    stats = {
        'total': 0,
        'converted': 0,
        'validated': 0,
        'no_final_answer': 0,
        'no_equations': 0,
        'validation_failed': 0,
        'steps_counts': [],
    }

    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break

        stats['total'] += 1

        question = example['question']
        answer_text = example['answer']

        # Extract final answer
        gold_answer = parse_final_answer(answer_text)
        if gold_answer is None:
            stats['no_final_answer'] += 1
            annotations.append({
                'problem': question,
                'gold_answer': None,
                'sympy_steps': [],
                'raw_cot': clean_cot_text(answer_text),
            })
            continue

        # Convert to SymPy steps
        sympy_steps = cot_to_sympy_steps(answer_text, gold_answer)

        if not sympy_steps:
            stats['no_equations'] += 1
        else:
            stats['converted'] += 1
            stats['steps_counts'].append(len(sympy_steps))

            # Validate the steps
            if validate_sympy_steps(sympy_steps, gold_answer):
                stats['validated'] += 1
            else:
                stats['validation_failed'] += 1

        # Format gold_answer as int if whole
        gold_answer_fmt = int(gold_answer) if gold_answer == int(gold_answer) else gold_answer

        annotations.append({
            'problem': question,
            'gold_answer': gold_answer_fmt,
            'sympy_steps': sympy_steps,
            'raw_cot': clean_cot_text(answer_text),
        })

    return annotations, stats


def print_stats(stats: Dict):
    """Print conversion statistics."""
    print("\n" + "=" * 60)
    print("CONVERSION STATISTICS")
    print("=" * 60)
    print(f"Total problems:        {stats['total']}")
    print(f"Successfully converted: {stats['converted']} ({100*stats['converted']/max(1,stats['total']):.1f}%)")
    print(f"Validated correct:     {stats['validated']} ({100*stats['validated']/max(1,stats['total']):.1f}%)")
    print(f"No final answer:       {stats['no_final_answer']}")
    print(f"No equations found:    {stats['no_equations']}")
    print(f"Validation failed:     {stats['validation_failed']}")

    if stats['steps_counts']:
        avg_steps = sum(stats['steps_counts']) / len(stats['steps_counts'])
        min_steps = min(stats['steps_counts'])
        max_steps = max(stats['steps_counts'])
        print(f"\nSteps per problem:")
        print(f"  Average: {avg_steps:.2f}")
        print(f"  Min:     {min_steps}")
        print(f"  Max:     {max_steps}")

        # Distribution
        from collections import Counter
        dist = Counter(stats['steps_counts'])
        print(f"\nSteps distribution:")
        for n_steps in sorted(dist.keys()):
            count = dist[n_steps]
            bar = '#' * min(50, count // 10)
            print(f"  {n_steps} steps: {count:5d} {bar}")


def save_annotations(annotations: List[Dict], output_path: Path):
    """Save annotations to JSONL file."""
    with open(output_path, 'w') as f:
        for ann in annotations:
            # Remove raw_cot from output (only used for debugging)
            output = {
                'problem': ann['problem'],
                'gold_answer': ann['gold_answer'],
                'sympy_steps': ann['sympy_steps'],
            }
            f.write(json.dumps(output) + '\n')
    print(f"\nSaved {len(annotations)} annotations to {output_path}")


def show_examples(annotations: List[Dict], n: int = 5):
    """Show example conversions."""
    print("\n" + "=" * 60)
    print("EXAMPLE CONVERSIONS")
    print("=" * 60)

    # Show some successful conversions
    successful = [a for a in annotations if a['sympy_steps']][:n]

    for i, ann in enumerate(successful):
        print(f"\n--- Example {i+1} ---")
        print(f"Problem: {ann['problem'][:200]}...")
        print(f"Gold answer: {ann['gold_answer']}")
        print(f"SymPy steps:")
        for step in ann['sympy_steps']:
            print(f"  {step}")

        # Validate
        if ann['gold_answer'] is not None:
            valid = validate_sympy_steps(ann['sympy_steps'], ann['gold_answer'])
            print(f"Validates: {'YES' if valid else 'NO'}")


def show_failures(annotations: List[Dict], n: int = 3):
    """Show examples that failed conversion."""
    print("\n" + "=" * 60)
    print("FAILED CONVERSIONS (for debugging)")
    print("=" * 60)

    # Show some that have no equations
    no_eq = [a for a in annotations if not a['sympy_steps'] and a['gold_answer'] is not None][:n]

    for i, ann in enumerate(no_eq):
        print(f"\n--- Failure {i+1} ---")
        print(f"Problem: {ann['problem'][:150]}...")
        print(f"Gold answer: {ann['gold_answer']}")
        print(f"Raw CoT: {ann['raw_cot'][:300]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate SymPy annotations for GSM8K')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--output', type=str, default='data/gsm8k_sympy_annotations.jsonl',
                        help='Output file path')
    parser.add_argument('--show_examples', type=int, default=5,
                        help='Number of examples to show')
    args = parser.parse_args()

    # Process dataset
    annotations, stats = process_gsm8k(args.max_samples)

    # Print statistics
    print_stats(stats)

    # Show examples
    show_examples(annotations, args.show_examples)
    show_failures(annotations, n=3)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_annotations(annotations, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
