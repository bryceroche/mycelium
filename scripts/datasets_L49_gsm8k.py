"""
Datasets for bridging procedural word problems to full GSM8K.

1. L49GSM8KEasyDataset — Easy GSM8K subset (<=3 steps, max number < 1000)
2. GSM8KAugmentedDataset — Full GSM8K with number-swapping augmentation per epoch
"""
import re
import random
from torch.utils.data import Dataset
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers (compatible with train_dual_lora_gsm8k.py)
# ---------------------------------------------------------------------------

def parse_final(answer_text):
    """Extract the final numeric answer after ####."""
    m = re.search(r'####\s*(.+)', answer_text)
    if not m:
        return None
    try:
        return float(m.group(1).strip().replace(',', ''))
    except ValueError:
        return None


def clean_cot(answer_text):
    """Strip <<calc=result>> annotations and #### line for clean CoT target."""
    cleaned = re.sub(r'<<.*?>>', '', answer_text)
    cleaned = re.sub(r'\n####.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    final = parse_final(answer_text)
    if final is not None:
        final_str = str(int(final)) if final == int(final) else str(final)
        cleaned += f" The answer is {final_str}."
    return cleaned


def _extract_numbers(text):
    """Find all numbers (int/float, possibly with commas) in text.

    Returns list of (match_object, float_value) so we can do positional
    replacement later.
    """
    results = []
    for m in re.finditer(r'-?\d[\d,]*\.?\d*', text):
        try:
            val = float(m.group().replace(',', ''))
            results.append((m, val))
        except ValueError:
            continue
    return results


def _max_number_in_text(text):
    """Return the largest absolute number found in text, or 0."""
    nums = _extract_numbers(text)
    if not nums:
        return 0
    return max(abs(v) for _, v in nums)


def _count_steps(answer_text):
    """Count reasoning steps = number of non-empty lines before ####."""
    before_hash = re.sub(r'\n####.*$', '', answer_text, flags=re.MULTILINE)
    lines = [l.strip() for l in before_hash.strip().split('\n') if l.strip()]
    return len(lines)


# ---------------------------------------------------------------------------
# 1. L4.9 GSM8K Easy Subset
# ---------------------------------------------------------------------------

class L49GSM8KEasyDataset(Dataset):
    """GSM8K filtered for easy problems: <=3 reasoning steps AND
    max number in solution < 1000.

    This bridges procedural word problems (L4) to full GSM8K (L5) by
    exposing the model to real GSM8K formatting and language on problems
    simple enough for the current architecture.
    """

    def __init__(self, split='train', max_samples=None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.samples = []
        total = 0
        skipped_no_final = 0
        skipped_steps = 0
        skipped_numbers = 0

        for ex in ds:
            total += 1
            final = parse_final(ex['answer'])
            if final is None:
                skipped_no_final += 1
                continue

            # Count steps (lines before ####)
            steps = _count_steps(ex['answer'])
            if steps > 3:
                skipped_steps += 1
                continue

            # Max number in the FULL answer text (includes calc annotations)
            max_num = _max_number_in_text(ex['answer'])
            if max_num >= 1000:
                skipped_numbers += 1
                continue

            cot = clean_cot(ex['answer'])
            final_val = int(final) if final == int(final) else final
            self.samples.append({
                'problem': ex['question'],
                'answer': cot,
                'final': final_val,
            })
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"L4.9 GSM8K Easy: {len(self.samples)}/{total} pass filter "
              f"(skipped: {skipped_no_final} no-final, "
              f"{skipped_steps} >3 steps, {skipped_numbers} numbers>=1000)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 2. GSM8K Augmented Dataset
# ---------------------------------------------------------------------------

def _augment_problem(question, answer_text, factor, rng):
    """Augment a GSM8K problem by scaling all numbers by `factor`.

    Strategy:
    1. Find all numbers in the problem text.
    2. Find each step's equation in the answer (pattern: "X op Y = Z").
    3. Replace numbers in problem text with scaled versions.
    4. For each step in the CoT, re-parse the equation and recompute
       using the new (scaled) input numbers.
    5. If any result is non-integer or negative, return None (caller
       uses original).

    Returns (new_question, new_answer_text) or None on failure.
    """
    if factor == 1.0:
        return question, answer_text

    # --- Step 1: Build a mapping old_number -> new_number for problem text ---
    prob_nums = _extract_numbers(question)
    if not prob_nums:
        return None  # nothing to augment

    num_map = {}  # old_float -> new_float
    for _, val in prob_nums:
        if val == 0:
            continue
        new_val = val * factor
        # Round to integer
        new_val = round(new_val)
        if new_val <= 0:
            return None
        num_map[val] = float(new_val)

    # --- Step 2: Replace numbers in problem text ---
    new_question = _replace_numbers_in_text(question, num_map)

    # --- Step 3: Rebuild the answer/CoT ---
    # Split into lines before #### and the #### line
    parts = answer_text.split('####')
    if len(parts) < 2:
        return None

    cot_part = parts[0]
    # Build a complete number map including computed values
    full_map = dict(num_map)

    # Process line by line, recomputing equations
    new_lines = []
    for line in cot_part.split('\n'):
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue

        new_line = _recompute_line(line, full_map, factor)
        if new_line is None:
            return None  # recomputation failed
        new_lines.append(new_line)

    # Recompute final answer
    original_final = parse_final(answer_text)
    if original_final is None:
        return None
    new_final = original_final * factor
    new_final = round(new_final)
    if new_final <= 0:
        return None

    # For multi-step problems the final answer is usually the result of
    # chained operations. Rather than trying to perfectly recompute every
    # chain, we find the last equation in the CoT and use its result as
    # the final answer. If that fails, fall back to scaled original.
    last_result = _find_last_equation_result('\n'.join(new_lines))
    if last_result is not None:
        new_final = last_result

    new_answer = '\n'.join(new_lines) + f'#### {_format_number(new_final)}'
    return new_question, new_answer


def _format_number(val):
    """Format number: integer if whole, else float."""
    if val == int(val):
        return str(int(val))
    return str(val)


def _replace_numbers_in_text(text, num_map):
    """Replace numbers in text using num_map (old_float -> new_float).

    Replaces from right to left to preserve positions.
    """
    matches = list(re.finditer(r'-?\d[\d,]*\.?\d*', text))
    # Process from right to left so positions stay valid
    result = text
    for m in reversed(matches):
        try:
            old_val = float(m.group().replace(',', ''))
        except ValueError:
            continue
        if old_val in num_map:
            new_val = num_map[old_val]
            result = result[:m.start()] + _format_number(new_val) + result[m.end():]
    return result


def _recompute_line(line, full_map, factor):
    """Recompute a single CoT line, updating full_map with any new results.

    Handles patterns like:
    - "36 / 3 = 12"  (equation)
    - "She has 36 - 12 = 24 apples left"  (equation in context)
    - Plain text with numbers

    Returns the new line, or None if recomputation produces invalid results.
    """
    # Strip <<calc>> annotations for processing
    clean_line = re.sub(r'<<.*?>>', '', line)

    # Try to find an equation: A op B = C
    eq_match = re.search(
        r'(-?\d[\d,]*\.?\d*)\s*([+\-*/x])\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)',
        clean_line
    )

    if eq_match:
        try:
            a_old = float(eq_match.group(1).replace(',', ''))
            op = eq_match.group(2)
            b_old = float(eq_match.group(3).replace(',', ''))
            c_old = float(eq_match.group(4).replace(',', ''))
        except ValueError:
            return _replace_numbers_in_text(line, full_map)

        # Map old values to new values
        a_new = full_map.get(a_old, a_old * factor)
        b_new = full_map.get(b_old, b_old * factor)
        a_new = round(a_new)
        b_new = round(b_new)

        # Recompute
        if op in ('+',):
            c_new = a_new + b_new
        elif op in ('-',):
            c_new = a_new - b_new
        elif op in ('*', 'x'):
            c_new = a_new * b_new
        elif op in ('/',):
            if b_new == 0:
                return None
            c_new = a_new / b_new
            if c_new != int(c_new):
                return None  # non-integer result
            c_new = int(c_new)
        else:
            return _replace_numbers_in_text(line, full_map)

        if c_new < 0:
            return None

        c_new = float(c_new)

        # Register new mappings
        full_map[a_old] = float(a_new)
        full_map[b_old] = float(b_new)
        full_map[c_old] = c_new

        # Now replace all numbers in the line using the updated map
        return _replace_numbers_in_text(line, full_map)
    else:
        # No equation — just replace known numbers
        return _replace_numbers_in_text(line, full_map)


def _find_last_equation_result(text):
    """Find the result of the last equation (A op B = C) in text."""
    matches = list(re.finditer(
        r'(-?\d[\d,]*\.?\d*)\s*[+\-*/x]\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)',
        text
    ))
    if not matches:
        return None
    try:
        val = float(matches[-1].group(3).replace(',', ''))
        return val
    except ValueError:
        return None


class GSM8KAugmentedDataset(Dataset):
    """Full GSM8K with per-epoch number augmentation.

    Each epoch_seed produces a different augmentation of each problem.
    Numbers in both problem and CoT are multiplied by a random factor,
    equations are recomputed. If augmentation fails (non-integer result,
    negative), the original problem is kept.
    """

    FACTORS = [0.5, 1.5, 2, 3]

    def __init__(self, split='train', epoch_seed=0, max_samples=None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.raw = []
        for ex in ds:
            final = parse_final(ex['answer'])
            if final is None:
                continue
            self.raw.append({
                'question': ex['question'],
                'answer_text': ex['answer'],
                'final': final,
            })
            if max_samples and len(self.raw) >= max_samples:
                break

        self.epoch_seed = epoch_seed
        self.samples = self._augment_all()
        print(f"GSM8K Augmented (seed={epoch_seed}): {len(self.samples)} problems "
              f"({self._aug_count} augmented, {len(self.samples) - self._aug_count} original)")

    def _augment_all(self):
        rng = random.Random(self.epoch_seed)
        samples = []
        self._aug_count = 0

        for item in self.raw:
            factor = rng.choice(self.FACTORS)
            result = _augment_problem(
                item['question'], item['answer_text'], factor, rng
            )
            if result is not None:
                new_q, new_a = result
                new_final = parse_final(new_a)
                if new_final is not None:
                    cot = clean_cot(new_a)
                    final_val = int(new_final) if new_final == int(new_final) else new_final
                    samples.append({
                        'problem': new_q,
                        'answer': cot,
                        'final': final_val,
                    })
                    if factor != 1.0:
                        self._aug_count += 1
                    continue

            # Fallback: use original
            cot = clean_cot(item['answer_text'])
            final_val = int(item['final']) if item['final'] == int(item['final']) else item['final']
            samples.append({
                'problem': item['question'],
                'answer': cot,
                'final': final_val,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Main: demo both datasets
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("L4.9 GSM8K Easy Subset")
    print("=" * 70)

    l49 = L49GSM8KEasyDataset(split='train')
    print(f"\nTotal easy problems: {len(l49)}")
    print(f"\nFirst 5 examples:")
    for i in range(min(5, len(l49))):
        s = l49[i]
        print(f"\n--- Example {i+1} ---")
        print(f"  Problem: {s['problem']}")
        print(f"  Answer:  {s['answer']}")
        print(f"  Final:   {s['final']}")

    print("\n" + "=" * 70)
    print("GSM8K Augmented Dataset")
    print("=" * 70)

    aug0 = GSM8KAugmentedDataset(split='train', epoch_seed=0, max_samples=20)
    aug1 = GSM8KAugmentedDataset(split='train', epoch_seed=42, max_samples=20)

    print(f"\nShowing same problem with two different seeds:")
    for i in range(min(3, len(aug0))):
        print(f"\n--- Problem {i+1} ---")
        print(f"  [seed=0]  Problem: {aug0[i]['problem']}")
        print(f"            Answer:  {aug0[i]['answer']}")
        print(f"            Final:   {aug0[i]['final']}")
        print(f"  [seed=42] Problem: {aug1[i]['problem']}")
        print(f"            Answer:  {aug1[i]['answer']}")
        print(f"            Final:   {aug1[i]['final']}")
