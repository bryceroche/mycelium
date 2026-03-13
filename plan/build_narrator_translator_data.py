"""
Build training data for the narrator/translator assembly line.

LoRA C (Narrator): Problem text + scaffold + previous → natural language description
    "What does this step do?" → "Find the prime factorization of 196"

LoRA D (Translator): Natural language task + values → rough SymPy expression
    "Find the prime factorization of 196" → "factorint(196)"

Both models do natural text continuation. Neither makes the full jump
from problem text to SymPy.

Source data:
    - parsed_steps.jsonl (50K CoT steps with raw_cot_text, operands, results)
    - iaf_training.jsonl (80K IAF-grounded operand mappings)
    - verified_training.jsonl (4,182 execution-verified steps)

Usage:
    python build_narrator_translator_data.py \
        --parsed-steps parsed_steps.jsonl \
        --verified verified_training.jsonl \
        --output-dir /tmp/narrator_translator/
"""

import argparse
import json
import re
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Extract narration from CoT text
# ─────────────────────────────────────────────────────────────

def extract_narration(raw_cot_text: str) -> Optional[str]:
    """
    Extract the first natural language sentence from CoT text.
    This is what the narrator should learn to produce.

    "Next, we expand (x+y)^2 to get x^2 + 2xy + y^2" 
        → "Expand (x+y)^2"

    "We need to find the prime factorization of 196"
        → "Find the prime factorization of 196"

    "Substituting x = 3 into the equation"
        → "Substitute x = 3 into the equation"
    """
    if not raw_cot_text or not raw_cot_text.strip():
        return None

    text = raw_cot_text.strip()

    # Strip LaTeX delimiters
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$$", "")

    # Take first sentence
    for delim in [". ", ".\n", ":\n", ";"]:
        if delim in text:
            text = text[:text.index(delim)]
            break

    # Clean up common prefixes
    for prefix in ["Next, ", "Then, ", "Now, ", "So, ", "Thus, ",
                    "First, ", "Finally, ", "Therefore, ",
                    "We need to ", "We can ", "We have ",
                    "We know that ", "We see that ",
                    "Let's ", "Let us "]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]
            break

    # Truncate to reasonable length
    if len(text) > 150:
        # Find a natural break point
        for i in range(100, min(150, len(text))):
            if text[i] in " ,":
                text = text[:i]
                break
        else:
            text = text[:150]

    text = text.strip()

    # Reject if too short or looks like pure math
    if len(text) < 5:
        return None

    # Reject if it's all math symbols (no natural language)
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.3:
        return None

    return text


def extract_expression_from_cot(raw_cot_text: str, telegram: str) -> Optional[str]:
    """
    Extract the mathematical expression from the CoT step.
    If we have a verified telegram, use that. Otherwise extract from CoT.
    """
    if telegram:
        return telegram.strip()

    # Try to find math expression in CoT
    # Look for content between LaTeX delimiters
    import re
    latex_matches = re.findall(r'\\\((.*?)\\\)', raw_cot_text)
    if latex_matches:
        return latex_matches[-1].strip()  # usually the result is last

    bracket_matches = re.findall(r'\\\[(.*?)\\\]', raw_cot_text)
    if bracket_matches:
        return bracket_matches[-1].strip()

    return None


def extract_values_from_step(step: dict, previous_results: List[str]) -> str:
    """
    Build a "values available" string for the translator.
    Combines operands from the step + previous results.
    """
    values = []

    # Operands from the step
    operands = step.get("operands", [])
    for op in operands:
        op_str = str(op).strip()
        if op_str and op_str != "_prev":
            values.append(op_str)

    # Recent previous results
    for prev in previous_results[-3:]:
        if prev and prev != "None":
            values.append(f"prev={prev}")

    if not values:
        return "none"

    return ", ".join(values)


# ─────────────────────────────────────────────────────────────
# Build LoRA C data (Narrator)
# ─────────────────────────────────────────────────────────────

SCAFFOLD_TO_VERB_HINT = {
    "SETUP": "State or define",
    "COMPUTE": "Calculate or evaluate",
    "SIMPLIFY": "Simplify",
    "EXPAND": "Expand",
    "SUBSTITUTE": "Substitute values into",
    "SOLVE": "Solve for",
    "THEOREM": "Apply a theorem or formula to",
    "OTHER": "Perform",
}


def build_narrator_data(steps_by_problem: Dict[str, List[dict]],
                         problem_texts: Dict[str, str]) -> List[dict]:
    """
    Build LoRA C training data.

    Input: problem text + scaffold type + previous steps
    Target: natural language description of what this step does

    This is pure text continuation — LMs are great at this.
    """
    training = []
    skipped = {"no_narration": 0, "too_short": 0, "no_problem": 0}

    for pid, steps in steps_by_problem.items():
        steps.sort(key=lambda s: s.get("step_idx", 0))

        problem_text = problem_texts.get(pid, "")
        if not problem_text:
            # Try to get from step data
            for s in steps:
                if s.get("problem_text"):
                    problem_text = s["problem_text"]
                    break

        if not problem_text:
            skipped["no_problem"] += 1
            continue

        previous_narrations = []
        previous_results = []

        for step in steps:
            step_idx = step.get("step_idx", 0)
            scaffold_type = step.get("scaffold_type", step.get("step_type", "COMPUTE"))
            total_steps = len(steps)

            # Extract narration target from CoT
            narration = extract_narration(step.get("raw_cot_text", ""))
            if narration is None:
                skipped["no_narration"] += 1
                # Still track for chaining
                result = step.get("result")
                if result:
                    previous_results.append(str(result))
                continue

            if len(narration) < 8:
                skipped["too_short"] += 1
                result = step.get("result")
                if result:
                    previous_results.append(str(result))
                continue

            # Build previous context
            prev_str = ""
            if previous_narrations:
                prev_lines = []
                for narr, res in zip(previous_narrations[-3:], previous_results[-3:]):
                    if res and res != "None":
                        prev_lines.append(f"- {narr} → {res}")
                    else:
                        prev_lines.append(f"- {narr}")
                prev_str = "Previous steps:\n" + "\n".join(prev_lines) + "\n"

            # Scaffold hint
            verb_hint = SCAFFOLD_TO_VERB_HINT.get(scaffold_type, "Perform")

            prompt = (
                f"Problem: {problem_text}\n"
                f"Step type: {scaffold_type} (step {step_idx + 1} of {total_steps})\n"
                f"{prev_str}"
                f"What does this step do?\n"
            )

            training.append({
                "problem_id": pid,
                "step_idx": step_idx,
                "scaffold_type": scaffold_type,
                "prompt": prompt,
                "target": f" {narration}\n",
                "model": "narrator",
            })

            # Track for chaining
            previous_narrations.append(narration)
            result = step.get("result")
            previous_results.append(str(result) if result else "None")

    print(f"  Narrator examples: {len(training)}")
    print(f"  Skipped: {skipped}")

    return training


# ─────────────────────────────────────────────────────────────
# Build LoRA D data (Translator)
# ─────────────────────────────────────────────────────────────

def build_translator_data(steps_by_problem: Dict[str, List[dict]],
                           verified_telegrams: Dict[str, dict]) -> List[dict]:
    """
    Build LoRA D training data.

    Input: natural language task description + available values
    Target: rough SymPy expression

    The jump is small: "Find the prime factorization of 196" → "factorint(196)"
    """
    training = []
    skipped = {"no_narration": 0, "no_expression": 0}

    for pid, steps in steps_by_problem.items():
        steps.sort(key=lambda s: s.get("step_idx", 0))

        previous_results = []

        for step in steps:
            step_idx = step.get("step_idx", 0)
            scaffold_type = step.get("scaffold_type", step.get("step_type", "COMPUTE"))

            # Get narration (same extraction as narrator)
            narration = extract_narration(step.get("raw_cot_text", ""))
            if narration is None:
                skipped["no_narration"] += 1
                result = step.get("result")
                if result:
                    previous_results.append(str(result))
                continue

            # Get expression target
            # Prefer verified telegram if available
            verify_key = f"{pid}_{step_idx}"
            telegram = None
            if verify_key in verified_telegrams:
                telegram = verified_telegrams[verify_key].get("telegram")

            if telegram is None:
                telegram = extract_expression_from_cot(
                    step.get("raw_cot_text", ""),
                    step.get("telegram", "")
                )

            if telegram is None:
                skipped["no_expression"] += 1
                result = step.get("result")
                if result:
                    previous_results.append(str(result))
                continue

            # Build values string
            values_str = extract_values_from_step(step, previous_results)

            # Build prompt — narration + values → expression
            prompt = (
                f"Task: {narration}\n"
                f"Values: {values_str}\n"
                f"Write as math expression:\n"
            )

            training.append({
                "problem_id": pid,
                "step_idx": step_idx,
                "scaffold_type": scaffold_type,
                "prompt": prompt,
                "target": f" {telegram}\n",
                "narration": narration,
                "model": "translator",
            })

            # Track for chaining
            result = step.get("result")
            previous_results.append(str(result) if result else "None")

    print(f"  Translator examples: {len(training)}")
    print(f"  Skipped: {skipped}")

    return training


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build narrator + translator training data"
    )
    parser.add_argument("--parsed-steps", required=True,
                        help="parsed_steps.jsonl (50K CoT steps)")
    parser.add_argument("--verified", default=None,
                        help="verified_training.jsonl (for better translator targets)")
    parser.add_argument("--output-dir", default="/tmp/narrator_translator/",
                        help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load parsed steps
    print("Loading parsed steps...")
    all_steps = []
    with open(args.parsed_steps) as f:
        for line in f:
            if line.strip():
                all_steps.append(json.loads(line))
    print(f"  Loaded {len(all_steps)} steps")

    # Group by problem
    steps_by_problem = defaultdict(list)
    problem_texts = {}
    for step in all_steps:
        pid = step.get("problem_id", "unknown")
        steps_by_problem[pid].append(step)
        if step.get("problem_text"):
            problem_texts[pid] = step["problem_text"]

    print(f"  {len(steps_by_problem)} unique problems")

    # Load verified telegrams for better translator targets
    verified_telegrams = {}
    if args.verified:
        print("Loading verified telegrams...")
        with open(args.verified) as f:
            for line in f:
                if line.strip():
                    v = json.loads(line)
                    key = f"{v.get('problem_id')}_{v.get('step_idx')}"
                    verified_telegrams[key] = v
        print(f"  Loaded {len(verified_telegrams)} verified telegrams")

    # Build narrator data (LoRA C)
    print("\n═══ Building Narrator (LoRA C) ═══")
    narrator_data = build_narrator_data(steps_by_problem, problem_texts)

    # Build translator data (LoRA D)
    print("\n═══ Building Translator (LoRA D) ═══")
    translator_data = build_translator_data(steps_by_problem, verified_telegrams)

    # Split and save
    def split_save(data, name):
        random.shuffle(data)
        n_val = max(1, int(len(data) * args.val_split))
        train = data[n_val:]
        val = data[:n_val]

        train_path = Path(args.output_dir) / f"{name}_train.jsonl"
        val_path = Path(args.output_dir) / f"{name}_val.jsonl"

        with open(train_path, "w") as f:
            for item in train:
                f.write(json.dumps(item) + "\n")
        with open(val_path, "w") as f:
            for item in val:
                f.write(json.dumps(item) + "\n")

        print(f"  {name}: {len(train)} train, {len(val)} val")
        return train_path, val_path

    print("\n═══ Saving ═══")
    split_save(narrator_data, "narrator")
    split_save(translator_data, "translator")

    # Distribution stats
    print("\n═══ Narrator scaffold distribution ═══")
    narr_types = defaultdict(int)
    for d in narrator_data:
        narr_types[d["scaffold_type"]] += 1
    for t, c in sorted(narr_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/len(narrator_data):.1f}%)")

    print("\n═══ Translator scaffold distribution ═══")
    trans_types = defaultdict(int)
    for d in translator_data:
        trans_types[d["scaffold_type"]] += 1
    for t, c in sorted(trans_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/len(translator_data):.1f}%)")

    # Show examples
    print("\n═══ Narrator examples ═══")
    for ex in narrator_data[:3]:
        print(f"  PROMPT: {ex['prompt'][:200]}")
        print(f"  TARGET: {ex['target'].strip()}")
        print()

    print("═══ Translator examples ═══")
    for ex in translator_data[:3]:
        print(f"  PROMPT: {ex['prompt'][:200]}")
        print(f"  TARGET: {ex['target'].strip()}")
        print()

    # Summary
    print("\n═══ Summary ═══")
    print(f"  Narrator (LoRA C):   {len(narrator_data)} examples")
    print(f"  Translator (LoRA D): {len(translator_data)} examples")
    print(f"  Output: {args.output_dir}")
    print(f"\n  Train with:")
    print(f"    LoRA r=16, alpha=16 (scale=1.0)")
    print(f"    LR=1e-4, 3 epochs")
    print(f"    stop=['\\n'] at inference")
    print(f"    float32 on A10G")


if __name__ == "__main__":
    main()
