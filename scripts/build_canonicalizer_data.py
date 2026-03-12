"""
Build canonicalizer training data: parsed CoT steps → rough telegrams.

Level 1: Sonnet batch API converts individual steps into telegrams.
Level 2: Group by problem_id, order, wire _prev, attach scaffold.

Usage:
    # Level 1: Generate telegrams via Sonnet batch
    python build_canonicalizer_data.py level1 \
        --input s3://mycelium-data/c2c3_training_data_v2/parsed_steps.jsonl \
        --output telegrams_raw.jsonl \
        --sample 200  # start small (Rule 14)

    # Level 2: Assemble into full training examples
    python build_canonicalizer_data.py level2 \
        --telegrams telegrams_raw.jsonl \
        --output canonicalizer_training.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Deterministic VERB mapping — Sonnet doesn't choose the verb
# ─────────────────────────────────────────────────────────────

VERB_MAP = {
    # From parsed_steps step_type → telegram VERB
    "setup":            "GIVEN",
    "define":           "GIVEN",
    "assign":           "GIVEN",
    "given":            "GIVEN",
    "state":            "GIVEN",
    "evaluate":         "EVAL",
    "compute":          "EVAL",
    "calculate":        "EVAL",
    "arithmetic":       "EVAL",
    "solve_equation":   "SOLVE",
    "solve":            "SOLVE",
    "find":             "SOLVE",
    "expand":           "EXPAND",
    "factor":           "EXPAND",   # factor is structural expansion
    "simplify":         "SIMPLIFY",
    "reduce":           "SIMPLIFY",
    "combine":          "SIMPLIFY",
    "substitute":       "SUBS",
    "replace":          "SUBS",
    "apply":            "APPLY",
    "theorem":          "APPLY",
    "formula":          "APPLY",
    "answer":           "ANSWER",
    "final":            "ANSWER",
    "conclude":         "ANSWER",
    "compare":          "EVAL",     # comparisons are evaluations
    "convert":          "EVAL",     # unit conversions are evaluations
    "other":            "EVAL",     # default fallback
}

VALID_VERBS = {"GIVEN", "EVAL", "SOLVE", "EXPAND", "SIMPLIFY", "SUBS", "APPLY", "ANSWER"}

# English words allowed in telegrams (only the verbs themselves)
ALLOWED_ENGLISH = VALID_VERBS | {"_prev"}

# ─────────────────────────────────────────────────────────────
# Sonnet prompt — verb is determined, Sonnet fills arguments
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You convert math reasoning steps into minimal telegraphic instructions.

Rules:
- Output EXACTLY one line: {VERB} [arguments]
- The VERB is already determined. You fill in the arguments only.
- 3-6 tokens maximum. Fewer is better.
- Use rough math notation: ^ not **, fractions as a/b, no Eq() wrappers
- NO English words except the VERB itself and _prev
- _prev references the previous step's result
- No Symbol() declarations, no import statements
- No parentheses around the entire expression (only within math as needed)
- If the equation is long, keep the VERB and core structure. Omit intermediate expansion.
- Theorem names allowed as arguments: difference_of_squares, pythagorean, quadratic_formula, vietas, binomial (use underscores, no spaces)

Examples:
  GIVEN x^2+y^2=90
  GIVEN xy=27
  EXPAND (x+y)^2
  SUBS _prev x^2+y^2 90
  EVAL _prev
  SOLVE x^2-5x+6 x
  SIMPLIFY _prev
  APPLY difference_of_squares a^2-b^2
  APPLY pythagorean a b c
  ANSWER _prev
"""


def build_user_prompt(step: dict, verb: str) -> str:
    """Build the user prompt for a single step conversion."""
    parts = [f"Convert this math step into a telegraphic instruction."]
    parts.append(f"The verb is: {verb}")
    parts.append(f"")

    if step.get("raw_cot_text"):
        parts.append(f"Raw step text: {step['raw_cot_text']}")

    if step.get("operands"):
        parts.append(f"Operands: {json.dumps(step['operands'])}")

    if step.get("result"):
        # Show result context but instruct not to include it
        parts.append(f"(Context — this step produces: {step['result']})")

    parts.append(f"")
    parts.append(f"Output exactly one line: {verb} [arguments]")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Validation gates
# ─────────────────────────────────────────────────────────────

def validate_telegram(telegram: str) -> dict:
    """Validate a generated telegram. Returns {valid, telegram, errors}."""
    errors = []
    telegram = telegram.strip()

    # Remove markdown code fences if Sonnet wrapped it
    if telegram.startswith("```"):
        telegram = telegram.strip("`").strip()

    # Must start with a valid verb
    parts = telegram.split(None, 1)
    if not parts:
        return {"valid": False, "telegram": telegram, "errors": ["empty"]}

    verb = parts[0].upper()
    if verb not in VALID_VERBS:
        errors.append(f"invalid_verb:{parts[0]}")

    # Normalize verb to uppercase
    telegram = verb + (" " + parts[1] if len(parts) > 1 else "")

    # Token count (rough: split on whitespace)
    tokens = telegram.split()
    if len(tokens) > 10:
        errors.append(f"too_long:{len(tokens)}_tokens")

    # No English words except verb and _prev
    # Allowed: single letters (variables), math functions, Greek letters, theorem names
    ALLOWED_WORDS = {'prev', 'pi', 'e', 'sin', 'cos', 'tan', 'log', 'ln', 'sqrt',
                     'inf', 'oo', 'alpha', 'beta', 'theta', 'phi', 'lambda',
                     'delta', 'sigma', 'omega', 'frac', 'cdot',
                     # Theorem names (underscored in telegrams but validation sees parts)
                     'difference', 'squares', 'pythagorean', 'quadratic', 'formula',
                     'vietas', 'binomial', 'sum', 'product', 'roots'}

    for token in tokens[1:]:  # skip verb
        # Extract alphabetic runs from the token
        alpha_runs = re.findall(r'[a-zA-Z]+', token)
        for word in alpha_runs:
            word_lower = word.lower()
            # Allow single letters (variables) and known math words
            if len(word) == 1:
                continue  # single letter variable - OK
            if len(word) == 2:
                # Allow coefficient-variable combos like Bx, xy, an
                continue  # 2-letter math notation - OK
            if word_lower in ALLOWED_WORDS:
                continue  # known math word - OK
            # Multi-letter word (3+) that's not in allowed list - flag it
            errors.append(f"english_word:{word}")

    return {
        "valid": len(errors) == 0,
        "telegram": telegram,
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────
# Level 1: Build Sonnet batch requests
# ─────────────────────────────────────────────────────────────

def build_batch_requests(input_path: str, output_path: str, sample: int = None):
    """Read parsed steps, build Anthropic batch API request file."""

    steps = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    print(f"Loaded {len(steps)} steps")

    if sample:
        steps = steps[:sample]
        print(f"Sampling first {sample} steps (Rule 14: validate small first)")

    # Build batch requests
    requests = []
    skipped = 0

    for i, step in enumerate(steps):
        step_type = step.get("step_type", "").lower().strip()

        # Map to verb
        verb = VERB_MAP.get(step_type)
        if verb is None:
            # Try partial match
            for key, v in VERB_MAP.items():
                if key in step_type:
                    verb = v
                    break

        if verb is None:
            skipped += 1
            continue

        user_prompt = build_user_prompt(step, verb)

        request = {
            "custom_id": f"step_{step.get('problem_id', 'unk')}_{step.get('step_idx', i)}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            }
        }
        requests.append(request)

        # Carry forward metadata for Level 2 assembly
        request["_meta"] = {
            "problem_id": step.get("problem_id"),
            "step_idx": step.get("step_idx", i),
            "verb": verb,
            "step_type": step_type,
            "result": step.get("result"),
        }

    print(f"Built {len(requests)} requests, skipped {skipped} (unmapped step_type)")

    # Write batch request file (JSONL for Anthropic batch API)
    # Separate the API requests from metadata
    batch_file = output_path.replace(".jsonl", "_batch.jsonl")
    meta_file = output_path.replace(".jsonl", "_meta.jsonl")

    with open(batch_file, "w") as fb, open(meta_file, "w") as fm:
        for req in requests:
            meta = req.pop("_meta")
            meta["custom_id"] = req["custom_id"]
            fb.write(json.dumps(req) + "\n")
            fm.write(json.dumps(meta) + "\n")

    print(f"Batch requests: {batch_file}")
    print(f"Metadata: {meta_file}")
    print(f"\nNext: submit batch via Anthropic API")
    print(f"  python -c \"import anthropic; c = anthropic.Anthropic(); ...")
    print(f"\nOr for quick testing, use level1_sync mode for first 20 steps")

    return batch_file, meta_file


# ─────────────────────────────────────────────────────────────
# Level 1 (sync mode): Direct API calls for small test runs
# ─────────────────────────────────────────────────────────────

def run_sync(input_path: str, output_path: str, sample: int = 20):
    """Direct API calls for testing. Use batch for production."""
    import anthropic
    import os

    # Load API key from secrets folder
    secrets_dir = Path(__file__).parent.parent / "secrets"
    api_key_file = secrets_dir / "anthropic_key.txt"
    if api_key_file.exists():
        api_key = api_key_file.read_text().strip()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    steps = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    steps = steps[:sample]
    print(f"Running sync on {len(steps)} steps")

    results = []
    for i, step in enumerate(steps):
        step_type = step.get("step_type", "").lower().strip()
        verb = VERB_MAP.get(step_type)
        if verb is None:
            for key, v in VERB_MAP.items():
                if key in step_type:
                    verb = v
                    break
        if verb is None:
            print(f"  [{i}] SKIP — unmapped step_type: {step_type}")
            continue

        user_prompt = build_user_prompt(step, verb)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            telegram = response.content[0].text.strip()
        except Exception as e:
            print(f"  [{i}] ERROR — {e}")
            continue

        validation = validate_telegram(telegram)

        result = {
            "problem_id": step.get("problem_id"),
            "step_idx": step.get("step_idx", i),
            "verb": verb,
            "step_type": step_type,
            "raw_cot_text": step.get("raw_cot_text", ""),
            "telegram": validation["telegram"],
            "valid": validation["valid"],
            "errors": validation["errors"],
            "result": step.get("result"),
        }
        results.append(result)

        status = "✓" if validation["valid"] else f"✗ {validation['errors']}"
        print(f"  [{i}] {validation['telegram']:40s} {status}")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid_count = sum(1 for r in results if r["valid"])
    print(f"\nResults: {valid_count}/{len(results)} valid ({100*valid_count/max(len(results),1):.1f}%)")
    print(f"Saved to {output_path}")

    return results


# ─────────────────────────────────────────────────────────────
# Level 1: Process batch results
# ─────────────────────────────────────────────────────────────

def process_batch_results(results_path: str, meta_path: str, output_path: str):
    """Merge batch API results with metadata, validate telegrams."""

    # Load metadata
    meta_by_id = {}
    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            meta_by_id[m["custom_id"]] = m

    # Load batch results
    results = []
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            custom_id = r["custom_id"]
            meta = meta_by_id.get(custom_id, {})

            # Extract telegram from response
            telegram_raw = ""
            if r.get("result", {}).get("message", {}).get("content"):
                telegram_raw = r["result"]["message"]["content"][0].get("text", "")

            validation = validate_telegram(telegram_raw)

            results.append({
                "problem_id": meta.get("problem_id"),
                "step_idx": meta.get("step_idx"),
                "verb": meta.get("verb"),
                "step_type": meta.get("step_type"),
                "telegram": validation["telegram"],
                "valid": validation["valid"],
                "errors": validation["errors"],
                "result": meta.get("result"),
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid_count = sum(1 for r in results if r["valid"])
    print(f"Processed {len(results)} results: {valid_count} valid ({100*valid_count/len(results):.1f}%)")

    # Error breakdown
    error_counts = {}
    for r in results:
        for e in r["errors"]:
            key = e.split(":")[0]
            error_counts[key] = error_counts.get(key, 0) + 1

    if error_counts:
        print("\nError breakdown:")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")

    return results


# ─────────────────────────────────────────────────────────────
# Level 2: Assemble into full training examples
# ─────────────────────────────────────────────────────────────

def assemble_training_data(
    telegrams_path: str,
    output_path: str,
    problem_texts_path: str = None,
    c1a_scaffolds_path: str = None,
    val_ratio: float = 0.1
):
    """Group telegrams by problem_id, join with problem texts and C1-A scaffolds."""

    # Load validated telegrams
    telegrams = []
    with open(telegrams_path) as f:
        for line in f:
            t = json.loads(line)
            if t.get("valid"):
                telegrams.append(t)

    print(f"Loaded {len(telegrams)} valid telegrams")

    # Load problem texts if provided
    problem_texts = {}
    if problem_texts_path:
        with open(problem_texts_path) as f:
            for line in f:
                p = json.loads(line)
                problem_texts[p["problem_id"]] = p.get("problem_text", "")
        print(f"Loaded {len(problem_texts)} problem texts")

    # Load C1-A scaffolds if provided (optional override)
    c1a_scaffolds = {}
    if c1a_scaffolds_path:
        with open(c1a_scaffolds_path) as f:
            for line in f:
                s = json.loads(line)
                c1a_scaffolds[s["problem_id"]] = s.get("scaffold_types", [])
        print(f"Loaded {len(c1a_scaffolds)} C1-A scaffolds")

    # Group by problem_id
    by_problem = {}
    for t in telegrams:
        pid = t.get("problem_id")
        if pid is None:
            continue
        if pid not in by_problem:
            by_problem[pid] = []
        by_problem[pid].append(t)

    print(f"Found {len(by_problem)} unique problems")

    # Assemble each problem
    training_examples = []
    skipped_no_text = 0

    for pid, steps in by_problem.items():
        # Must have problem text
        problem_text = problem_texts.get(pid, "")
        if not problem_text or len(problem_text) < 10:
            skipped_no_text += 1
            continue

        # Sort by step_idx
        steps.sort(key=lambda s: s.get("step_idx", 0))

        # Build scaffold (list of verbs from telegrams)
        # Use C1-A scaffold if available, otherwise derive from telegrams
        if pid in c1a_scaffolds:
            scaffold = c1a_scaffolds[pid]
        else:
            scaffold = [s["verb"] for s in steps]

        # Wire _prev references
        wired_telegrams = []
        for i, s in enumerate(steps):
            telegram = s["telegram"]
            # Keep telegram as-is; _prev insertion handled by Sonnet
            wired_telegrams.append(telegram)

        # Build target string
        target = "\n".join(wired_telegrams)

        training_examples.append({
            "problem_id": pid,
            "problem_text": problem_text,
            "scaffold": scaffold,
            "target": target,
            "n_steps": len(steps),
        })

    print(f"Skipped {skipped_no_text} problems without text")
    print(f"Assembled {len(training_examples)} training examples")

    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * (1 - val_ratio))
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Save train and val
    train_path = output_path.replace(".jsonl", "_train.jsonl")
    val_path = output_path.replace(".jsonl", "_val.jsonl")

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved: {train_path}, {val_path}")

    # Stats
    all_examples = train_examples + val_examples
    step_counts = [ex["n_steps"] for ex in all_examples]
    print(f"\nSteps per problem: min={min(step_counts)}, "
          f"max={max(step_counts)}, "
          f"mean={sum(step_counts)/len(step_counts):.1f}")

    # Verb distribution
    verb_counts = {}
    for ex in all_examples:
        for v in ex["scaffold"]:
            verb_counts[v] = verb_counts.get(v, 0) + 1

    print(f"\nVerb distribution:")
    for verb, count in sorted(verb_counts.items(), key=lambda x: -x[1]):
        print(f"  {verb}: {count} ({100*count/sum(verb_counts.values()):.1f}%)")

    return train_examples, val_examples


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build canonicalizer training data")
    sub = parser.add_subparsers(dest="command")

    # Level 1: build batch requests
    p1 = sub.add_parser("level1", help="Build Sonnet batch request file")
    p1.add_argument("--input", required=True, help="parsed_steps.jsonl path")
    p1.add_argument("--output", default="telegrams_raw.jsonl")
    p1.add_argument("--sample", type=int, default=None, help="Sample N steps (Rule 14)")

    # Level 1 sync: direct API calls for testing
    p1s = sub.add_parser("level1_sync", help="Direct API calls (testing only)")
    p1s.add_argument("--input", required=True, help="parsed_steps.jsonl path")
    p1s.add_argument("--output", default="telegrams_sync.jsonl")
    p1s.add_argument("--sample", type=int, default=20)

    # Level 1: process batch results
    p1r = sub.add_parser("level1_results", help="Process batch API results")
    p1r.add_argument("--results", required=True, help="Batch results JSONL")
    p1r.add_argument("--meta", required=True, help="Metadata JSONL from level1")
    p1r.add_argument("--output", default="telegrams_validated.jsonl")

    # Level 2: assemble full training examples
    p2 = sub.add_parser("level2", help="Assemble training examples")
    p2.add_argument("--telegrams", required=True, help="Validated telegrams JSONL")
    p2.add_argument("--problem-texts", required=True, help="Problem texts JSONL")
    p2.add_argument("--c1a-scaffolds", default=None, help="C1-A scaffold predictions (optional)")
    p2.add_argument("--output", default="canonicalizer_training.jsonl")
    p2.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")

    args = parser.parse_args()

    if args.command == "level1":
        build_batch_requests(args.input, args.output, args.sample)
    elif args.command == "level1_sync":
        run_sync(args.input, args.output, args.sample)
    elif args.command == "level1_results":
        process_batch_results(args.results, args.meta, args.output)
    elif args.command == "level2":
        assemble_training_data(
            args.telegrams,
            args.output,
            problem_texts_path=args.problem_texts,
            c1a_scaffolds_path=args.c1a_scaffolds,
            val_ratio=args.val_ratio
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
