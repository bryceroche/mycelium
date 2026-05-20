"""Generate per-step training targets for GSM8K via Haiku.

Bridges the gap between v59's working K-breath paradigm (which needs per-step
gen_targets) and GSM8K (which only ships final-answer solutions). For each
GSM8K problem we ask Haiku to decompose the solution into N atomic steps
matching the L4.7 format: digit-spaced numbers, one arithmetic op per step,
each step references the previous step's result, last step's number is the
answer.

We then VALIDATE that Haiku's final answer matches GSM8K's gold. Only kept if
match. Expected acceptance rate: ~80-90% (Haiku is reliable on small-integer
arithmetic).

Output: JSONL file, one example per line:
  {"problem": <digit-spaced>, "gen_targets": [<step1>, <step2>, ...], "answer": <int>, "n_steps": <int>}

Usage:
    ANTHROPIC_API_KEY=sk-... \\
    python scripts/generate_gsm8k_step_targets.py \\
        --num 200 --split train --output .cache/gsm8k_steps_v1_train.jsonl

Then load via mycelium.l3_data.load_gsm8k_steps (next file to write).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy import — only need this if we actually run
try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

from mycelium.l3_data import load_gsm8k_spaced

# Haiku 4.5 is current best-in-class for math accuracy at low latency
MODEL = "claude-haiku-4-5-20251001"

PROMPT_TEMPLATE = """You are a math tutor. Solve the problem below step by step.

CRITICAL FORMAT REQUIREMENTS:
- Numbers must be written with single spaces between digits, matching the problem text (e.g., "1 7 0" means one hundred seventy).
- Each step does exactly ONE arithmetic operation.
- Each step's text MUST reference the previous step's numeric result.
- Each step ends with a period.
- Write the arithmetic as: "operand1 op operand2 = result unit"
- The LAST step's result is the final answer.

Output ONLY the steps, one per line. Do NOT number them. Do NOT add headers or commentary.

EXAMPLE
Problem: "Sam had 8 5 cookies. Sam doubled the collection, then gave 1 3 2 away. How many cookies does Sam have left?"
Output:
Sam had 8 5 cookies and doubled them. 8 5 * 2 = 1 7 0 cookies now.
Then Sam gave 1 3 2 away. 1 7 0 - 1 3 2 = 3 8 cookies remaining.

EXAMPLE
Problem: "Ryan reads 5 pages per hour. Ryan read for 6 hours in the morning and 3 hours in the afternoon. The book has 1 2 6 pages. How many pages are left to read?"
Output:
Ryan read 5 pages/hour for 6 hours. 5 * 6 = 3 0 pages in the morning.
Then Ryan read for 3 more hours. 5 * 3 = 1 5 pages in the afternoon.
Total pages read: 3 0 + 1 5 = 4 5 pages.
The book has 1 2 6 pages. 1 2 6 - 4 5 = 8 1 pages left.

NOW SOLVE THIS PROBLEM
Problem: {problem}
Output:"""


def parse_final_answer(steps: list[str]) -> int | None:
    """Pull the last numeric result from the final step.

    Steps are in digit-spaced format. The last "= N" pattern is the answer.
    """
    if not steps:
        return None
    last = steps[-1]
    # Find all "= number" patterns; take the last one
    matches = re.findall(r"=\s*(-?\d(?:\s\d)*)", last)
    if not matches:
        # Fallback: last number anywhere in the line
        matches = re.findall(r"(-?\d(?:\s\d)*)", last)
        if not matches:
            return None
    digit_spaced = matches[-1]
    # Remove spaces between digits → integer
    try:
        return int(digit_spaced.replace(" ", ""))
    except ValueError:
        return None


def parse_haiku_output(text: str) -> list[str]:
    """Parse Haiku's response into a list of step strings.

    Strips empty lines, removes any "Step N:" prefixes Haiku might add, ensures
    each step is a clean single line. Does NOT add trailing " ####" — that's
    appended by encode_cycles at training time.
    """
    raw = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []
    for line in raw:
        # Strip optional "Step N:" or "N." prefixes
        line = re.sub(r"^(?:Step\s+\d+[:.\s]*|^\d+[.)]\s*)", "", line)
        if line:
            cleaned.append(line)
    return cleaned


def generate_steps(client: Anthropic, problem: str, max_retries: int = 2) -> list[str] | None:
    """Call Haiku and parse step targets. Returns None on API failure."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}],
            )
            text = response.content[0].text
            return parse_haiku_output(text)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
    print(f"  API error after {max_retries + 1} attempts: {last_err}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100, help="Number of GSM8K problems to process")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--max-steps", type=int, default=10, help="Reject if Haiku produces more steps than this")
    parser.add_argument("--min-steps", type=int, default=2, help="Reject if fewer steps (GSM8K problems are multi-step by definition)")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N problems (for resuming/sharding)")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GSM8K {args.split} split...", flush=True)
    examples = load_gsm8k_spaced(args.split)
    print(f"  loaded {len(examples)} problems", flush=True)

    examples = examples[args.start:args.start + args.num]
    print(f"Processing {len(examples)} problems starting at index {args.start}", flush=True)

    client = Anthropic()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected_api = 0
    rejected_format = 0
    rejected_answer = 0
    rejected_step_count = 0
    step_count_hist: dict[int, int] = {}

    t0 = time.perf_counter()
    with args.output.open("w") as f:
        for i, ex in enumerate(examples):
            steps = generate_steps(client, ex.problem)
            if steps is None:
                rejected_api += 1
                continue
            if not (args.min_steps <= len(steps) <= args.max_steps):
                rejected_step_count += 1
                continue
            parsed = parse_final_answer(steps)
            if parsed is None:
                rejected_format += 1
                continue
            if parsed != ex.answer:
                rejected_answer += 1
                if i < 5:  # Show first few rejections for debugging
                    print(f"  REJECT[{i}] answer mismatch: parsed={parsed} gold={ex.answer}")
                    print(f"    problem: {ex.problem[:100]}")
                    print(f"    last step: {steps[-1]}")
                continue

            record = {
                "problem": ex.problem,
                "gen_targets": steps,
                "answer": ex.answer,
                "n_steps": len(steps),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            kept += 1
            step_count_hist[len(steps)] = step_count_hist.get(len(steps), 0) + 1

            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(examples) - (i + 1)) / rate
                print(f"  [{i+1}/{len(examples)}] kept={kept} "
                      f"reject(api={rejected_api}, fmt={rejected_format}, ans={rejected_answer}, steps={rejected_step_count}) "
                      f"rate={rate:.2f}/s eta={eta:.0f}s",
                      flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n=== done in {elapsed:.0f}s ===")
    print(f"Kept: {kept}/{len(examples)} ({kept/len(examples)*100:.1f}%)")
    print(f"Rejected: api={rejected_api}, format={rejected_format}, answer={rejected_answer}, step_count={rejected_step_count}")
    print(f"Step-count histogram of kept examples:")
    for k in sorted(step_count_hist):
        print(f"  K={k}: {step_count_hist[k]}")
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
