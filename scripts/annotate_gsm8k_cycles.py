"""
Annotate GSM8K problems with per-step intermediate numeric results via Claude API.

For each GSM8K problem, sends it to Claude to decompose into numbered
computation steps, then parses the step results into cycle_targets.

Output format (JSONL):
  {"problem": "...", "cycle_targets": [120, 60, 180], "final_answer": 180,
   "num_steps": 3, "level": "GSM8K", "gold_answer": 180, "validated": true}

Usage:
  python scripts/annotate_gsm8k_cycles.py --max_problems 20   # test run
  python scripts/annotate_gsm8k_cycles.py                      # full run
  python scripts/annotate_gsm8k_cycles.py --resume             # continue
"""
import argparse
import asyncio
import json
import os
import re
import time

import anthropic
from datasets import load_dataset


# ---------------------------------------------------------------------------
# GSM8K helpers (same as datasets_L49_gsm8k.py)
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


def normalize_number(val):
    """Convert float to int if it's a whole number."""
    if val == int(val):
        return int(val)
    return val


# ---------------------------------------------------------------------------
# Claude API decomposition
# ---------------------------------------------------------------------------

DECOMPOSITION_PROMPT = """\
Break this math problem into numbered computation steps.
For each step, give ONLY the numeric result (integer or simple decimal).
The last step's result must equal the final answer.

Problem: {problem}
Answer: {answer}

Respond in EXACTLY this format (one step per line, no extra text):
Step 1 result: <number>
Step 2 result: <number>
...
"""


def parse_step_results(response_text):
    """Parse Claude's response to extract step results as a list of numbers."""
    results = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        m = re.match(r'Step\s+\d+\s+result:\s*(.+)', line, re.IGNORECASE)
        if m:
            val_str = m.group(1).strip().rstrip('.')
            # Remove dollar signs, commas, percent signs
            val_str = val_str.replace('$', '').replace(',', '').replace('%', '')
            try:
                val = float(val_str)
                results.append(normalize_number(val))
            except ValueError:
                continue
    return results


async def decompose_batch(client, problems, batch_size=10, delay=0.5):
    """Decompose a batch of problems using Claude API with rate limiting.

    Args:
        client: Anthropic client
        problems: list of dicts with 'problem', 'answer_text', 'gold_answer'
        batch_size: concurrent requests per batch
        delay: seconds between batches

    Returns:
        list of result dicts (or None for failures)
    """
    results = [None] * len(problems)
    sem = asyncio.Semaphore(batch_size)

    async def process_one(idx, prob):
        async with sem:
            prompt = DECOMPOSITION_PROMPT.format(
                problem=prob['problem'],
                answer=prob['answer_text'],
            )
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                steps = parse_step_results(text)

                if not steps:
                    results[idx] = {'error': 'no_steps_parsed', 'raw': text}
                    return

                validated = False
                gold = prob['gold_answer']
                last_step = steps[-1]
                if isinstance(gold, float) and isinstance(last_step, (int, float)):
                    validated = abs(last_step - gold) < 0.01
                elif isinstance(gold, int) and isinstance(last_step, int):
                    validated = last_step == gold
                else:
                    validated = abs(float(last_step) - float(gold)) < 0.01

                results[idx] = {
                    'problem': prob['problem'],
                    'cycle_targets': steps,
                    'final_answer': steps[-1],
                    'num_steps': len(steps),
                    'level': 'GSM8K',
                    'gold_answer': gold,
                    'validated': validated,
                }
            except anthropic.APIError as e:
                results[idx] = {'error': f'api_error: {e}'}
            except Exception as e:
                results[idx] = {'error': f'exception: {e}'}

    # Process in batches
    for batch_start in range(0, len(problems), batch_size):
        batch_end = min(batch_start + batch_size, len(problems))
        tasks = []
        for i in range(batch_start, batch_end):
            tasks.append(process_one(i, problems[i]))
        await asyncio.gather(*tasks)
        if batch_end < len(problems):
            await asyncio.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_gsm8k_problems(split='train', max_problems=None):
    """Load GSM8K problems from HuggingFace."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    skipped = 0
    for ex in ds:
        gold = parse_final(ex['answer'])
        if gold is None:
            skipped += 1
            continue
        problems.append({
            'problem': ex['question'],
            'answer_text': ex['answer'],
            'gold_answer': normalize_number(gold),
        })
        if max_problems and len(problems) >= max_problems:
            break
    if skipped:
        print(f"Skipped {skipped} problems with unparseable answers")
    print(f"Loaded {len(problems)} GSM8K problems ({split})")
    return problems


def load_existing_results(path):
    """Load already-processed problem texts for resume support."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if 'problem' in rec:
                        done.add(rec['problem'])
                except json.JSONDecodeError:
                    continue
    return done


def main():
    parser = argparse.ArgumentParser(
        description='Annotate GSM8K problems with per-step cycle targets via Claude API'
    )
    parser.add_argument('--max_problems', type=int, default=None,
                        help='Max problems to process (default: all)')
    parser.add_argument('--split', type=str, default='train',
                        help='GSM8K split (default: train)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Concurrent API requests per batch (default: 10)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Seconds between batches (default: 0.5)')
    parser.add_argument('--output', type=str,
                        default='data/per_cycle/gsm8k_decomposed.jsonl',
                        help='Output path')
    args = parser.parse_args()

    # Load problems
    problems = load_gsm8k_problems(args.split, args.max_problems)

    # Resume support
    done_problems = set()
    if args.resume:
        done_problems = load_existing_results(args.output)
        if done_problems:
            print(f"Resuming: {len(done_problems)} already processed")
            problems = [p for p in problems if p['problem'] not in done_problems]
            print(f"Remaining: {len(problems)} problems")

    if not problems:
        print("No problems to process.")
        return

    # Set up client
    client = anthropic.Anthropic()

    # Process
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mode = 'a' if args.resume else 'w'

    total = len(problems)
    validated_count = 0
    failed_count = 0
    mismatch_count = 0
    t0 = time.time()

    # Process in chunks and write incrementally
    chunk_size = args.batch_size * 5  # process 5 batches worth at a time

    with open(args.output, mode) as f:
        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = problems[chunk_start:chunk_end]

            results = asyncio.run(
                decompose_batch(client, chunk, args.batch_size, args.delay)
            )

            for r in results:
                if r is None or 'error' in r:
                    failed_count += 1
                    continue
                if r.get('validated'):
                    validated_count += 1
                else:
                    mismatch_count += 1
                f.write(json.dumps(r) + '\n')

            f.flush()

            processed = min(chunk_end, total)
            if processed % 100 <= chunk_size or processed == total:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(
                    f"Progress: {processed}/{total} "
                    f"({validated_count} validated, {mismatch_count} mismatch, "
                    f"{failed_count} failed) "
                    f"[{rate:.1f} prob/s, {elapsed:.0f}s elapsed]"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Validated: {validated_count}")
    print(f"  Mismatch (last step != gold): {mismatch_count}")
    print(f"  Failed (API/parse errors): {failed_count}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
