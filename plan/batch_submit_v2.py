"""Submit 7-dimensional extraction batch from local problem file."""

import json
import os
from pathlib import Path
from datetime import datetime

import boto3
import anthropic

# Load API key
API_KEY_PATH = Path(__file__).parent.parent / "secrets" / "anthropic_key.txt"
ANTHROPIC_API_KEY = API_KEY_PATH.read_text().strip() if API_KEY_PATH.exists() else os.environ.get("ANTHROPIC_API_KEY", "")

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

SYSTEM_PROMPT = """You are a mathematical reasoning analyzer. You will receive a math problem and its chain-of-thought solution. Parse each reasoning step and extract structural properties.

For each step, extract ALL of the following:

1. step_type (exactly one):
   - evaluate: compute a numerical result from an expression
   - simplify: reduce an expression to simpler form
   - substitute: replace a variable or expression with another
   - solve_equation: isolate a variable or find a solution
   - factor: decompose into factors
   - expand: distribute or multiply out expressions
   - apply_theorem: invoke a named theorem, formula, or identity
   - count: enumerate or compute combinations/permutations
   - compare: compare values, test conditions, check divisibility
   - convert: change representation (decimal↔fraction, degrees↔radians, etc.)
   - setup: define variables, establish equations from problem text
   - other: none of the above

2. complexity_change (exactly one):
   - reduces: output is simpler than input (fewer terms, resolved variable)
   - neutral: roughly same complexity (substitution, conversion)
   - increases: output is more complex (expanding, introducing cases)

3. n_operands: integer, how many distinct mathematical entities are inputs to this step (numbers, variables, expressions). Count each unique entity once.

4. has_dependency: boolean, does this step use a result computed in a previous step? true if it references an intermediate value, false if it only uses values from the original problem.

5. output_type (exactly one):
   - number: produces a concrete numerical value
   - expression: produces a symbolic expression
   - equation: produces an equation or inequality
   - boolean: produces true/false or yes/no
   - set: produces a set or list of values

6. step_position (exactly one, based on where this step falls in the solution):
   - early: first 25% of steps (setup, initial reading)
   - middle: middle 50% of steps (core computation)
   - late: last 25% of steps (final evaluation, answer extraction)

7. reference_distance (exactly one, how far back does this step reference?):
   - none: does not reference any previous step (uses only original problem values)
   - local: references the immediately previous step (step N-1)
   - medium: references 2-3 steps back
   - distant: references 4+ steps back or combines results from multiple earlier steps

Also extract:
- operands: list of the mathematical entities used as input (as strings)
- result: what this step produces (as string)
- text_reference: quote the specific phrase from the original problem text that this step addresses. Use EXACT words. Set to null if purely computational.

Return ONLY valid JSON, no markdown, no explanation:
{
    "steps": [
        {
            "step_idx": 0,
            "raw_cot_text": "the exact text of this CoT step",
            "step_type": "...",
            "complexity_change": "...",
            "n_operands": int,
            "has_dependency": bool,
            "output_type": "...",
            "step_position": "...",
            "reference_distance": "...",
            "operands": ["...", "..."],
            "result": "...",
            "text_reference": "..." or null
        }
    ]
}"""


def submit_batch(problems_file: str):
    """Submit batch from local problems file."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load problems
    with open(problems_file) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems")

    # Create batch requests
    requests = []
    for p in problems:
        if not p.get("problem_text") or not p.get("cot_text"):
            continue

        user_content = f"Problem: {p['problem_text'][:3000]}\n\nSolution (Chain of Thought):\n{p['cot_text'][:5000]}"

        requests.append({
            "custom_id": p["problem_id"],
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}]
            }
        })

    print(f"Created {len(requests)} batch requests")

    # Submit batch
    print("Submitting to Anthropic Batch API...")
    batch = client.beta.messages.batches.create(requests=requests)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save batch info
    batch_info = {
        "batch_id": batch.id,
        "created_at": datetime.now().isoformat(),
        "n_requests": len(requests),
        "status": batch.processing_status,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_training_data_v2/batch_info.json",
        Body=json.dumps(batch_info, indent=2).encode("utf-8"),
    )

    print(f"\nBatch info saved to s3://{BUCKET}/c2c3_training_data_v2/batch_info.json")
    return batch


if __name__ == "__main__":
    import sys
    problems_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/problems_sample.json"
    submit_batch(problems_file)
