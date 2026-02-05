#!/usr/bin/env python3
"""Generate sub-graph DSL expressions for deduplicated templates using frontier LLM.

Sends template patterns + examples to Groq (Llama 3.1 70B) to classify
what arithmetic operation each pattern represents.

DSL vocabulary:
  "value"           — assignment (entity = extracted number)
  "entity + value"  — addition (entity = entity + number)
  "entity - value"  — subtraction (entity = entity - number)
  "entity * value"  — multiplication (entity = entity * number)
  "entity / value"  — division (entity = entity / number)
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import List, Dict


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# Load from .env if not in environment
if not GROQ_API_KEY:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                GROQ_API_KEY = line.split("=", 1)[1].strip()


SYSTEM_PROMPT = """You are a math operation classifier. Given a pattern template from math word problems, determine what arithmetic operation it represents.

IMPORTANT: You must classify each pattern into EXACTLY ONE of these DSL expressions:
- "value" — The pattern introduces/assigns a quantity. Entity gets this number. Examples: "has 5 apples", "costs $10", "weighs 3 pounds"
- "entity + value" — The pattern adds to an existing quantity. Examples: "gets 3 more", "earns an additional 5", "increases by 2"
- "entity - value" — The pattern subtracts from an existing quantity. Examples: "gives away 3", "loses 2", "spent 5 dollars"
- "entity * value" — The pattern multiplies an existing quantity. Examples: "triples the amount", "3 times as many", "doubled"
- "entity / value" — The pattern divides an existing quantity. Examples: "splits evenly among 3", "divides into 4 groups", "one third of"

Think about what COMPUTATION this pattern performs, not what words it uses. A pattern that says "each costs $5" is a "value" (introducing a rate), not multiplication.

Respond with ONLY a JSON array of objects, each with "id" and "dsl_expr" fields. No explanation."""


def classify_batch(templates: List[Dict], batch_size: int = 50) -> Dict[str, str]:
    """Classify a batch of templates into DSL expressions."""
    results = {}

    for i in range(0, len(templates), batch_size):
        batch = templates[i:i + batch_size]

        # Build the user prompt
        items = []
        for t in batch:
            examples = t.get("pattern_examples", [])[:3]
            examples_str = "; ".join(examples) if examples else "no examples"
            items.append(f'  {{"id": "{t["template_id"]}", "pattern": "{t["pattern"]}", "examples": "{examples_str}"}}')

        user_msg = "Classify these patterns:\n[\n" + ",\n".join(items) + "\n]"

        # Call Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }

        retries = 3
        for attempt in range(retries):
            try:
                resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 30)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

                content = resp.json()["choices"][0]["message"]["content"].strip()

                # Parse JSON response
                # Handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]

                classifications = json.loads(content)

                valid_dsls = {"value", "entity + value", "entity - value", "entity * value", "entity / value"}
                for item in classifications:
                    dsl = item.get("dsl_expr", "value")
                    if dsl not in valid_dsls:
                        dsl = "value"  # Default fallback
                    results[item["id"]] = dsl

                print(f"  Batch {i//batch_size + 1}/{(len(templates) + batch_size - 1)//batch_size}: "
                      f"classified {len(classifications)} templates")
                break

            except json.JSONDecodeError as e:
                print(f"  JSON parse error (attempt {attempt+1}): {e}")
                print(f"  Raw response: {content[:200]}")
                if attempt == retries - 1:
                    # Fallback: assign "value" to all in batch
                    for t in batch:
                        results[t["template_id"]] = "value"
            except Exception as e:
                print(f"  Error (attempt {attempt+1}): {e}")
                if attempt == retries - 1:
                    for t in batch:
                        results[t["template_id"]] = "value"

        # Rate limit respect
        time.sleep(1)

    return results


def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "deduplicated_1k_templates.json"
    output_path = project_root / "deduplicated_1k_templates.json"  # Overwrite with DSLs

    print(f"Loading templates from {input_path}...")
    with open(input_path) as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} templates")

    # Show distribution of existing dsl_expr
    existing = {}
    for t in templates:
        dsl = t.get("dsl_expr", "NONE")
        existing[dsl] = existing.get(dsl, 0) + 1
    print(f"Existing DSL distribution: {existing}")

    # Classify all templates
    print(f"\nClassifying {len(templates)} templates via {MODEL}...")
    t0 = time.time()
    classifications = classify_batch(templates, batch_size=50)
    elapsed = time.time() - t0
    print(f"Classification complete in {elapsed:.1f}s")

    # Apply classifications
    updated = 0
    for t in templates:
        tid = t["template_id"]
        if tid in classifications:
            t["dsl_expr"] = classifications[tid]
            updated += 1

    # Stats
    dsl_dist = {}
    for t in templates:
        dsl = t.get("dsl_expr", "value")
        dsl_dist[dsl] = dsl_dist.get(dsl, 0) + 1

    print(f"\n=== DSL Distribution ===")
    for dsl, count in sorted(dsl_dist.items(), key=lambda x: -x[1]):
        pct = count / len(templates) * 100
        print(f"  {dsl:<20} {count:>5} ({pct:.1f}%)")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"Updated {updated}/{len(templates)} templates")


if __name__ == "__main__":
    main()
