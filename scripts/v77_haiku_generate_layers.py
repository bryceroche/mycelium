"""v77/v78 Phase 2 — generate progressive-layer supervision targets via Haiku 4.5.

Reads N problems from a JSONL input (e.g. `.cache/gsm8k_steps_v1_train.jsonl`),
calls Haiku to produce N progressive layers per problem (4, 6, or 7 supported),
validates the final DAG via SymPy, and writes results to a JSONL output.

Seven-layer SMOOTH coarse->fine ramp (v78 default — each consecutive pair has
similar translation work, no large jumps or adjacent-layer duplicates):
    Layer 0: problem-text surface form (light paraphrase)
    Layer 1: drop narrative fluff, keep entities + numbers + math intent
    Layer 2: tighten to math-focused statement, retain entities + numbers
    Layer 3: equation form with verbal scaffolding, entities still named
    Layer 4: equation with placeholder variables, light entity reference
    Layer 5: equation with variables only, structural form
    Layer 6: pure SymPy-executable DAG (variable assignments + final answer)

The final DAG MUST be executable by `scripts/v77_sympy_eval.py:dag_to_answer`
and produce the gold answer. We validate this and report success rate.

Usage:
    # Default: 10 problems -> .cache/gsm8k_steps_v77_sample.jsonl
    .venv/bin/python scripts/v77_haiku_generate_layers.py

    # v78 (7 layers, full train+test):
    .venv/bin/python scripts/v77_haiku_generate_layers.py \\
        --src .cache/gsm8k_steps_v1_train.jsonl \\
        --dst .cache/gsm8k_steps_v78_train.jsonl \\
        --n_layers 7 \\
        --num_problems 7000 \\
        --concurrency 16

The API key is read from ANTHROPIC_API_KEY env var, falling back to
`/home/bryce/Desktop/keys/key1.txt`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: .venv/bin/pip install anthropic", file=sys.stderr)
    sys.exit(1)

from v77_sympy_eval import dag_to_answer


MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_KEY_FALLBACK = "/home/bryce/Desktop/keys/key1.txt"


# Haiku pricing (Haiku 4.5): $1/MTok input, $5/MTok output
_PRICE_IN_PER_MTOK = 1.0
_PRICE_OUT_PER_MTOK = 5.0


PROMPT_TEMPLATE_6 = """You are converting a math word problem and its gold step-by-step solution into SIX progressively more refined representations. Each representation is a SMOOTH REFINEMENT of the previous, like a diffusion model denoising from text toward executable code.

ALL SIX LEVELS must cover the FULL solution (all reasoning steps). Not just one step the entire chain.

LEVEL 0 -- paraphrase the situation in plain English. DESCRIBE WHO is doing WHAT and what is BEING ASKED. Do NOT identify the math operations. Do NOT use math-class names. Keep entity names. ~1-3 sentences. Stay at the story level.

LEVEL 1 -- same content as Level 0, but now NAME the kind(s) of math needed in plain language ("requires addition and division", "multi-step problem involving multiplication then subtraction"). Do NOT yet write any numbers or operation labels. Still narrative prose. ~2-3 sentences.

LEVEL 2 -- same as Level 1, plus pull in the CONCRETE NUMBERS tied to entities ("Janet has 50 eggs, gets 15 more"). Still no operation labels in caps and no equations. Just attach numbers to the entities from Level 1. ~2-4 sentences.

LEVEL 3 -- same as Level 2, plus EXPLICITLY NAME each operation in CAPS (ADD, SUBTRACT, MULTIPLY, DIVIDE). Still English sentences. Reads like a tutor explaining what operation to apply where. NO equations yet. ~2-4 sentences.

LEVEL 4 -- equation form with light verbal context. Each step becomes "named quantity = an equation = a value". Like "Janet's eggs after laying = 50 + 15 = 65 eggs." Connect equations with brief prose like "Then" or "So". The equations are arithmetic over plain numbers (no variable names). ~2-6 short lines.

LEVEL 5 -- pure DAG syntax, fully executable, no English. Use only ASCII variables x0, x1, x2, ... and the final variable `answer`. Each statement is `xN = EXPR`. Operators: + - * / ** and parentheses. Statements are separated by ` ; ` (space-semicolon-space). The FINAL statement MUST be `answer = ...`. Every variable referenced must be defined earlier. Numbers are plain decimal (no commas, no units). Example: `x0 = 50 + 15 ; x1 = x0 * 2 ; answer = x1`.

CRITICAL RULES for Level 5:
- ASCII only. Variables x0, x1, x2, ... in order. Last assignment is `answer = ...`.
- Statements separated by ` ; `.
- Operators: + - * / ** (power). NO functions, NO comparisons, NO if/else.
- Numbers must be plain (e.g. "50" not "50 dollars" and not "50,000").
- The DAG, when evaluated, MUST equal the gold answer exactly.
- The DAG should mirror the structure of the gold solution: one xN per reasoning step.

CRITICAL RULES for the LEVEL RAMP:
- L0 must NOT identify the math (no "addition", "subtraction", "multi-step", etc.).
- L1 introduces the math class name but uses NO numbers and NO operation tags.
- L2 introduces numbers but no operation tags in CAPS yet.
- L3 introduces operation tags in CAPS but still no equations.
- L4 has equations but with verbal scaffolding ("X's eggs = 50 + 15 = 65").
- L5 strips all verbal scaffolding into pure DAG.

Each level adds EXACTLY ONE refinement dimension over the previous. Do not skip ahead.

Output format -- EXACTLY six lines, each prefixed with the level marker. Do NOT add any other text, headers, blank lines, or commentary.

L0: <level 0 text>
L1: <level 1 text>
L2: <level 2 text>
L3: <level 3 text>
L4: <level 4 text>
L5: <level 5 DAG>

EXAMPLE INPUT
Problem: Sam had 8 5 cookies. Sam doubled the collection, then gave 1 3 2 away. How many cookies does Sam have left?
Gold steps:
  - Sam had 8 5 cookies and doubled them. 8 5 * 2 = 1 7 0 cookies now.
  - Then Sam gave 1 3 2 away. 1 7 0 - 1 3 2 = 3 8 cookies remaining.
Gold answer: 38

EXAMPLE OUTPUT
L0: Sam starts with a collection of cookies, doubles the collection, and then gives some away. The question asks how many cookies Sam has left at the end.
L1: Sam starts with some cookies, doubles them, and then gives some away. This is a two-step problem involving multiplication and subtraction.
L2: Sam starts with 85 cookies, doubles them, and then gives 132 away. This is a two-step problem involving multiplication and subtraction.
L3: Sam starts with 85 cookies and MULTIPLIES by 2 to double them. Then he SUBTRACTS 132 to find the cookies remaining.
L4: Sam's doubled cookies = 85 * 2 = 170 cookies. Then cookies remaining = 170 - 132 = 38 cookies.
L5: x0 = 85 * 2 ; x1 = x0 - 132 ; answer = x1

NOW DO THIS PROBLEM
Problem: {problem}
Gold steps:
{steps_block}
Gold answer: {answer}

OUTPUT (six lines, L0-L5 only):"""


# v78 7-layer prompt: SMOOTH ramp emphasizing equal translation work between
# consecutive pairs. No huge jumps; no adjacent-layer duplicates. The intermediate
# layers (3, 4, 5) gradually strip verbal scaffolding (entities -> variables ->
# pure structure) so the final L6 DAG isn't a giant leap from L5.
PROMPT_TEMPLATE_7 = """For this math problem and its gold solution, produce 7 representations forming a SMOOTH ramp from problem-surface form (L0) to pure DAG (L6).

CRITICAL: each consecutive pair (L_k -> L_k+1) should require similar translation work. Avoid huge jumps and adjacent-layer duplicates.

L0: ~problem-text surface form (light paraphrase, keep entity names and the question intact, do not yet name math operations)
L1: drop narrative fluff, keep entities + numbers + math intent (state what KIND of math the problem requires, in plain words; still no operation tags in CAPS, still no equations)
L2: tighten to math-focused statement, retain entities + numbers (concise math-class description with entity names and concrete numbers; still no equations)
L3: equation form with verbal scaffolding, entities still named (e.g., "Janet's eggs after laying = 50 + 15 = 65 eggs"; lines may use "Then"/"So" connectors)
L4: equation with placeholder variables, light entity reference (e.g., "eggs_total = 50 + 15 = 65 (Janet)"; entity names appear as comments or parentheticals, equation centers on the variable)
L5: equation with variables only, structural form (e.g., "x0 = 50 + 15"; no entity names at all, no comments, just variable-and-number equations connected by ; and culminating in answer = ...)
L6: pure SymPy-executable DAG (variable assignments + final answer, identical in form to L5 but stripped of any whitespace inconsistencies, using ASCII variables x0, x1, x2, ... and ending with `answer = ...`)

EVERY L_k must be SUFFICIENT to derive the gold answer (a downstream parser reading L_k should be able to produce the right number). L0..L2 are verbal but unambiguous; L3..L5 are equation-form; L6 is the executable DAG.

CRITICAL RULES for Level 6 (the DAG):
- ASCII only. Variables x0, x1, x2, ... in order. Last assignment is `answer = ...`.
- Statements separated by ` ; ` (space-semicolon-space).
- Operators: + - * / ** (power). NO functions, NO comparisons, NO if/else.
- Numbers must be plain (e.g. "50" not "50 dollars" and not "50,000").
- The DAG, when evaluated, MUST equal the gold answer exactly.
- The DAG should mirror the structure of the gold solution: one xN per reasoning step.

CRITICAL RULES for the LEVEL RAMP:
- L0 must NOT name math operations or write any equations.
- L1 introduces the math class name (addition / subtraction / etc.) but uses NO equations.
- L2 tightens L1 into a more math-focused statement, but still no equations.
- L3 introduces equations with verbal scaffolding (named quantities = arithmetic).
- L4 switches the equation to using variable-style names, but still references entities lightly.
- L5 is variable-only equations, structurally identical to L6 but possibly with extra whitespace or non-DAG flourishes.
- L6 strips any remaining verbal residue into pure DAG syntax.

Each level adds AT MOST ONE refinement dimension over the previous. Do not skip ahead. Adjacent layers should NOT be identical.

Output format -- EXACTLY seven lines, each prefixed with the level marker. Do NOT add any other text, headers, blank lines, or commentary.

L0: <level 0 text>
L1: <level 1 text>
L2: <level 2 text>
L3: <level 3 text>
L4: <level 4 text>
L5: <level 5 text>
L6: <level 6 DAG>

EXAMPLE INPUT
Problem: Sam had 8 5 cookies. Sam doubled the collection, then gave 1 3 2 away. How many cookies does Sam have left?
Gold steps:
  - Sam had 8 5 cookies and doubled them. 8 5 * 2 = 1 7 0 cookies now.
  - Then Sam gave 1 3 2 away. 1 7 0 - 1 3 2 = 3 8 cookies remaining.
Gold answer: 38

EXAMPLE OUTPUT
L0: Sam starts with a collection of cookies, doubles the collection, and then gives some away. The question is how many cookies Sam has left at the end.
L1: Sam starts with cookies, doubles them, and then gives some away. This is a two-step problem involving multiplication and subtraction.
L2: Sam has 85 cookies; he doubles them, then gives 132 away; the result is the remaining count. Two-step: multiplication then subtraction.
L3: Sam's doubled cookies = 85 * 2 = 170 cookies. Then cookies remaining = 170 - 132 = 38 cookies.
L4: doubled_count = 85 * 2 = 170 (Sam) ; remaining = 170 - 132 = 38 (Sam)
L5: x0 = 85 * 2 ; x1 = x0 - 132 ; answer = x1
L6: x0 = 85 * 2 ; x1 = x0 - 132 ; answer = x1

NOW DO THIS PROBLEM
Problem: {problem}
Gold steps:
{steps_block}
Gold answer: {answer}

OUTPUT (seven lines, L0-L6 only):"""


# Legacy 4-layer prompt (Phase 1) — kept around for reproducibility.
PROMPT_TEMPLATE_4 = """You are converting a math word problem and its gold step-by-step solution into FOUR progressively more refined representations. Each representation is a SMOOTH REFINEMENT of the previous, like a diffusion model denoising from text toward executable code.

ALL FOUR LEVELS must cover the FULL solution (all reasoning steps). Not just one step -- the entire chain.

LEVEL 0 -- paraphrased problem in plain English. State what kind of math is involved (addition, subtraction, multiplication, division, multi-step) but without committing to operations or numbers. Keep entity names. ~1-3 sentences.

LEVEL 1 -- same content as Level 0, but pull in the concrete numbers from the problem and explicitly NAME each operation in capitals. Still English sentences. Reads like a tutor explaining what to do. ~2-4 sentences.

LEVEL 2 -- equation form with light verbal context. Each step becomes a named quantity = an equation = a value. Like "Janet's eggs after laying = 50 + 15 = 65 eggs." Connect equations with prose like "Then" or "So". The equations are arithmetic over numbers (no variable names). ~2-6 short lines.

LEVEL 3 -- pure DAG syntax, fully executable, no English. Use only ASCII variables x0, x1, x2, ... and the final variable `answer`. Each statement is `xN = EXPR`. Operators: + - * / ** and parentheses. Statements are separated by ` ; ` (space-semicolon-space). The FINAL statement MUST be `answer = ...`. Every variable referenced must be defined earlier. Numbers are plain decimal (no commas, no units). Example: `x0 = 50 + 15 ; x1 = x0 * 2 ; answer = x1`.

CRITICAL RULES for Level 3:
- ASCII only. Variables x0, x1, x2, ... in order. Last assignment is `answer = ...`.
- Statements separated by ` ; `.
- Operators: + - * / ** (power). NO functions, NO comparisons, NO if/else.
- Numbers must be plain (e.g. "50" not "50 dollars" and not "50,000").
- The DAG, when evaluated, MUST equal the gold answer exactly.
- The DAG should mirror the structure of the gold solution: one xN per reasoning step.

Output format -- EXACTLY four lines, each prefixed with the level marker. Do NOT add any other text, headers, blank lines, or commentary.

L0: <level 0 text>
L1: <level 1 text>
L2: <level 2 text>
L3: <level 3 DAG>

EXAMPLE INPUT
Problem: Sam had 8 5 cookies. Sam doubled the collection, then gave 1 3 2 away. How many cookies does Sam have left?
Gold steps:
  - Sam had 8 5 cookies and doubled them. 8 5 * 2 = 1 7 0 cookies now.
  - Then Sam gave 1 3 2 away. 1 7 0 - 1 3 2 = 3 8 cookies remaining.
Gold answer: 38

EXAMPLE OUTPUT
L0: Sam starts with some cookies, doubles the amount, then gives some away. This is a two-step problem: multiplication then subtraction to find the remaining cookies.
L1: Sam has 85 cookies and doubles them (MULTIPLY by 2). Then Sam gives 132 cookies away (SUBTRACT 132). The result is the cookies remaining.
L2: Sam's doubled cookies = 85 * 2 = 170 cookies. Then cookies remaining = 170 - 132 = 38 cookies.
L3: x0 = 85 * 2 ; x1 = x0 - 132 ; answer = x1

NOW DO THIS PROBLEM
Problem: {problem}
Gold steps:
{steps_block}
Gold answer: {answer}

OUTPUT (four lines, L0-L3 only):"""


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    p = Path(ANTHROPIC_KEY_FALLBACK)
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError(
        f"No API key: set ANTHROPIC_API_KEY env var or place key at {ANTHROPIC_KEY_FALLBACK}"
    )


def load_problems(input_path: Path, num: int, start: int = 0) -> list[dict]:
    out = []
    with input_path.open("r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i < start:
                continue
            out.append(json.loads(line))
            if len(out) >= num:
                break
    return out


def format_steps_block(steps: list[str]) -> str:
    return "\n".join(f"  - {s}" for s in steps)


def parse_haiku_layers(text: str, n_layers: int) -> dict[str, str] | None:
    """Parse Haiku output expecting L0..L{n_layers-1} markers, tolerant of extra
    whitespace and blank lines.
    """
    layers: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(rf"^L(\d+)\s*:\s*(.*)$", line)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n_layers:
                layers[f"L{idx}"] = m.group(2).strip()
    required = {f"L{i}" for i in range(n_layers)}
    if required.issubset(layers.keys()):
        return layers
    return None


def call_haiku(client: Anthropic, problem: str, steps: list[str], answer: int,
               n_layers: int, max_retries: int = 2) -> tuple[dict[str, str] | None, dict | None]:
    """Call Haiku and parse n_layers output. Returns (layers_dict, usage_dict) or (None, None)."""
    if n_layers == 7:
        template = PROMPT_TEMPLATE_7
    elif n_layers == 6:
        template = PROMPT_TEMPLATE_6
    else:
        template = PROMPT_TEMPLATE_4
    prompt = template.format(
        problem=problem,
        steps_block=format_steps_block(steps),
        answer=answer,
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            parsed = parse_haiku_layers(text, n_layers)
            if parsed is not None:
                parsed["__raw__"] = text
                return parsed, usage
            last_err = f"failed to parse {n_layers} layers from output:\n{text[:300]}"
        except Exception as e:
            last_err = str(e)
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    return None, {"error": last_err}


# Counters used by the worker pool (locked by a Lock for thread safety).
class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.kept = 0
        self.failed_api = 0
        self.failed_sympy = 0
        self.failed_match = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.processed = 0


def _process_one(client: Anthropic, ex: dict, n_layers: int) -> dict:
    """Generate layers for one problem; return the record dict (or a failure marker)."""
    layers, usage = call_haiku(client, ex["problem"], ex["gen_targets"], ex["answer"],
                                n_layers=n_layers)
    if layers is None:
        return {"__failed_api__": True, "usage": usage or {}}

    last_key = f"L{n_layers - 1}"
    last_layer = layers[last_key]
    sympy_val = dag_to_answer(last_layer)
    sympy_ok = sympy_val is not None
    sympy_matches = sympy_ok and abs(sympy_val - ex["answer"]) < 1e-6

    record = {
        "problem": ex["problem"],
        "gen_targets": ex["gen_targets"],
        "answer": ex["answer"],
        "n_steps": ex.get("n_steps", len(ex["gen_targets"])),
        "layers": {f"L{i}": layers[f"L{i}"] for i in range(n_layers)},
        "sympy_value": sympy_val,
        "sympy_executable": sympy_ok,
        "sympy_matches_gold": sympy_matches,
        "usage": usage,
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    # Backwards-compatible flags
    parser.add_argument("--num", type=int, default=None,
                        help="Legacy alias for --num_problems")
    parser.add_argument("--input", type=Path, default=None,
                        help="Legacy alias for --src")
    parser.add_argument("--output", type=Path, default=None,
                        help="Legacy alias for --dst")
    # New (Phase 2) flags
    parser.add_argument("--src", type=Path, default=None,
                        help="Input JSONL (defaults to .cache/gsm8k_steps_v1_train.jsonl)")
    parser.add_argument("--dst", type=Path, default=None,
                        help="Output JSONL (defaults to .cache/gsm8k_steps_v77_sample.jsonl)")
    parser.add_argument("--num_problems", type=int, default=None,
                        help="Number of problems to process")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="7 (v78 default), 6 (v77 default) or 4 (legacy Phase 1)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N problems in the input file")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of parallel Haiku API calls (default 8)")
    parser.add_argument("--keep_failures", action="store_true",
                        help="Keep records that didn't match gold (default: keep all that produce executable DAG)")
    args = parser.parse_args()

    input_path = args.src or args.input or Path(".cache/gsm8k_steps_v1_train.jsonl")
    output_path = args.dst or args.output or Path(".cache/gsm8k_steps_v77_sample.jsonl")
    num = args.num_problems if args.num_problems is not None else (args.num if args.num is not None else 10)
    n_layers = int(args.n_layers)
    assert n_layers in (4, 6, 7), "n_layers must be 4, 6, or 7"

    os.environ["ANTHROPIC_API_KEY"] = load_api_key()

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    problems = load_problems(input_path, num, start=args.start)
    if not problems:
        print(f"ERROR: no problems found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(problems)} problems from {input_path} (start={args.start})")
    print(f"Generating {n_layers}-layer targets via {MODEL} (concurrency={args.concurrency})...")
    print(f"Output: {output_path}")

    client = Anthropic()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _Stats()
    t0 = time.perf_counter()
    write_lock = threading.Lock()

    with output_path.open("w") as out_f:
        # Submit all problems concurrently and write records as they complete.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_process_one, client, ex, n_layers): (i, ex)
                       for i, ex in enumerate(problems)}
            for fut in concurrent.futures.as_completed(futures):
                i, _ = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"__failed_api__": True, "error": str(e)}

                with stats.lock:
                    stats.processed += 1
                    if rec.get("__failed_api__"):
                        stats.failed_api += 1
                        progress = stats.processed
                        if progress % 100 == 0 or progress == len(problems):
                            elapsed = time.perf_counter() - t0
                            in_cost = (stats.in_tokens / 1e6) * _PRICE_IN_PER_MTOK
                            out_cost = (stats.out_tokens / 1e6) * _PRICE_OUT_PER_MTOK
                            print(f"  [{progress}/{len(problems)}]  kept={stats.kept}  "
                                  f"sympy_ok={stats.kept - stats.failed_sympy}  "
                                  f"matches={stats.kept - stats.failed_sympy - stats.failed_match}  "
                                  f"api_fail={stats.failed_api}  "
                                  f"cost=${in_cost + out_cost:.2f}  "
                                  f"({elapsed:.0f}s)",
                                  flush=True)
                        continue
                    usage = rec.get("usage") or {}
                    stats.in_tokens += int(usage.get("input_tokens", 0))
                    stats.out_tokens += int(usage.get("output_tokens", 0))
                    if not rec["sympy_executable"]:
                        stats.failed_sympy += 1
                    elif not rec["sympy_matches_gold"]:
                        stats.failed_match += 1

                # Write the record (sympy_executable success or not — caller can filter).
                # If keep_failures=False (default), still write — filtering is a separate step.
                with write_lock:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                with stats.lock:
                    stats.kept += 1
                    if stats.processed % 100 == 0 or stats.processed == len(problems):
                        elapsed = time.perf_counter() - t0
                        in_cost = (stats.in_tokens / 1e6) * _PRICE_IN_PER_MTOK
                        out_cost = (stats.out_tokens / 1e6) * _PRICE_OUT_PER_MTOK
                        rate = stats.processed / max(elapsed, 1e-6)
                        eta = (len(problems) - stats.processed) / max(rate, 1e-6)
                        print(f"  [{stats.processed}/{len(problems)}]  kept={stats.kept}  "
                              f"sympy_ok={stats.kept - stats.failed_sympy}  "
                              f"matches={stats.kept - stats.failed_sympy - stats.failed_match}  "
                              f"api_fail={stats.failed_api}  "
                              f"cost=${in_cost + out_cost:.2f}  "
                              f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                              flush=True)

    elapsed = time.perf_counter() - t0
    in_cost = (stats.in_tokens / 1e6) * _PRICE_IN_PER_MTOK
    out_cost = (stats.out_tokens / 1e6) * _PRICE_OUT_PER_MTOK
    print(f"\n=== generation done in {elapsed:.0f}s ===")
    print(f"Records written:  {stats.kept}/{len(problems)}")
    print(f"  sympy executes:    {stats.kept - stats.failed_sympy}/{stats.kept}")
    print(f"  sympy matches gold: {stats.kept - stats.failed_sympy - stats.failed_match}/{stats.kept}")
    print(f"  api failures:      {stats.failed_api}")
    print(f"Token usage:  in={stats.in_tokens:,}  out={stats.out_tokens:,}")
    print(f"Cost:         ${in_cost + out_cost:.2f}  (in=${in_cost:.2f} + out=${out_cost:.2f})")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
