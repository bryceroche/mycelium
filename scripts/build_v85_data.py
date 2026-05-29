"""Build v85 data: differentiable queryable structures.

v85 replaces v82-v84's text-based DAG supervision with STRUCTURED slot
supervision. Each problem gets:
  - numbers[]: literal numeric values in the prompt + char spans
  - verbs[]: action verbs in the prompt + char spans
  - entities[]: named entities (optional)
  - dag_slots[]: gold DAG with args pointing to numbers[] (or earlier dag_slots)

The model learns to bind args to ACTUAL prompt numbers (via pointer attention),
not to memorize digit sequences. Numbers/verbs/entities become queryable
differentiable tensors that attention heads natively query.

Input source: .cache/gsm8k_steps_v80_train.jsonl + _test.jsonl (Haiku-annotated
L0-L6 layers). We use L4 (`x_k := <OP>(args)`) as the structural source and ask
Haiku to extract numbers + verbs + bind args to numbers[] indices.

NL semantics (e.g. "half" = 0.5) stay in Pythia. We only extract STRUCTURE.

Strict JSON output schema (per record):
{
  "numbers": [
    {"value": 50, "span_start_char": 47, "span_end_char": 49, "is_literal": true},
    ...
  ],
  "verbs": [
    {"verb": "earns", "span_start_char": 6, "span_end_char": 11},
    ...
  ],
  "entities": [
    {"text": "Weng", "span_start_char": 0, "span_end_char": 4}
  ],
  "dag_slots": [
    {"op": "DIV", "type_path": "0.1.1", "args": [
      {"source": "numbers", "index": 0, "value": 50},
      {"source": "numbers", "index": 1, "value": 60}
    ], "is_active": true},
    {"op": "MUL", "type_path": "0.0.1", "args": [
      {"source": "dag",     "index": 0, "value_ref": "x0"},
      {"source": "numbers", "index": 2, "value": 12}
    ], "is_active": true}
  ],
  "n_steps": 2
}

Validation gate:
- Every dag_slots[k].args[i].source == "numbers" MUST reference a real
  numbers[j] entry
- Every dag_slots[k].args[i].source == "dag" MUST reference k' < k
- Spans must lie within the problem text
- SymPy execution of the DAG (using numbers values as operands) MUST equal
  the gold answer (within 1e-3)
- n_steps <= K_max (=10)

Drop records that fail validation.

Budget: ~$5-15 for ~5000 records (Haiku is ~$1-5/MT input, ~$5/MT output).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: .venv/bin/pip install anthropic",
          file=sys.stderr)
    sys.exit(1)

import numpy as np

from v77_sympy_eval import dag_to_answer  # type: ignore

# Reuse v82's IB tree helpers + types path lookup so v85 types_path encodings
# remain compatible with the existing 32-leaf IB codebook.
import diag_ib_clustering as ibc  # type: ignore
from build_v82 import (  # type: ignore
    load_tree, assign_leaf, types_path_at_depth, OP_TO_IDX, parse_l2_nl, parse_l4_args,
)


MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_KEY_FALLBACK = "/home/bryce/Desktop/keys/key1.txt"

_PRICE_IN_PER_MTOK = 1.0
_PRICE_OUT_PER_MTOK = 5.0


HAIKU_PROMPT = r"""You will extract STRUCTURAL information from a math word problem so a neural net can bind its DAG to actual operand positions.

# Problem
__PROBLEM__

# Gold answer
__ANSWER__

# Existing DAG (L4 format, for reference — your dag_slots MUST mirror these ops + step count)
__L4__

# Numbers already extracted from the problem text (by regex)
__NUMBERS_LIST__

You DO NOT need to find character positions. The numbers list above is canonical — use those indices.

# What to extract

Output a SINGLE JSON object with these keys (and nothing else — no prose, no markdown):

1. "implicit_numbers": list of literal numbers that are NEEDED for the DAG computation but DO NOT
   appear in the problem text (e.g. unit conversion constants like 60 for minutes/hour, or 100 for
   percentages). Each entry:
     - "value": numeric value (int or float)
   The model will treat these as "implicit_K" pseudo-positions. If none are needed, return [].

2. "verbs": list of action verbs in the problem text (just the verb string). MAX 10 verbs.
   Each entry: {"verb": "earns"}

3. "entities": list of named entities (people, places, things). MAX 5 entities.
   Each entry: {"text": "Weng"}

4. "dag_slots": list of computation steps mirroring the L4 DAG. Each entry:
   - "op": one of "ADD", "SUB", "MUL", "DIV" (matching L4 step k's op)
   - "type_path": semantic cluster path string (e.g. "0.1.1"). Best-effort; ignored by validator.
   - "args": list of EXACTLY 2 arg objects. For unary L4 steps, repeat the single arg.
     Each arg is one of:
       {"source": "numbers",  "index": <i>, "value": <numeric_value>}
         — pointer to numbers[i] from the input list above
       {"source": "implicit", "index": <i>, "value": <numeric_value>}
         — pointer to implicit_numbers[i] (the list you return in field 1)
       {"source": "dag",      "index": <k>, "value_ref": "x<k>"}
         — pointer to a previous dag_slot's output; index k is 0-indexed (x0 = dag_slots[0])
   - "is_active": always true

5. "n_steps": INTEGER — must equal len(dag_slots) and match L4's step count.

# Validation rules (failures cause record drop)
- args[i].source="numbers": index must reference a real numbers[j] in the input list; value must
  match numbers[index].value
- args[i].source="implicit": index must reference a real implicit_numbers[j] in your output; value
  must match
- args[i].source="dag": index must be < k (earlier step)
- Executing the DAG (using actual numeric values) must equal the gold answer (within 1e-3)
- n_steps == len(dag_slots) == L4 step count
- Output MUST be a single JSON object, no prose, no markdown

# Examples

Example 1:
Problem: "Weng earns $1 2 an hour for babysitting. Yesterday, she just did 5 0 minutes of babysitting. How much did she earn?"
Gold answer: 10
L4:
x_1 := <DIV>(50, 60)
x_2 := <MUL>(x_1, 12)
ANSWER := x_2
Numbers list:
  [0]: 12
  [1]: 50

Output:
{
  "implicit_numbers": [{"value": 60}],
  "verbs": [
    {"verb": "earns"},
    {"verb": "did"},
    {"verb": "earn"}
  ],
  "entities": [{"text": "Weng"}],
  "dag_slots": [
    {"op": "DIV", "type_path": "0.1.1", "args": [
      {"source": "numbers",  "index": 1, "value": 50},
      {"source": "implicit", "index": 0, "value": 60}
    ], "is_active": true},
    {"op": "MUL", "type_path": "2.2.1", "args": [
      {"source": "dag",     "index": 0, "value_ref": "x0"},
      {"source": "numbers", "index": 0, "value": 12}
    ], "is_active": true}
  ],
  "n_steps": 2
}

Example 2:
Problem: "A bag has 1 5 apples. James adds 7 more. How many apples are in the bag?"
Gold answer: 22
L4:
x_1 := <ADD>(15, 7)
ANSWER := x_1
Numbers list:
  [0]: 15
  [1]: 7

Output:
{
  "implicit_numbers": [],
  "verbs": [{"verb": "has"}, {"verb": "adds"}],
  "entities": [{"text": "James"}],
  "dag_slots": [
    {"op": "ADD", "type_path": "0.0.0", "args": [
      {"source": "numbers", "index": 0, "value": 15},
      {"source": "numbers", "index": 1, "value": 7}
    ], "is_active": true}
  ],
  "n_steps": 1
}

# Now process the problem above. Output ONLY the JSON object.
"""


def load_key() -> str:
    """Load Anthropic API key from env or fallback file."""
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    if k:
        return k
    if os.path.exists(ANTHROPIC_KEY_FALLBACK):
        with open(ANTHROPIC_KEY_FALLBACK) as f:
            return f.read().strip()
    raise RuntimeError("No ANTHROPIC_API_KEY in env and key fallback missing.")


def parse_json_object(text: str) -> dict | None:
    """Best-effort JSON parse: strip fences, find first { ... } block."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


_NUMBER_RE = re.compile(r"(?<!\d)(\d(?:\s\d)*(?:\.\s?\d+)?)(?!\s?\d)")


def extract_numbers_from_problem(problem: str) -> list[dict]:
    """Regex-based extractor for digit-spaced numbers in the problem text.

    Examples that match:
      "1 2"      → value 12
      "5 0"      → value 50
      "0.5"      → value 0.5
      "1 2.5"    → value 12.5 (less common but supported)
    Returns: list of {"value", "span_start_char", "span_end_char", "is_literal": True}
    """
    out = []
    for m in _NUMBER_RE.finditer(problem):
        raw = m.group(1)
        compressed = raw.replace(" ", "")
        try:
            if "." in compressed:
                v = float(compressed)
                # Drop bizarre floats like "0.5.0" — re will not produce them, but be safe.
                if v != v:
                    continue
            else:
                v = int(compressed)
        except ValueError:
            continue
        out.append({
            "value": v,
            "span_start_char": m.start(1),
            "span_end_char": m.end(1),
            "is_literal": True,
        })
    return out


def execute_dag(dag_slots, numbers, implicit_numbers) -> float | None:
    """Execute a v85 DAG using actual numeric values from numbers + implicit_numbers."""
    op_table = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}
    parts = []
    for k, slot in enumerate(dag_slots):
        op = slot.get("op", "")
        if op not in op_table:
            return None
        sym = op_table[op]
        args = slot.get("args", [])
        if len(args) != 2:
            return None
        rendered = []
        for arg in args:
            src = arg.get("source", "")
            idx = arg.get("index", -1)
            if src == "numbers":
                if not (0 <= idx < len(numbers)):
                    return None
                rendered.append(str(numbers[idx]["value"]))
            elif src == "implicit":
                if not (0 <= idx < len(implicit_numbers)):
                    return None
                rendered.append(str(implicit_numbers[idx]["value"]))
            elif src == "dag":
                if not (0 <= idx < k):
                    return None
                rendered.append(f"x{idx}")
            else:
                return None
        parts.append(f"x{k} = {rendered[0]} {sym} {rendered[1]}")
    parts.append(f"answer = x{len(dag_slots) - 1}")
    dag_str = " ; ".join(parts)
    return dag_to_answer(dag_str)


def validate_v85_obj(obj: dict, gold_answer: float, problem: str, numbers: list[dict],
                     K_max: int = 10, tol: float = 1e-3):
    """Validate a Haiku output object. Returns (ok, error_string).

    `numbers` is the canonical (regex-extracted) numbers list — we DON'T accept Haiku's numbers.
    """
    for k in ("implicit_numbers", "verbs", "entities", "dag_slots", "n_steps"):
        if k not in obj:
            return False, f"missing key {k!r}"
    if not isinstance(obj["implicit_numbers"], list):
        return False, "implicit_numbers must be list"
    if not isinstance(obj["dag_slots"], list):
        return False, "dag_slots must be list"
    n_steps = obj["n_steps"]
    if not isinstance(n_steps, int):
        return False, f"n_steps must be int, got {type(n_steps).__name__}"
    if n_steps != len(obj["dag_slots"]):
        return False, f"n_steps={n_steps} != len(dag_slots)={len(obj['dag_slots'])}"
    if n_steps > K_max:
        return False, f"n_steps={n_steps} > K_max={K_max}"
    if n_steps < 1:
        return False, "n_steps must be >= 1"

    # Validate implicit numbers
    for i, n in enumerate(obj["implicit_numbers"]):
        if "value" not in n:
            return False, f"implicit_numbers[{i}] missing value"
        if not isinstance(n["value"], (int, float)):
            return False, f"implicit_numbers[{i}].value not numeric"

    # Validate dag_slots
    for k, slot in enumerate(obj["dag_slots"]):
        for kk in ("op", "type_path", "args", "is_active"):
            if kk not in slot:
                return False, f"dag_slots[{k}] missing key {kk!r}"
        if slot["op"] not in OP_TO_IDX:
            return False, f"dag_slots[{k}].op {slot['op']!r} not in {list(OP_TO_IDX)}"
        if not isinstance(slot["args"], list) or len(slot["args"]) != 2:
            return False, f"dag_slots[{k}].args must be list of length 2"
        for ai, arg in enumerate(slot["args"]):
            if "source" not in arg or "index" not in arg:
                return False, f"dag_slots[{k}].args[{ai}] missing source/index"
            src = arg["source"]
            idx = arg["index"]
            if src == "numbers":
                if not (0 <= idx < len(numbers)):
                    return False, f"dag_slots[{k}].args[{ai}] numbers idx {idx} out of range (n={len(numbers)})"
                if "value" in arg and isinstance(arg["value"], (int, float)):
                    if abs(float(arg["value"]) - float(numbers[idx]["value"])) > 1e-6:
                        return False, (f"dag_slots[{k}].args[{ai}] value {arg['value']} "
                                       f"!= numbers[{idx}].value {numbers[idx]['value']}")
            elif src == "implicit":
                if not (0 <= idx < len(obj["implicit_numbers"])):
                    return False, f"dag_slots[{k}].args[{ai}] implicit idx {idx} out of range"
            elif src == "dag":
                if not (0 <= idx < k):
                    return False, f"dag_slots[{k}].args[{ai}] dag idx {idx} not earlier than step {k}"
            else:
                return False, f"dag_slots[{k}].args[{ai}] source {src!r} not in (numbers, implicit, dag)"

    # Validate via SymPy execution
    val = execute_dag(obj["dag_slots"], numbers, obj["implicit_numbers"])
    if val is None:
        return False, "DAG failed to execute"
    if abs(val - gold_answer) > tol:
        return False, f"DAG executed to {val} != gold {gold_answer}"

    return True, ""


def call_haiku(client, problem: str, answer: float, l4: str, numbers: list[dict],
               max_retries: int = 2) -> tuple[dict | None, dict, str | None]:
    """Call Haiku to extract v85 structure. Returns (validated_obj, usage, error_string)."""
    numbers_list_str = "\n".join(f"  [{i}]: {n['value']}" for i, n in enumerate(numbers))
    if not numbers_list_str:
        numbers_list_str = "  (none — all operands must come from implicit_numbers or dag refs)"
    prompt = (HAIKU_PROMPT
              .replace("__PROBLEM__", problem)
              .replace("__ANSWER__", str(answer))
              .replace("__L4__", l4)
              .replace("__NUMBERS_LIST__", numbers_list_str))

    in_tok = 0
    out_tok = 0
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            last_err = f"API error: {e}"
            continue
        in_tok += msg.usage.input_tokens
        out_tok += msg.usage.output_tokens
        text = msg.content[0].text if msg.content else ""
        obj = parse_json_object(text)
        if obj is None:
            last_err = "JSON parse failed"
            continue
        ok, err = validate_v85_obj(obj, gold_answer=float(answer), problem=problem,
                                    numbers=numbers)
        if ok:
            return obj, {"input_tokens": in_tok, "output_tokens": out_tok}, None
        last_err = f"validation: {err}"

    return None, {"input_tokens": in_tok, "output_tokens": out_tok}, last_err


def annotate_with_ib(obj: dict, leaves, centroids, leaf_ops, tok, embed_w,
                     l2_text: str) -> dict:
    """Augment dag_slots with IB-leaf-aligned type_path (overrides Haiku's guess).

    Same logic as v82: pull L2's NL step descriptions, do OP-constrained nearest-
    centroid lookup → leaf_id → types_path. Falls back to Haiku's type_path if
    L2 has fewer step descriptions than dag_slots.
    """
    nl_steps = parse_l2_nl(l2_text)
    new_slots = []
    for k, slot in enumerate(obj["dag_slots"]):
        new_slot = dict(slot)
        if k < len(nl_steps) and nl_steps[k]["op"] == slot["op"]:
            idx = assign_leaf(nl_steps[k]["nl"], slot["op"], tok, embed_w, centroids, leaf_ops)
            new_slot["leaf_id"] = leaves[idx]["leaf_id"]
            new_slot["type_path"] = types_path_at_depth(leaves[idx]["leaf_id"], 3)
        new_slots.append(new_slot)
    obj["dag_slots"] = new_slots
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".cache/gsm8k_steps_v80_train.jsonl")
    ap.add_argument("--dst", default=".cache/gsm8k_steps_v85_train.jsonl")
    ap.add_argument("--tree", default=".cache/ib_tree.json")
    ap.add_argument("--centroids", default=".cache/ib_centroids.npz")
    ap.add_argument("--pythia", default=".cache/pythia-410m/model.safetensors")
    ap.add_argument("--num", type=int, default=0, help="limit records (0=all)")
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--budget_usd", type=float, default=15.0,
                    help="stop early if estimated cost exceeds this")
    ap.add_argument("--print_every", type=int, default=50)
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[1/4] Loading Pythia + IB tree for type_path augmentation...")
    embed_w = ibc.load_pythia_embed_numpy(args.pythia)
    tok = ibc.load_tokenizer()
    leaves, centroids, leaf_ops, max_depth = load_tree(args.tree, args.centroids)
    print(f"  embed: {embed_w.shape}  leaves: {len(leaves)}")

    print(f"[2/4] Initializing Anthropic client...")
    key = load_key()
    client = Anthropic(api_key=key)

    out_path = Path(args.dst)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"[3/4] Processing {args.src} -> {args.dst}  K_max={args.K_max}")
    kept = 0
    dropped = {"parse_fail": 0, "validation": 0, "api": 0, "k_max": 0}
    total_in_tok = 0
    total_out_tok = 0
    sample_records = []

    with open(args.src) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if args.num and i >= args.num:
                break

            # Budget guard.
            est_cost = (total_in_tok * _PRICE_IN_PER_MTOK +
                        total_out_tok * _PRICE_OUT_PER_MTOK) / 1e6
            if est_cost > args.budget_usd:
                print(f"\n[BUDGET STOP] ${est_cost:.2f} exceeds budget ${args.budget_usd:.2f}",
                      flush=True)
                break

            rec = json.loads(line)
            problem = rec["problem"]
            answer = float(rec["answer"])
            layers = rec.get("layers", {})
            l4 = layers.get("L4", "")
            l2 = layers.get("L2", "")
            arg_steps = parse_l4_args(l4)
            if len(arg_steps) > args.K_max:
                dropped["k_max"] += 1
                continue
            if len(arg_steps) < 1:
                dropped["parse_fail"] += 1
                continue

            # Extract numbers via regex (canonical, deterministic).
            numbers = extract_numbers_from_problem(problem)
            if not numbers:
                # Some problems have ZERO literal numbers (rare). Skip these — there's
                # nothing for the model to bind to.
                dropped["parse_fail"] += 1
                continue

            obj, usage, err = call_haiku(client, problem, answer, l4, numbers)
            total_in_tok += usage.get("input_tokens", 0)
            total_out_tok += usage.get("output_tokens", 0)
            if obj is None:
                if err and err.startswith("validation"):
                    dropped["validation"] += 1
                else:
                    dropped["api"] += 1
                # Periodic logging — even drops should be visible.
                total = kept + sum(dropped.values())
                if total % args.print_every == 0:
                    est_cost = (total_in_tok * _PRICE_IN_PER_MTOK +
                                total_out_tok * _PRICE_OUT_PER_MTOK) / 1e6
                    print(f"  [{total}] kept={kept}  drop={dict(dropped)}  ${est_cost:.2f}  "
                          f"({time.perf_counter() - t0:.1f}s)  last_err={err!r}",
                          flush=True)
                continue

            # Inject regex-extracted numbers into obj (canonical).
            obj["numbers"] = numbers
            # Augment type_path from IB clustering (replaces Haiku's guess).
            obj = annotate_with_ib(obj, leaves, centroids, leaf_ops, tok, embed_w, l2)

            # Write the output record.
            out_rec = {
                "problem": problem,
                "answer": answer,
                "n_steps": obj["n_steps"],
                "v85": obj,  # full v85 structured annotation
                # Keep v80/v82 layer fields for compatibility with eval/v77 loader.
                "layers": layers,
                "problem_type": rec.get("problem_type", ""),
                "sympy_value": rec.get("sympy_value"),
                "sympy_matches_gold": rec.get("sympy_matches_gold", True),
                "gen_targets": rec.get("gen_targets", []),
            }
            fout.write(json.dumps(out_rec) + "\n")
            kept += 1
            if len(sample_records) < 5:
                sample_records.append(out_rec)

            total = kept + sum(dropped.values())
            if total % args.print_every == 0:
                est_cost = (total_in_tok * _PRICE_IN_PER_MTOK +
                            total_out_tok * _PRICE_OUT_PER_MTOK) / 1e6
                print(f"  [{total}] kept={kept}  drop={dict(dropped)}  ${est_cost:.2f}  "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)

    est_cost = (total_in_tok * _PRICE_IN_PER_MTOK +
                total_out_tok * _PRICE_OUT_PER_MTOK) / 1e6
    print(f"\n[4/4] Done. kept={kept}  drop={dict(dropped)}  "
          f"in_tok={total_in_tok}  out_tok={total_out_tok}  ${est_cost:.2f}  "
          f"wall={time.perf_counter() - t0:.1f}s")

    # n_steps and N distribution stats.
    n_steps_hist = {}
    n_numbers_hist = {}
    n_verbs_hist = {}
    n_implicit_hist = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("v85", {})
            ns = v.get("n_steps", 0)
            nn = len(v.get("numbers", []))
            ni = len(v.get("implicit_numbers", []))
            nv = len(v.get("verbs", []))
            n_steps_hist[ns] = n_steps_hist.get(ns, 0) + 1
            n_numbers_hist[nn] = n_numbers_hist.get(nn, 0) + 1
            n_implicit_hist[ni] = n_implicit_hist.get(ni, 0) + 1
            n_verbs_hist[nv] = n_verbs_hist.get(nv, 0) + 1
    print(f"\n=== Distribution stats (n_kept={kept}) ===")
    print(f"  n_steps:   {dict(sorted(n_steps_hist.items()))}")
    print(f"  n_numbers: {dict(sorted(n_numbers_hist.items()))}")
    print(f"  n_implicit: {dict(sorted(n_implicit_hist.items()))}")
    print(f"  n_verbs:   {dict(sorted(n_verbs_hist.items()))}")

    print(f"\n=== 5 sample v85 records ===")
    for i, r in enumerate(sample_records):
        print(f"\n--- sample {i + 1} ---")
        print(f"problem: {r['problem'][:200]}")
        print(f"answer: {r['answer']}")
        print(f"n_steps: {r['n_steps']}")
        print(json.dumps(r["v85"], indent=2)[:1500])


if __name__ == "__main__":
    main()
