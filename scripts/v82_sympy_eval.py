"""v82 SymPy executor for the 3-list B6 format.

The model's last breath (B6) emits text of the form:

    "3,2 | 0.2.1,0.0.1 | 50,60,-1,12"

i.e. 3 lists separated by " | ":
  1. ops_list   — integer op codes (ADD=0, SUB=1, MUL=2, DIV=3); n_steps entries
  2. types_list — dotted cluster path per step (unused at execution); n_steps entries
  3. args_list  — INTERLEAVED args (step0.arg0, step0.arg1, step1.arg0, ...);
                  n_steps * 2 entries.

`b6_string_to_dag(text) -> str | None` parses this into a v77/v80-style DAG:

    "x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1"

`dag_to_answer` is re-exported from v77_sympy_eval.

Args encoding:
  int >= 0   → literal numeric value
  int < 0    → x_{-arg} ref (so -1 -> x_1 -> x0 in 0-indexed DAG)
  '?'/'r'    → unfilled / magnitude marker → malformed at B6, returns None
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

# Re-export the standard DAG executor.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v77_sympy_eval import dag_to_answer  # type: ignore  # noqa: F401


_OP_TABLE = ["+", "-", "*", "/"]


def _parse_int(s: str) -> Optional[int]:
    """Parse a strict integer (used only for op codes 0..3)."""
    s = s.strip()
    if not s or s == "?":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(round(float(s)))
        except ValueError:
            return None


def _parse_arg(s: str):
    """Parse an arg token into either:
      - int (>= 0) → literal numeric value
      - int (< 0)  → x_k ref where k = -int
      - float      → literal numeric value
    Returns None on '?', 'r', or unparseable.
    """
    s = s.strip()
    if not s or s == "?" or s == "r":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return None
    if "." in s or f != int(f):
        return f
    return int(f)


def b6_string_to_dag(text: str) -> Optional[str]:
    """Parse v82 B6 3-list text into a SymPy-executable DAG string."""
    if not text or not text.strip():
        return None
    segments = text.split("|")
    if len(segments) < 3:
        return None
    seg_ops = segments[0].strip()
    seg_types = segments[1].strip()
    seg_args = segments[2].strip()
    # Args segment may have trailing garbage after the last expected arg.
    # Allowed chars: digits, ',', '-', '.', whitespace, '?', 'r'.
    seg_args = re.sub(r"[^0-9,\-. \tr?].*$", "", seg_args).strip()

    def split_list(s):
        if not s:
            return []
        return [x.strip() for x in s.split(",")]

    ops = split_list(seg_ops)
    types = split_list(seg_types)
    args_flat = split_list(seg_args)
    if not ops:
        return None
    n_steps = len(ops)
    # types length should match (just sanity, not used for execution).
    if len(args_flat) != n_steps * 2:
        return None

    def render_arg(enc) -> Optional[str]:
        if enc is None:
            return None
        if isinstance(enc, float):
            return repr(enc)
        if enc < 0:
            k = -enc - 1
            if k < 0:
                return None
            return f"x{k}"
        return str(enc)

    parts = []
    for k in range(n_steps):
        op_enc = _parse_int(ops[k])
        if op_enc is None or not (0 <= op_enc < len(_OP_TABLE)):
            return None
        sym = _OP_TABLE[op_enc]
        a1_enc = _parse_arg(args_flat[k * 2])
        a2_enc = _parse_arg(args_flat[k * 2 + 1])
        if a1_enc is None or a2_enc is None:
            return None
        a1 = render_arg(a1_enc)
        a2 = render_arg(a2_enc)
        if a1 is None or a2 is None:
            return None
        parts.append(f"x{k} = {a1} {sym} {a2}")
    parts.append(f"answer = x{n_steps - 1}")
    return " ; ".join(parts)


def _t(label, text, expected_answer):
    dag = b6_string_to_dag(text)
    val = dag_to_answer(dag) if dag else None
    if expected_answer is None:
        ok = val is None
        exp_str = "None"
    else:
        ok = val is not None and abs(val - expected_answer) < 1e-6
        exp_str = str(expected_answer)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    print(f"          input: {text!r}")
    print(f"          dag:   {dag!r}")
    print(f"          val={val}  expected={exp_str}")
    return ok


def main():
    print("=== v82 SymPy DAG eval — self-test ===")
    tests = [
        # Weng: DIV(50,60), MUL(x_1,12) -> 10
        ("Weng 2-step",
         "3,2 | 0.2.1,0.0.1 | 50,60,-1,12",
         10.0),
        # Joy: DIV(20,8), MUL(120,x_1), DIV(x_2,60) -> 5
        ("Joy 3-step",
         "3,2,3 | 0.2.0,1.0.0,0.1.0 | 20,8,120,-1,-2,60",
         5.0),
        # ADD with two literals
        ("simple add",
         "0 | 0 | 50,15",
         65.0),
        # SUB
        ("simple sub",
         "1 | 1.0.0 | 100,25",
         75.0),
        # placeholder in B6 → None
        ("unfilled placeholder",
         "3 | 0.2.1 | ?,60",
         None),
        # 'r' magnitude marker in B6 → None
        ("magnitude marker",
         "3 | 0.2.1 | r,60",
         None),
        # malformed: only 2 segments
        ("missing list",
         "3 | 0",
         None),
        # arg count mismatch (1 step but 1 arg)
        ("arg count off",
         "3 | 0.2.1 | 50",
         None),
    ]
    passed = sum(_t(*t) for t in tests)
    print(f"\n=== {passed}/{len(tests)} passed ===")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
