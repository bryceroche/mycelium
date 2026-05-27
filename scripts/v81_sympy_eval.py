"""v81 SymPy executor for the multi-list B6 format.

The model's last breath (B6) emits text of the form:

    "3,2 | 0.2.1,0.0.1 | 50,-1 | 60,12"

i.e. 4 lists separated by " | ":
  1. ops_list      — integer op codes (ADD=0, SUB=1, MUL=2, DIV=3)
  2. types_path    — dotted-integer cluster id at leaf depth (unused at execution)
  3. args1_list    — encoded arg1 per step (positive=literal, negative=x_k ref)
  4. args2_list    — encoded arg2 per step (or '?' for unary ops)

`b6_string_to_dag(text) -> str | None` parses this into a v77/v80-style DAG:

    "x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1"

`dag_to_answer` is re-exported from v77_sympy_eval (we don't duplicate it).

Encoding:
  op_idx in {0,1,2,3} maps to {+, -, *, /}.
  arg encoding:
    int >= 0  → literal numeric value
    int < 0   → reference to x_{-arg} (so -1 -> x_1 -> x0 in 0-indexed DAG)
    '?'       → placeholder (unfilled); presence at B6 means malformed → None

The function tolerates trailing whitespace / spurious tokens after the last list.
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
    """Parse a strict integer (used only for op codes, where the value is
    bounded to {0,1,2,3})."""
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
      - int (< 0) → x_k ref where k = -int
      - float → literal numeric value
    Returns None on '?' or unparseable.
    """
    s = s.strip()
    if not s or s == "?":
        return None
    # Int first.
    try:
        return int(s)
    except ValueError:
        pass
    # Float (positive or negative); fractional values are KEPT.
    try:
        f = float(s)
    except ValueError:
        return None
    # If it has a decimal point or otherwise isn't integer-valued, return float.
    if "." in s or f != int(f):
        return f
    return int(f)


def b6_string_to_dag(text: str) -> Optional[str]:
    """Parse v81 B6 multi-list text into a SymPy-executable DAG string.

    Returns None if any list is missing, the lists disagree in length, or any
    op/arg is unparseable.
    """
    if not text or not text.strip():
        return None
    # The model may emit extra junk after the 4th list. Truncate after we see 3
    # ' | ' separators (i.e. take the first 4 list segments only).
    segments = text.split("|")
    if len(segments) < 4:
        return None
    # Re-join in case extra '|' appear inside args (shouldn't, but be safe):
    # We take the first 4 segments and discard the rest. The 4th segment may
    # have trailing junk; we'll let the comma-split handle it.
    seg_ops = segments[0].strip()
    seg_types = segments[1].strip()
    seg_a1 = segments[2].strip()
    seg_a2 = segments[3].strip()
    # The 4th segment may have garbage after the last expected step; trim at
    # the first non-arg character. Arg tokens may contain digits, ',', '-', '?',
    # '.', and whitespace.
    seg_a2 = re.sub(r"[^0-9,\-. \t?].*$", "", seg_a2).strip()

    def split_list(s):
        if not s:
            return []
        return [x.strip() for x in s.split(",")]

    ops = split_list(seg_ops)
    types = split_list(seg_types)
    args1 = split_list(seg_a1)
    args2 = split_list(seg_a2)
    if not ops:
        return None
    # types may have different validity but length should match (we don't use
    # the values for execution).
    n_steps = len(ops)
    if len(args1) != n_steps or len(args2) != n_steps:
        return None
    # types length should match (just sanity, not used for execution)
    if len(types) != n_steps:
        # Don't bail — we don't need types for execution.
        pass

    def render_arg(enc) -> Optional[str]:
        if enc is None:
            return None
        # Float literal — render as-is (SymPy parses floats).
        if isinstance(enc, float):
            return repr(enc)
        # Int: negative = x_k ref (k = -enc - 1, so -1 -> x0).
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
        a1_enc = _parse_arg(args1[k])
        a2_enc = _parse_arg(args2[k])
        # Unary ops are not supported at execution — every B6 step needs 2 args.
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
    print("=== v81 SymPy DAG eval — self-test ===")
    tests = [
        # Weng: DIV(50,60), MUL(x_1,12) -> 10
        ("Weng 2-step",
         "3,2 | 0.2.1,0.0.1 | 50,-1 | 60,12",
         10.0),
        # Joy: DIV(20,8), MUL(120,x1), DIV(x2,60) -> assume 5 (20/8=2.5, *120=300, /60=5)
        ("Joy 3-step",
         "3,2,3 | 0.2.0,1,0.1 | 20,120,-2 | 8,-1,60",
         5.0),
        # ADD with two literals
        ("simple add",
         "0 | 0 | 50 | 15",
         65.0),
        # SUB
        ("simple sub",
         "1 | 1 | 100 | 25",
         75.0),
        # placeholder in B6 → None
        ("unfilled placeholder",
         "3 | 0 | ? | 60",
         None),
        # malformed: only 3 segments
        ("missing list",
         "3 | 0 | 50",
         None),
    ]
    passed = sum(_t(*t) for t in tests)
    print(f"\n=== {passed}/{len(tests)} passed ===")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
