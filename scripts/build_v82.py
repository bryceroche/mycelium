"""Build v82: parallel-diffusion supervision for the SINGLE-HEAD WaistController.

v82 is the next architecture after v81. v81 failed because multi-head per-position
CE diluted content learning into structural-token CE. v82 fixes this with SINGLE-
HEAD full-sequence CE on a PARALLEL DIFFUSION target schedule.

Each breath emits the FULL 3-list sequence (not 4 — args1/args2 are combined into
one interleaved args list). Coarser breaths use LOWER PRECISION values, not
placeholders. Every breath refines ALL lists by one precision step.

Data format (3 parallel lists separated by " | "):

    <ops_csv> | <types_csv> | <args_csv>

    ops:   n_steps entries     — OP indices (ADD=0, SUB=1, MUL=2, DIV=3)
    types: n_steps entries     — IB tree cluster paths (e.g. "0.2.1")
    args:  n_steps * 2 entries — interleaved per step: step0.arg0, step0.arg1,
                                  step1.arg0, step1.arg1, ...

Args encoding:
  positive int           = literal numeric value (e.g. 50, 12)
  negative int (-k)      = reference to x_k     (e.g. -1 = x_1)
  '?'                    = placeholder
  'r'                    = MAGNITUDE marker for an x_k ref at P1 precision
  power-of-10 integer    = magnitude category for a literal at P1 precision

Precision schedule:

  OPs (P0/P1):
    P0: '?'
    P1: exact (0..3)

  Types (P0..P3):
    P0: '?'
    P1: depth-1 prefix (single digit, e.g. '0', '2')
    P2: depth-2 prefix (e.g. '0.2')
    P3: leaf path (e.g. '0.2.1')

  Args (P0..P3 — but P2 == P3 since exact saturates):
    P0: '?'
    P1: literal -> nearest lower power of 10 (preserves sign); x_k -> 'r'
    P2: exact
    P3: exact

Breath schedule (lockstep refinement across lists):

  Breath | ops | types | args
    B0  |  P0  |  P0   | P0    (all placeholders)
    B1  |  P1  |  P1   | P1    (ops exact + types depth-1 + args magnitude)
    B2  |  P1  |  P2   | P2    (types depth-2 + args exact)
    B3  |  P1  |  P3   | P3    (types leaf + everything exact)
    B4  |  P1  |  P3   | P3    (refinement pass — same target)
    B5  |  P1  |  P3   | P3    (refinement pass)
    B6  |  P1  |  P3   | P3    (refinement pass — final)

Edge cases:
  - Shallow leaves (e.g. "DIV.0" — only depth-1 types_path): pad by repeating
    the last index for depth-2 / depth-3 targets ("0.0" / "0.0.0").
  - >2 args per step: REJECTED (we drop the record).
  - Unary ops (1 arg): args list emits the lone arg + '?' as the second arg.
"""
import os
import sys
import json
import re
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import diag_ib_clustering as ibc  # type: ignore


L2_STEP_RE = re.compile(r"^Step \d+:\s*(.+?)\.\s*OP=(ADD|SUB|MUL|DIV)\.\s*ARG=([-\d.]+)\.\s*$")
L4_STEP_RE = re.compile(r"^x_(\d+)\s*:=\s*<(ADD|SUB|MUL|DIV|MOD|POW)>\(([^()]*)\)\s*$")


# Alphabetical op encoding: ADD=0, SUB=1, MUL=2, DIV=3.
OP_TO_IDX = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}


def parse_l2_nl(l2_text):
    """Yield {nl, op} per step."""
    out = []
    for ln in l2_text.split("\n"):
        m = L2_STEP_RE.match(ln.strip())
        if m:
            out.append({"nl": m.group(1).strip(), "op": m.group(2)})
    return out


def parse_l4_args(l4_text):
    """Yield {idx, op, args} per step (args = list of raw strings as in L4)."""
    out = []
    for ln in l4_text.split("\n"):
        m = L4_STEP_RE.match(ln.strip())
        if m:
            args_raw = m.group(3).strip()
            args = [a.strip() for a in args_raw.split(",")] if args_raw else []
            out.append({"idx": int(m.group(1)), "op": m.group(2), "args": args})
    return out


def load_tree(tree_path, cent_path):
    with open(tree_path) as f:
        meta = json.load(f)
    leaves = meta["leaves"]
    cent = np.load(cent_path)
    npz_ids = list(cent["leaf_ids"])
    centroids = cent["centroids"]
    leaf_idx_by_id = {lid: i for i, lid in enumerate(npz_ids)}
    aligned = []
    for l in leaves:
        idx = leaf_idx_by_id[l["leaf_id"]]
        aligned.append({**l, "_centroid_idx": idx})
    leaf_ops = np.array([l["op"] for l in aligned])
    return aligned, centroids, leaf_ops, meta["max_depth"]


def assign_leaf(nl, op, tok, embed_w, centroids, leaf_ops):
    """OP-constrained nearest-centroid leaf assignment."""
    emb = ibc.embed_text(nl, tok, embed_w)
    emb_n = emb / (np.linalg.norm(emb) + 1e-12)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = cent_n @ emb_n
    op_mask = (leaf_ops == op)
    sims = np.where(op_mask, sims, -np.inf)
    return int(np.argmax(sims))


# ---- types_path extraction ----

def leaf_to_types_path(leaf_id):
    """Drop the OP prefix. e.g. 'DIV.0.2.1' → ['0', '2', '1']; 'MUL.1' → ['1']."""
    parts = leaf_id.split(".")
    return parts[1:]  # everything after OP


def types_path_at_depth(leaf_id, depth):
    """Return the dotted types_path string at the requested depth (1, 2, or 3)."""
    parts = leaf_to_types_path(leaf_id)
    if not parts:
        return ""
    if len(parts) >= depth:
        return ".".join(parts[:depth])
    last = parts[-1]
    padded = parts + [last] * (depth - len(parts))
    return ".".join(padded)


# ---- args encoding ----

_X_REF_RE = re.compile(r"^x_(\d+)$")


def encode_arg_exact(arg_str):
    """Exact (P2/P3) arg encoding:
      - 'x_k'        → '-k'  (e.g. '-1', '-2', ...)
      - integer      → unchanged
      - decimal      → unchanged (e.g. '0.5')
      - anything     → raw (will likely fail SymPy)
    """
    s = arg_str.strip()
    m = _X_REF_RE.match(s)
    if m:
        return str(-int(m.group(1)))
    try:
        v = int(s)
        return str(v)
    except ValueError:
        pass
    try:
        float(s)
        return s.lstrip("+")
    except ValueError:
        return s


def _power_of_10_floor(x):
    """Return the largest power of 10 less-than-or-equal-to |x|, preserving sign.

    Examples:
      50    -> 10
      234   -> 100
      7     -> 1
      0.5   -> 0.1
      -25   -> -10
      0     -> 0    (special — no defined floor; emit '0')
      0.05  -> 0.01
    """
    if x == 0:
        return 0
    sign = -1 if x < 0 else 1
    ax = abs(float(x))
    # log10 floor.
    import math
    exp = int(math.floor(math.log10(ax)))
    mag = 10 ** exp
    # For integer values, return integer if exp >= 0; else preserve float (e.g. 0.1).
    if exp >= 0:
        return sign * int(mag)
    return sign * mag


def encode_arg_magnitude(arg_str):
    """P1 arg encoding (magnitude):
      - x_k                → 'r'  (single character; we don't predict WHICH ref at P1)
      - integer literal    → nearest lower power of 10 (preserves sign)
      - decimal literal    → nearest lower power of 10 as decimal string
      - unparseable        → raw
    """
    s = arg_str.strip()
    m = _X_REF_RE.match(s)
    if m:
        return "r"
    # Try int first.
    try:
        v = int(s)
        out = _power_of_10_floor(v)
        return str(out)
    except ValueError:
        pass
    # Float.
    try:
        f = float(s)
    except ValueError:
        return s
    out = _power_of_10_floor(f)
    # Preserve original decimal-ness when input was float; render small powers of 10
    # like 0.1 / 0.01 as decimals not "1e-1".
    if isinstance(out, int):
        return str(out)
    # Format with up to 12 fractional digits then strip trailing zeros.
    formatted = f"{out:.12f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def encode_args_for_step_exact(args):
    """Return (a1, a2) exact-encoded for a single step. Unary -> a2='?'."""
    if len(args) == 0:
        return "?", "?"
    if len(args) == 1:
        return encode_arg_exact(args[0]), "?"
    return encode_arg_exact(args[0]), encode_arg_exact(args[1])


def encode_args_for_step_mag(args):
    """Return (a1, a2) magnitude-encoded for a single step. Unary -> a2='?'."""
    if len(args) == 0:
        return "?", "?"
    if len(args) == 1:
        return encode_arg_magnitude(args[0]), "?"
    return encode_arg_magnitude(args[0]), encode_arg_magnitude(args[1])


# ---- Per-breath 3-list construction ----

def build_breath(breath_idx, steps, leaf_ids):
    """Build the 3-list space-pipe-space string for breath B<breath_idx>.

    steps: list of {idx, op, args}.
    leaf_ids: parallel list of leaf_id strings.

    args are emitted INTERLEAVED: step0.arg0, step0.arg1, step1.arg0, step1.arg1...
    """
    n_steps = len(steps)
    # ops precision
    if breath_idx >= 1:
        ops_items = [str(OP_TO_IDX[s["op"]]) for s in steps]
    else:
        ops_items = ["?"] * n_steps

    # types precision (P0=?, P1=depth1, P2=depth2, P3=depth3)
    if breath_idx == 0:
        types_items = ["?"] * n_steps
    else:
        if breath_idx == 1:
            depth = 1
        elif breath_idx == 2:
            depth = 2
        else:  # B3..B6
            depth = 3
        types_items = [types_path_at_depth(lid, depth) for lid in leaf_ids]

    # args precision (interleaved over steps)
    args_items = []
    for step in steps:
        if breath_idx == 0:
            a1, a2 = "?", "?"
        elif breath_idx == 1:
            a1, a2 = encode_args_for_step_mag(step["args"])
        else:  # B2..B6 — exact
            a1, a2 = encode_args_for_step_exact(step["args"])
        args_items.append(a1)
        args_items.append(a2)

    ops_str = ",".join(ops_items)
    types_str = ",".join(types_items)
    args_str = ",".join(args_items)
    return f"{ops_str} | {types_str} | {args_str}"


# ---- Per-record processing ----

def process_record(rec, tok, embed_w, leaves, centroids, leaf_ops):
    """Build a v82 record. Returns None to drop the input."""
    v80_layers = rec["layers"]
    l2_text = v80_layers.get("L2", "")
    l4_text = v80_layers.get("L4", "")
    nl_steps = parse_l2_nl(l2_text)
    arg_steps = parse_l4_args(l4_text)
    if not nl_steps or not arg_steps or len(nl_steps) != len(arg_steps):
        return None
    for ns, ar in zip(nl_steps, arg_steps):
        if ns["op"] != ar["op"]:
            return None
    for ar in arg_steps:
        if ar["op"] not in OP_TO_IDX:
            return None
        if len(ar["args"]) == 0 or len(ar["args"]) > 2:
            return None
    leaf_ids = []
    for ns, ar in zip(nl_steps, arg_steps):
        idx = assign_leaf(ns["nl"], ns["op"], tok, embed_w, centroids, leaf_ops)
        leaf_ids.append(leaves[idx]["leaf_id"])
    n_steps = len(arg_steps)
    layers = {
        f"L{i}": build_breath(i, arg_steps, leaf_ids) for i in range(7)
    }
    return {
        "problem": rec["problem"],
        "answer": rec["answer"],
        "n_steps": n_steps,
        "leaf_ids": leaf_ids,
        "layers": layers,
        **layers,
        "problem_type": rec.get("problem_type", ""),
        "gen_targets": rec.get("gen_targets", []),
        "sympy_value": rec.get("sympy_value"),
        "sympy_matches_gold": rec.get("sympy_matches_gold", True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".cache/gsm8k_steps_v80_train.jsonl")
    ap.add_argument("--dst", default=".cache/gsm8k_steps_v82_train.jsonl")
    ap.add_argument("--tree", default=".cache/ib_tree.json")
    ap.add_argument("--centroids", default=".cache/ib_centroids.npz")
    ap.add_argument("--pythia", default=".cache/pythia-410m/model.safetensors")
    ap.add_argument("--num", type=int, default=0, help="limit input records (0=all)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[1/3] Loading Pythia + IB tree...")
    embed_w = ibc.load_pythia_embed_numpy(args.pythia)
    tok = ibc.load_tokenizer()
    leaves, centroids, leaf_ops, max_depth = load_tree(args.tree, args.centroids)
    print(f"  embed: {embed_w.shape}  leaves: {len(leaves)}  max_depth: {max_depth}")

    print(f"\n[2/3] Processing {args.src} -> {args.dst}")
    out_path = Path(args.dst)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    kept = 0
    drop_parse = 0
    drop_op = 0
    drop_args = 0
    drop_other = 0
    with open(args.src) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if args.num and i >= args.num:
                break
            rec = json.loads(line)
            v80_layers = rec["layers"]
            nl_steps = parse_l2_nl(v80_layers.get("L2", ""))
            arg_steps = parse_l4_args(v80_layers.get("L4", ""))
            drop_reason = None
            if (not nl_steps or not arg_steps
                or len(nl_steps) != len(arg_steps)):
                drop_reason = "parse"
            else:
                for ns, ar in zip(nl_steps, arg_steps):
                    if ns["op"] != ar["op"]:
                        drop_reason = "parse"
                        break
                    if ar["op"] not in OP_TO_IDX:
                        drop_reason = "op"
                        break
                    if len(ar["args"]) == 0 or len(ar["args"]) > 2:
                        drop_reason = "args"
                        break
            if drop_reason is None:
                out_rec = process_record(rec, tok, embed_w, leaves,
                                         centroids, leaf_ops)
                if out_rec is None:
                    drop_other += 1
                    continue
                fout.write(json.dumps(out_rec) + "\n")
                kept += 1
            else:
                if drop_reason == "parse":
                    drop_parse += 1
                elif drop_reason == "op":
                    drop_op += 1
                elif drop_reason == "args":
                    drop_args += 1
            total = kept + drop_parse + drop_op + drop_args + drop_other
            if total % 500 == 0:
                print(f"  [{total}]  kept={kept}  drop[parse/op/args/other]="
                      f"{drop_parse}/{drop_op}/{drop_args}/{drop_other}  "
                      f"({time.perf_counter() - t0:.1f}s)")

    print(f"\n[3/3] Done.  kept={kept}  drop[parse/op/args/other]="
          f"{drop_parse}/{drop_op}/{drop_args}/{drop_other}  out={out_path}")

    # 5 samples — print ALL 7 breaths for each.
    print(f"\n=== 5 sample v82 records (verbatim B0..B6) ===")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            r = json.loads(line)
            print(f"\n--- sample {i + 1} (n_steps={r['n_steps']}, leaves={r['leaf_ids']}) ---")
            print(f"problem: {r['problem'][:120]}")
            for L in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
                print(f"  {L}: {r[L]}")


if __name__ == "__main__":
    main()
