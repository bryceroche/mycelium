"""Build v81: multi-list parallel supervision for the multi-head WaistController.

Each breath emits 4 PARALLEL LISTS separated by " | ":

    <ops_list> | <types_path_list> | <args1_list> | <args2_list>

Within a list, items are separated by ",". Within an item, dotted integers are
separated by ".".

Op encoding: ADD=0, SUB=1, MUL=2, DIV=3 (alphabetical).
Args encoding: positive integer = literal numeric value; negative integer = x_k
ref where x_1 = -1, x_2 = -2, etc.
Placeholder: "?".

Per-breath schedule (for DIV(50,60), MUL(x_1,12) — Weng's 2-step problem):

  B0: "?,? | ?,? | ?,? | ?,?"                         (skeleton — all placeholders)
  B1: "3,2 | ?,? | ?,? | ?,?"                         (+ ops)
  B2: "3,2 | 0,0 | ?,? | ?,?"                         (+ types_path depth 1)
  B3: "3,2 | 0.2,0.0 | ?,? | ?,?"                     (+ types_path depth 2)
  B4: "3,2 | 0.2.1,0.0.1 | ?,? | ?,?"                 (+ types_path leaf, depth 3)
  B5: "3,2 | 0.2.1,0.0.1 | 50,-1 | ?,?"               (+ args1)
  B6: "3,2 | 0.2.1,0.0.1 | 50,-1 | 60,12"             (+ args2 — full/executable)

Edge cases:
  - Shallow leaves (e.g. "DIV.0" → types_path "0"): for breaths B3/B4 requiring
    deeper depths, pad by repeating the last index. "DIV.0" at B3 (depth 2) →
    "0.0"; at B4 (depth 3) → "0.0.0".
  - >2 args per step: REJECTED in this first build (>2-arg records are dropped).
  - Unary ops (1 arg): args2 list item is "?" for that step (never filled).
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
    """Return the dotted types_path string at the requested depth.

    depth = 1, 2, 3 (relative to types_path, i.e. excluding OP).
    Pad with the last index if the leaf is shallower than requested.

    Examples (DIV.0.2.1):
      depth=1 → '0'
      depth=2 → '0.2'
      depth=3 → '0.2.1'

    Examples (MUL.1, only depth-1 types_path available):
      depth=1 → '1'
      depth=2 → '1.1'    (pad with last)
      depth=3 → '1.1.1'  (pad with last)
    """
    parts = leaf_to_types_path(leaf_id)
    if not parts:
        # Shouldn't happen for our 32-leaf tree (every leaf has at least one
        # sub-index), but be safe.
        return ""
    if len(parts) >= depth:
        return ".".join(parts[:depth])
    # Pad: repeat the LAST element until we hit depth.
    last = parts[-1]
    padded = parts + [last] * (depth - len(parts))
    return ".".join(padded)


# ---- args encoding ----

_X_REF_RE = re.compile(r"^x_(\d+)$")


def encode_arg(arg_str):
    """v81 arg encoding:
      - 'x_k'  → '-k'  (e.g. '-1', '-2', ...)
      - integer literal → unchanged integer string
      - decimal literal (e.g. '0.5') → unchanged decimal string (PRESERVED,
        not rounded — many GSM8K problems carry fractional rates).
      - anything else → returned raw (will likely fail SymPy execution).

    Returns the SYMBOL emitted in the supervision text. The parser
    (v81_sympy_eval.b6_string_to_dag) splits args1/args2 on ",", so commas
    are the only forbidden character inside an arg.
    """
    s = arg_str.strip()
    m = _X_REF_RE.match(s)
    if m:
        return str(-int(m.group(1)))
    # Try int — keep raw form.
    try:
        v = int(s)
        return str(v)
    except ValueError:
        pass
    # Float — keep the original string so we preserve the literal value.
    try:
        float(s)
        # Strip a leading "+" if present, normalise.
        return s.lstrip("+")
    except ValueError:
        # Unparseable → return raw (will likely fail SymPy later).
        return s


def encode_args(args):
    """Return (arg1, arg2) encoded strings. For 1-arg ops, arg2='?'.
    Caller has already filtered out >2-arg records.
    """
    if len(args) == 0:
        return "?", "?"
    if len(args) == 1:
        return encode_arg(args[0]), "?"
    # len == 2
    return encode_arg(args[0]), encode_arg(args[1])


# ---- Per-breath multi-list construction ----

def build_breath(breath_idx, steps, leaf_ids):
    """Build the 4-list space-pipe-space string for breath B<breath_idx>.

    steps: list of {idx, op, args} dicts.
    leaf_ids: parallel list of leaf_id strings.

    Layout (per step k=0..n_steps-1):
      ops_list[k]         = OP_TO_IDX[op] when B1+, else '?'
      types_path_list[k]  = padded types_path string at depth d when B2+, else '?'
                            d = 1 at B2, 2 at B3, 3 at B4+
      args1_list[k]       = encoded arg1 when B5+, else '?'
      args2_list[k]       = encoded arg2 when B6+, else '?'
    """
    n_steps = len(steps)
    ops_items = []
    types_items = []
    args1_items = []
    args2_items = []
    for step, lid in zip(steps, leaf_ids):
        # ops
        if breath_idx >= 1:
            ops_items.append(str(OP_TO_IDX[step["op"]]))
        else:
            ops_items.append("?")
        # types_path
        if breath_idx >= 2:
            if breath_idx == 2:
                depth = 1
            elif breath_idx == 3:
                depth = 2
            else:  # B4, B5, B6 — full leaf depth
                depth = 3
            types_items.append(types_path_at_depth(lid, depth))
        else:
            types_items.append("?")
        # args
        a1, a2 = encode_args(step["args"])
        if breath_idx >= 5:
            args1_items.append(a1)
        else:
            args1_items.append("?")
        if breath_idx >= 6:
            args2_items.append(a2)
        else:
            args2_items.append("?")
    ops_str = ",".join(ops_items)
    types_str = ",".join(types_items)
    args1_str = ",".join(args1_items)
    args2_str = ",".join(args2_items)
    return f"{ops_str} | {types_str} | {args1_str} | {args2_str}"


# ---- Per-record processing ----

def process_record(rec, tok, embed_w, leaves, centroids, leaf_ops):
    """Build a v81 record. Returns None to drop the input.

    Drop reasons:
      - L2/L4 don't parse
      - L2 step count != L4 step count
      - L2 op disagrees with L4 op for any step
      - any step has >2 args (multi-arg filter)
      - any step has 0 args (degenerate)
      - any step uses an op not in {ADD, SUB, MUL, DIV} (MOD/POW are dropped)
    """
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
    # Filter: only ADD/SUB/MUL/DIV, only 1-2 args.
    for ar in arg_steps:
        if ar["op"] not in OP_TO_IDX:
            return None
        if len(ar["args"]) == 0 or len(ar["args"]) > 2:
            return None
    # Assign each step to a leaf using OP-constrained nearest centroid.
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
        "leaf_ids": leaf_ids,  # debug aid
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
    ap.add_argument("--dst", default=".cache/gsm8k_steps_v81_train.jsonl")
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
            # Pre-classify to count drop categories.
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

    # 5 samples — print ALL 7 breaths for each
    print(f"\n=== 5 sample v81 records (verbatim B0..B6) ===")
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
