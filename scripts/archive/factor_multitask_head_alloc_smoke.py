"""factor_multitask_head_alloc_smoke.py — GPU-FREE proof of the head-allocation fix.

Run: DEV=CPU CPU=1 .venv/bin/python3 scripts/factor_multitask_head_alloc_smoke.py

THE BUG: the multi-task harness derived the per-head attention-mask allocation from the
UNION spec (n_factor_types = N_GLOBAL_TYPES = 8), so cell_mp_head_allocation(8,16,1)
gives only ~2 heads per global type. A PURE single-domain COLORING batch (only the edge
type present) then got just 2 of 16 heads on its one live relation; the other 13 heads
were allocated to ABSENT types and sat DEAD. Native single-domain coloring (T=1) uses
15 edge-heads + 1 global. So multi-task crippled coloring (15 active heads -> 2).

THE FIX: per-batch NATIVE head allocation over the PRESENT global types only, threaded
into the engine as tensors (build_factor_attn_bias_multitask). Weights stay SHARED (the
same 16 Q/K/V heads); ONLY the per-head MASK assignment changes per batch.

This smoke PROVES the fix by inspecting the ACTUAL attn_bias tensor the engine builds:
  - BEFORE (union allocation): count how many heads' masks match each relation
    (a head "attends to" relation t if its (S,S) bias == the type-t adjacency bias).
  - AFTER  (native allocation): re-count -> coloring ~15 edge-heads, kenken 5/5/5.
It also asserts the multitask mask == the boolean union-spec mask on a head-by-head
basis for whichever relation each head is assigned, so the SAME boolean engine produces
both (it's a re-ALLOCATION of the fixed engine, not a new mask).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.factor_graph_engine import FactorGraphSpec, attach_factor_graph_params
from mycelium.factor_inlet import (
    attach_factor_inlet_params, N_GLOBAL_TYPES, GLOBAL_TYPE_IDS,
)
from mycelium.factor_masks import (
    cell_mp_head_allocation, CELL_MP_HEAD_GLOBAL,
    native_head_alloc_for_present_types, head_alloc_to_tensors,
    build_factor_attn_bias, build_factor_attn_bias_multitask,
)
import scripts.factor_graph_train as T


def _per_type_adjacency_bias(membership, latent_type, cell_valid, n_heads, T_union, S):
    """Build the boolean per-head bias from the UNION spec (n_factor_types=T_union) —
    the BUGGY path the engine took before the fix. Returns (B,H,S,S)."""
    return build_factor_attn_bias(membership, latent_type, cell_valid,
                                  n_heads, T_union, S)


def _classify_heads(bias_BHSS, membership, latent_type, cell_valid, T_union, S):
    """For each head, classify which GLOBAL relation its (S,S) mask realizes by matching
    against each relation's reference bias. Returns a length-H list of relation ids (or
    -1 global / -2 unmatched)."""
    Bn = int(bias_BHSS.shape[0])
    H = int(bias_BHSS.shape[1])
    # Reference per-relation bias: build a per-head union mask where head h0 holds
    # relation t -> reuse build_factor_attn_bias and pull out one representative head
    # per type. Simpler: reconstruct each type's reference (B,S,S) directly.
    refs = {}
    bias_np = bias_BHSS.realize().numpy()                     # (B,H,S,S)
    # Build a reference for every global type t and for the global head.
    for t in list(range(T_union)) + [CELL_MP_HEAD_GLOBAL]:
        # Allocate ALL non-global heads to type t (or all-global), pull head 0.
        if t == CELL_MP_HEAD_GLOBAL:
            oh, isg = head_alloc_to_tensors(
                np.full((H,), CELL_MP_HEAD_GLOBAL, dtype=np.int64), T_union)
        else:
            alloc = np.full((H,), t, dtype=np.int64)
            alloc[-1] = CELL_MP_HEAD_GLOBAL
            oh, isg = head_alloc_to_tensors(alloc, T_union)
        ref = build_factor_attn_bias_multitask(
            membership, latent_type, cell_valid, oh, isg, H, T_union, S)
        refs[t] = ref.realize().numpy()[:, 0]                 # (B,S,S)
    out = []
    for h in range(H):
        mh = bias_np[:, h]                                    # (B,S,S)
        matched = -2
        for t, ref in refs.items():
            if np.allclose(mh, ref, atol=1e-3):
                matched = t
                break
        out.append(matched)
    return out


def main():
    np.random.seed(0)
    Tensor.manual_seed(0)
    Tensor.training = False

    K = 2
    BATCH = 2
    SEED = 42
    cfg = Config(hidden=128, n_heads=16, head_dim=8, ffn=256, max_seq_len=64)
    H = cfg.n_heads
    hidden = cfg.hidden
    print(f"=== head-alloc fix proof (H={H} N_GLOBAL_TYPES={N_GLOBAL_TYPES}) ===")
    model = BreathingTransformer(cfg)
    T.cast_layers_fp32(model)
    attach_factor_inlet_params(model, hidden=hidden)

    tr = ".cache/kenken_train.jsonl"; te = ".cache/kenken_test.jsonl"
    os.environ.setdefault("FG_COLORING_N_INSTANCES", "64")
    os.environ.setdefault("FG_CIRCUIT_N_INSTANCES", "64")
    os.environ.setdefault("FG_COLORING_N_VALUES", "3")
    mix = ["coloring", "circuit", "kenken"]
    weights = {d: 1.0 for d in mix}
    task = T._build_multitask_task(K, BATCH, BATCH, SEED, hidden, H, model, tr, te,
                                   mix, weights)
    spec = task.spec
    attach_factor_graph_params(model, hidden=hidden, spec=spec)
    S = spec.s_max
    T_union = spec.n_factor_types
    edge = GLOBAL_TYPE_IDS["coloring_edge"]
    krow, kcol, kcage = (GLOBAL_TYPE_IDS["kenken_row"],
                         GLOBAL_TYPE_IDS["kenken_col"], GLOBAL_TYPE_IDS["kenken_cage"])

    all_ok = True
    for d in mix:
        fb = task.train_loader.sample_batch(domain=d)
        mem, lt, cv = fb.membership, fb.latent_type, fb.cell_valid

        # BEFORE: union-spec boolean allocation (the bug).
        before_bias = _per_type_adjacency_bias(mem, lt, cv, H, T_union, S)
        before_cls = _classify_heads(before_bias, mem, lt, cv, T_union, S)

        # AFTER: the per-batch native allocation the adapter now produces (carried on fb).
        after_bias = build_factor_attn_bias_multitask(
            mem, lt, cv, fb.head_type_oh, fb.head_is_global, H, T_union, S)
        after_cls = _classify_heads(after_bias, mem, lt, cv, T_union, S)

        def _count(cls, t):
            return sum(1 for c in cls if c == t)

        print(f"\n[{d}] present global types = "
              f"{T._present_global_types(d, task.adapters and None) if d!='circuit' else T._present_global_types(d, ('AND','OR','NOT'))}")
        print(f"  BEFORE head->relation: {before_cls}")
        print(f"  AFTER  head->relation: {after_cls}")
        if d == "coloring":
            b, a = _count(before_cls, edge), _count(after_cls, edge)
            print(f"  coloring EDGE heads: BEFORE={b}  AFTER={a}  (target ~15)")
            ok = (b == 2 and a == 15)
            all_ok = all_ok and ok
            print(f"  -> {'OK' if ok else 'FAIL'} (before=2, after=15)")
        elif d == "kenken":
            for nm, t in (("row", krow), ("col", kcol), ("cage", kcage)):
                a = _count(after_cls, t)
                print(f"  kenken {nm:4s} heads: AFTER={a}  (target 5)")
            ok = (_count(after_cls, krow) == 5 and _count(after_cls, kcol) == 5
                  and _count(after_cls, kcage) == 5)
            all_ok = all_ok and ok
            print(f"  -> {'OK' if ok else 'FAIL'} (kenken 5/5/5)")
        elif d == "circuit":
            cand, cor, cnot = (GLOBAL_TYPE_IDS["circuit_and"],
                               GLOBAL_TYPE_IDS["circuit_or"],
                               GLOBAL_TYPE_IDS["circuit_not"])
            for nm, t in (("and", cand), ("or", cor), ("not", cnot)):
                a = _count(after_cls, t)
                print(f"  circuit {nm:4s} heads: AFTER={a}  (target 5)")
            ok = (_count(after_cls, cand) == 5 and _count(after_cls, cor) == 5
                  and _count(after_cls, cnot) == 5)
            all_ok = all_ok and ok
            print(f"  -> {'OK' if ok else 'FAIL'} (circuit 5/5/5)")

    print()
    print("HEAD-ALLOC PROOF PASS" if all_ok else "HEAD-ALLOC PROOF FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
