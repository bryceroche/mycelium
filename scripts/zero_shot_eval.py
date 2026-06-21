"""zero_shot_eval.py — ZERO-SHOT generalization harness for the general-weights model.

THE QUESTION
------------
The multi-task checkpoint (one DENSE shared Pythia-410M backbone, co-trained on
{coloring, circuit, kenken} with semantics-as-INPUT via the gated generic inlet)
claims to be a general MODEL, not just general CODE. The cleanest test of that claim
is ZERO-SHOT (no retraining, forward-only) transfer to factor graphs the model never
saw at train time. This harness runs that test in two HONEST flavors that, together,
map exactly what semantics-as-input buys:

  1. FAIR FLAVOR (expected to WORK) — NEW PARAMS of a TRAINED relation type.
     The model trained on KenKen (cage + Latin-square all-different relations; the
     kenken_row/col/cage gates open). We generate a HELD-OUT KenKen distribution —
     N=4 boards — that the training corpus NEVER contains (train/test are exclusively
     N in {5,6,7}; see §HELD-OUT AXIS). The relation TYPE is identical (same cage +
     all-different logic, same global type-ids, same inlet) — only the params (a
     smaller board, a new value-domain {1..4}) are new. If semantics-as-input gives a
     general model, the model solves N=4 zero-shot at acc COMPARABLE to its
     in-distribution N in {5,6,7} acc, NOT collapsed to chance. This tests the
     inlet/predicate-registry claim: "new params of a known relation = free."

  2. NEGATIVE CONTROL (expected to FAIL) — a genuinely NEW RELATION TYPE.
     We build small Boolean circuits whose ONLY gates are XOR (output = a XOR b),
     remapped to the GLOBAL type-id `circuit_xor` (= 4). The fair checkpoint trained
     circuit with gate_types = {AND, OR, NOT} (measured_config), so XOR's per-type
     inlet gate `fg_inlet_gate[4]` NEVER received gradient and is ~0 -> the inlet
     contributes ~nothing for XOR latents -> the shared backbone has NO learned way to
     USE this relation's semantics. Per the deduction-not-induction + attention-
     bootstrap boundary, NEW relation LOGIC needs gradient; it cannot be inferred
     zero-shot from the type-id alone. Predict: ~chance. This is the control that
     PROVES the boundary: if the fair flavor works AND this fails, we have cleanly
     separated "new params of a trained relation = free" from "new relation type =
     needs retrain."

  HONEST FRAMING (printed in the verdict): the fair flavor tests new-PARAMS-of-
  trained-relations (the inlet's real value); the negative control confirms that a
  new-RELATION-TYPE needs a retrain (the boundary). We do NOT claim a new relation
  type SHOULD work zero-shot — its failure is the EXPECTED, informative result.

VERDICT
-------
SEMANTICS-AS-INPUT GENERALIZES iff:
  (a) fair-flavor zero-shot acc is COMPARABLE to in-distribution (not chance), AND
  (b) the negative control FAILS (~chance).
Comparable := fair cell_acc within FAIR_TOL of in-dist cell_acc AND well above chance.

HELD-OUT AXIS (the fair flavor) — WHY N=4 IS GENUINELY HELD OUT
--------------------------------------------------------------
Surveyed .cache/kenken_train.jsonl + .cache/kenken_test.jsonl (the corpus the
fair checkpoint's measured_config points at): BOTH are exclusively N in {5,6,7}
(train: N5=340 N6=341 N7=339; test: N5=60 N6=59 N7=61). The board size axis is the
cleanest held-out axis because:
  - N=4 appears NOWHERE in train or test (zero leakage by construction).
  - N=4 lays on the SAME fixed 7x7=49-cell grid the engine pins (s_max=49), exactly
    as N=5/6 already do — it is a sub-grid, not a new topology. (N>=8 is impossible:
    the engine hard-pins s_max=49; N=4 is the natural held-out direction.)
  - It is the SAME relation type (KenKen cages + Latin-square all-different), so the
    SAME global type-ids (kenken_row/col/cage = 5/6/7) and the SAME inlet fire — this
    is "new params of a TRAINED relation," exactly the claim under test.
  - Generation is supported by the existing build_kenken_data generator (verified
    100% unique-yield at N=4) and verified by the EXACT count_solutions verifier.
This is strictly stronger than "same structure, new clue numbers" (the by-instance
split) — it is a board size the model has never seen.

NEGATIVE-CONTROL CONSTRUCTION (the new relation type)
-----------------------------------------------------
A Boolean circuit of 2-input XOR gates: inputs are free Boolean cells; each gate cell
g has two parents (p1, p2) and gold(g) = gold(p1) XOR gold(p2). Each gate is ONE
factor (membership = {g, p1, p2}) with GLOBAL type-id `circuit_xor` (= 4). The model
trained circuit on {AND, OR, NOT} only, so:
  - The membership topology is a valid DAG the engine can consume.
  - The type-id 4 indexes a real inlet-table row, BUT its gate is ~0 (never trained),
    so the inlet adds ~nothing -> no learned XOR semantics.
  - XOR is the ONE 2-input gate whose output is NOT a monotone/threshold function of
    its inputs, so even the structural mask cannot leak the answer the way AND/OR
    might — it genuinely requires the (absent) learned relation logic.
Predict: ~chance (cell_acc ~ 0.5 on the 2-value codebook; puzzle_acc ~ 0).

GPU-FREE BUILD + CPU SELFTEST
-----------------------------
This file is GPU-FREE to build: `python -c "import ast; ast.parse(open(...).read())"`
parses it, and `--selftest` runs the FULL harness path (load -> generate held-out
batch -> forward -> metric -> verdict) against a MOCK tiny model + a MOCK tiny
checkpoint, with NO GPU and NO dependence on the real (still-training) checkpoint.
The real eval is a one-liner (see RUN, bottom of this docstring), run post-verdict on
a GPU once fg_multi_fair_final.safetensors exists.

REUSE (no engine / kenken.py / trainer-training-path modification — this is additive)
------------------------------------------------------------------------------------
  scripts.factor_graph_train : the multi-task model build, the ckpt load (load_ckpt),
                               the FactorGraphSpec, the per-domain ADAPTERS (kenken +
                               a generic adapter pattern), cast_layers_fp32,
                               collect_backbone_params, model_state_dict_fg.
  scripts.build_kenken_data  : gen_unique_puzzle (N=4 generation) + count_solutions /
                               cage_ok / propagate (the EXACT verifier).
  mycelium.factor_inlet      : GLOBAL_TYPE_IDS, N_GLOBAL_TYPES,
                               attach_factor_inlet_params, build_generic_factor_inlet.
  mycelium.factor_graph_engine : factor_breathing_forward (the eager eval forward).
  mycelium.factor_masks      : native_head_alloc_for_present_types, head_alloc_to_tensors.
  mycelium.kenken / kenken_data : the verification-inlet ids (op/target/size) for the
                               kenken adapter (only the inlet tables, never the heads).

RUN (the real zero-shot eval, post-verdict, on a GPU)
-----------------------------------------------------
  DEV=AMD .venv/bin/python3 scripts/zero_shot_eval.py \
      --ckpt .cache/fg_ckpts/fg_multi_fair/fg_multi_fair_final.safetensors

CPU SELFTEST (GPU-free, no real ckpt; runs here)
------------------------------------------------
  .venv/bin/python3 scripts/zero_shot_eval.py --selftest
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ---------------------------------------------------------------------------
# Constants — the harness's pinned knobs.
# ---------------------------------------------------------------------------

HELD_OUT_N = 4                  # the fair-flavor held-out KenKen board size.
IN_DIST_NS = (5, 6, 7)         # the trained KenKen board sizes (in-distribution).
FAIR_TOL = 0.20                 # cell_acc within this of in-dist -> "comparable".
CHANCE_MARGIN = 0.10            # acc must beat (chance + this) to count as non-chance.
DEFAULT_K = 16                  # breaths (matches the fair checkpoint's K).
DEFAULT_CKPT = ".cache/fg_ckpts/fg_multi_fair/fg_multi_fair_final.safetensors"
DEFAULT_TRAIN = ".cache/kenken_train.jsonl"
DEFAULT_TEST = ".cache/kenken_test.jsonl"


# ===========================================================================
# DATA GENERATION (CPU-only, no tinygrad needed for the puzzle generation).
# ===========================================================================

def generate_heldout_kenken(n: int, count: int, seed: int) -> list[dict]:
    """Generate `count` uniqueness-verified KenKen puzzles of board size `n`.

    Reuses scripts.build_kenken_data.gen_unique_puzzle — the SAME generator that built
    the training corpus, just at a board size the corpus does not contain. Every puzzle
    is count_solutions==1 (unique) by construction, so puzzle_acc's proper-solution
    check (the exact verifier) is well-defined.
    """
    from scripts.build_kenken_data import gen_unique_puzzle
    rng = random.Random(seed)
    out: list[dict] = []
    tries = 0
    while len(out) < count and tries < count * 60:
        tries += 1
        bb = rng.random()                       # vary cage structure -> vary depth
        pz = gen_unique_puzzle(n, rng, bb)
        if pz is not None:
            out.append(pz)
    return out


def verify_kenken_solution(rec: dict, pred_grid: "list[list[int]]") -> bool:
    """EXACT verifier: is `pred_grid` a PROPER solution to puzzle `rec`?

    A proper KenKen solution must (a) be a Latin square (each value 1..N once per row
    and per col) and (b) satisfy EVERY cage's arithmetic constraint. This is the
    count_solutions/cage_ok verifier from build_kenken_data, NOT a gold-match — a
    different-but-valid solution would still pass (though the corpus is uniqueness-
    verified, so the proper solution is unique). pred_grid is the model's N x N
    prediction (values 1..N).
    """
    from scripts.build_kenken_data import cage_ok
    n = int(rec["N"])
    g = pred_grid
    # (a) Latin-square: each row and each col is a permutation of 1..n.
    target = set(range(1, n + 1))
    for r in range(n):
        if set(g[r][c] for c in range(n)) != target:
            return False
    for c in range(n):
        if set(g[r][c] for r in range(n)) != target:
            return False
    # (b) every cage's arithmetic constraint holds.
    for cage, clue in zip(rec["cages"], rec["clues"]):
        typ, tgt = clue[0], clue[1]
        asg = [g[int(r)][int(c)] for (r, c) in cage]
        if not cage_ok(typ, tgt, asg):
            return False
    return True


def build_xor_circuit(n_inputs: int, n_gates: int, seed: int) -> dict:
    """Construct ONE Boolean circuit whose every gate is a 2-input XOR.

    Returns a domain-agnostic factor-graph record (NOT a KenKen rec):
      {
        "kind": "xor_circuit",
        "n_cells": int,                  # total cells (inputs + gates), <= 49
        "gold": [int,...],               # gold Boolean value per cell (0/1)
        "input_cells": [int,...],        # observed value+1 per cell (0=unknown);
                                          #   inputs are GIVEN, gates are unknown.
        "factors": [ {"members":[g,p1,p2], "type":"circuit_xor"}, ... ],
      }

    Topology: cells [0, n_inputs) are free Boolean inputs (GIVEN/observed). Each gate
    cell g in [n_inputs, n_inputs+n_gates) picks two distinct EARLIER cells (p1, p2)
    and gold(g) = gold(p1) XOR gold(p2). One factor per gate: members = {g, p1, p2},
    type = circuit_xor. The inputs are observed; the gates must be DEDUCED — this is
    the deduction the model has no learned XOR rule for.
    """
    rng = random.Random(seed)
    n_cells = n_inputs + n_gates
    assert n_cells <= 49, f"xor circuit n_cells={n_cells} exceeds the 49-cell grid"
    gold = [0] * n_cells
    for i in range(n_inputs):
        gold[i] = rng.randint(0, 1)
    factors = []
    for gi in range(n_inputs, n_cells):
        p1, p2 = rng.sample(range(gi), 2)       # two distinct earlier cells
        gold[gi] = gold[p1] ^ gold[p2]
        factors.append({"members": [gi, p1, p2], "type": "circuit_xor"})
    # inputs are GIVEN (observed = value+1); gates unknown (0).
    input_cells = [0] * n_cells
    for i in range(n_inputs):
        input_cells[i] = gold[i] + 1            # 1 or 2 (value+1 convention)
    return {
        "kind": "xor_circuit",
        "n_cells": n_cells,
        "gold": gold,
        "input_cells": input_cells,
        "factors": factors,
        "n_inputs": n_inputs,
        "n_gates": n_gates,
    }


def generate_xor_circuits(count: int, seed: int,
                          n_inputs: int = 8, n_gates: int = 16) -> list[dict]:
    """Generate `count` XOR-only Boolean circuits (the negative-control corpus)."""
    rng = random.Random(seed)
    return [build_xor_circuit(n_inputs, n_gates, rng.randint(0, 2**31 - 1))
            for _ in range(count)]


# ===========================================================================
# TENSOR BATCH BUILDERS — turn generated puzzles into the unified _MultiTaskBatch
# contract (s_max=49, value_domain padded to 7, GLOBAL type-ids, native head alloc).
# These mirror the trainer's adapters but are eval-only (no JIT, eager).
# ===========================================================================

def _make_kenken_eval_batch(recs: list[dict], model, K: int, hidden: int,
                            n_heads: int, spec, n_cages_max: int):
    """Build a unified multi-task eval batch from a list of KenKen puzzle records.

    Reuses the trainer's KenKen path exactly: KenKenLoader-style encode + the kenken
    multi-task adapter (latent_type remapped to GLOBAL row/col/cage ids; per-latent
    op/target/size for the inlet; native 5/5/5+1 head allocation). The generic inlet
    is built eagerly (build_generic_factor_inlet), so the model reads the TRAINED
    kenken_row/col/cage inlet gates.
    """
    from tinygrad import Tensor, dtypes
    from mycelium.kenken_data import encode_puzzle, N_MAX
    from mycelium.factor_graph_engine import make_kenken_factor_batch
    from mycelium.factor_inlet import GLOBAL_TYPE_IDS, build_generic_factor_inlet
    from mycelium.factor_masks import (
        native_head_alloc_for_present_types, head_alloc_to_tensors,
    )

    B = len(recs)
    L_max = int(getattr(model, "_zse_L_max", N_MAX + N_MAX + n_cages_max))

    # ---- stack the per-puzzle KenKen tensors (KenKenLoader._stack, inlined). ----
    encs = [encode_puzzle(r, n_cages_max) for r in recs]

    def stack_int(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                      dtype=dtypes.int).contiguous().realize()

    def stack_f(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                      dtype=dtypes.float).contiguous().realize()

    class _KB:
        pass
    kb = _KB()
    kb.input_cells = stack_int("input_cells")
    kb.gold = stack_int("gold")
    kb.cell_valid = stack_f("cell_valid")
    kb.cell_cage_id = stack_int("cell_cage_id")
    kb.value_domain_mask = stack_f("value_domain_mask")
    kb.cage_op = stack_int("cage_op")
    kb.cage_target = stack_int("cage_target")
    kb.cage_size = stack_int("cage_size")
    kb.deduction_depth = [int(r.get("deduction_depth", 0)) for r in recs]

    # ---- KenKen factor batch (rows/cols/cages membership + local 0/1/2 types). ----
    kk_spec_T3 = type(spec)(s_max=49, n_values=7, n_factor_types=3,
                            n_heads=n_heads, k_max=K, has_factor_inlet=True)
    fb = make_kenken_factor_batch(kb, kk_spec_T3)

    # ---- remap local 0/1/2 -> GLOBAL row/col/cage ids; pad to L_max. ----
    row_gid = GLOBAL_TYPE_IDS["kenken_row"]
    col_gid = GLOBAL_TYPE_IDS["kenken_col"]
    cage_gid = GLOBAL_TYPE_IDS["kenken_cage"]
    local_to_global = np.array([row_gid, col_gid, cage_gid], dtype=np.int32)
    sentinel = int(spec.n_factor_types)         # MT sentinel = N_GLOBAL_TYPES

    lt = fb.latent_type.realize().numpy().astype(np.int32)        # (B, L) local 0/1/2
    lt_g = local_to_global[np.clip(lt, 0, 2)].astype(np.int32)
    mem = fb.membership.realize().numpy().astype(np.float32)      # (B, L, 49)
    L = mem.shape[1]
    if L < L_max:
        mem_p = np.zeros((B, L_max, 49), dtype=np.float32)
        mem_p[:, :L, :] = mem
        lt_p = np.full((B, L_max), sentinel, dtype=np.int32)
        lt_p[:, :L] = lt_g
    else:
        mem_p, lt_p = mem, lt_g

    # vdm pad 7-wide is a no-op for kenken (already 7).
    vdm = kb.value_domain_mask

    # ---- per-latent op/target/size aligned to the cage latents (rows/cols = 0). ----
    C = int(kb.cage_op.shape[1])
    n_rowcol = 2 * N_MAX
    op_np = np.zeros((B, L_max), dtype=np.int32)
    tgt_np = np.zeros((B, L_max), dtype=np.int32)
    sz_np = np.zeros((B, L_max), dtype=np.int32)
    op_np[:, n_rowcol:n_rowcol + C] = kb.cage_op.realize().numpy().astype(np.int32)
    tgt_np[:, n_rowcol:n_rowcol + C] = kb.cage_target.realize().numpy().astype(np.int32)
    sz_np[:, n_rowcol:n_rowcol + C] = kb.cage_size.realize().numpy().astype(np.int32)

    membership_t = Tensor(mem_p, dtype=dtypes.float).contiguous().realize()
    latent_type_t = Tensor(lt_p, dtype=dtypes.int).contiguous().realize()
    op_t = Tensor(op_np, dtype=dtypes.int).contiguous().realize()
    tgt_t = Tensor(tgt_np, dtype=dtypes.int).contiguous().realize()
    sz_t = Tensor(sz_np, dtype=dtypes.int).contiguous().realize()

    # ---- native head allocation for the PRESENT kenken types (5/5/5 + 1 global). ----
    present = [row_gid, col_gid, cage_gid]
    hga = native_head_alloc_for_present_types(present, n_heads)
    head_type_oh, head_is_global = head_alloc_to_tensors(hga, spec.n_factor_types)
    head_type_oh = head_type_oh.realize()
    head_is_global = head_is_global.realize()

    # ---- build the generic inlet eagerly (reads the TRAINED kenken inlet gates). ----
    inlet = build_generic_factor_inlet(
        model, membership_t, latent_type_t, kb.cell_valid,
        op=op_t, target=tgt_t, size=sz_t).realize()

    return _EvalBatch(
        input_cells=kb.input_cells, cell_valid=kb.cell_valid,
        value_domain_mask=vdm, gold=kb.gold,
        membership=membership_t, latent_type=latent_type_t,
        factor_inlet=inlet, head_type_oh=head_type_oh,
        head_is_global=head_is_global, recs=recs, domain="kenken")


def _make_xor_eval_batch(circuits: list[dict], model, K: int, hidden: int,
                         n_heads: int, spec):
    """Build a unified multi-task eval batch from XOR-circuit records (negative control).

    Mirrors the trainer's circuit adapter, but the gate-type is the NOVEL circuit_xor
    GLOBAL id. The inlet carries that type-id (op/target/size = 0); since the XOR gate's
    inlet gate is ~0 (never trained), the inlet contributes ~nothing -> the model has no
    learned way to use this relation. The native head allocation gives circuit_xor its
    own heads (single present type -> 15 + 1 global), so the model gets every chance
    structurally; only the LEARNED relation logic is absent.
    """
    from tinygrad import Tensor, dtypes
    from mycelium.factor_inlet import GLOBAL_TYPE_IDS, build_generic_factor_inlet
    from mycelium.factor_masks import (
        native_head_alloc_for_present_types, head_alloc_to_tensors,
    )

    S = 49
    B = len(circuits)
    xor_gid = GLOBAL_TYPE_IDS["circuit_xor"]
    sentinel = int(spec.n_factor_types)
    L_max = int(getattr(model, "_zse_L_max", max(len(c["factors"]) for c in circuits)))
    L_max = max(L_max, max(len(c["factors"]) for c in circuits))

    input_np = np.zeros((B, S), dtype=np.int32)
    gold_np = np.zeros((B, S), dtype=np.int32)
    valid_np = np.zeros((B, S), dtype=np.float32)
    vdm_np = np.zeros((B, S, 7), dtype=np.float32)            # universal 7-wide codebook
    mem_np = np.zeros((B, L_max, S), dtype=np.float32)
    lt_np = np.full((B, L_max), sentinel, dtype=np.int32)

    for b, c in enumerate(circuits):
        nc = c["n_cells"]
        for j in range(nc):
            valid_np[b, j] = 1.0
            gold_np[b, j] = c["gold"][j] + 1     # Boolean 0/1 -> value 1/2
            input_np[b, j] = c["input_cells"][j]  # 0=unknown, 1/2 given
            vdm_np[b, j, 0] = 1.0                 # value 1 legal (Boolean false)
            vdm_np[b, j, 1] = 1.0                 # value 2 legal (Boolean true)
        for li, fac in enumerate(c["factors"]):
            lt_np[b, li] = xor_gid
            for cell in fac["members"]:
                mem_np[b, li, cell] = 1.0

    input_t = Tensor(input_np, dtype=dtypes.int).contiguous().realize()
    gold_t = Tensor(gold_np, dtype=dtypes.int).contiguous().realize()
    valid_t = Tensor(valid_np, dtype=dtypes.float).contiguous().realize()
    vdm_t = Tensor(vdm_np, dtype=dtypes.float).contiguous().realize()
    mem_t = Tensor(mem_np, dtype=dtypes.float).contiguous().realize()
    lt_t = Tensor(lt_np, dtype=dtypes.int).contiguous().realize()

    # op/target/size = 0 (circuit carries no arithmetic).
    zsem = Tensor(np.zeros((B, L_max), dtype=np.int32), dtype=dtypes.int).contiguous().realize()

    present = [xor_gid]
    hga = native_head_alloc_for_present_types(present, n_heads)
    head_type_oh, head_is_global = head_alloc_to_tensors(hga, spec.n_factor_types)
    head_type_oh = head_type_oh.realize()
    head_is_global = head_is_global.realize()

    inlet = build_generic_factor_inlet(
        model, mem_t, lt_t, valid_t, op=zsem, target=zsem, size=zsem).realize()

    return _EvalBatch(
        input_cells=input_t, cell_valid=valid_t, value_domain_mask=vdm_t,
        gold=gold_t, membership=mem_t, latent_type=lt_t, factor_inlet=inlet,
        head_type_oh=head_type_oh, head_is_global=head_is_global,
        recs=circuits, domain="xor_circuit")


class _EvalBatch:
    """Minimal batch object satisfying the FactorGraphBatch contract + the multi-task
    head-allocation attrs the engine routes on. Carries `recs` (the source puzzle
    records) for the exact verifier."""
    def __init__(self, input_cells, cell_valid, value_domain_mask, gold,
                 membership, latent_type, factor_inlet,
                 head_type_oh, head_is_global, recs, domain):
        self.input_cells = input_cells
        self.cell_valid = cell_valid
        self.value_domain_mask = value_domain_mask
        self.gold = gold
        self.membership = membership
        self.latent_type = latent_type
        self.factor_inlet = factor_inlet
        self.head_type_oh = head_type_oh
        self.head_is_global = head_is_global
        self.recs = recs
        self.domain = domain
        self.deduction_depth = [0] * int(input_cells.shape[0])


# ===========================================================================
# METRICS
# ===========================================================================

def _forward_predict(model, batch: _EvalBatch, spec, K: int):
    """Run the eager K-breath forward, return the final per-cell predictions (np int).

    Reuses factor_breathing_forward (the SAME forward the trainer's eval uses). The
    multi-task head-allocation attrs on `batch` route the engine to
    build_factor_attn_bias_multitask, exactly as in evaluate_multitask.
    """
    from tinygrad import Tensor
    from mycelium.factor_graph_engine import factor_breathing_forward
    Tensor.training = False
    logits_history, _ = factor_breathing_forward(model, batch, spec, K=K)
    final = logits_history[-1]
    pred = (final.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)   # (B, S)
    return pred


def _cell_puzzle_acc(pred: np.ndarray, batch: _EvalBatch):
    """cell_acc (over valid cells) + puzzle_acc (GOLD-match exact, all valid cells)."""
    gold = batch.gold.realize().numpy().astype(np.int32)
    valid = batch.cell_valid.realize().numpy() > 0.5
    cell_eq = 0
    n_cells = 0
    puzzle_eq = 0
    n_puzzles = 0
    B = pred.shape[0]
    for b in range(B):
        v = valid[b]
        nv = int(v.sum())
        if nv == 0:
            continue
        cell_eq += int((pred[b][v] == gold[b][v]).sum())
        n_cells += nv
        puzzle_eq += int(np.all(pred[b][v] == gold[b][v]))
        n_puzzles += 1
    return (cell_eq / max(n_cells, 1), puzzle_eq / max(n_puzzles, 1),
            n_cells, n_puzzles)


def _proper_solution_rate_kenken(pred: np.ndarray, batch: _EvalBatch):
    """EXACT-verifier proper-solution rate for KenKen (count_solutions/cage_ok), NOT a
    gold-match. Reshapes each puzzle's prediction to its N x N grid and runs
    verify_kenken_solution."""
    from mycelium.kenken_data import N_MAX
    n_proper = 0
    n_total = 0
    for b, rec in enumerate(batch.recs):
        n = int(rec["N"])
        grid = [[int(pred[b, r * N_MAX + col]) for col in range(n)] for r in range(n)]
        n_total += 1
        if verify_kenken_solution(rec, grid):
            n_proper += 1
    return n_proper / max(n_total, 1), n_total


def _xor_correct_rate(pred: np.ndarray, batch: _EvalBatch):
    """EXACT-verifier rate for the XOR control: a circuit is 'correct' iff EVERY gate
    cell's predicted value equals (parent1 XOR parent2) of the PREDICTED parents (i.e.
    the prediction is a self-consistent XOR assignment). Since inputs are given, this
    reduces to predicting every gate's XOR correctly."""
    n_proper = 0
    n_total = 0
    for b, c in enumerate(batch.recs):
        ok = True
        # predicted Boolean = value-1 (value 1->0, value 2->1).
        pv = [int(pred[b, j]) - 1 for j in range(c["n_cells"])]
        for fac in c["factors"]:
            g, p1, p2 = fac["members"]
            if pv[g] != (pv[p1] ^ pv[p2]):
                ok = False
                break
        n_total += 1
        if ok:
            n_proper += 1
    return n_proper / max(n_total, 1), n_total


# ===========================================================================
# MODEL LOAD (real path) — reuse the trainer's multi-task build + ckpt load.
# ===========================================================================

def _load_real_model(ckpt_path: str, K: int, train_path: str, test_path: str,
                     seed: int = 42):
    """Build the multi-task model EXACTLY as the trainer does, then load the ckpt.

    Returns (model, spec, n_heads, hidden, n_cages_max). Reuses scripts.factor_graph_train:
    the Pythia backbone, attach_factor_inlet_params, the unified spec, the fg params, and
    load_ckpt. GPU is used here (Device.DEFAULT); the CPU SELFTEST never calls this.
    """
    from tinygrad import Tensor, Device
    import scripts.factor_graph_train as fgt
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec, attach_factor_graph_params,
    )
    from mycelium.factor_inlet import attach_factor_inlet_params, N_GLOBAL_TYPES
    from mycelium.kenken_data import load_jsonl, N_MAX

    Tensor.manual_seed(seed)
    np.random.seed(seed)
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer ...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    fgt.cast_layers_fp32(model)
    hidden = cfg.hidden
    n_heads = cfg.n_heads

    # Unified multi-task spec (s_max=49, N=7, T=N_GLOBAL_TYPES=8, inlet on).
    spec = FactorGraphSpec(s_max=49, n_values=7, n_factor_types=N_GLOBAL_TYPES,
                           n_heads=n_heads, k_max=K, has_factor_inlet=True)

    # Generic inlet params FIRST (the build_generic_factor_inlet caller needs them).
    attach_factor_inlet_params(model, hidden=hidden)
    # The kenken verification-inlet tables are NOT needed (the generic inlet replaces
    # them in the multi-task path), but n_cages_max must cover the eval corpus.
    train_recs = load_jsonl(train_path)
    test_recs = load_jsonl(test_path)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    n_cages_max = corpus_n_cages_max
    # L_max for the multi-task topology = max membership width over the mix at train.
    # The fair checkpoint's measured_config records L_max=120; pin it so the eval batch
    # padding matches the trained width (the engine is L-agnostic, but keeping it stable
    # mirrors training). We attach it for the batch builders to read.
    model._zse_L_max = int(N_MAX + N_MAX + n_cages_max)

    attach_factor_graph_params(model, hidden=hidden, spec=spec)
    Device[Device.DEFAULT].synchronize()

    fgt.load_ckpt(model, ckpt_path)
    return model, spec, n_heads, hidden, n_cages_max


# ===========================================================================
# CPU SELFTEST — a MOCK tiny model + tiny ckpt that exercises the full path.
# ===========================================================================

def _build_mock_model(K: int, seed: int = 0):
    """Build a TINY model that satisfies every attribute the harness path touches:
    a 4-layer breathing block, the fg_* params, and the generic-inlet tables. No
    Pythia load, no GPU — tiny hidden so it runs in seconds on CPU.

    The mock exercises the FULL harness path (build batch -> build_generic_factor_inlet
    -> factor_breathing_forward -> readout -> metric) so a green selftest proves the
    plumbing, shapes, type-ids, head allocation, and verifier all line up. It does NOT
    test ACCURACY (a random tiny model is at chance) — accuracy is the real-ckpt job.
    """
    from tinygrad import Tensor
    from mycelium import Config, BreathingTransformer
    from mycelium.factor_graph_engine import FactorGraphSpec, attach_factor_graph_params
    from mycelium.factor_inlet import attach_factor_inlet_params, N_GLOBAL_TYPES
    from mycelium.kenken_data import N_MAX

    Tensor.manual_seed(seed)
    np.random.seed(seed)
    # Tiny config: small hidden, still 16 heads (head alloc needs >=4 layers / 16 heads
    # to mirror the real spec). Config() defaults to the Pythia dims; shrink hidden so
    # the mock is fast but keep n_heads=16 so head_type_oh shapes match the real path.
    cfg = Config()
    # Shrink for CPU speed where the Config allows. We keep the real n_heads (16) and
    # hidden divisible by it; use a small multiple of 16.
    try:
        cfg.hidden = 64
        cfg.n_heads = 16
        cfg.intermediate_size = 128
        if hasattr(cfg, "ffn_dim"):
            cfg.ffn_dim = 128
    except Exception:
        pass
    model = BreathingTransformer(cfg)
    hidden = cfg.hidden
    n_heads = cfg.n_heads

    spec = FactorGraphSpec(s_max=49, n_values=7, n_factor_types=N_GLOBAL_TYPES,
                           n_heads=n_heads, k_max=K, has_factor_inlet=True)
    attach_factor_inlet_params(model, hidden=hidden)
    attach_factor_graph_params(model, hidden=hidden, spec=spec)
    model._zse_L_max = int(N_MAX + N_MAX + 30)   # plenty for tiny N=4 boards
    return model, spec, n_heads, hidden


def _mock_ckpt_roundtrip(model, tmp_path: str):
    """Save the mock model's fg state dict and load it back via the trainer's load_ckpt,
    exercising the REAL ckpt save/load code path (model_state_dict_fg + load_ckpt) on a
    tiny checkpoint — so the selftest covers 'load' for real, just not the GPU ckpt."""
    import scripts.factor_graph_train as fgt
    from tinygrad.nn.state import safe_save
    sd = fgt.model_state_dict_fg(model)
    safe_save(sd, tmp_path)
    fgt.load_ckpt(model, tmp_path)
    return tmp_path


def run_selftest(K: int = 4) -> int:
    """GPU-free end-to-end selftest with a mock model. Returns process exit code."""
    import tempfile
    print("=" * 72)
    print("CPU SELFTEST — mock tiny model, no GPU, no real checkpoint")
    print("=" * 72)
    # Force tinygrad onto CPU (DEV=CPU is the current selector; older CPU=1 is deprecated).
    if "DEV" not in os.environ and not any(
            os.environ.get(d) == "1" for d in ("CPU", "AMD", "GPU", "CUDA", "METAL")):
        os.environ["DEV"] = "CPU"

    # 1) MOCK MODEL + MOCK CKPT ROUNDTRIP (the real save/load path).
    model, spec, n_heads, hidden = _build_mock_model(K=K)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "mock_fg.safetensors")
        _mock_ckpt_roundtrip(model, ckpt)
        print(f"[selftest] mock ckpt save+load OK ({ckpt})")

        # 2) FAIR FLAVOR — generate a tiny N=4 held-out KenKen set, run the full path.
        kk = generate_heldout_kenken(n=HELD_OUT_N, count=4, seed=1)
        assert len(kk) == 4, "N=4 generation failed in selftest"
        n_cages_max = max(len(r["cages"]) for r in kk)
        fair_batch = _make_kenken_eval_batch(kk, model, K, hidden, n_heads, spec,
                                             n_cages_max)
        fair_pred = _forward_predict(model, fair_batch, spec, K)
        f_cell, f_puz, f_nc, f_np = _cell_puzzle_acc(fair_pred, fair_batch)
        f_proper, _ = _proper_solution_rate_kenken(fair_pred, fair_batch)
        print(f"[selftest] FAIR (N=4 mock): cell_acc={f_cell:.3f} "
              f"puzzle_acc(gold)={f_puz:.3f} proper_rate(verifier)={f_proper:.3f} "
              f"n_cells={f_nc} n_puzzles={f_np}")

        # 3) NEGATIVE CONTROL — tiny XOR circuits, full path + exact XOR verifier.
        xc = generate_xor_circuits(count=4, seed=2, n_inputs=4, n_gates=6)
        ctrl_batch = _make_xor_eval_batch(xc, model, K, hidden, n_heads, spec)
        ctrl_pred = _forward_predict(model, ctrl_batch, spec, K)
        c_cell, c_puz, c_nc, c_np = _cell_puzzle_acc(ctrl_pred, ctrl_batch)
        c_proper, _ = _xor_correct_rate(ctrl_pred, ctrl_batch)
        print(f"[selftest] CONTROL (XOR mock): cell_acc={c_cell:.3f} "
              f"puzzle_acc(gold)={c_puz:.3f} xor_consistent_rate={c_proper:.3f} "
              f"n_cells={c_nc} n_circuits={c_np}")

        # 4) VERIFIER UNIT CHECK — a known-good N=4 grid must verify True; a corrupted
        #    one must verify False (proves the exact verifier is not a rubber stamp).
        rec0 = kk[0]
        n0 = int(rec0["N"])
        good = [[int(rec0["solution"][r][c]) for c in range(n0)] for r in range(n0)]
        assert verify_kenken_solution(rec0, good) is True, \
            "exact verifier rejected the gold solution"
        bad = [row[:] for row in good]
        bad[0][0] = (bad[0][0] % n0) + 1         # perturb one cell -> breaks Latin row
        assert verify_kenken_solution(rec0, bad) is False, \
            "exact verifier accepted a corrupted grid"
        print("[selftest] exact verifier unit check OK (gold->True, corrupted->False)")

        # 5) XOR verifier unit check.
        c0 = xc[0]
        good_pred = np.zeros((1, 49), dtype=np.int32)
        for j in range(c0["n_cells"]):
            good_pred[0, j] = c0["gold"][j] + 1
        gb = _EvalBatch(
            input_cells=ctrl_batch.input_cells, cell_valid=ctrl_batch.cell_valid,
            value_domain_mask=ctrl_batch.value_domain_mask, gold=ctrl_batch.gold,
            membership=ctrl_batch.membership, latent_type=ctrl_batch.latent_type,
            factor_inlet=ctrl_batch.factor_inlet, head_type_oh=ctrl_batch.head_type_oh,
            head_is_global=ctrl_batch.head_is_global, recs=[c0], domain="xor_circuit")
        r_good, _ = _xor_correct_rate(good_pred, gb)
        assert r_good == 1.0, "XOR verifier rejected the gold XOR assignment"
        print("[selftest] XOR verifier unit check OK (gold XOR assignment -> consistent)")

    print("\n[selftest] PASS — full harness path runs on CPU with a mock model.")
    print("           (accuracy is at chance: the mock is random; the REAL eval needs")
    print("            the trained checkpoint on a GPU — see RUN in the module docstring.)")
    return 0


# ===========================================================================
# REAL EVAL + VERDICT
# ===========================================================================

def run_real_eval(ckpt_path: str, K: int, train_path: str, test_path: str,
                  n_fair: int, n_ctrl: int, seed: int) -> int:
    """Load the real multi-task checkpoint and run both flavors + the in-distribution
    baseline, then print the verdict. GPU path."""
    from mycelium.kenken_data import load_jsonl, N_MAX

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("       (the fair checkpoint is still training; build was verified via")
        print("        --selftest. Re-run with --ckpt pointing at the final ckpt.)")
        return 2

    model, spec, n_heads, hidden, n_cages_max = _load_real_model(
        ckpt_path, K, train_path, test_path, seed=seed)

    print("\n" + "=" * 72)
    print("ZERO-SHOT GENERALIZATION EVAL (forward-only, NO retraining)")
    print(f"ckpt={ckpt_path}  K={K}  seed={seed}")
    print("=" * 72)

    # ---- (b) IN-DISTRIBUTION baseline: the trained KenKen test set (N in {5,6,7}). ----
    test_recs = load_jsonl(test_path)
    # Cap to a manageable count; the test set is the in-dist reference for "comparable".
    id_recs = test_recs[: min(len(test_recs), max(n_fair, 60))]
    id_batch = _make_kenken_eval_batch(id_recs, model, K, hidden, n_heads, spec,
                                       n_cages_max)
    id_pred = _forward_predict(model, id_batch, spec, K)
    id_cell, id_puz, _, id_np = _cell_puzzle_acc(id_pred, id_batch)
    id_proper, _ = _proper_solution_rate_kenken(id_pred, id_batch)
    id_chance = float(np.mean([1.0 / int(r["N"]) for r in id_recs]))
    print(f"\n[IN-DIST  N in {IN_DIST_NS}]  n={id_np}")
    print(f"    cell_acc={id_cell:.3f}  puzzle_acc(gold)={id_puz:.3f}  "
          f"proper_rate(exact verifier)={id_proper:.3f}")
    print(f"    chance(cell)~={id_chance:.3f} (mean 1/N)")

    # ---- (1) FAIR FLAVOR: held-out N=4 KenKen. ----
    fair_recs = generate_heldout_kenken(n=HELD_OUT_N, count=n_fair, seed=seed + 100)
    fair_ncm = max(len(r["cages"]) for r in fair_recs)
    fair_batch = _make_kenken_eval_batch(fair_recs, model, K, hidden, n_heads, spec,
                                         max(fair_ncm, n_cages_max))
    fair_pred = _forward_predict(model, fair_batch, spec, K)
    f_cell, f_puz, _, f_np = _cell_puzzle_acc(fair_pred, fair_batch)
    f_proper, _ = _proper_solution_rate_kenken(fair_pred, fair_batch)
    f_chance = 1.0 / HELD_OUT_N
    print(f"\n[FAIR  held-out KenKen N={HELD_OUT_N}]  n={f_np}  "
          f"(NEW PARAMS of a TRAINED relation — inlet/predicate-registry claim)")
    print(f"    cell_acc={f_cell:.3f}  puzzle_acc(gold)={f_puz:.3f}  "
          f"proper_rate(exact verifier)={f_proper:.3f}")
    print(f"    chance(cell)={f_chance:.3f} (1/N)   in-dist cell_acc={id_cell:.3f}")

    # ---- (2) NEGATIVE CONTROL: novel XOR relation. ----
    ctrl_circuits = generate_xor_circuits(count=n_ctrl, seed=seed + 200)
    ctrl_batch = _make_xor_eval_batch(ctrl_circuits, model, K, hidden, n_heads, spec)
    ctrl_pred = _forward_predict(model, ctrl_batch, spec, K)
    c_cell, c_puz, _, c_np = _cell_puzzle_acc(ctrl_pred, ctrl_batch)
    c_proper, _ = _xor_correct_rate(ctrl_pred, ctrl_batch)
    c_chance = 0.5                                # Boolean codebook
    print(f"\n[CONTROL  novel XOR relation]  n={c_np}  "
          f"(NEW RELATION TYPE — gate fg_inlet_gate[circuit_xor]~0, never trained)")
    print(f"    cell_acc={c_cell:.3f}  puzzle_acc(gold)={c_puz:.3f}  "
          f"xor_consistent_rate(exact verifier)={c_proper:.3f}")
    print(f"    chance(cell)={c_chance:.3f}")
    # Report the actual XOR gate value for transparency.
    try:
        gate = model.fg_inlet_gate.realize().numpy()
        from mycelium.factor_inlet import GLOBAL_TYPE_IDS
        xg = float(gate[GLOBAL_TYPE_IDS["circuit_xor"]])
        print(f"    (fg_inlet_gate[circuit_xor] = {xg:+.5f}  — ~0 confirms 'never opened')")
    except Exception:
        pass

    # ---- VERDICT ----
    fair_beats_chance = f_cell > (f_chance + CHANCE_MARGIN)
    fair_comparable = abs(f_cell - id_cell) <= FAIR_TOL and fair_beats_chance
    ctrl_fails = c_cell <= (c_chance + CHANCE_MARGIN)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  FAIR comparable-to-in-dist : {fair_comparable}  "
          f"(|{f_cell:.3f}-{id_cell:.3f}|={abs(f_cell-id_cell):.3f} <= {FAIR_TOL} "
          f"AND > chance+{CHANCE_MARGIN})")
    print(f"  CONTROL fails (~chance)    : {ctrl_fails}  "
          f"(cell={c_cell:.3f} <= chance {c_chance:.3f}+{CHANCE_MARGIN})")
    if fair_comparable and ctrl_fails:
        print("\n  ==> SEMANTICS-AS-INPUT GENERALIZES.")
        print("      New PARAMS of a TRAINED relation transfer zero-shot (the inlet's")
        print("      value); a NEW RELATION TYPE does not (it needs a retrain — the")
        print("      boundary). The general model is real, with a mapped limit.")
    elif fair_comparable and not ctrl_fails:
        print("\n  ==> AMBIGUOUS: fair transfers but control did NOT fail. Either the")
        print("      mask leaks the XOR answer structurally, or the registry transfers")
        print("      more than expected. Inspect the control before claiming the boundary.")
    elif not fair_comparable:
        print("\n  ==> SEMANTICS-AS-INPUT DID NOT GENERALIZE (fair flavor collapsed).")
        print("      Held-out params of a trained relation did not transfer; the model")
        print("      may be general CODE but not a general MODEL on this axis.")
    print("\n  HONEST FRAMING: the fair flavor tests new-PARAMS-of-trained-relations;")
    print("  the control confirms a new-RELATION-TYPE needs retrain. A failing control")
    print("  is the EXPECTED, informative result — NOT a defect.")
    return 0


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="multi-task checkpoint (fg_multi_fair_final.safetensors)")
    p.add_argument("--train", default=DEFAULT_TRAIN, help="kenken train jsonl (n_cages_max)")
    p.add_argument("--test", default=DEFAULT_TEST, help="kenken test jsonl (in-dist baseline)")
    p.add_argument("--K", type=int, default=DEFAULT_K, help="breaths (match the ckpt)")
    p.add_argument("--n-fair", type=int, default=120, help="held-out N=4 puzzles to eval")
    p.add_argument("--n-ctrl", type=int, default=120, help="XOR control circuits to eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="GPU-free CPU selftest with a mock model (no real ckpt)")
    args = p.parse_args(argv)

    if args.selftest:
        return run_selftest(K=min(args.K, 4))
    return run_real_eval(args.ckpt, args.K, args.train, args.test,
                         args.n_fair, args.n_ctrl, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
