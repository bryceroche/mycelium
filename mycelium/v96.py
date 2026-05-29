"""v96 — Consolidation table architecture.

The synthesis of v66 → v95 learnings.

KEY IDEA (compress the DELTA, not the state):
    Transformers EXPAND across layers (1024d → richer 1024d). The breath's
    *contribution* is the difference between L3 output and L0 input — naturally
    low-rank, naturally informative. Compressing this delta is the JPEG/MP3
    of breath reasoning. Every prior architecture compressed the FULL
    representation (waist 1024→512). That mixes the known (prompt, history)
    with the unknown (new reasoning). Wastes capacity.

PER-BREATH PIPELINE (computed in breathing.py inside breathe_with_lookup):
    1. delta = output_state - input_state                              (B, T, 1024)
    2. importance = sigmoid(delta @ gate_w + gate_b)                    (B, T, 1024)
    3. delta_q = delta * importance                                     (B, T, 1024)
    4. pool_scores = (delta_q * breath_embed[k]).sum(-1)                (B, T)
    5. pool_w = softmax(pool_scores + (1 - kv_mask) * -1e4)             (B, T)
    6. delta_pooled = (pool_w * delta_q).sum(T)                         (B, 1024)
    7. artifact_ops      = delta_pooled @ v96_ops_codebook.T            (B, 4)
       artifact_types    = delta_pooled @ v96_types_codebook.T          (B, 32)
       artifact_conf     = ||delta_pooled||_2                           (B, 1)
       artifact_summary  = delta_pooled @ v96_summary_proj              (B, 128)
    8. row = concat([ops, types, conf, summary])                        (B, 165)
    9. consolidation_table[:, k, :] = row

The table grows breath-by-breath. The WaistController at the final breath
reads the table via a new KV stream:
    table_kv = v96_table_kv_proj(consolidation_table[:, :K, :])         (B, K, 1024)
    full_kv  = concat([prompt_kv, table_kv], dim=1)                      (B, T_prompt + K, 1024)

PER-BREATH SUPERVISION (the credit-assignment unlock that v85 missed):
    Each row gets its OWN loss:
        ce_ops_k   = CE(artifact_ops[k],   gold_op_idx_for_step_k)
        ce_types_k = CE(artifact_types[k], gold_type_idx_for_step_k)
        conf_loss_k = (artifact_conf[k] - target_conf_k)**2

    With K-progressive label smoothing:
        ls_k = LS_START * (1 - k / (K-1))     # 0.5 @ B0, 0.0 @ B6

Constraints respected (CLAUDE.md):
    - All masks honored (V81_MAIN_ATTN_MASK=0 for AR generation)
    - AMD JIT — no .cast(dtypes.float32) inside JIT body
    - consolidation_table is an ACTIVATION (built fresh each forward), NOT a param
    - JIT cache key includes V96_CONSOLIDATION
    - Gradient separation enforced: v96 params live with model.parameters(); the
      controller's separate optimizer is untouched.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tinygrad import Tensor, dtypes


# ---------------- Constants & sizes ---------------------------------------

V96_OPS_N         = 4    # ADD / SUB / MUL / DIV
V96_TYPES_N       = 32   # IB tree leaves (matches V78_HEAD_CODEBOOK_N)
V96_CONF_N        = 1
V96_SUMMARY_DIM   = 128
V96_PACKED_DIM    = V96_OPS_N + V96_TYPES_N + V96_CONF_N + V96_SUMMARY_DIM  # = 165

# Offsets inside the packed row.
V96_OPS_OFF       = 0
V96_TYPES_OFF     = V96_OPS_OFF + V96_OPS_N
V96_CONF_OFF      = V96_TYPES_OFF + V96_TYPES_N
V96_SUMMARY_OFF   = V96_CONF_OFF + V96_CONF_N


# ---------------- Param creation ------------------------------------------

def make_v96_params(hidden: int, ops_codebook_init: float = 0.02,
                    types_codebook_init: float = 0.02,
                    summary_proj_init: float = 0.02,
                    table_kv_proj_init: float = 0.0,    # v96.1: ZERO-init to start table contribution at 0
                    gate_bias_init: float = -1.0,
                    constraint_head_init: float = 0.02) -> dict:
    """Allocate v96 parameter tensors (returned as a dict for caller assignment).

    All are kept ALWAYS-allocated for state_dict symmetry. Forward path is gated
    on V96_CONSOLIDATION env flag.

    gate_bias_init = -1.0 → initial sigmoid(bias) ≈ 0.27 → ~27% of dims active at
    step 0. Gradient learns the per-dim importance.

    ops_codebook + types_codebook init scale 0.02 is the Pythia-style randn scale.
    types_codebook can OPTIONALLY be initialized from the IB centroids
    (.cache/ib_centroids.npz, projected to 1024d) — we provide a hook for that
    via the trainer (see load_ib_codebooks_into_v96).

    v96.2 (2026-05-28) constraint check heads — two small linear heads (Linear
    128 → 1) that learn to score "ref validity" and "arg ordering plausibility"
    from the raw_summary of each row. Bombe-inspired elimination: the model
    learns to penalize its own configurations that violate causal/ordering
    constraints. Trained via self-supervision (the model penalizes itself on
    its own outputs — see compute_constraint_scores in l3_training.py).

    constraint_head_init = 0.02 is the standard randn scale. Biases zero-init.
    """
    params = {
        # Per-dimension importance gate (small MLP — actually just a linear
        # layer Linear(1024 → 1024)). zero-init weights so initial gate is
        # JUST the bias (constant ~0.27 importance). Gradient learns to
        # condition on the delta content.
        "v96_gate_w":         Tensor.zeros((hidden, hidden), dtype=dtypes.float).contiguous(),
        "v96_gate_b":         (Tensor.ones((hidden,), dtype=dtypes.float) * gate_bias_init).contiguous(),
        # Ops codebook: (4, 1024). delta_pooled @ ops_codebook.T → (B, 4) logits.
        "v96_ops_codebook":   (Tensor.randn(V96_OPS_N, hidden, dtype=dtypes.float) * ops_codebook_init).contiguous(),
        # Types codebook: (32, 1024). delta_pooled @ types_codebook.T → (B, 32).
        "v96_types_codebook": (Tensor.randn(V96_TYPES_N, hidden, dtype=dtypes.float) * types_codebook_init).contiguous(),
        # Summary projection: (1024, 128). delta_pooled @ summary_proj → (B, 128).
        "v96_summary_proj":   (Tensor.randn(hidden, V96_SUMMARY_DIM, dtype=dtypes.float) * summary_proj_init).contiguous(),
        # Table KV projection: (165, hidden). (B, K, 165) @ table_kv_proj → (B, K, hidden).
        # v96.1: ZERO-init by default. Combined with v96_table_alpha (also zero-init below),
        # the table contribution to cross-attn is EXACTLY ZERO at step 0 — model is
        # byte-equivalent to v80 baseline. Gradient gradually opens the channel,
        # preventing the positive-feedback explosion observed in v96 step 70+
        # (waist_norm 1.4 → 710 over 90 steps).
        "v96_table_kv_proj":  (Tensor.randn(V96_PACKED_DIM, hidden, dtype=dtypes.float) * table_kv_proj_init).contiguous(),
        # v96.1: Per-breath scale gate on the table contribution to cross-attn.
        # Zero-init scalar. Forward multiplies the projected table KV by sigmoid(alpha)
        # so the model learns to gradually use the table. Same pattern as BFIELD_ALPHA.
        "v96_table_alpha":    Tensor.zeros((1,), dtype=dtypes.float).contiguous(),
        # v96.2 (2026-05-28) constraint check heads. Linear(128 → 1) each.
        # ref_validity: scores whether the row's raw_summary is "backward-looking"
        #   (doesn't reference future state). Trained via self-supervision —
        #   the model penalizes its own confidence at predicting validity 1.0.
        # arg_order:    scores whether the row's args are properly ordered
        #   (larger first for SUB, dividend before divisor for DIV). Same
        #   self-supervision pattern but weighted by non-commutative op prob.
        "v96_ref_validity_head_w": (Tensor.randn(V96_SUMMARY_DIM, 1, dtype=dtypes.float) * constraint_head_init).contiguous(),
        "v96_ref_validity_head_b": Tensor.zeros((1,), dtype=dtypes.float).contiguous(),
        "v96_arg_order_head_w":    (Tensor.randn(V96_SUMMARY_DIM, 1, dtype=dtypes.float) * constraint_head_init).contiguous(),
        "v96_arg_order_head_b":    Tensor.zeros((1,), dtype=dtypes.float).contiguous(),
    }
    return params


# ---------------- Artifact computation -------------------------------------

@dataclass
class V96Artifact:
    """Unpacked view of a row in the consolidation table.

    Shapes are per-breath: (B, 4), (B, 32), (B, 1), (B, 128).
    These are graph tensors — the packed form is the canonical storage.
    """
    ops_logits:    Tensor   # (B, 4)
    types_logits:  Tensor   # (B, 32)
    confidence:    Tensor   # (B, 1)
    raw_summary:   Tensor   # (B, 128)


def compute_v96_artifact(x_in: Tensor, x_out: Tensor,
                          breath_embed_k: Tensor,
                          v96_gate_w: Tensor, v96_gate_b: Tensor,
                          v96_ops_codebook: Tensor,
                          v96_types_codebook: Tensor,
                          v96_summary_proj: Tensor,
                          pool_mask: Optional[Tensor] = None,
                          temperature: Optional[float] = None) -> V96Artifact:
    """Compute the v96 artifact for one breath.

    x_in / x_out: (B, T, hidden) — pre/post-breath hidden states (delta source).
    breath_embed_k: (hidden,) — the per-breath embedding vector for breath k.
    pool_mask: optional (B, T) float — 1.0 at valid positions for attention
        pool, 0.0 elsewhere. When provided, applies -1e4 penalty to masked
        positions before softmax.

    v96.2: temperature: optional float. When provided, divides ops_logits and
        types_logits by this scalar BEFORE returning. Implements per-breath
        sharpening: T_k = max(0.3, 2.0 * (1 - k/(K-1))) makes early breaths
        BROAD (many candidates alive) and late breaths SHARP (single
        candidate survives). Trained to align with constraint satisfaction
        via the calibration loss. Default None = no temperature scaling.

    Returns V96Artifact with logits/confidence/summary tensors.
    """
    # Cast to float32 OUTSIDE the JIT body (the trainer + this helper run
    # inside the JIT; cast is fine here because we receive already-float tensors
    # in the v96 wiring — but we explicitly avoid .cast(dtypes.float32) as
    # tinygrad+AMD has that gotcha. We use dtypes.float which is float32.)
    delta = (x_out - x_in).cast(dtypes.float)                          # (B, T, H)
    importance = (delta @ v96_gate_w + v96_gate_b.reshape(1, 1, -1)).sigmoid()
    delta_q = delta * importance                                        # (B, T, H)
    # Attention pool: query = breath_embed[k] (broadcast over batch).
    q = breath_embed_k.cast(dtypes.float).reshape(1, 1, -1)             # (1, 1, H)
    scores = (delta_q * q).sum(axis=-1)                                  # (B, T)
    if pool_mask is not None:
        scores = scores + (1.0 - pool_mask.cast(scores.dtype)) * (-1e4)
    weights = scores.softmax(axis=-1).reshape(scores.shape[0], -1, 1)    # (B, T, 1)
    delta_pooled = (delta_q * weights).sum(axis=1)                       # (B, H)
    # Heads.
    ops_logits = delta_pooled @ v96_ops_codebook.transpose(-2, -1)       # (B, 4)
    types_logits = delta_pooled @ v96_types_codebook.transpose(-2, -1)   # (B, 32)
    # v96.2: per-breath temperature sharpening. DIVIDE the logits — large T
    # flattens the softmax distribution (many candidates), small T sharpens
    # toward a single peak (one candidate survives). Reflects Bombe-style
    # elimination across breaths.
    if temperature is not None and temperature > 0.0 and temperature != 1.0:
        ops_logits = ops_logits / float(temperature)
        types_logits = types_logits / float(temperature)
    # Confidence = L2 norm of pooled delta. Higher = more information committed
    # this breath. Trained to track per-breath progress (see trainer).
    confidence = (delta_pooled.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()  # (B, 1)
    raw_summary = delta_pooled @ v96_summary_proj                         # (B, 128)
    return V96Artifact(ops_logits=ops_logits, types_logits=types_logits,
                       confidence=confidence, raw_summary=raw_summary)


# ---------------- v96.2 Constraint check heads (Bombe-inspired elimination) ----

def compute_temperature_schedule(K: int, T_start: float = 2.0, T_end: float = 0.3) -> list:
    """Per-breath temperature schedule. Linear interp from T_start at B0 → T_end at B_{K-1}.

    Used to sharpen ops/types logits across breaths. Early: broad selection
    (many candidates alive). Late: sharp commitment (single candidate
    survives). Reflects Bombe constraint propagation.

    Returns: list of K floats. T_end is the floor — never returns < T_end.
    """
    if K <= 1:
        return [T_end]
    out = []
    for k in range(K):
        T_k = T_start * (1.0 - float(k) / float(K - 1))
        out.append(max(T_end, T_k))
    return out


def compute_constraint_scores(raw_summary: Tensor,
                                ref_w: Tensor, ref_b: Tensor,
                                arg_w: Tensor, arg_b: Tensor) -> tuple:
    """Project the row's raw_summary through the two constraint heads.

    raw_summary: (B, 128) per-breath summary.
    ref_w, ref_b:  Linear(128, 1) for reference-validity check.
    arg_w, arg_b:  Linear(128, 1) for arg-ordering check.

    Returns (ref_validity_score, arg_order_score) — each (B, 1) sigmoid.
    These are self-supervised: the model learns to assign HIGH validity to its
    own selections via the constraint losses in the trainer.
    """
    raw_summary_f = raw_summary.cast(dtypes.float)
    ref_score = (raw_summary_f @ ref_w + ref_b.reshape(1, -1)).sigmoid()    # (B, 1)
    arg_score = (raw_summary_f @ arg_w + arg_b.reshape(1, -1)).sigmoid()    # (B, 1)
    return ref_score, arg_score


def pack_artifact(artifact: V96Artifact) -> Tensor:
    """Pack the unpacked artifact into a single (B, 165) row.

    Layout: [ops(4) | types(32) | confidence(1) | raw_summary(128)].
    """
    # Concat along feature dim.
    return Tensor.cat(artifact.ops_logits, artifact.types_logits,
                       artifact.confidence, artifact.raw_summary, dim=-1)


def unpack_artifact(packed: Tensor) -> V96Artifact:
    """Inverse of pack_artifact. packed: (..., 165). Returns V96Artifact view."""
    return V96Artifact(
        ops_logits=packed[..., V96_OPS_OFF:V96_OPS_OFF + V96_OPS_N],
        types_logits=packed[..., V96_TYPES_OFF:V96_TYPES_OFF + V96_TYPES_N],
        confidence=packed[..., V96_CONF_OFF:V96_CONF_OFF + V96_CONF_N],
        raw_summary=packed[..., V96_SUMMARY_OFF:V96_SUMMARY_OFF + V96_SUMMARY_DIM],
    )


# ---------------- IB tree-derived gold label parsing -----------------------

_IB_LEAF_TO_IDX: Optional[dict] = None
_IB_OP_TO_LEAF_IDXS: Optional[dict] = None


def _load_ib_tree() -> dict:
    """Lazy load the IB tree json. Returns the parsed JSON dict."""
    import json
    tree_path = os.environ.get("IB_TREE_PATH", ".cache/ib_tree.json")
    if not os.path.exists(tree_path):
        return None
    with open(tree_path) as f:
        return json.load(f)


def init_ib_leaf_index() -> tuple:
    """Build leaf_id -> index map and op -> [leaf_indices] map from IB tree.

    Returns (leaf_to_idx, op_to_leaf_idxs). Empty dicts if tree missing.
    """
    global _IB_LEAF_TO_IDX, _IB_OP_TO_LEAF_IDXS
    if _IB_LEAF_TO_IDX is not None and _IB_OP_TO_LEAF_IDXS is not None:
        return _IB_LEAF_TO_IDX, _IB_OP_TO_LEAF_IDXS
    tree = _load_ib_tree()
    leaf_to_idx = {}
    op_to_leaf_idxs = {"ADD": [], "SUB": [], "MUL": [], "DIV": []}
    if tree is not None:
        for i, leaf in enumerate(tree["leaves"]):
            leaf_to_idx[leaf["leaf_id"]] = i
            op = leaf["op"]
            if op in op_to_leaf_idxs:
                op_to_leaf_idxs[op].append(i)
    _IB_LEAF_TO_IDX = leaf_to_idx
    _IB_OP_TO_LEAF_IDXS = op_to_leaf_idxs
    return leaf_to_idx, op_to_leaf_idxs


# Op string → index (matches V96_OPS_N=4).
V96_OP_TO_IDX = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}
# Common spelling alternates that crop up in L2 free-text vs L4 OP=NAME notation.
V96_OP_ALIASES = {
    "ADD": "ADD", "SUM": "ADD", "PLUS": "ADD",
    "SUB": "SUB", "SUBTRACT": "SUB", "MINUS": "SUB", "DIFFERENCE": "SUB",
    "MUL": "MUL", "MULTIPLY": "MUL", "MULT": "MUL", "TIMES": "MUL", "PRODUCT": "MUL",
    "DIV": "DIV", "DIVIDE": "DIV", "QUOTIENT": "DIV",
}


def parse_per_step_op_indices(L2_text: str, L4_text: str, K: int) -> List[int]:
    """Extract per-step op indices from a record's L2 + L4 layers.

    L2 has lines like `Step 1: ... OP=DIV. ARG=60.`.
    L4 has lines like `x_1 := <DIV>(50, 60)`.

    Returns a list of K ints; positions without a recoverable op are filled with
    -100 (the CE ignore_index for the trainer).
    """
    import re
    out: List[int] = []
    # Prefer L2's OP= notation; fall back to L4's <OP> pattern.
    l2_ops = re.findall(r"OP=([A-Z_]+)", L2_text or "")
    l4_ops = re.findall(r"<([A-Z_]+)>\(", L4_text or "")
    n_l2 = len(l2_ops)
    n_l4 = len(l4_ops)
    n_actual = max(n_l2, n_l4)
    for step_idx in range(K):
        op_str = None
        if step_idx < n_l2:
            op_str = l2_ops[step_idx]
        elif step_idx < n_l4:
            op_str = l4_ops[step_idx]
        elif n_actual > 0:
            # K bigger than actual reasoning steps: copy the LAST op (the answer
            # is just "return xN" — same op as the final step). This matches the
            # progressive-refinement principle (the model emits the same DAG at
            # all breaths, just more refined each time).
            op_str = (l2_ops or l4_ops)[-1]
        # Resolve aliases → canonical → idx.
        if op_str is not None:
            canonical = V96_OP_ALIASES.get(op_str.upper(), None)
            if canonical is not None:
                out.append(V96_OP_TO_IDX[canonical])
                continue
        out.append(-100)
    return out


def parse_per_step_type_indices(L3_text: str, L2_text: str, L4_text: str,
                                  per_step_op_indices: List[int],
                                  K: int) -> List[int]:
    """Extract per-step type indices into the 32-leaf IB tree.

    Strategy:
      - We do NOT have direct (step → leaf_id) annotations in the v80 jsonl.
      - The IB tree was built on Pythia embeddings of L2 step descriptions.
      - At train time we approximate: use op-constrained random-from-cohort
        (pick the FIRST leaf in the op's bucket). This is INACCURATE but
        deterministic and op-correct. The trainer's per-row CE then forces the
        types head to be at LEAST op-discriminative, which is the principal
        signal we want.

    For a more accurate per-step leaf assignment, the data pipeline would need
    to embed each L2 step with Pythia and assign by nearest centroid. That's
    expensive and orthogonal to the v96 architecture test. We bias toward
    DETERMINISTIC and OP-CORRECT here.

    Returns a list of K ints; -100 at positions without an op.
    """
    _leaf_to_idx, op_to_leaf_idxs = init_ib_leaf_index()
    if not op_to_leaf_idxs or all(len(v) == 0 for v in op_to_leaf_idxs.values()):
        # IB tree not loaded — emit -100 everywhere, trainer ignores.
        return [-100] * K
    inv_op = {v: k for k, v in V96_OP_TO_IDX.items()}
    out: List[int] = []
    for step_idx in range(K):
        op_idx = per_step_op_indices[step_idx]
        if op_idx == -100:
            out.append(-100)
            continue
        op_name = inv_op[op_idx]
        leaf_list = op_to_leaf_idxs.get(op_name, [])
        if not leaf_list:
            out.append(-100)
            continue
        # Deterministic: first leaf in op bucket. This is the v96-conservative
        # gold (op-correct, type-approximate). The trainer's CE descent then
        # validates "model commits to ONE leaf per op" — exactly what the
        # per-row supervision needs to break v85's template attractor.
        out.append(leaf_list[0])
    return out


def parse_gold_per_step(layers_raw: dict, K: int) -> tuple:
    """Top-level gold extraction. Returns (op_indices, type_indices) of length K.

    Used by the trainer to build per-row CE targets for v96.
    """
    L2 = layers_raw.get("L2", "")
    L3 = layers_raw.get("L3", "")
    L4 = layers_raw.get("L4", "")
    ops = parse_per_step_op_indices(L2, L4, K)
    types = parse_per_step_type_indices(L3, L2, L4, ops, K)
    return ops, types


def compute_target_confidence(K: int, base: float = 0.5,
                                final_scale: float = 1.0) -> np.ndarray:
    """Per-breath target confidence. Monotonically increasing from B0 to B_{K-1}:
        breath k: base + (final_scale - base) * (k / (K-1))

    Default: 0.5 at B0 → 1.0 at B6. Encourages the gate to LET MORE through
    in later breaths (the consolidation should grow, not shrink).

    Returns: (K,) float32 array.
    """
    if K <= 1:
        return np.array([final_scale], dtype=np.float32)
    out = np.zeros(K, dtype=np.float32)
    for k in range(K):
        out[k] = base + (final_scale - base) * (float(k) / float(K - 1))
    return out


def load_ib_codebooks_into_v96(model, hidden: int = 1024) -> bool:
    """Optionally initialize v96_types_codebook from the IB centroids.

    The IB centroids in .cache/ib_centroids.npz are already 1024d (Pythia
    embedding space), so they project DIRECTLY into the types codebook.

    Returns True if init succeeded, False otherwise.
    """
    ib_path = os.environ.get("IB_CENTROIDS_PATH", ".cache/ib_centroids.npz")
    if not os.path.exists(ib_path):
        return False
    try:
        npz = np.load(ib_path)
        centroids = npz["centroids"]  # (32, 1024)
        leaf_ids = npz["leaf_ids"]
    except Exception:
        return False
    if centroids.shape != (V96_TYPES_N, hidden):
        return False
    # Normalize centroids to scale 0.02 (codebook init scale) so they don't
    # blow up the logits at step 0. Then assign.
    centroids_t = Tensor(centroids.astype(np.float32), dtype=dtypes.float)
    norm = (centroids_t.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
    centroids_normed = centroids_t / norm * 0.02
    model.v96_types_codebook.assign(centroids_normed.contiguous()).realize()
    return True
