"""Training step + accuracy eval for math curriculum (L3, L4, L4.5).

Single-cycle path (L3): standard masked-loss CE, one forward pass per step.
Multi-cycle path (L4+): per-cycle forward passes — each outer cycle is its own
breathe-then-speak event. Equal-weighted loss across cycles (equal-reward
decomposition for teacher-forced training).
"""
import os
from typing import List, Tuple
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.l3_data import (
    MathExample, encode_example, encode_cycles, parse_int_answer, collate, SEP,
)
# v77 (2026-05-24): DAG-layered supervision. Each breath b is supervised against
# the b-th progressively-refined layer (paraphrase -> ... -> pure DAG). Imported
# lazily inside the V77 code path to avoid forcing V77 imports on legacy code.
try:
    from mycelium.l3_data import V77Example  # type: ignore
except ImportError:
    V77Example = None  # type: ignore
from mycelium.lookup_table import op_label_from_text, find_eq_position, eq_token_ids_for
from mycelium.calibration import (
    find_all_eq_positions, extract_digit_runs_after_eq, digit_token_ids_for,
)


# Label smoothing on the main answer-CE. Applied to training-time CE paths;
# eval CE (multi_cycle_eval_loss) gates on Tensor.training so reported eval
# loss stays comparable across LABEL_SMOOTHING values.
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.0"))

# v75 (2026-05-23) Multi-well supervision (diffusion paradigm).
# When PER_BREATH_FULL_ANSWER=1, every breath decodes the FULL structured answer
# (the union of all per-step token ranges) instead of step-k-specific tokens.
# Each breath gets the SAME CE target. The decomposition into K steps lives in
# the output format, not in the supervision schedule. Combined with the bounded
# per-breath delta (MAX_STEP_SIZE in breathing.py), this is the v75-conservative
# variant: each breath refines toward the same target; earlier breaths rougher,
# later breaths sharper.
PER_BREATH_FULL_ANSWER = int(os.environ.get("PER_BREATH_FULL_ANSWER", "0")) > 0
if PER_BREATH_FULL_ANSWER:
    print(f"[PER_BREATH_FULL_ANSWER] v75 multi-well supervision: every breath decodes the FULL answer (union of all step ranges)", flush=True)

# v77 (2026-05-24) DAG-layered supervision: each breath b decodes a progressively-
# refined layer-b target (paraphrase → ... → pure DAG). N_BREATHS is FIXED at
# the layer count for all problems. Default 0 preserves v66/v75 behavior.
V77_DAG_TRAINING = int(os.environ.get("V77_DAG_TRAINING", "0")) > 0
V77_N_LAYERS = int(os.environ.get("V77_N_LAYERS", "6"))
if V77_DAG_TRAINING:
    print(f"[V77_DAG_TRAINING] v77 DAG-layered supervision ON: N_BREATHS={V77_N_LAYERS} for all problems; "
          f"breath b decodes layer-b target.", flush=True)

# v78b (2026-05-25) Attention supervision aux loss. For each digit token in the
# gold L6 DAG target, supervise the WaistController's LAST cross-attn layer to
# peak at matching digit positions in the prompt. Tells the model WHERE to look
# for operands — addresses v77's wrong-operand failure ("x0 = 2 + 2" vs
# "x0 = 2 + 1"). WAIST_ATTN_SUPERVISION lives in breathing.py (gates the stash);
# this weight scales the aux loss in BOTH JIT entries.
WAIST_ATTN_AUX_WEIGHT = float(os.environ.get("WAIST_ATTN_AUX_WEIGHT", "0.5"))

# Module-level cache of digit token IDs. Computed lazily on first use to avoid
# forcing tokenizer initialization at import time.
_DIGIT_TOKEN_IDS: set | None = None


def _get_digit_token_ids(tok) -> set:
    """Lazy cache of single-digit token IDs. Used by attention-supervision aux loss
    to mask positions whose gold label is a digit (the only positions where prompt
    operand matches are well-defined)."""
    global _DIGIT_TOKEN_IDS
    if _DIGIT_TOKEN_IDS is None:
        _DIGIT_TOKEN_IDS = digit_token_ids_for(tok)
        print(f"[WAIST_ATTN_SUPERVISION] digit_token_ids cached: {len(_DIGIT_TOKEN_IDS)} ids", flush=True)
    return _DIGIT_TOKEN_IDS

# JIT-train cache: (id(model), id(opt), n_loops_tuple, fixed_len, B) → compiled fn
# The canonical tinygrad pattern wraps the WHOLE step (forward + backward + opt.step)
# in one TinyJit. Outputs are scalar Tensor losses; .numpy() returns the value.
_JIT_TRAIN_CACHE: dict = {}
# Separate cache for the calibration JIT (different signature & key).
_JIT_CALIB_CACHE: dict = {}
# Separate cache for the per-breath JIT (K-breath supervision path).
_JIT_PER_BREATH_CACHE: dict = {}


# v73: token-level rename augmentation. Forces v72's copy mechanism to learn
# "copy is required when vocab fails" by replacing content tokens with rare-vocab
# IDs in 50% of training examples. Without this, training data uses common names
# (Janet/Sam) that Pythia already knows, so p_vocab >> p_copy and the gate never
# bends.
WAIST_COPY_RENAME_AUG = int(os.environ.get("WAIST_COPY_RENAME_AUG", "0")) > 0
WAIST_COPY_RENAME_RATE = float(os.environ.get("WAIST_COPY_RENAME_RATE", "0.5"))
WAIST_COPY_RENAME_NUM_TOKENS = int(os.environ.get("WAIST_COPY_RENAME_NUM_TOKENS", "3"))
# Rare-vocab IDs: high IDs in Pythia's 50304 vocab are low-frequency. Built lazily
# from the tokenizer on first use (cached in the module).
_RENAME_CONTENT_VOCAB: set | None = None
_RENAME_RARE_IDS: list | None = None
_RENAME_FIRST_BATCH_LOGGED = False


def _build_rename_vocabs(tok):
    """Build the content-vocab set and rare-ID list once per process.

    Content tokens are heuristically "entity-like": decoded form starts with a
    space marker ('Ġ' in Pythia BPE) + a letter, with token length > 3. Rare
    IDs are the intersection of [40000, 50000) with the content vocab — high IDs
    that Pythia barely knows.
    """
    global _RENAME_CONTENT_VOCAB, _RENAME_RARE_IDS
    if _RENAME_CONTENT_VOCAB is not None:
        return _RENAME_CONTENT_VOCAB, _RENAME_RARE_IDS
    vocab_size = tok.get_vocab_size()
    content = set()
    for tid in range(vocab_size):
        s = tok.id_to_token(tid)
        if s is None:
            continue
        # Pythia BPE uses 'Ġ' as the space-prefix marker (codepoint 0x120).
        if len(s) > 3 and s[0] == 'Ġ' and len(s) > 1 and s[1].isalpha():
            content.add(tid)
    rare = [tid for tid in range(40000, min(50000, vocab_size)) if tid in content]
    _RENAME_CONTENT_VOCAB = content
    _RENAME_RARE_IDS = rare
    print(f"[v73 rename-aug] content vocab size: {len(content)}, rare IDs: {len(rare)}", flush=True)
    return content, rare


def _apply_rename_aug(tokens_np: np.ndarray, per_step_labels_np: np.ndarray,
                      tok, rate: float, num_tokens: int):
    """v73 rename augmentation. Replaces content tokens with rare IDs in-place.

    Args:
        tokens_np: (B, T) int32 — prompt+answer token ids, padded.
        per_step_labels_np: (K, B, T-1) int32 — per-step CE labels (-100 outside
            step k's positions; otherwise shifted token ids).
        tok: HuggingFace tokenizer (used to build content vocab once).
        rate: probability of applying augmentation to each example (default 0.5).
        num_tokens: target number of distinct content tokens to rename per example.

    For each example b independently, with probability rate:
      - Find content tokens present in tokens_np[b].
      - Pick min(num_tokens, n_content) distinct ones uniformly at random.
      - For each, pick a random rare ID (also distinct from other picks for this
        example) and replace ALL occurrences in tokens_np[b] AND per_step_labels_np[:, b, :].
    If an example has fewer than 2 content tokens, skip it.

    Returns: list of per-example summaries (orig_ids, new_ids, n_replacements_total).
    """
    content_set, rare_ids = _build_rename_vocabs(tok)
    if not rare_ids:
        return []
    B, T = tokens_np.shape
    K = per_step_labels_np.shape[0]
    summaries = []
    rng = np.random
    for b in range(B):
        if rng.random() >= rate:
            summaries.append(None)
            continue
        # Find content tokens present in this example's prompt.
        ex_ids = tokens_np[b]
        present_content = sorted({int(t) for t in ex_ids if int(t) in content_set})
        if len(present_content) < 2:
            summaries.append(None)
            continue
        # Pick up to num_tokens distinct content tokens; assign distinct rare IDs.
        k_pick = min(num_tokens, len(present_content))
        picked = list(rng.choice(present_content, size=k_pick, replace=False))
        # Sample distinct rare IDs (without replacement, restricted to ones not
        # already in the prompt to avoid accidental collision).
        rare_pool = [r for r in rare_ids if r not in present_content]
        if len(rare_pool) < k_pick:
            rare_pool = rare_ids  # fall back to full pool if dataset is huge
        new_ids = list(rng.choice(rare_pool, size=k_pick, replace=False))
        total_repl = 0
        for orig, new in zip(picked, new_ids):
            mask_tokens = (tokens_np[b] == orig)
            n_tok = int(mask_tokens.sum())
            if n_tok == 0:
                continue
            tokens_np[b, mask_tokens] = new
            # Replace in labels for ALL K steps. Labels are shifted-by-one
            # of the same ids, so this keeps them consistent with tokens_np.
            mask_labels = (per_step_labels_np[:, b, :] == orig)
            n_lbl = int(mask_labels.sum())
            per_step_labels_np[:, b, :][mask_labels] = int(new)
            total_repl += n_tok + n_lbl
        summaries.append({
            "orig_ids": [int(x) for x in picked],
            "new_ids": [int(x) for x in new_ids],
            "n_replacements": total_repl,
        })
    return summaries


def _compile_jit_train_step(model, opt, n_loops_per_cycle: Tuple[int, ...],
                            fixed_len: int, B: int, lookup_aux_weight: float):
    """Compile and cache a JIT'd train step for the given (n_loops_per_cycle, B,
    fixed_len, aux_weight) tuple.

    Inputs to the returned function (all Tensors with stable shapes):
      n_cycles == 1: (tokens0, labels0, eq_mask, op_labels)
      n_cycles == 2: (tokens0, labels0, tokens1, labels1, eq_mask, op_labels)
    Returns: scalar loss Tensor (already .realize()'d).
    """
    key = (id(model), id(opt), tuple(n_loops_per_cycle), fixed_len, B, float(lookup_aux_weight))
    if key in _JIT_TRAIN_CACHE:
        return _JIT_TRAIN_CACHE[key]

    n_cycles = len(n_loops_per_cycle)
    aw = float(lookup_aux_weight)
    import time as _t_jit
    _jit_compile_start = _t_jit.perf_counter()
    print(f"[JIT] compile train step: n_loops={n_loops_per_cycle} B={B} fixed_len={fixed_len}...", flush=True)

    if n_cycles == 1:
        nl0 = int(n_loops_per_cycle[0])
        # v39 waist aux supervision: capture compressed at end-of-breath waist,
        # classify op from it via waist_head. Only active when BFIELD_AUX_WEIGHT > 0
        # and the end-of-breath waist is enabled.
        from mycelium.breathing import BFIELD_AUX_WEIGHT as _BAW, BFIELD_END_OF_BREATH as _BEOB, BFIELD_WAIST as _BFW
        use_waist_aux = (_BAW > 0.0) and _BEOB and (_BFW > 0)
        waw_local = float(_BAW)

        @TinyJit
        def _step(tokens0, labels0, eq_mask, op_labels):
            opt.zero_grad()
            if use_waist_aux:
                final_h, match_weights, _, waist_compressed_per_breath = model.breathe_with_lookup(
                    tokens0, nl0, return_waist_compressed=True)
                # Last breath's 512d compressed tensor at "=" position
                last_compressed = waist_compressed_per_breath[-1].cast(dtypes.float)  # (B, T, 512)
                waist_logits_all = last_compressed @ model.waist_head_w + model.waist_head_b   # (B, T, 4)
                waist_gathered = (waist_logits_all * eq_mask).sum(axis=1)                     # (B, 4)
                waist_aux_logits = waist_gathered * 10.0
                waist_aux_ce = waist_aux_logits.sparse_categorical_crossentropy(
                    op_labels, ignore_index=-100, reduction="mean")
            else:
                final_h, match_weights, _ = model.breathe_with_lookup(tokens0, nl0)
                waist_aux_ce = Tensor.zeros((), dtype=dtypes.float).contiguous()
            logits = (final_h @ model.embed_out).cast(dtypes.float)
            pred = logits[:, :-1, :]
            main_ce = pred.sparse_categorical_crossentropy(labels0, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            last_mw = match_weights[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(op_labels, ignore_index=-100, reduction="mean")
            l2_reg = (model.lookup_table.weight.square().mean()
                      + model.lookup_table.values.square().mean()
                      + model.lookup_table.value_proj_up.square().mean()) * 1e-6
            ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                         Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
            be_reg = (model.block.breath_embed.square().mean()
                      + model.block.handoff_w.square().mean()
                      + model.block.handoff_b.square().mean()
                      + model.block.rope.pitch.square().mean()
                      + model.block.crp_mix_alpha.square().mean()
                      + model.block.crp_target_norm.square().mean()
                      + model.block.notebook_write_w.square().mean()
                      + model.block.notebook_write_b.square().mean()
                      + model.block.notebook_read_w.square().mean()
                      + model.block.notebook_read_b.square().mean()
                      + model.block.notebook_write_query.square().mean()
                      + model.block.notebook_rep_write_w.square().mean()
                      + model.block.notebook_rep_write_b.square().mean()
                      + model.block.notebook_rep_read_w.square().mean()
                      + model.block.notebook_rep_read_b.square().mean()
                      + model.block.notebook_rep_query.square().mean()
                      + model.block.bfield_proj_down.square().mean()
                      + model.block.bfield_proj_up.square().mean()
                      + model.block.bfield_bias.square().mean()
                      + model.block.waist_codebook_keys.square().mean()
                      + model.block.waist_codebook_values.square().mean()
                      + model.waist_head_w.square().mean()
                      + model.waist_head_b.square().mean()
                      + sum((p.square().mean() for lb in model.block.layers_b
                                                for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                            Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7
            total = main_ce + aw * aux_ce + waw_local * waist_aux_ce + l2_reg + ch_reg + be_reg
            total.backward()
            opt.step()
            return total.realize()

    elif n_cycles == 2:
        nl0 = int(n_loops_per_cycle[0])
        nl1 = int(n_loops_per_cycle[1])
        # v26: thread notebook across cycles when CROSS_CYCLE_NOTEBOOK=1
        from mycelium.breathing import CROSS_CYCLE_NOTEBOOK as _CCN_2

        @TinyJit
        def _step(tokens0, labels0, tokens1, labels1, eq_mask, op_labels):
            opt.zero_grad()
            # Cycle 0 — breathe_with_lookup (heavy, provides match weights for aux)
            if _CCN_2:
                final0, mw, _, nb_c0, nb_r_c0 = model.breathe_with_lookup(tokens0, nl0, return_notebook=True)
            else:
                final0, mw, _ = model.breathe_with_lookup(tokens0, nl0)
                nb_c0, nb_r_c0 = None, None
            logits0 = (final0 @ model.embed_out).cast(dtypes.float)
            pred0 = logits0[:, :-1, :]
            main_ce0 = pred0.sparse_categorical_crossentropy(labels0, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            # Cycle 1 — plain forward (light, just CE on the execution gen).
            # v26c: seed notebook from cycle 0's final state when CROSS_CYCLE_NOTEBOOK=1.
            # Gradient flows through (no detach). v26-first-attempt collapsed with this
            # path when warm-starting from v24c (notebook trained on fresh-zero inputs,
            # OOD on cross-cycle non-zero). v26c warm-starts from v23a (no trained
            # notebook) + NOTEBOOK_STATE_INIT_SCALE>0 (random init regularizes the
            # model to handle any notebook state from the start).
            if _CCN_2:
                final1, _, _ = model.forward_with_notebook(tokens1, nl1,
                                                            initial_notebook=nb_c0,
                                                            initial_notebook_r=nb_r_c0)
            else:
                final1 = model(tokens1, nl1)
            logits1 = (final1 @ model.embed_out).cast(dtypes.float)
            pred1 = logits1[:, :-1, :]
            main_ce1 = pred1.sparse_categorical_crossentropy(labels1, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            # Aux CE from cycle 0's last-breath match weights
            last_mw = mw[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(op_labels, ignore_index=-100, reduction="mean")
            l2_reg = (model.lookup_table.weight.square().mean()
                      + model.lookup_table.values.square().mean()
                      + model.lookup_table.value_proj_up.square().mean()) * 1e-6
            ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                         Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
            be_reg = (model.block.breath_embed.square().mean()
                      + model.block.handoff_w.square().mean()
                      + model.block.handoff_b.square().mean()
                      + model.block.rope.pitch.square().mean()
                      + model.block.crp_mix_alpha.square().mean()
                      + model.block.crp_target_norm.square().mean()
                      + model.block.notebook_write_w.square().mean()
                      + model.block.notebook_write_b.square().mean()
                      + model.block.notebook_read_w.square().mean()
                      + model.block.notebook_read_b.square().mean()
                      + model.block.notebook_write_query.square().mean()
                      + model.block.notebook_rep_write_w.square().mean()
                      + model.block.notebook_rep_write_b.square().mean()
                      + model.block.notebook_rep_read_w.square().mean()
                      + model.block.notebook_rep_read_b.square().mean()
                      + model.block.notebook_rep_query.square().mean()
                      + model.block.bfield_proj_down.square().mean()
                      + model.block.bfield_proj_up.square().mean()
                      + model.block.bfield_bias.square().mean()
                      + model.block.waist_codebook_keys.square().mean()
                      + model.block.waist_codebook_values.square().mean()
                      + model.waist_head_w.square().mean()
                      + model.waist_head_b.square().mean()
                      + sum((p.square().mean() for lb in model.block.layers_b
                                                for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                            Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7
            avg_main = (main_ce0 + main_ce1) / 2.0
            total = avg_main + aw * aux_ce + l2_reg + ch_reg + be_reg
            total.backward()
            opt.step()
            return total.realize()

    elif n_cycles == 3:
        nl0 = int(n_loops_per_cycle[0])
        nl1 = int(n_loops_per_cycle[1])
        nl2 = int(n_loops_per_cycle[2])
        # v26: thread notebook across cycles when CROSS_CYCLE_NOTEBOOK=1
        from mycelium.breathing import CROSS_CYCLE_NOTEBOOK as _CCN_3

        @TinyJit
        def _step(tokens0, labels0, tokens1, labels1, tokens2, labels2, eq_mask, op_labels):
            opt.zero_grad()
            # Cycle 0 — breathe_with_lookup (heavy, provides match weights for aux)
            if _CCN_3:
                final0, mw, _, nb_c0, nb_r_c0 = model.breathe_with_lookup(tokens0, nl0, return_notebook=True)
            else:
                final0, mw, _ = model.breathe_with_lookup(tokens0, nl0)
                nb_c0, nb_r_c0 = None, None
            logits0 = (final0 @ model.embed_out).cast(dtypes.float)
            pred0 = logits0[:, :-1, :]
            main_ce0 = pred0.sparse_categorical_crossentropy(labels0, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            # Cycles 1 and 2 — plain forwards (light, just CE on the execution gen).
            # v26c: thread notebook from cycle N to cycle N+1, gradient through.
            if _CCN_3:
                final1, nb_c1, nb_r_c1 = model.forward_with_notebook(tokens1, nl1,
                                                                       initial_notebook=nb_c0,
                                                                       initial_notebook_r=nb_r_c0)
            else:
                final1 = model(tokens1, nl1)
                nb_c1, nb_r_c1 = None, None
            logits1 = (final1 @ model.embed_out).cast(dtypes.float)
            pred1 = logits1[:, :-1, :]
            main_ce1 = pred1.sparse_categorical_crossentropy(labels1, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            if _CCN_3:
                final2, _, _ = model.forward_with_notebook(tokens2, nl2,
                                                            initial_notebook=nb_c1,
                                                            initial_notebook_r=nb_r_c1)
            else:
                final2 = model(tokens2, nl2)
            logits2 = (final2 @ model.embed_out).cast(dtypes.float)
            pred2 = logits2[:, :-1, :]
            main_ce2 = pred2.sparse_categorical_crossentropy(labels2, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            # Aux CE from cycle 0's last-breath match weights
            last_mw = mw[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(op_labels, ignore_index=-100, reduction="mean")
            l2_reg = (model.lookup_table.weight.square().mean()
                      + model.lookup_table.values.square().mean()
                      + model.lookup_table.value_proj_up.square().mean()) * 1e-6
            ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                         Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
            be_reg = (model.block.breath_embed.square().mean()
                      + model.block.handoff_w.square().mean()
                      + model.block.handoff_b.square().mean()
                      + model.block.rope.pitch.square().mean()
                      + model.block.crp_mix_alpha.square().mean()
                      + model.block.crp_target_norm.square().mean()
                      + model.block.notebook_write_w.square().mean()
                      + model.block.notebook_write_b.square().mean()
                      + model.block.notebook_read_w.square().mean()
                      + model.block.notebook_read_b.square().mean()
                      + model.block.notebook_write_query.square().mean()
                      + model.block.notebook_rep_write_w.square().mean()
                      + model.block.notebook_rep_write_b.square().mean()
                      + model.block.notebook_rep_read_w.square().mean()
                      + model.block.notebook_rep_read_b.square().mean()
                      + model.block.notebook_rep_query.square().mean()
                      + model.block.bfield_proj_down.square().mean()
                      + model.block.bfield_proj_up.square().mean()
                      + model.block.bfield_bias.square().mean()
                      + model.block.waist_codebook_keys.square().mean()
                      + model.block.waist_codebook_values.square().mean()
                      + model.waist_head_w.square().mean()
                      + model.waist_head_b.square().mean()
                      + sum((p.square().mean() for lb in model.block.layers_b
                                                for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                            Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7
            avg_main = (main_ce0 + main_ce1 + main_ce2) / 3.0
            total = avg_main + aw * aux_ce + l2_reg + ch_reg + be_reg
            total.backward()
            opt.step()
            return total.realize()

    elif n_cycles == 4:
        nl0 = int(n_loops_per_cycle[0])
        nl1 = int(n_loops_per_cycle[1])
        nl2 = int(n_loops_per_cycle[2])
        nl3 = int(n_loops_per_cycle[3])
        from mycelium.breathing import CROSS_CYCLE_NOTEBOOK as _CCN_4

        @TinyJit
        def _step(tokens0, labels0, tokens1, labels1, tokens2, labels2,
                  tokens3, labels3, eq_mask, op_labels):
            opt.zero_grad()
            # Cycle 0 — breathe_with_lookup (heavy, provides match weights for aux)
            if _CCN_4:
                final0, mw, _, nb_c0, nb_r_c0 = model.breathe_with_lookup(tokens0, nl0, return_notebook=True)
            else:
                final0, mw, _ = model.breathe_with_lookup(tokens0, nl0)
                nb_c0, nb_r_c0 = None, None
            logits0 = (final0 @ model.embed_out).cast(dtypes.float)
            pred0 = logits0[:, :-1, :]
            main_ce0 = pred0.sparse_categorical_crossentropy(labels0, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            # Cycles 1, 2, 3 — plain forwards, optionally with cross-cycle notebook threading
            if _CCN_4:
                final1, nb_c1, nb_r_c1 = model.forward_with_notebook(tokens1, nl1,
                                                                       initial_notebook=nb_c0,
                                                                       initial_notebook_r=nb_r_c0)
            else:
                final1 = model(tokens1, nl1)
                nb_c1, nb_r_c1 = None, None
            logits1 = (final1 @ model.embed_out).cast(dtypes.float)
            pred1 = logits1[:, :-1, :]
            main_ce1 = pred1.sparse_categorical_crossentropy(labels1, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            if _CCN_4:
                final2, nb_c2, nb_r_c2 = model.forward_with_notebook(tokens2, nl2,
                                                                       initial_notebook=nb_c1,
                                                                       initial_notebook_r=nb_r_c1)
            else:
                final2 = model(tokens2, nl2)
                nb_c2, nb_r_c2 = None, None
            logits2 = (final2 @ model.embed_out).cast(dtypes.float)
            pred2 = logits2[:, :-1, :]
            main_ce2 = pred2.sparse_categorical_crossentropy(labels2, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            if _CCN_4:
                final3, _, _ = model.forward_with_notebook(tokens3, nl3,
                                                            initial_notebook=nb_c2,
                                                            initial_notebook_r=nb_r_c2)
            else:
                final3 = model(tokens3, nl3)
            logits3 = (final3 @ model.embed_out).cast(dtypes.float)
            pred3 = logits3[:, :-1, :]
            main_ce3 = pred3.sparse_categorical_crossentropy(labels3, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="mean")
            # Aux CE from cycle 0's last-breath match weights
            last_mw = mw[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(op_labels, ignore_index=-100, reduction="mean")
            l2_reg = (model.lookup_table.weight.square().mean()
                      + model.lookup_table.values.square().mean()
                      + model.lookup_table.value_proj_up.square().mean()) * 1e-6
            ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                         Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
            be_reg = (model.block.breath_embed.square().mean()
                      + model.block.handoff_w.square().mean()
                      + model.block.handoff_b.square().mean()
                      + model.block.rope.pitch.square().mean()
                      + model.block.crp_mix_alpha.square().mean()
                      + model.block.crp_target_norm.square().mean()
                      + model.block.notebook_write_w.square().mean()
                      + model.block.notebook_write_b.square().mean()
                      + model.block.notebook_read_w.square().mean()
                      + model.block.notebook_read_b.square().mean()
                      + model.block.notebook_write_query.square().mean()
                      + model.block.notebook_rep_write_w.square().mean()
                      + model.block.notebook_rep_write_b.square().mean()
                      + model.block.notebook_rep_read_w.square().mean()
                      + model.block.notebook_rep_read_b.square().mean()
                      + model.block.notebook_rep_query.square().mean()
                      + model.block.bfield_proj_down.square().mean()
                      + model.block.bfield_proj_up.square().mean()
                      + model.block.bfield_bias.square().mean()
                      + model.block.waist_codebook_keys.square().mean()
                      + model.block.waist_codebook_values.square().mean()
                      + model.waist_head_w.square().mean()
                      + model.waist_head_b.square().mean()
                      + sum((p.square().mean() for lb in model.block.layers_b
                                                for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                            Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7
            avg_main = (main_ce0 + main_ce1 + main_ce2 + main_ce3) / 4.0
            total = avg_main + aw * aux_ce + l2_reg + ch_reg + be_reg
            total.backward()
            opt.step()
            return total.realize()

    else:
        raise NotImplementedError(f"JIT-train n_cycles={n_cycles} not implemented (only 1, 2, 3, 4)")

    _JIT_TRAIN_CACHE[key] = _step
    print(f"[JIT] compiled in {_t_jit.perf_counter() - _jit_compile_start:.1f}s "
          f"(cache size={len(_JIT_TRAIN_CACHE)})", flush=True)
    return _step


def _compile_jit_per_breath_step(model, opt, K: int, fixed_len: int, B: int,
                                  lookup_aux_weight: float, grad_clip: float = 0.0):
    """JIT'd per-breath supervision step (v54/v55/v56 paradigm).

    Forward: K-breath breathe_with_lookup with return_per_breath_x and
    return_waist_compressed. Per-breath CE: each breath k decoded via the
    WaistController (if CONTROLLER_DECODE) or ln_f + embed_out (otherwise),
    supervised against step-k labels.

    Inputs (stable shapes):
      tokens              (B, fixed_len)
      labels_stk          (K, B, fixed_len - 1)   — per-step labels, -100 outside step k
      full_answer_labels  (B, fixed_len - 1)      — v75 full-answer labels (union of step ranges)
      eq_mask             (B, fixed_len, 1)       — 1.0 at "=" position
      op_labels           (B,)                    — op index 0..3 or -100

    Returns: (total_loss, ce_0, ce_1, ..., ce_{K-1}) — all scalar Tensors,
    each .realize()'d. Total = avg_main + lookup_aux_weight * aux_ce + regs.
    """
    # JIT cache key includes PER_BREATH_FULL_ANSWER, V77_DAG_TRAINING, V78_HEAD_CODEBOOK,
    # WAIST_ATTN_SUPERVISION, V79_CAUSAL_MASKS, MULTI_HEAD_WAIST, V81_MAIN_ATTN_MASK
    # so the multi-well, per-step, v77 DAG, v78 kitchen-sink, v78b attention-supervised,
    # v79 causal-masked, and v81 multi-head graphs don't collide.
    from mycelium.breathing import V78_HEAD_CODEBOOK as _V78_HC_KEY
    from mycelium.breathing import WAIST_ATTN_SUPERVISION as _WAS_KEY
    from mycelium.breathing import V79_CAUSAL_MASKS as _V79CM_KEY
    from mycelium.breathing import MULTI_HEAD_WAIST as _MHW_KEY
    from mycelium.breathing import V81_MAIN_ATTN_MASK as _V81MAM_KEY
    key = (id(model), id(opt), int(K), int(fixed_len), int(B), float(lookup_aux_weight), float(grad_clip), bool(PER_BREATH_FULL_ANSWER), bool(V77_DAG_TRAINING), bool(_V78_HC_KEY), bool(_WAS_KEY), bool(_V79CM_KEY), bool(_MHW_KEY), bool(_V81MAM_KEY))
    if key in _JIT_PER_BREATH_CACHE:
        return _JIT_PER_BREATH_CACHE[key]

    aw = float(lookup_aux_weight)
    gc_val = float(grad_clip)
    params = opt.params  # used for grad-norm computation if gc_val > 0
    pbfa = bool(PER_BREATH_FULL_ANSWER)  # captured at compile time
    v77_flag = bool(V77_DAG_TRAINING)  # captured at compile time (for log only; v77 doesn't change graph topology)
    v78_hc_flag = bool(_V78_HC_KEY)  # captured at compile time (for log only; topology depends on it inside _forward)
    was_flag = bool(_WAS_KEY)  # v78b attention supervision
    was_w = float(WAIST_ATTN_AUX_WEIGHT) if was_flag else 0.0
    v79cm_flag = bool(_V79CM_KEY)  # v79 causal masks during training
    mhw_flag = bool(_MHW_KEY)       # v81 multi-head WaistController
    v81mam_flag = bool(_V81MAM_KEY)  # v81 main-attn answer-span masking
    import time as _t_jit
    _jit_compile_start = _t_jit.perf_counter()
    print(f"[JIT] compile per_breath step: K={K} B={B} fixed_len={fixed_len} aw={aw} clip={gc_val} full_answer={pbfa} v77={v77_flag} v78_hc={v78_hc_flag} was={was_flag} was_w={was_w} v79cm={v79cm_flag} mhw={mhw_flag} v81mam={v81mam_flag}...", flush=True)

    from mycelium.breathing import CONTROLLER_DECODE as _CD
    from mycelium.breathing import _layernorm as _ln
    from mycelium.breathing import BOUNDARY_AUX_WEIGHT, WAIST_COPY as _WC, WAIST_COPY_AUX_WEIGHT as _WC_AUX_W
    HASH_TOKEN_ID = 1835  # `####` token in our tokenizer
    cfg_eps = model.cfg.layer_norm_eps
    bpw = BOUNDARY_POS_WEIGHT  # captured at compile time (default 5.0)

    @TinyJit
    def _step(tokens, labels_stk, full_answer_labels, eq_mask, op_labels, prompt_dropout_mask_t,
              attn_target_t, attn_mask_t, kv_mask_t, per_head_labels_t, v81_main_mask_t):
        opt.zero_grad()
        # v79 causal-mask plumbing: gate on v79cm_flag (compile-time bool). When ON,
        # pass kv_mask to breathe_with_lookup (notebook attn-pool) AND to every
        # WaistController.forward call (cross-attn). When OFF, both are None, so
        # behavior is byte-identical to v78b.
        _nb_pool_mask = kv_mask_t if v79cm_flag else None
        _wc_kv_mask  = kv_mask_t if v79cm_flag else None
        # v81 main_attn_mask: thread the prompt-range mask into breathe_with_lookup,
        # which (a) zeros the input embedding at answer-span positions and (b) blocks
        # main-self-attn keys at answer-span positions. When v81mam_flag is OFF,
        # behavior is byte-identical to v79.
        _main_attn_mask = v81_main_mask_t if v81mam_flag else None
        if _CD:
            _final, match_weights, per_breath_x, waist_compressed_per_breath = model.breathe_with_lookup(
                tokens, K, return_per_breath_x=True, return_waist_compressed=True,
                notebook_pool_mask=_nb_pool_mask, main_attn_mask=_main_attn_mask)
            prompt_emb = model.embed(tokens).cast(dtypes.float)
            # v81: when the embedding mask is on at training, the cross-attn KV embeddings
            # at answer-span positions are STILL the unmasked prompt_emb. The kv_mask
            # already blocks them in the cross-attn, so this is consistent. We could
            # also zero prompt_emb but it adds complexity for no behavioral change
            # (kv_mask additively penalizes those positions to ~0 weight).
        else:
            _final, match_weights, per_breath_x = model.breathe_with_lookup(
                tokens, K, return_per_breath_x=True,
                notebook_pool_mask=_nb_pool_mask, main_attn_mask=_main_attn_mask)

        losses_per_breath = []
        boundary_losses = []
        copy_aux_losses = []  # v72: per-breath aux loss supervising copy_attn → match positions
        last_cross_attn_for_aux = None  # v78b: stashed from LAST breath's WaistController.forward
        # v81 per-head CE losses (separately tracked so the trainer can log breakdowns).
        # Each is the SUM over breaths of the head's mean CE. Always built so the JIT
        # signature is stable; zero when mhw_flag=0. v81 only fires multi-head at the
        # FINAL breath (K-1) — earlier breaths use single-head decode to keep graph small.
        per_head_total_losses = [Tensor.zeros((), dtype=dtypes.float).contiguous() for _ in range(4)]
        for k in range(K):
            if _CD:
                waist_k = waist_compressed_per_breath[k].cast(dtypes.float)
                # v63: pass (k_idx, K_total) for K-position embedding lookup; pass
                # prompt_dropout_mask (1.0 = use prompt, 0.0 = zero prompt).
                # v72: also pass prompt_tokens when WAIST_COPY=1 so the controller
                # computes the copy-attention components (stashed on the instance).
                # v79: also pass kv_mask to mask cross-attn KV positions past prompt_len.
                # v81: force_single_head=True for breaths < K-1 (reduces graph
                # complexity; only the final breath needs multi-head for B6 emission).
                _force_sh = (mhw_flag and k < K - 1)
                logits_or_dict = model.waist_controller.forward(
                    waist_k, prompt_emb, model.embed_out,
                    k_idx=k, K_total=K,
                    prompt_dropout_mask=prompt_dropout_mask_t,
                    prompt_tokens=(tokens if _WC else None),
                    kv_mask=_wc_kv_mask,
                    force_single_head=_force_sh)
                # v78b: capture cross-attn ONLY on the last breath (the one that will
                # carry the attention-supervision aux loss). Overwriting each breath
                # gives us breath K-1's value at loop end. Gated on was_flag so when
                # the supervision is OFF the stash slot in the controller stays None
                # and we never construct a graph that depends on it.
                if was_flag and k == K - 1:
                    last_cross_attn_for_aux = model.waist_controller._last_cross_attn
                # v81: WaistController returns a dict {ops, types, args1, args2} only at
                # the FINAL breath (when force_single_head=False). Earlier breaths return
                # a single tensor. mhw_flag is captured at compile time so the graph
                # branch is specialized.
                if mhw_flag and k == K - 1:
                    logits = logits_or_dict["ops"]  # representative for downstream boundary aux
                else:
                    logits = logits_or_dict
            else:
                x_k = per_breath_x[k]
                x_normed = _ln(x_k, model.ln_f_g, model.ln_f_b, cfg_eps)
                logits = (x_normed @ model.embed_out).cast(dtypes.float)

            # v81 multi-head CE path — only fires for the FINAL breath (graph cost).
            # Earlier breaths fall through to the legacy single-head CE below.
            if _CD and mhw_flag and k == K - 1:
                head_names_local = ["ops", "types", "args1", "args2"]
                breath_loss_acc = Tensor.zeros((), dtype=dtypes.float).contiguous()
                for hi, name in enumerate(head_names_local):
                    lh = logits_or_dict[name][:, :-1, :]  # (B, T-1, V)
                    target_h_k = per_head_labels_t[hi, k]  # (B, T-1)
                    ce_h = lh.sparse_categorical_crossentropy(
                        target_h_k, ignore_index=-100,
                        label_smoothing=LABEL_SMOOTHING, reduction="mean")
                    breath_loss_acc = breath_loss_acc + ce_h
                    per_head_total_losses[hi] = per_head_total_losses[hi] + ce_h
                ce_k = breath_loss_acc / 4.0
                losses_per_breath.append(ce_k)
                target_k = full_answer_labels if pbfa else labels_stk[k]
                pred = logits[:, :-1, :]  # for the boundary-aux block
                if BOUNDARY_AUX_WEIGHT > 0.0:
                    x_k_b = per_breath_x[k].cast(dtypes.float)
                    B_b = x_k_b.shape[0]; T_b = x_k_b.shape[1]
                    blogits = (x_k_b @ model.boundary_head_w + model.boundary_head_b).reshape(B_b, T_b)
                    blogits_pred = blogits[:, :-1]
                    labels_k_b = target_k
                    btarget = (labels_k_b == HASH_TOKEN_ID).cast(dtypes.float)
                    valid = (labels_k_b != -100).cast(dtypes.float)
                    bce_per = blogits_pred.maximum(0.0) - blogits_pred * btarget + (1.0 + (-blogits_pred.abs()).exp()).log()
                    weight_per_pos = 1.0 + (bpw - 1.0) * btarget
                    bce_per = bce_per * weight_per_pos
                    bce_k = (bce_per * valid).sum() / (valid.sum() + 1.0)
                    boundary_losses.append(bce_k)
                continue  # skip the legacy single-head CE block below
            pred = logits[:, :-1, :]
            # v75 multi-well supervision: every breath shares the same target
            # (full_answer_labels) when PER_BREATH_FULL_ANSWER=1. pbfa was captured at
            # compile time, so the JIT graph is fully specialized — no runtime branch.
            target_k = full_answer_labels if pbfa else labels_stk[k]
            if _WC and _CD and model.waist_controller._last_copy_attn is not None:
                # v72 mixed CE: p_final = (1 - gate) * p_vocab + gate * p_copy
                # pred: (B, T-1, V); copy_attn: (B, T, T_prompt); copy_gate: (B, T, 1)
                y_target = target_k                                     # (B, T-1)
                log_p_vocab = pred.log_softmax(axis=-1)                  # (B, T-1, V)
                # Gather log-prob at target; clamp y_target to >= 0 to avoid OOB on -100.
                y_safe = y_target.maximum(0).reshape(*y_target.shape, 1)
                log_p_vocab_y = log_p_vocab.gather(-1, y_safe).reshape(y_target.shape)  # (B, T-1)
                p_vocab_y = log_p_vocab_y.exp()                          # (B, T-1)
                # Copy distribution: build mask (B, T-1, T_prompt) where prompt[b,i] == y_target[b,t].
                # tokens is the full sequence; T_prompt == T (the full conditioning context).
                copy_attn_full = model.waist_controller._last_copy_attn  # (B, T, T_prompt)
                copy_attn_pred = copy_attn_full[:, :-1, :]               # (B, T-1, T_prompt)
                copy_gate_full = model.waist_controller._last_copy_gate  # (B, T, 1)
                copy_gate_pred = copy_gate_full[:, :-1, 0]               # (B, T-1)
                # Match mask: y_target_b_t == tokens_b_i
                # tokens: (B, T); reshape to (B, 1, T) and y_target to (B, T-1, 1)
                match_mask = (y_target.reshape(*y_target.shape, 1) == tokens.reshape(tokens.shape[0], 1, tokens.shape[1])).cast(dtypes.float)
                p_copy_y = (copy_attn_pred * match_mask).sum(axis=-1)    # (B, T-1)
                p_final = (1.0 - copy_gate_pred) * p_vocab_y + copy_gate_pred * p_copy_y
                # CE over valid positions (label != -100). Avoid log(0) with +1e-12.
                valid = (y_target != -100).cast(dtypes.float)
                log_p_final = (p_final + 1e-12).log()
                ce_k = -(log_p_final * valid).sum() / (valid.sum() + 1.0)
                # v72 AUX LOSS: supervise copy_attn → matching prompt positions.
                # Use log_softmax of pre-softmax scores (stable). EXCLUDE pad-pad matches:
                # when y_target == 0 and prompt has pad tokens, match_mask fires at pad
                # positions where copy_scores was masked to -1e4, log_softmax there ≈ -1e4
                # → blows up aux to ~1e4. Mask both sides to legit token matches only.
                copy_scores_full = model.waist_controller._last_copy_scores       # (B, T, T_p)
                copy_scores_pred = copy_scores_full[:, :-1, :]                    # (B, T-1, T_p)
                log_attn = copy_scores_pred.log_softmax(axis=-1)                  # (B, T-1, T_p)
                # Exclude pad positions in the PROMPT (token 0) — log_softmax there is ~-1e4.
                prompt_nonpad = (tokens != 0).cast(dtypes.float)                  # (B, T)
                prompt_nonpad_r = prompt_nonpad.reshape(prompt_nonpad.shape[0], 1, prompt_nonpad.shape[1])
                match_mask_clean = match_mask * prompt_nonpad_r                    # (B, T-1, T_p)
                # Exclude pad targets (y_target == 0) — degenerate target distribution.
                y_nonpad = (y_target != 0).cast(dtypes.float)                     # (B, T-1)
                neg_sum = -(match_mask_clean * log_attn).sum(axis=-1)             # (B, T-1)
                match_count = match_mask_clean.sum(axis=-1)                       # (B, T-1)
                has_match = (match_count > 0.5).cast(dtypes.float)                # (B, T-1)
                aux_per = neg_sum / (match_count + 1e-9)                          # (B, T-1)
                # Clip per-position aux to log(T_p)*2 to defend against any remaining outlier.
                aux_per = aux_per.clip(0.0, 16.0)
                aux_weight = has_match * valid * y_nonpad                          # (B, T-1)
                copy_aux_k = (aux_per * aux_weight).sum() / (aux_weight.sum() + 1.0)
                copy_aux_losses.append(copy_aux_k)
            else:
                ce_k = pred.sparse_categorical_crossentropy(
                    target_k, ignore_index=-100,
                    label_smoothing=LABEL_SMOOTHING, reduction="mean")
            losses_per_breath.append(ce_k)
            # v65 boundary aux: per-position binary head predicting "is next token ####?".
            # Applied to per_breath_x[k] (1024d post-breath hidden state).
            # Stable BCE from logits: max(z,0) - z*y + log(1 + exp(-|z|)).
            # v75: mask same positions as main CE (full-answer or per-step).
            if BOUNDARY_AUX_WEIGHT > 0.0:
                x_k_b = per_breath_x[k].cast(dtypes.float)  # (B, T, hidden)
                B_b = x_k_b.shape[0]; T_b = x_k_b.shape[1]
                blogits = (x_k_b @ model.boundary_head_w + model.boundary_head_b).reshape(B_b, T_b)
                blogits_pred = blogits[:, :-1]
                labels_k_b = target_k  # (B, T-1) — v75: shares with main CE
                btarget = (labels_k_b == HASH_TOKEN_ID).cast(dtypes.float)
                valid = (labels_k_b != -100).cast(dtypes.float)
                bce_per = blogits_pred.maximum(0.0) - blogits_pred * btarget + (1.0 + (-blogits_pred.abs()).exp()).log()
                # v66: up-weight positive class (#### is rare ~1 per 10-30 negatives)
                weight_per_pos = 1.0 + (bpw - 1.0) * btarget  # 1.0 for negatives, bpw for positives
                bce_per = bce_per * weight_per_pos
                bce_k = (bce_per * valid).sum() / (valid.sum() + 1.0)
                boundary_losses.append(bce_k)
        avg_main = sum(losses_per_breath[1:], losses_per_breath[0]) / float(K)
        if BOUNDARY_AUX_WEIGHT > 0.0 and len(boundary_losses) > 0:
            avg_boundary = sum(boundary_losses[1:], boundary_losses[0]) / float(len(boundary_losses))
        else:
            avg_boundary = Tensor.zeros((), dtype=dtypes.float).contiguous()

        if aw > 0.0:
            last_mw = match_weights[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(
                op_labels, ignore_index=-100, reduction="mean")
        else:
            aux_ce = Tensor.zeros((), dtype=dtypes.float).contiguous()

        l2_reg = (model.lookup_table.weight.square().mean()
                  + model.lookup_table.values.square().mean()
                  + model.lookup_table.value_proj_up.square().mean()) * 1e-6
        ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                     Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
        be_reg = (model.block.breath_embed.square().mean()
                  + model.block.handoff_w.square().mean()
                  + model.block.handoff_b.square().mean()
                  + model.block.rope.pitch.square().mean()
                  + model.block.crp_mix_alpha.square().mean()
                  + model.block.crp_target_norm.square().mean()
                  + model.block.notebook_write_w.square().mean()
                  + model.block.notebook_write_b.square().mean()
                  + model.block.notebook_read_w.square().mean()
                  + model.block.notebook_read_b.square().mean()
                  + model.block.notebook_write_query.square().mean()
                  + model.block.notebook_rep_write_w.square().mean()
                  + model.block.notebook_rep_write_b.square().mean()
                  + model.block.notebook_rep_read_w.square().mean()
                  + model.block.notebook_rep_read_b.square().mean()
                  + model.block.notebook_rep_query.square().mean()
                  + model.block.bfield_proj_down.square().mean()
                  + model.block.bfield_proj_up.square().mean()
                  + model.block.bfield_bias.square().mean()
                  + model.block.waist_codebook_keys.square().mean()
                  + model.block.waist_codebook_values.square().mean()
                  + model.waist_head_w.square().mean()
                  + model.waist_head_b.square().mean()
                  + model.ln_f_g.square().mean()        # v77: ln_f params have no gradient path
                  + model.ln_f_b.square().mean()        #      when CONTROLLER_DECODE=1 AND aux_ce=0
                  + sum((p.square().mean() for lb in model.block.layers_b
                                            for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                        Tensor.zeros((), dtype=dtypes.float).contiguous())
                  # v81 multi-head MLP params: ALWAYS L2-regged for ckpt symmetry. When
                  # MULTI_HEAD_WAIST=0 they're inert; the L2 keeps their gradient defined
                  # so AdamW doesn't assert on None grad. When MULTI_HEAD_WAIST=1, the
                  # simplified bias-only forward path only touches b1+b2; w1/w2 stay at
                  # zero-init via L2 alone (effectively frozen).
                  + sum((p.square().mean() for mlp in model.waist_controller.head_mlps
                                            for p in [mlp["w1"], mlp["b1"], mlp["w2"], mlp["b2"]]),
                        Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7

        # v70/v71: per-breath sparsity losses are collected by apply_collapse_v70/71 into a
        # Python list. We sum them ONCE here, OUTSIDE the per-breath chain. This decouples
        # the K-deep sparsity computation from the per-breath chain and lets AMD schedule
        # the K reductions independently (vs the self.X = self.X + s pattern which serializes).
        from mycelium.breathing import COLLAPSE_V70 as _CV70_inner, COLLAPSE_V71 as _CV71_inner
        if _CV71_inner and model.block._collapse_v71_sparsity_list:
            _sl = model.block._collapse_v71_sparsity_list
            v70_sparsity = _sl[0] if len(_sl) == 1 else sum(_sl[1:], _sl[0])
        elif _CV70_inner and model.block._collapse_v70_sparsity_list:
            _sl = model.block._collapse_v70_sparsity_list
            v70_sparsity = _sl[0] if len(_sl) == 1 else sum(_sl[1:], _sl[0])
        else:
            v70_sparsity = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # v72: average copy-attn aux loss across breaths; gated by WAIST_COPY at runtime.
        if _WC and copy_aux_losses:
            avg_copy_aux = sum(copy_aux_losses[1:], copy_aux_losses[0]) / float(len(copy_aux_losses))
        else:
            avg_copy_aux = Tensor.zeros((), dtype=dtypes.float).contiguous()
        # v78b ATTENTION SUPERVISION AUX LOSS — last-breath only.
        # CE between WaistController's last cross-attn weights and the per-token
        # uniform-over-matches target distribution. Masked to digit-token positions
        # in the output (attn_mask_t = 1 at supervised positions, 0 elsewhere).
        # attn_target_t: (B, T-1, T)  — distribution over prompt positions; rows sum to 1 at supervised t, 0 elsewhere.
        # last_cross_attn_for_aux: (B, T_q, T_kv) = (B, T, T) — head-mean attention.
        # We slice the Q axis to positions [0..T-1] so it aligns with the labels/mask.
        if was_flag and last_cross_attn_for_aux is not None:
            cross_attn_pred = last_cross_attn_for_aux[:, :-1, :]              # (B, T-1, T_kv)
            log_attn = (cross_attn_pred + 1e-12).log()                        # (B, T-1, T_kv)
            neg_ce_per_t = -(attn_target_t * log_attn).sum(axis=-1)           # (B, T-1)
            attn_aux_loss = (neg_ce_per_t * attn_mask_t).sum() / (attn_mask_t.sum() + 1.0)
        else:
            attn_aux_loss = Tensor.zeros((), dtype=dtypes.float).contiguous()
        total = avg_main + aw * aux_ce + l2_reg + ch_reg + be_reg + BOUNDARY_AUX_WEIGHT * avg_boundary + v70_sparsity + _WC_AUX_W * avg_copy_aux + was_w * attn_aux_loss
        total.backward()
        # NaN-skip: if loss is NaN (forward overflow), zero all gradients so
        # opt.step is a no-op. Checked via total.isfinite() — single kernel,
        # avoids the per-param isnan() loop that crashes the AMD driver in JIT.
        healthy = total.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)
        # Global-norm gradient clipping — guards against NaN spikes at bigger model
        # sizes. Pythia-1B (H=2048) hit gradient explosion around step 2000-2500
        # without it. GRAD_CLIP env var (default 0 = off for 410M; 1.0 for 1B).
        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6))
            clip_coef = clip_coef.minimum(Tensor(1.0, dtype=dtypes.float))
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)
        opt.step()
        # v66 waist norm: mean L2 norm of last breath's compressed waist.
        # Monitored for skip-connection leakage (if model ignores waist, norm → 0).
        # Computed here (inside JIT) so it's part of the same fused graph with no extra sync.
        if _CD and waist_compressed_per_breath:
            waist_last = waist_compressed_per_breath[-1].cast(dtypes.float)
            waist_norm = waist_last.square().mean(axis=-1).sqrt().mean()
        else:
            waist_norm = Tensor.zeros((), dtype=dtypes.float).contiguous()
        return (total.realize(), healthy.realize(), waist_norm.realize(), attn_aux_loss.realize(),
                *(ce.realize() for ce in losses_per_breath),
                # v81 per-head sums (4 values appended at the END so legacy callers can
                # still slice ce_ts = outs[4:4+K] without disruption).
                *(ph.realize() for ph in per_head_total_losses))

    _JIT_PER_BREATH_CACHE[key] = _step
    print(f"[JIT] compiled per_breath in {_t_jit.perf_counter() - _jit_compile_start:.1f}s "
          f"(cache size={len(_JIT_PER_BREATH_CACHE)})", flush=True)
    return _step


# Separate cache for scheduled-sampling JIT (different input signature).
_JIT_PER_BREATH_SS_CACHE: dict = {}

SCHED_SAMPLE_RATE = float(os.environ.get("SCHED_SAMPLE_RATE", "0.0"))
BOUNDARY_POS_WEIGHT = float(os.environ.get("BOUNDARY_POS_WEIGHT", "5.0"))

if SCHED_SAMPLE_RATE > 0.0:
    print(f"[SCHED_SAMPLE] rate={SCHED_SAMPLE_RATE} — scheduled sampling enabled "
          f"(warmup=500 steps, ramp to {SCHED_SAMPLE_RATE} at step 1500)", flush=True)
if BOUNDARY_POS_WEIGHT != 1.0:
    print(f"[BOUNDARY_POS_WEIGHT] pos_weight={BOUNDARY_POS_WEIGHT}", flush=True)

# Waist norm tracking state (module-level so it persists across calls).
_waist_norm_low_streak: int = 0
_WAIST_NORM_WARN_THRESHOLD = 0.01
_WAIST_NORM_WARN_STREAK = 100


def _compile_jit_per_breath_step_ss(model, opt, K: int, fixed_len: int, B: int,
                                     lookup_aux_weight: float, grad_clip: float = 0.0):
    """JIT'd per-breath supervision with scheduled sampling (v66).

    Same as _compile_jit_per_breath_step but unrolls K breaths manually so that
    after breath k, argmax predictions can replace some gold tokens before breath
    k+1 (scheduled sampling). Forces the model to handle its own prediction errors.

    Additional inputs vs the base version:
      sched_sample_rate_t  (1,) float — current scheduled-sample rate (0.0 = off).
                           Passed as Tensor so the JIT graph doesn't recompile per step.
      bernoulli_mask_t     (B, fixed_len - 1) float — 1.0 at positions to be replaced
                           by argmax predictions. Precomputed in Python each step using
                           np.random.binomial(..., p=rate). Zeros when rate=0.

    Scheduled sampling mechanism:
      After breath k (k < K-1):
        1. Compute WaistController logits for breath k.
        2. Take argmax → predicted tokens (B, T-1).
        3. Replace gold tokens at Bernoulli-selected positions in the token sequence.
        4. Re-embed the modified sequence → x_emb for breath k+1 (BREATHE_FRESH_INPUT=0
           path: x flows as hidden states; we inject modified embeddings at the START
           of the next breath via x_in perturbation).

    The argmax is computed with stop_gradient (detached from the training graph) — we
    want the model to SEE its own errors, not backprop through the argmax decision.

    AMD JIT constraints respected:
      - No .cast(dtypes.float32) on new large tensors inside JIT (only dtypes.half).
      - Bernoulli mask is precomputed outside JIT and passed as float Tensor input.
      - Scheduled rate is a scalar Tensor input (no Python branch per rate value).
    """
    # JIT cache key includes PER_BREATH_FULL_ANSWER, V77_DAG_TRAINING, V78_HEAD_CODEBOOK,
    # WAIST_ATTN_SUPERVISION, V79_CAUSAL_MASKS, MULTI_HEAD_WAIST, V81_MAIN_ATTN_MASK
    # so the v75 multi-well, per-step, v77 DAG, v78 kitchen-sink, v78b attention-supervised,
    # v79 causal-masked, and v81 multi-head graphs don't collide.
    from mycelium.breathing import V78_HEAD_CODEBOOK as _V78_HC_KEY_SS
    from mycelium.breathing import WAIST_ATTN_SUPERVISION as _WAS_KEY_SS
    from mycelium.breathing import V79_CAUSAL_MASKS as _V79CM_KEY_SS
    from mycelium.breathing import MULTI_HEAD_WAIST as _MHW_KEY_SS
    from mycelium.breathing import V81_MAIN_ATTN_MASK as _V81MAM_KEY_SS
    key = (id(model), id(opt), int(K), int(fixed_len), int(B), float(lookup_aux_weight), float(grad_clip), bool(PER_BREATH_FULL_ANSWER), bool(V77_DAG_TRAINING), bool(_V78_HC_KEY_SS), bool(_WAS_KEY_SS), bool(_V79CM_KEY_SS), bool(_MHW_KEY_SS), bool(_V81MAM_KEY_SS))
    if key in _JIT_PER_BREATH_SS_CACHE:
        return _JIT_PER_BREATH_SS_CACHE[key]

    aw = float(lookup_aux_weight)
    gc_val = float(grad_clip)
    params = opt.params
    pbfa = bool(PER_BREATH_FULL_ANSWER)  # captured at compile time
    v77_flag = bool(V77_DAG_TRAINING)
    v78_hc_flag = bool(_V78_HC_KEY_SS)
    was_flag = bool(_WAS_KEY_SS)  # v78b attention supervision
    was_w = float(WAIST_ATTN_AUX_WEIGHT) if was_flag else 0.0
    v79cm_flag = bool(_V79CM_KEY_SS)  # v79 causal masks during training
    mhw_flag = bool(_MHW_KEY_SS)       # v81 multi-head WaistController
    v81mam_flag = bool(_V81MAM_KEY_SS)  # v81 main-attn answer-span masking
    import time as _t_jit
    _jit_compile_start = _t_jit.perf_counter()
    print(f"[JIT-SS] compile per_breath_ss step: K={K} B={B} fixed_len={fixed_len} aw={aw} clip={gc_val} full_answer={pbfa} v77={v77_flag} v78_hc={v78_hc_flag} was={was_flag} was_w={was_w} v79cm={v79cm_flag} mhw={mhw_flag} v81mam={v81mam_flag}...", flush=True)

    from mycelium.breathing import CONTROLLER_DECODE as _CD
    from mycelium.breathing import _layernorm as _ln
    from mycelium.breathing import BOUNDARY_AUX_WEIGHT, WAIST_COPY as _WC, WAIST_COPY_AUX_WEIGHT as _WC_AUX_W
    from mycelium.breathing import (
        NOTEBOOK_V24, NOTEBOOK_ACCUMULATE_ENABLED, NOTEBOOK_DUAL, NOTEBOOK_POOL_MODE,
        CROSS_BREATH_HANDOFF, BREATHE_FRESH_INPUT, PROMPT_REFRESH_ALPHA,
        NOTEBOOK_DAG, BFIELD_SIN_MOD, STOCH_DEPTH_P, BFIELD_WAIST, BFIELD_END_OF_BREATH,
        LOOKUP_VALUE_INJECT, LOOKUP_VALUE_SCALE, _initial_notebook_state, _sine_temp_baseline,
    )
    import math as _math
    HASH_TOKEN_ID = 1835  # `####` token
    cfg_eps = model.cfg.layer_norm_eps
    T = fixed_len
    bpw = BOUNDARY_POS_WEIGHT  # captured at compile time

    @TinyJit
    def _step_ss(tokens, labels_stk, full_answer_labels, eq_mask, op_labels, prompt_dropout_mask_t,
                 sched_sample_rate_t, bernoulli_mask_t, attn_target_t, attn_mask_t, kv_mask_t,
                 per_head_labels_t, v81_main_mask_t):
        opt.zero_grad()
        # v79 causal-mask plumbing — see base JIT for rationale.
        _nb_pool_mask_ss = kv_mask_t if v79cm_flag else None
        _wc_kv_mask_ss  = kv_mask_t if v79cm_flag else None
        # v81 main-attn answer-span mask — threaded into the inline breathe_once calls.
        _main_attn_mask_ss = v81_main_mask_t if v81mam_flag else None

        # --- Manually unrolled K-breath forward pass (mirrors breathe_with_lookup) ---
        x_emb = model.embed(tokens).cast(model.block.bfield_proj_down.dtype)  # (B, T, H), half precision
        # v81 embed mask: zero the input embeddings at answer-span positions when the
        # main-attn mask is on. Matches breathe_with_lookup's behavior; required for the
        # masking-audit pass-condition at non-prompt query positions.
        if _main_attn_mask_ss is not None:
            x_emb = x_emb * _main_attn_mask_ss.cast(x_emb.dtype).reshape(x_emb.shape[0], -1, 1)
        x = x_emb
        integral = Tensor.zeros_like(x)
        handoff = None
        notebook = None
        notebook_r = None
        weight_sum = 0.0

        if NOTEBOOK_V24:
            B_nb = x.shape[0]
            notebook = _initial_notebook_state(B_nb, model.block.nb_dim)
            if NOTEBOOK_DUAL:
                notebook_r = _initial_notebook_state(B_nb, model.block.nb_dim)

        per_breath_x = []
        waist_compressed_per_breath = []
        match_weights_list = []
        prompt_emb = None
        if _CD:
            prompt_emb = model.embed(tokens).cast(dtypes.float)

        for l in range(K):
            # --- x_in setup (mirrors breathe_with_lookup) ---
            if BREATHE_FRESH_INPUT:
                x_in = x_emb
            else:
                x_in = x
                if CROSS_BREATH_HANDOFF and handoff is not None:
                    x_in = x_in + handoff
            if PROMPT_REFRESH_ALPHA > 0.0:
                x_in = x_in + (PROMPT_REFRESH_ALPHA * x_emb).cast(x_in.dtype)
            if NOTEBOOK_V24:
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    read_vec = (notebook @ model.block.notebook_read_w + model.block.notebook_read_b)
                    x_in = x_in + read_vec.reshape(x_in.shape[0], 1, -1).cast(x_in.dtype)
                if NOTEBOOK_DUAL:
                    read_vec_r = (notebook_r @ model.block.notebook_rep_read_w + model.block.notebook_rep_read_b)
                    x_in = x_in + read_vec_r.reshape(x_in.shape[0], 1, -1).cast(x_in.dtype)

            # --- Breath forward ---
            temp_mult = _sine_temp_baseline(l, K)
            if _CD and BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                x, waist_compressed = model.block.breathe_once(x_in, l, temp_mult=temp_mult,
                                                                  return_waist_compressed=True, n_loops=K,
                                                                  attn_mask=_main_attn_mask_ss)
                waist_compressed_per_breath.append(waist_compressed)
            else:
                x = model.block.breathe_once(x_in, l, temp_mult=temp_mult, n_loops=K,
                                              attn_mask=_main_attn_mask_ss)

            # --- Notebook write ---
            # v79: when v79cm_flag is ON, mask the attn-pool to prompt range only.
            if NOTEBOOK_V24:
                x_f = x.cast(dtypes.float)
                if NOTEBOOK_POOL_MODE == "attn":
                    scores = (x_f * model.block.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                    if _nb_pool_mask_ss is not None:
                        scores = scores + (1.0 - _nb_pool_mask_ss.cast(scores.dtype)) * (-1e4)
                    weights = scores.softmax(axis=-1).reshape(x.shape[0], -1, 1)
                    x_pool = (x_f * weights).sum(axis=1)
                else:
                    x_pool = x_f.mean(axis=1)
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    notebook = notebook + (x_pool @ model.block.notebook_write_w + model.block.notebook_write_b)
                if NOTEBOOK_DUAL:
                    if NOTEBOOK_POOL_MODE == "attn":
                        scores_r = (x_f * model.block.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                        if _nb_pool_mask_ss is not None:
                            scores_r = scores_r + (1.0 - _nb_pool_mask_ss.cast(scores_r.dtype)) * (-1e4)
                        weights_r = scores_r.softmax(axis=-1).reshape(x.shape[0], -1, 1)
                        x_pool_r = (x_f * weights_r).sum(axis=1)
                    else:
                        x_pool_r = x_pool
                    notebook_r = x_pool_r @ model.block.notebook_rep_write_w + model.block.notebook_rep_write_b

            # --- Handoff ---
            if CROSS_BREATH_HANDOFF:
                handoff = model.block.compute_handoff(x)

            per_breath_x.append(x)

            # --- Integral accumulation ---
            if BFIELD_SIN_MOD:
                beta_l = _math.sin((l + 0.5) * _math.pi / float(K))
            else:
                beta_l = 1.0
            if STOCH_DEPTH_P > 0.0 and Tensor.training:
                keep_l = model.block.stoch_keep_mask[l].cast(x.dtype)
                integral = integral + beta_l * keep_l * x
            else:
                integral = integral + beta_l * x
            weight_sum = weight_sum + beta_l
            running = integral / weight_sum
            running_normed = _ln(running, model.ln_f_g, model.ln_f_b, cfg_eps)
            match_weights_list.append(model.lookup_table(running_normed))

            # --- v66 Scheduled sampling: after breath l, mix some gold tokens with argmax ---
            # Only applies for l < K-1 (no need to modify input after the last breath).
            # The argmax is computed with stop_gradient (detached from the graph).
            if _CD and l < K - 1:
                # Compute logits for breath l. v81: force single-head decode at l<K-1
                # (the multi-head dict only matters for the final breath's B6 emission).
                waist_l = waist_compressed_per_breath[l].cast(dtypes.float)
                logits_l = model.waist_controller.forward(
                    waist_l, prompt_emb, model.embed_out,
                    k_idx=l, K_total=K,
                    prompt_dropout_mask=prompt_dropout_mask_t,
                    prompt_tokens=(tokens if _WC else None),
                    kv_mask=_wc_kv_mask_ss,
                    force_single_head=mhw_flag)  # single-head when v81 is on (returns single tensor)
                # Argmax predictions: (B, T-1) integer tokens — no gradient.
                # v72: when WAIST_COPY=1, the scheduled-sampling argmax is computed on
                # the VOCAB logits only (not the copy distribution). The copy distribution
                # is dense over T_prompt and would require scatter-into-vocab; for the
                # mid-training scheduled-sampling decision it's acceptable to use the
                # vocab argmax since (a) it doesn't affect the loss directly, (b) the
                # main CE still trains the copy mechanism via the mixed-loss path below.
                argmax_tokens_l = logits_l[:, :-1, :].detach().argmax(axis=-1)  # (B, T-1) int
                # Gold tokens for positions 1..T-1 (align with argmax labels)
                gold_tail = tokens[:, 1:]  # (B, T-1) int
                # Bernoulli mask (float 0.0/1.0) selects which positions use argmax.
                # Use float arithmetic to avoid int-cast issues; round-trip via int at end.
                mask_l_f = bernoulli_mask_t  # (B, T-1) float, 0.0 or 1.0
                gold_tail_f = gold_tail.cast(dtypes.float)   # (B, T-1) float
                argmax_f = argmax_tokens_l.cast(dtypes.float)  # (B, T-1) float
                mixed_tail_f = gold_tail_f * (1.0 - mask_l_f) + argmax_f * mask_l_f
                mixed_tail = mixed_tail_f.cast(dtypes.int)  # (B, T-1) int
                # Reconstruct full token sequence (position 0 always gold)
                gold_head = tokens[:, :1]  # (B, 1) int
                mixed_tokens = gold_head.cat(mixed_tail, dim=1)  # (B, T) int
                # Re-embed mixed token sequence for next breath's x_emb
                mixed_emb = model.embed(mixed_tokens).cast(x_emb.dtype)  # (B, T, H)
                # Scale the embedding delta by sched_sample_rate_t (0.0 = no change).
                # delta=0 when mask=all-zeros (rate=0 case), so this is exact.
                rate_scalar = sched_sample_rate_t.cast(x_emb.dtype).reshape(1, 1, 1)
                x_emb = x_emb + rate_scalar * (mixed_emb - x_emb)

        # --- Per-breath CE losses ---
        losses_per_breath = []
        boundary_losses = []
        copy_aux_losses = []  # v72: per-breath aux loss supervising copy_attn → match positions
        last_cross_attn_for_aux = None  # v78b: stashed from LAST breath's WaistController.forward
        # v81 per-head loss accumulators (always built so JIT signature stable).
        per_head_total_losses_ss = [Tensor.zeros((), dtype=dtypes.float).contiguous() for _ in range(4)]
        for k in range(K):
            if _CD:
                waist_k = waist_compressed_per_breath[k].cast(dtypes.float)
                _force_sh_ss = (mhw_flag and k < K - 1)
                logits_or_dict = model.waist_controller.forward(
                    waist_k, prompt_emb, model.embed_out,
                    k_idx=k, K_total=K,
                    prompt_dropout_mask=prompt_dropout_mask_t,
                    prompt_tokens=(tokens if _WC else None),
                    kv_mask=_wc_kv_mask_ss,
                    force_single_head=_force_sh_ss)
                # v78b: capture cross-attn ONLY on the last breath. Same as base JIT.
                if was_flag and k == K - 1:
                    last_cross_attn_for_aux = model.waist_controller._last_cross_attn
                if mhw_flag and k == K - 1:
                    logits = logits_or_dict["ops"]  # representative for downstream
                else:
                    logits = logits_or_dict
            else:
                x_k = per_breath_x[k]
                x_normed = _ln(x_k, model.ln_f_g, model.ln_f_b, cfg_eps)
                logits = (x_normed @ model.embed_out).cast(dtypes.float)
            # v81 per-head CE — only fires at the FINAL breath (graph cost).
            if _CD and mhw_flag and k == K - 1:
                head_names_local = ["ops", "types", "args1", "args2"]
                breath_loss_acc = Tensor.zeros((), dtype=dtypes.float).contiguous()
                for hi, name in enumerate(head_names_local):
                    lh = logits_or_dict[name][:, :-1, :]
                    target_h_k = per_head_labels_t[hi, k]
                    ce_h = lh.sparse_categorical_crossentropy(
                        target_h_k, ignore_index=-100,
                        label_smoothing=LABEL_SMOOTHING, reduction="mean")
                    breath_loss_acc = breath_loss_acc + ce_h
                    per_head_total_losses_ss[hi] = per_head_total_losses_ss[hi] + ce_h
                ce_k = breath_loss_acc / 4.0
                losses_per_breath.append(ce_k)
                target_k = full_answer_labels if pbfa else labels_stk[k]
                pred = logits[:, :-1, :]
                if BOUNDARY_AUX_WEIGHT > 0.0:
                    x_k_b = per_breath_x[k].cast(dtypes.float)
                    B_b = x_k_b.shape[0]; T_b = x_k_b.shape[1]
                    blogits = (x_k_b @ model.boundary_head_w + model.boundary_head_b).reshape(B_b, T_b)
                    blogits_pred = blogits[:, :-1]
                    labels_k_b = target_k
                    btarget = (labels_k_b == HASH_TOKEN_ID).cast(dtypes.float)
                    valid = (labels_k_b != -100).cast(dtypes.float)
                    bce_per = blogits_pred.maximum(0.0) - blogits_pred * btarget + (1.0 + (-blogits_pred.abs()).exp()).log()
                    weight_per_pos = 1.0 + (bpw - 1.0) * btarget
                    bce_per = bce_per * weight_per_pos
                    bce_k = (bce_per * valid).sum() / (valid.sum() + 1.0)
                    boundary_losses.append(bce_k)
                continue
            pred = logits[:, :-1, :]
            # v75 multi-well supervision: every breath shares the full-answer target
            # when PER_BREATH_FULL_ANSWER=1. pbfa captured at compile time.
            target_k = full_answer_labels if pbfa else labels_stk[k]
            if _WC and _CD and model.waist_controller._last_copy_attn is not None:
                # v72 mixed CE: p_final = (1 - gate) * p_vocab + gate * p_copy
                y_target = target_k                                     # (B, T-1)
                log_p_vocab = pred.log_softmax(axis=-1)                  # (B, T-1, V)
                y_safe = y_target.maximum(0).reshape(*y_target.shape, 1)
                log_p_vocab_y = log_p_vocab.gather(-1, y_safe).reshape(y_target.shape)  # (B, T-1)
                p_vocab_y = log_p_vocab_y.exp()                          # (B, T-1)
                copy_attn_full = model.waist_controller._last_copy_attn  # (B, T, T_prompt)
                copy_attn_pred = copy_attn_full[:, :-1, :]               # (B, T-1, T_prompt)
                copy_gate_full = model.waist_controller._last_copy_gate
                copy_gate_pred = copy_gate_full[:, :-1, 0]               # (B, T-1)
                # Match mask: y_target_b_t == tokens_b_i
                match_mask = (y_target.reshape(*y_target.shape, 1) == tokens.reshape(tokens.shape[0], 1, tokens.shape[1])).cast(dtypes.float)
                p_copy_y = (copy_attn_pred * match_mask).sum(axis=-1)    # (B, T-1)
                p_final = (1.0 - copy_gate_pred) * p_vocab_y + copy_gate_pred * p_copy_y
                valid = (y_target != -100).cast(dtypes.float)
                log_p_final = (p_final + 1e-12).log()
                ce_k = -(log_p_final * valid).sum() / (valid.sum() + 1.0)
                # v72 AUX LOSS — direct supervision for copy_attn (see regular JIT for rationale).
                copy_target_sum = match_mask.sum(axis=-1, keepdim=True)         # (B, T-1, 1)
                has_match = (copy_target_sum.squeeze(-1) > 0.5).cast(dtypes.float)  # (B, T-1)
                copy_targets = match_mask / (copy_target_sum + 1e-9)            # (B, T-1, T_p)
                copy_aux_per = -(copy_targets * (copy_attn_pred + 1e-12).log()).sum(axis=-1)
                copy_aux_mask = has_match * valid                                # (B, T-1)
                copy_aux_k = (copy_aux_per * copy_aux_mask).sum() / (copy_aux_mask.sum() + 1.0)
                copy_aux_losses.append(copy_aux_k)
            else:
                ce_k = pred.sparse_categorical_crossentropy(
                    target_k, ignore_index=-100,
                    label_smoothing=LABEL_SMOOTHING, reduction="mean")
            losses_per_breath.append(ce_k)
            # v65 boundary aux with v66 pos_weight
            # v75: mask same positions as main CE.
            if BOUNDARY_AUX_WEIGHT > 0.0:
                x_k_b = per_breath_x[k].cast(dtypes.float)
                B_b = x_k_b.shape[0]; T_b = x_k_b.shape[1]
                blogits = (x_k_b @ model.boundary_head_w + model.boundary_head_b).reshape(B_b, T_b)
                blogits_pred = blogits[:, :-1]
                labels_k_b = target_k
                btarget = (labels_k_b == HASH_TOKEN_ID).cast(dtypes.float)
                valid = (labels_k_b != -100).cast(dtypes.float)
                bce_per = blogits_pred.maximum(0.0) - blogits_pred * btarget + (1.0 + (-blogits_pred.abs()).exp()).log()
                # v66: up-weight positive class to counter class imbalance
                weight_per_pos = 1.0 + (bpw - 1.0) * btarget  # 1.0 for negatives, bpw for positives
                bce_per = bce_per * weight_per_pos
                bce_k = (bce_per * valid).sum() / (valid.sum() + 1.0)
                boundary_losses.append(bce_k)

        avg_main = sum(losses_per_breath[1:], losses_per_breath[0]) / float(K)
        if BOUNDARY_AUX_WEIGHT > 0.0 and len(boundary_losses) > 0:
            avg_boundary = sum(boundary_losses[1:], boundary_losses[0]) / float(len(boundary_losses))
        else:
            avg_boundary = Tensor.zeros((), dtype=dtypes.float).contiguous()

        if aw > 0.0:
            last_mw = match_weights_list[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(
                op_labels, ignore_index=-100, reduction="mean")
        else:
            aux_ce = Tensor.zeros((), dtype=dtypes.float).contiguous()

        l2_reg = (model.lookup_table.weight.square().mean()
                  + model.lookup_table.values.square().mean()
                  + model.lookup_table.value_proj_up.square().mean()) * 1e-6
        ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                     Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
        be_reg = (model.block.breath_embed.square().mean()
                  + model.block.handoff_w.square().mean()
                  + model.block.handoff_b.square().mean()
                  + model.block.rope.pitch.square().mean()
                  + model.block.crp_mix_alpha.square().mean()
                  + model.block.crp_target_norm.square().mean()
                  + model.block.notebook_write_w.square().mean()
                  + model.block.notebook_write_b.square().mean()
                  + model.block.notebook_read_w.square().mean()
                  + model.block.notebook_read_b.square().mean()
                  + model.block.notebook_write_query.square().mean()
                  + model.block.notebook_rep_write_w.square().mean()
                  + model.block.notebook_rep_write_b.square().mean()
                  + model.block.notebook_rep_read_w.square().mean()
                  + model.block.notebook_rep_read_b.square().mean()
                  + model.block.notebook_rep_query.square().mean()
                  + model.block.bfield_proj_down.square().mean()
                  + model.block.bfield_proj_up.square().mean()
                  + model.block.bfield_bias.square().mean()
                  + model.block.waist_codebook_keys.square().mean()
                  + model.block.waist_codebook_values.square().mean()
                  + model.waist_head_w.square().mean()
                  + model.waist_head_b.square().mean()
                  + model.ln_f_g.square().mean()        # v77: ln_f params have no gradient path
                  + model.ln_f_b.square().mean()        #      when CONTROLLER_DECODE=1 AND aux_ce=0
                  + sum((p.square().mean() for lb in model.block.layers_b
                                            for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                        Tensor.zeros((), dtype=dtypes.float).contiguous())
                  # v81 multi-head MLP params: ALWAYS L2-regged for SS path too.
                  + sum((p.square().mean() for mlp in model.waist_controller.head_mlps
                                            for p in [mlp["w1"], mlp["b1"], mlp["w2"], mlp["b2"]]),
                        Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7

        # v70/v71 sparsity loss — same list-sum pattern as base JIT. v71 takes precedence.
        from mycelium.breathing import COLLAPSE_V70 as _CV70_inner_ss, COLLAPSE_V71 as _CV71_inner_ss
        if _CV71_inner_ss and model.block._collapse_v71_sparsity_list:
            _sl_ss = model.block._collapse_v71_sparsity_list
            v70_sparsity = _sl_ss[0] if len(_sl_ss) == 1 else sum(_sl_ss[1:], _sl_ss[0])
        elif _CV70_inner_ss and model.block._collapse_v70_sparsity_list:
            _sl_ss = model.block._collapse_v70_sparsity_list
            v70_sparsity = _sl_ss[0] if len(_sl_ss) == 1 else sum(_sl_ss[1:], _sl_ss[0])
        else:
            v70_sparsity = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # v72: average copy-attn aux loss across breaths; gated by WAIST_COPY at runtime.
        if _WC and copy_aux_losses:
            avg_copy_aux = sum(copy_aux_losses[1:], copy_aux_losses[0]) / float(len(copy_aux_losses))
        else:
            avg_copy_aux = Tensor.zeros((), dtype=dtypes.float).contiguous()
        # v78b ATTENTION SUPERVISION AUX LOSS — last-breath only (same as base JIT).
        # NOTE: per memory/reference_tinygrad_am_quirks.md, adding aux loss in the SS path
        # has historically hung (v72 copy aux loss). The smoke uses SCHED_SAMPLE_RATE=0
        # which routes through the base JIT path; this branch is here for symmetry only.
        if was_flag and last_cross_attn_for_aux is not None:
            cross_attn_pred = last_cross_attn_for_aux[:, :-1, :]
            log_attn = (cross_attn_pred + 1e-12).log()
            neg_ce_per_t = -(attn_target_t * log_attn).sum(axis=-1)
            attn_aux_loss = (neg_ce_per_t * attn_mask_t).sum() / (attn_mask_t.sum() + 1.0)
        else:
            attn_aux_loss = Tensor.zeros((), dtype=dtypes.float).contiguous()
        total = avg_main + aw * aux_ce + l2_reg + ch_reg + be_reg + BOUNDARY_AUX_WEIGHT * avg_boundary + v70_sparsity + _WC_AUX_W * avg_copy_aux + was_w * attn_aux_loss
        total.backward()
        healthy = total.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)
        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6))
            clip_coef = clip_coef.minimum(Tensor(1.0, dtype=dtypes.float))
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)
        opt.step()
        # v66 waist norm monitoring (same pattern as base JIT)
        if _CD and waist_compressed_per_breath:
            waist_last = waist_compressed_per_breath[-1].cast(dtypes.float)
            waist_norm = waist_last.square().mean(axis=-1).sqrt().mean()
        else:
            waist_norm = Tensor.zeros((), dtype=dtypes.float).contiguous()
        return (total.realize(), healthy.realize(), waist_norm.realize(), attn_aux_loss.realize(),
                *(ce.realize() for ce in losses_per_breath),
                # v81 per-head sums (4 values appended at the END so legacy callers can
                # slice ce_ts = outs[4:4+K] without disruption). Matches base JIT layout.
                *(ph.realize() for ph in per_head_total_losses_ss))

    _JIT_PER_BREATH_SS_CACHE[key] = _step_ss
    print(f"[JIT-SS] compiled per_breath_ss in {_t_jit.perf_counter() - _jit_compile_start:.1f}s "
          f"(cache size={len(_JIT_PER_BREATH_SS_CACHE)})", flush=True)
    return _step_ss


def _build_aux_tensors(batch_examples, tokens_np: np.ndarray, eq_token_ids):
    """Build (eq_mask, op_labels) tensors for the aux CE loss.
    eq_mask: (B, T, 1) float — 1.0 at the "=" position for valid examples, 0 elsewhere
    op_labels: (B,) int — op index 0/1/2/3, or -100 to mark "ignore in CE"
    """
    B, T = tokens_np.shape
    eq_positions = np.array([find_eq_position(tokens_np[b].tolist(), eq_token_ids)
                             for b in range(B)], dtype=np.int32)
    op_labels = np.array([op_label_from_text(ex.problem + " " + " ".join(ex.gen_targets))
                          for ex in batch_examples], dtype=np.int32)
    valid = (eq_positions >= 0) & (op_labels >= 0)
    eq_safe = np.where(valid, eq_positions, 0).astype(np.int32)
    eq_mask_np = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        if valid[b]:
            eq_mask_np[b, eq_safe[b]] = 1.0
    op_labels_masked = np.where(valid, op_labels, -100).astype(np.int32)
    eq_mask = Tensor(eq_mask_np, dtype=dtypes.float).reshape(B, T, 1).realize()
    y = Tensor(op_labels_masked, dtype=dtypes.int).realize()
    return eq_mask, y


def _compile_jit_calibration_step(model, opt, n_loops: int, n_cycles: int,
                                   fixed_len: int, B: int, max_digits: int,
                                   calibration_weight: float):
    """JIT-compile a single calibration training step.

    Inputs (all fixed-shape Tensors, prepared per cycle outside the JIT):
      For each cycle c in [0, n_cycles): tokens_c, labels_c, eq_mask_c,
        digit_select_mask_c (B, max_digits, T-1), true_digits_c (B, max_digits int),
        digit_valid_mask_c (B, max_digits float).

    Inside the JIT (unrolled over n_cycles × n_loops):
      - breathe_with_lookup forward (already JIT-friendly; uses sine-temp baseline)
      - per-breath answer-CE (mean over labels != -100 positions, computed from labels)
      - per-breath argmax → gather at digit positions via digit_select_mask
      - per-step correctness = (#matched valid digits == #valid digits)
      - per-breath confidence at eq via eq_mask
      - BCE(conf, correctness)
      - sum cycles + lookup L2 reg, backward, opt.step

    Returns:
      _step(...): the JIT'd callable. Outputs are (total_loss, cpb_acc, answer_ce_acc,
        calib_bce_acc, conf_correct_sum, conf_wrong_sum, n_correct_total, n_wrong_total).
    """
    key = (id(model), id(opt), n_loops, n_cycles, fixed_len, B, max_digits, float(calibration_weight))
    if key in _JIT_CALIB_CACHE:
        return _JIT_CALIB_CACHE[key]

    import time as _t_jit
    _jit_compile_start = _t_jit.perf_counter()
    print(f"[JIT] compile calibration step: n_loops={n_loops} n_cycles={n_cycles} "
          f"B={B} fixed_len={fixed_len} max_digits={max_digits}...", flush=True)

    T = fixed_len
    cw = float(calibration_weight)
    if n_cycles != 2:
        raise NotImplementedError(f"calibration JIT supports n_cycles=2 only (got {n_cycles})")

    @TinyJit
    def _step(
        tokens0, labels0, eq_mask0, digit_select_mask0, true_digits0, digit_valid_mask0,
        tokens1, labels1, eq_mask1, digit_select_mask1, true_digits1, digit_valid_mask1,
    ):
        opt.zero_grad()
        total_ans = Tensor.zeros((), dtype=dtypes.float).contiguous()
        total_bce = Tensor.zeros((), dtype=dtypes.float).contiguous()
        cpb_acc = Tensor.zeros((n_loops,), dtype=dtypes.float).contiguous()
        conf_correct_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        conf_wrong_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        n_correct_total = Tensor.zeros((), dtype=dtypes.float).contiguous()
        n_wrong_total = Tensor.zeros((), dtype=dtypes.float).contiguous()

        for cycle_idx in range(2):
            if cycle_idx == 0:
                tokens, labels = tokens0, labels0
                eq_mask, dsm = eq_mask0, digit_select_mask0
                td, dvm = true_digits0, digit_valid_mask0
            else:
                tokens, labels = tokens1, labels1
                eq_mask, dsm = eq_mask1, digit_select_mask1
                td, dvm = true_digits1, digit_valid_mask1

            answer_mask = (labels != -100).cast(dtypes.float)   # (B, T-1)
            answer_count = answer_mask.sum(axis=1).maximum(1.0)  # (B,)

            # Per-example step validity (at least one valid digit)
            n_valid_digits = dvm.sum(axis=1)                       # (B,) float
            step_valid = (n_valid_digits > 0).cast(dtypes.float)   # (B,)
            n_step_valid = step_valid.sum().maximum(1.0)
            td_f = td.cast(dtypes.float)

            _, _, integrated_per_breath = model.breathe_with_lookup(tokens, n_loops)

            cycle_ans_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            cycle_bce_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()

            for breath_idx in range(n_loops):
                rep = integrated_per_breath[breath_idx]
                logits = (rep @ model.embed_out).cast(dtypes.float)   # (B, T, vocab)
                pred = logits[:, :-1, :]                               # (B, T-1, vocab)

                per_tok_ce = pred.sparse_categorical_crossentropy(
                    labels, ignore_index=-100, label_smoothing=LABEL_SMOOTHING, reduction="none"
                )                                                       # (B, T-1)
                per_ex_ans = (per_tok_ce * answer_mask).sum(axis=1) / answer_count
                cycle_ans_sum = cycle_ans_sum + per_ex_ans.mean()

                # Per-step correctness via tensor gather
                argmax = pred.argmax(axis=-1).cast(dtypes.float)        # (B, T-1)
                argmax_b = argmax.reshape(B, 1, T - 1)                  # (B, 1, T-1)
                argmax_at_digits = (argmax_b * dsm).sum(axis=2)         # (B, max_digits)
                match = (argmax_at_digits == td_f).cast(dtypes.float)
                matched_valid = (match * dvm).sum(axis=1)               # (B,) — #valid digits matched
                # all-correct iff matched_valid == n_valid_digits (and n_valid_digits > 0)
                all_correct = (matched_valid >= n_valid_digits).cast(dtypes.float)
                correctness = all_correct * step_valid                  # (B,)

                # Confidence at eq position
                rep_f = rep.cast(dtypes.float)
                gathered = (rep_f * eq_mask).sum(axis=1)                # (B, hidden)
                conf = model.confidence_head(gathered)                  # (B,)

                # BCE
                c = conf.clamp(1e-7, 1.0 - 1e-7)
                bce = -(correctness * c.log() + (1.0 - correctness) * (1.0 - c).log())
                bce_step = (bce * step_valid).sum() / n_step_valid
                cycle_bce_sum = cycle_bce_sum + bce_step

                # Diagnostics
                # cpb: count correct steps per breath (across cycles, accumulator)
                n_cor_this = (correctness * step_valid).sum()           # scalar
                cpb_one_hot = Tensor.zeros((n_loops,), dtype=dtypes.float).contiguous()
                # One-hot trick: build a (n_loops,) tensor with 1 at position breath_idx
                # tinygrad doesn't have nice indexed-assignment; use arange trick
                idx_t = Tensor.arange(n_loops, dtype=dtypes.float)
                mask_breath = (idx_t == float(breath_idx)).cast(dtypes.float)
                cpb_acc = cpb_acc + mask_breath * n_cor_this

                # Confidence sums
                conf_correct_sum = conf_correct_sum + (conf * correctness).sum()
                conf_wrong_sum = conf_wrong_sum + (conf * (1.0 - correctness) * step_valid).sum()
                n_correct_total = n_correct_total + (correctness).sum()
                n_wrong_total = n_wrong_total + ((1.0 - correctness) * step_valid).sum()

            cycle_ans = cycle_ans_sum / float(n_loops)
            cycle_bce = cycle_bce_sum / float(n_loops)
            total_ans = total_ans + cycle_ans
            total_bce = total_bce + cycle_bce

        avg_ans = total_ans / float(n_cycles)
        avg_bce = total_bce / float(n_cycles)
        l2_reg = model.lookup_table.weight.square().mean() * 1e-6
        total_loss = avg_ans + cw * avg_bce + l2_reg

        total_loss.backward()
        opt.step()
        return (
            total_loss.realize(),
            cpb_acc.realize(),
            avg_ans.realize(),
            avg_bce.realize(),
            conf_correct_sum.realize(),
            conf_wrong_sum.realize(),
            n_correct_total.realize(),
            n_wrong_total.realize(),
        )

    _JIT_CALIB_CACHE[key] = _step
    print(f"[JIT] calibration step compiled in {_t_jit.perf_counter() - _jit_compile_start:.1f}s "
          f"(cache size={len(_JIT_CALIB_CACHE)})", flush=True)
    return _step


def masked_forward_loss(model, tokens: Tensor, labels: Tensor, n_loops: int,
                        use_jit: bool = False) -> Tensor:
    """Next-token CE loss, ignoring positions where label == -100.

    tokens: (B, T) int — full input
    labels: (B, T-1) int — targets for tokens[:, 1:], with -100 in masked positions

    use_jit=False by default (eager path). The JIT-forward path (call_jit) is
    8.8× faster on n_loops=8 BUT breaks backward — TinyJit captures the forward
    compute into a fused op whose output tensors are leaves in the autograd
    graph, so backward stops at the JIT boundary and transformer params end up
    with grad=None. Use the JIT methods (model.call_jit, model.breathe_with_lookup_jit)
    only in inference / diagnostic paths where no backward is needed. Full
    JIT'd training (forward + backward + opt.step in one TinyJit) is the right
    fix and is tracked as a follow-up.
    """
    h = model.call_jit(tokens, n_loops) if use_jit else model(tokens, n_loops)
    logits = (h @ model.embed_out).cast(dtypes.float)          # (B, T, vocab)
    pred = logits[:, :-1, :]                                   # (B, T-1, vocab)
    ls = LABEL_SMOOTHING if Tensor.training else 0.0
    return pred.sparse_categorical_crossentropy(labels, ignore_index=-100, label_smoothing=ls, reduction="mean")


def masked_forward_loss_with_lookup(model, tokens: Tensor, labels: Tensor,
                                    n_loops: int, use_jit: bool = False):
    """Combined main-CE forward that also returns per-breath lookup match weights.
    One forward pass instead of two — used when joint lookup-aux training is on,
    halving the main-step compute relative to running plain forward + a separate
    breathe_with_lookup for the aux loss.

    use_jit=False by default — see masked_forward_loss's docstring for the
    backward-through-JIT issue. The JIT method is available for inference paths.

    Returns (main_ce_loss, match_weights) where match_weights is a list of
    per-breath (B, T, n_entries) tensors.
    """
    if use_jit:
        final_h, last_mw = model.breathe_with_lookup_jit(tokens, n_loops)
        match_weights = [last_mw]
    else:
        final_h, match_weights, _ = model.breathe_with_lookup(tokens, n_loops)
    logits = (final_h @ model.embed_out).cast(dtypes.float)
    pred = logits[:, :-1, :]
    ls = LABEL_SMOOTHING if Tensor.training else 0.0
    main_ce = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, label_smoothing=ls, reduction="mean")
    return main_ce, match_weights


def train_step(model, opt, tokens: Tensor, labels: Tensor, n_loops: int) -> float:
    Tensor.training = True
    opt.zero_grad()
    loss = masked_forward_loss(model, tokens, labels, n_loops)
    loss.backward()
    opt.step()
    Device[Device.DEFAULT].synchronize()
    return float(loss.numpy())


def eval_loss(model, tokens: Tensor, labels: Tensor, n_loops: int) -> float:
    Tensor.training = False
    loss = masked_forward_loss(model, tokens, labels, n_loops)
    return float(loss.realize().numpy())


def _resolve_loops_per_cycle(n_loops, n_cycles: int) -> List[int]:
    """Accept either a single int (same loops per cycle) or a list (one per cycle).
    Pads/truncates a list to length n_cycles. Used to support three-phase scheduling
    where cycle 0 (Phase A) gets heavy breathing and later cycles (Phase C) get light.
    """
    if isinstance(n_loops, int):
        return [n_loops] * n_cycles
    out = list(n_loops)
    if len(out) < n_cycles:
        out = out + [out[-1]] * (n_cycles - len(out))
    return out[:n_cycles]


def controller_train_step(model, ctrl_opt, batch_examples: List[MathExample], tok,
                          eq_token_ids, max_loops: int = 8,
                          n_classes: int = 4, profile: bool = False,
                          compute_penalty: float = 0.0,
                          stop_calib_weight: float = 0.01):
    """Train ONLY the controller via lookup-CE loss on op classification.

    Forwards through breathe_controlled with decisions NOT detached, so the
    gradient from the auxiliary CE loss flows back through every breath's
    decisions into controller params. transformer params receive gradient too
    (the loss reaches them via the breathing path) but are NOT updated because
    ctrl_opt only contains controller_parameters() — those grads are simply
    discarded on the next main_opt.zero_grad().

    Loss: cross-entropy on op label using match_weights at the eq position
    averaged over all breaths (encourages every breath, not just the final
    one, to produce op-discriminable representations).

    Returns the scalar loss value.
    """
    from mycelium.controller import Notebook
    import time as _time
    Tensor.training = True

    if profile: _t = _time.perf_counter()
    cycles_per_ex = [encode_cycles(tok, ex) for ex in batch_examples]
    encoded = [ex_cycles[0] for ex_cycles in cycles_per_ex]      # cycle 0 only
    tokens_np, labels_np = collate(encoded, fixed_len=64 + 40)     # capture labels for answer-CE
    B, T = tokens_np.shape
    tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
    labels = Tensor(labels_np, dtype=dtypes.int).realize()         # (B, T-1) next-token targets, -100 mask
    answer_mask = Tensor((labels_np != -100).astype(np.float32), dtype=dtypes.float).realize()  # (B, T-1)
    encode_time = (_time.perf_counter() - _t) if profile else 0.0

    eq_positions = np.array([find_eq_position(tokens_np[b].tolist(), eq_token_ids)
                             for b in range(B)], dtype=np.int32)
    op_labels = np.array([op_label_from_text(ex.problem + " " + " ".join(ex.gen_targets))
                          for ex in batch_examples], dtype=np.int32)
    valid = (eq_positions >= 0) & (op_labels >= 0)
    if int(valid.sum()) == 0:
        return (0.0, {"encode": encode_time, "forward": 0.0, "backward_step": 0.0}) if profile else 0.0

    if profile: _t = _time.perf_counter()
    # Forward through the closed loop. Decisions NOT detached so gradient flows back.
    # return_per_breath_reps=True so we can compute per-breath answer-prediction CE
    # (the signal that actually differentiates problem difficulty, vs the previous
    # op-classification CE which is trivially solved by the operator symbol in the
    # prompt). See project_arith_mixed_v3_result.md.
    notebook = Notebook()
    _, decisions, n_breaths, match_weights, integrated_per_breath = model.breathe_controlled(
        tokens, max_loops=max_loops, notebook=notebook,
        detach_rep_for_ctrl=False, detach_decisions_into_transformer=False,
        return_per_breath_reps=True,
    )

    # Gather match weights at eq positions for every breath, mean across breaths.
    eq_safe = np.where(valid, eq_positions, 0).astype(np.int32)
    eq_mask_np = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        if valid[b]:
            eq_mask_np[b, eq_safe[b]] = 1.0
    eq_mask = Tensor(eq_mask_np, dtype=dtypes.float).reshape(B, T, 1).realize()

    op_labels_masked = np.where(valid, op_labels, -100).astype(np.int32)
    y = Tensor(op_labels_masked, dtype=dtypes.int).realize()

    # Per-breath OP-classification CE (the existing main signal — still useful for
    # training the lookup table). reduction="none" gives (B,) per breath.
    per_breath_per_ex_op_losses = []
    for mw in match_weights:
        gathered = (mw.cast(dtypes.float) * eq_mask).sum(axis=1)        # (B, n_entries)
        op_logits = gathered[:, :n_classes] * 10.0
        per_ex_op = op_logits.sparse_categorical_crossentropy(y, ignore_index=-100, reduction="none")  # (B,)
        per_breath_per_ex_op_losses.append(per_ex_op)

    # Per-breath PER-EXAMPLE ANSWER-prediction CE (the new stop_calib target).
    # The prior implementation used op-classification CE — trivially solved in
    # single-cycle arithmetic because the operator symbol is in the prompt. This
    # meant per-example targets were uniformly small across the batch and the stop
    # head couldn't learn per-problem differentiation (v3 instrumentation showed
    # within-batch std going DOWN with more training: 0.47 → 0.27). Answer-prediction
    # CE actually varies by problem difficulty: easy = low CE early, hard = high CE
    # until late breaths.
    per_breath_per_ex_answer_losses = []
    for rep in integrated_per_breath:
        full_logits = (rep @ model.embed_out).cast(dtypes.float)           # (B, T, vocab)
        pred = full_logits[:, :-1, :]                                       # next-token prediction
        per_tok_ce = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="none")  # (B, T-1)
        per_ex_ans = (per_tok_ce * answer_mask).sum(axis=1) / answer_mask.sum(axis=1).maximum(1.0)      # (B,) mean over answer positions
        per_breath_per_ex_answer_losses.append(per_ex_ans)

    # Main ctrl loss: average over breaths (op-CE signal — keeps lookup table sharp).
    total = per_breath_per_ex_op_losses[0].mean()
    for l in per_breath_per_ex_op_losses[1:]:
        total = total + l.mean()
    avg_ctrl_loss = total / float(len(per_breath_per_ex_op_losses))

    # Stop calibration: supervise stop_logit against -answer_loss (per example).
    # Easy problems: answer_loss low → target ~0 → stop_logit pulled up → "stop OK."
    # Hard problems: answer_loss high → target very negative → stop_logit pulled down → "keep going."
    # This is the signal that actually differentiates problems — the v1-v3 plan's
    # op-CE target couldn't.
    stop_calib = None
    for i, (d, pb_loss) in enumerate(zip(decisions[1:], per_breath_per_ex_answer_losses)):
        target = pb_loss * -1.0                          # (B,) — per example
        diff = d["stop_logit"] - target.detach()         # (B,)
        term = diff.square().mean()
        stop_calib = term if stop_calib is None else stop_calib + term
    if stop_calib is not None:
        avg_ctrl_loss = avg_ctrl_loss + stop_calib * stop_calib_weight

    # Compute penalty (optional). ReLU(-stop_logit) per breath: when stop_logit is
    # negative ("keep going"), penalty is |stop_logit|; when positive ("stop"), zero.
    # Combined with per-example calibration this AMPLIFIES easy/hard differentiation:
    # easy problems get both forces pushing stop_logit up; hard problems have
    # calibration pulling down vs. penalty pushing up, net stays modestly negative.
    if compute_penalty > 0.0:
        cp_total = None
        for d in decisions[1:]:
            sl = d["stop_logit"]                          # (B,)
            term = (-sl).maximum(0.0).mean()              # ReLU(-stop_logit) — push positive
            cp_total = term if cp_total is None else cp_total + term
        if cp_total is not None:
            avg_ctrl_loss = avg_ctrl_loss + cp_total * compute_penalty

    # Tiny L2 reg on the step_w/b params so they get a defined gradient. step_mult
    # is consumed via float(...) inside the loop (non-differentiable rounding into
    # the integer-indexed RoPE table), so no other path reaches it.
    avg_ctrl_loss = avg_ctrl_loss + (model.controller.step_w.square().mean()
                                     + model.controller.step_b.square().mean()) * 1e-6

    if profile: py_overhead = _time.perf_counter() - _t
    if profile: _t2 = _time.perf_counter()
    ctrl_opt.zero_grad()
    avg_ctrl_loss.backward()
    ctrl_opt.step()
    Device[Device.DEFAULT].synchronize()
    if profile: gpu_compute = _time.perf_counter() - _t2

    loss_val = float(avg_ctrl_loss.numpy())
    if profile:
        return loss_val, {
            "encode": encode_time,
            "py_overhead": py_overhead,
            "gpu_compute": gpu_compute,
        }
    return loss_val


def calibration_train_step(model, opt, batch_examples: List[MathExample], tok,
                           digit_token_ids: set, eq_token_ids: list,
                           n_loops: int = 8, fixed_len: int = 128,
                           calibration_weight: float = 0.1,
                           profile: bool = False, use_jit: bool = False,
                           max_digits: int = 4):
    """Per-cycle optimal-stopping calibration training.

    Multi-cycle aware (mirrors multi_cycle_train_step's encoding). For each
    outer cycle:
      - One forward pass via breathe_controlled (return_per_breath_reps=True)
      - The cycle's target contains exactly one "=" (a single step)
      - Per breath: answer-CE on the answer-digit positions in the target;
        BCE(conf, argmax-correct) at the eq position
    Losses are summed across cycles (equal-reward decomposition).

    Why multi-cycle, not single-cycle: training distribution must match the
    eval distribution. Single-cycle encoding causes catastrophic forgetting of
    multi-cycle generation. See project_calibration_v3_failed.md.

    Args:
      digit_token_ids: from calibration.digit_token_ids_for(tok)
      eq_token_ids: from lookup_table.eq_token_ids_for(tok)
      n_loops: max breaths per cycle forward pass
      fixed_len: padded sequence length
      calibration_weight: λ for the BCE-calibration term (0 = pure answer-CE)

    Returns dict with diagnostics aggregated across cycles:
      loss, answer_ce, calib_bce, correct_per_breath, mean_conf_correct,
      mean_conf_wrong, n_correct, n_wrong, n_valid_steps.
    """
    from mycelium.controller import Notebook
    Tensor.training = True

    eq_set = set(eq_token_ids)
    cycles_per_ex = [encode_cycles(tok, ex) for ex in batch_examples]
    n_cycles = len(cycles_per_ex[0])
    B = len(batch_examples)

    # JIT fast path — prepare fixed-shape per-cycle tensors and call the compiled step.
    if use_jit and n_cycles == 2:
        T = fixed_len
        cycle_tensors = []
        for cycle_idx in range(n_cycles):
            encoded = [ex_cycles[cycle_idx] for ex_cycles in cycles_per_ex]
            tokens_np, labels_np = collate(encoded, fixed_len=T)
            eq_mask_np = np.zeros((B, T), dtype=np.float32)
            dsm_np = np.zeros((B, max_digits, T - 1), dtype=np.float32)
            td_np = np.zeros((B, max_digits), dtype=np.int32)
            dvm_np = np.zeros((B, max_digits), dtype=np.float32)
            for b in range(B):
                _, prefix_len, total_len = cycles_per_ex[b][cycle_idx]
                scan_end = min(T, total_len)
                target_eq = -1
                for i in range(prefix_len, scan_end):
                    if int(tokens_np[b, i]) in eq_set:
                        target_eq = i
                        break
                if target_eq < 0:
                    continue
                eq_mask_np[b, target_eq] = 1.0
                i = target_eq + 1
                d = 0
                while i < scan_end and d < max_digits and int(tokens_np[b, i]) in digit_token_ids:
                    pp = i - 1
                    if 0 <= pp < T - 1:
                        dsm_np[b, d, pp] = 1.0
                        td_np[b, d] = int(tokens_np[b, i])
                        dvm_np[b, d] = 1.0
                    i += 1
                    d += 1
            cycle_tensors.append((
                Tensor(tokens_np, dtype=dtypes.int).realize(),
                Tensor(labels_np, dtype=dtypes.int).realize(),
                Tensor(eq_mask_np, dtype=dtypes.float).reshape(B, T, 1).realize(),
                Tensor(dsm_np, dtype=dtypes.float).realize(),
                Tensor(td_np, dtype=dtypes.int).realize(),
                Tensor(dvm_np, dtype=dtypes.float).realize(),
            ))

        jit_step = _compile_jit_calibration_step(model, opt, n_loops, n_cycles,
                                                  fixed_len, B, max_digits, calibration_weight)
        (total_loss_t, cpb_t, ans_t, bce_t,
         conf_corr_t, conf_wr_t, n_cor_t, n_wr_t) = jit_step(
            *cycle_tensors[0], *cycle_tensors[1]
        )
        n_correct = float(n_cor_t.numpy())
        n_wrong = float(n_wr_t.numpy())
        mean_conf_correct = (float(conf_corr_t.numpy()) / n_correct) if n_correct > 0 else float("nan")
        mean_conf_wrong = (float(conf_wr_t.numpy()) / n_wrong) if n_wrong > 0 else float("nan")
        return {
            "loss": float(total_loss_t.numpy()),
            "answer_ce": float(ans_t.numpy()),
            "calib_bce": float(bce_t.numpy()),
            "correct_per_breath": cpb_t.numpy().tolist(),
            "n_valid_steps": int(n_correct + n_wrong),
            "mean_conf_correct": mean_conf_correct,
            "mean_conf_wrong": mean_conf_wrong,
            "n_correct": int(n_correct),
            "n_wrong": int(n_wrong),
        }

    # Eager path
    opt.zero_grad()

    cycle_answer_losses = []
    cycle_calib_losses = []
    cpb_aggregate = None
    n_correct_total = 0
    n_wrong_total = 0
    correct_conf_sum = 0.0
    wrong_conf_sum = 0.0
    n_valid_steps_total = 0

    for cycle_idx in range(n_cycles):
        encoded = [ex_cycles[cycle_idx] for ex_cycles in cycles_per_ex]
        tokens_np, labels_np = collate(encoded, fixed_len=fixed_len)
        T = tokens_np.shape[1]

        # Per-example: find the eq position in the TARGET (i.e., at idx >= prefix_len)
        # and the digit run that follows. One step per cycle (each cycle target has
        # exactly one "=").
        per_ex_info = []   # list of (eq_pos, digit_positions, true_digit_ids) or None
        for b in range(B):
            _, prefix_len, total_len = cycles_per_ex[b][cycle_idx]
            target_eq = -1
            scan_end = min(T, total_len)
            for i in range(prefix_len, scan_end):
                if int(tokens_np[b, i]) in eq_set:
                    target_eq = i
                    break
            if target_eq < 0:
                per_ex_info.append(None)
                continue
            digit_positions = []
            true_digit_ids = []
            i = target_eq + 1
            while i < scan_end and len(digit_positions) < 4 and int(tokens_np[b, i]) in digit_token_ids:
                digit_positions.append(i)
                true_digit_ids.append(int(tokens_np[b, i]))
                i += 1
            if digit_positions:
                per_ex_info.append((target_eq, digit_positions, true_digit_ids))
            else:
                per_ex_info.append(None)

        # Validity vector & eq mask
        step_valid_np = np.array(
            [1.0 if info is not None else 0.0 for info in per_ex_info],
            dtype=np.float32,
        )
        n_valid_this_cycle = int(step_valid_np.sum())
        n_valid_steps_total += n_valid_this_cycle
        if n_valid_this_cycle == 0:
            continue   # skip cycle if no valid eqs (shouldn't happen for L4)

        eq_mask_np = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if per_ex_info[b] is not None:
                eq_mask_np[b, per_ex_info[b][0]] = 1.0

        # Answer-mask: supervise EVERY target token (matches multi_cycle_train_step).
        # Earlier v4 limited supervision to digit positions only — caught the bug
        # that non-digit positions (operators, separators, spaces) got no training
        # signal and the model produced garbage there at autoregressive eval. With
        # the full labels mask, the standard generation distribution is preserved.
        # See project_calibration_v3_failed.md and v4 = 0/1/0 result.
        answer_mask_np = (labels_np != -100).astype(np.float32)

        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        labels = Tensor(labels_np, dtype=dtypes.int).realize()
        eq_mask = Tensor(eq_mask_np, dtype=dtypes.float).reshape(B, T, 1).realize()
        answer_mask = Tensor(answer_mask_np, dtype=dtypes.float).realize()
        step_valid_t = Tensor(step_valid_np, dtype=dtypes.float).realize()
        n_valid_t = step_valid_t.sum().maximum(1.0)

        # Forward — per-breath integrated reps. Decisions detached so controller
        # path doesn't contaminate main-grad updates.
        notebook = Notebook()
        _, _decisions, _n_breaths_actual, _match_weights, integrated_per_breath = model.breathe_controlled(
            tokens, max_loops=n_loops, notebook=notebook,
            detach_rep_for_ctrl=True,
            detach_decisions_into_transformer=True,
            adaptive=False,
            return_per_breath_reps=True,
        )
        n_breaths = len(integrated_per_breath)

        cycle_ans_sum = None
        conf_per_breath: list = []
        argmax_per_breath_tensors: list = []

        for rep in integrated_per_breath:
            logits = (rep @ model.embed_out).cast(dtypes.float)
            pred = logits[:, :-1, :]
            ls = LABEL_SMOOTHING if Tensor.training else 0.0
            per_tok_ce = pred.sparse_categorical_crossentropy(
                labels, ignore_index=-100, label_smoothing=ls, reduction="none"
            )
            per_ex_ans = (per_tok_ce * answer_mask).sum(axis=1) / answer_mask.sum(axis=1).maximum(1.0)
            if cycle_ans_sum is None:
                cycle_ans_sum = per_ex_ans.mean()
            else:
                cycle_ans_sum = cycle_ans_sum + per_ex_ans.mean()
            argmax_per_breath_tensors.append(pred.argmax(axis=-1))
            rep_f = rep.cast(dtypes.float)
            gathered = (rep_f * eq_mask).sum(axis=1)   # (B, hidden)
            conf_per_breath.append(model.confidence_head(gathered))   # (B,)

        cycle_ans = cycle_ans_sum / float(n_breaths)
        cycle_answer_losses.append(cycle_ans)

        # Sync argmaxes once for correctness
        all_argmax = Tensor.stack(*argmax_per_breath_tensors, dim=0).numpy()  # (n_breaths, B, T-1)
        correctness_np = np.zeros((n_breaths, B), dtype=np.float32)
        for breath_idx in range(n_breaths):
            for b in range(B):
                info = per_ex_info[b]
                if info is None:
                    continue
                _, digit_positions, true_digit_ids = info
                all_correct = True
                for j, dp in enumerate(digit_positions):
                    pp = dp - 1
                    if pp < 0 or pp >= all_argmax.shape[2]:
                        all_correct = False
                        break
                    if int(all_argmax[breath_idx, b, pp]) != int(true_digit_ids[j]):
                        all_correct = False
                        break
                correctness_np[breath_idx, b] = 1.0 if all_correct else 0.0

        # BCE per breath
        correctness_t = Tensor(correctness_np, dtype=dtypes.float).realize()
        eps_bce = 1e-7
        cycle_bce_sum = None
        for breath_idx, conf in enumerate(conf_per_breath):
            target = correctness_t[breath_idx]   # (B,)
            c = conf.clamp(eps_bce, 1.0 - eps_bce)
            bce = -(target * c.log() + (1.0 - target) * (1.0 - c).log())   # (B,)
            bce_masked = bce * step_valid_t
            breath_loss = bce_masked.sum() / n_valid_t
            cycle_bce_sum = breath_loss if cycle_bce_sum is None else cycle_bce_sum + breath_loss
        cycle_bce = cycle_bce_sum / float(n_breaths)
        cycle_calib_losses.append(cycle_bce)

        # Per-cycle diagnostics → aggregate
        cycle_cpb = correctness_np.sum(axis=1)   # (n_breaths,)
        if cpb_aggregate is None:
            cpb_aggregate = cycle_cpb.copy()
        elif len(cpb_aggregate) == len(cycle_cpb):
            cpb_aggregate = cpb_aggregate + cycle_cpb

        conf_stack_np = Tensor.stack(*conf_per_breath, dim=0).numpy()   # (n_breaths, B)
        valid_bcast = np.broadcast_to(step_valid_np[None, :] > 0, conf_stack_np.shape)
        correct_mask = (correctness_np > 0) & valid_bcast
        wrong_mask = (correctness_np == 0) & valid_bcast
        n_correct_total += int(correct_mask.sum())
        n_wrong_total += int(wrong_mask.sum())
        if int(correct_mask.sum()) > 0:
            correct_conf_sum += float(conf_stack_np[correct_mask].sum())
        if int(wrong_mask.sum()) > 0:
            wrong_conf_sum += float(conf_stack_np[wrong_mask].sum())

    if not cycle_answer_losses:
        return {"loss": 0.0, "answer_ce": 0.0, "calib_bce": 0.0,
                "correct_per_breath": [], "n_valid_steps": 0,
                "mean_conf_correct": float("nan"), "mean_conf_wrong": float("nan"),
                "n_correct": 0, "n_wrong": 0}

    avg_answer_ce = cycle_answer_losses[0]
    for ce in cycle_answer_losses[1:]:
        avg_answer_ce = avg_answer_ce + ce
    avg_answer_ce = avg_answer_ce / float(len(cycle_answer_losses))

    avg_calib_bce = cycle_calib_losses[0]
    for bce in cycle_calib_losses[1:]:
        avg_calib_bce = avg_calib_bce + bce
    avg_calib_bce = avg_calib_bce / float(len(cycle_calib_losses))

    total_loss = avg_answer_ce + calibration_weight * avg_calib_bce
    # Lookup-table L2 reg (mirrors multi_cycle_train_step)
    total_loss = total_loss + model.lookup_table.weight.square().mean() * 1e-6

    total_loss.backward()
    opt.step()
    Device[Device.DEFAULT].synchronize()

    correct_per_breath = cpb_aggregate.tolist() if cpb_aggregate is not None else []
    mean_conf_correct = (correct_conf_sum / n_correct_total) if n_correct_total > 0 else float("nan")
    mean_conf_wrong = (wrong_conf_sum / n_wrong_total) if n_wrong_total > 0 else float("nan")

    return {
        "loss": float(total_loss.numpy()),
        "answer_ce": float(avg_answer_ce.numpy()),
        "calib_bce": float(avg_calib_bce.numpy()),
        "correct_per_breath": correct_per_breath,
        "n_valid_steps": n_valid_steps_total,
        "mean_conf_correct": mean_conf_correct,
        "mean_conf_wrong": mean_conf_wrong,
        "n_correct": n_correct_total,
        "n_wrong": n_wrong_total,
    }


def _aux_loss_from_match_weights(match_weights, tokens_np: np.ndarray,
                                  batch_examples: List[MathExample],
                                  eq_token_id, n_classes: int = 4) -> Tensor | None:
    """Compute lookup-CE aux loss from already-available per-breath match_weights
    (no extra forward). Used by the shared-forward fast path."""
    B, T = tokens_np.shape
    eq_positions = np.array([find_eq_position(tokens_np[b].tolist(), eq_token_id)
                             for b in range(B)], dtype=np.int32)
    op_labels = np.array([op_label_from_text(ex.problem + " " + " ".join(ex.gen_targets))
                          for ex in batch_examples], dtype=np.int32)
    valid = (eq_positions >= 0) & (op_labels >= 0)
    if int(valid.sum()) == 0:
        return None

    last_mw = match_weights[-1]
    eq_safe = np.where(valid, eq_positions, 0).astype(np.int32)
    eq_mask_np = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        if valid[b]:
            eq_mask_np[b, eq_safe[b]] = 1.0
    eq_mask = Tensor(eq_mask_np, dtype=dtypes.float).reshape(B, T, 1).realize()
    gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
    logits = gathered[:, :n_classes] * 10.0
    op_labels_masked = np.where(valid, op_labels, -100).astype(np.int32)
    y = Tensor(op_labels_masked, dtype=dtypes.int).realize()
    return logits.sparse_categorical_crossentropy(y, ignore_index=-100, reduction="mean")


def _lookup_aux_loss(model, tokens: Tensor, tokens_np: np.ndarray,
                     batch_examples: List[MathExample], n_loops: int,
                     eq_token_id: int, n_classes: int = 4) -> Tensor | None:
    """Auxiliary CE loss on the model's lookup table at the "=" token position.

    Returns a 0-D loss tensor (mean CE over examples where eq position + op label
    are both valid), or None if no example in the batch has a usable eq+op pair.

    Uses breathe_with_lookup so the gradient flows through the same forward as
    the caller's main loss when chained together.
    """
    B, T = tokens_np.shape
    eq_positions = np.array([find_eq_position(tokens_np[b].tolist(), eq_token_id)
                             for b in range(B)], dtype=np.int32)
    op_labels = np.array([op_label_from_text(ex.problem + " " + " ".join(ex.gen_targets))
                          for ex in batch_examples], dtype=np.int32)
    valid = (eq_positions >= 0) & (op_labels >= 0)
    if int(valid.sum()) == 0:
        return None

    _, match_weights, _ = model.breathe_with_lookup(tokens, n_loops)
    last_mw = match_weights[-1]                                    # (B, T, n_entries)

    # Gather match_weights[b, eq_positions[b], :] via mask + sum
    eq_safe = np.where(valid, eq_positions, 0).astype(np.int32)    # avoid -1 indexing
    eq_mask_np = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        if valid[b]:
            eq_mask_np[b, eq_safe[b]] = 1.0
    eq_mask = Tensor(eq_mask_np, dtype=dtypes.float).reshape(B, T, 1).realize()
    gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)  # (B, n_entries)

    # CE on first n_classes entries, ignore_index=-100 masks invalid examples
    logits = gathered[:, :n_classes] * 10.0                        # temperature scaling matches diagnostic
    op_labels_masked = np.where(valid, op_labels, -100).astype(np.int32)
    y = Tensor(op_labels_masked, dtype=dtypes.int).realize()
    return logits.sparse_categorical_crossentropy(y, ignore_index=-100, reduction="mean")


def per_breath_train_step(model, opt, batch_examples: List[MathExample], tok,
                          fixed_len: int,
                          lookup_aux_weight: float = 0.0,
                          lookup_eq_token_id=None,
                          use_jit: bool = False,
                          grad_clip: float = 0.0,
                          step_idx: int = 0):
    """v52 Stage 1: per-breath supervision.

    Drops the outer cycle structure entirely. For a K-step problem, runs ONE
    forward pass with n_loops=K breaths. Each breath's end-of-breath output
    is decoded via ln_f + embed_out (or WaistController if CONTROLLER_DECODE)
    and supervised against THAT step's tokens.

    Result: A=1 (single breath) can only solve step 1; K-step problems need
    exactly K breaths. Depth-helps signal becomes architecturally required.

    use_jit=True dispatches to a JIT'd version (forward + backward + opt.step
    fused). Eliminates the linear per-step time growth of the eager path.

    Assumes uniform K across the batch (use single-level training for Stage 1).

    step_idx: used to compute the linear scheduled-sampling ramp (v66).
    """
    global _waist_norm_low_streak
    from mycelium.breathing import _layernorm

    Tensor.training = True

    # v77 DAG-layered supervision path (2026-05-24).
    # When V77_DAG_TRAINING=1, batch_examples are V77Examples (not MathExample);
    # we build a layered per-breath supervision target from per_layer_target.
    # The number of breaths K is FIXED at V77_N_LAYERS (default 6) across all problems.
    # Input answer span = LAST layer's text (the pure DAG) + EOS — the model's
    # final goal. Per-breath labels: breath b is supervised against layer-b's
    # tokens placed at positions [prompt_len-1 .. prompt_len-1+len(L_b)-1] in
    # the labels tensor; -100 elsewhere. Where L_b is longer than the input span
    # (rare; happens when L3 verbal description exceeds L5's DAG length), labels
    # extend up to fixed_len-1 — the model is still supervised correctly even
    # if input tokens at those positions are PAD (post-EOS), since labels still
    # carry the gold token-id and the gradient flows through the breath's hidden
    # state, which doesn't strictly require teacher-forced input alignment.
    if V77_DAG_TRAINING:
        if V77Example is None:
            raise RuntimeError("V77_DAG_TRAINING=1 but V77Example failed to import")
        K = V77_N_LAYERS
        assert all(isinstance(ex, V77Example) for ex in batch_examples), (
            "V77_DAG_TRAINING=1 requires V77Example batch entries (use load_gsm8k_v77)")
        assert all(len(ex.per_layer_target) == K for ex in batch_examples), (
            f"V77_DAG_TRAINING expects {K} layers per example; got mixed.")

        B = len(batch_examples)
        # Per-example: tokenize prompt, tokenize each layer (with a leading space),
        # build the canonical input as prompt + L_{K-1} (the DAG) + EOS, and per-
        # breath label sequences with layer-b tokens at the answer span.
        tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
        per_step_labels_np = np.full((K, B, fixed_len - 1), -100, dtype=np.int32)
        # v81 (2026-05-26) per-head labels for the MULTI_HEAD_WAIST path.
        # Shape (4_heads, K, B, fixed_len - 1). Always built so the JIT signature stays
        # stable; the JIT body picks per-head vs single-head based on MULTI_HEAD_WAIST
        # at compile time. The 4 head order matches WaistController.head_names:
        # ops=0, types=1, args1=2, args2=3.
        per_head_per_step_labels_np = np.full((4, K, B, fixed_len - 1), -100, dtype=np.int32)
        prompt_lens = []
        max_layer_lens = []
        eos_id = 0  # Pythia EOS == 0 in our tokenizer wrapping
        for b, ex in enumerate(batch_examples):
            p_ids = tok.encode(ex.problem).ids
            prompt_len = len(p_ids)
            # Tokenize each layer with a leading space; this matches encode_cycles' convention.
            per_layer_ids = [tok.encode(" " + ex.per_layer_target[ell]).ids for ell in range(K)]
            # Canonical input answer = the FINAL layer (= the DAG). This is what
            # the model is ultimately trying to emit at breath K-1.
            final_layer_ids = per_layer_ids[-1]
            full_ids = list(p_ids) + list(final_layer_ids) + [eos_id]
            full_ids = full_ids[:fixed_len]
            tokens_np[b, :len(full_ids)] = full_ids
            prompt_lens.append(prompt_len)
            max_layer_lens.append(max(len(ids) for ids in per_layer_ids))

            # Per-breath labels: layer-b's tokens at positions [prompt_len-1 .. ).
            # position p in the labels tensor corresponds to predicting tokens_np[b, p+1]
            # given tokens_np[b, :p+1]. So labels[k, b, prompt_len-1] = first token of layer k.
            for ell in range(K):
                layer_ids = per_layer_ids[ell]
                # Place layer-b's tokens starting at label index prompt_len - 1.
                for j, tid in enumerate(layer_ids):
                    label_pos = (prompt_len - 1) + j
                    if 0 <= label_pos < fixed_len - 1:
                        per_step_labels_np[ell, b, label_pos] = tid

            # v81 per-head labels — tokenize each of the 4 lists in L<ell> separately,
            # place each list's tokens at the corresponding head's positions, -100 elsewhere.
            # Layout: " " + ops_str + " | " + types_str + " | " + args1_str + " | " + args2_str.
            # We tokenize: " " + ops, " | " + types, " | " + args1, " | " + args2.
            # Each segment's tokens go in head_h's label array at the running offset.
            for ell in range(K):
                target_text = ex.per_layer_target[ell]
                # Split into 4 lists by " | ". Build expect_4 with sanity guard.
                segs = target_text.split(" | ")
                if len(segs) != 4:
                    # Fall back: skip per-head labels for this example/breath (will yield no
                    # per-head supervision). Should never happen for v81 data.
                    continue
                head_prefixes = [" ", " | ", " | ", " | "]
                offset = 0  # position offset within the L<ell> labels
                for hi in range(4):
                    head_text = head_prefixes[hi] + segs[hi]
                    seg_ids = tok.encode(head_text).ids
                    for j, tid in enumerate(seg_ids):
                        label_pos = (prompt_len - 1) + offset + j
                        if 0 <= label_pos < fixed_len - 1:
                            per_head_per_step_labels_np[hi, ell, b, label_pos] = tid
                    offset += len(seg_ids)

        # v75-style full-answer labels: union of all layer spans. Used when
        # PER_BREATH_FULL_ANSWER=1 (mostly off for V77; we want per-layer specialization).
        full_answer_labels_np = np.full((B, fixed_len - 1), -100, dtype=np.int32)
        for b in range(B):
            p_len = prompt_lens[b]
            # Use the FINAL layer's labels as the canonical "full answer" target.
            final_layer_ids = tok.encode(" " + batch_examples[b].per_layer_target[-1]).ids
            for j, tid in enumerate(final_layer_ids):
                label_pos = (p_len - 1) + j
                if 0 <= label_pos < fixed_len - 1:
                    full_answer_labels_np[b, label_pos] = tid

        # No rename aug for V77 (the DAG vocabulary is structured; aug would break it).
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        per_step_labels_t = [Tensor(per_step_labels_np[ell], dtype=dtypes.int).realize() for ell in range(K)]
        full_answer_labels_t = Tensor(full_answer_labels_np, dtype=dtypes.int).realize()
        # v81 per-head label tensor; always built so the JIT signature stays stable.
        per_head_per_step_labels_t = Tensor(per_head_per_step_labels_np, dtype=dtypes.int).realize()
        # v81 per-example prompt-range mask (1.0 at prompt positions, 0 at answer-span).
        # Used as `main_attn_mask` for breathe_with_lookup AND as `notebook_pool_mask`
        # AND as `kv_mask` for the WaistController. Single mask source for full
        # answer-span isolation. Static across the breath/decode loop.
        v81_main_mask_np = np.zeros((B, fixed_len), dtype=np.float32)
        for b in range(B):
            v81_main_mask_np[b, :prompt_lens[b]] = 1.0
        v81_main_mask_t = Tensor(v81_main_mask_np, dtype=dtypes.float).realize()

        # Jump to the shared JIT/eager dispatch using V77's K/labels.
        # We skip the cycles_per_ex/step_ranges_per_ex setup below.
        v77_active = True
    else:
        v77_active = False
        per_head_per_step_labels_t = None
        v81_main_mask_t = None

    if not v77_active:
        # Determine K from the first example. K must be uniform across batch (sample
        # from a single level for Stage 1).
        cycles_per_ex = [encode_cycles(tok, ex) for ex in batch_examples]
        K = len(cycles_per_ex[0])
        assert all(len(c) == K for c in cycles_per_ex), "per_breath_train_step needs uniform K across batch"

        # Use the LAST cycle's ids (= cumulative full sequence with EOS).
        # Build per-step token ranges relative to the full sequence.
        full_ids_per_ex = []
        step_ranges_per_ex = []
        for ex_cycles in cycles_per_ex:
            full = ex_cycles[-1][0]
            ranges = []
            for k in range(K):
                start_pos_k = ex_cycles[k][1] - 1  # position whose label is step k's first token
                if k + 1 < K:
                    end_pos_k = ex_cycles[k + 1][1] - 2
                else:
                    end_pos_k = ex_cycles[k][2] - 2  # last position before EOS
                ranges.append((start_pos_k, end_pos_k))
            full_ids_per_ex.append(full)
            step_ranges_per_ex.append(ranges)

        # Pad to fixed_len
        B = len(batch_examples)
        tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
        for b in range(B):
            ids = full_ids_per_ex[b][:fixed_len]
            tokens_np[b, :len(ids)] = ids

        # Per-step labels: shape (K, B, fixed_len - 1) with -100 outside step k's positions.
        per_step_labels_np = np.full((K, B, fixed_len - 1), -100, dtype=np.int32)
        for b in range(B):
            ids = full_ids_per_ex[b]
            ranges = step_ranges_per_ex[b]
            for k in range(K):
                start, end = ranges[k]
                for p in range(max(0, start), min(end + 1, fixed_len - 1)):
                    if p + 1 < len(ids):
                        per_step_labels_np[k, b, p] = ids[p + 1]

        # v75 (2026-05-23) Full-answer labels for multi-well supervision.
        # Shape (B, fixed_len - 1), -100 outside the full answer span (ranges[0][0] to
        # ranges[K-1][1]), gold next-token within. Every breath gets the SAME target
        # when PER_BREATH_FULL_ANSWER=1. Always built so the JIT signature stays
        # stable (the JIT body gates which labels are used at compile time).
        full_answer_labels_np = np.full((B, fixed_len - 1), -100, dtype=np.int32)
        for b in range(B):
            ids = full_ids_per_ex[b]
            ranges = step_ranges_per_ex[b]
            full_start = max(0, ranges[0][0])
            full_end = min(ranges[K - 1][1], fixed_len - 2)
            for p in range(full_start, full_end + 1):
                if p + 1 < len(ids):
                    full_answer_labels_np[b, p] = ids[p + 1]

        # v73: token-level rename augmentation. Fires AFTER tokens+labels are built
        # but BEFORE they're moved to GPU. Modifies numpy arrays in-place; tensor
        # shapes are unchanged (JIT cache stays valid).
        if WAIST_COPY_RENAME_AUG:
            global _RENAME_FIRST_BATCH_LOGGED
            summaries = _apply_rename_aug(
                tokens_np, per_step_labels_np, tok,
                rate=WAIST_COPY_RENAME_RATE,
                num_tokens=WAIST_COPY_RENAME_NUM_TOKENS,
            )
            if not _RENAME_FIRST_BATCH_LOGGED:
                n_aug = sum(1 for s in summaries if s is not None)
                print(f"[v73 rename-aug] first batch: {n_aug}/{B} examples renamed "
                      f"(rate={WAIST_COPY_RENAME_RATE}, num_tokens={WAIST_COPY_RENAME_NUM_TOKENS})",
                      flush=True)
                for b_idx, s in enumerate(summaries):
                    if s is None:
                        continue
                    orig_decoded = [tok.id_to_token(i) for i in s["orig_ids"]]
                    new_decoded = [tok.id_to_token(i) for i in s["new_ids"]]
                    print(f"  ex{b_idx}: orig={list(zip(s['orig_ids'], orig_decoded))} "
                          f"-> new={list(zip(s['new_ids'], new_decoded))} "
                          f"replacements={s['n_replacements']}", flush=True)
                _RENAME_FIRST_BATCH_LOGGED = True

        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        per_step_labels_t = [Tensor(per_step_labels_np[k], dtype=dtypes.int).realize() for k in range(K)]
        # v75 full-answer labels for multi-well supervision. Always built; the JIT body
        # picks step-k labels vs full-answer labels based on PER_BREATH_FULL_ANSWER at
        # compile time (the env var enters the JIT cache key).
        full_answer_labels_t = Tensor(full_answer_labels_np, dtype=dtypes.int).realize()
        # v81 per-head label dummy + v81 main mask dummy (non-v77 path doesn't use them
        # but the JIT signature is stable).
        per_head_per_step_labels_t = Tensor(np.full((4, K, B, fixed_len - 1), -100, dtype=np.int32),
                                              dtype=dtypes.int).realize()
        v81_main_mask_t = Tensor(np.zeros((B, fixed_len), dtype=np.float32), dtype=dtypes.float).realize()

    # JIT fast path — fused forward+backward+opt.step, flat per-step time.
    if use_jit:
        # Build aux tensors regardless of weight so the JIT signature is stable;
        # the compiled function gates aux_ce on lookup_aux_weight at compile time.
        # V77 path: op-label aux isn't meaningful for DAG targets, so pass dummies
        # (lookup_aux_weight should be 0 when V77_DAG_TRAINING=1 anyway).
        if lookup_eq_token_id is not None and not v77_active:
            eq_mask_jit, op_labels_jit = _build_aux_tensors(batch_examples, tokens_np, lookup_eq_token_id)
        else:
            eq_mask_jit = Tensor(np.zeros((B, fixed_len, 1), dtype=np.float32),
                                  dtype=dtypes.float).realize()
            op_labels_jit = Tensor(np.full((B,), -100, dtype=np.int32),
                                    dtype=dtypes.int).realize()
        labels_stk = Tensor(per_step_labels_np, dtype=dtypes.int).realize()

        # v78b ATTENTION SUPERVISION TARGETS (built always for stable JIT signature;
        # the JIT body multiplies by attn_mask which is all-zeros when supervision is
        # off, so the loss contribution is 0). For each digit token in the LAST
        # breath's labels (= L6 DAG for V77; = step K-1 labels otherwise), find
        # matching ORIGINAL-PROMPT positions (excluding the answer span, which would
        # be trivial self-attend); uniform distribution over matches; mask=1 at
        # supervised positions. Shape:
        #   attn_target_np : (B, fixed_len - 1, fixed_len)  — distribution over prompt positions
        #   attn_mask_np   : (B, fixed_len - 1)             — 1.0 at supervised positions
        #
        # Prompt range: positions [0, prompt_lens[b]). For V77 prompt_lens is the
        # original-problem-tokens length. For the legacy (non-v77) path we don't
        # have a separate `prompt_lens` list, but the labels in per_step_labels_np
        # are -100 outside the answer span — so digit-label positions are naturally
        # in the answer-span only. We allow matches over the whole sequence in that
        # case, treating positions before the answer span as the "prompt".
        from mycelium.breathing import WAIST_ATTN_SUPERVISION as _WAS
        attn_target_np = np.zeros((B, fixed_len - 1, fixed_len), dtype=np.float32)
        attn_mask_np   = np.zeros((B, fixed_len - 1), dtype=np.float32)
        if _WAS:
            digit_ids = _get_digit_token_ids(tok)
            # Use the LAST breath's labels (per spec: same prompt-attention target at every
            # breath since operands live in the same prompt positions across all breaths).
            last_labels = per_step_labels_np[K - 1]  # (B, fixed_len - 1)
            # Build per-example prompt range. For V77 we have explicit prompt_lens;
            # for the legacy path we approximate it as the smallest position with a
            # non-(-100) label minus 1 (i.e., the first label position is at
            # prompt_len - 1). Falls back to scanning all of `tokens_np[b]` if
            # there are no labels.
            if v77_active:
                prompt_ranges = prompt_lens  # already a list[int]
            else:
                prompt_ranges = []
                for b in range(B):
                    # find earliest non-(-100) over all K breaths' labels
                    earliest = fixed_len - 1
                    for k_idx in range(K):
                        nz = np.where(per_step_labels_np[k_idx, b] != -100)[0]
                        if nz.size > 0:
                            earliest = min(earliest, int(nz[0]))
                    prompt_ranges.append(earliest + 1)  # labels[earliest] = next_token after prompt[earliest]
            for b in range(B):
                prompt_end = int(prompt_ranges[b])
                p_ids_prompt = tokens_np[b, :prompt_end].tolist()
                for t in range(fixed_len - 1):
                    tok_id = int(last_labels[b, t])
                    if tok_id == -100 or tok_id not in digit_ids:
                        continue
                    matches = [i for i, p_tid in enumerate(p_ids_prompt) if p_tid == tok_id]
                    if not matches:
                        continue
                    w = 1.0 / float(len(matches))
                    for i in matches:
                        attn_target_np[b, t, i] = w
                    attn_mask_np[b, t] = 1.0
        attn_target_t = Tensor(attn_target_np, dtype=dtypes.float).realize()
        attn_mask_t   = Tensor(attn_mask_np, dtype=dtypes.float).realize()
        # v63 prompt dropout: per-step Bernoulli mask (1.0 = use prompt, 0.0 = zero prompt).
        # Drop fraction = CONTROLLER_PROMPT_DROPOUT env var; 0 disables.
        import os as _os, random as _random
        _pdrop = float(_os.environ.get("CONTROLLER_PROMPT_DROPOUT", "0.0"))
        _pd_val = 0.0 if (_pdrop > 0.0 and _random.random() < _pdrop) else 1.0
        prompt_dropout_mask_t = Tensor(np.array([_pd_val], dtype=np.float32), dtype=dtypes.float).realize()

        # v79 (2026-05-25) Causal kv_mask for cross-attn + notebook attention pool.
        # mask=1.0 at positions [0, prompt_lens[b]), 0.0 elsewhere. Built ALWAYS so
        # the JIT signature is stable; the JIT body uses it only when v79 flag is on.
        # The flag is captured at compile time; when off, the body ignores the mask.
        from mycelium.breathing import V79_CAUSAL_MASKS as _V79CM_OUTER
        kv_mask_np = np.zeros((B, fixed_len), dtype=np.float32)
        if _V79CM_OUTER:
            # prompt_ranges built above when WAIST_ATTN_SUPERVISION on; if it's off
            # we need to derive prompt ranges from labels (same logic as the attn
            # supervision block above).
            if v77_active:
                pl_list = prompt_lens  # already a list[int]
            else:
                pl_list = []
                for b in range(B):
                    earliest = fixed_len - 1
                    for k_idx in range(K):
                        nz = np.where(per_step_labels_np[k_idx, b] != -100)[0]
                        if nz.size > 0:
                            earliest = min(earliest, int(nz[0]))
                    pl_list.append(earliest + 1)
            for b in range(B):
                pe = min(int(pl_list[b]), fixed_len)
                kv_mask_np[b, :pe] = 1.0
        kv_mask_t_v79 = Tensor(kv_mask_np, dtype=dtypes.float).realize()

        if SCHED_SAMPLE_RATE > 0.0:
            # v66 Scheduled sampling path: unrolled breath loop in the JIT.
            # Linear ramp: 0.0 for steps 0-500, ramping to SCHED_SAMPLE_RATE at step 1500+.
            _ss_warmup = 500
            _ss_full_ramp = 1500
            _ss_target = SCHED_SAMPLE_RATE
            _current_rate = max(0.0, min(_ss_target,
                (_ss_target * (step_idx - _ss_warmup) / float(_ss_full_ramp - _ss_warmup))
                if step_idx > _ss_warmup else 0.0))
            # Bernoulli mask: 1.0 where we replace gold with argmax predictions.
            # Shape (B, fixed_len - 1) — positions 1..T in the token sequence.
            if _current_rate > 0.0:
                _bern_np = np.random.binomial(1, _current_rate, size=(B, fixed_len - 1)).astype(np.float32)
            else:
                _bern_np = np.zeros((B, fixed_len - 1), dtype=np.float32)
            sched_sample_rate_t = Tensor(np.array([_current_rate], dtype=np.float32), dtype=dtypes.float).realize()
            bernoulli_mask_t = Tensor(_bern_np, dtype=dtypes.float).realize()
            step_fn_ss = _compile_jit_per_breath_step_ss(model, opt, K, fixed_len, B, lookup_aux_weight, grad_clip)
            outs = step_fn_ss(tokens, labels_stk, full_answer_labels_t, eq_mask_jit, op_labels_jit, prompt_dropout_mask_t,
                              sched_sample_rate_t, bernoulli_mask_t, attn_target_t, attn_mask_t, kv_mask_t_v79,
                              per_head_per_step_labels_t, v81_main_mask_t)
        else:
            step_fn = _compile_jit_per_breath_step(model, opt, K, fixed_len, B, lookup_aux_weight, grad_clip)
            outs = step_fn(tokens, labels_stk, full_answer_labels_t, eq_mask_jit, op_labels_jit, prompt_dropout_mask_t,
                           attn_target_t, attn_mask_t, kv_mask_t_v79,
                           per_head_per_step_labels_t, v81_main_mask_t)

        total_t = outs[0]
        healthy_t = outs[1]
        waist_norm_t = outs[2]
        attn_aux_t = outs[3]  # v78b: WaistController cross-attn supervision aux loss
        # v81: JIT returns (total, healthy, waist_norm, attn_aux) + K per-breath CE
        # + 4 per-head sums (ALWAYS present; zero when mhw_flag=0).
        ce_ts = outs[4:4 + K]
        per_head_ts = outs[4 + K:4 + K + 4]
        if float(healthy_t.numpy()) < 0.5:
            print("[NaN-skip] NaN grad detected — step skipped", flush=True)
        # v78b: print attn aux value periodically so the smoke can verify supervision firing.
        from mycelium.breathing import WAIST_ATTN_SUPERVISION as _WAS_LOG
        from mycelium.breathing import MULTI_HEAD_WAIST as _MHW_LOG
        if _WAS_LOG and (step_idx == 0 or step_idx % 10 == 0):
            print(f"[attn_aux] step {step_idx}: {float(attn_aux_t.numpy()):.4f}", flush=True)
        if _MHW_LOG and (step_idx == 0 or step_idx % 10 == 0):
            head_names_log = ["ops", "types", "args1", "args2"]
            ph_vals = [float(t.numpy()) for t in per_head_ts]
            ph_str = "  ".join(f"{n}={v / float(K):.3f}" for n, v in zip(head_names_log, ph_vals))
            print(f"[per_head_ce] step {step_idx}: {ph_str}", flush=True)

        # v66 waist norm logging + collapse guardrail.
        waist_norm_val = float(waist_norm_t.numpy())
        if step_idx % 10 == 0:
            # Reported alongside the per-breath CE in l3_train.py's log line.
            # Store as return extra so caller can include it.
            pass  # caller reads waist_norm_val from the returned extras tuple below
        if waist_norm_val < _WAIST_NORM_WARN_THRESHOLD:
            _waist_norm_low_streak += 1
            if _waist_norm_low_streak >= _WAIST_NORM_WARN_STREAK:
                print(f"\n[WARNING] waist_norm < {_WAIST_NORM_WARN_THRESHOLD} for "
                      f"{_waist_norm_low_streak} consecutive steps "
                      f"(current: {waist_norm_val:.4f}) — possible skip-connection leakage "
                      f"(model may be ignoring waist, relying entirely on prompt refresh).\n",
                      flush=True)
        else:
            _waist_norm_low_streak = 0

        return float(total_t.numpy()), [float(c.numpy()) for c in ce_ts], waist_norm_val

    # Forward — fetch per-breath x AND per-breath waist_compressed if controller is active
    from mycelium.breathing import CONTROLLER_DECODE
    opt.zero_grad()
    if CONTROLLER_DECODE:
        _final, match_weights, per_breath_x, waist_compressed_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True)
        # Prompt embeddings (used as cross-attn keys/values by the controller)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
    else:
        _final, match_weights, per_breath_x = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True)
        waist_compressed_per_breath = None
        prompt_emb = None

    # Per-breath CE (equal-weighted). Decode path depends on CONTROLLER_DECODE.
    # v75: when PER_BREATH_FULL_ANSWER=1, every breath shares the same target
    # (full-answer labels) instead of step-k-specific labels — the multi-well
    # supervision paradigm (diffusion-style: K refinements toward one target).
    losses_per_breath = []
    cfg_eps = model.cfg.layer_norm_eps
    for k in range(K):
        if CONTROLLER_DECODE:
            # v54 Phase 1: route through WaistController (cross-attention to prompt).
            waist_k = waist_compressed_per_breath[k].cast(dtypes.float)
            logits = model.waist_controller.forward(waist_k, prompt_emb, model.embed_out)
        else:
            x_k = per_breath_x[k]
            x_normed = _layernorm(x_k, model.ln_f_g, model.ln_f_b, cfg_eps)
            logits = (x_normed @ model.embed_out).cast(dtypes.float)
        pred = logits[:, :-1, :]
        ls = LABEL_SMOOTHING
        target_k = full_answer_labels_t if PER_BREATH_FULL_ANSWER else per_step_labels_t[k]
        ce_k = pred.sparse_categorical_crossentropy(
            target_k, ignore_index=-100, label_smoothing=ls, reduction="mean")
        losses_per_breath.append(ce_k)

    avg_main = sum(losses_per_breath[1:], losses_per_breath[0]) / float(K)

    # Lookup aux (op classification on last breath's match weights). Skipped for V77
    # (DAG targets don't have a single "=" position for op classification — each xN
    # node has its own =, and the op label heuristic is built for L4/L4.5/L4.7 templates).
    if lookup_aux_weight > 0 and lookup_eq_token_id is not None and not v77_active:
        eq_mask, op_labels_t = _build_aux_tensors(batch_examples, tokens_np, lookup_eq_token_id)
        last_mw = match_weights[-1]
        gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
        logits_aux = gathered[:, :4] * 10.0
        aux_ce = logits_aux.sparse_categorical_crossentropy(
            op_labels_t, ignore_index=-100, reduction="mean")
    else:
        aux_ce = Tensor.zeros((), dtype=dtypes.float).contiguous()

    # L2 regs (mirror multi_cycle_train_step's regs)
    l2_reg = (model.lookup_table.weight.square().mean()
              + model.lookup_table.values.square().mean()
              + model.lookup_table.value_proj_up.square().mean()) * 1e-6
    ch_reg = sum((p.square().mean() for p in model.confidence_head.parameters()),
                 Tensor.zeros((), dtype=dtypes.float).contiguous()) * 1e-7
    be_reg = (model.block.breath_embed.square().mean()
              + model.block.handoff_w.square().mean()
              + model.block.handoff_b.square().mean()
              + model.block.rope.pitch.square().mean()
              + model.block.crp_mix_alpha.square().mean()
              + model.block.crp_target_norm.square().mean()
              + model.block.notebook_write_w.square().mean()
              + model.block.notebook_write_b.square().mean()
              + model.block.notebook_read_w.square().mean()
              + model.block.notebook_read_b.square().mean()
              + model.block.notebook_write_query.square().mean()
              + model.block.notebook_rep_write_w.square().mean()
              + model.block.notebook_rep_write_b.square().mean()
              + model.block.notebook_rep_read_w.square().mean()
              + model.block.notebook_rep_read_b.square().mean()
              + model.block.notebook_rep_query.square().mean()
              + model.block.bfield_proj_down.square().mean()
              + model.block.bfield_proj_up.square().mean()
              + model.block.bfield_bias.square().mean()
              + model.block.waist_codebook_keys.square().mean()
              + model.block.waist_codebook_values.square().mean()
              + model.waist_head_w.square().mean()
              + model.waist_head_b.square().mean()
              + model.ln_f_g.square().mean()        # v77: ln_f params have no gradient path
              + model.ln_f_b.square().mean()        #      when CONTROLLER_DECODE=1 AND aux_ce=0
              + sum((p.square().mean() for lb in model.block.layers_b
                                          for p in [lb.wq, lb.bq, lb.wk, lb.bk, lb.w_in, lb.b_in]),
                    Tensor.zeros((), dtype=dtypes.float).contiguous())) * 1e-7

    total = avg_main + lookup_aux_weight * aux_ce + l2_reg + ch_reg + be_reg
    total.backward()
    opt.step()
    # Eager path: compute waist norm too (for parity with JIT return signature)
    from mycelium.breathing import CONTROLLER_DECODE as _CD_eager
    if _CD_eager and waist_compressed_per_breath:
        waist_last = waist_compressed_per_breath[-1].cast(dtypes.float)
        waist_norm_val = float(waist_last.square().mean(axis=-1).sqrt().mean().realize().numpy())
    else:
        waist_norm_val = None
    return float(total.realize().numpy()), [float(c.realize().numpy()) for c in losses_per_breath], waist_norm_val


def multi_cycle_train_step(model, opt, batch_examples: List[MathExample], tok,
                           n_loops, fixed_len: int,
                           lookup_aux_weight: float = 0.0,
                           lookup_eq_token_id: int | None = None,
                           profile: bool = False,
                           use_jit: bool = False):
    """Per-cycle forward+backward. Each outer cycle gets its own breathing pass.
    Losses are summed across cycles and normalized by num_cycles (equal-weight
    decomposition).

    n_loops: int (uniform) or list[int] of length >= n_cycles (per-cycle scheduling).
    For three-phase: pass [phase_a_loops, phase_c_loops, phase_c_loops, ...].

    lookup_aux_weight: if > 0, adds a cross-entropy loss on the model's lookup
    table at the "=" token position (cycle 0 only). Drives the table entries
    toward the model's actual operation directions during joint training.

    use_jit: when True, dispatches to a JIT-compiled train step (forward +
    backward + opt.step in one TinyJit, cached per loops_per_cycle tuple).
    First call per unique config compiles (~10s); subsequent calls replay as a
    single fused graph at ~2-3× the eager speed. Requires lookup_aux_weight > 0
    and lookup_eq_token_id set (the JIT path always includes the aux loss).

    profile: when True, returns (loss, timings_dict) with keys
      {encode, py_overhead, gpu_compute}. Off by default.
    """
    import time as _time
    Tensor.training = True

    # JIT fast path — compiles the whole step (forward + backward + opt.step) into
    # one fused graph per unique loops_per_cycle tuple. Requires aux to be on.
    if use_jit and lookup_aux_weight > 0 and lookup_eq_token_id is not None:
        if profile: _t = _time.perf_counter()
        cycles_per_ex = [encode_cycles(tok, ex) for ex in batch_examples]
        n_cycles = len(cycles_per_ex[0])
        loops_per_cycle = _resolve_loops_per_cycle(n_loops, n_cycles)
        encode_time = (_time.perf_counter() - _t) if profile else 0.0

        if profile: _t = _time.perf_counter()
        tokens_per_cycle = []
        labels_per_cycle = []
        for c in range(n_cycles):
            encoded = [ex_cycles[c] for ex_cycles in cycles_per_ex]
            tokens_np, labels_np = collate(encoded, fixed_len=fixed_len)
            tokens_per_cycle.append(Tensor(tokens_np, dtype=dtypes.int).realize())
            labels_per_cycle.append(Tensor(labels_np, dtype=dtypes.int).realize())
            if c == 0:
                eq_mask, op_labels_t = _build_aux_tensors(batch_examples, tokens_np,
                                                          lookup_eq_token_id)
        B = int(tokens_per_cycle[0].shape[0])
        jit_step = _compile_jit_train_step(model, opt, tuple(loops_per_cycle),
                                           fixed_len, B, lookup_aux_weight)
        if n_cycles == 1:
            loss_t = jit_step(tokens_per_cycle[0], labels_per_cycle[0],
                              eq_mask, op_labels_t)
        elif n_cycles == 2:
            loss_t = jit_step(tokens_per_cycle[0], labels_per_cycle[0],
                              tokens_per_cycle[1], labels_per_cycle[1],
                              eq_mask, op_labels_t)
        elif n_cycles == 3:
            loss_t = jit_step(tokens_per_cycle[0], labels_per_cycle[0],
                              tokens_per_cycle[1], labels_per_cycle[1],
                              tokens_per_cycle[2], labels_per_cycle[2],
                              eq_mask, op_labels_t)
        elif n_cycles == 4:
            loss_t = jit_step(tokens_per_cycle[0], labels_per_cycle[0],
                              tokens_per_cycle[1], labels_per_cycle[1],
                              tokens_per_cycle[2], labels_per_cycle[2],
                              tokens_per_cycle[3], labels_per_cycle[3],
                              eq_mask, op_labels_t)
        else:
            raise NotImplementedError(f"use_jit doesn't support n_cycles={n_cycles}")
        if profile:
            Device[Device.DEFAULT].synchronize()
            gpu_compute = _time.perf_counter() - _t
        loss_val = float(loss_t.numpy())
        if profile:
            return loss_val, {"encode": encode_time, "py_overhead": 0.0, "gpu_compute": gpu_compute}
        return loss_val

    # Eager path (original)
    opt.zero_grad()

    if profile: _t = _time.perf_counter()
    cycles_per_ex = [encode_cycles(tok, ex) for ex in batch_examples]
    n_cycles = len(cycles_per_ex[0])
    loops_per_cycle = _resolve_loops_per_cycle(n_loops, n_cycles)
    encode_time = (_time.perf_counter() - _t) if profile else 0.0

    cycle_losses = []
    aux_loss = None
    use_shared_forward = (lookup_aux_weight > 0 and lookup_eq_token_id is not None)

    if profile: _t = _time.perf_counter()
    for c in range(n_cycles):
        encoded = [ex_cycles[c] for ex_cycles in cycles_per_ex]
        tokens_np, labels_np = collate(encoded, fixed_len=fixed_len)
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        labels = Tensor(labels_np, dtype=dtypes.int).realize()
        if c == 0 and use_shared_forward:
            # Shared forward: one breathe_with_lookup pass returns both the
            # per-token hidden states (for main CE) and per-breath match weights
            # (for aux CE) — no second forward through the transformer.
            main_ce, match_weights = masked_forward_loss_with_lookup(
                model, tokens, labels, loops_per_cycle[c]
            )
            cycle_losses.append(main_ce)
            aux_loss = _aux_loss_from_match_weights(
                match_weights, tokens_np, batch_examples, lookup_eq_token_id
            )
        else:
            cycle_losses.append(masked_forward_loss(model, tokens, labels, loops_per_cycle[c]))

    total = cycle_losses[0]
    for l in cycle_losses[1:]:
        total = total + l
    avg_loss = total / float(n_cycles)

    # Always include lookup_table in the graph via a tiny L2 reg. Two purposes:
    #   1. opt.step() requires every parameter to have a defined gradient, even if
    #      this batch had no valid eq+op pair (or aux is off). The L2 reg gives
    #      lookup_table a small, always-defined gradient.
    #   2. Spec calls for a regularizer keeping the prime entries from drifting
    #      toward each other; the L2 norm is a mild form of that.
    # Coefficient is tiny (1e-6) so behavior impact is negligible.
    avg_loss = (avg_loss
                + model.lookup_table.weight.square().mean() * 1e-6
                + model.lookup_table.values.square().mean() * 1e-6
                + model.lookup_table.value_proj_up.square().mean() * 1e-6)
    # Confidence head: tiny L2 reg so its grad is defined when CALIBRATION_MODE=0
    # (the path that doesn't otherwise touch the head). Behavior impact negligible;
    # the head doesn't move in this regime because the gradient is microscopic.
    for p in model.confidence_head.parameters():
        avg_loss = avg_loss + p.square().mean() * 1e-7
    # Breath-time embedding: same idea — keep gradient defined when BREATH_TIME_EMBED=0.
    avg_loss = avg_loss + model.block.breath_embed.square().mean() * 1e-7
    # Cross-breath handoff weights: same idea — keep gradient defined when CROSS_BREATH_HANDOFF=0.
    avg_loss = avg_loss + model.block.handoff_w.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.handoff_b.square().mean() * 1e-7
    # Helix pitch scalar: same idea — keep gradient defined when LEARN_PITCH=0.
    avg_loss = avg_loss + model.block.rope.pitch.square().mean() * 1e-7
    # Constant-radius projection scalars: same idea — gradient defined when CONSTANT_RADIUS=0.
    avg_loss = avg_loss + model.block.crp_mix_alpha.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.crp_target_norm.square().mean() * 1e-7
    # v24 notebook projections: gradient defined when NOTEBOOK_V24=0 (or for 1-breath
    # cycles where notebook never propagates). Tiny L2 keeps opt.step() happy.
    avg_loss = avg_loss + model.block.notebook_write_w.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_write_b.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_read_w.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_read_b.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_write_query.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_rep_write_w.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_rep_write_b.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_rep_read_w.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_rep_read_b.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.notebook_rep_query.square().mean() * 1e-7
    # v38 B-field IB bottleneck: same idea — gradient defined when BFIELD_WAIST=0.
    avg_loss = avg_loss + model.block.bfield_proj_down.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.bfield_proj_up.square().mean() * 1e-7
    avg_loss = avg_loss + model.block.bfield_bias.square().mean() * 1e-7
    avg_loss = avg_loss + model.waist_head_w.square().mean() * 1e-7
    avg_loss = avg_loss + model.waist_head_b.square().mean() * 1e-7
    if aux_loss is not None:
        avg_loss = avg_loss + lookup_aux_weight * aux_loss

    # Time breakdown:
    #   encode_time: Python tokenization (already captured above)
    #   py_overhead: Python time spent building the forward graph (no GPU work
    #                — tinygrad is lazy, so the whole forward block above is
    #                graph construction, not compute)
    #   gpu_compute: actual GPU work (forward + backward + opt.step), bounded
    #                by the Device.synchronize at the end
    # We don't try to split forward-vs-backward GPU time because realize()-ing
    # mid-step breaks autograd (produces grad=None on some params).
    if profile: py_overhead = _time.perf_counter() - _t
    if profile: _t2 = _time.perf_counter()
    avg_loss.backward()
    opt.step()
    Device[Device.DEFAULT].synchronize()
    if profile: gpu_compute = _time.perf_counter() - _t2

    loss_val = float(avg_loss.numpy())
    if profile:
        return loss_val, {
            "encode": encode_time,
            "py_overhead": py_overhead,
            "gpu_compute": gpu_compute,
        }
    return loss_val


def multi_cycle_eval_loss(model, batch_examples: List[MathExample], tok,
                          n_loops, fixed_len: int) -> float:
    """Per-cycle eval loss (no backward). Equal-weighted across cycles. Same per-cycle
    loop scheduling as multi_cycle_train_step."""
    Tensor.training = False
    cycles_per_ex = [encode_cycles(tok, ex) for ex in batch_examples]
    n_cycles = len(cycles_per_ex[0])
    loops_per_cycle = _resolve_loops_per_cycle(n_loops, n_cycles)
    cycle_losses = []
    for c in range(n_cycles):
        encoded = [ex_cycles[c] for ex_cycles in cycles_per_ex]
        tokens_np, labels_np = collate(encoded, fixed_len=fixed_len)
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        labels = Tensor(labels_np, dtype=dtypes.int).realize()
        cycle_losses.append(masked_forward_loss(model, tokens, labels, loops_per_cycle[c]))
    total = cycle_losses[0]
    for l in cycle_losses[1:]:
        total = total + l
    return float((total / float(n_cycles)).realize().numpy())


def multi_cycle_generate(model, tok, problem_ids: List[int], n_loops, n_cycles: int,
                         max_new_per_cycle: int = 40, eos_id: int = 0,
                         vocab_active: int = 50277, use_kv_cache: bool = False) -> List[List[int]]:
    """Per-cycle inference — explicit outer cycle loop with three-phase scheduling.

    n_loops: int (uniform) or list[int] (per-cycle). For three-phase eval:
    pass [phase_a_loops, phase_c_loops, ...] so cycle 0 does the heavy analysis
    and subsequent cycles do light execution.

    use_kv_cache: when True, each cycle uses the model's cached_generate (breathing
    once on the cycle prefix, then incremental token generation with cached K/V).
    Massively faster than re-breathing per token.
    """
    Tensor.training = False
    loops_per_cycle = _resolve_loops_per_cycle(n_loops, n_cycles)
    sep_ids = tok.encode(SEP).ids
    sep_len = len(sep_ids)
    context = list(problem_ids)
    cycle_outputs: List[List[int]] = []

    for cyc in range(n_cycles):
        nl = loops_per_cycle[cyc]
        if use_kv_cache:
            ctx = context[-model.cfg.max_seq_len:]
            gen = model.cached_generate(
                ctx, n_loops=nl, max_new=max_new_per_cycle,
                stop_token_ids=[eos_id], stop_seq=sep_ids,
                vocab_active=vocab_active,
            )
        else:
            gen = []
            for _ in range(max_new_per_cycle):
                ctx = (context + gen)[-model.cfg.max_seq_len:]
                toks = Tensor([ctx], dtype=dtypes.int).realize()
                h = model(toks, nl)
                last = h[:, -1, :]
                logits = (last @ model.embed_out).cast(dtypes.float)
                logits = logits[:, :vocab_active]
                next_id = int(logits.argmax(axis=-1).realize().numpy()[0])
                gen.append(next_id)
                if next_id == eos_id:
                    break
                if len(gen) >= sep_len and gen[-sep_len:] == sep_ids:
                    break
        cycle_outputs.append(gen)
        context.extend(gen)

    return cycle_outputs


def accuracy_at_loops_multi(model, tok, examples: List[MathExample], n_loops,
                            max_new_per_cycle: int = 40,
                            batch_size: int = 64,
                            cache_max_len: int | None = None) -> Tuple[float, List[Tuple[MathExample, int | None, str]]]:
    """Multi-cycle accuracy eval. Single-cycle (L3) examples use the batched cached
    generation path. The JIT compile is keyed on (B, n_loops, vocab_active), so we pad
    the last chunk up to batch_size to keep B uniform — that way the JIT compiles
    on the first eval call and is reused on every subsequent call (40s compile
    amortized to one-time cost).

    Sets Tensor.training = False so tinygrad doesn't track autograd state during
    the eval forward — matters because main training leaves it at True.

    cache_max_len: override the K/V buffer length for short sequences (e.g., 32 for
    L3-spaced arithmetic). Defaults to cfg.max_seq_len. Smaller values cut cache
    memory linearly, allowing larger batch_size within the GPU memory budget.

    n_loops can be int (uniform) or list (per-cycle). For three-phase eval pass
    [phase_a, phase_c, phase_c, ...].
    """
    Tensor.training = False
    correct = 0
    rows = []
    n_cycles = len(examples[0].gen_targets) if examples else 1
    sep_ids = tok.encode(SEP).ids
    # Phase A loops = first entry if list, else the int
    phase_a_loops = n_loops[0] if isinstance(n_loops, list) else int(n_loops)

    if n_cycles == 1:
        # Batched path: process in fixed-size chunks of batch_size.
        # Pad the last (potentially short) chunk up to batch_size so all JIT calls
        # share the same B (== same compiled graph). Padding uses a benign repeat
        # of an existing prompt — its outputs are discarded.
        prompt_ids_all = [tok.encode(ex.problem).ids for ex in examples]
        n_total = len(examples)
        for chunk_start in range(0, n_total, batch_size):
            chunk_end = min(chunk_start + batch_size, n_total)
            real_n = chunk_end - chunk_start
            chunk = examples[chunk_start:chunk_end]
            chunk_prompts = prompt_ids_all[chunk_start:chunk_end]
            # Pad up to batch_size if last chunk is short (keeps B uniform → JIT reuse)
            if real_n < batch_size and prompt_ids_all:
                pad_n = batch_size - real_n
                chunk_prompts = chunk_prompts + [prompt_ids_all[0]] * pad_n
            outs_batched = model.cached_generate_batch(
                chunk_prompts, n_loops=phase_a_loops, max_new=max_new_per_cycle,
                stop_token_ids=[0], stop_seq=sep_ids,
                cache_max_len=cache_max_len,
            )
            # Only score the real (non-padded) outputs
            for ex, gen_ids in zip(chunk, outs_batched[:real_n]):
                gen_text = tok.decode(gen_ids)
                parsed = parse_int_answer(gen_text)
                ok = (parsed == ex.answer)
                if ok:
                    correct += 1
                rows.append((ex, parsed, gen_text))
    else:
        # Batched multi-cycle path. One cached_generate_batch call per cycle: each call
        # processes all B prompts in parallel with cached K/V, then we splice the prior
        # cycle's generated tokens onto each running prompt for the next cycle. Internal
        # right-padding of cached_generate_batch already handles per-example variable
        # prompt lengths, so the assembled cycle-N prompts can have different lengths
        # across the batch.
        #
        # cache_max_len is sized for the last cycle (longest prompt: problem + all prior
        # cycle outputs + this cycle's own max_new). One JIT compile per (B, n_loops),
        # reused across all subsequent eval calls.
        prompt_ids_all = [tok.encode(ex.problem).ids for ex in examples]
        n_total = len(examples)
        loops_per_cycle = _resolve_loops_per_cycle(n_loops, n_cycles)
        max_prompt = max((len(p) for p in prompt_ids_all), default=0)
        eff_cache_max_len = max(cache_max_len or 0,
                                max_prompt + n_cycles * max_new_per_cycle)
        for chunk_start in range(0, n_total, batch_size):
            chunk_end = min(chunk_start + batch_size, n_total)
            real_n = chunk_end - chunk_start
            chunk = examples[chunk_start:chunk_end]
            chunk_prompts = prompt_ids_all[chunk_start:chunk_end]
            # Pad up to batch_size if last chunk is short (keeps B uniform → JIT reuse)
            if real_n < batch_size and prompt_ids_all:
                pad_n = batch_size - real_n
                chunk_prompts = chunk_prompts + [prompt_ids_all[0]] * pad_n
            B_eff = len(chunk_prompts)
            cycle_outs_per_ex: List[List[List[int]]] = [[] for _ in range(B_eff)]
            running_prompts = list(chunk_prompts)
            for cyc in range(n_cycles):
                nl = loops_per_cycle[cyc]
                outs = model.cached_generate_batch(
                    running_prompts, n_loops=nl, max_new=max_new_per_cycle,
                    stop_token_ids=[0], stop_seq=sep_ids,
                    cache_max_len=eff_cache_max_len,
                )
                for b in range(B_eff):
                    cycle_outs_per_ex[b].append(outs[b])
                running_prompts = [running_prompts[b] + outs[b] for b in range(B_eff)]
            # Only score the real (non-padded) outputs
            for ex, all_cycles in zip(chunk, cycle_outs_per_ex[:real_n]):
                last_text = tok.decode(all_cycles[-1])
                full_text = " ".join(tok.decode(co) for co in all_cycles)
                parsed = parse_int_answer(last_text)
                ok = (parsed == ex.answer)
                if ok:
                    correct += 1
                rows.append((ex, parsed, full_text))
    return correct / max(1, len(examples)), rows
