"""Training step + accuracy eval for math curriculum (L3, L4, L4.5).

Single-cycle path (L3): standard masked-loss CE, one forward pass per step.
Multi-cycle path (L4+): per-cycle forward passes — each outer cycle is its own
breathe-then-speak event. Equal-weighted loss across cycles (equal-reward
decomposition for teacher-forced training).
"""
from typing import List, Tuple
import numpy as np
from tinygrad import Tensor, Device, dtypes

from mycelium.l3_data import MathExample, encode_example, encode_cycles, parse_int_answer, collate, SEP
from mycelium.lookup_table import op_label_from_text, find_eq_position


def masked_forward_loss(model, tokens: Tensor, labels: Tensor, n_loops: int) -> Tensor:
    """Next-token CE loss, ignoring positions where label == -100.

    tokens: (B, T) int — full input
    labels: (B, T-1) int — targets for tokens[:, 1:], with -100 in masked positions
    """
    h = model(tokens, n_loops)                                 # (B, T, hidden)
    logits = (h @ model.embed_out).cast(dtypes.float)          # (B, T, vocab)
    pred = logits[:, :-1, :]                                   # (B, T-1, vocab)
    return pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")


def masked_forward_loss_with_lookup(model, tokens: Tensor, labels: Tensor,
                                    n_loops: int):
    """Combined main-CE forward that also returns per-breath lookup match weights.
    One forward pass instead of two — used when joint lookup-aux training is on,
    halving the main-step compute relative to running plain forward + a separate
    breathe_with_lookup for the aux loss.

    Returns (main_ce_loss, match_weights) where match_weights is a list of
    (B, T, n_entries) tensors, one per breath.
    """
    final_h, match_weights, _ = model.breathe_with_lookup(tokens, n_loops)
    logits = (final_h @ model.embed_out).cast(dtypes.float)
    pred = logits[:, :-1, :]
    main_ce = pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")
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
                          n_classes: int = 4, profile: bool = False):
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
    tokens_np, _ = collate(encoded, fixed_len=64 + 40)             # FIXED_LEN+max_new for safety
    B, T = tokens_np.shape
    tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
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
    notebook = Notebook()
    _, decisions, n_breaths, match_weights = model.breathe_controlled(
        tokens, max_loops=max_loops, notebook=notebook,
        detach_rep_for_ctrl=False, detach_decisions_into_transformer=False,
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

    per_breath_losses = []
    for mw in match_weights:
        gathered = (mw.cast(dtypes.float) * eq_mask).sum(axis=1)        # (B, n_entries)
        logits = gathered[:, :n_classes] * 10.0
        per_breath_losses.append(logits.sparse_categorical_crossentropy(y, ignore_index=-100, reduction="mean"))

    total = per_breath_losses[0]
    for l in per_breath_losses[1:]:
        total = total + l
    avg_ctrl_loss = total / float(len(per_breath_losses))

    # Auxiliary: stop calibration. The stop_logit at breath l should be HIGH when
    # the per-breath lookup-CE has plateaued (no further improvement from breathing).
    # Crude target: target_stop = sigmoid_inverse(1 - normalized_per_breath_loss),
    # i.e., when per-breath loss is low, target is high (model thinks "we're done").
    # Use squared-error on stop_logit (unbounded, before sigmoid).
    stop_calib = None
    for i, (d, pb_loss) in enumerate(zip(decisions[1:], per_breath_losses)):  # skip init decision
        # Per-batch per-breath loss is a scalar (mean reduction); broadcast to (B,).
        # Use the loss MAGNITUDE as the inverse-stop signal: high loss → don't stop.
        # Target: stop_logit large when loss is low. Use -loss as the supervised target.
        target = (pb_loss * -1.0).expand(d["stop_logit"].shape[0])
        diff = (d["stop_logit"] - target.detach())
        term = diff.square().mean()
        stop_calib = term if stop_calib is None else stop_calib + term
    if stop_calib is not None:
        avg_ctrl_loss = avg_ctrl_loss + stop_calib * 0.01

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


def multi_cycle_train_step(model, opt, batch_examples: List[MathExample], tok,
                           n_loops, fixed_len: int,
                           lookup_aux_weight: float = 0.0,
                           lookup_eq_token_id: int | None = None,
                           profile: bool = False):
    """Per-cycle forward+backward. Each outer cycle gets its own breathing pass.
    Losses are summed across cycles and normalized by num_cycles (equal-weight
    decomposition).

    n_loops: int (uniform) or list[int] of length >= n_cycles (per-cycle scheduling).
    For three-phase: pass [phase_a_loops, phase_c_loops, phase_c_loops, ...].

    lookup_aux_weight: if > 0, adds a cross-entropy loss on the model's lookup
    table at the "=" token position (cycle 0 only). Drives the table entries
    toward the model's actual operation directions during joint training.

    profile: when True, returns (loss, timings_dict) with keys
      {encode, forward, backward_step}. The phases are forced to sync at their
      boundaries (so timings are accurate but the lazy graph is partially
      materialized in slices instead of one big realize). Off by default.
    """
    import time as _time
    Tensor.training = True
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
    avg_loss = avg_loss + model.lookup_table.weight.square().mean() * 1e-6
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
        # Multi-cycle (sequential per problem for now)
        for ex in examples:
            prompt_ids = tok.encode(ex.problem).ids
            cycle_outs = multi_cycle_generate(model, tok, prompt_ids, n_loops=n_loops,
                                              n_cycles=n_cycles, max_new_per_cycle=max_new_per_cycle)
            last_text = tok.decode(cycle_outs[-1])
            full_text = " ".join(tok.decode(co) for co in cycle_outs)
            parsed = parse_int_answer(last_text)
            ok = (parsed == ex.answer)
            if ok:
                correct += 1
            rows.append((ex, parsed, full_text))
    return correct / max(1, len(examples)), rows
