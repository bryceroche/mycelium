"""Training step + accuracy eval for math curriculum (L3, L4, L4.5).

Single-cycle path (L3): standard masked-loss CE, one forward pass per step.
Multi-cycle path (L4+): per-cycle forward passes — each outer cycle is its own
breathe-then-speak event. Equal-weighted loss across cycles (equal-reward
decomposition for teacher-forced training).
"""
from typing import List, Tuple
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.l3_data import MathExample, encode_example, encode_cycles, parse_int_answer, collate, SEP
from mycelium.lookup_table import op_label_from_text, find_eq_position


# JIT-train cache: (id(model), id(opt), n_loops_tuple, fixed_len, B) → compiled fn
# The canonical tinygrad pattern wraps the WHOLE step (forward + backward + opt.step)
# in one TinyJit. Outputs are scalar Tensor losses; .numpy() returns the value.
_JIT_TRAIN_CACHE: dict = {}


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

        @TinyJit
        def _step(tokens0, labels0, eq_mask, op_labels):
            opt.zero_grad()
            final_h, match_weights, _ = model.breathe_with_lookup(tokens0, nl0)
            logits = (final_h @ model.embed_out).cast(dtypes.float)
            pred = logits[:, :-1, :]
            main_ce = pred.sparse_categorical_crossentropy(labels0, ignore_index=-100, reduction="mean")
            last_mw = match_weights[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(op_labels, ignore_index=-100, reduction="mean")
            l2_reg = model.lookup_table.weight.square().mean() * 1e-6
            total = main_ce + aw * aux_ce + l2_reg
            total.backward()
            opt.step()
            return total.realize()

    elif n_cycles == 2:
        nl0 = int(n_loops_per_cycle[0])
        nl1 = int(n_loops_per_cycle[1])

        @TinyJit
        def _step(tokens0, labels0, tokens1, labels1, eq_mask, op_labels):
            opt.zero_grad()
            # Cycle 0 — breathe_with_lookup (heavy, provides match weights for aux)
            final0, mw, _ = model.breathe_with_lookup(tokens0, nl0)
            logits0 = (final0 @ model.embed_out).cast(dtypes.float)
            pred0 = logits0[:, :-1, :]
            main_ce0 = pred0.sparse_categorical_crossentropy(labels0, ignore_index=-100, reduction="mean")
            # Cycle 1 — plain forward (light, just CE on the execution gen)
            final1 = model(tokens1, nl1)
            logits1 = (final1 @ model.embed_out).cast(dtypes.float)
            pred1 = logits1[:, :-1, :]
            main_ce1 = pred1.sparse_categorical_crossentropy(labels1, ignore_index=-100, reduction="mean")
            # Aux CE from cycle 0's last-breath match weights
            last_mw = mw[-1]
            gathered = (last_mw.cast(dtypes.float) * eq_mask).sum(axis=1)
            logits_aux = gathered[:, :4] * 10.0
            aux_ce = logits_aux.sparse_categorical_crossentropy(op_labels, ignore_index=-100, reduction="mean")
            l2_reg = model.lookup_table.weight.square().mean() * 1e-6
            avg_main = (main_ce0 + main_ce1) / 2.0
            total = avg_main + aw * aux_ce + l2_reg
            total.backward()
            opt.step()
            return total.realize()

    else:
        raise NotImplementedError(f"JIT-train n_cycles={n_cycles} not implemented (only 1 and 2)")

    _JIT_TRAIN_CACHE[key] = _step
    print(f"[JIT] compiled in {_t_jit.perf_counter() - _jit_compile_start:.1f}s "
          f"(cache size={len(_JIT_TRAIN_CACHE)})", flush=True)
    return _step


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
    return pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")


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
                          n_classes: int = 4, profile: bool = False,
                          compute_penalty: float = 0.0):
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

    # Per-breath, PER-EXAMPLE op-CE losses (shape (B,) each). The prior implementation
    # used reduction="mean" → scalar, which made the stop-calibration target identical
    # across examples and prevented the stop head from learning per-problem
    # differentiation. With reduction="none" each example gets its own loss value,
    # so the stop calibration can supervise "easy problems → stop high, hard → stop low."
    per_breath_per_ex_losses = []
    for mw in match_weights:
        gathered = (mw.cast(dtypes.float) * eq_mask).sum(axis=1)        # (B, n_entries)
        logits = gathered[:, :n_classes] * 10.0
        per_ex = logits.sparse_categorical_crossentropy(y, ignore_index=-100, reduction="none")  # (B,)
        per_breath_per_ex_losses.append(per_ex)

    # Main ctrl loss: average over breaths and examples (matches prior scalar behavior).
    total = per_breath_per_ex_losses[0].mean()
    for l in per_breath_per_ex_losses[1:]:
        total = total + l.mean()
    avg_ctrl_loss = total / float(len(per_breath_per_ex_losses))

    # Auxiliary: PER-EXAMPLE stop calibration. The stop_logit at breath l should be
    # HIGH when this example's per-breath lookup-CE is low (model says "I've got this
    # one") and LOW when it's still high. Squared-error pull toward -loss as target.
    # Per-example targets are the key: with scalar batch-mean targets (prior code),
    # the stop head couldn't learn per-problem differentiation, which we verified
    # empirically (std on stop_logit was ~0.27 across 32 diverse problems).
    stop_calib = None
    for i, (d, pb_loss) in enumerate(zip(decisions[1:], per_breath_per_ex_losses)):
        target = pb_loss * -1.0                          # (B,) — per example
        diff = d["stop_logit"] - target.detach()         # (B,)
        term = diff.square().mean()
        stop_calib = term if stop_calib is None else stop_calib + term
    if stop_calib is not None:
        avg_ctrl_loss = avg_ctrl_loss + stop_calib * 0.01

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
