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


def masked_forward_loss(model, tokens: Tensor, labels: Tensor, n_loops: int) -> Tensor:
    """Next-token CE loss, ignoring positions where label == -100.

    tokens: (B, T) int — full input
    labels: (B, T-1) int — targets for tokens[:, 1:], with -100 in masked positions
    """
    h = model(tokens, n_loops)                                 # (B, T, hidden)
    logits = (h @ model.embed_out).cast(dtypes.float)          # (B, T, vocab)
    pred = logits[:, :-1, :]                                   # (B, T-1, vocab)
    return pred.sparse_categorical_crossentropy(labels, ignore_index=-100, reduction="mean")


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


def generate_answer(model, prompt_ids: List[int], n_loops: int, max_new: int = 32,
                    eos_id: int = 0, vocab_active: int = 50277) -> List[int]:
    """Greedy generation from a problem-tokenized prompt. Stops on EOS or after
    max_new tokens. Returns the generated tokens (without the prompt).
    """
    Tensor.training = False
    out = list(prompt_ids)
    for _ in range(max_new):
        ctx = out[-model.cfg.max_seq_len:]
        toks = Tensor([ctx], dtype=dtypes.int).realize()
        h = model(toks, n_loops)                              # (1, S, hidden)
        last = h[:, -1, :]                                    # (1, hidden)
        logits = (last @ model.embed_out).cast(dtypes.float)
        logits = logits[:, :vocab_active]
        next_id = int(logits.argmax(axis=-1).realize().numpy()[0])
        out.append(next_id)
        if next_id == eos_id:
            break
    return out[len(prompt_ids):]


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


def multi_cycle_train_step(model, opt, batch_examples: List[MathExample], tok,
                           n_loops, fixed_len: int) -> float:
    """Per-cycle forward+backward. Each outer cycle gets its own breathing pass.
    Losses are summed across cycles and normalized by num_cycles (equal-weight
    decomposition).

    n_loops: int (uniform) or list[int] of length >= n_cycles (per-cycle scheduling).
    For three-phase: pass [phase_a_loops, phase_c_loops, phase_c_loops, ...].
    """
    Tensor.training = True
    opt.zero_grad()

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
    avg_loss = total / float(n_cycles)
    avg_loss.backward()
    opt.step()
    Device[Device.DEFAULT].synchronize()
    return float(avg_loss.numpy())


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

    cache_max_len: override the K/V buffer length for short sequences (e.g., 32 for
    L3-spaced arithmetic). Defaults to cfg.max_seq_len. Smaller values cut cache
    memory linearly, allowing larger batch_size within the GPU memory budget.

    n_loops can be int (uniform) or list (per-cycle). For three-phase eval pass
    [phase_a, phase_c, phase_c, ...].
    """
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


def accuracy_at_loops(model, tok, examples: List[MathExample], n_loops: int,
                      max_new: int = 48) -> Tuple[float, List[Tuple[MathExample, int | None, str]]]:
    """Greedy-generate an answer for each example at n_loops; compare parsed
    integer to ground truth. Returns (accuracy, list of (example, parsed, gen_text)).
    """
    correct = 0
    rows = []
    for ex in examples:
        # Prompt the model with the problem text + a leading space (matches training format).
        # The model has been trained: after the problem text, emit the gen text starting with " ".
        prompt_ids = tok.encode(ex.problem).ids
        gen_ids = generate_answer(model, prompt_ids, n_loops=n_loops, max_new=max_new)
        gen_text = tok.decode(gen_ids)
        parsed = parse_int_answer(gen_text)
        ok = (parsed == ex.answer)
        if ok:
            correct += 1
        rows.append((ex, parsed, gen_text))
    return correct / max(1, len(examples)), rows
