"""Per-checkpoint lookup-table eval — second axis of training signal.

Train a fresh 16×1024 cosine-similarity lookup table on operation classification
from the integrated rep at "=" position. Report held-out accuracy, per-op modal
entries, and on-target purity. Costs ~5-15 seconds per call depending on dataset
size — cheap enough to run every accuracy eval.

Tracks: when does op-discriminability saturate during training? Does it ever
drop (a sign of representation collapse)? For multi-step problems (L4+), do
the per-cycle operations remain separable?
"""
from __future__ import annotations
import time
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.optim import Adam

from mycelium.l3_data import generate_math, MathExample


_OP_TO_IDX = {"+": 0, "-": 1, "*": 2, "/": 3}
_OP_LIST = ["+", "-", "*", "/"]


def _classify_op(problem: str) -> str:
    for ch in problem:
        if ch in "+-*/":
            return ch
    return "?"


def _cache_eq_reps_batched(model, tok, examples: list[MathExample], n_loops: int,
                           batch_size: int = 64) -> tuple[np.ndarray, list[str]]:
    """Forward each example in length-bucketed batches; return (reps NxD, ops).

    All ARITH problems end with " =" — the eq position is the last token. We
    bucket by sequence length to avoid padding inside attention (no mask path
    in the standard forward) and batch within each bucket.
    """
    Tensor.training = False
    by_len: dict[int, list[tuple[MathExample, list[int]]]] = {}
    for ex in examples:
        ids = tok.encode(ex.problem).ids
        by_len.setdefault(len(ids), []).append((ex, ids))

    reps: list[np.ndarray] = []
    ops: list[str] = []
    for L, items in sorted(by_len.items()):
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            tokens_np = np.array([ids for _, ids in chunk], dtype=np.int32)
            toks_t = Tensor(tokens_np, dtype=dtypes.int).realize()
            h = model(toks_t, n_loops)                 # (B, L, hidden)
            h_np = h[:, -1, :].realize().numpy()        # (B, hidden) — eq position
            for i, (ex, _) in enumerate(chunk):
                reps.append(h_np[i].astype(np.float32))
                ops.append(_classify_op(ex.problem))
    return np.stack(reps), ops


def _train_table(reps_train: np.ndarray, y_train: np.ndarray,
                 hidden: int, n_classes: int = 4, n_entries: int = 16,
                 epochs: int = 30, lr: float = 0.02, seed: int = 11) -> np.ndarray:
    """Train a (n_entries × hidden) cosine-similarity table. Returns numpy table."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((hidden, n_entries))
    Q, _ = np.linalg.qr(M)                              # (hidden, n_entries)
    table = Tensor(Q.T.astype(np.float32), requires_grad=True).realize()
    X = Tensor(reps_train.astype(np.float32)).realize()
    y = Tensor(y_train.astype(np.int32), dtype=dtypes.int).realize()
    opt = Adam([table], lr=lr)
    Tensor.training = True
    for _ in range(epochs):
        opt.zero_grad()
        x_n = X / (X.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        t_n = table / (table.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        logits = (x_n @ t_n.T)[:, :n_classes] * 10.0
        logits.sparse_categorical_crossentropy(y).backward()
        opt.step()
    Tensor.training = False
    return table.realize().numpy()


def lookup_eval(model, tok, n_loops: int = 8,
                n_per_op_train: int = 150, n_per_op_eval: int = 50,
                seed: int = 2024, batch_size: int = 64,
                verbose: bool = False) -> dict:
    """Train + eval a fresh lookup table. Returns metrics dict.

    Total cost: ~5-15s for 600 train + 200 eval problems at n_loops=8.
    """
    t0 = time.perf_counter()

    # Generate balanced problems
    raw = generate_math("ARITH", (n_per_op_train + n_per_op_eval) * 12,
                        seed=seed, digit_spacing=True)
    by_op: dict[str, list[MathExample]] = {o: [] for o in _OP_LIST}
    for ex in raw:
        op = _classify_op(ex.problem)
        if op in by_op and len(by_op[op]) < n_per_op_train + n_per_op_eval:
            by_op[op].append(ex)
    train: list[MathExample] = []
    eval_: list[MathExample] = []
    for o in _OP_LIST:
        train.extend(by_op[o][:n_per_op_train])
        eval_.extend(by_op[o][n_per_op_train:n_per_op_train + n_per_op_eval])

    # Cache reps
    reps_tr, ops_tr = _cache_eq_reps_batched(model, tok, train, n_loops, batch_size)
    reps_ev, ops_ev = _cache_eq_reps_batched(model, tok, eval_, n_loops, batch_size)
    Device[Device.DEFAULT].synchronize()
    t_reps = time.perf_counter() - t0

    # NCM baseline
    means = np.stack([reps_tr[[i for i, x in enumerate(ops_tr) if x == o]].mean(axis=0)
                      for o in _OP_LIST])
    rn = reps_ev / (np.linalg.norm(reps_ev, axis=-1, keepdims=True) + 1e-12)
    mn = means / (np.linalg.norm(means, axis=-1, keepdims=True) + 1e-12)
    ncm_pred = (rn @ mn.T).argmax(axis=1)
    y_ev = np.array([_OP_TO_IDX[o] for o in ops_ev])
    ncm_acc = float((ncm_pred == y_ev).mean())

    # Train + eval table
    y_tr = np.array([_OP_TO_IDX[o] for o in ops_tr])
    table = _train_table(reps_tr, y_tr, hidden=reps_tr.shape[1])
    tn = table / (np.linalg.norm(table, axis=-1, keepdims=True) + 1e-12)
    sims = rn @ tn.T                                    # (N_eval, n_entries)
    pred_class = sims[:, :4].argmax(axis=1)
    cls_acc = float((pred_class == y_ev).mean())
    best_entry = sims.argmax(axis=1)
    modal = {}
    purity = {}
    on_target = 0
    for o in _OP_LIST:
        bs = [int(best_entry[i]) for i in range(len(ops_ev)) if ops_ev[i] == o]
        if not bs:
            continue
        counts = np.bincount(bs, minlength=table.shape[0])
        modal[o] = int(counts.argmax())
        purity[o] = float(counts.max() / len(bs))
        if modal[o] == _OP_TO_IDX[o]:
            on_target += 1
    distinct = len(set(modal.values())) == len(modal)
    elapsed = time.perf_counter() - t0

    metrics = {
        "n_loops": n_loops,
        "ncm_acc": ncm_acc,
        "trained_acc": cls_acc,
        "modal_entry_per_op": modal,
        "purity_per_op": purity,
        "on_target_count": on_target,        # 0..4
        "distinct_modes_across_ops": distinct,
        "elapsed_s": elapsed,
        "elapsed_reps_s": t_reps,
        "n_train": len(train),
        "n_eval": len(eval_),
    }
    if verbose:
        print(f"  lookup-eval @ A={n_loops}: trained={cls_acc*100:.1f}%  ncm={ncm_acc*100:.1f}%  "
              f"on-target={on_target}/4  purity={ {o: f'{purity[o]:.2f}' for o in purity} }  "
              f"({elapsed:.1f}s)")
    return metrics
