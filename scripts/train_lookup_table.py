"""Train a 16×1024 lookup table on operation classification from the integrated
rep at "=" position. The table entries are the prototypes the controller would
use for spectral matching — we want to know whether they can be aligned with the
model's actual operation directions via a tiny supervised step.

Pipeline:
1. Cache the integrated representation at "=" for ~800 ARITH problems
2. Split 600 train / 200 eval, balanced across the 4 operations
3. Train: cosine_sim(rep, table_entries) → softmax → CE on the op-index
4. Eval: classification accuracy + per-op modal entry + purity + distinctness

Compare against the NCM baseline (op-mean prototypes, no training): if the
trained table achieves ≥ NCM accuracy AND each op locks onto its own modal
entry with high purity, the closed-loop foundation is empirically validated.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load
from tinygrad.nn.optim import Adam

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math


def cast_fp32(model):
    def _c(o, a):
        t = getattr(o, a)
        if t.dtype == dtypes.half:
            setattr(o, a, t.cast(dtypes.float).contiguous().realize())
    _c(model.embed, "weight"); _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv","bv","wo","bo","w_out","b_out"): _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq","bq","wk","bk","w_in","b_in"): _c(layer, a)


def named_state(model):
    sd = {"embed.weight": model.embed.weight, "embed_out": model.embed_out,
          "ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv","bv","wo","bo","w_out","b_out","in_ln_g","in_ln_b","post_ln_g","post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq","bq","wk","bk","w_in","b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    return sd


def load_ckpt(model, path):
    sd_ck = safe_load(path)
    targets = named_state(model)
    for name, dst in targets.items():
        src = sd_ck[name].to(dst.device).realize()
        if src.shape != dst.shape: src = src.reshape(dst.shape)
        if src.dtype != dst.dtype: src = src.cast(dst.dtype)
        dst.assign(src).realize()
    Device[Device.DEFAULT].synchronize()


def classify_op(problem: str) -> str:
    for ch in problem:
        if ch in "+-*/":
            return ch
    return "?"


def find_eq_pos(token_ids: list, tok) -> int:
    eq_ids = tok.encode("=").ids
    if not eq_ids:
        return len(token_ids) - 1
    target = eq_ids[-1]
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == target:
            return i
    return len(token_ids) - 1


def cache_reps(model, tok, problems, n_loops: int) -> tuple:
    """Forward each problem; return (reps NxD numpy float32, ops length-N list)."""
    Tensor.training = False
    reps = []
    ops = []
    for ex in problems:
        ids = tok.encode(ex.problem).ids
        eq_pos = find_eq_pos(ids, tok)
        toks = Tensor([ids], dtype=dtypes.int).realize()
        h = model(toks, n_loops)
        v = h[:, eq_pos, :].realize().numpy().reshape(-1).astype(np.float32)
        reps.append(v)
        ops.append(classify_op(ex.problem))
    return np.stack(reps), ops


def train_lookup(reps_train: np.ndarray, ops_train: list, n_entries: int = 16,
                 n_classes: int = 4, op_to_idx: dict = None, hidden: int = 1024,
                 epochs: int = 30, lr: float = 0.02, seed: int = 11) -> tuple:
    """Train a (n_entries × hidden) table such that cosine(rep, table) → logits over
    n_entries, sliced to n_classes for CE on op-index. The first n_classes entries
    are the "live" ops (one per op in op_to_idx); the remaining n_entries-n_classes
    are inactive prototypes (their gradient is zero unless picked up by the softmax).
    """
    rng = np.random.default_rng(seed)
    # Random orthogonal init for all 16 entries (matches the spec's prime basis)
    M = rng.standard_normal((hidden, n_entries))
    Q, _ = np.linalg.qr(M)
    init = Q.T  # (n_entries, hidden), rows orthonormal
    table = Tensor(init.astype(np.float32), requires_grad=True).realize()

    y_np = np.array([op_to_idx[o] for o in ops_train], dtype=np.int64)
    X = Tensor(reps_train.astype(np.float32)).realize()
    y = Tensor(y_np, dtype=dtypes.int).realize()

    opt = Adam([table], lr=lr)
    Tensor.training = True
    for epoch in range(epochs):
        opt.zero_grad()
        # Cosine similarity → logits over all n_entries
        x_n = X / (X.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        t_n = table / (table.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        logits_all = x_n @ t_n.T  # (N, n_entries)
        # CE only over the first n_classes entries (one per op)
        logits_cls = logits_all[:, :n_classes] * 10.0  # temperature scaling for sharper softmax
        loss = logits_cls.sparse_categorical_crossentropy(y)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            preds = logits_cls.argmax(axis=-1).realize().numpy()
            acc = float((preds == y_np).mean())
            print(f"    epoch {epoch+1:3d}: loss={float(loss.realize().numpy()):.4f}  train_acc={acc*100:.1f}%")
    Tensor.training = False
    return table.realize().numpy(), op_to_idx


def lookup_metrics(reps: np.ndarray, ops: list, table: np.ndarray,
                   n_classes: int, op_to_idx: dict) -> dict:
    """Diagnostic metrics for the trained table on held-out reps."""
    # Cosine sim → logits
    rn = reps / (np.linalg.norm(reps, axis=-1, keepdims=True) + 1e-12)
    tn = table / (np.linalg.norm(table, axis=-1, keepdims=True) + 1e-12)
    sims = rn @ tn.T              # (N, n_entries)
    best = sims.argmax(axis=1)    # (N,)
    pred_class_logits = sims[:, :n_classes]
    pred = pred_class_logits.argmax(axis=1)
    y = np.array([op_to_idx[o] for o in ops])
    acc = float((pred == y).mean())

    op_set = sorted(set(ops))
    op_to_best = {o: [int(best[i]) for i in range(len(ops)) if ops[i] == o] for o in op_set}
    modal = {o: int(np.bincount(op_to_best[o], minlength=table.shape[0]).argmax()) for o in op_set}
    purity = {o: float(np.bincount(op_to_best[o]).max() / max(len(op_to_best[o]), 1))
              for o in op_set}
    distinct = len(set(modal.values())) == len(op_set)
    return {
        "classification_acc": acc,
        "modal_entry_per_op": modal,
        "purity_per_op": purity,
        "distinct_modes_across_ops": distinct,
        "best_idx_distribution": {o: list(np.bincount(op_to_best[o], minlength=table.shape[0]))
                                  for o in op_set},
    }


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l3_ckpts/l3_spaced_step600.safetensors")
    n_per_op = getenv("N_PER_OP", 200)
    loop_sweep = [int(x) for x in getenv("LOOPS", "1,4,8").split(",")]

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    load_ckpt(model, ckpt)
    tok = load_tokenizer()

    raw = generate_math("ARITH", n_per_op * 12, seed=2024, digit_spacing=True)
    by_op = {"+": [], "-": [], "*": [], "/": []}
    for ex in raw:
        op = classify_op(ex.problem)
        if op in by_op and len(by_op[op]) < n_per_op:
            by_op[op].append(ex)
    op_to_idx = {"+": 0, "-": 1, "*": 2, "/": 3}
    n_classes = 4

    # Split per op: 75% train, 25% eval
    train, eval_ = [], []
    for o in ["+", "-", "*", "/"]:
        cut = int(0.75 * len(by_op[o]))
        train.extend(by_op[o][:cut])
        eval_.extend(by_op[o][cut:])
    print(f"=== Lookup table training on {os.path.basename(ckpt)} ===")
    print(f"train: {len(train)}  eval: {len(eval_)}")
    print()

    for n_loops in loop_sweep:
        print(f"=== N_LOOPS={n_loops} ===")
        t0 = time.perf_counter()
        reps_tr, ops_tr = cache_reps(model, tok, train, n_loops)
        reps_ev, ops_ev = cache_reps(model, tok, eval_, n_loops)
        print(f"  cached reps in {time.perf_counter() - t0:.1f}s")

        # NCM baseline (no training)
        means = {o: reps_tr[[i for i, x in enumerate(ops_tr) if x == o]].mean(axis=0)
                 for o in ["+", "-", "*", "/"]}
        rn_ev = reps_ev / (np.linalg.norm(reps_ev, axis=-1, keepdims=True) + 1e-12)
        means_arr = np.stack([means[o] for o in ["+", "-", "*", "/"]])
        mn = means_arr / (np.linalg.norm(means_arr, axis=-1, keepdims=True) + 1e-12)
        ncm_pred = (rn_ev @ mn.T).argmax(axis=1)
        ncm_y = np.array([op_to_idx[o] for o in ops_ev])
        ncm_acc = float((ncm_pred == ncm_y).mean())
        print(f"  NCM baseline (op-mean prototypes, no training): {ncm_acc*100:.1f}%")

        # Train the 16-entry lookup table
        print(f"  training 16×1024 table on {len(reps_tr)} examples...")
        table, _ = train_lookup(reps_tr, ops_tr, n_entries=16, n_classes=n_classes,
                                op_to_idx=op_to_idx, hidden=reps_tr.shape[1],
                                epochs=30, lr=0.02, seed=11)

        # Evaluate
        m = lookup_metrics(reps_ev, ops_ev, table, n_classes=n_classes, op_to_idx=op_to_idx)
        print(f"  TRAINED table classification acc: {m['classification_acc']*100:.1f}%")
        print(f"    (NCM baseline:               {ncm_acc*100:.1f}%)")
        print(f"  modal best-entry per op (over all 16 entries):")
        for o in ["+", "-", "*", "/"]:
            target_idx = op_to_idx[o]
            modal_idx = m['modal_entry_per_op'][o]
            purity = m['purity_per_op'][o]
            on_target = "✓" if modal_idx == target_idx else " "
            print(f"    {o} → entry {modal_idx:>2}  purity={purity:.2f}  "
                  f"target={target_idx}  {on_target}")
        print(f"  distinct modal entries across ops: {m['distinct_modes_across_ops']}")
        print(f"  best-entry distribution per op (first 6 of 16):")
        for o in ["+", "-", "*", "/"]:
            print(f"    {o}: {m['best_idx_distribution'][o][:6]} ...")
        print()


if __name__ == "__main__":
    main()
