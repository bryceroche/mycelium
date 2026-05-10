"""Lookup table diagnostic — does π-cycled rotation produce distinguishable
spectral signals per arithmetic operation, or do operations mush together?

For each ARITH problem (digit-spaced), forward through the model, take the
integrated hidden state at the "=" token, and group by operation (+, -, *, /).

Headline question: do problems with the same operation cluster together in
representation space, while different operations stay separated? If yes →
the breathing produces real spectral structure that a lookup table can index.
If no → the closed loop has no foundation; building the controller would be
training on noise.

We also test a 16-entry random-orthogonal lookup table to check the
operation→best-match-entry mapping consistency. No training of the table —
this is purely diagnostic.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

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
    """Map a problem string to its arithmetic operation (+, -, *, /)."""
    for ch in problem:
        if ch in "+-*/":
            return ch
    return "?"


def find_eq_pos(token_ids: list, tok) -> int:
    """Return the index of the '=' token in the prompt. Falls back to last position."""
    eq_ids = tok.encode("=").ids
    if not eq_ids:
        return len(token_ids) - 1
    target = eq_ids[-1]
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == target:
            return i
    return len(token_ids) - 1


def find_op_pos(token_ids: list, tok) -> int:
    """Return the index of the operator token (+, -, *, /). Falls back to mid-prompt."""
    for op_str in [" +", " -", " *", " /", "+", "-", "*", "/"]:
        op_ids = tok.encode(op_str).ids
        if not op_ids: continue
        target = op_ids[-1]
        for i in range(len(token_ids)):
            if token_ids[i] == target:
                return i
    return len(token_ids) // 2


def get_reps(model, tok, problems, n_loops: int, position: str = "eq") -> tuple:
    """Forward each problem; return (reps NxD numpy, ops length-N list).

    position: 'eq' = position of '=' token (final pre-gen).
              'op' = position of operator token.
              'last_operand' = right after the second operand finishes.
    """
    Tensor.training = False
    reps = []
    ops = []
    for ex in problems:
        ids = tok.encode(ex.problem).ids
        if position == "eq":
            pos = find_eq_pos(ids, tok)
        elif position == "op":
            pos = find_op_pos(ids, tok)
        elif position == "last_operand":
            # right before "=", which is the last digit of the second operand
            eq_p = find_eq_pos(ids, tok)
            pos = max(0, eq_p - 1)
        else:
            pos = len(ids) - 1
        toks = Tensor([ids], dtype=dtypes.int).realize()
        h = model(toks, n_loops)                      # (1, T, hidden), post-LN
        v = h[:, pos, :].realize().numpy().reshape(-1)  # (hidden,)
        reps.append(v.astype(np.float64))
        ops.append(classify_op(ex.problem))
    return np.stack(reps), ops


def ncm_loo_accuracy(reps: np.ndarray, ops: list) -> float:
    """Leave-one-out nearest-class-mean (cosine) classifier accuracy.
    Honest test of 'is the op info linearly readable from these reps' — robust to
    random orthogonal directions, only checks whether class means are separable."""
    op_set = sorted(set(ops))
    op_idx = {o: i for i, o in enumerate(op_set)}
    y = np.array([op_idx[o] for o in ops])
    N = len(ops)
    correct = 0
    # Precompute group sums + counts so we can do leave-one-out by subtraction
    sums = {o: np.zeros(reps.shape[1]) for o in op_set}
    counts = {o: 0 for o in op_set}
    for v, o in zip(reps, ops):
        sums[o] += v
        counts[o] += 1
    for i in range(N):
        oi = ops[i]
        loo_sums = {o: (sums[o] - reps[i]) if o == oi else sums[o] for o in op_set}
        loo_counts = {o: (counts[o] - 1) if o == oi else counts[o] for o in op_set}
        means = {o: loo_sums[o] / max(loo_counts[o], 1) for o in op_set}
        # cosine sim
        v_n = reps[i] / (np.linalg.norm(reps[i]) + 1e-12)
        best_op = max(op_set, key=lambda o: float(v_n @ (means[o] / (np.linalg.norm(means[o]) + 1e-12))))
        if best_op == oi:
            correct += 1
    return correct / N


def cos(a, b):
    """Cosine similarity between vectors / batches. Inputs (D,) or (N, D)."""
    a = np.asarray(a); b = np.asarray(b)
    if a.ndim == 1: a = a[None, :]
    if b.ndim == 1: b = b[None, :]
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def cluster_metrics(reps: np.ndarray, ops: list) -> dict:
    """Within-op vs between-op cosine separation."""
    op_set = sorted(set(ops))
    by_op = {o: reps[[i for i, x in enumerate(ops) if x == o]] for o in op_set}
    means = {o: by_op[o].mean(axis=0) for o in op_set}

    within_sims = []
    for o in op_set:
        m = means[o]
        sims = cos(by_op[o], m)[:, 0]   # each member vs its op mean
        within_sims.extend(sims.tolist())

    between_sims = []
    for i, o1 in enumerate(op_set):
        for o2 in op_set[i+1:]:
            between_sims.append(float(cos(means[o1], means[o2])[0, 0]))

    within = float(np.mean(within_sims))
    between = float(np.mean(between_sims))
    # Discriminability: how much closer to your own mean than to other ops' means.
    # >0 means within-op tighter than between-op (real clustering).
    discrim = within - between
    return {
        "op_set": op_set,
        "n_per_op": {o: int(by_op[o].shape[0]) for o in op_set},
        "within_op_mean_cos": within,
        "between_op_mean_cos": between,
        "discriminability": discrim,
        "means": means,
    }


def random_orthogonal(n: int, d: int, seed: int) -> np.ndarray:
    """Random orthogonal n x d matrix (rows orthonormal). n <= d required."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, n))
    Q, _ = np.linalg.qr(M)
    return Q.T  # (n, d)


def lookup_diagnostic(reps: np.ndarray, ops: list, n_entries: int = 16,
                      seed: int = 7) -> dict:
    """Match each rep against a 16-entry random-orthogonal lookup table.
    Report op → best-match-entry distribution."""
    table = random_orthogonal(n_entries, reps.shape[1], seed)  # (16, D)
    sims = cos(reps, table)                                    # (N, 16)
    best = sims.argmax(axis=1)                                 # (N,)
    op_set = sorted(set(ops))
    op_to_best = {o: [int(best[i]) for i in range(len(ops)) if ops[i] == o] for o in op_set}

    # Modal entry per op + purity (fraction of problems mapped to the modal entry)
    modal = {}
    purity = {}
    for o in op_set:
        bs = op_to_best[o]
        if not bs:
            modal[o] = -1; purity[o] = 0.0; continue
        counts = np.bincount(bs, minlength=n_entries)
        modal[o] = int(counts.argmax())
        purity[o] = float(counts.max() / len(bs))

    # Are the modal entries DIFFERENT across ops? (good = yes)
    modal_entries = set(modal.values())
    distinct_modes = len(modal_entries) == len(op_set)

    return {
        "modal_entry_per_op": modal,
        "purity_per_op": purity,
        "distinct_modes_across_ops": distinct_modes,
        "n_entries": n_entries,
        "best_idx_distribution": {o: list(np.bincount(op_to_best[o], minlength=n_entries))
                                  for o in op_set},
    }


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l3_ckpts/l3_spaced_step600.safetensors")
    n_per_op_target = getenv("N_PER_OP", 50)
    loop_sweep = [int(x) for x in getenv("LOOPS", "1,2,4,8").split(",")]

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    load_ckpt(model, ckpt)
    tok = load_tokenizer()

    # Generate enough problems to get N_PER_OP per operation.
    # ARITH has 6 generators (+, -, *, *2, *3, /2) → ops {+, -, *, /}.
    # */2/3 all classify as "*". /2 is the only "/". Generate plenty and bucket.
    raw = generate_math("ARITH", n_per_op_target * 12, seed=2024, digit_spacing=True)
    by_op = {"+": [], "-": [], "*": [], "/": []}
    for ex in raw:
        op = classify_op(ex.problem)
        if op in by_op and len(by_op[op]) < n_per_op_target:
            by_op[op].append(ex)
    problems = []
    for o in ["+", "-", "*", "/"]:
        problems.extend(by_op[o])
    print(f"=== Lookup-table diagnostic on {os.path.basename(ckpt)} ===")
    print(f"problems per op: { {o: len(by_op[o]) for o in by_op} } total={len(problems)}\n")

    print("=== Sample problems per operation ===")
    for o in by_op:
        sample = by_op[o][0].problem if by_op[o] else "?"
        print(f"  {o}: {sample!r}")
    print()

    # Per-position scan first: for each candidate position, where does op live?
    print("=" * 70)
    print("PER-POSITION SCAN — does operation info live somewhere other than '='?")
    print("=" * 70)
    print(f"  {'position':<14} {'n_loops':>7} {'discrim':>8} {'NCM acc':>8} {'baseline':>9}")
    baseline = 1.0 / 4  # 4-class chance
    for position in ["op", "last_operand", "eq"]:
        for n_loops in [1, 8]:
            reps, ops = get_reps(model, tok, problems, n_loops, position=position)
            clust = cluster_metrics(reps, ops)
            ncm = ncm_loo_accuracy(reps, ops)
            print(f"  {position:<14} {n_loops:>7} {clust['discriminability']:>+8.4f} "
                  f"{ncm*100:>7.1f}% {baseline*100:>8.1f}%")
    print()

    print("=" * 70)
    print("LOOP SWEEP at '=' position (what feeds generation)")
    print("=" * 70)
    for n_loops in loop_sweep:
        t0 = time.perf_counter()
        reps, ops = get_reps(model, tok, problems, n_loops, position="eq")
        elapsed = time.perf_counter() - t0
        clust = cluster_metrics(reps, ops)
        ncm = ncm_loo_accuracy(reps, ops)
        lk = lookup_diagnostic(reps, ops, n_entries=16, seed=7)

        print(f"--- N_LOOPS={n_loops}  ({elapsed:.1f}s) ---")
        print(f"  within-op mean cos:  {clust['within_op_mean_cos']:+.4f}")
        print(f"  between-op mean cos: {clust['between_op_mean_cos']:+.4f}")
        print(f"  discriminability:    {clust['discriminability']:+.4f}")
        print(f"  NCM-LOO accuracy:    {ncm*100:.1f}%   (4-class chance = 25%)")
        print(f"  pairwise op-mean cos:")
        ops_list = clust["op_set"]
        for i, o1 in enumerate(ops_list):
            for o2 in ops_list[i+1:]:
                c = float(cos(clust["means"][o1], clust["means"][o2])[0, 0])
                print(f"    {o1} <-> {o2}: {c:+.4f}")
        print(f"  16-entry orthogonal lookup table:")
        print(f"    modal best-entry per op: {lk['modal_entry_per_op']}")
        print(f"    purity (frac on modal):  { {o: f'{v:.2f}' for o, v in lk['purity_per_op'].items()} }")
        print(f"    distinct modes across ops: {lk['distinct_modes_across_ops']}")
        print()


if __name__ == "__main__":
    main()
