"""Verification head Phase 1: frozen-model MLP probe on v6 reps.

Foundational question: do the v6 integrated representations CONTAIN enough
information to distinguish "problem = correct_answer" from "problem = wrong_answer"?

Procedure:
  1. Freeze the v6 ckpt.
  2. Generate N (problem, correct, wrong) triples (1200 default). Wrong answers
     are one-digit corruptions of gold.
  3. For each input ("problem = candidate"), forward through breathe_controlled
     with SINE_TEMP=1 matching training. Extract integrated_per_breath[-1] at the
     last token position → 1024d rep.
  4. Train a 2-layer MLP (1024 → 512 → 1) on (rep, is_correct) with BCE loss.
     ~520K params. Hold out 20% for eval.
  5. Report eval accuracy + AUC.

Decision rule:
  - eval acc > 0.80: strong signal — verification information is in the rep.
                    Phase 2 (integrated training) will work.
  - eval acc 0.60-0.80: weak signal — possible but need bigger head or joint training.
  - eval acc ~0.50: NO signal — reps don't encode verification. Need deeper fix.

Usage:
    DEV=PCI+AMD CKPT=.cache/arith_mixed_ckpts/arith_mixed_v6_step300.safetensors \\
        SINE_TEMP=1 SINE_TEMP_MAX=2.0 SINE_TEMP_MIN=0.7 \\
        .venv/bin/python scripts/probe_verification_mlp.py
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, space_digits
from mycelium.controller import Notebook


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


def corrupt_answer_digit(answer_int: int, rng: random.Random) -> int:
    s = str(answer_int)
    if len(s) == 1:
        choices = [d for d in range(10) if d != answer_int]
        return rng.choice(choices)
    pos = rng.randrange(len(s))
    orig = int(s[pos])
    delta_choices = [d for d in (-3, -2, -1, 1, 2, 3) if 0 <= orig + d <= 9]
    delta = rng.choice(delta_choices)
    new_digit = orig + delta
    new_s = s[:pos] + str(new_digit) + s[pos+1:]
    if new_s[0] == '0' and len(new_s) > 1:
        new_s = '1' + new_s[1:]
    return int(new_s)


def main():
    ckpt = getenv("CKPT", ".cache/arith_mixed_ckpts/arith_mixed_v6_step300.safetensors")
    N = getenv("N", 600)              # generate 2N labeled pairs (correct + wrong)
    MAX_LOOPS = getenv("MAX_LOOPS", 8)
    SEED = getenv("SEED", 42)
    BATCH = getenv("BATCH", 32)
    MLP_HIDDEN = getenv("MLP_HIDDEN", 512)
    MLP_HIDDEN2 = getenv("MLP_HIDDEN2", 0)   # 0 = 2-layer MLP, >0 = 3-layer
    MLP_STEPS = getenv("MLP_STEPS", 200)
    MLP_LR = float(getenv("MLP_LR", "1e-3"))
    EVAL_FRAC = float(getenv("EVAL_FRAC", "0.2"))
    # Breath aggregation: "last" (final breath only), "concat" (concat all 8 breaths),
    # or "mean" (average across breaths)
    AGGREGATE = getenv("AGGREGATE", "last")

    cfg = Config()
    input_dim = cfg.hidden * (MAX_LOOPS if AGGREGATE == "concat" else 1)
    print(f"=== verification head Phase 1: MLP probe on {os.path.basename(ckpt)} ===")
    print(f"N={N} (→ {2*N} pairs)  MAX_LOOPS={MAX_LOOPS}  BATCH={BATCH}")
    print(f"AGGREGATE={AGGREGATE}  input_dim={input_dim}")
    print(f"MLP_HIDDEN={MLP_HIDDEN}  MLP_HIDDEN2={MLP_HIDDEN2}  "
          f"MLP_STEPS={MLP_STEPS}  MLP_LR={MLP_LR}")
    print(f"SINE_TEMP={os.environ.get('SINE_TEMP', '0')}  "
          f"SINE_TEMP_MAX={os.environ.get('SINE_TEMP_MAX', 'n/a')}  "
          f"SINE_TEMP_MIN={os.environ.get('SINE_TEMP_MIN', 'n/a')}\n")

    # === Step 1: generate input strings ===
    raw = generate_math("ARITH_MIXED", N + 100, seed=SEED, digit_spacing=False)[:N]
    rng = random.Random(SEED + 13)
    correct_strs = []
    wrong_strs = []
    for ex in raw:
        wrong_ans = corrupt_answer_digit(ex.answer, rng)
        correct_strs.append(space_digits(f"{ex.problem} {ex.answer}."))
        wrong_strs.append(space_digits(f"{ex.problem} {wrong_ans}."))
    print(f"Generated {len(correct_strs)} correct + {len(wrong_strs)} wrong inputs.")
    print(f"Sample correct: {correct_strs[0]!r}")
    print(f"Sample wrong:   {wrong_strs[0]!r}\n")

    # === Step 2: load model ===
    print("Loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    info = model.load_state_dict(safe_load(ckpt), strict=False)
    print(f"  loaded. missing={len(info['missing'])} unexpected={len(info['unexpected'])}\n")
    tok = load_tokenizer()
    Tensor.training = False

    # === Step 3: forward pass — extract rep at last token, per breath ===
    def tokenize_batch(strs, fixed_len=64):
        ids_list = [tok.encode(s).ids for s in strs]
        max_len = max(fixed_len, max(len(ids) for ids in ids_list))
        out = np.zeros((len(strs), max_len), dtype=np.int32)
        last_pos = []
        for b, ids in enumerate(ids_list):
            L = min(len(ids), max_len)
            out[b, :L] = ids[:L]
            last_pos.append(L - 1)
        return Tensor(out, dtype=dtypes.int).realize(), last_pos

    all_strs = correct_strs + wrong_strs
    all_labels = np.array([1.0] * len(correct_strs) + [0.0] * len(wrong_strs), dtype=np.float32)

    print(f"Extracting reps for {len(all_strs)} inputs (B={BATCH})...")
    t0 = time.perf_counter()
    all_reps = np.zeros((len(all_strs), input_dim), dtype=np.float32)
    for batch_start in range(0, len(all_strs), BATCH):
        batch_end = min(batch_start + BATCH, len(all_strs))
        batch_strs = all_strs[batch_start:batch_end]
        tokens, last_pos = tokenize_batch(batch_strs)
        notebook = Notebook()
        _, _, _, _, ipb = model.breathe_controlled(
            tokens, max_loops=MAX_LOOPS, notebook=notebook, return_per_breath_reps=True
        )
        # Materialize all breaths' reps to numpy
        breath_reps_np = [r.numpy() for r in ipb]  # list of (B, T, hidden)
        H = cfg.hidden
        for b in range(batch_end - batch_start):
            lp = last_pos[b]
            if AGGREGATE == "last":
                all_reps[batch_start + b] = breath_reps_np[-1][b, lp, :]
            elif AGGREGATE == "mean":
                accum = np.zeros(H, dtype=np.float32)
                for l in range(MAX_LOOPS):
                    accum += breath_reps_np[l][b, lp, :]
                all_reps[batch_start + b] = accum / MAX_LOOPS
            elif AGGREGATE == "concat":
                for l in range(MAX_LOOPS):
                    all_reps[batch_start + b, l*H:(l+1)*H] = breath_reps_np[l][b, lp, :]
            else:
                raise ValueError(f"unknown AGGREGATE={AGGREGATE}")
        if batch_start % (10 * BATCH) == 0:
            print(f"  {batch_end}/{len(all_strs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"  done in {time.perf_counter()-t0:.0f}s. reps shape: {all_reps.shape}")

    # === Step 4: split + train MLP ===
    rng_np = np.random.default_rng(SEED + 7)
    n_total = len(all_strs)
    perm = rng_np.permutation(n_total)
    n_eval = int(n_total * EVAL_FRAC)
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:]
    train_x = Tensor(all_reps[train_idx], dtype=dtypes.float).realize()
    train_y = Tensor(all_labels[train_idx], dtype=dtypes.float).realize()
    eval_x = Tensor(all_reps[eval_idx], dtype=dtypes.float).realize()
    eval_y = Tensor(all_labels[eval_idx], dtype=dtypes.float).realize()
    print(f"\nSplit: {len(train_idx)} train, {len(eval_idx)} eval")

    # MLP: input_dim → MLP_HIDDEN → [MLP_HIDDEN2 →] 1 (BCE on logit)
    deep = int(MLP_HIDDEN2) > 0
    w1 = (Tensor.randn(input_dim, MLP_HIDDEN, dtype=dtypes.float) * (1.0 / (input_dim ** 0.5))).contiguous()
    b1 = Tensor.zeros(MLP_HIDDEN, dtype=dtypes.float).contiguous()
    if deep:
        w2 = (Tensor.randn(MLP_HIDDEN, MLP_HIDDEN2, dtype=dtypes.float) * (1.0 / (MLP_HIDDEN ** 0.5))).contiguous()
        b2 = Tensor.zeros(MLP_HIDDEN2, dtype=dtypes.float).contiguous()
        w3 = (Tensor.randn(MLP_HIDDEN2, 1, dtype=dtypes.float) * 0.02).contiguous()
        b3 = Tensor.zeros(1, dtype=dtypes.float).contiguous()
        params = [w1, b1, w2, b2, w3, b3]
    else:
        w2 = (Tensor.randn(MLP_HIDDEN, 1, dtype=dtypes.float) * 0.02).contiguous()
        b2 = Tensor.zeros(1, dtype=dtypes.float).contiguous()
        params = [w1, b1, w2, b2]
    opt = Adam(params, lr=MLP_LR)

    def forward(x):
        h = (x @ w1 + b1).gelu()
        if deep:
            h = (h @ w2 + b2).gelu()
            logits = (h @ w3 + b3).reshape(-1)
        else:
            logits = (h @ w2 + b2).reshape(-1)
        return logits

    arch_str = f"{input_dim} → {MLP_HIDDEN}" + (f" → {MLP_HIDDEN2}" if deep else "") + " → 1"
    print(f"\nTraining MLP ({arch_str}) for {MLP_STEPS} steps...")
    Tensor.training = True
    n_train = len(train_idx)
    mlp_batch = 64
    losses = []
    for step in range(MLP_STEPS):
        idx = rng_np.integers(0, n_train, size=mlp_batch)
        x_batch = Tensor(all_reps[train_idx[idx]], dtype=dtypes.float).realize()
        y_batch = Tensor(all_labels[train_idx[idx]], dtype=dtypes.float).realize()
        logits = forward(x_batch)
        loss = ((logits.maximum(0.0) - logits * y_batch + (1.0 + (-logits.abs()).exp()).log()).mean())  # BCE-with-logits
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == MLP_STEPS - 1:
            losses.append((step, float(loss.realize().numpy())))
            print(f"  step {step:3d}  loss={losses[-1][1]:.4f}", flush=True)
    Tensor.training = False

    # === Step 5: eval ===
    print(f"\nEval on {len(eval_idx)} held-out:")
    logits = forward(eval_x).realize().numpy()  # (n_eval,)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.float32)
    labels_np = all_labels[eval_idx]
    acc = float((preds == labels_np).mean())
    pos_correct = int(((preds == 1) & (labels_np == 1)).sum())
    pos_total = int(labels_np.sum())
    neg_correct = int(((preds == 0) & (labels_np == 0)).sum())
    neg_total = int((1 - labels_np).sum())
    # Quick AUC: rank correlation between probs and labels
    sort_idx = np.argsort(-probs)
    sorted_labels = labels_np[sort_idx]
    cumpos = np.cumsum(sorted_labels)
    n_pos = sorted_labels.sum(); n_neg = len(labels_np) - n_pos
    auc = float((cumpos[sorted_labels == 0].sum()) / (n_pos * n_neg + 1e-9)) if n_pos > 0 and n_neg > 0 else 0.5

    print(f"  accuracy: {acc:.3f}")
    print(f"  correct-detection: {pos_correct}/{pos_total} = {pos_correct/pos_total:.3f}")
    print(f"  wrong-detection:   {neg_correct}/{neg_total} = {neg_correct/neg_total:.3f}")
    print(f"  AUC: {auc:.3f}")
    print()

    if acc > 0.80:
        print(f"  → STRONG SIGNAL ({acc*100:.1f}% acc) — verification info IS in v6 reps.")
        print(f"     Phase 2 (integrated training) will work. Greenlight.")
    elif acc > 0.60:
        print(f"  → WEAK SIGNAL ({acc*100:.1f}% acc) — present but limited. Possible avenues:")
        print(f"     larger head, joint fine-tuning, multi-breath aggregation.")
    else:
        print(f"  → NO SIGNAL ({acc*100:.1f}% acc) — reps don't encode verification.")
        print(f"     Need to train the TRANSFORMER to produce verification-encoding reps.")


if __name__ == "__main__":
    main()
