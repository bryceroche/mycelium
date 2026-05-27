"""v81 Phase 2 — masking audit (CRITICAL).

The eval bug found 2026-05-25/26 was a missing `notebook_pool_mask` at eval —
that silently invalidated 9 iterations of CE chasing. Before training any v81
multi-head WaistController, we MUST verify that masking is airtight.

Audit protocol (BINARY pass/fail):

1. Load any v80 ckpt for plumbing (state_dict has the WaistController shape).
2. Build two synthetic inputs:
     A. [prompt, gold_full_target, EOS, zeros]
     B. [prompt, RANDOM_GARBAGE_target, EOS, zeros]
   Same prompt; only the answer-span tokens differ.
3. Forward both through `breathe_with_lookup` + `WaistController.forward` with
   the SAME kv_mask (1.0 at prompt range, 0.0 elsewhere) and notebook_pool_mask.
4. Compare predictions at:
     - position `prompt_len - 1` (predicts first answer token)
     - position `prompt_len - 1 + 5` (predicts 6th answer token, 5 tokens into answer)
   If they're identical (max abs diff < 1e-4 on logits) -> PASS.
   If they differ -> FAIL = masking has a leak.

We test this for BOTH the training-time forward pattern AND the eval-time forward
pattern, since the two had diverged historically (eval missed notebook_pool_mask
for 9 iterations).

Run:
    /home/bryce/mycelium/.venv/bin/python scripts/diag_v81_masking_audit.py

Exits non-zero on audit FAIL.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mirror the env stack from v80_smoke_train.sh BEFORE importing mycelium.
os.environ.setdefault("V77_DAG_TRAINING", "1")
os.environ.setdefault("V77_N_LAYERS", "7")
os.environ.setdefault("BREATH_EMBED_ORTHO_INIT", "2.0")
os.environ.setdefault("PER_BREATH_TEMP", "1")
os.environ.setdefault("BREATH_NORM_OSC", "1")
os.environ.setdefault("MAX_STEP_BASE", "2.0")
os.environ.setdefault("MAX_STEP_MIN", "0.1")
os.environ.setdefault("NOTEBOOK_ACCUMULATE_ENABLED", "1")
os.environ.setdefault("NOTEBOOK_NO_DETACH", "1")
os.environ.setdefault("V78_HEAD_CODEBOOK", "1")
os.environ.setdefault("V78_HEAD_CODEBOOK_N", "32")
os.environ.setdefault("CONTROLLER_N_LAYERS", "4")
os.environ.setdefault("WAIST_ATTN_SUPERVISION", "1")
os.environ.setdefault("WAIST_ATTN_AUX_WEIGHT", "0.5")
os.environ.setdefault("V79_CAUSAL_MASKS", "1")
os.environ.setdefault("SCHED_SAMPLE_RATE", "0.3")
os.environ.setdefault("CONTROLLER_DECODE", "1")
os.environ.setdefault("PER_BREATH_DECODE", "1")
os.environ.setdefault("BFIELD_WAIST", "512")
os.environ.setdefault("BFIELD_END_OF_BREATH", "1")
os.environ.setdefault("BFIELD_ENFORCED", "0")
os.environ.setdefault("BFIELD_ALPHA", "1.0")
os.environ.setdefault("WAIST_CODEBOOK_N", "64")
os.environ.setdefault("WAIST_CODEBOOK_INJECT_WEIGHT", "1.0")
os.environ.setdefault("NOTEBOOK_V24", "1")
os.environ.setdefault("NOTEBOOK_DUAL", "1")
os.environ.setdefault("NOTEBOOK_POOL_MODE", "attn")
os.environ.setdefault("NOTEBOOK_INIT_SCALE", "0.02")
os.environ.setdefault("STOCH_DEPTH_P", "0.0")  # AUDIT: deterministic
os.environ.setdefault("LABEL_SMOOTHING", "0.0")
os.environ.setdefault("WEIGHT_DECAY", "0.0")
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("SINE_TEMP", "1")
os.environ.setdefault("SINE_TEMP_MAX", "2.0")
os.environ.setdefault("SINE_TEMP_MIN", "0.7")
os.environ.setdefault("CONSTANT_RADIUS", "1")
os.environ.setdefault("BREATH_TIME_EMBED", "1")
os.environ.setdefault("BREATH_TIME_INIT_SCALE", "0.0")
os.environ.setdefault("CROSS_BREATH_HANDOFF", "1")
os.environ.setdefault("ABLATE_BREATH_ROTATION", "1")
os.environ.setdefault("QUADRATURE_HEADS", "0")
os.environ.setdefault("PROMPT_REFRESH_ALPHA", "0.1")
os.environ.setdefault("BOUNDARY_AUX_WEIGHT", "0.0")
os.environ.setdefault("BOUNDARY_POS_WEIGHT", "5.0")
os.environ.setdefault("PER_BREATH_FULL_ANSWER", "0")

# Audit-specific overrides
os.environ.setdefault("MULTI_HEAD_WAIST", "1")  # enable multi-head path

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer


def cast_model_fp32(model):
    """Force base model FP weights to float for stable audit numerics."""
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    _cast(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def build_inputs(tok, fixed_len: int, K: int):
    """Build two synthetic inputs sharing a prompt but with different answer spans."""
    np.random.seed(42)
    problem = "Sam has 5 apples. He eats 2. How many does he have?"
    p_ids = tok.encode(problem).ids
    prompt_len = len(p_ids)
    assert prompt_len + 30 < fixed_len, f"prompt_len={prompt_len} too long"

    # GOLD answer-span tokens (anything plausible)
    gold_answer_text = " 5 - 2 = 3 # 3"
    gold_ids = tok.encode(gold_answer_text).ids[:20]

    # GARBAGE answer-span tokens (random ids in vocab range)
    garbage_ids = list(np.random.randint(100, 50000, size=20).astype(np.int32))

    tokens_A = np.zeros((1, fixed_len), dtype=np.int32)
    tokens_B = np.zeros((1, fixed_len), dtype=np.int32)
    tokens_A[0, :prompt_len] = p_ids
    tokens_B[0, :prompt_len] = p_ids
    # Append EOS (0) after answer span — but tokens start as zeros so EOS pad is fine.
    tokens_A[0, prompt_len:prompt_len + len(gold_ids)] = gold_ids
    tokens_B[0, prompt_len:prompt_len + len(garbage_ids)] = garbage_ids
    return tokens_A, tokens_B, prompt_len


def _unwrap_head(o):
    """If WaistController returned a multi-head dict, use the 'ops' head for the audit
    (the audit checks for IDENTITY across A/B; all heads must equal A/B-wise since they
    share the cross-attn backbone)."""
    if isinstance(o, dict):
        return o["ops"]
    return o


def forward_train(model, tokens_t, kv_mask_t, K: int):
    """Mirror the training forward pattern (per-breath supervision)."""
    _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
        tokens_t, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
        notebook_pool_mask=kv_mask_t, main_attn_mask=kv_mask_t)
    prompt_emb = model.embed(tokens_t).cast(dtypes.float)
    # Last breath's logits across ALL positions (T_q = T_kv = fixed_len).
    waist_k = waist_per_breath[K - 1].cast(dtypes.float)
    logits_full = model.waist_controller.forward(
        waist_k, prompt_emb, model.embed_out,
        k_idx=K - 1, K_total=K,
        kv_mask=kv_mask_t)
    return _unwrap_head(logits_full)  # (1, T_q, vocab)


def forward_eval(model, tokens_t, kv_mask_t, K: int, t_pos: int):
    """Mirror the eval forward pattern: T_q=1, gathering only at t_pos."""
    fixed_len = tokens_t.shape[1]
    _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
        tokens_t, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
        notebook_pool_mask=kv_mask_t, main_attn_mask=kv_mask_t)
    prompt_emb = model.embed(tokens_t).cast(dtypes.float)
    wk = waist_per_breath[K - 1].cast(dtypes.float)
    # Gather row at t_pos
    positions = Tensor.arange(fixed_len)
    t_pos_t = Tensor([t_pos], dtype=dtypes.int)
    gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(1, 1)).reshape(1, fixed_len, 1).cast(dtypes.float)
    wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)
    lk = model.waist_controller.forward(
        wk_at_pos, prompt_emb, model.embed_out,
        k_idx=K - 1, K_total=K,
        kv_mask=kv_mask_t)
    return _unwrap_head(lk)  # (1, 1, vocab)


def audit_prompt_sensitivity(model, tok, fixed_len: int, K: int):
    """SANITY: changing the PROMPT (with answer span all zeros) MUST change the
    prediction at position prompt_len-1. Otherwise the mask is over-aggressive
    and we'd have a model that ignores its input entirely."""
    np.random.seed(7)
    prob1 = "Sam has 5 apples. He eats 2. How many does he have?"
    prob2 = "Lisa earns $10 per hour and works 4 hours. Total?"
    p1 = tok.encode(prob1).ids
    p2 = tok.encode(prob2).ids
    pl = min(len(p1), len(p2))  # use common prefix length for fair comparison
    p1 = p1[:pl]
    p2 = p2[:pl]
    t1 = np.zeros((1, fixed_len), dtype=np.int32)
    t2 = np.zeros((1, fixed_len), dtype=np.int32)
    t1[0, :pl] = p1
    t2[0, :pl] = p2
    kv_mask = np.zeros((1, fixed_len), dtype=np.float32)
    kv_mask[0, :pl] = 1.0

    Tensor.training = False
    kv_mask_t = Tensor(kv_mask, dtype=dtypes.float).realize()
    t1_t = Tensor(t1, dtype=dtypes.int).realize()
    t2_t = Tensor(t2, dtype=dtypes.int).realize()

    l1 = forward_train(model, t1_t, kv_mask_t, K).numpy()
    l2 = forward_train(model, t2_t, kv_mask_t, K).numpy()
    pos = pl - 1
    diff = float(np.abs(l1[0, pos, :] - l2[0, pos, :]).max())
    ok = diff > 0.1  # prompts should produce DISTINCT predictions
    print(f"\n=== SANITY [prompt sensitivity] ===")
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] pos={pos} (last prompt token)  max|logit_p1 - logit_p2|={diff:.6e}")
    if not ok:
        print(f"  FAIL: the model produces nearly-identical logits for different prompts —")
        print(f"        the mask is blocking the prompt itself.")
    return ok


def audit_one(model, tok, fixed_len: int, K: int, mode: str):
    """Run the audit in either 'train' (full T_q) or 'eval' (gather at t_pos) mode."""
    tokens_A, tokens_B, prompt_len = build_inputs(tok, fixed_len, K)
    kv_mask = np.zeros((1, fixed_len), dtype=np.float32)
    kv_mask[0, :prompt_len] = 1.0
    kv_mask_t = Tensor(kv_mask, dtype=dtypes.float).realize()
    tokens_A_t = Tensor(tokens_A, dtype=dtypes.int).realize()
    tokens_B_t = Tensor(tokens_B, dtype=dtypes.int).realize()

    Tensor.training = False  # deterministic; freeze STOCH_DEPTH masks

    print(f"\n=== AUDIT [{mode}] ===")
    print(f"  prompt_len={prompt_len}  fixed_len={fixed_len}  K={K}")
    print(f"  prompt tokens: {tokens_A[0, :prompt_len].tolist()}")
    print(f"  gold span:    {tokens_A[0, prompt_len:prompt_len + 20].tolist()}")
    print(f"  garbage span: {tokens_B[0, prompt_len:prompt_len + 20].tolist()}")

    # Positions to compare
    pos1 = prompt_len - 1            # predicts first answer token
    pos2 = prompt_len - 1 + 5        # predicts 6th answer token
    failed = False

    if mode == "train":
        logits_A = forward_train(model, tokens_A_t, kv_mask_t, K).numpy()  # (1, T, V)
        logits_B = forward_train(model, tokens_B_t, kv_mask_t, K).numpy()
        for pos in (pos1, pos2):
            la = logits_A[0, pos, :]
            lb = logits_B[0, pos, :]
            diff = float(np.abs(la - lb).max())
            ok = diff < 1e-4
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] pos={pos}  max|logit_A - logit_B|={diff:.6e}")
            if not ok:
                failed = True
                # Print argmax of each for sanity
                print(f"            argmax_A={int(np.argmax(la))}  argmax_B={int(np.argmax(lb))}")
    else:
        for pos in (pos1, pos2):
            la = forward_eval(model, tokens_A_t, kv_mask_t, K, pos).numpy()[0, 0]
            lb = forward_eval(model, tokens_B_t, kv_mask_t, K, pos).numpy()[0, 0]
            diff = float(np.abs(la - lb).max())
            ok = diff < 1e-4
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] pos={pos}  max|logit_A - logit_B|={diff:.6e}")
            if not ok:
                failed = True
                print(f"            argmax_A={int(np.argmax(la))}  argmax_B={int(np.argmax(lb))}")

    return not failed


def main():
    CKPT = os.environ.get("CKPT", ".cache/gsm8k_steps_ckpts/v80_v8v3_smoke_step70.safetensors")
    fixed_len = int(os.environ.get("FIXED_LEN", "320"))
    K = int(os.environ.get("K", "7"))  # v81 = 7 layers

    cfg_kwargs = {}
    for env_key, cfg_key in [("HIDDEN", "hidden"), ("N_HEADS", "n_heads"),
                              ("HEAD_DIM", "head_dim"), ("FFN", "ffn"),
                              ("CONTROLLER_HIDDEN", "controller_hidden")]:
        v = os.environ.get(env_key)
        if v is not None:
            cfg_kwargs[cfg_key] = int(v)
    cfg = Config(**cfg_kwargs)

    print("=== v81 Phase 2 masking audit ===")
    print(f"  ckpt: {CKPT}")
    print(f"  fixed_len={fixed_len}  K={K}")

    tok = load_tokenizer()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    if os.path.exists(CKPT):
        print(f"\nloading ckpt for plumbing: {CKPT}")
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
        del ckpt_sd
    else:
        print(f"\nWARNING: ckpt not found at {CKPT}. Audit will run on untrained model.")

    train_ok = audit_one(model, tok, fixed_len, K, mode="train")
    eval_ok = audit_one(model, tok, fixed_len, K, mode="eval")

    # Sanity: predictions SHOULD differ when the prompt itself changes — otherwise
    # masking is over-aggressive and the model can't see anything.
    prompt_ok = audit_prompt_sensitivity(model, tok, fixed_len, K)

    overall_ok = train_ok and eval_ok and prompt_ok
    print(f"\n=== AUDIT RESULT ===")
    print(f"  train pattern: {'PASS' if train_ok else 'FAIL'}")
    print(f"  eval pattern:  {'PASS' if eval_ok else 'FAIL'}")
    print(f"  overall:       {'PASS' if overall_ok else 'FAIL'}")
    if not overall_ok:
        print("\nMASKING IS BROKEN. Fix mycelium/breathing.py + mycelium/l3_training.py.")
        sys.exit(1)
    print("\nMasking is airtight — safe to proceed to multi-head training.")


if __name__ == "__main__":
    main()
