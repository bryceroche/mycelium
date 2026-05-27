"""Waist-zero ablation diagnostic for the WaistController.

Question: is the 512d waist representation LOAD-BEARING for the WaistController's
decode, or does the controller bypass it and decode from prompt cross-attn alone?

Method (teacher-forced):
    For each breath k in [0..K):
      1. ce_normal[k] = CE( WaistController(waist[k], prompt_emb) , gold_L_k )
      2. ce_zero[k]   = CE( WaistController(zeros_like(waist[k]), prompt_emb) , gold_L_k )
      3. ce_swap[k]   = CE( WaistController(waist[0],   prompt_emb) , gold_L_k )
         (cross-breath substitution: always use breath-0's waist content;
          k_idx stays at k so the k_pos_embed is unchanged)

Interpretations:
    ce_normal ≈ ce_zero  → waist DECORATIVE (decoder uses prompt only — STRUCTURAL bug)
    ce_normal << ce_zero → waist LOAD-BEARING (decoder genuinely reads waist)
    ce_normal ≈ ce_swap  → no breath-specificity (waist content same across breaths)
    ce_normal << ce_swap → breath-specific signal in waist content

Env vars:
    CKPT          path to ckpt (default: latest v79 prod)
    V77_TEST_PATH test JSONL (default: v78c test)
    NUM_EVAL      problems per run (default 64)
    K             breaths (default 7 for v79)
    FIXED_LEN     padding length (default 320)
    OUT_PATH      result text dump (default /tmp/waist_zero_diag.txt)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def env_v79():
    os.environ.setdefault("DEV", "PCI+AMD")
    os.environ["V77_DAG_TRAINING"] = "1"
    os.environ["V77_N_LAYERS"] = "7"
    os.environ["BREATH_EMBED_ORTHO_INIT"] = "2.0"
    os.environ["PER_BREATH_TEMP"] = "1"
    os.environ["BREATH_NORM_OSC"] = "1"
    os.environ["MAX_STEP_BASE"] = "2.0"
    os.environ["MAX_STEP_MIN"] = "0.1"
    os.environ["NOTEBOOK_ACCUMULATE_ENABLED"] = "1"
    os.environ["NOTEBOOK_NO_DETACH"] = "1"
    os.environ["V78_HEAD_CODEBOOK"] = "1"
    os.environ["V78_HEAD_CODEBOOK_N"] = "12"
    os.environ["CONTROLLER_N_LAYERS"] = "4"
    os.environ["WAIST_ATTN_SUPERVISION"] = "1"
    os.environ["WAIST_ATTN_AUX_WEIGHT"] = "0.5"
    os.environ["V79_CAUSAL_MASKS"] = "1"
    os.environ["NOTEBOOK_DAG"] = "0"
    os.environ["CONTROLLER_DECODE"] = "1"
    os.environ["PER_BREATH_DECODE"] = "1"
    os.environ["BFIELD_WAIST"] = "512"
    os.environ["BFIELD_END_OF_BREATH"] = "1"
    os.environ["BFIELD_ENFORCED"] = "0"
    os.environ["BFIELD_ALPHA"] = "1.0"
    os.environ["WAIST_CODEBOOK_N"] = "64"
    os.environ["WAIST_CODEBOOK_INJECT_WEIGHT"] = "1.0"
    os.environ["NOTEBOOK_V24"] = "1"
    os.environ["NOTEBOOK_DUAL"] = "1"
    os.environ["NOTEBOOK_POOL_MODE"] = "attn"
    os.environ["NOTEBOOK_INIT_SCALE"] = "0.02"
    os.environ["STOCH_DEPTH_P"] = "0.10"
    os.environ["LABEL_SMOOTHING"] = "0.1"
    os.environ["WEIGHT_DECAY"] = "0.05"
    os.environ["PER_HEAD_PITCH"] = "1"
    os.environ["SINE_TEMP"] = "1"
    os.environ["SINE_TEMP_MAX"] = "2.0"
    os.environ["SINE_TEMP_MIN"] = "0.7"
    os.environ["CONSTANT_RADIUS"] = "1"
    os.environ["BREATH_TIME_EMBED"] = "1"
    os.environ["BREATH_TIME_INIT_SCALE"] = "0.0"
    os.environ["CROSS_BREATH_HANDOFF"] = "1"
    os.environ["ABLATE_BREATH_ROTATION"] = "1"
    os.environ["QUADRATURE_HEADS"] = "0"
    os.environ["PROMPT_REFRESH_ALPHA"] = "0.1"
    os.environ["BOUNDARY_AUX_WEIGHT"] = "0.0"


env_v79()


import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_v77


def cast_model_fp32(model):
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


def build_batch(tok, examples, K, fixed_len, eos_id=0):
    """Build (tokens, per_step_labels, kv_mask, prompt_lens).

    Mirrors per_breath_train_step's V77 branch:
      - tokens = [prompt + final_layer (L_{K-1}) + EOS, 0...]   (one row per example)
      - per_step_labels[k, b, p] = layer-k's gold tok at position p ; -100 elsewhere
      - kv_mask is 1.0 at positions [0, prompt_len+len(final_layer)+1), 0 elsewhere
        — at eval the controller would only see prompt positions; for the TF
        diagnostic we expose the same full sequence the trainer does so per-breath
        decode lines up with the trained behavior.
    """
    B = len(examples)
    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    per_step_labels_np = np.full((K, B, fixed_len - 1), -100, dtype=np.int32)
    kv_mask_np = np.zeros((B, fixed_len), dtype=np.float32)
    prompt_lens = []
    for b, ex in enumerate(examples):
        p_ids = tok.encode(ex.problem).ids
        prompt_len = len(p_ids)
        per_layer_ids = [tok.encode(" " + ex.per_layer_target[ell]).ids for ell in range(K)]
        final_layer_ids = per_layer_ids[-1]
        full_ids = list(p_ids) + list(final_layer_ids) + [eos_id]
        full_ids = full_ids[:fixed_len]
        tokens_np[b, :len(full_ids)] = full_ids
        kv_mask_np[b, :len(full_ids)] = 1.0
        prompt_lens.append(prompt_len)
        for ell in range(K):
            layer_ids = per_layer_ids[ell]
            for j, tid in enumerate(layer_ids):
                label_pos = (prompt_len - 1) + j
                if 0 <= label_pos < fixed_len - 1:
                    per_step_labels_np[ell, b, label_pos] = tid
    return tokens_np, per_step_labels_np, kv_mask_np, prompt_lens


def compute_ce(logits_t: Tensor, labels_t: Tensor) -> float:
    """logits_t: (B, T, V); labels_t: (B, T-1) with -100 outside the supervised span.
    Returns mean CE over non-ignored positions (matches the trainer's reduction)."""
    pred = logits_t[:, :-1, :].cast(dtypes.float)
    ce = pred.sparse_categorical_crossentropy(
        labels_t, ignore_index=-100, reduction="mean")
    return float(ce.realize().numpy())


def main():
    CKPT = os.environ.get("CKPT",
        "/home/bryce/mycelium/.cache/gsm8k_steps_ckpts/v79_causal_prod_step500.safetensors")
    TEST_PATH = os.environ.get("V77_TEST_PATH",
        "/home/bryce/mycelium/.cache/gsm8k_steps_v78c_test.jsonl")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "64"))
    K = int(os.environ.get("K", "7"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "320"))
    OUT_PATH = os.environ.get("OUT_PATH", "/tmp/waist_zero_diag.txt")
    BATCH = int(os.environ.get("BATCH", "8"))

    print(f"=== Waist-zero ablation diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval={NUM_EVAL}  K={K}  fixed_len={FIXED_LEN}  batch={BATCH}")
    print(f"  out:  {OUT_PATH}")

    if not os.path.exists(CKPT):
        print(f"ERROR: ckpt not found at {CKPT}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: test set not found at {TEST_PATH}", file=sys.stderr)
        sys.exit(1)

    examples = load_gsm8k_v77(TEST_PATH, min_k=2, max_k=6,
                              require_sympy_match=True, bucket_by_k=False)
    examples = examples[:NUM_EVAL]
    print(f"  loaded {len(examples)} examples")

    cfg = Config()
    tok = load_tokenizer()
    print(f"  loading Pythia + ckpt...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt_sd = safe_load(CKPT)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  loaded; missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    del ckpt_sd

    Tensor.training = False

    # Accumulators per-breath: list of per-batch CE values.
    ce_normal = [[] for _ in range(K)]
    ce_zero   = [[] for _ in range(K)]
    ce_swap   = [[] for _ in range(K)]

    n_batches = (len(examples) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        batch = examples[bi * BATCH : (bi + 1) * BATCH]
        if not batch:
            break
        # Pad to BATCH if needed (last batch can be short — we just process it as-is).
        B = len(batch)
        tokens_np, per_step_labels_np, kv_mask_np, prompt_lens = build_batch(
            tok, batch, K, FIXED_LEN)
        tokens_t = Tensor(tokens_np, dtype=dtypes.int).realize()
        kv_mask_t = Tensor(kv_mask_np, dtype=dtypes.float).realize()
        per_step_labels_t = [
            Tensor(per_step_labels_np[k], dtype=dtypes.int).realize() for k in range(K)
        ]

        # ONE breathe_with_lookup pass — get all per-breath waists.
        prompt_emb = model.embed(tokens_t).cast(dtypes.float).realize()
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens_t, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask_t)
        # waist_per_breath[k]: (B, T, waist_dim) — we'll feed each variant into
        # waist_controller.forward and compare CE.

        # Build the zero-waist baseline ONCE (same for all k).
        waist_shape = waist_per_breath[0].shape
        waist_zero_t = Tensor.zeros(*waist_shape, dtype=dtypes.float).contiguous().realize()
        waist_breath_0 = waist_per_breath[0].cast(dtypes.float).realize()

        for k in range(K):
            wk = waist_per_breath[k].cast(dtypes.float).realize()
            # Normal: actual breath-k waist, k_idx=k.
            logits_n = model.waist_controller.forward(
                wk, prompt_emb, model.embed_out,
                k_idx=k, K_total=K, kv_mask=kv_mask_t)
            ce_n = compute_ce(logits_n, per_step_labels_t[k])
            ce_normal[k].append(ce_n)

            # Zero waist: shape-matched zeros, k_idx=k unchanged.
            logits_z = model.waist_controller.forward(
                waist_zero_t, prompt_emb, model.embed_out,
                k_idx=k, K_total=K, kv_mask=kv_mask_t)
            ce_z = compute_ce(logits_z, per_step_labels_t[k])
            ce_zero[k].append(ce_z)

            # Cross-breath swap: breath 0's waist content but k_idx=k.
            logits_s = model.waist_controller.forward(
                waist_breath_0, prompt_emb, model.embed_out,
                k_idx=k, K_total=K, kv_mask=kv_mask_t)
            ce_s = compute_ce(logits_s, per_step_labels_t[k])
            ce_swap[k].append(ce_s)

        print(f"  batch {bi+1}/{n_batches} done (B={B})")

    # Aggregate (sample-weighted means).
    def mean(xs): return sum(xs) / max(len(xs), 1)
    ce_normal_mean = [mean(c) for c in ce_normal]
    ce_zero_mean   = [mean(c) for c in ce_zero]
    ce_swap_mean   = [mean(c) for c in ce_swap]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("")
    emit("=" * 72)
    emit("=== Per-breath waist-ablation CE ===")
    emit(f"  ckpt = {CKPT}")
    emit(f"  N={len(examples)} examples, K={K} breaths")
    emit("")
    emit(f"  {'breath':>7} | {'CE(normal)':>11} | {'CE(zero)':>10} | {'CE(swap0)':>10} | {'Δzero':>8} | {'Δswap':>8}")
    emit(f"  {'-'*7} | {'-'*11} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
    for k in range(K):
        dz = ce_zero_mean[k] - ce_normal_mean[k]
        ds = ce_swap_mean[k] - ce_normal_mean[k]
        emit(f"  L{k:>5} | {ce_normal_mean[k]:>11.4f} | {ce_zero_mean[k]:>10.4f} | "
             f"{ce_swap_mean[k]:>10.4f} | {dz:>+8.4f} | {ds:>+8.4f}")
    emit("")
    emit("Interpretation:")
    emit("  Δzero = CE(zero_waist) - CE(normal_waist).")
    emit("    Δzero ≈ 0   → waist DECORATIVE: decoder works without it (STRUCTURAL bug).")
    emit("    Δzero >> 0  → waist LOAD-BEARING: decoder genuinely uses it.")
    emit("  Δswap = CE(waist_0 substituted) - CE(normal).")
    emit("    Δswap ≈ 0   → no breath-specific signal in waist content.")
    emit("    Δswap >> 0  → waist carries breath-distinct content the controller reads.")
    emit("")
    # Aggregate verdict
    mean_dz = mean([ce_zero_mean[k] - ce_normal_mean[k] for k in range(K)])
    mean_ds = mean([ce_swap_mean[k] - ce_normal_mean[k] for k in range(K)])
    emit(f"  Aggregate Δzero (mean over breaths): {mean_dz:+.4f}")
    emit(f"  Aggregate Δswap (mean over breaths): {mean_ds:+.4f}")
    emit("")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[wrote {OUT_PATH}]")


if __name__ == "__main__":
    main()
