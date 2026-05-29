"""v81 TF vs AR diagnostic — verify whether v81's multi-head decode works.

For 5 test problems, runs the model TWO ways at the FINAL breath (K-1):

  TF (teacher-forced):
      tokens = [prompt, gold_L6, EOS, zeros]   (matches training-time input layout)
      Forward once with full T_q = fixed_len; argmax each head at the position
      RANGE that head was supervised on (NOT at every position — heads only
      learned to predict at their specific position range).

  Sequential per-head TF "walk":
      Walk ops head argmax starting at prompt_len-1 until it emits the next
      structural delimiter (token-id 1040 = " |", or EOS=0). Then switch to
      the types head, starting at the position AFTER the last ops token.
      Etc. for args1 and args2. This is how the segmented decode should
      actually work — each head is queried only at its supervised range.

  AR (autoregressive, current eval_v81_dag.py approach):
      tokens = [prompt, zeros]
      Same final-breath forward, then for each head argmax at every position.
      Walk that argmax to extract each head's list. This is what the current
      eval does — it queries ALL heads at the SAME starting position.

Compares TF vs AR overlap with gold L6.

Outputs (per problem): prompt_len, gold_L6 text, TF text, AR text, TF parse
accuracy, AR parse accuracy.

Env vars:
    CKPT (required)         = ckpt to evaluate
    V77_TEST_PATH           = test set jsonl (default v81 train data)
    N_PROBLEMS              = how many to run (default 5)
    FIXED_LEN               = 320
    K                       = 7

Mirrors v81_smoke_train.sh env stack so model topology / JIT flags match.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mirror v81 production env stack BEFORE importing mycelium.
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
os.environ.setdefault("SCHED_SAMPLE_RATE", "0.0")
os.environ.setdefault("MULTI_HEAD_WAIST", "1")
os.environ.setdefault("V81_MAIN_ATTN_MASK", "1")
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
os.environ.setdefault("STOCH_DEPTH_P", "0.0")  # deterministic
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

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_v77

from v81_sympy_eval import b6_string_to_dag, dag_to_answer


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


def forward_all_heads(model, tokens_t, kv_mask_t, K, force_single_head=False):
    """Run K breaths, return dict {ops/types/args1/args2}: (B, T, vocab) each at FINAL breath.
    If force_single_head=True, returns a Tensor (B, T, vocab) — the single-head logits without
    any per-head bias."""
    _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
        tokens_t, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
        notebook_pool_mask=kv_mask_t, main_attn_mask=kv_mask_t)
    prompt_emb = model.embed(tokens_t).cast(dtypes.float)
    wk = waist_per_breath[K - 1].cast(dtypes.float)
    out = model.waist_controller.forward(
        wk, prompt_emb, model.embed_out,
        k_idx=K - 1, K_total=K,
        kv_mask=kv_mask_t,
        force_single_head=force_single_head)
    return out  # dict (B, T, vocab) per head, OR Tensor when force_single_head=True


def argmax_per_head_per_pos(out_dict, B, T):
    """Return dict head -> np.ndarray (B, T) of argmax token ids (vocab restricted to 50277)."""
    res = {}
    for name in ["ops", "types", "args1", "args2"]:
        logits = out_dict[name][:, :, :50277]
        ids = logits.argmax(axis=-1).cast(dtypes.int).numpy()
        res[name] = ids
    return res


def decode_walk_until_pipe(argmax_row, start_pos, eos_id=0, pipe_id=1040, max_len=60):
    """Walk argmax_row from start_pos, collecting tokens until EOS, pipe, or max_len.

    Returns (toks, end_pos) where end_pos is the position AFTER the last token
    collected (so the next head should start at end_pos)."""
    out = []
    p = start_pos
    n = len(argmax_row)
    for _ in range(max_len):
        if p >= n:
            break
        tid = int(argmax_row[p])
        if tid == eos_id or tid == pipe_id:
            break
        out.append(tid)
        p += 1
    return out, p


def assemble_AR_b6(tok, head_ids_dict, prompt_len, b_idx=0):
    """Current eval_v81_dag.py approach (approach A): argmax each head at the FIRST
    answer position. Walk each head independently from prompt_len - 1 until ` |` or EOS."""
    PIPE = 1040
    EOS = 0
    out = {}
    for name in ["ops", "types", "args1", "args2"]:
        toks, _end = decode_walk_until_pipe(head_ids_dict[name][b_idx], prompt_len - 1,
                                              eos_id=EOS, pipe_id=PIPE, max_len=60)
        out[name] = toks
    ops_str = tok.decode(out["ops"]).strip()
    types_str = tok.decode(out["types"]).strip()
    a1_str = tok.decode(out["args1"]).strip()
    a2_str = tok.decode(out["args2"]).strip()
    return f"{ops_str} | {types_str} | {a1_str} | {a2_str}"


def assemble_segmented_b6(tok, head_ids_dict, prompt_len, b_idx=0):
    """Each head argmaxes at its OWN trained position range. ops head walks from
    prompt_len-1 until pipe/eos; types head walks from the next position (after ops's
    range) until pipe/eos; etc. This matches how training supervised each head."""
    PIPE = 1040
    EOS = 0
    # ops head: walk from prompt_len - 1
    ops_toks, pos_after_ops = decode_walk_until_pipe(head_ids_dict["ops"][b_idx], prompt_len - 1,
                                                       eos_id=EOS, pipe_id=PIPE, max_len=60)
    # types head: starts AFTER ops's range. The pipe token at the boundary was supervised
    # under TYPES head (because head_prefixes[1] = " | "), so we INCLUDE pos_after_ops as
    # the start of types' range. types_head was supervised on " | <types>" starting at pos_after_ops.
    # We walk from pos_after_ops (where ops decoded the next-pipe trigger) past the pipe token,
    # until the next pipe.
    # types head: walk starting at pos_after_ops + 1 (one past the pipe that ops emitted)
    # OR pos_after_ops if ops walked off the end without seeing pipe.
    types_start = pos_after_ops
    # types_head's first label is the " |" token at position pos_after_ops (where ops just
    # stopped on seeing pipe). So we walk types from pos_after_ops, expecting the " |" token,
    # then the space + ints. But for cleaner output, skip the leading " |" by walking through it.
    types_toks_raw, pos_after_types = decode_walk_until_pipe(head_ids_dict["types"][b_idx], types_start,
                                                               eos_id=EOS, pipe_id=PIPE, max_len=60)
    # Strip a leading " |" if present (it was supervised, but logically belongs to delimiter).
    # In practice the first token in types' range is the " |" token-id 1040 — walk_until_pipe
    # would stop there. So types_toks_raw is EMPTY if the model perfectly tracks this.
    # To get actual types content, we walk types ONE token past the initial pipe.
    # Simpler approach: walk from types_start, but allow the FIRST token to be a pipe
    # (skip it once), then walk-until-pipe normally.
    p = types_start
    n = len(head_ids_dict["types"][b_idx])
    # Skip the leading pipe token if present
    if p < n and int(head_ids_dict["types"][b_idx][p]) == PIPE:
        p += 1
    types_toks, pos_after_types = decode_walk_until_pipe(head_ids_dict["types"][b_idx], p,
                                                          eos_id=EOS, pipe_id=PIPE, max_len=60)
    # args1 head: walk starting at pos_after_types, skip leading pipe.
    p = pos_after_types
    if p < n and int(head_ids_dict["args1"][b_idx][p]) == PIPE:
        p += 1
    args1_toks, pos_after_args1 = decode_walk_until_pipe(head_ids_dict["args1"][b_idx], p,
                                                          eos_id=EOS, pipe_id=PIPE, max_len=60)
    # args2 head: walk starting at pos_after_args1, skip leading pipe.
    p = pos_after_args1
    if p < n and int(head_ids_dict["args2"][b_idx][p]) == PIPE:
        p += 1
    args2_toks, _ = decode_walk_until_pipe(head_ids_dict["args2"][b_idx], p,
                                            eos_id=EOS, pipe_id=PIPE, max_len=60)

    ops_str = tok.decode(ops_toks).strip()
    types_str = tok.decode(types_toks).strip()
    a1_str = tok.decode(args1_toks).strip()
    a2_str = tok.decode(args2_toks).strip()
    return f"{ops_str} | {types_str} | {a1_str} | {a2_str}"


def main():
    CKPT = os.environ.get("CKPT", ".cache/gsm8k_steps_ckpts/v81_prod_step2000.safetensors")
    TEST_PATH = os.environ.get("V77_TEST_PATH", ".cache/gsm8k_steps_v81_train.jsonl")
    N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "5"))
    K = int(os.environ.get("K", "7"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "320"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    print(f"=== v81 TF/AR per-breath diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  K={K} fixed_len={FIXED_LEN} n_problems={N_PROBLEMS}")

    if not os.path.exists(CKPT):
        print(f"ERROR: ckpt not found at {CKPT}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: test set not found at {TEST_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\nloading test set...")
    examples = load_gsm8k_v77(TEST_PATH, min_k=MIN_K, max_k=MAX_K,
                               require_sympy_match=True, bucket_by_k=False)
    examples = examples[:N_PROBLEMS]
    print(f"  using {len(examples)} examples")

    tok = load_tokenizer()
    cfg = Config()
    print(f"\nloading Pythia + ckpt...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt_sd = safe_load(CKPT)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
    del ckpt_sd
    Tensor.training = False

    n_tf_parse = 0
    n_tf_ok = 0
    n_ar_parse = 0
    n_ar_ok = 0
    n_seg_parse = 0
    n_seg_ok = 0
    PIPE = 1040
    EOS = 0

    for i, ex in enumerate(examples):
        print(f"\n--- Problem {i+1} ---")
        print(f"  Q: {ex.problem[:90]!r}...")
        gold_L6 = ex.per_layer_target[K - 1]
        print(f"  gold L6: {gold_L6!r}")
        print(f"  gold answer: {ex.gold_answer}")

        # Build TF and AR inputs.
        p_ids = tok.encode(ex.problem).ids
        prompt_len = len(p_ids)
        gold_ids = tok.encode(" " + gold_L6).ids  # leading space, matches training

        # TF input: [prompt, gold_L6, EOS, zeros]
        tf_tokens_np = np.zeros((1, FIXED_LEN), dtype=np.int32)
        tf_tokens_np[0, :prompt_len] = p_ids
        # Place gold L6 starting at position prompt_len (so labels at prompt_len-1 = first gold).
        for j, tid in enumerate(gold_ids):
            pos = prompt_len + j
            if pos < FIXED_LEN:
                tf_tokens_np[0, pos] = tid
        # EOS after gold
        eos_pos = prompt_len + len(gold_ids)
        if eos_pos < FIXED_LEN:
            tf_tokens_np[0, eos_pos] = EOS

        # AR input: [prompt, zeros]
        ar_tokens_np = np.zeros((1, FIXED_LEN), dtype=np.int32)
        ar_tokens_np[0, :prompt_len] = p_ids

        # kv_mask: 1.0 at prompt positions only (both TF and AR use the SAME mask;
        # this is the v81 paradigm — answer-span is always masked-out in cross-attn,
        # main-attn, and notebook).
        kv_mask_np = np.zeros((1, FIXED_LEN), dtype=np.float32)
        kv_mask_np[0, :prompt_len] = 1.0

        tf_tokens_t = Tensor(tf_tokens_np, dtype=dtypes.int).realize()
        ar_tokens_t = Tensor(ar_tokens_np, dtype=dtypes.int).realize()
        kv_mask_t = Tensor(kv_mask_np, dtype=dtypes.float).realize()

        # ---- TF forward ----
        tf_out = forward_all_heads(model, tf_tokens_t, kv_mask_t, K)
        tf_argmax = argmax_per_head_per_pos(tf_out, B=1, T=FIXED_LEN)
        # Argmax MUST be identical to AR's argmax at every position IF the masking
        # closes the leak (Phase 2 audit checked this). Confirm with one element.
        tf_at_prompt_last = int(tf_argmax["ops"][0][prompt_len - 1])

        # ---- AR forward ----
        ar_out = forward_all_heads(model, ar_tokens_t, kv_mask_t, K)
        ar_argmax = argmax_per_head_per_pos(ar_out, B=1, T=FIXED_LEN)
        ar_at_prompt_last = int(ar_argmax["ops"][0][prompt_len - 1])

        # ---- Single-head AR forward (no per-head bias; force_single_head=True) ----
        sh_out = forward_all_heads(model, ar_tokens_t, kv_mask_t, K, force_single_head=True)
        sh_argmax = sh_out[:, :, :50277].argmax(axis=-1).cast(dtypes.int).numpy()  # (1, T)

        leak_check = "PASS" if tf_at_prompt_last == ar_at_prompt_last else "FAIL"
        print(f"  TF/AR mask leak: {leak_check} (ops@pos{prompt_len-1}: TF={tf_at_prompt_last} vs AR={ar_at_prompt_last})")

        # ---- (1) AR / approach A: all heads start at prompt_len - 1 ----
        ar_text = assemble_AR_b6(tok, ar_argmax, prompt_len)
        ar_dag = b6_string_to_dag(ar_text)
        ar_val = dag_to_answer(ar_dag) if ar_dag else None
        ar_ok = ar_val is not None and abs(ar_val - ex.gold_answer) < 1e-3
        print(f"  AR (current approach A):")
        print(f"    text: {ar_text!r}")
        print(f"    dag:  {ar_dag!r}")
        print(f"    val={ar_val}  ok={ar_ok}")
        if ar_dag is not None:
            n_ar_parse += 1
        if ar_ok:
            n_ar_ok += 1

        # ---- (2) Segmented walk: each head argmaxes at its trained range ----
        seg_text = assemble_segmented_b6(tok, ar_argmax, prompt_len)
        seg_dag = b6_string_to_dag(seg_text)
        seg_val = dag_to_answer(seg_dag) if seg_dag else None
        seg_ok = seg_val is not None and abs(seg_val - ex.gold_answer) < 1e-3
        print(f"  SEGMENTED walk (each head at its trained range):")
        print(f"    text: {seg_text!r}")
        print(f"    dag:  {seg_dag!r}")
        print(f"    val={seg_val}  ok={seg_ok}")
        if seg_dag is not None:
            n_seg_parse += 1
        if seg_ok:
            n_seg_ok += 1

        # ---- (3) TF — same heads but with gold tokens IN INPUT. Note: when v81 masking
        # is airtight, TF should be IDENTICAL to AR (the model never sees the answer span).
        # We still compute to confirm.
        tf_text = assemble_segmented_b6(tok, tf_argmax, prompt_len)
        tf_dag = b6_string_to_dag(tf_text)
        tf_val = dag_to_answer(tf_dag) if tf_dag else None
        tf_ok = tf_val is not None and abs(tf_val - ex.gold_answer) < 1e-3
        print(f"  TF (gold L6 in input, segmented walk):")
        print(f"    text: {tf_text!r}")
        print(f"    dag:  {tf_dag!r}")
        print(f"    val={tf_val}  ok={tf_ok}")
        if tf_dag is not None:
            n_tf_parse += 1
        if tf_ok:
            n_tf_ok += 1

        # ---- BONUS: dump per-head argmax at the first 20 answer-span positions for ops head ----
        # Useful for visualizing what each head produces at its trained range vs out-of-range.
        first_20_pos = [prompt_len - 1 + j for j in range(min(20, FIXED_LEN - prompt_len))]
        ops_argmax_20 = [int(ar_argmax["ops"][0][p]) for p in first_20_pos]
        types_argmax_20 = [int(ar_argmax["types"][0][p]) for p in first_20_pos]
        a1_argmax_20 = [int(ar_argmax["args1"][0][p]) for p in first_20_pos]
        a2_argmax_20 = [int(ar_argmax["args2"][0][p]) for p in first_20_pos]
        print(f"  per-head argmax (first 20 answer positions):")
        print(f"    ops:    {ops_argmax_20}")
        print(f"    ops dec: {tok.decode(ops_argmax_20)!r}")
        print(f"    types:  {types_argmax_20}")
        print(f"    types dec: {tok.decode(types_argmax_20)!r}")
        print(f"    args1:  {a1_argmax_20}")
        print(f"    args1 dec: {tok.decode(a1_argmax_20)!r}")
        print(f"    args2:  {a2_argmax_20}")
        print(f"    args2 dec: {tok.decode(a2_argmax_20)!r}")

        # ---- Single-head walk (no per-head bias; pure cross-attn backbone) ----
        # Walk single-head argmax starting at prompt_len-1 until EOS or max_len. Split on " |".
        sh_toks = []
        p = prompt_len - 1
        for _ in range(80):
            if p >= FIXED_LEN:
                break
            tid = int(sh_argmax[0][p])
            if tid == EOS:
                break
            sh_toks.append(tid)
            p += 1
        sh_text = tok.decode(sh_toks).strip()
        # Take only first 4 segments split by " |"
        sh_segments = sh_text.split(" |")
        if len(sh_segments) >= 4:
            sh_text_trunc = " |".join(sh_segments[:4]).strip()
            # Construct B6 by re-joining with proper " | " separator
            sh_text_b6 = " | ".join(s.strip() for s in sh_segments[:4])
        else:
            sh_text_b6 = sh_text
        sh_dag = b6_string_to_dag(sh_text_b6)
        sh_val = dag_to_answer(sh_dag) if sh_dag else None
        sh_ok = sh_val is not None and abs(sh_val - ex.gold_answer) < 1e-3
        print(f"  SINGLE-HEAD walk (no per-head bias, first 4 pipe-segments):")
        print(f"    full: {sh_text!r}")
        print(f"    text: {sh_text_b6!r}")
        print(f"    dag:  {sh_dag!r}")
        print(f"    val={sh_val}  ok={sh_ok}")

        # ---- DECONSTRUCT TRAINING LABELS for this example ----
        # Show where each head was actually supervised, and whether the model's
        # argmax matches the gold token at those supervised positions.
        segs = gold_L6.split(" | ")
        if len(segs) == 4:
            head_prefixes = [" ", " | ", " | ", " | "]
            head_names = ["ops", "types", "args1", "args2"]
            offset = 0
            print(f"  TRAINING SUPERVISION per head (gold positions + model argmax):")
            for hi in range(4):
                head_text = head_prefixes[hi] + segs[hi]
                seg_ids = tok.encode(head_text).ids
                # Per-position gold + model argmax at each label position
                start_pos = (prompt_len - 1) + offset
                end_pos = start_pos + len(seg_ids)
                n_matches = 0
                gold_seg = []
                model_seg = []
                for j, tid in enumerate(seg_ids):
                    lp = start_pos + j
                    if lp < FIXED_LEN - 1:
                        gold_seg.append(int(tid))
                        model_seg.append(int(ar_argmax[head_names[hi]][0][lp]))
                        if int(ar_argmax[head_names[hi]][0][lp]) == int(tid):
                            n_matches += 1
                acc = n_matches / max(len(seg_ids), 1) * 100
                print(f"    {head_names[hi]:6s} [{start_pos:3d}..{end_pos:3d}] acc={acc:.1f}% ({n_matches}/{len(seg_ids)})")
                print(f"      gold:  {gold_seg} = {tok.decode(gold_seg)!r}")
                print(f"      model: {model_seg} = {tok.decode(model_seg)!r}")
                offset += len(seg_ids)

    print(f"\n=== summary (n={N_PROBLEMS}) ===")
    print(f"  AR (current eval_v81 approach A):  parse {n_ar_parse}/{N_PROBLEMS}  acc {n_ar_ok}/{N_PROBLEMS}")
    print(f"  SEGMENTED (each head at its range): parse {n_seg_parse}/{N_PROBLEMS}  acc {n_seg_ok}/{N_PROBLEMS}")
    print(f"  TF (gold L6 in input, segmented):  parse {n_tf_parse}/{N_PROBLEMS}  acc {n_tf_ok}/{N_PROBLEMS}")


if __name__ == "__main__":
    main()
