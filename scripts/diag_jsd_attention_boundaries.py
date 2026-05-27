"""JSD attention boundaries — Panama-hat diagnostic.

Question: where does the model NATURALLY draw boundaries on the problem text?
Jensen-Shannon divergence between consecutive attention distributions reveals
where the controller's gaze SHIFTS — those positions are inferred boundaries.

Two views, both extracted from `WaistController._last_cross_attn`
((B, T_q, T_kv) after head-averaging) when WAIST_ATTN_SUPERVISION=1:

  VIEW A — DECODE-SIDE BOUNDARIES (output-trajectory):
    For each decode position t in the answer span:
      attn_decode[t]  : distribution over prompt positions [0..prompt_len)
    JSD(attn_decode[t], attn_decode[t+1]) shows where the controller's gaze
    REDISTRIBUTES as it moves from output token t to t+1. Peaks mark output-side
    "Panama hat" moments — the controller stopped referencing one problem region
    and started referencing another. Useful for: does the model's gaze align with
    the operands it's emitting? E.g. when emitting "50", does it look at "50 minutes"?

  VIEW B — BREATH-SIDE BOUNDARIES (specialization):
    For each breath k, average attn_decode[t] over the answer span → a single
    distribution per breath per problem. JSD across consecutive breaths shows
    how breath k's "view of the prompt" differs from breath k+1. This is
    breath-level specialization, complementing the waist-zero Δswap=+4.59 result.

Output: per-problem table + summary statistics.

Env vars:
  CKPT          path to ckpt (default: v79 step 1000)
  V77_TEST_PATH test JSONL (default: v78c test)
  NUM_EVAL      problems (default 5; keep small — this is a qualitative diagnostic)
  K             breaths (default 7)
  FIXED_LEN     padding (default 320)
  OUT_PATH      result dump (default /tmp/jsd_boundaries.txt)
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
    os.environ["WAIST_ATTN_SUPERVISION"] = "1"   # REQUIRED for stash
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


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in nats. Bounded [0, log(2)] ≈ [0, 0.693].
    p, q: 1-D distributions (need not sum to 1; renormalized here)."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def top_attn_positions(attn_row: np.ndarray, top_k: int = 3) -> list[int]:
    """Return the indices of the top-k positions in an attention distribution."""
    return np.argsort(attn_row)[-top_k:][::-1].tolist()


def main():
    CKPT = os.environ.get("CKPT",
        "/home/bryce/mycelium/.cache/gsm8k_steps_ckpts/v79_causal_prod_step1000.safetensors")
    TEST_PATH = os.environ.get("V77_TEST_PATH",
        "/home/bryce/mycelium/.cache/gsm8k_steps_v78c_test.jsonl")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "5"))
    K = int(os.environ.get("K", "7"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "320"))
    OUT_PATH = os.environ.get("OUT_PATH", "/tmp/jsd_boundaries.txt")

    print(f"=== JSD attention boundaries diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval={NUM_EVAL}  K={K}  fixed_len={FIXED_LEN}")
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

    lines: list[str] = []
    def emit(s=""):
        print(s)
        lines.append(s)

    # Aggregates across problems
    all_decode_jsd_peaks: list[float] = []
    all_breath_jsd: list[list[float]] = []   # one [K-1] vector per problem
    boundary_alignment_hits = 0
    boundary_alignment_n = 0

    for prob_i, ex in enumerate(examples):
        p_ids = tok.encode(ex.problem).ids
        prompt_len = len(p_ids)
        per_layer_ids = [tok.encode(" " + ex.per_layer_target[ell]).ids for ell in range(K)]
        final_layer_ids = per_layer_ids[-1]   # L6 DAG tokens
        eos_id = 0
        full_ids = list(p_ids) + list(final_layer_ids) + [eos_id]
        full_ids = full_ids[:FIXED_LEN]
        tokens_np = np.zeros((1, FIXED_LEN), dtype=np.int32)
        tokens_np[0, :len(full_ids)] = full_ids
        tokens_t = Tensor(tokens_np, dtype=dtypes.int).realize()
        kv_mask_np = np.zeros((1, FIXED_LEN), dtype=np.float32)
        kv_mask_np[0, :len(full_ids)] = 1.0
        kv_mask_t = Tensor(kv_mask_np, dtype=dtypes.float).realize()

        # One forward pass to get all per-breath waists.
        prompt_emb = model.embed(tokens_t).cast(dtypes.float).realize()
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens_t, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask_t)

        # Per-breath cross-attn — we run WaistController.forward for each breath
        # and stash its head-mean attn weights (set by WAIST_ATTN_SUPERVISION=1).
        # Shape: (1, T_q=FIXED_LEN, T_kv=FIXED_LEN). We slice on T_kv to prompt
        # range, and on T_q to the answer span.
        per_breath_attn = []  # list of (T_q, T_kv) numpy arrays
        for k in range(K):
            wk = waist_per_breath[k].cast(dtypes.float)
            _logits = model.waist_controller.forward(
                wk, prompt_emb, model.embed_out,
                k_idx=k, K_total=K, kv_mask=kv_mask_t)
            attn = model.waist_controller._last_cross_attn   # (1, T_q, T_kv)
            assert attn is not None, "Cross-attn stash empty — WAIST_ATTN_SUPERVISION must be 1"
            per_breath_attn.append(attn.numpy()[0])

        # Answer span on the decode side: positions [prompt_len-1 .. prompt_len-1 + len(final_layer)]
        answer_start = max(0, prompt_len - 1)
        answer_end = min(FIXED_LEN - 1, prompt_len - 1 + len(final_layer_ids))

        # ---------- VIEW A — DECODE-SIDE JSD on FINAL BREATH (K-1) ----------
        attn_final = per_breath_attn[K - 1]                # (T_q, T_kv)
        attn_to_prompt = attn_final[:, :prompt_len]        # restrict to prompt
        decode_jsd = []
        for t in range(answer_start, min(answer_end, FIXED_LEN - 2)):
            decode_jsd.append(jsd(attn_to_prompt[t], attn_to_prompt[t + 1]))
        if decode_jsd:
            decode_jsd_arr = np.array(decode_jsd)
            top_decode_idx = np.argsort(decode_jsd_arr)[-3:][::-1]   # top-3 peaks
            all_decode_jsd_peaks.append(float(decode_jsd_arr.max()))
        else:
            top_decode_idx = []

        # ---------- VIEW B — BREATH-SIDE JSD ----------
        # Mean attn over answer span → (T_kv,) per breath. Then JSD across breaths.
        breath_summary = []
        for k in range(K):
            attn_to_prompt_k = per_breath_attn[k][:, :prompt_len]
            mean_attn = attn_to_prompt_k[answer_start:answer_end + 1].mean(axis=0)
            breath_summary.append(mean_attn)
        breath_jsd = [jsd(breath_summary[k], breath_summary[k + 1]) for k in range(K - 1)]
        all_breath_jsd.append(breath_jsd)

        # ---------- Decode the output tokens at each decode position (for context) ----------
        # The model PREDICTS token at position t+1 given input up to t. The actual
        # gold next-token is in the sequence at position t+1.
        gold_next_at = lambda t: int(tokens_np[0, t + 1]) if t + 1 < FIXED_LEN else 0

        # ---------- Print per-problem summary ----------
        emit(f"\n{'=' * 78}")
        emit(f"=== Problem {prob_i + 1} / {len(examples)} (n_steps={ex.n_steps}, gold={ex.gold_answer}) ===")
        emit(f"Q: {ex.problem[:160]!r}")
        emit(f"prompt_len = {prompt_len},  final_L6 = {ex.per_layer_target[-1]!r}")
        emit("")

        emit(f"--- VIEW A: decode-side JSD on FINAL BREATH (top-3 peaks) ---")
        if len(top_decode_idx) > 0:
            for rank, jd_local_idx in enumerate(top_decode_idx):
                t = answer_start + jd_local_idx
                j_val = decode_jsd_arr[jd_local_idx]
                # The output token "about to be emitted" at this transition is at position t+1.
                gold_at_t = gold_next_at(t)
                gold_at_tp1 = gold_next_at(t + 1)
                tok_at_t = tok.id_to_token(gold_at_t) if gold_at_t else "<EOS>"
                tok_at_tp1 = tok.id_to_token(gold_at_tp1) if gold_at_tp1 else "<EOS>"
                # Top-3 prompt positions the controller attended to BEFORE and AFTER the shift.
                top_before = top_attn_positions(attn_to_prompt[t])
                top_after = top_attn_positions(attn_to_prompt[t + 1])
                tokens_before = [tok.id_to_token(int(tokens_np[0, pp])) for pp in top_before]
                tokens_after = [tok.id_to_token(int(tokens_np[0, pp])) for pp in top_after]
                emit(f"  #{rank + 1}  JSD={j_val:.3f}  output[{tok_at_t!r} → {tok_at_tp1!r}]")
                emit(f"       gaze before:  positions {top_before} = {tokens_before}")
                emit(f"       gaze after:   positions {top_after}  = {tokens_after}")
        else:
            emit("  (no answer-span decode positions — skipped)")

        emit("")
        emit(f"--- VIEW B: breath-side JSD (k → k+1, over answer-span-mean attn) ---")
        for k, j in enumerate(breath_jsd):
            top_attended = top_attn_positions(breath_summary[k])
            tokens_at_top = [tok.id_to_token(int(tokens_np[0, pp])) for pp in top_attended]
            emit(f"  L{k} → L{k+1}:  JSD={j:.3f}    breath-L{k} gaze top-3: {top_attended} = {tokens_at_top}")
        # Also report the LAST breath's gaze (no JSD pair for it).
        top_attended_last = top_attn_positions(breath_summary[K - 1])
        tokens_at_top_last = [tok.id_to_token(int(tokens_np[0, pp])) for pp in top_attended_last]
        emit(f"  L{K-1}: (final)               breath-L{K-1} gaze top-3: {top_attended_last} = {tokens_at_top_last}")

    # ---------- Aggregate ----------
    emit("")
    emit("=" * 78)
    emit(f"=== Aggregate over {len(examples)} problems ===")
    if all_decode_jsd_peaks:
        emit(f"  Mean max decode-side JSD per problem: {np.mean(all_decode_jsd_peaks):.4f}")
        emit(f"  Max  max decode-side JSD per problem: {np.max(all_decode_jsd_peaks):.4f}")
        emit(f"  Min  max decode-side JSD per problem: {np.min(all_decode_jsd_peaks):.4f}")
    if all_breath_jsd:
        arr = np.array(all_breath_jsd)  # (N, K-1)
        emit(f"")
        emit(f"  Mean breath-side JSD per (k → k+1) over {arr.shape[0]} problems:")
        for k in range(arr.shape[1]):
            emit(f"    L{k} → L{k+1}: {arr[:, k].mean():.4f}   (std {arr[:, k].std():.4f})")
        emit(f"")
        emit(f"  JSD upper bound (log 2) = {np.log(2):.4f}")
        emit(f"  Interpretation:")
        emit(f"    breath JSD ≈ 0     → breaths read the SAME prompt regions")
        emit(f"    breath JSD ≈ log 2 → breaths read COMPLETELY DIFFERENT prompt regions")
        emit(f"    middle (0.2-0.5)   → meaningful specialization with some shared content")
    emit("")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[wrote {OUT_PATH}]")


if __name__ == "__main__":
    main()
