"""v77 DAG eval — final breath emits a SymPy-executable DAG, SymPy computes the answer.

Inference flow:
    1. Load a ckpt trained with V77_DAG_TRAINING=1 (N_BREATHS=6).
    2. For each test problem, run K=6 inner breaths and decode the FINAL breath's
       output greedily. The final breath should emit a Layer-5 DAG.
    3. Parse the decoded DAG string and execute it via dag_to_answer().
    4. Compare to gold_answer (tolerance 1e-3 OR exact equality).

Outputs (in addition to per-example samples):
    - Overall accuracy
    - DAG parse success rate (% executable)
    - DAG correctness rate among parseable (% matched gold)
    - Failure breakdown (malformed / parseable_but_wrong / undefined_var / div_by_zero)

Env vars:
    CKPT=...           required. Path to the trained safetensors.
    V77_TEST_PATH=...  test set JSONL (default .cache/gsm8k_steps_v77_test.jsonl)
    NUM_EVAL=60        max problems to eval
    K=6                inner breaths (matches training)
    FIXED_LEN=320      input padding (matches training)
    BATCH=8            eval batch size
    MAX_NEW=120        max tokens to generate per problem (DAG fits comfortably)
    USE_KV_CACHE=1     use the closure-fused JIT decoder

The eval is intentionally simple: greedy decode breath 5 only. Earlier-breath
"segmented" decoding is supported by eval_ckpt_controller_segmented.py if needed,
but v77's principle is that the final breath alone produces the executable target.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_v77

# Make scripts/ importable so we can pull in v77_sympy_eval.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v77_sympy_eval import dag_to_answer  # type: ignore


_EVAL_JIT_CACHE: dict = {}


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


def _compile_jit_final_breath_decode(model, K: int, fixed_len: int, B: int):
    """JIT: K breaths on a full sequence; return argmax token IDs at t_pos[b] for FINAL breath only.

    Inputs:
      tokens:   (B, fixed_len) int
      t_pos_t:  (B,) int — current position per example (predicts next token)
      kv_mask:  (B, fixed_len) float — 1.0 at valid (already-emitted) positions, 0.0 elsewhere

    **v78c (2026-05-25) inference fix:**
        Training had `tokens = [prompt, gold_L6, EOS, zeros]` so the WaistController's
        cross-attention KV (`prompt_emb = embed(tokens)`) included gold L6 token
        embeddings at positions [prompt_len .. prompt_len+len(L6)-1]. The model learned
        to leverage these positions (lookahead leakage; no causal mask in cross-attn).
        At autoregressive eval those positions contain zeros (= EOS embeddings since
        token id 0 has a non-zero embedding), which is out-of-distribution.

        Fix: pass `kv_mask` with 1.0 only at positions [0..current_len-1] so the
        WaistController's cross-attn applies an additive -1e4 penalty at non-valid
        positions and effectively ignores them. This eliminates the "answer = answer
        = answer = ..." degenerate cascade where the model attends to ghost EOS-pad
        positions for its structural-token predictions.

    **v79 (2026-05-26) eval fix — notebook_pool_mask plumbing:**
        Training-time `breathe_with_lookup` receives `notebook_pool_mask=kv_mask`
        when V79_CAUSAL_MASKS=1, so the notebook's attention-pool write only reads
        prompt positions (positions where actual content lives). Eval here was
        passing `kv_mask` to WaistController.forward (cross-attn) but NOT to
        breathe_with_lookup (notebook), so the notebook attended to zero-padding
        and the rest of the model saw an OOD running integral. Result: AR collapsed
        to "x1 = x1 = x1 ..." cascade even though TF (with proper masks) produced
        clean DAG output (verified via diag_v79_per_breath_decode on v80_prod_step400:
        TF "x0 = 2 * 2 ; x1 = x * x0 ; answer = x1" vs AR cascade).

        Fix below: pass kv_mask through breathe_with_lookup as notebook_pool_mask
        so eval byte-matches what training supervised.

    Returns:
      (B,) int — argmax of breath-(K-1)'s logits at t_pos per example.
    """
    key = (id(model), int(K), int(fixed_len), int(B))
    if key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[key]

    @TinyJit
    def _fwd(tokens, t_pos_t, kv_mask):
        # v79 (2026-05-26): pass kv_mask as notebook_pool_mask so the notebook
        # only attends to prompt positions, matching training-time behavior with
        # V79_CAUSAL_MASKS=1. Without this, AR generation collapsed into a cascade
        # even on well-trained ckpts (the notebook's pooled rep was OOD).
        # v81 (2026-05-26): also pass kv_mask as main_attn_mask — zeros input embeddings
        # at answer-span positions AND blocks main-self-attn keys there. Required for the
        # v81 paradigm to match its training-time masking (verified via diag_v81_masking_audit).
        from mycelium.breathing import V81_MAIN_ATTN_MASK as _V81MAM_EVAL, MULTI_HEAD_WAIST as _MHW_EVAL
        _main_attn = kv_mask if _V81MAM_EVAL else None
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask, main_attn_mask=_main_attn)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        positions = Tensor.arange(fixed_len)
        gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(B, 1)).reshape(B, fixed_len, 1).cast(dtypes.float)
        # FINAL breath only.
        wk = waist_per_breath[K - 1].cast(dtypes.float)
        wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)
        lk = model.waist_controller.forward(wk_at_pos, prompt_emb, model.embed_out,
                                              k_idx=K - 1, K_total=K,
                                              kv_mask=kv_mask)
        # v81: multi-head returns a dict; for DAG eval we want the OPS head as the
        # canonical decode head (the model's "main vocab" emission). The DAG parser
        # then handles the multi-head output structure if needed. For v81 inference,
        # ALL 4 heads need to be sampled to reconstruct the 4-list B6 text; this JIT
        # only returns ops head, so v81-aware eval uses a separate path (see scripts/
        # eval_v77_dag.py's multi-head branch below; this JIT is the legacy path).
        if _MHW_EVAL:
            lk = lk["ops"]
        return lk[:, :, :50277].argmax(axis=-1).reshape(B).realize()

    _EVAL_JIT_CACHE[key] = _fwd
    return _fwd


def generate_final_breath_batch(model, prompt_ids_list, tok, K, fixed_len,
                                  max_new=120, eos_id=0):
    """Greedy decode using only the FINAL breath's waist.

    Returns: list of lists of generated token IDs (one per example, excluding prompt).
    """
    B = len(prompt_ids_list)
    prompt_lens = [len(p) for p in prompt_ids_list]
    current_lens = list(prompt_lens)
    max_prompt = max(current_lens)
    assert max_prompt + max_new <= fixed_len, (
        f"need fixed_len >= {max_prompt + max_new}, got {fixed_len}")
    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    # v79 (2026-05-26): kv_mask covers ONLY the prompt range [0, prompt_lens[b]),
    # never extending into generated-token positions. This matches training where
    # kv_mask is prompt-range only (the gold L6 / answer-span positions are
    # masked-out in cross-attn and in notebook pool). Without this, the cross-attn
    # and notebook see OOD embeddings from positions training never trained for,
    # which collapses AR generation into a degenerate cascade.
    kv_mask_np = np.zeros((B, fixed_len), dtype=np.float32)
    for b in range(B):
        tokens_np[b, :prompt_lens[b]] = prompt_ids_list[b]
        kv_mask_np[b, :prompt_lens[b]] = 1.0
    generated_per_ex = [[] for _ in range(B)]
    active = [True] * B

    fwd = _compile_jit_final_breath_decode(model, K, fixed_len, B)
    t_pos_np = np.zeros((B,), dtype=np.int32)
    t_pos_t = Tensor(t_pos_np, dtype=dtypes.int).contiguous().realize()
    kv_mask_t = Tensor(kv_mask_np, dtype=dtypes.float).contiguous().realize()

    for _step in range(max_new):
        if not any(active):
            break
        for b in range(B):
            t_pos_np[b] = current_lens[b] - 1
            # kv_mask stays fixed at prompt-range (matches training).
        t_pos_t.assign(Tensor(t_pos_np, dtype=dtypes.int)).realize()
        # kv_mask_t is static (prompt-range) — no need to re-assign each step.
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        next_ids_t = fwd(tokens, t_pos_t, kv_mask_t)
        next_ids_np = next_ids_t.numpy()
        for b in range(B):
            if not active[b]:
                continue
            next_tok = int(next_ids_np[b])
            generated_per_ex[b].append(next_tok)
            current_lens[b] += 1
            if current_lens[b] < fixed_len:
                tokens_np[b, current_lens[b] - 1] = next_tok
            if next_tok == eos_id or current_lens[b] >= fixed_len:
                active[b] = False
    return generated_per_ex


# DAG extraction: the model emits text like "x0 = 50 + 15 ; answer = x0".
# We try to extract the DAG string from the generated text — strip pre-amble,
# trim post-EOS junk, etc. Be tolerant: take the longest "x" + "=" + "answer ="
# substring possible.
import re

# A DAG statement looks like: "xN = <expr> ;" or "answer = <expr>".
_STATEMENT_RE = re.compile(r"(x\d+\s*=\s*[^;]+?\s*;)|(answer\s*=\s*[^;\n]+)")


def extract_dag(text: str) -> str:
    """Pull out the DAG-shaped substring from the model's free-form output.

    **v78c (2026-05-25) extraction fix:**
        v78b's autoregressive eval often emits multiple `answer = ...` statements
        in a row (cascade from broken cross-attn that we fix elsewhere). The
        previous "use LAST `answer =`" heuristic picked the deepest reference
        which usually pointed to undefined variables.

        New approach: walk statements, accumulate xN = ... definitions, find the
        FIRST `answer = ...` whose RHS only references DEFINED variables. Return
        the substring `(all xN defs encountered up to that point) + ;(chosen answer)`.
        SKIPS intermediate invalid answer statements so dag_to_answer sees a clean DAG.

        If no valid answer found, fall back to the FIRST `answer = ...` (matches
        v78b behavior — will still flag undefined_variable).
    """
    text = text.strip()
    # Find first xN = ... statement.
    first = re.search(r"x\d+\s*=", text)
    if not first:
        return ""
    start = first.start()

    # Walk statements separated by `;` from `start`.
    defined_vars = set()
    xN_re = re.compile(r"x(\d+)\s*=\s*([^;]+)")
    ans_re = re.compile(r"answer\s*=\s*([^;\n]+)")
    var_ref_re = re.compile(r"x(\d+)")
    allowed = set("0123456789xX_+-*/(). ")

    pos = start
    xN_defs = []  # list of (var_name, rhs) tuples in order encountered
    chosen_answer_rhs = None
    first_answer_rhs = None

    while pos < len(text):
        # Skip leading whitespace and semicolons
        while pos < len(text) and text[pos] in " ;":
            pos += 1
        if pos >= len(text):
            break

        # Try xN = ...
        m_x = xN_re.match(text, pos)
        if m_x:
            var_num = int(m_x.group(1))
            rhs = m_x.group(2).strip()
            # Only add this var if its rhs is parseable (only refs to defined vars)
            refs = set(f"x{n}" for n in var_ref_re.findall(rhs))
            if refs.issubset(defined_vars):
                var_name = f"x{var_num}"
                # Skip duplicates (model sometimes redefines x0)
                if var_name not in defined_vars:
                    xN_defs.append((var_name, rhs))
                    defined_vars.add(var_name)
            pos = m_x.end()
            if pos < len(text) and text[pos] == ";":
                pos += 1
            continue

        # Try answer = ...
        m_a = ans_re.match(text, pos)
        if m_a:
            rhs = m_a.group(1).strip()
            ans_rhs_start = m_a.start(1)
            ans_end = ans_rhs_start
            while ans_end < len(text) and (text[ans_end] in allowed or text[ans_end].isspace()):
                ans_end += 1
            # Re-extract clean rhs
            clean_rhs = text[ans_rhs_start:ans_end].strip().rstrip(";").strip()
            refs = set(f"x{n}" for n in var_ref_re.findall(clean_rhs))
            all_defined = refs.issubset(defined_vars)
            if first_answer_rhs is None:
                first_answer_rhs = clean_rhs
            if all_defined and chosen_answer_rhs is None:
                chosen_answer_rhs = clean_rhs
                break  # found valid answer; stop walking
            pos = ans_end
            if pos < len(text) and text[pos] == ";":
                pos += 1
            continue

        # If neither pattern matches, advance by 1
        pos += 1

    # Pick chosen answer if found, else first answer
    ans_rhs = chosen_answer_rhs if chosen_answer_rhs is not None else first_answer_rhs
    if ans_rhs is None:
        return ""

    # Build clean DAG: xN defs in order + answer
    parts = [f"{name} = {rhs}" for name, rhs in xN_defs]
    parts.append(f"answer = {ans_rhs}")
    return " ; ".join(parts)


def classify_failure(dag_str: str, gold: float) -> str:
    """Classify why dag_to_answer didn't match gold. Returns a category string."""
    if not dag_str:
        return "no_dag_found"
    val = dag_to_answer(dag_str)
    if val is None:
        # Try to find the specific reason by attempting partial parses.
        if "answer" not in dag_str:
            return "no_answer_statement"
        # Check for undefined variable references.
        defined = set()
        statements = [s.strip() for s in dag_str.split(";") if s.strip()]
        for stmt in statements:
            if "=" not in stmt:
                return "malformed_statement"
            lhs, _, rhs = stmt.partition("=")
            lhs = lhs.strip()
            # Check rhs for variables not defined.
            used_vars = re.findall(r"\bx\d+\b", rhs)
            for uv in used_vars:
                if uv not in defined:
                    return "undefined_variable"
            defined.add(lhs)
        if "/0" in dag_str.replace(" ", "") or "/ 0" in dag_str:
            return "div_by_zero"
        return "sympy_parse_error"
    return "parseable_but_wrong_answer"


def main():
    cfg_kwargs = {}
    for env_key, cfg_key in [("HIDDEN", "hidden"), ("N_HEADS", "n_heads"),
                              ("HEAD_DIM", "head_dim"), ("FFN", "ffn"),
                              ("CONTROLLER_HIDDEN", "controller_hidden")]:
        v = os.environ.get(env_key)
        if v is not None:
            cfg_kwargs[cfg_key] = int(v)
    cfg = Config(**cfg_kwargs)
    if cfg_kwargs:
        print(f"[Config overrides] {cfg_kwargs}")

    CKPT = os.environ.get("CKPT", "")
    TEST_PATH = os.environ.get("V77_TEST_PATH", ".cache/gsm8k_steps_v77_test.jsonl")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "60"))
    K = int(os.environ.get("K", "6"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "320"))
    BATCH = int(os.environ.get("BATCH", "8"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "120"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    print(f"=== v77 DAG eval (K={K} breaths, final breath -> DAG -> SymPy) ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}  max_new: {MAX_NEW}")
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: test set not found at {TEST_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\nloading v77 test set...")
    examples = load_gsm8k_v77(TEST_PATH, min_k=MIN_K, max_k=MAX_K,
                                require_sympy_match=True, bucket_by_k=False)
    examples = examples[:NUM_EVAL]
    print(f"  using {len(examples)} examples")

    tok = load_tokenizer()
    print(f"\nloading Pythia + ckpt...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    if not CKPT:
        print("WARNING: CKPT not set — evaluating untrained model.")
    else:
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
        del ckpt_sd

    Tensor.training = False

    correct = 0
    parseable = 0
    parseable_correct = 0
    failures = {"no_dag_found": 0, "no_answer_statement": 0,
                "malformed_statement": 0, "undefined_variable": 0,
                "div_by_zero": 0, "sympy_parse_error": 0,
                "parseable_but_wrong_answer": 0}
    samples_to_show = 5
    t0 = time.perf_counter()

    for batch_start in range(0, len(examples), BATCH):
        batch = examples[batch_start:batch_start + BATCH]
        prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
        # Pad batch to BATCH for JIT shape stability (the JIT caches per B).
        if len(prompt_ids) < BATCH:
            # Replicate the last example to fill the batch.
            while len(prompt_ids) < BATCH:
                prompt_ids.append(prompt_ids[-1])
            pad_n = BATCH - len(batch)
        else:
            pad_n = 0

        gen_per_ex = generate_final_breath_batch(model, prompt_ids, tok,
                                                   K=K, fixed_len=FIXED_LEN,
                                                   max_new=MAX_NEW)
        for i, ex in enumerate(batch):  # ignore padded entries
            gen_text = tok.decode(gen_per_ex[i])
            dag = extract_dag(gen_text)
            sympy_val = dag_to_answer(dag) if dag else None
            ok = sympy_val is not None and abs(sympy_val - ex.gold_answer) < 1e-3
            if ok:
                correct += 1
                parseable += 1
                parseable_correct += 1
            elif sympy_val is not None:
                # Parses but wrong answer.
                parseable += 1
                failures["parseable_but_wrong_answer"] += 1
            else:
                cat = classify_failure(dag, ex.gold_answer)
                failures[cat] = failures.get(cat, 0) + 1
            if samples_to_show > 0 and batch_start == 0 and i < samples_to_show:
                print(f"\n  Q: {ex.problem[:100]!r}")
                print(f"  gen: {gen_text.strip()[:300]!r}")
                print(f"  extracted DAG: {dag!r}")
                print(f"  sympy: {sympy_val}  gold: {ex.gold_answer}  {'OK' if ok else 'WRONG'}")
                samples_to_show -= 1

    total = len(examples)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    parse_pct = parseable / max(total, 1) * 100
    parse_correct_pct = parseable_correct / max(parseable, 1) * 100
    print(f"\n=== v77 DAG eval results ({dt:.1f}s) ===")
    print(f"  accuracy:           {acc:.1f}% ({correct}/{total})")
    print(f"  DAG parse rate:     {parse_pct:.1f}% ({parseable}/{total})")
    if parseable > 0:
        print(f"  correct of parseable: {parse_correct_pct:.1f}% ({parseable_correct}/{parseable})")
    print(f"  failure breakdown:")
    for cat, n in sorted(failures.items(), key=lambda kv: -kv[1]):
        if n > 0:
            print(f"    {cat}: {n}")


if __name__ == "__main__":
    main()
