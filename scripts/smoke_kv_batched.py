"""Smoke test: batched cached generation vs B=1 uncached. Correctness + speed."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, parse_int_answer, SEP
from mycelium.l3_training import multi_cycle_generate


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


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l3_ckpts/l3_spaced_step600.safetensors")
    n_loops = getenv("LOOPS", 4)
    n_problems = getenv("N", 8)
    cache_max_len = getenv("CACHE_MAX_LEN", 0) or None  # 0 means use cfg.max_seq_len

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    load_ckpt(model, ckpt)

    tok = load_tokenizer()
    problems = generate_math("ARITH", n_problems, seed=999, digit_spacing=True)
    sep_ids = tok.encode(SEP).ids

    print(f"=== Batched KV cache smoke: ckpt={os.path.basename(ckpt)} A={n_loops} N={n_problems} ===\n")

    # --- 1) Uncached, sequential ---
    print("--- Uncached (sequential, re-breathe per token) ---")
    t0 = time.perf_counter()
    correct_un = 0
    uncached_outs = []
    for ex in problems:
        prompt_ids = tok.encode(ex.problem).ids
        outs = multi_cycle_generate(model, tok, prompt_ids, n_loops=[n_loops, 1],
                                    n_cycles=1, max_new_per_cycle=12, use_kv_cache=False)
        gen_text = tok.decode(outs[0])
        parsed = parse_int_answer(gen_text)
        ok = parsed == ex.answer
        if ok: correct_un += 1
        uncached_outs.append(gen_text)
    t_un = time.perf_counter() - t0
    print(f"  {correct_un}/{n_problems} in {t_un:.1f}s\n")

    # --- 2) Batched cached: warm-up then timed ---
    prompt_id_lists = [tok.encode(ex.problem).ids for ex in problems]
    print("--- Batched cached: warm-up (JIT compile) ---")
    t0 = time.perf_counter()
    outs_batched = model.cached_generate_batch(
        prompt_id_lists, n_loops=n_loops, max_new=12,
        stop_token_ids=[0], stop_seq=sep_ids,
        cache_max_len=cache_max_len,
    )
    Device[Device.DEFAULT].synchronize()
    t_warm = time.perf_counter() - t0
    print(f"  warm-up: {t_warm:.1f}s\n")

    print("--- Batched cached: steady-state (JIT cached) ---")
    t0 = time.perf_counter()
    outs_batched = model.cached_generate_batch(
        prompt_id_lists, n_loops=n_loops, max_new=12,
        stop_token_ids=[0], stop_seq=sep_ids,
        cache_max_len=cache_max_len,
    )
    Device[Device.DEFAULT].synchronize()
    t_b = time.perf_counter() - t0
    correct_b = 0
    batched_outs = []
    for ex, ids in zip(problems, outs_batched):
        gen_text = tok.decode(ids)
        parsed = parse_int_answer(gen_text)
        ok = parsed == ex.answer
        if ok: correct_b += 1
        batched_outs.append(gen_text)
        flag = "OK" if ok else "WRONG"
        print(f"  {ex.problem!r} -> {gen_text.strip()!r} parsed={parsed} gold={ex.answer} [{flag}]")
    print(f"  batched: {correct_b}/{n_problems} in {t_b:.1f}s\n")

    print(f"=== Summary ===")
    print(f"  uncached:        {correct_un}/{n_problems} ({t_un:.1f}s)")
    print(f"  batched warm-up: {t_warm:.1f}s")
    print(f"  batched steady:  {correct_b}/{n_problems} ({t_b:.1f}s)")
    print(f"  speedup vs uncached (steady): {t_un/max(t_b, 0.001):.1f}x")
    print(f"  speedup vs uncached (warm):   {t_un/max(t_warm, 0.001):.1f}x")
    matches = sum(1 for u, b in zip(uncached_outs, batched_outs) if u == b)
    print(f"  exact text match: {matches}/{n_problems}")


if __name__ == "__main__":
    main()
