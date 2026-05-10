"""Smoke test for KV-cached generation: correctness + speedup vs the uncached path."""
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
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/arith_ckpts/arith_step1500.safetensors")
    n_loops = getenv("LOOPS", 4)
    n_problems = getenv("N", 8)

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    load_ckpt(model, ckpt)

    tok = load_tokenizer()
    problems = generate_math("ARITH", n_problems, seed=999, digit_spacing=True)

    print(f"=== KV cache smoke: {ckpt} @ A={n_loops}, N={n_problems} ===\n")

    print("--- Uncached (re-breathe per token) ---")
    t0 = time.perf_counter()
    correct_uncached = 0
    uncached_outs = []
    for ex in problems:
        prompt_ids = tok.encode(ex.problem).ids
        outs = multi_cycle_generate(model, tok, prompt_ids, n_loops=[n_loops, 1],
                                    n_cycles=1, max_new_per_cycle=12, use_kv_cache=False)
        gen_text = tok.decode(outs[0])
        parsed = parse_int_answer(gen_text)
        ok = parsed == ex.answer
        if ok: correct_uncached += 1
        uncached_outs.append((gen_text, parsed))
        print(f"  {ex.problem!r} -> {gen_text.strip()!r} parsed={parsed} gold={ex.answer} {'OK' if ok else 'WRONG'}")
    t_uncached = time.perf_counter() - t0
    print(f"  uncached: {correct_uncached}/{n_problems} in {t_uncached:.1f}s\n")

    print("--- Cached (single-pass per token after Phase A) ---")
    t0 = time.perf_counter()
    correct_cached = 0
    cached_outs = []
    for ex in problems:
        prompt_ids = tok.encode(ex.problem).ids
        outs = multi_cycle_generate(model, tok, prompt_ids, n_loops=[n_loops, 1],
                                    n_cycles=1, max_new_per_cycle=12, use_kv_cache=True)
        gen_text = tok.decode(outs[0])
        parsed = parse_int_answer(gen_text)
        ok = parsed == ex.answer
        if ok: correct_cached += 1
        cached_outs.append((gen_text, parsed))
        print(f"  {ex.problem!r} -> {gen_text.strip()!r} parsed={parsed} gold={ex.answer} {'OK' if ok else 'WRONG'}")
    t_cached = time.perf_counter() - t0
    print(f"  cached: {correct_cached}/{n_problems} in {t_cached:.1f}s\n")

    print(f"=== Summary ===")
    print(f"  uncached: {correct_uncached}/{n_problems} ({t_uncached:.1f}s)")
    print(f"  cached:   {correct_cached}/{n_problems} ({t_cached:.1f}s)")
    print(f"  speedup:  {t_uncached/t_cached:.1f}x")
    matches = sum(1 for (gu, _), (gc, _) in zip(uncached_outs, cached_outs) if gu == gc)
    print(f"  exact text match: {matches}/{n_problems}")


if __name__ == "__main__":
    main()
