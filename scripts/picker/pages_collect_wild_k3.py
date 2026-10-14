"""pages_collect_wild_k3.py -- THE K3 COVERAGE COLLECT (2026-10-14, Fable): breath-0 attention
for the K3 chassis pair on wild, since the membrane-rack collect FAILED on K3 bodies for a
7-breath assertion (.cache/membrane_rack_K3_241.log / _242.log: `assert len(fat_all) == K_B,
(len(fat_all), K_B)` -> AssertionError: (3, 7) -- ALG_LOOP_STOP=3 makes forward()'s own
ALG_MINE_BREATHS tap hand back only 3 breaths of fat_all, not ALG_BREATH=7; membrane_rack.py
hard-asserts the full count. This script sizes its own output arrays from the ACTUAL len(fat_all)
instead of asserting against ALG_BREATH, which is all the (3,7) mismatch needed.

DEV=CPU throughout; never touches .cache/gpu.lock. Reuses, read-only, the tokenization/gold/
sentence-index fields already banked in .cache/membrane_raw_wild_PMS8_241.npz (ids, tokmask,
sent, g_presence, g_ftype, g_digits) -- those are checkpoint-independent (same 311-row wild
fixture, same frozen tokenizer, same custody gold) and are NOT recomputed; this script computes
and writes only the NEW, checkpoint-dependent arrays (argmax_breath, fat_mass) for K3_241 and
K3_242's own breath-0..2 pooled slot->token attention. The frozen Llama trunk forward is also
checkpoint-independent, so it runs ONCE and is reused for both bodies.

Outputs (NEW artifacts, pages_ prefix, nothing banked touched):
  .cache/pages_membrane_raw_wild_K3_241.npz
  .cache/pages_membrane_raw_wild_K3_242.npz
  (schema matches membrane_raw_wild_PMS8_241.npz, except fat_mass/argmax_breath's breath axis
  is K_EFF (<=3 under ALG_LOOP_STOP=3), not ALG_BREATH=7)

Usage: DEV=CPU .venv/bin/python3 scripts/picker/pages_collect_wild_k3.py
"""
import os
import sys
import json

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
RAW_BANKED_PMS8 = ".cache/membrane_raw_wild_PMS8_241.npz"   # read-only reuse of ids/tokmask/sent/gold
BODIES = ["K3_241", "K3_242"]

# Verbatim from pages_collect_valid2.py's own _FAM (THE CHASSIS / S8 env), plus THE LOOP STOP
# flag (rack_chain_K3.sh's own LSX="ALG_LOOP_STOP=3") -- this IS the K3_241/K3_242 training env.
_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_MINE_BREATHS": "1",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
    "ALG_LOOP_STOP": "3",   # THE K3 FLAG (rack_chain_K3.sh's LSX) -- the only delta from PMS8's own env
}
_FAM["BIND_CODES"] = ".cache/bindbus_codes512r.npz"


def _set_env():
    for k, v in _FAM.items():
        os.environ.setdefault(k, v)
    assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only by brief"
    assert os.environ.get("ALG_LOOP_STOP") == "3", "the K3 flag must be set before phase1_algebra_head import"


def main():
    _set_env()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                        LLAMA_3_2_1B_CFG, _rms_norm)
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, build_slot_masks, alt2_fact_buf,
                                      K_VARS, L_FAC, T_ALG, H_TRUNK)
    assert H.ALG_LOOP_STOP == 3, (H.ALG_LOOP_STOP, "module-level env read did not pick up LSX")

    # ---- reuse, read-only, the checkpoint-independent fields already banked for PMS8_241 ----
    bank = np.load(RAW_BANKED_PMS8)
    ids, tokmask, sent = bank["ids"], bank["tokmask"], bank["sent"]
    g_presence, g_ftype, g_digits = bank["g_presence"], bank["g_ftype"], bank["g_digits"]
    n = ids.shape[0]
    rows = [json.loads(l) for l in open(WILD_JSONL)]
    assert len(rows) == n, (len(rows), n, "wild_admitted_holdout.jsonl row count must match the banked raw npz")
    print(f"[pages-collect-k3] reused banked ids/tokmask/sent/gold from {RAW_BANKED_PMS8} (n={n})", flush=True)

    # ---- the frozen trunk forward (checkpoint-independent) runs ONCE, shared by both bodies ----
    class _Host:
        pass
    host = _Host()
    sd = load_llama_weights(os.path.join(H._ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    states = np.zeros((n, T_ALG, H_TRUNK), np.float16)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        x = host.llama_embed[Tensor(ids[sl].astype(np.int32), dtype=dtypes.int)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        c = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(c).all()
        states[sl] = c.astype(np.float16)
        print(f"[pages-collect-k3] trunk {min(s0+8,n)}/{n}", flush=True)
    del host

    for BODY in BODIES:
        ckpt = f".cache/sharp_{BODY}.safetensors"
        raw_out = f".cache/pages_membrane_raw_wild_{BODY}.npz"
        print(f"[pages-collect-k3] == {BODY} (ckpt={ckpt}) ==", flush=True)
        p = build_params(0)
        sdp = safe_load(ckpt)
        assert set(sdp.keys()) == set(p.keys()), (sorted(set(sdp) - set(p))[:4], sorted(set(p) - set(sdp))[:4])
        for k in p:
            p[k].assign(sdp[k].to(p[k].device).cast(p[k].dtype)).realize()

        argmax_breath = None
        fat_mass = None
        k_eff = None
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            ts = Tensor(np.ascontiguousarray(states[sl_p]), dtype=dtypes.half)
            tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
            se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)

            o0 = forward(p, ts, tk, se, tree=None)
            onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(onp0, sent[sl_p].astype(np.int32))
            _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
            _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
            _nv = np.array([rows[int(i)].get("n_vars", K_VARS) for i in sl_p])
            _ma = np.array([rows[int(i)].get("m", 0) for i in sl_p])
            fb = alt2_fact_buf(_oa, sent[sl_p].astype(np.int32), _nv, _ma)
            fact_t = Tensor(fb, dtype=dtypes.float)

            o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t, tree=None)
            fat_all = [t.realize().numpy() for t in o["fat_all"]]
            if k_eff is None:
                k_eff = len(fat_all)   # THE FIX: size from the ACTUAL count, not ALG_BREATH (7)
                argmax_breath = np.full((n, k_eff, L_FAC), -1, np.int32)
                fat_mass = np.zeros((n, k_eff, L_FAC, T_ALG), np.float16)
                print(f"[pages-collect-k3] {BODY}: len(fat_all)={k_eff} (ALG_LOOP_STOP=3 -> expect 3, not ALG_BREATH=7)", flush=True)
            assert len(fat_all) == k_eff, (BODY, len(fat_all), k_eff, "breath count changed mid-run")
            tkm = tokmask[sl_p].astype(np.float32) > 0.5
            neg = np.where(tkm, 0.0, -1e9).astype(np.float32)
            for kb in range(k_eff):
                fa = fat_all[kb] + neg[:, None, :]
                argmax_breath[sl, kb, :] = fa[:len(sl)].argmax(-1)
                fat_mass[sl, kb, :, :] = fat_all[kb][:len(sl)].astype(np.float16)
            print(f"[pages-collect-k3] {BODY} membrane {min(s0+8,n)}/{n}", flush=True)

        np.savez(raw_out, argmax_breath=argmax_breath, fat_mass=fat_mass,
                 ids=ids, tokmask=tokmask, sent=sent,
                 g_presence=g_presence, g_ftype=g_ftype, g_digits=g_digits)
        print(f"[pages-collect-k3] wrote {raw_out} (K_EFF={k_eff})", flush=True)


if __name__ == "__main__":
    main()
