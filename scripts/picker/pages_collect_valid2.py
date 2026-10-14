"""pages_collect_valid2.py -- THE READ NOTEBOOK (2026-10-14, Opus): collection step.

Builds the valid2-slice analogue of the banked wild membrane/correctness arrays, since no
valid2 attention array exists (dn_read_a.py's own finding, 2026-10-12: the gold-states npz for
pm35cslicevalid2 was deleted in the 2026-10-11 cleanup; regenerating needs a fresh frozen-trunk
L0-L3 CPU precompute). This task's brief explicitly authorizes exactly that for 128 rows, so this
script runs it: DEV=CPU throughout, no GPU, no .cache/gpu.lock touch.

What this runs (said plainly, per the brief's "say what you ran"):
  1. Loads the first 128 rows of .cache/form_pm35c_slice1024_valid2.jsonl (deterministic prefix
     slice, not a random sample -- stated so the choice is auditable) -- the diet slice PMS8_241's
     own training mix holds out (775 rows total).
  2. A FRESH frozen-trunk (Llama-3.2-1B L0-L3 + embed, never trained) CPU forward pass over those
     128 rows -- phase1_algebra_head.do_precompute()'s own method, inlined here because
     do_precompute() writes to the FIXED STATES_NPY/STATES_NPZ split-name paths (would collide
     with the canonical 'pm35cslicevalid2' name other scripts expect to mean the FULL 775-row
     slice) -- this script's own states never touch those paths; outputs are pages_-prefixed only.
  3. PMS8_241's own two-pass loop_val cycle (pass-1 unmasked parse -> build_slot_masks +
     alt2_fact_buf -> pass-2 masked walk) over those 128 rows' trunk states, ALG_MINE_BREATHS=1
     tapping out["fat_all"] for the per-breath pooled (head-mean) slot->token attention -- the
     SAME method goldmention_read.py / membrane_scale.py use, same checkpoint/env (sharp_PMS8_241,
     the _FAM dict copied verbatim from goldmention_read.py), run on CPU for 128 rows instead of
     the banked GPU run's 311/775.
  4. Correctness per GIVEN/REL slot via dn_common.score_row_slots (loop_val.py's own "ok"
     convention, reused verbatim) against gold built by build_gold() straight from the fixture's
     own `factors` field (the custody-gold law: never the pen `solution` field).

Outputs (NEW artifacts, pages_ prefix, nothing banked touched):
  .cache/pages_valid2_128.jsonl               -- the 128-row slice itself (reproducibility)
  .cache/pages_membrane_raw_valid2_PMS8_241.npz  -- schema-identical to membrane_raw_wild_PMS8_241.npz
      (argmax_breath, fat_mass, ids, tokmask, sent, g_presence, g_ftype, g_digits)
  .cache/pages_ps_legal_valid2_PMS8_241.npz      -- schema-identical to ps_legal_wild_PMS8_241.npz
      (rows, slots, ok) -- GIVEN+REL slots scored (score_row_slots covers both ftypes)

Usage: DEV=CPU .venv/bin/python3 scripts/picker/pages_collect_valid2.py
"""
import os
import sys
import json

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

VALID2_FULL = ".cache/form_pm35c_slice1024_valid2.jsonl"
N_ROWS = int(os.environ.get("PAGES_N_VALID2", "128"))   # smoke-test override only; brief's own number is 128
SLICE_JSONL = ".cache/pages_valid2_128.jsonl"
CKPT = ".cache/sharp_PMS8_241.safetensors"
RAW_OUT = ".cache/pages_membrane_raw_valid2_PMS8_241.npz"
OK_OUT = ".cache/pages_ps_legal_valid2_PMS8_241.npz"

# Verbatim from goldmention_read.py's _FAM (THE CHASSIS / S8 env for sharp_PMS8_241) -- the ALG_TEST
# entry is unused here (we build ts/tk/se/gold ourselves, bypassing load_alg's file-backed door
# entirely) but left in place so build_params()/forward() see an identical architecture env.
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
}
_FAM["BIND_CODES"] = ".cache/bindbus_codes512r.npz"


def _set_env():
    for k, v in _FAM.items():
        os.environ.setdefault(k, v)
    assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only by brief"


def main():
    _set_env()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                        LLAMA_3_2_1B_CFG, _rms_norm)
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, build_gold, sent_indices,
                                      build_slot_masks, alt2_fact_buf, K_VARS, L_FAC, T_ALG, H_TRUNK)
    import dn_common as DC

    # ---- step 1: the 128-row slice (deterministic prefix) ----
    all_rows = [json.loads(l) for l in open(VALID2_FULL)]
    rows = all_rows[:N_ROWS]
    n = len(rows)
    with open(SLICE_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[pages-collect] wrote {SLICE_JSONL} (n={n}/{len(all_rows)} of {VALID2_FULL})", flush=True)

    # ---- tokenize (pure CPU, no model) ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    ids = np.zeros((n, T_ALG), np.int32)
    mask = np.zeros((n, T_ALG), np.float32)
    offsets = []
    for i, s in enumerate(rows):
        e = tok.encode(s["text"])
        if len(e.ids) > T_ALG:
            raise RuntimeError(f"TRUNCATION {SLICE_JSONL}:{i} -- {len(e.ids)} > {T_ALG}")
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1.0
        offsets.append(list(e.offsets))

    # ---- gold, straight from `factors` (custody-gold law) ----
    gold = build_gold(rows, offsets)
    sent = np.stack([sent_indices(rows[i]["text"], offsets[i], mask[i]) for i in range(n)])
    print(f"[pages-collect] tokenized + gold built, n={n}", flush=True)

    # ---- step 2: fresh frozen-trunk CPU forward (do_precompute()'s method, inlined; own output
    # paths only, never STATES_NPY/STATES_NPZ) ----
    class _Host:
        pass
    host = _Host()
    sd = load_llama_weights(os.path.join(H._ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    states = np.zeros((n, T_ALG, H_TRUNK), np.float16)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        c = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(c).all()
        states[sl] = c.astype(np.float16)
        print(f"[pages-collect] trunk {min(s0+8,n)}/{n}", flush=True)
    del host

    # ---- step 3: PMS8_241's own two-pass cycle, CPU, ALG_MINE_BREATHS tap ----
    p = build_params(0)
    sdp = safe_load(CKPT)
    assert set(sdp.keys()) == set(p.keys()), (sorted(set(sdp) - set(p))[:4], sorted(set(p) - set(sdp))[:4])
    for k in p:
        p[k].assign(sdp[k].to(p[k].device).cast(p[k].dtype)).realize()
    K_B = int(os.environ.get("ALG_BREATH", "7"))

    argmax_breath = np.full((n, K_B, L_FAC), -1, np.int32)
    fat_mass = np.zeros((n, K_B, L_FAC, T_ALG), np.float16)
    ok_rows, ok_slots, ok_vals = [], [], []

    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(states[sl_p]), dtype=dtypes.half)
        tk = Tensor(mask[sl_p].astype(np.float32), dtype=dtypes.float)
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
        assert len(fat_all) == K_B, (len(fat_all), K_B)
        tkm = mask[sl_p].astype(np.float32) > 0.5
        neg = np.where(tkm, 0.0, -1e9).astype(np.float32)
        for kb in range(K_B):
            fa = fat_all[kb] + neg[:, None, :]
            argmax_breath[sl, kb, :] = fa[:len(sl)].argmax(-1)
            fat_mass[sl, kb, :, :] = fat_all[kb][:len(sl)].astype(np.float16)

        pass2_keys = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o else ())
        pass2_np = {k: o[k].realize().numpy() for k in pass2_keys}
        for b, i in enumerate(sl):   # only the real (non-pad) rows of this batch
            headrec = {k: pass2_np[k][b] for k in pass2_keys}
            cells = DC.score_row_slots(headrec, gold, int(i), l_fac=L_FAC)
            for j, c in cells.items():
                ok_rows.append(int(i)); ok_slots.append(int(j)); ok_vals.append(bool(c["ok"]))

        print(f"[pages-collect] membrane {min(s0+8,n)}/{n}", flush=True)

    np.savez(RAW_OUT, argmax_breath=argmax_breath, fat_mass=fat_mass,
             ids=ids, tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
             g_presence=gold["presence"], g_ftype=gold["ftype"], g_digits=gold["digits"])
    print(f"[pages-collect] wrote {RAW_OUT}", flush=True)

    np.savez(OK_OUT, rows=np.array(ok_rows, np.int64), slots=np.array(ok_slots, np.int64),
             ok=np.array(ok_vals, bool))
    n_given = int(sum(1 for r, j, in zip(ok_rows, ok_slots) if int(gold["ftype"][r, j]) == 1))
    print(f"[pages-collect] wrote {OK_OUT} (n_slots={len(ok_rows)}, n_given={n_given}, "
          f"n_right={sum(ok_vals)}, n_wrong={len(ok_vals)-sum(ok_vals)})", flush=True)


if __name__ == "__main__":
    main()
