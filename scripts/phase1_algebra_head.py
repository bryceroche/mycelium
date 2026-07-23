"""phase1_algebra_head.py — the ALGEBRA delta head: the §11 layout, built.

THE OBJECT: parse algebra-in-words into arith3 graphs — the first head whose output
feeds a domain where the symbolic jaw has GENUINE deciding to do (per-sample band
labels 0-3 decisions). Layout per the registered §11 frame:

  TWO SLOT BANKS over the shared waist (frozen Llama L0-L3 trunk, T=256):
    VARIABLE slots (24): identity = positional (slot i <-> letter i); text anchoring
      supervised by the generator's MENTION spans (name->slot binding AS STRUCTURE,
      the §6 law applied prospectively — the text-NACK lesson).
    FACTOR slots (24): span-supervised like the KenKen head.
  POINTERS are BILINEAR (factor-state x variable-state): args = 2-hot BCE over var
  slots; result = CE over var slots. Directly supervised from gold — the attention-
  bootstrap law's sanctioned escape, third deployment.
  RESULT = UNION TYPE: is-literal mode bit; rel factors -> result POINTER; given
  factors -> var pointer + value DIGITS (3 x 10-way, MSD-first, transplanted).
  QUERY POINTER: one global head over variable slots (solving is not answering).

EVAL IS PER-BAND from run one (registered): factor-exact / graph-solve / ANSWER
rate logged per decisions-band — the parse-vs-solve factorization question answers
itself. ANSWER rate = the honest end metric: decode -> problem_from_algebra ->
solve_symbolic -> solution[predicted query] == gold answer.

USAGE:
  Selftest:    .venv/bin/python3 scripts/phase1_algebra_head.py --selftest
  Precompute:  DEV=AMD .venv/bin/python3 scripts/phase1_algebra_head.py --precompute
  Train:       DEV=AMD STEPS=8000 .venv/bin/python3 scripts/phase1_algebra_head.py --train
  Eval:        DEV=AMD .venv/bin/python3 scripts/phase1_algebra_head.py --eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

T_ALG = 256
H_TRUNK = 2048
H_W = int(os.environ.get("ALG_HW", "512"))  # capacity-probe dial (2026-07-11)
K_VARS = 24
L_FAC = 24
N_DIG = 3
N_HEADS = 8
SENT_MAX = 32

ALG_TRAIN = os.environ.get("ALG_TRAIN", ".cache/algebra_nl_train.jsonl")
ALG_TEST = os.environ.get("ALG_TEST", ".cache/algebra_nl_test.jsonl")
TEST_NAME = os.environ.get("ALG_TEST_NAME", "test")   # states-file key for the test slice
TRAIN_NAME = os.environ.get("ALG_TRAIN_NAME", "train")  # states-file key for the train slice
STATES_NPZ = ".cache/phase1_alg_states_{split}.npz"
STATES_NPY = ".cache/phase1_alg_states_{split}_states.npy"  # memmap sibling (gen-7+)
ALG_CKPT = os.environ.get("ALG_CKPT", ".cache/phase1_algebra_head.safetensors")
TOKENIZER_JSON = ".cache/llama-3.2-1b-weights/tokenizer.json"


# ===========================================================================
# GOLD (CPU): jsonl + offsets -> slot tensors per the §11 layout
# ===========================================================================

def _spans_to_tokmask(spans, offs, out):
    for (cs, ce) in spans:
        for ti, (ts, te) in enumerate(offs):
            if ti >= T_ALG:
                break
            if ts < ce and te > cs:
                out[ti] = 1.0


def build_gold(samples, offsets):
    n = len(samples)
    g = {
        "presence": np.zeros((n, L_FAC), np.float32),
        "ftype": np.zeros((n, L_FAC), np.int32),         # 0=rel 1=given 2=mod 3=sel
        "op": np.zeros((n, L_FAC), np.int32),            # 0=add 1=mul
        "args": np.zeros((n, L_FAC, K_VARS), np.float32),
        "res": np.zeros((n, L_FAC), np.int32),
        "is_lit": np.zeros((n, L_FAC), np.float32),
        "digits": np.zeros((n, L_FAC, N_DIG), np.int32),
        "fspan": np.zeros((n, L_FAC, T_ALG), np.float32),
        "vspan": np.zeros((n, K_VARS, T_ALG), np.float32),
        "query": np.zeros((n,), np.int32),
        "band": np.zeros((n,), np.int32),
        # the tranche (2026-07-09): explicit per-kind masks (rel mask was
        # (1-is_lit) — wrong once mod/sel exist) + selector-type gold
        "sel": np.zeros((n, L_FAC), np.int32),           # SEL_TO_ID
        "is_rel": np.zeros((n, L_FAC), np.float32),
        "is_mod": np.zeros((n, L_FAC), np.float32),
        "is_sel": np.zeros((n, L_FAC), np.float32),
        # tranche 2 (2026-07-10): pct=4, fdiv=5 — the mod head-shape
        "is_pct": np.zeros((n, L_FAC), np.float32),
        "is_fdiv": np.zeros((n, L_FAC), np.float32),
        # gen-9 (2026-07-12): arg multiplicity — args=[a,a] was
        # UNREPRESENTABLE (multi-hot gold + top-2-distinct decode); the
        # [85] fix. 1.0 on rel factors whose two args are the same var.
        "arg_dup": np.zeros((n, L_FAC), np.float32),
        # gen-15 (2026-07-20): OP_APPLY macro floor (ALG_FTYPES=7) — second
        # digit bank (k2) + ordered second operand pointer (y). Structural
        # entry per the pointer law; gold-fed from birth (two-terminal law).
        "digits2": np.zeros((n, L_FAC, N_DIG), np.int32),
        "is_macro": np.zeros((n, L_FAC), np.float32),
        "is_frac": np.zeros((n, L_FAC), np.float32),   # mg2: FRAC_OF (ftype 7)
        "y": np.zeros((n, L_FAC), np.int32),
    }
    for i, (smp, offs) in enumerate(zip(samples, offsets)):
        # gen-10 prose rows: factors may carry NO spans (raw prose has no
        # letter anchors) — sort them last; span losses auto-mask on zeros
        facs = sorted(smp["factors"],
                      key=lambda f: (min(s for s, _ in f["spans"])
                                     if f.get("spans") else 10 ** 9))
        assert len(facs) <= L_FAC and smp["n_vars"] <= K_VARS
        g["query"][i] = smp["query_var"]
        g["band"][i] = smp["decisions"]
        for v_str, spans in smp["mentions"].items():
            _spans_to_tokmask(spans, offs, g["vspan"][i, int(v_str)])
        for j, f in enumerate(facs):
            g["presence"][i, j] = 1.0
            _spans_to_tokmask(f.get("spans") or [], offs, g["fspan"][i, j])
            if f["ftype"] == "rel":
                g["ftype"][i, j] = 0
                g["is_rel"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
                if f["args"][0] == f["args"][1]:
                    g["arg_dup"][i, j] = 1.0
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "given":
                g["ftype"][i, j] = 1
                g["is_lit"][i, j] = 1.0
                g["res"][i, j] = f["var"]
                t = int(f["value"])
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "mod":
                g["ftype"][i, j] = 2
                g["is_mod"][i, j] = 1.0
                g["args"][i, j, f["var"]] = 1.0
                g["res"][i, j] = f["result"]
                t = int(f["k"])                # modulus via the digit head
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "sel":
                from mycelium.csp_domains import SEL_TO_ID
                g["ftype"][i, j] = 3
                g["is_sel"][i, j] = 1.0
                g["sel"][i, j] = SEL_TO_ID[f["sel"]]
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "pct":
                g["ftype"][i, j] = 4
                g["is_pct"][i, j] = 1.0
                g["args"][i, j, f["args"][0]] = 1.0
                g["res"][i, j] = f["args"][1]
                t = int(f["p"])
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "fdiv":
                g["ftype"][i, j] = 5
                g["is_fdiv"][i, j] = 1.0
                g["args"][i, j, f["var"]] = 1.0
                g["res"][i, j] = f["result"]
                t = int(f["k"])
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "macro" and f.get("name") == "FRAC_OF":
                g["ftype"][i, j] = 7
                g["is_frac"][i, j] = 1.0
                g["args"][i, j, f["x"]] = 1.0
                g["res"][i, j] = f["result"]
                for t_, arr in ((int(f["a"]), "digits"), (int(f["k"]), "digits2")):
                    assert t_ < 10 ** N_DIG
                    for d in range(N_DIG):
                        g[arr][i, j, d] = (t_ // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "macro":
                assert f.get("name") == "OP_APPLY"
                g["ftype"][i, j] = 6
                g["is_macro"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1   # add/sub on macro slots
                g["args"][i, j, f["x"]] = 1.0                  # x via args argmax
                g["y"][i, j] = f["y"]                          # y via the fresh pointer
                g["res"][i, j] = f["result"]
                for t_, key, arr in ((int(f["k1"]), "k1", "digits"),
                                     (int(f["k2"]), "k2", "digits2")):
                    assert t_ < 10 ** N_DIG
                    for d in range(N_DIG):
                        g[arr][i, j, d] = (t_ // 10 ** (N_DIG - 1 - d)) % 10
            else:
                raise ValueError(f"unknown gold ftype {f['ftype']!r}")
    return g


def tokenize(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples = [json.loads(l) for l in open(path)]
    ids = np.zeros((len(samples), T_ALG), np.int32)
    mask = np.zeros((len(samples), T_ALG), np.float32)
    offsets = []
    for i, s in enumerate(samples):
        e = tok.encode(s["text"])
        if len(e.ids) > T_ALG:
            raise RuntimeError(f"TRUNCATION {path}:{i} — {len(e.ids)} > {T_ALG}")
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1.0
        offsets.append(list(e.offsets))
    return samples, ids, mask, offsets


def sent_indices(text, offs, mask_row):
    bounds = []
    i = text.find(". ")
    while i != -1:
        bounds.append(i + 1)
        i = text.find(". ", i + 1)
    out = np.zeros((T_ALG,), np.int32)
    ntk = int(mask_row.sum())
    arr = np.asarray(offs[:min(ntk, T_ALG)], dtype=np.int64)
    if len(arr):
        idx = np.searchsorted(np.asarray(bounds, dtype=np.int64), arr[:, 0], "right")
        out[:len(arr)] = np.minimum(idx, SENT_MAX - 1)
    return out


def build_slot_masks(o_np, sent_rows):
    """Evidence-sharing slot mask (B, L, L) from the model's OWN breath-0
    outputs (deployable): same-sentence (attention-argmax sentence) OR
    shared-variable (top-2 args + res argmax overlap) OR self."""
    B = o_np["fat"].shape[0]
    masks = np.zeros((B, L_FAC, L_FAC), np.float32)
    for bi in range(B):
        tok_star = o_np["fat"][bi].argmax(-1)              # (L,)
        s_id = sent_rows[bi][np.minimum(tok_star, T_ALG - 1)]
        same = s_id[:, None] == s_id[None, :]
        M = np.zeros((L_FAC, K_VARS), bool)
        for j in range(L_FAC):
            for a in np.argsort(-o_np["args"][bi, j])[:2]:
                M[j, a] = True
            M[j, int(o_np["res"][bi, j].argmax())] = True
        shared = (M.astype(np.int32) @ M.astype(np.int32).T) > 0
        masks[bi] = (same | shared | np.eye(L_FAC, dtype=bool)).astype(np.float32)
    return masks


# ===========================================================================
# PRECOMPUTE (GPU once) — small corpus, plain npz
# ===========================================================================

def do_precompute():
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm)

    class _H:
        pass
    host = _H()
    sd = load_llama_weights(os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    jobs = [(TRAIN_NAME, ALG_TRAIN), (TEST_NAME, ALG_TEST)]
    if os.environ.get("PRECOMPUTE_ONLY"):
        jobs = [(n_, p_) for n_, p_ in jobs if n_ == os.environ["PRECOMPUTE_ONLY"]]
    for split, path in jobs:
        samples, ids, mask, offsets = tokenize(path)
        n = len(samples)
        # states stream straight to a disk-backed memmap — holding the full
        # (n, T, 2048) fp16 array in RAM beside the AM driver's pinned pages
        # OOMs at gen-7 scale (three kills, all during the giant-array write)
        states = np.lib.format.open_memmap(
            STATES_NPY.format(split=split), mode="w+", dtype=np.float16,
            shape=(n, T_ALG, H_TRUNK))
        for s0 in range(0, n, 8):
            sl = slice(s0, min(s0 + 8, n))
            x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
            for layer in host.llama_layers:
                x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
            c = x.cast(dtypes.float).realize().numpy()
            assert np.isfinite(c).all()
            states[sl] = c.astype(np.float16)
            if (s0 // 8) % 200 == 0:
                print(f"[precompute] {split}: batch {s0//8}/{(n+7)//8}", flush=True)
        states.flush()
        shape = states.shape
        del states
        gold = build_gold(samples, offsets)
        sent = np.stack([sent_indices(s["text"], o, mask[i])
                         for i, (s, o) in enumerate(zip(samples, offsets))])
        np.savez(STATES_NPZ.format(split=split),
                 tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
                 **{f"g_{k}": v for k, v in gold.items()})
        print(f"[precompute] {split}: {shape} (states -> memmap npy)", flush=True)


def load_alg(split):
    is_train = split == "train"
    split = TRAIN_NAME if is_train else (TEST_NAME if split == "test" else split)
    z = np.load(STATES_NPZ.format(split=split))
    samples = [json.loads(l) for l in open(ALG_TRAIN if is_train else ALG_TEST)]
    gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    if os.path.exists(STATES_NPY.format(split=split)):
        states = np.load(STATES_NPY.format(split=split), mmap_mode="r")
    else:
        states = z["states"]   # legacy artifacts (gen<=6): states inside npz
    return samples, states, z["tokmask"], gold, z["sent"]


# ===========================================================================
# MODEL — two slot banks + bilinear pointers (all supervised)
# ===========================================================================

def build_params(seed=0):
    from tinygrad import Tensor, dtypes
    rng = np.random.RandomState(seed)

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x

    def lin(i, o):
        return t(rng.randn(i, o) / math.sqrt(i)), t(np.zeros((o,)))

    p = {}
    p["waist_w"], p["waist_b"] = lin(H_TRUNK, H_W)
    p["sent_emb"] = t(rng.randn(SENT_MAX, H_W) * 0.1)
    p["vq"] = t(rng.randn(K_VARS, H_W) * 0.02)
    p["fq"] = t(rng.randn(L_FAC, H_W) * 0.02)
    p["qq"] = t(rng.randn(1, H_W) * 0.02)
    for nm in ("wq", "wk", "wv", "wo"):
        p[f"attn_{nm}"], p[f"attn_{nm}_b"] = lin(H_W, H_W)
    p["ffn_w1"], p["ffn_b1"] = lin(H_W, 2 * H_W)
    p["ffn_w2"], p["ffn_b2"] = lin(2 * H_W, H_W)
    p["h_pres"], p["h_pres_b"] = lin(H_W, 1)
    # ALG2=1 -> tranche geometry (4-way ftype + selector head). Default keeps
    # the legacy 2-way build BYTE-COMPATIBLE with deployed checkpoints — every
    # lattice script loads the old ckpt through here.
    if int(os.environ.get("ALG2", "0")):
        nft = int(os.environ.get("ALG_FTYPES", "4"))  # 6 = +pct/fdiv (T2)
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, nft)
        p["h_sel"], p["h_sel_b"] = lin(H_W, 4)       # larger/smaller/even/odd
    else:
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, 2)
    p["h_op"], p["h_op_b"] = lin(H_W, 2)
    if int(os.environ.get("ALG_DUP", "0")):   # gen-9: arg-multiplicity bit
        p["h_dup"], p["h_dup_b"] = lin(H_W, 1)
    p["h_islit"], p["h_islit_b"] = lin(H_W, 1)
    p["h_dig"], p["h_dig_b"] = lin(H_W, N_DIG * 10)
    if int(os.environ.get("ALG2", "0")) and \
            int(os.environ.get("ALG_FTYPES", "4")) >= 7:   # gen-15: OP_APPLY
        p["h_dig2"], p["h_dig2_b"] = lin(H_W, N_DIG * 10)
        p["W_y"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    if K_B > 1:   # BRICK-P: masked slot-to-slot breathing (2026-07-09)
        p["W_bq"], p["W_bq_b"] = lin(H_W, H_W)
        p["W_bk"], p["W_bk_b"] = lin(H_W, H_W)
        p["W_bv"], p["W_bv_b"] = lin(H_W, H_W)
        p["W_bo"] = t(np.zeros((H_W, H_W)))          # zero-init: breath deltas
        p["W_bo_b"] = t(np.zeros(H_W))               # start silent
        p["breath_emb"] = t(rng.randn(K_B, H_W) * 0.02)
        p["breath_gate"] = t(np.full(K_B, -2.0))     # init-closed convex blend
    p["W_args"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_res"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_query"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    return p


def forward(p, trunk, tokmask, sent, slot_mask=None):
    B = trunk.shape[0]
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]

    def bank(queries, nq, extra=None):
        q_in = queries.unsqueeze(0) + (extra if extra is not None else 0)
        q = q_in @ p["attn_wq"] + p["attn_wq_b"]
        k = waist @ p["attn_wk"] + p["attn_wk_b"]
        v = waist @ p["attn_wv"] + p["attn_wv_b"]
        hd = H_W // N_HEADS
        qh = q.reshape(B if extra is not None else 1, nq, N_HEADS, hd).permute(0, 2, 1, 3)
        kh = k.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        sc = (qh @ kh.transpose(-2, -1)) / math.sqrt(hd)
        sc = sc.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
        at = sc.softmax(-1)
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)
        st = st @ p["attn_wo"] + p["attn_wo_b"] + q_in.reshape(-1, nq, H_W)
        st = st + ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])
        return st, at.mean(1)

    vst, vat = bank(p["vq"], K_VARS)
    fst, fat = bank(p["fq"], L_FAC)
    qst, _qa = bank(p["qq"], 1)

    # BRICK-P breathing (2026-07-09): K-1 refinement passes. Each breath
    # re-reads the text conditioned on current beliefs (bank-with-extra) +
    # MASKED slot-to-slot settling — evidence-sharing topology AS STRUCTURE
    # (the v98 escape; free-form slot attention is the perceiver trap and is
    # not built). Deltas enter via zero-init W_bo + init-closed gates:
    # at init the K-breath output is byte-identical to the incumbent.
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    breaths = [fst]
    if K_B > 1 and slot_mask is not None and "W_bo" in p:
        cur = fst
        for kb in range(1, K_B):
            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)
            h_tok, _ = bank(p["fq"], L_FAC, extra=q_extra)
            bq = cur @ p["W_bq"] + p["W_bq_b"]
            bk = cur @ p["W_bk"] + p["W_bk_b"]
            bv = cur @ p["W_bv"] + p["W_bv_b"]
            sc2 = (bq @ bk.transpose(-2, -1)) / math.sqrt(H_W)
            sc2 = sc2.clip(-1e4, 1e4) + (1.0 - slot_mask) * -1e4
            h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]
            # ABLATION arms (2026-07-10): zero-mult keeps every param in the
            # graph (defined zero grads — the None-grad lesson, applied)
            arm = os.environ.get("ALG_BREATH_ARM", "both")
            if arm == "tok":
                h_slot = h_slot * 0.0
            elif arm == "slot":
                h_tok = h_tok * 0.0
            elif arm == "depth":
                # the decider control: plain per-slot MLP second pass — same
                # params repurposed, NO attention, no mask, no re-read
                h_tok = h_tok * 0.0
                h_slot = h_slot * 0.0 + ((cur @ p["W_bq"] + p["W_bq_b"])
                                         .gelu() @ p["W_bv"] + p["W_bv_b"]) \
                    @ p["W_bo"] + p["W_bo_b"]
            g = p["breath_gate"][kb].sigmoid()
            cur = cur + g * (h_tok + h_slot - cur)
            breaths.append(cur)

    def heads_of(s):
        return {
            "pres": (s @ p["h_pres"] + p["h_pres_b"]).squeeze(-1),
            "ftype": s @ p["h_ftype"] + p["h_ftype_b"],
            "op": s @ p["h_op"] + p["h_op_b"],
            **({"sel": s @ p["h_sel"] + p["h_sel_b"]} if "h_sel" in p else {}),
            **({"dup": (s @ p["h_dup"] + p["h_dup_b"]).squeeze(-1)}
               if "h_dup" in p else {}),
            "islit": (s @ p["h_islit"] + p["h_islit_b"]).squeeze(-1),
            "dig": (s @ p["h_dig"] + p["h_dig_b"]).reshape(B, L_FAC, N_DIG, 10),
            "args": (s @ p["W_args"]) @ vst.transpose(-2, -1),
            "res": (s @ p["W_res"]) @ vst.transpose(-2, -1),
            **({"dig2": (s @ p["h_dig2"] + p["h_dig2_b"])
                .reshape(B, L_FAC, N_DIG, 10),
                "y": (s @ p["W_y"]) @ vst.transpose(-2, -1)}
               if "h_dig2" in p else {}),
        }
    out = heads_of(breaths[-1])
    out["query"] = ((qst @ p["W_query"]) @ vst.transpose(-2, -1)).reshape(B, K_VARS)
    out["fat"], out["vat"] = fat, vat
    if len(breaths) > 1:
        out["breaths"] = [heads_of(s) for s in breaths]
    return out


def loss_fn(o, g):
    """Ladder wrapper: with breaths, per-breath weighted CE (1 + k/(K-1)) —
    the v98 ladder; the shared query/span terms ride the final breath only."""
    if "breaths" in o:
        K_B = len(o["breaths"])
        tot = None
        for kb, ob in enumerate(o["breaths"]):
            full = dict(o, **ob)
            w = 1.0 + kb / max(K_B - 1, 1)
            term = _loss_single(full, g) * w
            tot = term if tot is None else tot + term
        return tot / K_B
    return _loss_single(o, g)


def _loss_single(o, g):
    from tinygrad import Tensor
    pres = g["presence"]
    n_p = pres.sum() + 1e-6
    # explicit per-kind masks (tranche); legacy callers without them fall back
    # to the old two-kind arithmetic (byte-identical behavior)
    is_rel = g["is_rel"] if "is_rel" in g else (1.0 - g["is_lit_f"])
    is_mod = g["is_mod"] if "is_mod" in g else g["is_lit_f"] * 0.0
    is_sel = g["is_sel"] if "is_sel" in g else g["is_lit_f"] * 0.0
    is_pct = g["is_pct"] if "is_pct" in g else g["is_lit_f"] * 0.0
    is_fdiv = g["is_fdiv"] if "is_fdiv" in g else g["is_lit_f"] * 0.0
    rel = pres * is_rel
    n_rel = rel.sum() + 1e-6

    def bce(lg, tg):
        return lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()

    def ce(lg, tg):
        return (lg.log_softmax(-1) * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)

    l = bce(o["pres"], pres).mean()
    l = l + (ce(o["ftype"], g["ftype"]) * pres).sum() / n_p
    l = l + (ce(o["op"], g["op"]) * rel).sum() / n_rel
    l = l + bce(o["islit"], g["is_lit_f"]).mean()
    is_macro = g["is_macro"] if "is_macro" in g else is_mod * 0.0
    is_frac = g["is_frac"] if "is_frac" in g else is_mod * 0.0
    dm = g["is_lit_f"] + is_mod + is_pct + is_fdiv + is_macro + is_frac
    l = l + (ce(o["dig"], g["digits"]).mean(-1) * dm).sum() / (dm.sum() + 1e-6)
    if "sel" in o and "sel" in g:               # selector-type CE (closed vocab)
        sm = pres * is_sel
        l = l + (ce(o["sel"], g["sel"]) * sm).sum() / (sm.sum() + 1e-6)
    if "dup" in o and "arg_dup" in g:           # gen-9: arg-multiplicity BCE
        l = l + (bce(o["dup"], g["arg_dup"]) * rel).sum() / n_rel
    if "dig2" in o and "is_macro" in g:        # gen-15: OP_APPLY terms
        if "is_frac" in g:                     # mg2: frac k rides the dig2 CE
            mac2 = pres * (g["is_macro"] + g["is_frac"])
            l = l + (ce(o["dig2"], g["digits2"]) .mean(-1) * mac2).sum() / (mac2.sum() + 1e-6)
        mac = pres * g["is_macro"]
        n_mac = mac.sum() + 1e-6
        l = l + (ce(o["op"], g["op"]) * mac).sum() / n_mac
        l = l + (ce(o["dig2"], g["digits2"]).mean(-1) * mac).sum() / n_mac
        l = l + (ce(o["y"], g["y"]) * mac).sum() / n_mac * 2.0
    args_w = 1.0 + 4.0 * g["args"]
    am = pres * (is_rel + is_sel + is_mod + is_pct + is_fdiv + is_macro + is_frac)
    n_am = am.sum() + 1e-6
    l = l + ((bce(o["args"], g["args"]) * args_w).mean(-1) * am).sum() / n_am * 2.0
    l = l + (ce(o["res"], g["res"]) * pres).sum() / n_p * 2.0
    l = l + ce(o["query"], g["query"]).mean() * 2.0
    fsn = g["fspan"] / (g["fspan"].sum(-1, keepdim=True) + 1e-6)
    l = l + ((-(o["fat"] + 1e-9).log() * fsn).sum(-1) * pres).sum() / n_p
    vsn = g["vspan"] / (g["vspan"].sum(-1, keepdim=True) + 1e-6)
    vmask = (g["vspan"].sum(-1) > 0).float()
    l = l + ((-(o["vat"] + 1e-9).log() * vsn).sum(-1) * vmask).sum() / (vmask.sum() + 1e-6)
    return l


# ===========================================================================
# DECODE + PER-BAND EVAL (factor-exact / graph-solve / ANSWER, per band)
# ===========================================================================

def decode(o_np):
    from mycelium.csp_domains import ID_TO_SEL
    facs = []
    for j in range(L_FAC):
        if o_np["pres"][j] <= 0:
            continue
        res = int(o_np["res"][j].argmax())
        ft = int(o_np["ftype"][j].argmax()) if o_np["ftype"].shape[-1] >= 4 \
            else (1 if o_np["islit"][j] > 0 else 0)

        def digval():
            digs = o_np["dig"][j].argmax(-1)
            return int(sum(d * 10 ** (N_DIG - 1 - i)
                           for i, d in enumerate(digs)))
        if ft == 1:
            facs.append({"ftype": "given", "var": res, "value": digval()})
        elif ft == 0:
            op = "add" if o_np["op"][j].argmax() == 0 else "mul"
            if "dup" in o_np and o_np["dup"][j] > 0:
                a0 = int(np.argmax(o_np["args"][j]))   # gen-9: args=[a,a]
                args = [a0, a0]
            else:
                args = sorted(int(a) for a in
                              np.argsort(-o_np["args"][j])[:2])
            facs.append({"ftype": "rel", "op": op,
                         "args": args, "result": res})
        elif ft == 2:
            var = int(np.argmax(o_np["args"][j]))
            facs.append({"ftype": "mod", "var": var, "k": max(digval(), 2),
                         "result": res})
        elif ft == 4:
            facs.append({"ftype": "pct",
                         "args": [int(np.argmax(o_np["args"][j])), res],
                         "p": max(digval(), 1)})
        elif ft == 5:
            facs.append({"ftype": "fdiv",
                         "var": int(np.argmax(o_np["args"][j])),
                         "k": max(digval(), 2), "result": res})
        elif ft == 7 and "dig2" in o_np:
            k_ = int(sum(d * 10 ** (N_DIG - 1 - i2) for i2, d in
                         enumerate(o_np["dig2"][j].argmax(-1))))
            facs.append({"ftype": "macro", "name": "FRAC_OF",
                         "a": max(digval(), 1), "k": max(k_, 2),
                         "x": int(np.argmax(o_np["args"][j])), "result": res})
        elif ft == 6 and "dig2" in o_np:
            k2 = int(sum(d * 10 ** (N_DIG - 1 - i2) for i2, d in
                         enumerate(o_np["dig2"][j].argmax(-1))))
            facs.append({"ftype": "macro", "name": "OP_APPLY",
                         "op": "add" if o_np["op"][j].argmax() == 0 else "sub",
                         "k1": max(digval(), 1), "x": int(np.argmax(o_np["args"][j])),
                         "k2": max(k2, 1), "y": int(np.argmax(o_np["y"][j])),
                         "result": res})
        else:
            args = list(np.argsort(-o_np["args"][j])[:2])
            sel = ID_TO_SEL[int(o_np["sel"][j].argmax())]
            facs.append({"ftype": "sel", "sel": sel,
                         "args": sorted(int(a) for a in args), "result": res})
    return facs, int(o_np["query"].argmax())


def do_eval():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    per_band = {}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        t_tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        t_se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        out = forward(p, t_tr, t_tk, t_se)
        if int(os.environ.get("ALG_BREATH", "1")) > 1 and "W_bo" in p:
            o0 = {k: out[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(o0, sent[sl_p])
            out = forward(p, t_tr, t_tk, t_se,
                          slot_mask=Tensor(mk, dtype=dtypes.float))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            band = int(gold["band"][i])
            st = per_band.setdefault(band, {"n": 0, "fac_ok": 0, "fac_tot": 0,
                                            "solve": 0, "answer": 0, "query_ok": 0})
            st["n"] += 1
            facs, q_pred = decode({k: o[k][bi] for k in o})

            def fkey(f):
                if f["ftype"] == "rel":
                    return ("rel", f["op"], tuple(sorted(f["args"])),
                            f["result"])
                if f["ftype"] == "given":
                    return ("given", f["var"], f["value"])
                if f["ftype"] == "mod":
                    return ("mod", f["var"], f["k"], f["result"])
                if f["ftype"] == "pct":
                    return ("pct", tuple(f["args"]), f["p"])
                if f["ftype"] == "fdiv":
                    return ("fdiv", f["var"], f["k"], f["result"])
                return ("sel", f["sel"], tuple(sorted(f["args"])), f["result"])
            gset = set(fkey(f) for f in smp["factors"])
            st["fac_ok"] += len(gset & set(fkey(f) for f in facs))
            st["fac_tot"] += len(gset)
            st["query_ok"] += int(q_pred == smp["query_var"])
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

            def fvars(f):
                if f["ftype"] in ("rel", "sel", "pct"):
                    return list(f["args"]) + ([f["result"]]
                                              if "result" in f else [])
                if f["ftype"] in ("mod", "fdiv"):
                    return [f["var"], f["result"]]
                return [f["var"]]
            try:
                from mycelium.csp_domains import problem_from_algebra3
                nv = max([smp["n_vars"]] + [v + 1 for f in facs
                                            for v in fvars(f)])
                res = solve_symbolic(problem_from_algebra3(nv, facs, gv,
                                                           smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] == "solved":
                    sol = [int(res["assignment"][v]) for v in range(nv)]
                    if sol[:smp["n_vars"]] == smp["solution"]:
                        st["solve"] += 1
                    gold_ans = smp["solution"][smp["query_var"]]
                    if q_pred < len(sol) and sol[q_pred] == gold_ans:
                        st["answer"] += 1
            except Exception:
                pass

    print(f"\n[algebra eval] per-BAND (the registered stratification):")
    print(f"  band |  n | fac-F1ish | query | graph-solve | ANSWER")
    tot = {"n": 0, "solve": 0, "answer": 0}
    for b in sorted(per_band):
        st = per_band[b]
        print(f"    {b:2d} | {st['n']:2d} |   {st['fac_ok']/max(st['fac_tot'],1):.3f}   "
              f"|  {st['query_ok']/max(st['n'],1):.2f} |     {st['solve']:2d}      |   {st['answer']:2d}")
        for k in tot:
            tot[k] += st[k] if k != "n" else st["n"]
    print(f"  TOTAL: {tot['solve']}/{tot['n']} graph-solve, "
          f"{tot['answer']}/{tot['n']} ANSWER")
    print(f"  (factorization read: flat fac-exact across bands = reading and"
          f" reasoning on independent axes)")




# ===========================================================================
# TAXONOMY (--errors): prediction #2-algebra's grader (spec §11)
# ===========================================================================
# Classes: CORRECT | DETECT_malformed | DETECT_unsat | DETECT_multi (ban-and-resolve
# uniqueness probe, gold-free) | SILENT (unique-solves to the WRONG answer). Literal
# errors attributed by the gold given's ROLE: chain_k vs pair_sum/pair_diff — the
# registered ordering: chain-literals >> coupled-literals > structural (~0).

def do_errors():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    cats = {"CORRECT": 0, "DETECT_malformed": 0, "DETECT_unsat": 0,
            "DETECT_multi": 0, "SILENT": 0}
    lit_err = {}     # class -> {role: count} for wrong-literal cases
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            facs, q_pred = decode({k: o[k][bi] for k in o})
            # wrong-literal attribution (pred given value != gold given value, by var)
            gold_gv = {f["var"]: (f["value"], f.get("role", "?"))
                       for f in smp["factors"] if f["ftype"] == "given"}
            wrong_lits = [(v, gold_gv[v][1]) for f in facs if f["ftype"] == "given"
                          for v in [f["var"]] if v in gold_gv
                          and f["value"] != gold_gv[v][0]]
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in facs if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            cat = None
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])])
                if nv > 26:
                    cat = "DETECT_malformed"
                else:
                    res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                         budget=200_000, seed=0)
                    if res["status"] != "solved":
                        cat = "DETECT_unsat"
                    else:
                        sol = [int(res["assignment"][v]) for v in range(nv)]
                        ans_ok = q_pred < len(sol) and sol[q_pred] == gold_ans
                        if ans_ok and not wrong_lits:
                            cat = "CORRECT"
                        else:
                            # uniqueness probe on the PREDICTED graph (gold-free)
                            multi = False
                            for v in range(nv):
                                if v in gv:
                                    continue
                                p2 = problem_from_algebra(nv, rels, gv, smp["m"])
                                p2.domains0[v].discard(sol[v])
                                if p2.domains0[v]:
                                    r2 = solve_symbolic(p2, budget=100_000, seed=0)
                                    if r2["status"] == "solved":
                                        multi = True
                                        break
                            if ans_ok:
                                cat = "CORRECT"   # right answer, benign literal drift
                            elif multi:
                                cat = "DETECT_multi"
                            else:
                                cat = "SILENT"
            except Exception:
                cat = "DETECT_malformed"
            cats[cat] += 1
            if wrong_lits and cat in ("SILENT", "DETECT_unsat", "DETECT_multi"):
                for _v, role in wrong_lits:
                    lit_err.setdefault(cat, {}).setdefault(role, 0)
                    lit_err[cat][role] += 1

    wrong = n - cats["CORRECT"]
    det = cats["DETECT_malformed"] + cats["DETECT_unsat"] + cats["DETECT_multi"]
    print(f"\n[algebra errors] n={n}")
    for k, v in cats.items():
        print(f"  {k:18s} {v}")
    if wrong:
        print(f"  DETECTABLE fraction: {det}/{wrong} = {det/wrong:.2f} "
              f"(KenKen was 1.00 seven times — the INVERSION is the prediction)")
    print(f"  wrong-literal attribution by role x class: {lit_err}")
    print(f"  (registered ordering: chain_k literals >> pair literals (parity ~half-"
          f"caught) > structural ~0)")


# ===========================================================================
# TRAIN
# ===========================================================================

def do_train(steps, lr, batch, seed):
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save

    samples, states, tokmask, gold, sent = load_alg("train")
    n = states.shape[0]
    p = build_params(seed)
    if int(os.environ.get("RESUME", "0")) and os.path.exists(ALG_CKPT):
        from tinygrad.nn.state import safe_load as _sl
        sd0 = _sl(ALG_CKPT)
        assert set(sd0.keys()) == set(p.keys()), "resume key mismatch (hard error)"
        for k in p:
            p[k].assign(sd0[k].to(p[k].device).cast(p[k].dtype)).realize()
        print(f"[train] RESUMED from {ALG_CKPT}", flush=True)
    elif os.environ.get("WARM_FROM"):
        # tranche warm-start: load every shape-matching key from a LEGACY ckpt;
        # skips are EXPLICIT AND PRINTED (warm-start may skip loudly — eval
        # loads must hard-error; this is the training-side allowance)
        from tinygrad.nn.state import safe_load as _sl
        sd0 = _sl(os.environ["WARM_FROM"])
        n_load = 0
        for k in p:
            if k in sd0 and tuple(sd0[k].shape) == tuple(p[k].shape):
                p[k].assign(sd0[k].to(p[k].device).cast(p[k].dtype)).realize()
                n_load += 1
            elif k in sd0 and len(sd0[k].shape) == len(p[k].shape) and all(
                    o <= n_ for o, n_ in zip(sd0[k].shape, p[k].shape)):
                # PAD-WARM (2026-07-10, the ftype-router lesson): old shape is
                # a prefix of new — copy the trained slice, keep fresh init on
                # the new rows. Discarding a trained ROUTER forces relearning
                # inside a converged circuit (the bootstrap-trap family).
                import numpy as _np
                cur = p[k].detach().numpy()
                old = sd0[k].to(p[k].device).cast(p[k].dtype).numpy()
                sl = tuple(slice(0, o) for o in old.shape)
                cur[sl] = old
                p[k].assign(Tensor(cur, dtype=p[k].dtype)).realize()
                n_load += 1
                print(f"[warm] PAD-WARM {k} {tuple(old.shape)} -> "
                      f"{tuple(cur.shape)}", flush=True)
            else:
                print(f"[warm] SKIP {k} (fresh init: "
                      f"{'missing' if k not in sd0 else 'shape'})", flush=True)
        print(f"[train] WARM from {os.environ['WARM_FROM']}: "
              f"{n_load}/{len(p)} keys", flush=True)
    opt = AdamW(list(p.values()), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    K_B = int(os.environ.get("ALG_BREATH", "1"))
    MASKS = None
    if K_B > 1:
        # mask-prep pass: masks from the WARM-STARTED head's own breath-0
        # parses (deployable-from-birth; frozen for training efficiency)
        print("[breath] mask-prep pass ...", flush=True)
        MASKS = np.zeros((n, L_FAC, L_FAC), np.float32)
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out0 = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]
        print(f"[breath] masks ready (mean degree "
              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float32), dtypes.float)
    b_tk = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float)
    b_se = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int)
    b_mask = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float) \
        if K_B > 1 else None
    bg = {}
    for k, shape, dt in (("presence", (L_FAC,), dtypes.float),
                         ("is_lit_f", (L_FAC,), dtypes.float),
                         ("args", (L_FAC, K_VARS), dtypes.float),
                         ("fspan", (L_FAC, T_ALG), dtypes.float),
                         ("vspan", (K_VARS, T_ALG), dtypes.float),
                         ("ftype", (L_FAC,), dtypes.int),
                         ("op", (L_FAC,), dtypes.int),
                         ("res", (L_FAC,), dtypes.int),
                         ("digits", (L_FAC, N_DIG), dtypes.int),
                         ("sel", (L_FAC,), dtypes.int),
                         ("is_rel", (L_FAC,), dtypes.float),
                         ("is_mod", (L_FAC,), dtypes.float),
                         ("is_sel", (L_FAC,), dtypes.float),
                         ("is_pct", (L_FAC,), dtypes.float),
                         ("is_fdiv", (L_FAC,), dtypes.float),
                         ("arg_dup", (L_FAC,), dtypes.float),
                         # gen-15: OP_APPLY gold buffers (two-terminal law —
                         # without these, h_dig2/W_y leave the graph: None grads)
                         *((("is_macro", (L_FAC,), dtypes.float),
                            ("digits2", (L_FAC, N_DIG), dtypes.int),
                            ("y", (L_FAC,), dtypes.int))
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 7 else ()),
                         *((("is_frac", (L_FAC,), dtypes.float),)
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 8 else ()),
                         ("query", (), dtypes.int)):
        npdt = np.float32 if dt == dtypes.float else np.int32
        bg[k] = fix(np.zeros((batch,) + shape, npdt), dt)

    @TinyJit
    def step():
        Tensor.training = True
        o = forward(p, b_tr, b_tk, b_se, slot_mask=b_mask)
        l = loss_fn(o, bg)
        opt.zero_grad()
        l.backward()
        opt.step()
        return l.realize()

    # HYGIENE (the stack-at-convergence protocol): cosine LR decay + periodic
    # validation on the SMALL test slice (bigtest stays untouched as measurement
    # set) + PICK-BEST-BY-VAL. The overnight constant-lr spike taught this.
    lr_min = lr / 30.0
    val_every = int(os.environ.get("VAL_EVERY", "4000"))

    def _quick_val():
        vs, vst, vtk, vg, vse = load_split_val
        n_ok = n_tot = 0
        for s0 in range(0, len(vs), 8):
            sl = np.arange(s0, min(s0 + 8, len(vs)))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            o = forward(p, Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float),
                        Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float),
                        Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int))
            onp = {k: o[k].realize().numpy() for k in
                   (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
            for bi, i in enumerate(sl):
                i = int(i)
                for j in range(L_FAC):
                    if vg["presence"][i, j] < 0.5:
                        continue
                    n_tot += 1
                    ok = (onp["pres"][bi, j] > 0)
                    ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
                    ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
                    if vg["ftype"][i, j] == 0:
                        ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                        gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                        if len(gset) == 1 and "dup" in onp:
                            # gen-9: repeated-arg rel — dup bit + top-1 match
                            ok &= bool(onp["dup"][bi, j] > 0)
                            ok &= int(np.argmax(onp["args"][bi, j])) in gset
                        else:
                            top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                            ok &= top2 == gset
                    else:
                        ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                                    vg["digits"][i, j]).all())
                    n_ok += ok
        return n_ok / max(n_tot, 1)

    load_split_val = load_alg("test")
    best_val, best_snap = -1.0, None

    # CURRICULUM=1 (2026-07-10 ablation): coarse->fine by SAMPLE ORDERING.
    # Teeth score from sample artifacts (post-hoc, deployable-blind): oblique
    # (mention span >2 chars) + shuffled (letter != LETTERS[v]) + irrelevant.
    # Phase 1/3: score==0 only; 2/3: score<=1; 3/3: full mix.
    pools = None
    if int(os.environ.get("CURRICULUM", "0")):
        from characterize_survivors import sample_teeth
        score = np.array([int(t["oblique"]) + int(t["shuffled"])
                          + int(t["irrelevant"])
                          for t in (sample_teeth(s_) for s_ in samples)])
        pools = [np.where(score == 0)[0], np.where(score <= 1)[0],
                 np.arange(n)]
        print(f"[curriculum] pools: easy={len(pools[0])} "
              f"mid={len(pools[1])} full={n}", flush=True)

    t0 = time.time()
    for s in range(steps):
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * s / steps))
        opt.lr.assign(Tensor([cur_lr], dtype=dtypes.float)).realize()
        pool = (pools[min(3 * s // steps, 2)] if pools is not None
                else np.arange(n))
        idx = rng.choice(pool, batch, replace=False)
        b_tr.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        if b_mask is not None:
            b_mask.assign(Tensor(MASKS[idx], dtype=dtypes.float).contiguous()).realize()
        feed = {"presence": gold["presence"][idx], "is_lit_f": gold["is_lit"][idx],
                "args": gold["args"][idx], "fspan": gold["fspan"][idx],
                "vspan": gold["vspan"][idx], "ftype": gold["ftype"][idx],
                "op": gold["op"][idx], "res": gold["res"][idx],
                "digits": gold["digits"][idx], "query": gold["query"][idx],
                "sel": gold["sel"][idx], "is_rel": gold["is_rel"][idx],
                "is_mod": gold["is_mod"][idx], "is_sel": gold["is_sel"][idx],
                "is_pct": gold["is_pct"][idx], "is_fdiv": gold["is_fdiv"][idx],
                "arg_dup": (gold["arg_dup"][idx] if "arg_dup" in gold
                            else np.zeros_like(gold["is_rel"][idx])),
                **({"is_macro": gold["is_macro"][idx],
                    "digits2": gold["digits2"][idx],
                    "y": gold["y"][idx]} if "is_macro" in bg else {}),
                **({"is_frac": gold["is_frac"][idx]} if "is_frac" in bg else {})}
        for k, v in feed.items():
            npdt = np.float32 if bg[k].dtype == dtypes.float else np.int32
            bg[k].assign(Tensor(v.astype(npdt), dtype=bg[k].dtype).contiguous()).realize()
        lv = step()
        if s % 500 == 0 or s == steps - 1:
            v = float(lv.numpy())
            assert np.isfinite(v)
            print(f"  step {s:5d} loss={v:.4f} lr={cur_lr:.1e} "
                  f"({(time.time()-t0)/(s+1):.2f}s/step)", flush=True)
        if (s + 1) % val_every == 0 or s == steps - 1:
            fv = _quick_val()
            mark = ""
            if fv > best_val:
                best_val = fv
                best_snap = {k: t.detach().numpy().copy() for k, t in p.items()}
                mark = "  <-- BEST"
            print(f"  [val @{s+1}] fac-exact-proxy={fv:.4f}{mark}", flush=True)
        # SNAP_EVERY (gut #57's snapshot rider): periodic trajectory ckpts for
        # the wobble census. Snapshots NEVER read bar fixtures (val picks,
        # bars judge); they exist so checkpoint variance can be measured.
        snap_every = int(os.environ.get("SNAP_EVERY", "0"))
        if snap_every and (s + 1) % snap_every == 0:
            sp = ALG_CKPT.replace(".safetensors", f"_s{s+1}.safetensors")
            safe_save(p, sp)
            print(f"  [snap @{s+1}] -> {sp}", flush=True)
    if best_snap is not None:
        for k in p:
            p[k].assign(Tensor(best_snap[k], dtype=p[k].dtype)).realize()
        print(f"[train] restored BEST ckpt (val {best_val:.4f})", flush=True)
    safe_save(p, ALG_CKPT)
    print(f"[train] saved {ALG_CKPT} (best-by-val)", flush=True)


# ===========================================================================
# SELFTEST
# ===========================================================================

def selftest():
    os.environ.setdefault("DEV", "CPU")
    from tinygrad import Tensor, dtypes
    p = build_params(0)
    B = 2
    o = forward(p, Tensor(np.random.RandomState(0).randn(B, T_ALG, H_TRUNK).astype(np.float32) * .1),
                Tensor(np.ones((B, T_ALG), np.float32)),
                Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int))
    assert o["args"].shape == (B, L_FAC, K_VARS) and o["query"].shape == (B, K_VARS)
    g = {"presence": Tensor(np.ones((B, L_FAC), np.float32)),
         "is_lit_f": Tensor(np.zeros((B, L_FAC), np.float32)),
         "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
         "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
         "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
         "ftype": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "digits": Tensor(np.zeros((B, L_FAC, N_DIG), np.int32), dtype=dtypes.int),
         "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int)}
    l = loss_fn(o, g)
    lv = float(l.numpy())
    assert np.isfinite(lv)
    l.backward()
    assert float(p["W_res"].grad.abs().max().numpy()) > 0
    print(f"  [OK] forward/loss/backward (loss={lv:.3f}); pointer grads flow")
    # decode round trip
    onp = {"pres": np.full((L_FAC,), -9., np.float32),
           "ftype": np.zeros((L_FAC, 2), np.float32),
           "op": np.zeros((L_FAC, 2), np.float32),
           "islit": np.full((L_FAC,), -9., np.float32),
           "dig": np.zeros((L_FAC, N_DIG, 10), np.float32),
           "args": np.full((L_FAC, K_VARS), -9., np.float32),
           "res": np.zeros((L_FAC, K_VARS), np.float32),
           "query": np.zeros((K_VARS,), np.float32)}
    onp["pres"][0] = 9.
    onp["args"][0, [2, 5]] = 9.
    onp["res"][0, 7] = 9.
    onp["query"][5] = 9.
    facs, q = decode(onp)
    assert facs == [{"ftype": "rel", "op": "add", "args": [2, 5], "result": 7}]
    assert q == 5
    print("  [OK] decode round trip (rel factor + query pointer)")
    # gold builder on a real corpus line
    samples, ids, mask, offsets = tokenize(ALG_TEST)
    g2 = build_gold(samples[:2], offsets[:2])
    assert g2["vspan"][0].sum() > 0 and g2["fspan"][0].sum() > 0
    print("  [OK] gold builder on real corpus (mentions + factor spans)")
    print("[selftest] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.precompute:
        do_precompute()
    elif args.train:
        do_train(int(os.environ.get("STEPS", "8000")),
                 float(os.environ.get("LR", "3e-4")),
                 int(os.environ.get("BATCH", "8")),
                 int(os.environ.get("SEED", "0")))
    elif args.eval:
        do_eval()
    elif args.errors:
        do_errors()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
