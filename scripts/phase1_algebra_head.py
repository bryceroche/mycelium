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
H_W = 512
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
    }
    for i, (smp, offs) in enumerate(zip(samples, offsets)):
        facs = sorted(smp["factors"], key=lambda f: min(s for s, _ in f["spans"]))
        assert len(facs) <= L_FAC and smp["n_vars"] <= K_VARS
        g["query"][i] = smp["query_var"]
        g["band"][i] = smp["decisions"]
        for v_str, spans in smp["mentions"].items():
            _spans_to_tokmask(spans, offs, g["vspan"][i, int(v_str)])
        for j, f in enumerate(facs):
            g["presence"][i, j] = 1.0
            _spans_to_tokmask(f["spans"], offs, g["fspan"][i, j])
            if f["ftype"] == "rel":
                g["ftype"][i, j] = 0
                g["is_rel"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
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
        gold = build_gold(samples, offsets)
        sent = np.stack([sent_indices(s["text"], o, mask[i])
                         for i, (s, o) in enumerate(zip(samples, offsets))])
        np.savez_compressed(STATES_NPZ.format(split=split), states=states,
                            tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
                            **{f"g_{k}": v for k, v in gold.items()})
        print(f"[precompute] {split}: {states.shape}", flush=True)


def load_alg(split):
    is_train = split == "train"
    split = TRAIN_NAME if is_train else (TEST_NAME if split == "test" else split)
    z = np.load(STATES_NPZ.format(split=split))
    samples = [json.loads(l) for l in open(ALG_TRAIN if is_train else ALG_TEST)]
    gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    return samples, z["states"], z["tokmask"], gold, z["sent"]


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
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, 4)   # rel/given/mod/sel
        p["h_sel"], p["h_sel_b"] = lin(H_W, 4)       # larger/smaller/even/odd
    else:
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, 2)
    p["h_op"], p["h_op_b"] = lin(H_W, 2)
    p["h_islit"], p["h_islit_b"] = lin(H_W, 1)
    p["h_dig"], p["h_dig_b"] = lin(H_W, N_DIG * 10)
    p["W_args"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_res"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_query"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    return p


def forward(p, trunk, tokmask, sent):
    B = trunk.shape[0]
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]

    def bank(queries, nq):
        q = queries @ p["attn_wq"] + p["attn_wq_b"]
        k = waist @ p["attn_wk"] + p["attn_wk_b"]
        v = waist @ p["attn_wv"] + p["attn_wv_b"]
        hd = H_W // N_HEADS
        qh = q.reshape(nq, N_HEADS, hd).transpose(0, 1)
        kh = k.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        sc = (qh.unsqueeze(0) @ kh.transpose(-2, -1)) / math.sqrt(hd)
        sc = sc.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
        at = sc.softmax(-1)
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)
        st = st @ p["attn_wo"] + p["attn_wo_b"] + queries.unsqueeze(0)
        st = st + ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])
        return st, at.mean(1)

    vst, vat = bank(p["vq"], K_VARS)
    fst, fat = bank(p["fq"], L_FAC)
    qst, _qa = bank(p["qq"], 1)

    def ptr(W):
        return (fst @ W) @ vst.transpose(-2, -1)         # (B, L, K)

    return {
        "pres": (fst @ p["h_pres"] + p["h_pres_b"]).squeeze(-1),
        "ftype": fst @ p["h_ftype"] + p["h_ftype_b"],
        "op": fst @ p["h_op"] + p["h_op_b"],
        **({"sel": fst @ p["h_sel"] + p["h_sel_b"]} if "h_sel" in p else {}),
        "islit": (fst @ p["h_islit"] + p["h_islit_b"]).squeeze(-1),
        "dig": (fst @ p["h_dig"] + p["h_dig_b"]).reshape(B, L_FAC, N_DIG, 10),
        "args": ptr(p["W_args"]),
        "res": ptr(p["W_res"]),
        "query": ((qst @ p["W_query"]) @ vst.transpose(-2, -1)).reshape(B, K_VARS),
        "fat": fat, "vat": vat,
    }


def loss_fn(o, g):
    from tinygrad import Tensor
    pres = g["presence"]
    n_p = pres.sum() + 1e-6
    # explicit per-kind masks (tranche); legacy callers without them fall back
    # to the old two-kind arithmetic (byte-identical behavior)
    is_rel = g["is_rel"] if "is_rel" in g else (1.0 - g["is_lit_f"])
    is_mod = g["is_mod"] if "is_mod" in g else g["is_lit_f"] * 0.0
    is_sel = g["is_sel"] if "is_sel" in g else g["is_lit_f"] * 0.0
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
    dm = g["is_lit_f"] + is_mod                 # digits: given values + moduli
    l = l + (ce(o["dig"], g["digits"]).mean(-1) * dm).sum() / (dm.sum() + 1e-6)
    if "sel" in o and "sel" in g:               # selector-type CE (closed vocab)
        sm = pres * is_sel
        l = l + (ce(o["sel"], g["sel"]) * sm).sum() / (sm.sum() + 1e-6)
    args_w = 1.0 + 4.0 * g["args"]
    am = pres * (is_rel + is_sel + is_mod)
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
        ft = int(o_np["ftype"][j].argmax()) if o_np["ftype"].shape[-1] == 4 \
            else (1 if o_np["islit"][j] > 0 else 0)

        def digval():
            digs = o_np["dig"][j].argmax(-1)
            return int(sum(d * 10 ** (N_DIG - 1 - i)
                           for i, d in enumerate(digs)))
        if ft == 1:
            facs.append({"ftype": "given", "var": res, "value": digval()})
        elif ft == 0:
            args = list(np.argsort(-o_np["args"][j])[:2])
            op = "add" if o_np["op"][j].argmax() == 0 else "mul"
            facs.append({"ftype": "rel", "op": op,
                         "args": sorted(int(a) for a in args), "result": res})
        elif ft == 2:
            var = int(np.argmax(o_np["args"][j]))
            facs.append({"ftype": "mod", "var": var, "k": max(digval(), 2),
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
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ())
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
                return ("sel", f["sel"], tuple(sorted(f["args"])), f["result"])
            gset = set(fkey(f) for f in smp["factors"])
            st["fac_ok"] += len(gset & set(fkey(f) for f in facs))
            st["fac_tot"] += len(gset)
            st["query_ok"] += int(q_pred == smp["query_var"])
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

            def fvars(f):
                if f["ftype"] == "rel" or f["ftype"] == "sel":
                    return list(f["args"]) + [f["result"]]
                if f["ftype"] == "mod":
                    return [f["var"], f["result"]]
                return [f["var"]]
            try:
                from mycelium.csp_domains import problem_from_algebra2
                nv = max([smp["n_vars"]] + [v + 1 for f in facs
                                            for v in fvars(f)])
                res = solve_symbolic(problem_from_algebra2(nv, facs, gv,
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
                "query") + (("sel",) if "sel" in out else ())
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
            else:
                print(f"[warm] SKIP {k} (fresh init: "
                      f"{'missing' if k not in sd0 else 'shape'})", flush=True)
        print(f"[train] WARM from {os.environ['WARM_FROM']}: "
              f"{n_load}/{len(p)} keys", flush=True)
    opt = AdamW(list(p.values()), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float32), dtypes.float)
    b_tk = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float)
    b_se = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int)
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
                         ("query", (), dtypes.int)):
        npdt = np.float32 if dt == dtypes.float else np.int32
        bg[k] = fix(np.zeros((batch,) + shape, npdt), dt)

    @TinyJit
    def step():
        Tensor.training = True
        o = forward(p, b_tr, b_tk, b_se)
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
                   ("pres", "ftype", "op", "islit", "dig", "args", "res")}
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
                        top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                        ok &= top2 == set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    else:
                        ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                                    vg["digits"][i, j]).all())
                    n_ok += ok
        return n_ok / max(n_tot, 1)

    load_split_val = load_alg("test")
    best_val, best_snap = -1.0, None

    t0 = time.time()
    for s in range(steps):
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * s / steps))
        opt.lr.assign(Tensor([cur_lr], dtype=dtypes.float)).realize()
        idx = rng.choice(n, batch, replace=False)
        b_tr.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        feed = {"presence": gold["presence"][idx], "is_lit_f": gold["is_lit"][idx],
                "args": gold["args"][idx], "fspan": gold["fspan"][idx],
                "vspan": gold["vspan"][idx], "ftype": gold["ftype"][idx],
                "op": gold["op"][idx], "res": gold["res"][idx],
                "digits": gold["digits"][idx], "query": gold["query"][idx],
                "sel": gold["sel"][idx], "is_rel": gold["is_rel"][idx],
                "is_mod": gold["is_mod"][idx], "is_sel": gold["is_sel"][idx]}
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
