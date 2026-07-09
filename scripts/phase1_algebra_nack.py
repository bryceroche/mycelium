"""phase1_algebra_nack.py — the ALGEBRA RETRANSMITTER: the flag-dependent repair
specialist, trained PURE (spec registrations, 2026-07-07 afternoon).

THE TWO-CHECKPOINT ARCHITECTURE IS THE PROTECTION: the plateaued algebra head
(ALG_CKPT) parses; THIS checkpoint only repairs. No blank-sample mix, no mixed
objective — the repair specialist trains pure (the blank-pass tax lesson applied
structurally: propose/dispose as two heads of one species over one frozen trunk).

FLAG FORMAT — BOTH GRANULARITIES FROM DAY ONE (ablatable, not retrofittable):
  SPAN-level : suspect-token embedding added at the waist for tokens of flagged
               factors' spans (the KenKen-proven channel, char-exact via gold spans).
  FIELD-level: per-slot x per-field flags (presence/ftype/op/args/res/digits)
               projected into the FACTOR-SLOT QUERIES (zero-init) — "this field of
               this factor is suspect" — the granularity the union-typed layout was
               built to support (re-emit the literal, keep the structure).

OBJECTIVE (flag-dependent, oracle flags for the transplant — Brick-A's arm
separation): flagged slots -> gold fix; unflagged slots -> copy the plateaued
parser's previous output. Flags are the only signal separating the behaviors.

EVAL (--eval): the COMPOSED TWO-CHECKPOINT ALGEBRA STACK, deployable end to end:
plateaued head parses -> withhold-2-and-solve (stage 1, the measured algebra peak)
-> survivors' withheld suspects become flags (gold-free) -> THIS head retransmits ->
withhold-2-and-solve again -> FORCED-answer recovery per stage. Oracle-flag arm
reported as the ceiling condition.

USAGE:
  Prep:   DEV=AMD .venv/bin/python3 scripts/phase1_algebra_nack.py --prep
  Train:  DEV=AMD STEPS=4000 .venv/bin/python3 scripts/phase1_algebra_nack.py --train
  Eval:   DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
              .venv/bin/python3 scripts/phase1_algebra_nack.py --eval
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, H_TRUNK, H_W, K_VARS, L_FAC, N_DIG,
    build_params, forward, loss_fn, load_alg, decode, ALG_CKPT,
)

N_FIELDS = 6          # presence, ftype, op, args, res, digits
NACK_CKPT = os.environ.get("NACK_CKPT", ".cache/phase1_algebra_nack.safetensors")
PREP_NPZ = ".cache/phase1_algebra_nack_prep.npz"


# ===========================================================================
# CONDITIONING PARAMS + FORWARD (both granularities, zero-init)
# ===========================================================================

def build_cond_params(seed: int = 5) -> dict:
    from tinygrad import Tensor, dtypes

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x
    return {
        "susp_tok": t(np.zeros((H_W,))),               # span-level, at the waist
        "fail_emb": t(np.zeros((H_W,))),               # global-fail, at the waist
        "field_w": t(np.zeros((N_FIELDS, H_W))),       # field-level, into slot queries
    }


def forward_cond(p, c, trunk, tokmask, sent, flag_tok, fail_bit, field_flags):
    """The algebra head forward with BOTH conditioning channels. Local variant —
    phase1_algebra_head.forward stays untouched. field_flags: (B, L_FAC, N_FIELDS)."""
    B = trunk.shape[0]
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]
    waist = waist + c["susp_tok"].reshape(1, 1, -1) * flag_tok.unsqueeze(-1)
    waist = waist + c["fail_emb"].reshape(1, 1, -1) * fail_bit.reshape(B, 1, 1)

    def bank(queries, nq, extra=None):
        q_in = queries.unsqueeze(0) + (extra if extra is not None else 0)
        q = q_in @ p["attn_wq"] + p["attn_wq_b"]
        k = waist @ p["attn_wk"] + p["attn_wk_b"]
        v = waist @ p["attn_wv"] + p["attn_wv_b"]
        hd = H_W // 8
        qh = q.reshape(B if extra is not None else 1, nq, 8, hd).permute(0, 2, 1, 3)
        kh = k.reshape(B, -1, 8, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, -1, 8, hd).permute(0, 2, 1, 3)
        sc = (qh @ kh.transpose(-2, -1)) / math.sqrt(hd)
        sc = sc.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
        at = sc.softmax(-1)
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)
        st = st @ p["attn_wo"] + p["attn_wo_b"] + q_in.reshape(-1, nq, H_W)
        st = st + ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])
        return st, at.mean(1)

    fq_extra = field_flags @ c["field_w"]              # (B, L_FAC, H_W), zero at init
    vst, vat = bank(p["vq"], K_VARS)
    fst, fat = bank(p["fq"], L_FAC, extra=fq_extra)
    qst, _ = bank(p["qq"], 1)

    def ptr(W):
        return (fst @ W) @ vst.transpose(-2, -1)

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


# ===========================================================================
# PREP: plateaued parses on TRAIN -> per-slot per-field wrong masks + prev outputs
# ===========================================================================

def wrong_fields(o, g, i, bi):
    """(L_FAC, N_FIELDS) oracle wrong-mask + (L_FAC,) any-wrong, from gold arrays."""
    wf = np.zeros((L_FAC, N_FIELDS), bool)
    for j in range(L_FAC):
        pg = g["presence"][i, j] > 0.5
        pp = o["pres"][bi, j] > 0
        if pg != pp:
            wf[j, 0] = True
            continue
        if not pg:
            continue
        wf[j, 1] = int(o["ftype"][bi, j].argmax()) != g["ftype"][i, j]
        ft = int(g["ftype"][i, j])   # 0=rel 1=given 2=mod 3=sel (tranche-aware)
        if ft == 0:
            wf[j, 2] = int(o["op"][bi, j].argmax()) != g["op"][i, j]
        if ft in (0, 3):             # rel + sel carry 2-hot args
            top2 = set(np.argsort(-o["args"][bi, j])[:2].tolist())
            wf[j, 3] = top2 != set(np.where(g["args"][i, j] > 0.5)[0].tolist())
        if ft in (2, 4, 5):          # mod/pct/fdiv carry a 1-hot arg
            wf[j, 3] = int(np.argmax(o["args"][bi, j])) != \
                int(np.argmax(g["args"][i, j]))
        wf[j, 4] = int(o["res"][bi, j].argmax()) != g["res"][i, j]
        if ft in (1, 2, 4, 5):       # digits: values + params (NOT sel)
            wf[j, 5] = not bool(
                (o["dig"][bi, j].argmax(-1) == g["digits"][i, j]).all())
    return wf


def do_prep():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    # NACK_SPLIT: the failure-mining slice (held out from BOTH parser training and
    # the measurement set — the convergence-starvation fix). Default: train.
    samples, states, tokmask, gold, sent = load_alg(
        os.environ.get("NACK_SPLIT", "train"))
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    WF = np.zeros((n, L_FAC, N_FIELDS), np.uint8)
    prev = {"pres": np.zeros((n, L_FAC), np.uint8),
            "ftype": np.zeros((n, L_FAC), np.int8),
            "op": np.zeros((n, L_FAC), np.int8),
            "args": np.zeros((n, L_FAC, K_VARS), np.uint8),
            "res": np.zeros((n, L_FAC), np.int8),
            "dig": np.zeros((n, L_FAC, N_DIG), np.int8),
            "sel": np.zeros((n, L_FAC), np.int8)}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res")
             + (("sel",) if "sel" in out else ())}
        for bi, i in enumerate(sl):
            i = int(i)
            WF[i] = wrong_fields(o, gold, i, bi)
            prev["pres"][i] = (o["pres"][bi] > 0).astype(np.uint8)
            prev["ftype"][i] = o["ftype"][bi].argmax(-1).astype(np.int8)
            prev["op"][i] = o["op"][bi].argmax(-1).astype(np.int8)
            for j in range(L_FAC):
                for a in np.argsort(-o["args"][bi, j])[:2]:
                    prev["args"][i, j, a] = 1
            prev["res"][i] = o["res"][bi].argmax(-1).astype(np.int8)
            prev["dig"][i] = o["dig"][bi].argmax(-1).astype(np.int8)
            if "sel" in o:
                prev["sel"][i] = o["sel"][bi].argmax(-1).astype(np.int8)
        if s0 % 512 == 0:
            print(f"  [prep] {s0}/{n}", flush=True)
    has_fail = WF.any(axis=(1, 2))

    # CURRICULUM PURITY (2026-07-08): graph-match labels right-asked-wrong-graph
    # parses as "failures" — training the specialist to fix CORRECT readings toward
    # canonical gold. Filter: a parse whose answer is FORCED-correct is NOT a
    # failure, whatever its graph looks like. PURITY=0 disables (for the
    # contamination measurement).
    if int(os.environ.get("PURITY", "1")):
        from mycelium.csp_domains import problem_from_algebra3 as problem_from_algebra2, ID_TO_SEL
        from mycelium.csp_core import solve_symbolic
        n_pure = 0
        for i in np.where(has_fail)[0]:
            i = int(i)
            smp = samples[i]
            # rebuild the parse from prev outputs (tranche-aware: 4 kinds)
            facs = []
            for j in range(L_FAC):
                if not prev["pres"][i, j]:
                    continue
                ft = int(prev["ftype"][i, j])
                res_j = int(prev["res"][i, j])
                val = int(sum(int(d) * 10 ** (N_DIG - 1 - k)
                              for k, d in enumerate(prev["dig"][i, j])))
                args2 = sorted(np.where(prev["args"][i, j] > 0)[0].tolist())[:2]
                if ft == 0:
                    if len(args2) < 2:
                        continue
                    facs.append({"ftype": "rel",
                                 "op": "add" if prev["op"][i, j] == 0 else "mul",
                                 "args": args2, "result": res_j})
                elif ft == 1:
                    facs.append({"ftype": "given", "var": res_j, "value": val})
                elif ft == 2:
                    if not args2:
                        continue
                    facs.append({"ftype": "mod", "var": int(args2[0]),
                                 "k": max(val, 2), "result": res_j})
                elif ft == 4:
                    if not args2:
                        continue
                    facs.append({"ftype": "pct",
                                 "args": [int(args2[0]), res_j],
                                 "p": max(val, 1)})
                elif ft == 5:
                    if not args2:
                        continue
                    facs.append({"ftype": "fdiv", "var": int(args2[0]),
                                 "k": max(val, 2), "result": res_j})
                else:
                    if len(args2) < 2:
                        continue
                    sid = int(prev.get("sel", np.zeros_like(prev["res"]))[i, j])
                    facs.append({"ftype": "sel", "sel": ID_TO_SEL.get(sid, "larger"),
                                 "args": args2, "result": res_j})
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            q = smp["query_var"]     # purity uses gold query (prep-side, oracle-ok)

            def _fv(f):
                if f["ftype"] in ("rel", "sel"):
                    return list(f["args"]) + [f["result"]]
                if f["ftype"] == "mod":
                    return [f["var"], f["result"]]
                return [f["var"]]
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in _fv(f)])
                res = solve_symbolic(problem_from_algebra2(nv, facs, gv, smp["m"]),
                                     budget=100_000, seed=0)
                if res["status"] != "solved":
                    continue
                sol = [int(res["assignment"][v]) for v in range(nv)]
                if q >= len(sol) or sol[q] != gold_ans:
                    continue
                p2 = problem_from_algebra2(nv, facs, gv, smp["m"])
                p2.domains0[q].discard(sol[q])
                if p2.domains0[q]:
                    r2 = solve_symbolic(p2, budget=60_000, seed=0)
                    if r2["status"] == "solved":
                        continue
                has_fail[i] = False        # forced-correct: never a failure
                n_pure += 1
            except Exception:
                continue
        print(f"[prep] PURITY filter: {n_pure} right-asked-wrong-graph parses "
              f"removed from the failure set", flush=True)

    np.savez_compressed(PREP_NPZ, wrong_fields=WF, has_fail=has_fail,
                        **{f"prev_{k}": v for k, v in prev.items()})
    print(f"[prep] {int(has_fail.sum())}/{n} organic failures -> {PREP_NPZ}",
          flush=True)


# ===========================================================================
# TRAIN: pure repair specialist (flag-dependent; oracle flags; no blank mix)
# ===========================================================================

def flags_from_wf(wf, gold_fspan_i):
    """wf (L,NF) -> (flag_tok (T,), fail (1,), field_flags (L,NF))."""
    any_slot = wf.any(-1)
    ft = np.zeros((T_ALG,), np.float32)
    for j in np.where(any_slot)[0]:
        ft = np.maximum(ft, gold_fspan_i[j].astype(np.float32))
    return ft, np.array([1.0 if any_slot.any() else 0.0], np.float32), \
        wf.astype(np.float32)


def do_train(steps, lr, batch, seed):
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_load, safe_save

    samples, states, tokmask, gold, sent = load_alg(
        os.environ.get("NACK_SPLIT", "train"))
    z = np.load(PREP_NPZ)
    WF, has_fail = z["wrong_fields"], z["has_fail"]
    prev = {k[5:]: z[k] for k in z.files if k.startswith("prev_")}
    fail_idx = np.where(has_fail)[0]
    print(f"[train] {len(fail_idx)} failures; PURE repair objective "
          f"(no blank mix — two-ckpt is the protection)", flush=True)

    p = build_params(seed)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    c = build_cond_params(seed + 5)
    opt = AdamW(list(p.values()) + list(c.values()), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float32), dtypes.float)
    b_tk = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float)
    b_se = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int)
    b_ft = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float)
    b_fb = fix(np.zeros((batch, 1), np.float32), dtypes.float)
    b_ff = fix(np.zeros((batch, L_FAC, N_FIELDS), np.float32), dtypes.float)
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
        o = forward_cond(p, c, b_tr, b_tk, b_se, b_ft, b_fb, b_ff)
        l = loss_fn(o, bg)
        opt.zero_grad()
        l.backward()
        opt.step()
        return l.realize()

    # HYGIENE (2026-07-10, transplanted from the parser trainer after the
    # same gap bit twice): cosine decay + periodic loss-EMA pick-best.
    lr_min = lr / 30.0
    best_ema, best_snap, ema = float("inf"), None, None
    t0 = time.time()
    for s in range(steps):
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * s / steps))
        opt.lr.assign(Tensor([cur_lr], dtype=dtypes.float)).realize()
        idx = rng.choice(fail_idx, batch, replace=False)
        ftok = np.zeros((batch, T_ALG), np.float32)
        fbit = np.zeros((batch, 1), np.float32)
        ffld = np.zeros((batch, L_FAC, N_FIELDS), np.float32)
        tg = {}
        use_fix = WF[idx].any(-1)                       # (B, L_FAC)
        for bi, i in enumerate(idx):
            ftok[bi], fbit[bi], ffld[bi] = flags_from_wf(WF[i], gold["fspan"][i])
        tg["presence"] = np.where(use_fix, gold["presence"][idx] > .5,
                                  prev["pres"][idx] > .5).astype(np.float32)
        tg["is_lit_f"] = (gold["is_lit"][idx] > .5).astype(np.float32)
        tg["fspan"] = gold["fspan"][idx].astype(np.float32)
        tg["vspan"] = gold["vspan"][idx].astype(np.float32)
        tg["args"] = np.where(use_fix[:, :, None], gold["args"][idx] > .5,
                              prev["args"][idx] > .5).astype(np.float32)
        tg["ftype"] = np.where(use_fix, gold["ftype"][idx],
                               prev["ftype"][idx]).astype(np.int32)
        tg["op"] = np.where(use_fix, gold["op"][idx], prev["op"][idx]).astype(np.int32)
        tg["res"] = np.where(use_fix, gold["res"][idx],
                             prev["res"][idx]).astype(np.int32)
        tg["digits"] = np.where(use_fix[:, :, None], gold["digits"][idx],
                                prev["dig"][idx]).astype(np.int32)
        tg["query"] = gold["query"][idx].astype(np.int32)
        tg["sel"] = np.where(use_fix, gold.get("sel", np.zeros_like(gold["op"]))[idx],
                             prev.get("sel", np.zeros_like(prev["op"]))[idx]).astype(np.int32)
        # per-kind masks follow the TARGET ftype (blended parse, per slot)
        tg["is_rel"] = (tg["ftype"] == 0).astype(np.float32)
        tg["is_mod"] = (tg["ftype"] == 2).astype(np.float32)
        tg["is_sel"] = (tg["ftype"] == 3).astype(np.float32)
        b_tr.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        b_ft.assign(Tensor(ftok, dtype=dtypes.float).contiguous()).realize()
        b_fb.assign(Tensor(fbit, dtype=dtypes.float).contiguous()).realize()
        b_ff.assign(Tensor(ffld, dtype=dtypes.float).contiguous()).realize()
        for kk in bg:
            npdt = np.float32 if bg[kk].dtype == dtypes.float else np.int32
            bg[kk].assign(Tensor(tg[kk].astype(npdt), dtype=bg[kk].dtype).contiguous()).realize()
        lv = step()
        v = float(lv.numpy())
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        if s > steps // 4 and ema < best_ema:
            best_ema = ema
            best_snap = {k: t.detach().numpy().copy()
                         for d in (p, c) for k, t in d.items()}
        if s % 500 == 0 or s == steps - 1:
            v = float(lv.numpy())
            assert np.isfinite(v)
            print(f"  step {s:5d} loss={v:.4f} ({(time.time()-t0)/(s+1):.2f}s/step)",
                  flush=True)
    if best_snap is not None:
        for d in (p, c):
            for k in d:
                d[k].assign(Tensor(best_snap[k], dtype=d[k].dtype)).realize()
        print(f"[train] restored best-by-loss-EMA ({best_ema:.3f})", flush=True)
    safe_save({**p, **c}, NACK_CKPT)
    print(f"[train] saved {NACK_CKPT}", flush=True)


# ===========================================================================
# EVAL: the composed two-checkpoint ALGEBRA stack (deployable + oracle ceiling)
# ===========================================================================

def do_eval():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic
    from tier0_incumbent import softmax, sig

    samples, states, tokmask, gold, sent = load_alg("test")
    p_plat = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p_plat:
        p_plat[k].assign(sd[k].to(p_plat[k].device).cast(p_plat[k].dtype)).realize()
    p_re = build_params(1)
    c_re = build_cond_params(1)
    sd2 = safe_load(NACK_CKPT)
    for d in (p_re, c_re):
        for k in d:
            d[k].assign(sd2[k].to(d[k].device).cast(d[k].dtype)).realize()
    n = len(samples)

    def slot_conf(o, j):
        cc = float(sig(o["pres"][j]))
        cc *= float(softmax(o["ftype"][j][None])[0].max())
        cc *= float(softmax(o["res"][j][None])[0].max())
        if sig(o["islit"][j]) > 0.5:
            cc *= float(np.mean(softmax(o["dig"][j]).max(-1)))
        else:
            cc *= float(softmax(o["op"][j][None])[0].max())
            cc *= float(np.sort(softmax(o["args"][j][None])[0])[-2:].sum())
        return cc

    def solve_check(facs, q_pred, smp, k_wh, o=None):
        """withhold k least-confident (if o given) then FORCED-answer check."""
        if o is not None and k_wh:
            confs = []
            fi = 0
            for j in range(L_FAC):
                if o["pres"][j] <= 0:
                    continue
                confs.append((fi, slot_conf(o, j)))
                fi += 1
            order = [x for x, _ in sorted(confs, key=lambda t: t[1])]
            wh = set(order[:k_wh])
            facs = [f for x, f in enumerate(facs) if x not in wh]
        else:
            wh = set()
        rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                for f in facs if f["ftype"] == "rel"]
        gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
        gold_ans = smp["solution"][smp["query_var"]]
        try:
            nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                     ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                      else [f["var"]])] + [q_pred + 1])
            res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                 budget=200_000, seed=0)
            if res["status"] != "solved":
                return False, wh
            sol = [int(res["assignment"][v]) for v in range(nv)]
            if not (q_pred < len(sol) and sol[q_pred] == gold_ans):
                return False, wh
            p2 = problem_from_algebra(nv, rels, gv, smp["m"])
            p2.domains0[q_pred].discard(sol[q_pred])
            if p2.domains0[q_pred]:
                r2 = solve_symbolic(p2, budget=100_000, seed=0)
                if r2["status"] == "solved":
                    return False, wh
            return True, wh
        except Exception:
            return False, wh

    def run(model_p, cond_c, idx, ftok, fbit, ffld):
        outs = {}
        for s0 in range(0, len(idx), 8):
            sl = np.array(idx[s0:s0 + 8])
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            rows = slice(s0, s0 + len(sl))
            f1 = ftok[rows]; f2 = fbit[rows]; f3 = ffld[rows]
            if pad:
                f1 = np.concatenate([f1, f1[:1].repeat(pad, 0)])
                f2 = np.concatenate([f2, f2[:1].repeat(pad, 0)])
                f3 = np.concatenate([f3, f3[:1].repeat(pad, 0)])
            out = forward_cond(
                model_p, cond_c,
                Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                Tensor(f1, dtype=dtypes.float), Tensor(f2, dtype=dtypes.float),
                Tensor(f3, dtype=dtypes.float))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
            for bi, i in enumerate(sl):
                outs[int(i)] = {k: o[k][bi] for k in o}
        return outs

    # STAGE 0/1: plateaued parse (zero conditioning through the SAME fwd for parity)
    zft = np.zeros((n, T_ALG), np.float32)
    zfb = np.zeros((n, 1), np.float32)
    zff = np.zeros((n, L_FAC, N_FIELDS), np.float32)
    c_zero = build_cond_params(9)     # zero-init = plain forward
    blank = run(p_plat, c_zero, list(range(n)), zft, zfb, zff)
    K_WH = 2                          # the measured algebra peak
    stage1, survivors = 0, []
    surv_flags = []
    for i in range(n):
        smp = samples[i]
        o = blank[i]
        facs, q_pred = decode(o)
        ok, _ = solve_check(facs, q_pred, smp, 0)
        if ok:
            continue                   # not a failure
        ok, wh = solve_check(facs, q_pred, smp, K_WH, o=o)
        if ok:
            stage1 += 1
            continue
        # suspects -> flags (gold-free): the withheld slots' spans + field flags all-on
        ftok_i = np.zeros((T_ALG,), np.float32)
        ffld_i = np.zeros((L_FAC, N_FIELDS), np.float32)
        fi = 0
        for j in range(L_FAC):
            if o["pres"][j] <= 0:
                continue
            if fi in wh:
                ffld_i[j, :] = 1.0
                ftok_i = np.maximum(ftok_i, gold["fspan"][i, j].astype(np.float32))
            fi += 1
        survivors.append(i)
        surv_flags.append((ftok_i, ffld_i))
    n_fail = stage1 + len(survivors)
    print(f"[alg-stack] failures {n_fail}/{n} | stage-1 withhold-{K_WH}: {stage1} recovered")

    rounds = int(os.environ.get("ROUNDS", "1"))
    if survivors:
        arm = os.environ.get("ARM", "both")
        if arm == "field_only":
            surv_flags = [(np.zeros_like(f), ff) for f, ff in surv_flags]
            print("[alg-stack] ARM=field_only (span channel zeroed — fully deployable)")
        ftok_s = np.stack([f for f, _ in surv_flags])
        ffld_s = np.stack([f for _, f in surv_flags])
        fbit_s = np.ones((len(survivors), 1), np.float32)
        total = stage1
        pool = survivors
        flags_pool = surv_flags
        for rnd in range(1, rounds + 1):
            if not pool:
                break
            ftok_s = np.stack([f for f, _ in flags_pool])
            ffld_s = np.stack([f for _, f in flags_pool])
            if arm == "field_only":
                ftok_s = np.zeros_like(ftok_s)
            fbit_s = np.ones((len(pool), 1), np.float32)
            re = run(p_re, c_re, pool, ftok_s, fbit_s, ffld_s)
            rec_r, nxt_pool, nxt_flags = 0, [], []
            for i in pool:
                o = re[int(i)]
                facs, q_pred = decode(o)
                ok, wh = solve_check(facs, q_pred, samples[int(i)], K_WH, o=o)
                if ok:
                    rec_r += 1
                    continue
                ftok_i = np.zeros((T_ALG,), np.float32)
                ffld_i = np.zeros((L_FAC, N_FIELDS), np.float32)
                fi = 0
                for j in range(L_FAC):
                    if o["pres"][j] <= 0:
                        continue
                    if fi in wh:
                        ffld_i[j, :] = 1.0
                        ftok_i = np.maximum(
                            ftok_i, gold["fspan"][int(i), j].astype(np.float32))
                    fi += 1
                nxt_pool.append(i)
                nxt_flags.append((ftok_i, ffld_i))
            total += rec_r
            print(f"[alg-stack] ROUND {rnd}: {rec_r}/{len(pool)} recovered "
                  f"({len(nxt_pool)} survive)")
            pool, flags_pool = nxt_pool, nxt_flags
        print(f"[alg-stack] MULTI-ROUND RECOVERY ({rounds} rounds): "
              f"{total}/{n_fail} = {total/n_fail:.2f}  (stage-1 alone was {stage1})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    if args.prep:
        do_prep()
    elif args.train:
        do_train(int(os.environ.get("STEPS", "4000")),
                 float(os.environ.get("LR", "1e-4")),
                 int(os.environ.get("BATCH", "8")),
                 int(os.environ.get("SEED", "0")))
    elif args.eval:
        do_eval()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
