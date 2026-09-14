"""perceiver_vote_read.py — THE PERCEIVER step 2 (2026-09-14, word given):
the VIEWS + VOTE lever, read on wild. Per row: the original parse and
PV_VIEWS sentence-permuted views (the trunk recomputed per view; the locus
read's machinery). Per present slot of the original: AGREEMENT = how many
views contain its coarse key (ftype, op, value). For GIVEN slots, the
DIGITS VOTE: given slots are matched across views by the token they read
(the bridge's source token — the numeral / phrase is the same string under
a sentence permutation) and the majority digit tuple (original + views,
ties to the original) replaces the original's digits. Reports:
  (a) THE VOTE: paired open vs voted slot-exact (McNemar), the gains on
      given slots;
  (b) THE ABSTAIN CURVE at slot level: precision vs coverage by agreement
      threshold, by the args margin, and combined;
  (c) THE SPEND CURVE: if views are spent only on the rows with the lowest
      min-args-margin (or the highest settle at b3), what fraction of the
      vote's gains is captured.
Bars (pinned before the read): VOTE-LEVER if paired diff >= +0.010 slot-
exact AND McNemar z >= 2.0; the perceiver SAVES VIEWS if the lowest-margin
50% of rows capture >= 80% of the gains. Curves are reported, not barred.
Env: family env + PV_CKPT, ALG_TEST(_NAME), PV_VIEWS (4), PV_OUT."""
import os, sys


def _main():
    import time, numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, decode, sent_indices, _slot_margins, L_FAC, K_VARS, T_ALG, TOKENIZER_JSON
    from beacon_closing_arm import recompute_states
    from tta_views import permuted_view
    from mycelium.loop_bridge import Bridge
    from mycelium.perceiver import auroc
    from tokenizers import Tokenizer
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    NV = int(os.environ.get("PV_VIEWS", "4"))
    vs, vst, vtk, vg, vse = load_alg("test"); N = len(vs)
    p = build_params(0); sd = safe_load(os.environ["PV_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())

    def two_pass(ts, tk, se, nv, ma):
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy())
        _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        fb = Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fb)
        out = {k: o[k].numpy() for k in KEYS}
        out["fat"] = o["fat_all"][-1].numpy(); out["breaths"] = [b.numpy() for b in o["breaths_all"]]
        return out

    def slots_of(onp, b):
        """per present slot: coarse key + the decoded factor type"""
        row = {k: onp[k][b] for k in KEYS}; row["query"] = np.zeros(K_VARS, np.float32); keys = {}; kinds = {}
        for j in range(L_FAC):
            if row["pres"][j] <= 0: continue
            rj = dict(row); pr = np.full_like(row["pres"], -1.0); pr[j] = row["pres"][j]; rj["pres"] = pr
            try: facs, _ = decode(rj)
            except Exception: facs = []
            keys[j] = tuple(sorted((f["ftype"], f.get("op", ""), f.get("value", "")) for f in facs))
            kinds[j] = facs[0]["ftype"] if facs else None
        return keys, kinds

    def score(onp, b, i, dig_override=None):
        out = {}
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5: continue
            ok = (onp["pres"][b, j] > 0) and int(onp["ftype"][b, j].argmax()) == vg["ftype"][i, j] and int(onp["res"][b, j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok = ok and int(onp["op"][b, j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                ok = ok and ((bool(onp["dup"][b, j] > 0) and int(np.argmax(onp["args"][b, j])) in gset) if (len(gset) == 1 and "dup" in onp) else set(np.argsort(-onp["args"][b, j])[:2].tolist()) == gset)
            else:
                digs = dig_override.get(j) if (dig_override and j in dig_override) else tuple(onp["dig"][b, j].argmax(-1))
                ok = ok and bool((np.array(digs) == vg["digits"][i, j]).all())
            out[j] = bool(ok)
        return out

    def tokenize(texts):
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
        for b, t in enumerate(texts):
            e = tok.encode(t); Ln = min(len(e.ids), T_ALG); ids[b, :Ln] = e.ids[:Ln]; msk[b, :Ln] = 1.0; snt[b] = sent_indices(t, list(e.offsets), msk[b])
        return ids, msk, snt

    def src_strings(onp, ids, snt, msk, b, slots):
        br = Bridge(onp["fat"][b:b + 1], snt[b:b + 1], msk[b:b + 1])
        return {j: tok.decode([int(ids[b, br.source_token(0, j)])]).strip().lower() for j in slots}

    rec = []      # per present slot: (i, j, open_ok, vote_ok, agree, is_given, changed, margin_args, margin_res, row_min_args, row_settle_b3)
    t0 = time.time()
    for s0 in range(0, N, 8):
        sl = np.arange(s0, min(s0 + 8, N)); pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o = two_pass(ts, tk, se, nv, ma)
        ids0, msk0, snt0 = tokenize([vs[int(i)]["text"] for i in sl_p])
        keys0, kinds0, src0 = [], [], []
        for b in range(8):
            kj, kd = slots_of(o, b); keys0.append(kj); kinds0.append(kd)
            src0.append(src_strings(o, ids0, snt0, msk0, b, [j for j, t in kd.items() if t == "given"]))
        agree = np.zeros((8, L_FAC)); votes = [{j: [tuple(o["dig"][b, j].argmax(-1))] for j in src0[b]} for b in range(8)]
        for v in range(1, NV + 1):
            texts = [permuted_view(vs[int(i)]["text"], 40000 + 10 * int(i) + v) for i in sl_p]
            ids, msk, snt = tokenize(texts)
            stv = recompute_states(ids)
            ov = two_pass(Tensor(stv, dtype=dtypes.half), Tensor(msk), Tensor(snt, dtype=dtypes.int), nv, ma)
            for b in range(8):
                kv_keys, kv_kinds = slots_of(ov, b); kv = list(kv_keys.values())
                for j, kj in keys0[b].items():
                    if kj in kv: agree[b, j] += 1; kv.remove(kj)
                gv = [j for j, t in kv_kinds.items() if t == "given"]
                sv = src_strings(ov, ids, snt, msk, b, gv)
                by_src = {}
                for j in gv: by_src.setdefault(sv[j], tuple(ov["dig"][b, j].argmax(-1)))
                for j, s in src0[b].items():
                    if s in by_src: votes[b][j].append(by_src[s])
        for b, i in enumerate(sl):
            i = int(i)
            over = {}
            for j, vl in votes[b].items():
                cnt = {}
                for d in vl: cnt[d] = cnt.get(d, 0) + 1
                best = max(cnt.items(), key=lambda kv_: (kv_[1], kv_[0] == vl[0]))[0]
                if best != vl[0] and cnt[best] > cnt[vl[0]]: over[j] = best
            so = score(o, b, i); sv_ = score(o, b, i, over)
            row = {k: o[k][b] for k in KEYS}
            marg = {j: {f: m for f, m, _ in _slot_margins(row, j)} for j in so}
            row_min_args = min([marg[j].get("args", np.inf) for j in so] or [np.nan])
            settle3 = float((np.linalg.norm(o["breaths"][3][b] - o["breaths"][2][b], axis=-1) / (np.linalg.norm(o["breaths"][2][b], axis=-1) + 1e-6)).mean())
            for j in so:
                rec.append((i, j, so[j], sv_[j], agree[b, j], kinds0[b].get(j) == "given", j in over, marg[j].get("args", np.nan), marg[j].get("res", np.nan), row_min_args, settle3))
        if s0 % 80 == 0: print(f"[vote] {s0}/{N} rows ({time.time() - t0:.0f}s)", flush=True)

    R = np.array(rec, dtype=object); i_ = R[:, 0].astype(int); o_ = R[:, 2].astype(bool); v_ = R[:, 3].astype(bool); ag = R[:, 4].astype(float)
    gv_ = R[:, 5].astype(bool); ch = R[:, 6].astype(bool); ma_ = R[:, 7].astype(float); mr_ = R[:, 8].astype(float); rma = R[:, 9].astype(float); st3 = R[:, 10].astype(float)
    n = len(o_); name = os.path.basename(os.environ["PV_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
    only_v = int((~o_ & v_).sum()); only_o = int((o_ & ~v_).sum()); z = (only_v - only_o) / max(np.sqrt(only_v + only_o), 1)
    print(f"[vote-read] {name} on {os.environ.get('ALG_TEST_NAME', '?')}: {n} slots ({gv_.sum()} given), {NV} views")
    print(f"[vote-read] THE VOTE: open {o_.mean():.4f} -> voted {v_.mean():.4f} (diff {v_.mean() - o_.mean():+.4f}; gained {only_v} / lost {only_o}; McNemar z {z:+.2f}); digits changed on {ch.sum()} given slots; "
          f"given-slot open {o_[gv_].mean():.4f} -> voted {v_[gv_].mean():.4f}; {'VOTE-LEVER' if (v_.mean() - o_.mean() >= 0.010 and z >= 2.0) else 'not a lever on this bar'}")
    print(f"[vote-read] agreement: P(wrong | agree<=1) {(~o_[ag <= 1]).mean():.3f} (n={int((ag <= 1).sum())}) vs P(wrong | agree=={NV}) {(~o_[ag == NV]).mean():.3f} (n={int((ag == NV).sum())}); AUROC(slot ok | agree) {auroc(ag, o_):.3f}; AUROC(slot ok | args margin) {auroc(np.nan_to_num(ma_, nan=-1), o_):.3f}")
    print("[vote-read] THE ABSTAIN CURVE (slot level, precision @ coverage):")
    for t in range(NV, -1, -1):
        m = ag >= t; print(f"    agree >= {t}: precision {o_[m].mean():.3f} @ coverage {m.mean():.3f} (n={int(m.sum())})")
    q = np.nanquantile(ma_, [0.8, 0.6, 0.4, 0.2, 0.0])
    for c, th in zip((0.2, 0.4, 0.6, 0.8, 1.0), q):
        m = np.nan_to_num(ma_, nan=-1) >= th; print(f"    args margin top {c:.0%}: precision {o_[m].mean():.3f} @ coverage {m.mean():.3f}")
    for t in (NV, NV - 1):
        m = (ag >= t) & (np.nan_to_num(ma_, nan=-1) >= q[2]); print(f"    agree >= {t} & args margin top 60%: precision {o_[m].mean():.3f} @ coverage {m.mean():.3f}")
    gains = (~o_ & v_).astype(float) - (o_ & ~v_).astype(float)
    tot = gains.sum()
    print("[vote-read] THE SPEND CURVE (rows sorted; fraction of the net gain captured):")
    rows = np.unique(i_)
    for label, key, rev in (("lowest min-args-margin", rma, False), ("highest settle b3", st3, True)):
        rk = {r: (key[i_ == r][0]) for r in rows}; order = sorted(rows, key=lambda r: (-rk[r] if rev else rk[r]) if np.isfinite(rk[r]) else np.inf)
        for frac in (0.25, 0.5, 0.75):
            chosen = set(order[:int(round(frac * len(order)))]); g = sum(gains[k] for k in range(n) if i_[k] in chosen)
            print(f"    views on the {label} {frac:.0%} of rows: {g / tot if tot else float('nan'):.2f} of the net gain ({g:+.0f} of {tot:+.0f})" + ("  <- SAVES VIEWS" if frac == 0.5 and tot > 0 and g / tot >= 0.8 else ""))
    out = os.environ.get("PV_OUT", f".cache/perceiver_vote_{name}_{os.environ.get('ALG_TEST_NAME', 'x')}.npz")
    np.savez(out, i=i_, j=R[:, 1].astype(int), open_ok=o_, vote_ok=v_, agree=ag, given=gv_, changed=ch, margin_args=ma_, margin_res=mr_, row_min_args=rma, settle_b3=st3)
    print(f"[vote-read] saved {out}")


if __name__ == "__main__":
    _main()
