"""attention_census.py — THE ATTENTION CENSUS (2026-09-15, word given): where
do the slots LOOK? Per present gold slot at the last breath (fat_all[-1],
head-mean slots<-tokens): the mass on DIGIT tokens, on the gold span's
tokens (mint has spans), on operator cues (each/total/per/times/more/
less/half/twice), and on the argmax token's class; split by given vs
relation slots and by slot-correct vs wrong; wild vs mint. A read.
Env: family env + ALG_MINE_BREATHS=1 (set), PV_CKPT, ALG_TEST(_NAME), PV_N."""
import os, sys


def _main():
    import re, numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, L_FAC, K_VARS, T_ALG, TOKENIZER_JSON
    from tokenizers import Tokenizer
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    tok = Tokenizer.from_file(TOKENIZER_JSON); CUES = re.compile(r"^(each|total|per|times|more|less|fewer|half|twice|double|altogether|combined|every|sum|difference|product|remaining|left|rest)$", re.I)
    vs, vst, vtk, vg, vse = load_alg("test"); N = min(int(os.environ.get("PV_N", "512")), len(vs)); rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(vs), N, replace=False))
    p = build_params(0); sd = safe_load(os.environ["PV_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    rec = []   # (is_rel, ok, mass_digit, mass_cue, mass_span, argmax_is_digit, argmax_is_cue, argmax_in_span, entropy)
    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32)); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float))
        fat = o["fat_all"][-1].numpy(); hd = {k: v.numpy() for k, v in o["heads_all"][-1].items() if k in ("pres", "ftype", "op", "dig", "args", "res", "dup")}
        for bi, i in enumerate(sl):
            i = int(i); r = vs[i]; enc = tok.encode(r["text"]); ids = enc.ids[:T_ALG]; offs = list(enc.offsets[:T_ALG]); T = len(ids)
            strs = [tok.decode([t]).strip() for t in ids]; is_digit = np.array([s.isdigit() for s in strs]); is_cue = np.array([bool(CUES.match(s)) for s in strs])
            for j in range(L_FAC):
                if vg["presence"][i, j] < 0.5: continue
                rel = vg["ftype"][i, j] == 0
                ok = (hd["pres"][bi, j] > 0) and int(hd["ftype"][bi, j].argmax()) == vg["ftype"][i, j] and int(hd["res"][bi, j].argmax()) == vg["res"][i, j]
                if rel:
                    ok = ok and int(hd["op"][bi, j].argmax()) == vg["op"][i, j]; gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    ok = ok and ((bool(hd["dup"][bi, j] > 0) and int(np.argmax(hd["args"][bi, j])) in gset) if (len(gset) == 1 and "dup" in hd) else set(np.argsort(-hd["args"][bi, j])[:2].tolist()) == gset)
                else:
                    ok = ok and bool((hd["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
                a = fat[bi, j, :T]; a = a / (a.sum() + 1e-9); am = int(a.argmax())
                spans = (r["factors"][j].get("spans") if j < len(r["factors"]) else None) or []
                in_span = np.zeros(T, bool)
                for s_, e_ in spans:
                    for t_, (x0, x1) in enumerate(offs):
                        if x1 > s_ and x0 < e_ and x1 > x0: in_span[t_] = True
                ent = float(-(a[a > 0] * np.log(a[a > 0])).sum())
                rec.append((bool(rel), bool(ok), float(a[is_digit].sum()), float(a[is_cue].sum()), float(a[in_span].sum()) if spans else np.nan, bool(is_digit[am]), bool(is_cue[am]), bool(in_span[am]) if spans else np.nan, ent))
    R = np.array(rec, dtype=float); rel, ok = R[:, 0] > 0, R[:, 1] > 0
    name = os.path.basename(os.environ["PV_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
    print(f"[attn-census] {name} on {os.environ.get('ALG_TEST_NAME', '?')}: {len(R)} present slots ({int((~rel).sum())} given, {int(rel.sum())} rel); slot-exact {ok.mean():.3f}")
    for lab, m in (("GIVEN ok", ~rel & ok), ("GIVEN wrong", ~rel & ~ok), ("REL ok", rel & ok), ("REL wrong", rel & ~ok)):
        if m.sum() == 0: continue
        S = R[m]
        print(f"  {lab:12s} n={int(m.sum()):5d} | mass on digits {np.nanmean(S[:, 2]):.3f}  on cues {np.nanmean(S[:, 3]):.3f}  on gold span {np.nanmean(S[:, 4]):.3f} | argmax is digit {np.nanmean(S[:, 5]):.3f}  is cue {np.nanmean(S[:, 6]):.3f}  in gold span {np.nanmean(S[:, 7]):.3f} | entropy {np.nanmean(S[:, 8]):.2f}")
    np.savez(f".cache/attn_census_{name}_{os.environ.get('ALG_TEST_NAME', 'x')}.npz", rec=R)


if __name__ == "__main__":
    _main()
