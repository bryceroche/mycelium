"""sine_shuffle_read.py -- THE MECHANISM READ for THE SINE BREATH (2026-10-10), the shuffle
read's own grouping-factor method (scripts/shuffle_read.py, itself ported from branch replay)
applied to THIS build's OWN taps: _depth_stack_sine's "depth_attn_k{i}" census entries (one per
ALG_DEPTH block, per breath, (B, nh, L_TOT, L_TOT) -- head-meaned here to match shuffle_read's
own (B, L_TOT, L_TOT) convention before the grouping-factor math, which is otherwise REUSED
verbatim: the governor census resolves each gold slot's OWNER (the entity/number it belongs to
in the text), then for every (block, breath, right/wrong) bucket the mean attention a slot pays
to OTHER slots sharing its own owner vs slots with a DIFFERENT owner, and the attention's
effective k (exp(entropy)).

THE REGISTERED BAR: k0's owner ratio (same-owner / different-owner mean attention) >= 1.5 (kill:
flat across widths = the crystal carries no ownership). Reports EVERY block, not just k0, so
ANTI4's own trough (at index 2, not 0) is read the same way without a second script.

Usage: sine_shuffle_read.py <TAG> -- caller exports the arm's own env first (FAM+SURF8+ALG_SINE+
ALG_SINE_SKIP), the rack_chain_SINE4.sh/ANTI4/SEAL convention. CKPT=.cache/sharp_<TAG>.safetensors,
right/wrong=.cache/ps_legal_wild_<TAG>.npz. Writes .cache/sine_shuffle_<TAG>.txt only.
"""
import os
import sys
import time
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: sine_shuffle_read.py <TAG> -- caller must export the arm's own env first")
TAG = sys.argv[1]
N_ROWS = 64
CKPT = f".cache/sharp_{TAG}.safetensors"
PS_LEGAL_PATH = f".cache/ps_legal_wild_{TAG}.npz"
OUT_TXT = f".cache/sine_shuffle_{TAG}.txt"


def main():
    P = []
    def log(s):
        print(s, flush=True); P.append(s)

    t0 = time.time()
    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    from tinygrad import Tensor, dtypes
    import spacy
    import governor_census as GC
    from tokenizers import Tokenizer

    assert H.ALG_SINE, "ALG_SINE=<sine|anti|flat> required (caller's own env)"
    log(f"[sine-shuffle] TAG={TAG} ALG_SINE={H.ALG_SINE} ALG_SINE_SKIP={H.ALG_SINE_SKIP} "
        f"widths={H._SINE_WIDTHS} trough_i={H._SINE_TROUGH_I} ckpt={CKPT}")

    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vs, vst_, vtk, vg, vse = H.load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)

    H._CENSUS = []
    o = H.forward(p, ts, tk, se, slot_mask=slot_mask)
    census = H._CENSUS
    H._CENSUS = None
    log(f"[sine-shuffle] census forward done ({time.time() - t0:.0f}s so far); {len(census)} tap entries")

    attn = {}   # (block_i, kb) -> (B, L_TOT, L_TOT), head-meaned
    for kb, tag, arr in census:
        if isinstance(tag, str) and tag.startswith("depth_attn_k"):
            i = int(tag[len("depth_attn_k"):])
            attn[(i, kb)] = arr.mean(axis=1)   # mean over heads
    blocks = sorted({i for (i, kb) in attn})
    breaths = sorted({kb for (i, kb) in attn})
    log(f"[sine-shuffle] depth_attn_k{{i}} captured: blocks={blocks} breaths={breaths}")
    if not blocks:
        log("[sine-shuffle] NO depth_attn_k{i} taps found -- is ALG_SINE set and this checkout's "
            "_depth_stack_sine current? exiting cleanly.")
        with open(OUT_TXT, "w") as f:
            f.write("\n".join(P) + "\n")
        return

    # ---- governor census (verbatim method, scripts/shuffle_read.py) ----
    NLP = spacy.load("en_core_web_sm")
    ps = np.load(PS_LEGAL_PATH)
    ok_lookup = {(int(r), int(c)): bool(o_) for r, c, o_ in zip(ps["rows"], ps["slots"], ps["ok"])}

    owner_of_slot = {}
    right_of_slot = {}
    n_resolved = n_unresolved = 0
    for i in range(n):
        row = vs[i]
        facs = row["factors"]
        text = row["text"]
        doc = NLP(text)
        sent_list = list(doc.sents)
        cands = GC.numeral_candidates(doc)
        given_list = [(f["var"], f["value"]) for f in facs if f["ftype"] == "given"]
        given_vars = {v for v, _ in given_list}
        bound = GC.bind_given_occurrences(doc, given_list, cands)
        owner_of_var = {}
        for var, sp_tok in bound.items():
            if sp_tok is None:
                owner_of_var[var] = "none"; n_unresolved += 1; continue
            ch = GC.governor_chain(doc, sent_list, sp_tok)
            if ch is None:
                owner_of_var[var] = "none"; n_unresolved += 1
            else:
                owner_of_var[var] = ch["owner_key"]; n_resolved += 1
        for j, fac in enumerate(facs):
            if j >= H.L_FAC:
                continue
            right_of_slot[(i, j)] = ok_lookup.get((i, j))
            if fac["ftype"] == "given":
                owner_of_slot[(i, j)] = owner_of_var.get(fac["var"], "none")
            elif fac["ftype"] == "rel":
                args = list(fac["args"])
                arg_owners = [owner_of_var.get(a, "none") if a in given_vars else None for a in args]
                if len(arg_owners) >= 2 and all(o_ is not None and o_ != "none" for o_ in arg_owners) and len(set(arg_owners)) == 1:
                    owner_of_slot[(i, j)] = arg_owners[0]
                else:
                    owner_of_slot[(i, j)] = "mixed"
            else:
                owner_of_slot[(i, j)] = "none"
        if i % 16 == 0:
            log(f"[sine-shuffle] governor census row {i}/{n} ({time.time() - t0:.0f}s so far)")

    gold_slots = [(i, j) for (i, j) in owner_of_slot if right_of_slot.get((i, j)) is not None]
    n_right = sum(1 for (i, j) in gold_slots if right_of_slot[(i, j)])
    n_facs_of_row = {i: len(vs[i]["factors"]) for i in range(n)}
    log(f"[sine-shuffle] {len(gold_slots)} gold slots ({n_right} right, {len(gold_slots)-n_right} wrong); "
        f"owner resolution: {n_resolved} resolved, {n_unresolved} unresolved")

    # ---- the grouping factor + effective k, per BLOCK x breath x right/wrong ----
    group_rows = {}
    k_eff_rows = {}
    for (blk, kb) in attn:
        A = attn[(blk, kb)]
        for (i, j) in gold_slots:
            right = right_of_slot[(i, j)]
            row_av = A[i, j]
            own = owner_of_slot[(i, j)]
            same_m, diff_m = [], []
            nfacs = n_facs_of_row[i]
            for k in range(min(H.L_FAC, A.shape[-1])):
                if k == j or k >= nfacs:
                    continue
                ok2 = owner_of_slot.get((i, k))
                if ok2 is None:
                    continue
                (same_m if ok2 == own else diff_m).append(row_av[k])
            ms = float(np.mean(same_m)) if same_m else float("nan")
            md = float(np.mean(diff_m)) if diff_m else float("nan")
            group_rows.setdefault((blk, kb, right), []).append((ms, md))
            p_ = row_av[row_av > 1e-12]
            ent = float(-np.sum(p_ * np.log(p_))) if p_.size else 0.0
            k_eff_rows.setdefault((blk, kb, right), []).append(float(np.exp(ent)))

    lines = []
    L = lines.append
    L("=" * 92)
    L(f"THE MECHANISM READ ON {TAG} -- THE SINE BREATH's depth_attn_k{{i}} taps (2026-10-10)")
    L("=" * 92)
    L(f"ALG_SINE={H.ALG_SINE} ALG_SINE_SKIP={H.ALG_SINE_SKIP} widths={H._SINE_WIDTHS} trough_i={H._SINE_TROUGH_I}")
    L(f"gold slots: {len(gold_slots)} (right={n_right}, wrong={len(gold_slots)-n_right})")
    L("")
    L("block  breath  split    n    same      diff      ratio(same/diff)   mean-k")
    bar_pass = {}
    for blk in blocks:
        for kb in breaths:
            for right, label in ((None, "all"), (True, "right"), (False, "wrong")):
                if right is None:
                    rows = [r for (b_, k_, rt), rs in group_rows.items() if b_ == blk and k_ == kb for r in rs]
                    keff = [r for (b_, k_, rt), rs in k_eff_rows.items() if b_ == blk and k_ == kb for r in rs]
                else:
                    rows = group_rows.get((blk, kb, right), [])
                    keff = k_eff_rows.get((blk, kb, right), [])
                if not rows:
                    continue
                arr = np.array(rows)
                ms, md = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])
                ratio = ms / (md + 1e-12)
                mk_ = float(np.mean(keff)) if keff else float("nan")
                L(f"  {blk}    b{kb}    {label:5s}  {len(rows):4d}  {ms:.5f}  {md:.5f}    {ratio:7.3f}         {mk_:6.2f}")
                if right is None:
                    bar_pass.setdefault(blk, []).append((kb, ratio))

    L(""); L("-" * 92)
    L(f"BAR (registered): k0's owner ratio >= 1.5 (whichever block is index 0 this arm)")
    if 0 in bar_pass:
        ratios0 = [r for (kb, r) in bar_pass[0]]
        L(f"  block 0 ratio by breath: {[round(r, 3) for r in ratios0]}  mean={np.nanmean(ratios0):.3f}  "
          f"-> {'PASS' if np.nanmean(ratios0) >= 1.5 else 'FAIL (flat across widths = the crystal carries no ownership)'}")
    for blk in blocks:
        ratios = [r for (kb, r) in bar_pass.get(blk, [])]
        trough_flag = "  <= TROUGH (the crystal)" if blk == H._SINE_TROUGH_I else ""
        L(f"  block {blk} mean ratio across breaths: {np.nanmean(ratios):.3f}{trough_flag}")
    L("-" * 92)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    log(f"[sine-shuffle] wrote {OUT_TXT} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
