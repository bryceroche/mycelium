"""shuffle_read.py -- THE SHUFFLE READ ON SRK_241 (2026-10-10, ported from branch
replay's scripts/shuffle_read.py [4feab380, 2026-10-08] onto gen-weights for the
sorting-room bodies).

Ported because: branch replay (worktree mycelium-wt8) forked off gen-weights at
6079b855 and was never merged forward; its phase1_algebra_head.py lacks every organ
built since (the sorting room / ALG_SORT, the owner-key shuffle / ALG_SORT_KEY among
them), so the ORIGINAL script cannot run against SR_241/SRK_241 at all. This is the
same read (same two _CENSUS taps: "mixer_attn1" the single-head mixer's real
post-mask attention, "mixer_attn_heads" the FED multi-head twin's per-head attention)
re-added at the identical call sites in gen-weights' breath_step (dark unless
_CENSUS is armed -- zero behavior change when it is not; the mixer's own softmax
output is bound to a new name and reused, not recomputed).

NEW for this read (THE KEY'S OWN GROUPING, independent of the gain): a third tap,
"sort_tok_key" (kb=-1, once per forward() call, not per breath) -- the sorting
room's raw per-TOKEN 64-d owner key, `waist @ sort_key_w + sort_key_b`, captured
BEFORE fat_cur turns it into a per-slot expectation and before sort_key_gain scales
anything into the mixer's logits. The ledger's "gain unread" (SRK_241: sort_key_gain
drifted to -0.021, near its zero-at-birth init) says the GAIN never used this key --
it says nothing about whether the key itself, geometrically, separates owners. This
script answers that separately: for every GIVEN-numeral token governor_census.py can
bind and resolve an owner for, the cosine between its sort_tok_key vector and every
other such token's vector in the same row, split same-owner vs different-owner.

BODY is a required positional argument (TAG, e.g. SR_241 or SRK_241) -- this reader
takes the body as an argument per the delegated task's instruction. Every port the
arm's own env threads (ALG_SORT, ALG_SORT_KEY, and everything else SURF8/the base
family set) must be exported by the CALLER before invoking this script (the
rack_chain_SR.sh / rack_chain_SRK.sh convention: $FAM $SURF8 $SRX, SRX differing by
body) -- this script does not default ALG_SORT or ALG_SORT_KEY itself, so a missing
export fails loud (SR_241 has no owner-key shuffle; running it without ALG_SORT_KEY
unset would silently skip the key census, which is correct for the control).

Fixture: .cache/wild_admitted_holdout.jsonl, first N_ROWS=64 rows (PMS8_241's own
64-row convention, for direct comparability with .cache/shuffle_read_PMS8_241.txt).
right/wrong: .cache/ps_legal_wild_<TAG>.npz (same convention as resonance_read.py /
replay_gate_check.py / the original shuffle_read.py).

DEV defaults to CPU (setdefault -- a caller exporting DEV=PCI+AMD under
`flock .cache/gpu.lock` wins); this script never touches .cache/gpu.lock itself.
Writes ONLY .cache/shuffle_SRK_<TAG>.txt -- never a banked artifact.
"""
import os
import sys
import time
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")

import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: shuffle_read.py <TAG> (e.g. SR_241 or SRK_241); "
                      "caller must export the arm's own env first (FAM+SURF8+W+SRX)")
TAG = sys.argv[1]
N_ROWS = 64
CKPT = f".cache/sharp_{TAG}.safetensors"
PS_LEGAL_PATH = f".cache/ps_legal_wild_{TAG}.npz"
OUT_TXT = f".cache/shuffle_SRK_{TAG}_mixer.txt"   # separate from shuffle_srk_keycensus.py's
                                                    # .cache/shuffle_SRK_<TAG>.txt (part e) --
                                                    # same .cache (symlinked), must not collide

# ---------------------------------------------------------------------------
# THE BASE FAMILY (unconditional, matching the original script's convention) --
# ALG_SORT / ALG_SORT_KEY and anything SURF8 overrides are NOT in this dict; the
# caller's env (rack_chain_SR.sh / rack_chain_SRK.sh's $FAM $SURF8 $SRX) wins there
# by simply never being touched here.
# ---------------------------------------------------------------------------
_FAM = {
    "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
    "BIND_CODES": ".cache/bindbus_codes512r.npz",   # SURF8's role bank (SURF8's own default)
}
for k, v in _FAM.items():
    os.environ.setdefault(k, v)


def main():
    P = []
    def log(s):
        print(s, flush=True); P.append(s)

    t0 = time.time()
    log(f"[shuffle] TAG={TAG} DEV={os.environ.get('DEV')} ALG_SORT={os.environ.get('ALG_SORT','0')} "
        f"ALG_SORT_KEY={os.environ.get('ALG_SORT_KEY','0')} ckpt={CKPT}")
    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    from tinygrad import Tensor, dtypes
    import spacy
    import governor_census as GC
    from tokenizers import Tokenizer

    KEYCENSUS = bool(int(os.environ.get("ALG_SORT_KEY", "0")))

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
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
    _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    fb = H.alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
    fact_t = Tensor(fb, dtype=dtypes.float)

    # ---- THE CENSUS: one forward pass, _CENSUS armed (eyes_autopsy.py's precedent) ----
    H._CENSUS = []
    o = H.forward(p, ts, tk, se, slot_mask=slot_mask, fact_buf=fact_t)
    census = H._CENSUS
    H._CENSUS = None
    log(f"[shuffle] census forward done ({time.time() - t0:.0f}s so far); {len(census)} tap entries")

    attn1 = {}; attn_heads = {}; sort_tok_key = None
    for kb, tag, arr in census:
        if tag == "mixer_attn1":
            attn1[kb] = arr          # (B, L_TOT, L_TOT)
        elif tag == "mixer_attn_heads":
            attn_heads[kb] = arr     # (B, MX_HEADS, L_TOT, L_TOT)
        elif tag == "sort_tok_key":
            sort_tok_key = arr       # (B, T_ALG, 64)
    K_B = int(os.environ.get("ALG_BREATH", "7"))
    breaths = sorted(attn1)
    log(f"[shuffle] mixer_attn1 captured at breaths {breaths}; mixer_attn_heads at {sorted(attn_heads)}; "
        f"sort_tok_key {'captured ' + str(sort_tok_key.shape) if sort_tok_key is not None else 'ABSENT (ALG_SORT_KEY unset or no sort_key_w)'}")
    if KEYCENSUS and sort_tok_key is None:
        # wt19/shuffletap carries only the mixer_attn1/mixer_attn_heads taps (parts a/b/d);
        # part (e), the key's own grouping, was already answered in the main checkout via
        # scripts/shuffle_srk_keycensus.py (a standalone replication, zero head edits) --
        # .cache/shuffle_SRK_SRK_241.txt. Downgrade to a note rather than hard-error so the
        # mixer-attention parts still run with ALG_SORT_KEY=1 left on (faithful to the real
        # trained body's forward pass -- turning it off here would read a DIFFERENT computation).
        log("[shuffle] NOTE: ALG_SORT_KEY=1 but no sort_tok_key tap on this branch -- part (e) "
            "skipped here (see .cache/shuffle_SRK_<TAG>.txt from scripts/shuffle_srk_keycensus.py "
            "in the main checkout); parts (a)/(b)/(d) below are unaffected.")
        KEYCENSUS = False

    # ---- THE GOVERNOR CENSUS, reused by import ----
    NLP = spacy.load("en_core_web_sm")
    tok_hf = Tokenizer.from_file(H.TOKENIZER_JSON)

    ps = np.load(PS_LEGAL_PATH)
    ok_lookup = {(int(r), int(c)): bool(o_) for r, c, o_ in zip(ps["rows"], ps["slots"], ps["ok"])}

    owner_of_slot = {}     # (i, j) -> owner label string
    kind_of_slot = {}      # (i, j) -> "given" / "rel"
    args_of_slot = {}      # (i, j) -> list of arg var indices (relation slots only)
    right_of_slot = {}     # (i, j) -> bool or None

    # per-row given-var -> (tok_idx_in_hf_encoding, owner_label), for the key census
    given_tok_of_row = {}

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
        tok_of_var = {}       # var -> spaCy token (for the key census's offset lookup)
        for var, sp_tok in bound.items():
            if sp_tok is None:
                owner_of_var[var] = "none"; n_unresolved += 1; continue
            tok_of_var[var] = sp_tok
            ch = GC.governor_chain(doc, sent_list, sp_tok)
            if ch is None:
                owner_of_var[var] = "none"; n_unresolved += 1
            else:
                owner_of_var[var] = ch["owner_key"]; n_resolved += 1

        if KEYCENSUS:
            enc = tok_hf.encode(text)
            offs = list(enc.offsets)[:H.T_ALG]
            entry = []
            for var, sp_tok in tok_of_var.items():
                own = owner_of_var.get(var, "none")
                if own == "none":
                    continue
                c0, c1 = sp_tok.idx, sp_tok.idx + len(sp_tok.text)
                best_k, best_ov = None, 0
                for k2, (o0_, o1_) in enumerate(offs):
                    if o1_ <= o0_:          # special token, zero-width offset
                        continue
                    ov = min(c1, o1_) - max(c0, o0_)
                    if ov > best_ov:
                        best_ov, best_k = ov, k2
                if best_k is not None:
                    entry.append((best_k, own))
            given_tok_of_row[i] = entry

        for j, fac in enumerate(facs):
            if j >= H.L_FAC:
                continue
            right_of_slot[(i, j)] = ok_lookup.get((i, j))
            if fac["ftype"] == "given":
                kind_of_slot[(i, j)] = "given"
                owner_of_slot[(i, j)] = owner_of_var.get(fac["var"], "none")
            elif fac["ftype"] == "rel":
                kind_of_slot[(i, j)] = "rel"
                args = list(fac["args"])
                args_of_slot[(i, j)] = args
                arg_owners = []
                for a in args:
                    if a in given_vars:
                        arg_owners.append(owner_of_var.get(a, "none"))
                    else:
                        arg_owners.append(None)
                if len(arg_owners) >= 2 and all(o_ is not None and o_ != "none" for o_ in arg_owners) and len(set(arg_owners)) == 1:
                    owner_of_slot[(i, j)] = arg_owners[0]
                else:
                    owner_of_slot[(i, j)] = "mixed"
            else:
                kind_of_slot[(i, j)] = fac["ftype"]
                owner_of_slot[(i, j)] = "none"
        if i % 16 == 0:
            log(f"[shuffle] governor census row {i}/{n} ({time.time() - t0:.0f}s so far)")

    gold_slots = [(i, j) for (i, j) in owner_of_slot if right_of_slot.get((i, j)) is not None]
    n_right = sum(1 for (i, j) in gold_slots if right_of_slot[(i, j)])
    log(f"[shuffle] {len(gold_slots)} gold slots ({n_right} right, {len(gold_slots)-n_right} wrong); "
        f"owner resolution: {n_resolved} resolved, {n_unresolved} unresolved")
    owner_counts = collections.Counter(owner_of_slot[(i, j)] for (i, j) in gold_slots)
    log(f"[shuffle] owner label distribution (top 8): {owner_counts.most_common(8)}")

    n_facs_of_row = {i: len(vs[i]["factors"]) for i in range(n)}

    # ================================================================================
    # (a)/(b): the grouping factor and effective k, per breath, right vs wrong
    # ================================================================================
    group_rows = {}
    k_eff_rows = {}
    for kb in breaths:
        A = attn1[kb]
        for (i, j) in gold_slots:
            right = right_of_slot[(i, j)]
            row_av = A[i, j]
            own = owner_of_slot[(i, j)]
            same_m, diff_m, empty_m = [], [], []
            nfacs = n_facs_of_row[i]
            for k in range(H.L_FAC):
                if k == j:
                    continue
                if k < nfacs:
                    ok2 = owner_of_slot.get((i, k))
                    if ok2 is None:
                        empty_m.append(row_av[k]); continue
                    (same_m if ok2 == own else diff_m).append(row_av[k])
                else:
                    empty_m.append(row_av[k])
            for k in range(H.L_FAC, A.shape[-1]):
                empty_m.append(row_av[k])
            ms = float(np.mean(same_m)) if same_m else float("nan")
            md = float(np.mean(diff_m)) if diff_m else float("nan")
            me = float(np.mean(empty_m)) if empty_m else float("nan")
            group_rows.setdefault((kb, right), []).append((ms, md, me))
            p_ = row_av[row_av > 1e-12]
            ent = float(-np.sum(p_ * np.log(p_)))
            k_eff_rows.setdefault((kb, right), []).append(float(np.exp(ent)))
    log(f"[shuffle] (a)/(b) aggregated ({time.time() - t0:.0f}s so far)")

    # ================================================================================
    # (d): the binding read -- gold-arg attention vs other-slot attention, relation slots only
    # ================================================================================
    bind_rows = {}
    rel_slots = [(i, j) for (i, j) in gold_slots if kind_of_slot.get((i, j)) == "rel"]
    for kb in breaths:
        A = attn1[kb]
        for (i, j) in rel_slots:
            right = right_of_slot[(i, j)]
            row_av = A[i, j]
            args = args_of_slot.get((i, j), [])
            arg_m, other_m = [], []
            nfacs = n_facs_of_row[i]
            for k in range(nfacs):
                if k == j:
                    continue
                (arg_m if k in args else other_m).append(row_av[k])
            ma = float(np.mean(arg_m)) if arg_m else float("nan")
            mo = float(np.mean(other_m)) if other_m else float("nan")
            bind_rows.setdefault((kb, right), []).append((ma, mo))
    log(f"[shuffle] (d) aggregated ({time.time() - t0:.0f}s so far); {len(rel_slots)} relation gold slots")

    # ================================================================================
    # (e) THE KEY'S OWN GROUPING (SRK_241 only, gated on ALG_SORT_KEY): cosine(sort_tok_key_i,
    # sort_tok_key_j) between GIVEN-numeral tokens, same-owner vs different-owner, independent
    # of sort_key_gain (never applied here) and of the mixer's attention (never used here).
    # ================================================================================
    key_same = []; key_diff = []
    if KEYCENSUS:
        for i in range(n):
            entry = given_tok_of_row.get(i, [])
            if len(entry) < 2:
                continue
            vecs = sort_tok_key[i]   # (T_ALG, 64)
            norm = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12)
            for a in range(len(entry)):
                ka, oa = entry[a]
                for b in range(a + 1, len(entry)):
                    kb_, ob = entry[b]
                    cos = float(norm[ka] @ norm[kb_])
                    (key_same if oa == ob else key_diff).append(cos)
        log(f"[shuffle] (e) key census: {len(key_same)} same-owner pairs, {len(key_diff)} diff-owner pairs "
            f"across {sum(1 for i in range(n) if len(given_tok_of_row.get(i, [])) >= 2)} rows with >=2 resolved given tokens")

    # ================================================================================
    # write the report
    # ================================================================================
    lines = []
    L = lines.append
    L("=" * 92)
    L(f"THE SHUFFLE READ ON {TAG} -- wild, {n} rows, single-head mixer attention (2026-10-10)")
    L("=" * 92)
    L(f"ALG_SORT={os.environ.get('ALG_SORT','0')} ALG_SORT_KEY={os.environ.get('ALG_SORT_KEY','0')} ckpt={CKPT}")
    L(f"gold slots: {len(gold_slots)} (right={n_right}, wrong={len(gold_slots)-n_right}); "
      f"owner resolution: {n_resolved} resolved / {n_unresolved} unresolved")
    L(f"owner label distribution across gold slots (top 8): {owner_counts.most_common(8)}")
    L("")
    L("-" * 92)
    L("(a) GROUPING: mean mixer attention same-owner / different-owner / empty, by breath x right/wrong")
    L("-" * 92)
    L("  breath  split    n    same      diff      empty    ratio(same/diff)")
    for kb in breaths:
        for right, label in ((True, "right"), (False, "wrong")):
            rows = group_rows.get((kb, right), [])
            if not rows:
                continue
            arr = np.array(rows)
            ms, md, me = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1]), np.nanmean(arr[:, 2])
            L(f"    b{kb}   {label:5s}  {len(rows):4d}  {ms:.5f}  {md:.5f}  {me:.5f}    {ms/(md+1e-12):7.3f}")
    L("")
    L("-" * 92)
    L("(b) THE MIXER's EFFECTIVE k (exp(attention entropy), head-mean), by breath x right/wrong")
    L("-" * 92)
    L("  breath  split    n     mean-k    p25      p50      p75")
    for kb in breaths:
        for right, label in ((True, "right"), (False, "wrong")):
            rows = k_eff_rows.get((kb, right), [])
            if not rows:
                continue
            arr = np.array(rows)
            L(f"    b{kb}   {label:5s}  {len(rows):4d}  {np.mean(arr):7.3f}  "
              f"{np.percentile(arr,25):7.3f}  {np.percentile(arr,50):7.3f}  {np.percentile(arr,75):7.3f}")
    L("")
    L("-" * 92)
    L("(d) THE BINDING READ (relation slots only): mean attention to gold ARGS vs to OTHER gold slots")
    L("-" * 92)
    L("  breath  split    n    to-args   to-other   ratio")
    for kb in breaths:
        for right, label in ((True, "right"), (False, "wrong")):
            rows = bind_rows.get((kb, right), [])
            if not rows:
                continue
            arr = np.array(rows)
            ma, mo = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])
            L(f"    b{kb}   {label:5s}  {len(rows):4d}  {ma:.5f}  {mo:.5f}   {ma/(mo+1e-12):7.3f}")

    if KEYCENSUS:
        L(""); L("-" * 92)
        L("(e) THE KEY'S OWN GROUPING (gain-independent): cosine(sort_tok_key) between GIVEN-numeral tokens")
        L("-" * 92)
        sm = float(np.mean(key_same)) if key_same else float("nan")
        dm = float(np.mean(key_diff)) if key_diff else float("nan")
        ss = float(np.std(key_same)) if key_same else float("nan")
        ds = float(np.std(key_diff)) if key_diff else float("nan")
        L(f"  same-owner  n={len(key_same):5d}  mean cos={sm:+.4f}  std={ss:.4f}")
        L(f"  diff-owner  n={len(key_diff):5d}  mean cos={dm:+.4f}  std={ds:.4f}")
        L(f"  gap (same - diff) = {sm - dm:+.4f}")

    L(""); L("-" * 92); L("THE SIX-LINE READING"); L("-" * 92)
    def _ratio_series(right=None):
        out = []
        for kb in breaths:
            if right is None:
                rows = group_rows.get((kb, True), []) + group_rows.get((kb, False), [])
            else:
                rows = group_rows.get((kb, right), [])
            if not rows:
                out.append(float("nan")); continue
            arr = np.array(rows)
            ms, md = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])
            out.append(ms / (md + 1e-12))
        return out
    r_all = _ratio_series(None); r_right = _ratio_series(True); r_wrong = _ratio_series(False)
    L(f"1. GROUPING FACTOR (same/diff) by breath, all gold slots: " + " ".join(f"{x:.2f}" for x in r_all))
    L(f"2. RIGHT vs WRONG grouping factor by breath: right " + " ".join(f"{x:.2f}" for x in r_right)
      + "; wrong " + " ".join(f"{x:.2f}" for x in r_wrong))
    L(f"3. TREND: {'rising' if r_all[-1] > r_all[0] + 0.1 else ('falling' if r_all[-1] < r_all[0] - 0.1 else 'flat')} "
      f"across breaths {breaths[0]}..{breaths[-1]} (b{breaths[0]}={r_all[0]:.2f} -> b{breaths[-1]}={r_all[-1]:.2f})")
    k_all = []
    for kb in breaths:
        rows = k_eff_rows.get((kb, True), []) + k_eff_rows.get((kb, False), [])
        k_all.append(float(np.mean(rows)) if rows else float("nan"))
    L(f"4. EFFECTIVE k by breath: " + " ".join(f"{x:.2f}" for x in k_all)
      + f" (L_TOT={H.L_TOT}; leader~1 / shoal~7 / mush~{H.L_TOT})")
    b_all = []
    for kb in breaths:
        rows = bind_rows.get((kb, True), []) + bind_rows.get((kb, False), [])
        if rows:
            arr = np.array(rows)
            b_all.append(float(np.nanmean(arr[:, 0]) / (np.nanmean(arr[:, 1]) + 1e-12)))
        else:
            b_all.append(float("nan"))
    L(f"5. BINDING (args-vs-other ratio) by breath: " + " ".join(f"{x:.2f}" for x in b_all))
    verdict = ("LATENT (grouping factor > 1.5)" if (not np.isnan(r_all[-1]) and r_all[-1] > 1.5)
               else ("ABSENT (~1)" if (not np.isnan(r_all[-1]) and 0.67 < r_all[-1] < 1.5)
                     else "THE WRONG WAY (< 1: attends MORE to different-owner slots)"))
    L(f"6. VERDICT: the shuffle is {verdict} at the last captured breath (b{breaths[-1]}, ratio={r_all[-1]:.2f}); "
      f"right-vs-wrong gap at b{breaths[-1]} = {r_right[-1]-r_wrong[-1]:+.2f}.")
    if KEYCENSUS:
        sm = float(np.mean(key_same)) if key_same else float("nan")
        dm = float(np.mean(key_diff)) if key_diff else float("nan")
        key_verdict = ("THE KEY CARRIES OWNERSHIP" if (sm - dm) > 0.05
                        else "THE KEY DOES NOT CARRY OWNERSHIP (gap <= 0.05, noise-level)")
        L(f"7. THE KEY CENSUS: same-owner cos {sm:+.4f} vs diff-owner cos {dm:+.4f} "
          f"(gap {sm-dm:+.4f}) -- {key_verdict}.")

    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    log(f"[shuffle] wrote {OUT_TXT} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
