"""shuffle_srk_keycensus.py -- THE SHUFFLE READ ON SRK_241, part (e) ONLY: THE KEY'S OWN
GROUPING, independent of the gain (2026-10-10, gen-weights, CPU, zero head edits).

Why this script exists instead of a _CENSUS tap: the natural place to capture the sorting
room's raw per-token 64-d owner key (`waist @ sort_key_w + sort_key_b`, computed in
phase1_algebra_head.forward() right after `_sort_room()`, before fat_cur turns it into a
per-slot expectation and before sort_key_gain scales anything into the mixer) is a new dark
_CENSUS tap in forward() -- but editing the live head mid-campaign (with the hourglass queue
behind it) is out of bounds for this task; see docs/phase1_skeleton_spec.md 2026-10-10 "THE
SHUFFLE READ ON SRK_241" / .cache/shuffle_SRK_head_edit.patch (the reverted tap, saved for a
future worktree port that ALSO wants the mixer_attn1/mixer_attn_heads taps for parts a/b/d).

Instead, this script REPLICATES forward()'s own pre-bank prefix as a standalone read (the
membrane_rack.py precedent: "copy the minimal cycle" rather than touch the shared function) --
every piece reused by IMPORT, unedited:
  build_params, _sort_room (plain, unedited top-level function), load_alg, FED_WAIST,
  ALG_SORT, ALG_SORT_HG, ALG_SORT_KEY, T_ALG, L_FAC, TOKENIZER_JSON.
The only code copied verbatim (read off phase1_algebra_head.py's own forward(), lines
~7642-7705 as of this commit) is the waist-formation prefix:
    waist = (trunk @ waist_w + waist_b).gelu() + sent_emb[sent]
    if FED_WAIST: waist = waist + FED-MLP(waist)        # ALG_FED=1 in this family
    if ALG_T1: ...                                       # NOT set in this family -- skipped
    if ALG_SORT: waist = _sort_room(p, waist, tokmask, B) # ALG_SORT_HG NOT set -- plain form
    sort_tok_key = waist @ sort_key_w + sort_key_b        # ALG_SORT_KEY only
No HUD (ALG_HUD unset in this family), no T1 (ALG_T1 unset), no hourglass (ALG_SORT_HG unset)
-- all three branches are no-ops here and are asserted off below rather than silently skipped.

OWNERS: identical method to shuffle_read.py's governor-census reuse (bind_given_occurrences +
governor_chain, by import, no reimplementation) -- but at TOKEN granularity, not slot
granularity: each GIVEN var's bound spaCy token's char span is aligned to the SAME tokenizer's
offsets (tokenizers.Tokenizer.from_file(TOKENIZER_JSON), the exact tokenizer load_alg's own
precompute used) to find its token index into `waist`'s T_ALG axis.

MEASURE: cosine(sort_tok_key[tok_a], sort_tok_key[tok_b]) for every pair of GIVEN-numeral
tokens in the same row with a resolved owner, split same-owner vs different-owner -- plus an
in-row OWNER-LABEL PERMUTATION NULL (200 reps, fixed seed) as the control this read's own
script can supply (SR_241 has no sort_key_w at all -- ALG_SORT_KEY was never part of that arm
-- so there is no second body to use as a between-body control for this specific tap; the
permutation null is the within-body control: "is the same/diff gap bigger than relabeling
owners at random would produce").

Usage: env <FAM+SURF8+SRX, ALG_SORT_KEY=1 required> .venv/bin/python3 scripts/shuffle_srk_keycensus.py <TAG>
DEV defaults to CPU (setdefault); never touches .cache/gpu.lock. Writes ONLY
.cache/shuffle_SRK_<TAG>.txt -- never a banked artifact.
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")

import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: shuffle_srk_keycensus.py <TAG> (e.g. SRK_241); "
                      "caller must export the arm's own env first (FAM+SURF8+W+SRX, "
                      "ALG_SORT_KEY=1 required -- this script only does the key census)")
TAG = sys.argv[1]
N_ROWS = 64
SEED = 1234
N_PERM = 200
CKPT = f".cache/sharp_{TAG}.safetensors"
OUT_TXT = f".cache/shuffle_SRK_{TAG}.txt"

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
    "BIND_CODES": ".cache/bindbus_codes512r.npz",
}
for k, v in _FAM.items():
    os.environ.setdefault(k, v)


def main():
    P = []
    def log(s):
        print(s, flush=True); P.append(s)

    t0 = time.time()
    assert int(os.environ.get("ALG_SORT_KEY", "0")), \
        "ALG_SORT_KEY=1 must be exported by the caller -- this script does not default it"
    assert int(os.environ.get("ALG_SORT", "0")), "ALG_SORT_KEY needs ALG_SORT>0"
    assert not int(os.environ.get("ALG_HUD", "0")), "this script's copied waist prefix omits HUD -- do not run it with ALG_HUD=1"
    assert not int(os.environ.get("ALG_T1", "0")), "this script's copied waist prefix omits T1 -- do not run it with ALG_T1=1"
    assert not int(os.environ.get("ALG_SORT_HG", "0")), "this script's copied waist prefix omits the hourglass form -- do not run it with ALG_SORT_HG set"
    log(f"[keycensus] TAG={TAG} DEV={os.environ.get('DEV')} ALG_SORT={os.environ.get('ALG_SORT')} "
        f"ALG_SORT_KEY={os.environ.get('ALG_SORT_KEY')} ckpt={CKPT}")

    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    from tinygrad import Tensor, dtypes
    import spacy
    import governor_census as GC
    from tokenizers import Tokenizer

    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert "sort_key_w" in sd and "sort_key_b" in sd, \
        f"{CKPT} has no sort_key_w/sort_key_b -- this arm never trained ALG_SORT_KEY"
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    gain_val = float(p["sort_key_gain"].numpy()[0])
    log(f"[keycensus] sort_key_gain (checkpoint value, FYI only -- unused below) = {gain_val:+.5f}")

    vs, vst_, vtk, vg, vse = H.load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    trunk = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tokmask = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    sent = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    B = n

    # ---- forward()'s own waist-formation prefix, copied verbatim (membrane_rack.py's own
    # "copy the minimal cycle" precedent) -- NOT a call into forward() itself, no head edit ----
    if trunk.dtype != dtypes.float:
        trunk = trunk.cast(dtypes.float)
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]
    if H.FED_WAIST and "fed_w2b" in p:
        waist = waist + ((waist @ p["fed_w2a"] + p["fed_w2a_b"]).gelu()
                         @ p["fed_w2b"] + p["fed_w2b_b"])
    assert H.ALG_SORT and "sort0_wq" in p
    waist = H._sort_room(p, waist, tokmask, B)
    sort_tok_key = (waist @ p["sort_key_w"] + p["sort_key_b"]).realize().numpy()   # (B, T_ALG, 64)
    log(f"[keycensus] sort_tok_key computed: shape={sort_tok_key.shape} ({time.time()-t0:.0f}s so far)")

    # ---- THE GOVERNOR CENSUS, reused by import ----
    NLP = spacy.load("en_core_web_sm")
    tok_hf = Tokenizer.from_file(H.TOKENIZER_JSON)

    given_tok_of_row = {}
    n_resolved = n_unresolved = 0
    for i in range(n):
        row = vs[i]
        facs = row["factors"]; text = row["text"]
        doc = NLP(text); sent_list = list(doc.sents)
        cands = GC.numeral_candidates(doc)
        given_list = [(f["var"], f["value"]) for f in facs if f["ftype"] == "given"]
        bound = GC.bind_given_occurrences(doc, given_list, cands)
        enc = tok_hf.encode(text)
        offs = list(enc.offsets)[:H.T_ALG]
        entry = []
        for var, sp_tok in bound.items():
            if sp_tok is None:
                n_unresolved += 1; continue
            ch = GC.governor_chain(doc, sent_list, sp_tok)
            if ch is None:
                n_unresolved += 1; continue
            own = ch["owner_key"]; n_resolved += 1
            c0, c1 = sp_tok.idx, sp_tok.idx + len(sp_tok.text)
            best_k, best_ov = None, 0
            for k2, (o0_, o1_) in enumerate(offs):
                if o1_ <= o0_:
                    continue
                ov = min(c1, o1_) - max(c0, o0_)
                if ov > best_ov:
                    best_ov, best_k = ov, k2
            if best_k is not None:
                entry.append((best_k, own))
        given_tok_of_row[i] = entry
        if i % 16 == 0:
            log(f"[keycensus] governor census row {i}/{n} ({time.time()-t0:.0f}s so far)")

    n_multi_rows = sum(1 for i in range(n) if len(given_tok_of_row.get(i, [])) >= 2)
    log(f"[keycensus] owner resolution: {n_resolved} resolved / {n_unresolved} unresolved; "
        f"{n_multi_rows}/{n} rows have >=2 resolved given tokens")

    def cos_split(entries_by_row):
        same, diff = [], []
        for i, entry in entries_by_row.items():
            if len(entry) < 2:
                continue
            vecs = sort_tok_key[i]
            norm = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12)
            for a in range(len(entry)):
                ka, oa = entry[a]
                for b in range(a + 1, len(entry)):
                    kb_, ob = entry[b]
                    c = float(norm[ka] @ norm[kb_])
                    (same if oa == ob else diff).append(c)
        return same, diff

    key_same, key_diff = cos_split(given_tok_of_row)

    # ---- THE PERMUTATION NULL: shuffle owner labels WITHIN each row, N_PERM reps ----
    rng = np.random.default_rng(SEED)
    perm_gaps = []
    for _ in range(N_PERM):
        perm_rows = {}
        for i, entry in given_tok_of_row.items():
            if len(entry) < 2:
                perm_rows[i] = entry; continue
            toks = [k for k, _ in entry]
            owns = [o for _, o in entry]
            rng.shuffle(owns)
            perm_rows[i] = list(zip(toks, owns))
        ps, pd = cos_split(perm_rows)
        perm_gaps.append((np.mean(ps) if ps else float("nan")) - (np.mean(pd) if pd else float("nan")))
    perm_gaps = np.array(perm_gaps)

    sm = float(np.mean(key_same)) if key_same else float("nan")
    dm = float(np.mean(key_diff)) if key_diff else float("nan")
    ss = float(np.std(key_same)) if key_same else float("nan")
    ds = float(np.std(key_diff)) if key_diff else float("nan")
    gap = sm - dm
    perm_mean = float(np.nanmean(perm_gaps)); perm_std = float(np.nanstd(perm_gaps))
    z = (gap - perm_mean) / (perm_std + 1e-12)

    lines = []
    L = lines.append
    L("=" * 92)
    L(f"THE SHUFFLE READ ON {TAG} -- part (e) THE KEY'S OWN GROUPING (2026-10-10)")
    L("=" * 92)
    L(f"ckpt={CKPT}  sort_key_gain(checkpoint)={gain_val:+.5f}  (the gain is NOT used anywhere below --")
    L("this is the raw per-token key's cosine structure, independent of it)")
    L(f"rows={n}  owner resolution: {n_resolved} resolved / {n_unresolved} unresolved given-numeral tokens; "
      f"{n_multi_rows} rows have >=2 resolved tokens to pair")
    L("")
    L(f"same-owner  n={len(key_same):5d}  mean cos={sm:+.4f}  std={ss:.4f}")
    L(f"diff-owner  n={len(key_diff):5d}  mean cos={dm:+.4f}  std={ds:.4f}")
    L(f"gap (same - diff) = {gap:+.4f}")
    L(f"PERMUTATION NULL ({N_PERM} within-row owner-label reshuffles): gap mean={perm_mean:+.4f} std={perm_std:.4f}; "
      f"observed gap z = {z:+.2f}")
    L("")
    if abs(gap) <= 0.02 or abs(z) < 2.0:
        verdict = "THE KEY DOES NOT CARRY OWNERSHIP (gap noise-level and/or not distinguishable from the permutation null)"
    elif gap > 0:
        verdict = "THE KEY CARRIES OWNERSHIP (same-owner cosine exceeds different-owner, beyond the permutation null)"
    else:
        verdict = "THE KEY CARRIES OWNERSHIP THE WRONG WAY (different-owner cosine exceeds same-owner)"
    L(f"VERDICT: {verdict}")

    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    log(f"[keycensus] wrote {OUT_TXT} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
