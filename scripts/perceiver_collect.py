"""perceiver_collect.py -- THE LEARNED PERCEIVER v1, PART 1 (2026-10-06, "WORD GIVEN: THE LEARNED
PERCEIVER v1", docs/phase1_skeleton_spec.md 06:45; extended same day by Bryce's "change of diet"
message -- raw GSM8K word problems as a second source for the STOP head only).

HARD CONSTRAINTS (the mission brief's own words, never relaxed by anything below): the GPU is busy
with a training chain (RK_241, 48000 steps, .cache/gpu.lock held at launch time -- confirmed with
`flock -n`); this script NEVER takes .cache/gpu.lock and NEVER sets DEV=PCI+AMD; every forward pass
here runs DEV=CPU. scripts/phase1_algebra_head.py, mycelium/jit_read.py, scripts/loop_val.py,
scripts/chain_acc.py and scripts/membrane_scale.py are READ (imported) but never edited.

TWO SOURCES (--source):
  fixture (default) -- an ANNOTATED jsonl (the diet slice .cache/form_pm35c_slice1024_valid2.jsonl,
    775 rows, held out from training per THE DIET FIXTURE law, or the wild measurement holdout
    .cache/wild_admitted_holdout.jsonl, 311 rows, MEASURED NEVER TRAINED ON) loaded through
    phase1_algebra_head.load_alg("test") -- the staged states+gold pipeline every reader in this
    codebase uses (STATES_NPZ already exists for both: .cache/phase1_alg_states_pm35cslicevalid2.npz,
    .cache/phase1_alg_states_wildhold.npz -- confirmed on disk, no restaging needed). Gold (`vg`) is
    available, so the full feature set AND the per-slot COMMIT label (RIGHT at the final breath) are
    both written.
  gsm8k -- RAW GSM8K TRAIN-split word problems (.cache/gsm8k/.../train-00000-of-00001.parquet, 7473
    rows), deduplicated against every wild-holdout GSM8K row by BOTH gen.src_idx and exact text (302
    of the wild holdout's 311 rows carry gen.src=="gsm8k"; the test split is never touched -- it is a
    measurement set, not a diet). These rows carry NO annotated factor graph (no `factors`/`solution`
    field) -- only the dataset's own "#### <answer>" line -- so load_alg's staged-states pipeline
    cannot be used (it requires build_gold()'s factor-graph gold at precompute time). Instead this
    script embeds raw text through the frozen trunk ON THE FLY, batch by batch, using the SAME two
    functions do_precompute() uses (phase1_algebra_head.tokenize, mycelium.llama_loader's
    attach_llama_layers/load_llama_weights/_rms_norm), host cached once per process (CLAUDE.md S5).
    No gold means: the per-slot COMMIT label and the gold-dependent BAND feature are both N/A (-1)
    for every gsm8k row; the STOP label (judge passes AND the decoded value matches the GSM8K
    answer) and every OTHER feature (entropy, numeral share/mass, atlas cosines, leaf/root-band
    change, nl_certifier, parse-space count) are gold-FREE already (kindpred below reads the model's
    OWN predicted ftype, never gold -- adaptive_stop.py's own convention, inherited verbatim) and so
    apply identically to both sources.

REUSE, NOT REIMPLEMENTATION (the codebase's own standing convention; see adaptive_stop.py's
docstring for the precedent this script extends):
  - imported from scripts/adaptive_stop.py (built in parallel this same day, per the mission brief's
    instruction to import rather than rewrite): FAMILY_ENVS, _build_family_env, _solve_task,
    _uniqueness_task, _nlc_cert, _cos, _rack_sidecar_path.
  - imported from scripts/phase1_algebra_head.py: build_params, forward, load_alg, build_slot_masks,
    alt2_fact_buf, _decode_slots, K_VARS, L_FAC, N_DIG, T_ALG, H_TRUNK, _hier_band_dims, tokenize,
    sent_indices, TOKENIZER_JSON.
  - imported from scripts/membrane_scale.py (band-by-scale method, 2026-09-24): _digit_runs,
    _decode_tokens, _segment_ids, _band_arrays, BANDS, CONJ_CLAUSE, CONJ_MENTION_EXTRA,
    PUNCT_MENTION_EXTRA, VERBISH.
  - imported from mycelium/rulebook.py: legal_digit_logits, legal_values.
  - imported from mycelium/custody_gold.py: row_gold (fixture source only).
  - imported from scripts/beam_oracle.py (mask_row) and scripts/combined_oracle.py
    (value_variants, apply_value_variant) for THE PARSE-SPACE METER (Bryce, "CONCENTRATION",
    2026-10-05 20:31, extended to GSM8K by the 2026-10-06 diet-change message): SCOPED to the
    VALUES axis only (sinkhorn_claim's given-value reassignment candidates via combined_oracle's own
    helper) -- the ARGUMENTS and TYPE/OP axes of combined_oracle.py's full three-axis bound are NOT
    exercised here (stated, not silently dropped: this meter is registered as a feature/diagnostic,
    never a bar, and the VALUES axis alone already bounds the count at a handful of candidates per
    breath, keeping the extra solver load small enough for an overnight CPU run across ~2000 GSM8K
    rows x 7 breaths).
  - imported from mycelium/llama_loader.py (NOT edited, only imported): attach_llama_layers,
    load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm -- the frozen trunk, for the gsm8k source's
    on-the-fly embedding only (the fixture source never needs this -- its states are already staged).

THE DEDUP (gsm8k source): exclude_idx = {gen.src_idx for wild rows with gen.src=="gsm8k"} (302
values); exclude_text = {text.strip() for the same rows}. A GSM8K-train row is admitted to the pool
iff its index is NOT in exclude_idx AND its question text is NOT in exclude_text (the text check is
the belt-and-suspenders one -- src_idx's indexing convention into the train split is asserted, not
assumed, by cross-checking a sample of wild rows' own text against train[src_idx] at pool-build time;
any mismatch widens the exclusion to the text-only check and says so loudly, never silently).

THE KEY for a gsm8k row = the parquet's own "#### <int>" tail (comma/dollar stripped, int-parsed;
a non-integer tail refuses the row, matching custody_gold.row_gold's own hard-error convention for
pen rows, never a silent zero).

LABELS WRITTEN:
  right_final[i,j]  (fixture only, -1 for gsm8k) -- RIGHT at the final breath, reproducing loop_val's
    LV_LEGAL=num per-slot "ok" (read.py:447-476, imported logic re-implemented here verbatim against
    the SAME onp/vg fields loop_val reads, since loop_val.read() is a monolithic function with no
    smaller reusable piece to import -- ported, not copied-and-modified: f_pres/f_ftype/f_res and,
    branching on gold ftype, (f_op and f_args) or f_dig, the numeral mask applied first exactly as
    loop_val's own LV_LEGAL=num branch does).
  stop_kb_label[i]  (both sources) -- the first breath whose decode passes THE CONSISTENCY JUDGE
    (solved + unique, key never consulted by the judge itself) AND whose solved value matches the
    row's key; -1 if no breath qualifies. This is the SUPERVISED label the perceiver's STOP head is
    trained to predict -- distinct from stop_kb_judge (adaptive_stop.py's own judge-only rule, kept
    alongside for comparison; a key-blind quantity, usable as an inference-time POLICY, never as a
    training label by itself since it does not know whether the breath that passed the judge was
    actually RIGHT).

usage:
  DEV=CPU .venv/bin/python3 scripts/perceiver_collect.py .cache/sharp_PMS8_241.safetensors \\
      --tag PMS8_241 --source fixture --rows .cache/form_pm35c_slice1024_valid2.jsonl --limit 8
  DEV=CPU .venv/bin/python3 scripts/perceiver_collect.py .cache/sharp_PMS8_241.safetensors \\
      --tag PMS8_241 --source gsm8k --limit 500
"""
import os
import sys
import re
import json
import time
import pickle
import argparse
import collections

# THE HARD DOOR: DEV=CPU, always, before anything else imports tinygrad. A caller's own DEV=CPU in
# the environment is left alone; anything else (including an unset DEV, or a stray DEV=PCI+AMD) is
# overridden here -- this script is never the one that takes the GPU.
os.environ["DEV"] = "CPU"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np

import adaptive_stop as AS   # the per-breath decode + judge functions (imported, never rewritten)

GSM8K_TRAIN_PARQUET = (".cache/gsm8k/datasets--openai--gsm8k/snapshots/"
                       "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet")
WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"

# fixture path -> the states-file split name already staged on disk (twin_stamp_mint.py's
# DEFAULT_SPLITS table, verbatim mapping -- these two names' .npz/_states.npy already exist, see
# `ls .cache/phase1_alg_states_pm35cslicevalid2.npz .cache/phase1_alg_states_wildhold.npz`).
FIXTURE_NAMES = {
    ".cache/form_pm35c_slice1024_valid2.jsonl": "pm35cslicevalid2",
    WILD_JSONL: "wildhold",
}


def _build_family_env_cpu(tag, extra):
    """AS._build_family_env, but DEV/ALG_TEST/ALG_TEST_NAME must already be set by the caller (the
    CPU door above + the fixture override below) before this runs, since _build_family_env only ever
    os.environ.setdefault()s the family table's own values -- whatever the caller set first wins."""
    return AS._build_family_env(tag, extra)


# ===========================================================================================================
# THE BAND-BY-SCALE FEATURE (fixture source only; imported from membrane_scale.py, never reimplemented)
# ===========================================================================================================
import membrane_scale as MS   # module-level import only -- MS.collect()/MS.report() are never called


def _row_bands(tok, ids_row, tokmask_row, sent_row):
    """Per-row CLAUSE/MENTION segment ids, membrane_scale.py's own _segment_ids verbatim (imported)."""
    dec = MS._decode_tokens(tok, ids_row, len(ids_row))
    clause_id = MS._segment_ids(dec, tokmask_row, sent_row, MS.CONJ_CLAUSE)
    mention_id = MS._segment_ids(dec, tokmask_row, sent_row,
                                  MS.CONJ_MENTION_EXTRA | MS.PUNCT_MENTION_EXTRA | MS.VERBISH)
    runs = MS._digit_runs(tok, ids_row, len(ids_row))
    return dec, clause_id, mention_id, runs


def _given_slot_band(v, runs, mention_id, clause_id, sent_row, tokmask_row):
    """band[T] for ONE gold-given slot's value v (membrane_scale.py's per-slot method, factored out
    of its report() loop body here since that loop is not itself a function). -1 array (all no-match)
    if v has no textual occurrence."""
    matches = [(a, b, val) for a, b, val in runs if val == v]
    if not matches:
        return None
    occ_tok = sorted({t for a, b, val in matches for t in range(a, b)})
    occ_sent = {int(sent_row[t]) for t in occ_tok}
    occ_clause = {int(clause_id[t]) for t in occ_tok}
    occ_mention = {int(mention_id[t]) for t in occ_tok}
    return MS._band_arrays(None, occ_tok, mention_id, occ_mention, clause_id, occ_clause,
                            sent_row, occ_sent, tokmask_row)


# ===========================================================================================================
# THE PARSE-SPACE METER (VALUES axis only; combined_oracle.py's own helpers, imported verbatim)
# ===========================================================================================================
import beam_oracle as BO
import combined_oracle as CO


def _value_variant_tasks(task_key, raw_row, text, nd, val_max=4):
    """Build up to val_max candidate rows off ONE breath's RAW (pre-mask) heads for ONE row (the
    VALUES axis of combined_oracle's three-axis bound, scoped per this file's docstring), decode each,
    and return a list of (ps_task_id, q, key_placeholder, parse, gv, nvv) solver tasks -- `key` is
    filled in by the caller (the row's own key is identical across every variant of the same row/
    breath; the judge never needs it, only the solved VALUE does, so a placeholder of -1 here is safe
    -- _solve_task's `key` argument is unused inside the function, kept only for interface parity with
    _uniqueness_task's call sites elsewhere). Distinct-parse dedup by signature (args, so two variants
    landing on the identical graph are not double-counted)."""
    from phase1_algebra_head import _decode_slots as _DS
    masked = BO.mask_row(raw_row, text)
    gslots, variants = CO.value_variants(raw_row, text, nd, val_max)
    out = []
    seen_sigs = set()
    ti, tkb = task_key
    for vi, (label, values, prio) in enumerate(variants[:val_max]):
        cand = CO.apply_value_variant(masked, gslots, values, nd)
        parse = _DS(cand)
        if not parse:
            continue
        sig = tuple(sorted((f["ftype"], f.get("var"), f.get("op"), tuple(f.get("args", ())),
                            f.get("result"), f.get("value")) for f in parse))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        used = ([f.get("var") for f in parse if f["ftype"] == "given"]
                + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]])
        q_dummy = 0  # the meter counts SOLVED candidates, not the row's own query -- _solve_task's
                     # `q`/`key` args are irrelevant to that count (used only to pick a return VALUE,
                     # which this meter never reads), so a placeholder is safe here.
        nvv = max([1] + [v + 1 for v in used if v is not None])
        gvd = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
        out.append(((ti, tkb, vi), q_dummy, -1, parse, gvd, nvv))   # FLAT 3-tuple id (i, kb, vi) --
                                                                     # never nest task_key inside it
    return out


# ===========================================================================================================
# GSM8K POOL (dedup against the wild holdout; train split only; no network, local parquet)
# ===========================================================================================================

def build_gsm8k_pool(n_target):
    import pyarrow.parquet as pq
    wild = [json.loads(l) for l in open(WILD_JSONL)]
    wild_gsm = [r for r in wild if r.get("gen", {}).get("src") == "gsm8k"]
    exclude_idx = {int(r["gen"]["src_idx"]) for r in wild_gsm}
    exclude_text = {r["text"].strip() for r in wild_gsm}
    t = pq.read_table(GSM8K_TRAIN_PARQUET).to_pydict()
    questions, answers = t["question"], t["answer"]
    n_train = len(questions)
    # THE INDEXING ASSERTION: src_idx is claimed to index the TRAIN split -- verify on every wild
    # gsm8k row that train[src_idx] (when in range) matches its OWN text; any mismatch means src_idx's
    # convention is NOT "index into this exact train parquet" and the exclusion must fall back to
    # TEXT ONLY (still a valid, if weaker, dedup) -- loud, not silent.
    idx_verified = idx_mismatch = idx_oor = 0
    for r in wild_gsm:
        si = int(r["gen"]["src_idx"])
        if 0 <= si < n_train:
            if questions[si].strip() == r["text"].strip():
                idx_verified += 1
            else:
                idx_mismatch += 1
        else:
            idx_oor += 1
    idx_trustworthy = (idx_mismatch == 0 and idx_oor == 0 and idx_verified == len(wild_gsm))
    print(f"[gsm8k-pool] wild gsm8k rows={len(wild_gsm)} src_idx-verified={idx_verified} "
          f"mismatch={idx_mismatch} out-of-range={idx_oor} -- "
          f"{'src_idx trustworthy, using idx+text exclusion' if idx_trustworthy else 'src_idx NOT fully trustworthy, falling back to TEXT-ONLY exclusion'}",
          flush=True)
    pool = []
    n_excl_idx = n_excl_text = 0
    for i in range(n_train):
        qtext = questions[i].strip()
        if idx_trustworthy and i in exclude_idx:
            n_excl_idx += 1
            continue
        if qtext in exclude_text:
            n_excl_text += 1
            continue
        ans = str(answers[i])
        tail = ans.split("####")[-1].strip().replace("$", "").replace(",", "")
        if not re.fullmatch(r"-?\d+", tail):
            continue
        pool.append({"text": questions[i], "key": int(tail), "train_idx": i})
        if len(pool) >= n_target:
            break
    print(f"[gsm8k-pool] train rows={n_train} excluded(idx)={n_excl_idx} excluded(text)={n_excl_text} "
          f"admitted(this pass)={len(pool)} (target {n_target})", flush=True)
    return pool, dict(n_train=n_train, n_wild_gsm=len(wild_gsm), idx_trustworthy=idx_trustworthy,
                      n_excl_idx=n_excl_idx, n_excl_text=n_excl_text)


# ===========================================================================================================
# ON-THE-FLY TRUNK EMBEDDING (gsm8k source only; host cached once per process, CLAUDE.md S5)
# ===========================================================================================================
_TRUNK_HOST = {}


def _get_trunk_host():
    if "host" not in _TRUNK_HOST:
        from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                           LLAMA_3_2_1B_CFG, _rms_norm)
        import phase1_algebra_head as H

        class _Host:
            pass
        host = _Host()
        sd = load_llama_weights(os.path.join(H._ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
        attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
        del sd
        _TRUNK_HOST["host"] = host
        _TRUNK_HOST["_rms_norm"] = _rms_norm
        print("[perceiver-collect] trunk host loaded (CPU, cached for this process)", flush=True)
    return _TRUNK_HOST["host"], _TRUNK_HOST["_rms_norm"]


def _embed_raw_batch(ids_batch):
    """mirrors phase1_algebra_head.do_precompute()'s inner loop body verbatim (imported constants,
    not copied logic beyond the four lines that loop does) -- ids_batch: (B, T_ALG) int32 -> (B,
    T_ALG, H_TRUNK) float16, the SAME states a staged fixture's vst already holds."""
    from tinygrad import Tensor, dtypes
    host, rms = _get_trunk_host()
    x = host.llama_embed[Tensor(ids_batch, dtype=dtypes.int)]
    for layer in host.llama_layers:
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
    x = rms(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
    c = x.cast(dtypes.float).realize().numpy()
    assert np.isfinite(c).all()
    return c.astype(np.float16)


# ===========================================================================================================
# THE PER-SLOT "ok" LABEL -- loop_val.py's LV_LEGAL=num branch, ported (that function has no smaller
# reusable piece to import; verbatim field-by-field logic against loop_val.py:439-476)
# ===========================================================================================================

def _slot_ok(onp_row, j, vg_row, legal_digit_logits_fn, text):
    """onp_row: {"pres","ftype","op","args","res","dig",["dup"]} for ONE row at ONE breath (already
    .numpy()'d, un-batched -- i.e. onp_row[k] is heads_all[kb][k][bi]). vg_row: {k: vg[k][i] for k in
    vg} the SAME row's gold. Returns (ok: bool, f_pres, f_ftype, f_res, f_op_or_None, f_args_or_None,
    f_dig_or_None) -- same tuple shape loop_val computes, so a caller can also do LV_FIELDS-style
    per-field tallies if it wants them later without re-deriving anything."""
    dig = onp_row["dig"].copy()
    if vg_row["ftype"][j] != 0 and int(onp_row["ftype"][j].argmax()) != 0:
        fake = legal_digit_logits_fn(dig[j], text)
        if fake is not None:
            dig[j] = fake
    f_pres = bool(onp_row["pres"][j] > 0)
    f_ftype = int(onp_row["ftype"][j].argmax()) == vg_row["ftype"][j]
    f_res = int(onp_row["res"][j].argmax()) == vg_row["res"][j]
    ok = f_pres and f_ftype and f_res
    f_op = f_args = f_dig = None
    if vg_row["ftype"][j] == 0:
        gset = set(np.where(vg_row["args"][j] > .5)[0].tolist())
        f_op = int(onp_row["op"][j].argmax()) == vg_row["op"][j]
        if len(gset) == 1 and "dup" in onp_row:
            f_args = bool(onp_row["dup"][j] > 0) and int(np.argmax(onp_row["args"][j])) in gset
        else:
            top2 = set(np.argsort(-onp_row["args"][j])[:2].tolist())
            f_args = top2 == gset
        ok = ok and f_op and f_args
    else:
        f_dig = bool((dig[j].argmax(-1) == vg_row["digits"][j]).all())
        ok = ok and f_dig
    return ok, f_pres, f_ftype, f_res, f_op, f_args, f_dig


# ===========================================================================================================
# MAIN
# ===========================================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", choices=["fixture", "gsm8k"], default="fixture")
    ap.add_argument("--rows", default="", help="fixture source: the jsonl path (.cache/form_pm35c_slice1024_valid2.jsonl or .cache/wild_admitted_holdout.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="row cap (0 = all); for gsm8k, the pool target size too")
    ap.add_argument("--gsm8k-offset", type=int, default=0, help="gsm8k source: skip this many admitted pool rows before taking --limit (for chunked overnight runs)")
    ap.add_argument("--family-env", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("AS_WORKERS", "6")))
    ap.add_argument("--wall", type=float, default=float(os.environ.get("AS_WALL", "3")))
    ap.add_argument("--ps-max", type=int, default=4, help="parse-space meter: VALUES-axis candidates per breath, capped")
    ap.add_argument("--no-parse-space", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ["AS_WALL"] = str(args.wall)

    assert os.environ.get("DEV") == "CPU", "perceiver-collect: CPU-only, always -- the GPU is a training chain's"
    import subprocess
    locked = subprocess.run(["flock", "-n", ".cache/gpu.lock", "-c", "true"]).returncode != 0
    print(f"[perceiver-collect] .cache/gpu.lock currently {'HELD (a training chain)' if locked else 'free'} -- untouched either way (DEV=CPU)", flush=True)

    if args.source == "fixture":
        if not args.rows:
            raise SystemExit("perceiver-collect: --source fixture needs --rows <jsonl>")
        test_name = FIXTURE_NAMES.get(args.rows)
        if test_name is None:
            raise SystemExit(f"perceiver-collect: no staged states-file name known for {args.rows!r} "
                             f"-- add it to FIXTURE_NAMES (see .cache/phase1_alg_states_<name>.npz)")
        os.environ["ALG_TEST"] = args.rows
        os.environ["ALG_TEST_NAME"] = test_name
        fixture_tag = test_name
    else:
        fixture_tag = "gsm8kTRAIN"
        # the gsm8k path never calls load_alg -- ALG_TEST/_NAME are irrelevant to it, but still set to
        # something harmless (the wild path's own default) so FAMILY_ENVS' table entry is consistent
        # if any downstream helper reads it incidentally.
        os.environ.setdefault("ALG_TEST", WILD_JSONL)
        os.environ.setdefault("ALG_TEST_NAME", "wildhold")

    _build_family_env_cpu(args.tag, args.family_env)
    assert os.environ["DEV"] == "CPU"

    t0 = time.time()
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks, alt2_fact_buf,
                                     _decode_slots, K_VARS, L_FAC, N_DIG, T_ALG)
    from mycelium.rulebook import legal_digit_logits
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    K_B = int(os.environ.get("ALG_BREATH", "1"))
    assert K_B > 1, "perceiver-collect needs a breathing body (ALG_BREATH > 1)"
    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(np.intersect1d(CONTENT, clock_dims)) == 0
    C = len(CONTENT)
    ROOT_IN_C = np.isin(CONTENT, bands[0])
    BRANCH_IN_C = np.isin(CONTENT, bands[1])
    LEAF_IN_C = np.isin(CONTENT, bands[2])
    assert ROOT_IN_C.sum() == len(bands[0]) and LEAF_IN_C.sum() == len(bands[2])

    p = build_params(0)
    sd = safe_load(args.ckpt)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[perceiver-collect] tag={args.tag} source={args.source} ckpt={args.ckpt} K_B={K_B} "
          f"C={C} ({time.time()-t0:.0f}s)", flush=True)

    # -------------------------------------------------------------------------------------------
    # BUILD THE ROW LIST + PER-BATCH (ts, tk, se, text, vg_or_None, key_or_None) SOURCE
    # -------------------------------------------------------------------------------------------
    pool_meta = None
    if args.source == "fixture":
        vs, vst, vtk, vg, vse = load_alg("test")
        n_full = len(vs)
        n = min(n_full, args.limit) if args.limit else n_full
        texts_all = [vs[i]["text"] for i in range(n)]
        from mycelium.custody_gold import row_gold
        keys_all = []
        for i in range(n):
            try:
                keys_all.append(int(row_gold(vs[i])))
            except Exception:
                keys_all.append(None)
        has_gold = True
        tok_ids_all = None  # filled lazily below, per-batch via the tokenizer (band feature only)
    else:
        pool, pool_meta = build_gsm8k_pool(args.gsm8k_offset + (args.limit or 2000))
        pool = pool[args.gsm8k_offset:args.gsm8k_offset + args.limit] if args.limit else pool[args.gsm8k_offset:]
        n = len(pool)
        texts_all = [r["text"] for r in pool]
        keys_all = [r["key"] for r in pool]
        has_gold = False
        vg = None
        # tokenize the WHOLE pool at once via phase1_algebra_head.tokenize (imported, not reimplemented)
        import tempfile
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for t in texts_all:
            tf.write(json.dumps({"text": t}) + "\n")
        tf.close()
        _samples, _ids_all, _mask_all, _offsets_all = H.tokenize(tf.name)
        os.unlink(tf.name)
        _sent_all = np.stack([H.sent_indices(texts_all[i], _offsets_all[i], _mask_all[i]) for i in range(n)])
        vtk = _mask_all
        vse = _sent_all
        gsm8k_ids = _ids_all  # kept for the embedding step below

    print(f"[perceiver-collect] n={n} ({args.source})", flush=True)
    if n == 0:
        raise SystemExit("perceiver-collect: 0 rows selected -- nothing to do")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(H.TOKENIZER_JSON) if has_gold else None

    # -------------------------------------------------------------------------------------------
    # OUTPUT ARRAYS
    # -------------------------------------------------------------------------------------------
    ent = np.full((n, K_B, L_FAC), np.nan, np.float32)
    numarg = np.full((n, K_B, L_FAC), -1, np.int8)
    nummass = np.full((n, K_B, L_FAC), np.nan, np.float32)
    band = np.full((n, K_B, L_FAC), -1, np.int8)
    state_store = np.zeros((n, K_B, L_FAC, C), np.float32)
    kindpred = np.full((n, K_B, L_FAC), -1, np.int8)
    right_final = np.full((n, L_FAC), -1, np.int8)
    pres_gold_out = np.full((n, L_FAC), -1, np.int8)
    ftype_gold_out = np.full((n, L_FAC), -1, np.int8)
    q_out = np.full((n,), -1, np.int32)
    key_out = np.array([k if k is not None else -1 for k in keys_all], np.int64)

    row_info = {}
    tasks_by_id = {}
    ps_tasks_by_id = {}  # (i, kb, vi) -> solver task for the parse-space meter

    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        if has_gold:
            nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
            ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
            st_np = np.ascontiguousarray(vst[sl_p])
            tk_np = vtk[sl_p].astype(np.float32)
            se_np = vse[sl_p].astype(np.int32)
        else:
            nv = np.full(8, K_VARS)
            ma = np.zeros(8, np.int64)
            st_np = _embed_raw_batch(gsm8k_ids[sl_p])
            tk_np = vtk[sl_p].astype(np.float32)
            se_np = vse[sl_p].astype(np.int32)
        ts = Tensor(st_np, dtype=dtypes.half)
        tk = Tensor(tk_np, dtype=dtypes.float)
        se = Tensor(se_np, dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se_np)
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].numpy() for k in _ka}}
        fb = alt2_fact_buf(_oa, se_np, nv, ma)
        fact_t = Tensor(fb, dtype=dtypes.float)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
        qlog = o["query"].numpy()
        qv = qlog.argmax(-1)
        heads_all = o["heads_all"]
        fat_all = [t.numpy() for t in o["fat_all"]]
        breaths_all = [t.numpy() for t in o["breaths_all"]]
        assert len(heads_all) == K_B and len(fat_all) == K_B and len(breaths_all) == K_B
        KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in heads_all[0] else ())
        tkm_np = tk_np > 0.5
        # per-row tokenization/band machinery computed ONCE per batch (not once per breath -- the
        # text, its tokenization, and the digit-run/clause/mention segmentation are breath-invariant;
        # only the attention argmax varies by breath).
        rowcache = {}
        for bi, i in enumerate(sl):
            i = int(i)
            text = texts_all[i]
            q_out[i] = int(qv[bi])
            if has_gold:
                pres_gold_out[i] = vg["presence"][i]
                ftype_gold_out[i] = vg["ftype"][i]
                ids_row_enc = tok.encode(text)
                ids_row = np.zeros(T_ALG, np.int32)
                Lr = min(len(ids_row_enc.ids), T_ALG)
                ids_row[:Lr] = ids_row_enc.ids[:Lr]
                dec_i = MS._decode_tokens(tok, ids_row, T_ALG)
                is_num = np.array([d.isdigit() and d != "" for d in dec_i])
                _, clause_id, mention_id, runs = _row_bands(tok, ids_row, tkm_np[bi], se_np[bi])
                rowcache[bi] = dict(is_num=is_num, clause_id=clause_id, mention_id=mention_id, runs=runs)
            else:
                rowcache[bi] = dict(is_num=np.zeros(T_ALG, dtype=bool), clause_id=None, mention_id=None, runs=None)
            row_info.setdefault(i, {"text": text, "q": int(qv[bi]), "key": keys_all[i], "parses": {}})
        for kb in range(K_B):
            if kb == K_B - 1:
                # IDENTITY FIX (bug audit 2026-10-06, the same gap adaptive_stop.py carried): heads_all[K_B-1]
                # is heads_of() on the raw final-breath state, built (phase1_algebra_head.py:6774) AFTER the
                # final-breath-only args injections that live in out["args"] alone (ALG_PTR_SURF role:add:2.0
                # at :6740, the router pointer :6752, busreg :6768). The tap's last breath therefore decodes a
                # DIFFERENT graph from chain_acc.py's read of `o` (the banked 11-vs-14). Read the last breath
                # from `o` so the stop head's final-breath labels mean what the chain means.
                hk = {k: (o[k].numpy() if k in o else heads_all[kb][k].numpy()) for k in KEYS}
            else:
                hk = {k: heads_all[kb][k].numpy() for k in KEYS}
            fa_kb = fat_all[kb]
            st_kb = breaths_all[kb][:, :, CONTENT]
            for bi, i in enumerate(sl):
                i = int(i)
                row = {k: hk[k][bi].copy() for k in KEYS}
                pres_mask = row["pres"] > 0
                ftype_am = row["ftype"].argmax(-1)
                kindpred[i, kb] = np.where(~pres_mask, -1, np.where(ftype_am == 0, 0, np.where(ftype_am == 1, 1, 2)))
                state_store[i, kb] = st_kb[bi]
                # ---- per-slot entropy / numeral-argmax / mass-on-numerals ----
                real = tkm_np[bi] > 0.5
                p_tok = fa_kb[bi] * real[None, :].astype(np.float32)
                p_tok = p_tok / (p_tok.sum(-1, keepdims=True) + 1e-12)
                ent_row = -(p_tok * np.log(p_tok + 1e-12)).sum(-1)
                ent[i, kb, :] = ent_row
                is_num = rowcache[bi]["is_num"]
                amax = p_tok.argmax(-1)
                for j in range(L_FAC):
                    numarg[i, kb, j] = int(is_num[amax[j]]) if real.any() else -1
                    nummass[i, kb, j] = float(p_tok[j][is_num & real].sum()) if is_num.any() else 0.0
                # ---- band (given-gold slots only) ----
                if has_gold:
                    clause_id, mention_id, runs = rowcache[bi]["clause_id"], rowcache[bi]["mention_id"], rowcache[bi]["runs"]
                    for j in range(L_FAC):
                        if vg["presence"][i, j] < 0.5 or int(vg["ftype"][i, j]) != 1:
                            continue
                        v = int("".join(str(int(x)) for x in vg["digits"][i, j]))
                        band_arr = _given_slot_band(v, runs, mention_id, clause_id, se_np[bi], tkm_np[bi])
                        if band_arr is not None:
                            band[i, kb, j] = int(band_arr[amax[j]])
                # ---- numeral mask + decode (chain_acc's CA_MASK=1) ----
                masked_row = dict(row)
                masked_row["dig"] = row["dig"].copy()
                for j in range(row["ftype"].shape[0]):
                    if int(row["ftype"][j].argmax()) == 0:
                        continue
                    fake = legal_digit_logits(masked_row["dig"][j], texts_all[i])
                    if fake is not None:
                        masked_row["dig"][j] = fake
                parse = _decode_slots(masked_row)
                ri = row_info[i]
                ri["parses"][kb] = parse
                if kb == K_B - 1 and has_gold:
                    onp_row = {k: row[k] for k in row}
                    vg_row = {k: vg[k][i] for k in vg}
                    for j in range(L_FAC):
                        if vg_row["presence"][j] < 0.5:
                            continue
                        ok, *_ = _slot_ok(onp_row, j, vg_row, legal_digit_logits, texts_all[i])
                        right_final[i, j] = int(ok)
                q = ri["q"]
                if ri["key"] is None or not parse:
                    ri.setdefault("refused_kb", set()).add(kb)
                else:
                    used = ([f.get("var") for f in parse if f["ftype"] == "given"]
                            + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]])
                    nvv = max([q + 1] + [v + 1 for v in used if v is not None])
                    gvd = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
                    tasks_by_id[(i, kb)] = (q, ri["key"], parse, gvd, nvv)
                # ---- THE PARSE-SPACE METER (VALUES axis, imported from combined_oracle.py) ----
                if not args.no_parse_space and parse:
                    try:
                        for tid, qd, kd, pparse, gvd2, nvv2 in _value_variant_tasks((i, kb), row, texts_all[i], N_DIG, args.ps_max):
                            ps_tasks_by_id[tid] = (qd, kd, pparse, gvd2, nvv2)
                    except Exception as e:
                        pass  # the meter is a diagnostic, never a bar -- one row's failure never halts the run
        if (s0 // 8) % 10 == 0:
            print(f"[perceiver-collect] forward pass {s0}/{n} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[perceiver-collect] forward passes done ({time.time()-t0:.0f}s); "
          f"{len(tasks_by_id)} primary solver tasks, {len(ps_tasks_by_id)} parse-space tasks over {n} rows", flush=True)

    # -------------------------------------------------------------------------------------------
    # SOLVE PHASE (primary: judge needs solved + unique; parse-space meter: solved only)
    # -------------------------------------------------------------------------------------------
    import multiprocessing as mp
    all_solve_tasks = ([(tid, q, key, parse, gv, nvv) for tid, (q, key, parse, gv, nvv) in tasks_by_id.items()]
                       + [(("ps",) + tid, q, key, parse, gv, nvv) for tid, (q, key, parse, gv, nvv) in ps_tasks_by_id.items()])
    solve_out = {}
    if all_solve_tasks:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            it = pool.imap_unordered(AS._solve_task, all_solve_tasks, chunksize=1)
            for _ in range(len(all_solve_tasks)):
                try:
                    tid, st, val, asg, m = it.next(timeout=args.wall * 3 + 30)
                except mp.TimeoutError:
                    break
                solve_out[tid] = (st, val, asg, m)
    print(f"[perceiver-collect] solve phase done ({time.time()-t0:.0f}s); "
          f"{sum(1 for v in solve_out.values() if v[0] == 'solved')}/{len(all_solve_tasks)} solved", flush=True)

    uniq_tasks = []
    for tid, (q, key, parse, gv, nvv) in tasks_by_id.items():
        st, val, asg, m = solve_out.get(tid, ("hung", None, None, None))
        if st == "solved":
            uniq_tasks.append((tid, q, parse, gv, nvv, m, val))
    uniq_out = {}
    if uniq_tasks:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            it = pool.imap_unordered(AS._uniqueness_task, uniq_tasks, chunksize=1)
            for _ in range(len(uniq_tasks)):
                try:
                    tid, uniq = it.next(timeout=args.wall * 3 + 30)
                except mp.TimeoutError:
                    break
                uniq_out[tid] = uniq
    print(f"[perceiver-collect] uniqueness phase done ({time.time()-t0:.0f}s); "
          f"{sum(1 for v in uniq_out.values() if v is True)}/{len(uniq_tasks)} unique", flush=True)

    # parse-space counts: per (i, kb), number of DISTINCT solved variants (dedup already applied at
    # task-build time by parse signature)
    parse_space = np.full((n, K_B), -1, np.int16)   # -1 = no parse-space task queued for this (row, breath)
    ps_count = collections.Counter()
    for tid in ps_tasks_by_id:   # tid is the FLAT 3-tuple (i, kb, vi) built in _value_variant_tasks
        st, val, asg, m = solve_out.get(("ps",) + tid, ("hung", None, None, None))
        i, kb, vi = tid
        if st == "solved":
            ps_count[(i, kb)] += 1
    ps_queued = collections.Counter((i, kb) for (i, kb, vi) in ps_tasks_by_id)
    for (i, kb) in ps_queued:   # explicit 0 for "queued but nothing solved" (never left at -1 "not computed")
        parse_space[i, kb] = ps_count.get((i, kb), 0)

    # -------------------------------------------------------------------------------------------
    # PER-ROW STOP LABELS
    # -------------------------------------------------------------------------------------------
    solver_status = np.full((n, K_B), "hung", dtype="<U11")
    stop_kb_judge = np.full((n,), -1, np.int32)
    stop_kb_label = np.full((n,), -1, np.int32)
    # PER-BREATH (not just "first occurrence") ground truth, so perceiver_train.py can build the
    # STOP head's target at EVERY breath, not only reconstruct it from the first qualifying one:
    #   judge_pass[i,kb]  = 1 iff solved+unique at this breath (key-blind, THE CONSISTENCY JUDGE);
    #                       0 if attempted and failed; -1 if no task was queued (no parse/refused).
    #   key_match[i,kb]   = 1 iff solved AND the solved value equals the row's key; 0 if solved but
    #                       wrong; -1 if not solved or the key is unknown (N/A, never a silent 0).
    # stop_target[i,kb] = (judge_pass==1) & (key_match==1) is then the row x breath binary target the
    # STOP head is trained against DIRECTLY (every breath, not only the first one) -- stop_kb_label
    # above is kept too, as the single-number "first qualifying breath" summary the inference-time
    # policy and the rows-bar both read.
    judge_pass = np.full((n, K_B), -1, np.int8)
    key_match = np.full((n, K_B), -1, np.int8)
    last_kb = K_B - 1
    last_label = [None] * n

    def _solved(tid):
        if tid not in tasks_by_id:
            return "refused", None, None
        st, val, asg, m = solve_out.get(tid, ("hung", None, None, None))
        return st, val, asg

    for i in range(n):
        key = row_info[i]["key"]
        for kb in range(K_B):
            solver_status[i, kb] = _solved((i, kb))[0]
        for kb in range(K_B):
            st, val, asg = _solved((i, kb))
            if (i, kb) in tasks_by_id:
                u = uniq_out.get((i, kb))
                if st == "solved" and u is not None:
                    judge_pass[i, kb] = int(st == "solved" and u is True)
                if st == "solved" and key is not None:
                    key_match[i, kb] = int(val == key)
            if st == "solved" and uniq_out.get((i, kb)) is True:
                if stop_kb_judge[i] < 0:
                    stop_kb_judge[i] = kb
                if key is not None and val == key and stop_kb_label[i] < 0:
                    stop_kb_label[i] = kb
        st_last, val_last, asg_last = _solved((i, last_kb))
        if st_last != "solved":
            last_label[i] = "refused"
        else:
            last_label[i] = "correct" if (key is not None and val_last == key) else "wrong"

    # -------------------------------------------------------------------------------------------
    # ATLAS CENTROIDS (per breath, over THIS run's own predicted-kind slots -- never gold)
    # -------------------------------------------------------------------------------------------
    mu_given = np.full((K_B, C), np.nan, np.float32)
    mu_rel = np.full((K_B, C), np.nan, np.float32)
    for kb in range(K_B):
        gm = kindpred[:, kb, :] == 1
        rm = kindpred[:, kb, :] == 0
        if gm.any():
            mu_given[kb] = state_store[:, kb, :, :][gm].mean(0)
        if rm.any():
            mu_rel[kb] = state_store[:, kb, :, :][rm].mean(0)
    cos_given = np.full((n, K_B, L_FAC), np.nan, np.float32)
    cos_rel = np.full((n, K_B, L_FAC), np.nan, np.float32)
    leaf_chg = np.full((n, K_B, L_FAC), np.nan, np.float32)
    root_chg = np.full((n, K_B, L_FAC), np.nan, np.float32)
    branch_chg = np.full((n, K_B, L_FAC), np.nan, np.float32)
    for kb in range(K_B):
        if not np.isnan(mu_given[kb]).any():
            cos_given[:, kb, :] = AS._cos(state_store[:, kb, :, :], mu_given[kb][None, None, :])
        if not np.isnan(mu_rel[kb]).any():
            cos_rel[:, kb, :] = AS._cos(state_store[:, kb, :, :], mu_rel[kb][None, None, :])
        if kb > 0:
            # NOTE: state_store[:, kb, :, MASK] with a boolean MASK on the last axis, mixed with the
            # earlier ":"/"kb" indices in ONE bracket, triggers numpy's advanced-indexing axis
            # reorder (the mask's axis jumps to the FRONT) -- chain two plain indexing ops instead so
            # the boolean selection is the only advanced index in its own bracket, keeping (n, L_FAC,
            # n_band) as the result shape.
            cur_full, prev_full = state_store[:, kb, :, :], state_store[:, kb - 1, :, :]
            root_chg[:, kb, :] = AS._cos(cur_full[:, :, ROOT_IN_C], prev_full[:, :, ROOT_IN_C])
            branch_chg[:, kb, :] = AS._cos(cur_full[:, :, BRANCH_IN_C], prev_full[:, :, BRANCH_IN_C])
            leaf_chg[:, kb, :] = AS._cos(cur_full[:, :, LEAF_IN_C], prev_full[:, :, LEAF_IN_C])

    # -------------------------------------------------------------------------------------------
    # NL CERTIFIER (row-level, last-breath decode; adaptive_stop._nlc_cert, imported)
    # -------------------------------------------------------------------------------------------
    nlc = np.full((n,), np.nan, np.float32)
    for i in range(n):
        ri = row_info[i]
        parse_last = ri["parses"].get(last_kb, [])
        st_last, val_last, asg_last = _solved((i, last_kb))
        try:
            nlc[i] = AS._nlc_cert(ri["text"], parse_last, ri["q"], asg_last)
        except Exception as e:
            print(f"[perceiver-collect] nl_certifier row {i} failed: {e}", flush=True)
    nlc_bykb = np.repeat(nlc[:, None], K_B, axis=1)

    rack_path = AS._rack_sidecar_path(args.tag)
    dry = np.full((n, K_B), np.nan, np.float32)
    if rack_path is not None and has_gold:
        rz = np.load(rack_path, allow_pickle=True)
        rrows = {int(r): k for k, r in enumerate(rz["rows"])}
        for i in range(n):
            if i in rrows:
                dry[i, last_kb] = float(rz["dry"][rrows[i]].sum())

    # -------------------------------------------------------------------------------------------
    # WRITE
    # -------------------------------------------------------------------------------------------
    out_path = args.out or f".cache/perceiver_telemetry_{args.tag}_{fixture_tag}.npz"
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), tag=args.tag, ckpt=args.ckpt,
               source=args.source, rows_path=args.rows, has_gold=has_gold, n=n, K_B=K_B,
               commit_hash="")
    np.savez(out_path,
             meta=json.dumps(meta), texts=np.array(texts_all, dtype=object),
             key=key_out, q=q_out,
             ent=ent, numarg=numarg, nummass=nummass, band=band,
             cos_given=cos_given, cos_rel=cos_rel,
             leaf_chg=leaf_chg, root_chg=root_chg, branch_chg=branch_chg,
             solver_status=solver_status, nl_certifier=nlc_bykb, dry=dry, parse_space=parse_space,
             stop_kb_judge=stop_kb_judge, stop_kb_label=stop_kb_label,
             judge_pass=judge_pass, key_match=key_match,
             right_final=right_final, pres_gold=pres_gold_out, ftype_gold=ftype_gold_out,
             last_label=np.array(last_label, dtype="<U8"))
    print(f"[perceiver-collect] wrote {out_path}", flush=True)
    n_correct_last = sum(1 for l in last_label if l == "correct")
    n_stop_label = sum(1 for v in stop_kb_label if v >= 0)
    print(f"[perceiver-collect] last-breath correct (vs key) = {n_correct_last}/{n}; "
          f"rows with a stop_kb_label >= 0 = {n_stop_label}/{n}", flush=True)
    if pool_meta is not None:
        print(f"[perceiver-collect] gsm8k pool meta: {pool_meta}", flush=True)
    print(f"[perceiver-collect] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
