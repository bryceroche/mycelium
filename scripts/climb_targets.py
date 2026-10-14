"""climb_targets.py -- THE GOVERNOR CLIMB's training targets (2026-10-14,
zero-GPU DATA PREP, delegated). Spec: docs/phase1_skeleton_spec.md's
2026-10-13 "A TREE OF RELATIONSHIPS ..." and 2026-10-14 "OPUS'S FEEDBACK ...".
CLAUDE.md governs: DEV=CPU only, no GPU/.cache/gpu.lock touch, no overwrite
of banked artifacts (this script and its outputs are all new, climb_*-
prefixed), no ledger writes.

THE OBJECT: the climb trains one attention head per sorting-room block to
walk, LISA-style (Strubell et al. 2018), one hop of a numeral's GOVERNOR
CHAIN per block: numeral -> HEAD NOUN (the noun it modifies/counts) -> VERB
(that noun's governing predicate) -> OWNER (nsubj of that verb, or a direct
`poss` possessor of the noun) -- plus RECIPIENT (indirect object / dative,
"Tom gave Bob 5 apples") as a fourth, separate link off the same verb.
Coreference is explicitly OUT OF SCOPE here (spaCy's small model has none;
a pronoun owner is just flagged as a pronoun, never resolved further).
spaCy's dependency parse is the TRAINING TEACHER ONLY -- it never ships;
this script produces a sidecar of target POSITIONS for the trainer to
supervise against, nothing the deployed head reads at inference.

REUSE (so the owner definition matches THE GOVERNOR CENSUS,
scripts/governor_census.py, 2026-10-07, exactly): `numeral_candidates`,
`find_owner`, `sent_index_of` are imported unmodified from governor_census.
The "head noun" / "governing verb" ancestor walk below is governor_chain's
own traversal (governor_census.py:226-260ish), kept in lock-step but
inlined because governor_chain() returns TEXT/lemmas for its census tables,
not the token objects this script needs for character-offset mapping.
RECIPIENT is new (the census never needed it): `find_recipient` searches
the verb's own children for a `dative`/`iobj` dependent, or a `prep`
"to" whose `pobj` is the recipient ("gave 5 apples to Bob"); as a one-step
fallback (not a climb) it also tries `verb_used` (the verb find_owner
actually resolved the subject from, e.g. through `conj` coordination)
when `verb` itself has none -- the recipient is a sibling argument of
whichever clause supplied the subject.

DESIGN DECISION on the "verb" link: `verb` (the nearest VERB/AUX ancestor
of the numeral, found independently) is reported as the noun's verb link
EVEN WHEN no subject/owner is found for it (e.g. "There are 5 apples") --
noun/verb/owner/recipient coverage is reported BY LINK, independently, per
the task's own framing, rather than letting owner failure silently erase
a found verb.

TOKENIZATION (so target positions line up with the precompute's token
axis): replicates scripts/phase1_algebra_head.py's `tokenize()` exactly --
same TOKENIZER_JSON (.cache/llama-3.2-1b-weights/tokenizer.json), same
T_ALG=256, same `tok.encode(text)` call, same truncation rule (>256 raises
there; here it is SKIPPED and counted, never crashes the whole pool).
Verified against the precompute's own gold npz
(.cache/phase1_alg_states_formpm35c.npz: tokmask.npy) in `main()` below --
see the "tokmask cross-check" line in the JSON summary.

ALIGNMENT (spaCy tokens vs Llama BPE pieces do not line up 1:1): each
target word's [char_start, char_end) is mapped to every tokenizer PIECE
whose own offset interval overlaps it ("the full piece span"); the HEAD
PIECE is defined as the first (lowest-index) piece in that span. A link
is COUNTED AS AN ALIGNMENT FAILURE when spaCy resolved it to a real token
but no piece overlaps its character span at all (should be rare/zero given
the precompute's own truncation guarantee, but measured, not assumed).

POOL: .cache/form_mix_pm35c.jsonl (the diet's training rows, 50653 lines,
matching the precompute's row count exactly) -- the TRAINING pool only.
THE WILD HOLDOUT and MATH-500 are measured-only and are NEVER read by this
script for training targets; an optional --wild-diagnostic pass over
.cache/wild_admitted_holdout.jsonl is written ONLY to a file stamped
climb_targets_wild_DIAGNOSTIC_ONLY.* with a header saying so, and no
trainer may read it.

OUTPUT:
  .cache/climb_targets_formpm35c.npz   -- per-row target positions (see
      the npz layout comment in main() below).
  .cache/climb_targets_formpm35c_summary.json -- coverage by link, the
      self-consistency read, alignment failure rate, spaCy version/model,
      timing.
  .cache/climb_targets_sample.txt      -- 10 rows, human-readable.

USAGE:
  .venv/bin/python3 scripts/climb_targets.py [--limit N] [--wild-diagnostic]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from governor_census import numeral_candidates, find_owner, sent_index_of  # noqa: E402

T_ALG = 256  # scripts/phase1_algebra_head.py:45
TOKENIZER_JSON = os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/tokenizer.json")  # :2337
POOL = os.path.join(_ROOT, ".cache/form_mix_pm35c.jsonl")
STATES_NPZ = os.path.join(_ROOT, ".cache/phase1_alg_states_formpm35c.npz")
WILD_HOLDOUT = os.path.join(_ROOT, ".cache/wild_admitted_holdout.jsonl")

OUT_NPZ = os.path.join(_ROOT, ".cache/climb_targets_formpm35c.npz")
OUT_SUMMARY = os.path.join(_ROOT, ".cache/climb_targets_formpm35c_summary.json")
OUT_SAMPLE = os.path.join(_ROOT, ".cache/climb_targets_sample.txt")
OUT_WILD_DIAG_NPZ = os.path.join(_ROOT, ".cache/climb_targets_wild_DIAGNOSTIC_ONLY.npz")
OUT_WILD_DIAG_SUMMARY = os.path.join(_ROOT, ".cache/climb_targets_wild_DIAGNOSTIC_ONLY_summary.json")

MAX_NUM = 32  # safe above the observed max (17 in a 2000-row sample) of
# governor_census.numeral_candidates() hits per row; rows that exceed it
# are TRUNCATED to the first MAX_NUM numerals in text order and counted.

TIME_BUDGET_S = 100 * 60  # ~100 min soft wall; stop early and say so
# rather than silently running past the ~2h CPU instruction.


def find_recipient(verb):
    """NEW (not in governor_census -- the census never needed a recipient):
    verb's own `dative`/`iobj` child ("gave Bob 5 apples"), else the `pobj`
    of a `to`-headed `prep` child ("gave 5 apples to Bob"). One step, no
    coordination climb (the climb is done by the caller via verb_used)."""
    if verb is None:
        return None
    for c in verb.children:
        if c.dep_ in ("dative", "iobj"):
            return c
    for c in verb.children:
        if c.dep_ == "prep" and c.lemma_.lower() == "to":
            for cc in c.children:
                if cc.dep_ == "pobj":
                    return cc
    return None


def climb_links(doc, sent_list, tok):
    """governor_chain's own ancestor walk (governor_census.py), inlined to
    keep the token objects (governor_chain() only returns text/lemmas).
    Returns a dict of token-or-None per link, plus owner_kind and the
    numeral's/owner's sentence index (for the self-consistency read)."""
    head_noun = None
    verb = None
    for anc in tok.ancestors:
        if head_noun is None and anc.pos_ in ("NOUN", "PROPN"):
            head_noun = anc
        if verb is None and anc.pos_ in ("VERB", "AUX"):
            verb = anc
            break
    if head_noun is None and tok.head is not tok:
        head_noun = tok.head
    owner_tok, owner_kind, verb_used = find_owner(verb, head_noun)
    recip_tok = find_recipient(verb)
    if recip_tok is None and verb_used is not None and verb_used is not verb:
        recip_tok = find_recipient(verb_used)
    num_sent = sent_index_of(doc, sent_list, tok)
    owner_sent = (sent_index_of(doc, sent_list, owner_tok)
                  if owner_tok is not None else -1)
    return dict(noun=head_noun, verb=verb, owner=owner_tok,
                owner_kind=owner_kind, recipient=recip_tok,
                num_sent=num_sent, owner_sent=owner_sent)


def _tokenize_row(hf_tok, text):
    """Replicates scripts/phase1_algebra_head.py:tokenize()'s per-row body
    exactly (same tokenizer file, same T_ALG, same encode call). Returns
    (ids, offsets, truncated) instead of raising on truncation -- this
    script's job is to SKIP+COUNT a truncated row, never crash the pool."""
    e = hf_tok.encode(text)
    if len(e.ids) > T_ALG:
        return None, None, True
    return list(e.ids), list(e.offsets), False


def char_span_to_piece(offsets, char_start, char_end):
    """-> (head_piece, span_start, span_end_exclusive) or None. 'head piece'
    = the first (lowest-index) tokenizer piece whose own [ps, pe) overlaps
    [char_start, char_end); 'full piece span' = [span_start, span_end)."""
    hits = [p for p, (ps, pe) in enumerate(offsets)
            if ps < char_end and pe > char_start]
    if not hits:
        return None
    return hits[0], hits[0], hits[-1] + 1


def process_pool(rows, nlp, hf_tok, max_num=MAX_NUM, time_budget_s=None,
                  label="pool"):
    """Returns (arrays dict, stats dict, sample list of first-10 processed
    rows' detail for the human-readable file)."""
    n = len(rows)
    row_idx = np.full((n,), -1, np.int32)
    n_num = np.zeros((n,), np.int32)
    truncated = np.zeros((n,), np.uint8)
    numeral_pos = np.full((n, max_num), -1, np.int32)
    numeral_span = np.full((n, max_num, 2), -1, np.int32)
    links = {}
    for link in ("noun", "verb", "owner", "recipient"):
        links[f"{link}_pos"] = np.full((n, max_num), -1, np.int32)
        links[f"{link}_span"] = np.full((n, max_num, 2), -1, np.int32)
        links[f"{link}_valid"] = np.zeros((n, max_num), np.uint8)
    owner_pronoun = np.zeros((n, max_num), np.uint8)
    owner_kind_code = np.full((n, max_num), -1, np.int8)  # 0=possessor,1=subject
    owner_same_sent = np.zeros((n, max_num), np.uint8)

    # coverage / failure counters
    resolved = collections.Counter()      # link -> #numerals spaCy resolved it for
    aligned = collections.Counter()       # link -> #resolved that also mapped to a piece
    total_numerals = 0
    skipped_truncated_llama = 0
    skipped_oversize_numerals = 0
    pronoun_owners = 0
    resolved_owners = 0
    same_sent_owners = 0

    sample = []
    t0 = time.time()
    kept = 0
    texts = [r["text"] for r in rows]
    for i, (row, doc) in enumerate(zip(rows, nlp.pipe(texts, batch_size=64))):
        if time_budget_s is not None and i % 2000 == 0 and i > 0:
            if time.time() - t0 > time_budget_s:
                print(f"[climb_targets] {label}: time budget hit at row {i}/{n}, stopping early", flush=True)
                break
        kept = i + 1  # this row-slot has been visited, whatever happens below
        ids, offsets, trunc = _tokenize_row(hf_tok, row["text"])
        if trunc:
            skipped_truncated_llama += 1
            continue
        sent_list = list(doc.sents)
        full_cands = numeral_candidates(doc)
        cands = full_cands
        if len(full_cands) > max_num:
            skipped_oversize_numerals += 1
            truncated[i] = 1
            cands = full_cands[:max_num]
        row_idx[i] = row.get("_orig_idx", i)
        n_num[i] = len(cands)

        detail_rows = []
        for k, (ti, tok, v) in enumerate(cands):
            total_numerals += 1
            ns, ne = tok.idx, tok.idx + len(tok.text)
            hp = char_span_to_piece(offsets, ns, ne)
            if hp is None:
                continue  # the numeral itself failed to align; no link positions make sense
            numeral_pos[i, k] = hp[0]
            numeral_span[i, k] = (hp[1], hp[2])

            ch = climb_links(doc, sent_list, tok)
            owner_kind = ch["owner_kind"]
            if owner_kind is not None:
                resolved_owners += 1
                owner_kind_code[i, k] = 0 if owner_kind == "possessor" else 1
                if ch["owner"].pos_ == "PRON":
                    owner_pronoun[i, k] = 1
                    pronoun_owners += 1
                if ch["num_sent"] == ch["owner_sent"] and ch["num_sent"] != -1:
                    owner_same_sent[i, k] = 1
                    same_sent_owners += 1

            per_tok = {}
            for name in ("noun", "verb", "owner", "recipient"):
                t = ch[name]
                per_tok[name] = t
                if t is None:
                    continue
                resolved[name] += 1
                s, e = t.idx, t.idx + len(t.text)
                hp2 = char_span_to_piece(offsets, s, e)
                if hp2 is None:
                    continue
                aligned[name] += 1
                links[f"{name}_pos"][i, k] = hp2[0]
                links[f"{name}_span"][i, k] = (hp2[1], hp2[2])
                links[f"{name}_valid"][i, k] = 1

            if i < 10:
                detail_rows.append((tok.text, ns, ne,
                                     {name: (per_tok[name].text if per_tok[name] is not None else None)
                                      for name in ("noun", "verb", "owner", "recipient")},
                                     owner_kind,
                                     (ch["owner"].pos_ == "PRON") if ch["owner"] is not None else None))
        if i < 10:
            sample.append((row["text"], detail_rows))

    arrays = dict(row_idx=row_idx[:kept], n_num=n_num[:kept], truncated=truncated[:kept],
                  numeral_pos=numeral_pos[:kept], numeral_span=numeral_span[:kept],
                  owner_pronoun=owner_pronoun[:kept], owner_kind_code=owner_kind_code[:kept],
                  owner_same_sent=owner_same_sent[:kept])
    for k, v in links.items():
        arrays[k] = v[:kept]
    arrays["max_num"] = np.array(max_num)
    arrays["t_alg"] = np.array(T_ALG)

    elapsed = time.time() - t0
    stats = dict(
        n_rows_in=n, n_rows_processed=kept,
        n_rows_skipped_truncated_llama=skipped_truncated_llama,
        n_rows_numerals_truncated_to_max=int(truncated[:kept].sum()),
        n_rows_oversize_numerals=skipped_oversize_numerals,
        total_numerals=total_numerals,
        resolved_by_link={k: resolved[k] for k in ("noun", "verb", "owner", "recipient")},
        aligned_by_link={k: aligned[k] for k in ("noun", "verb", "owner", "recipient")},
        coverage_by_link={k: (resolved[k] / total_numerals if total_numerals else 0.0)
                           for k in ("noun", "verb", "owner", "recipient")},
        alignment_failure_rate_by_link={
            k: (1.0 - aligned[k] / resolved[k]) if resolved[k] else None
            for k in ("noun", "verb", "owner", "recipient")},
        owner_pronoun_share=(pronoun_owners / resolved_owners if resolved_owners else 0.0),
        owner_self_consistency=(same_sent_owners / resolved_owners if resolved_owners else 0.0),
        elapsed_s=elapsed,
        rows_per_s=(kept / elapsed if elapsed > 0 else None),
    )
    return arrays, stats, sample


def write_sample(path, sample, header):
    with open(path, "w") as f:
        f.write(header + "\n\n")
        for ri, (text, detail_rows) in enumerate(sample):
            f.write(f"=== row {ri} ===\n{text}\n")
            if not detail_rows:
                f.write("  (no numerals)\n\n")
                continue
            for (ntext, ns, ne, links_, owner_kind, owner_pron) in detail_rows:
                f.write(f"  numeral '{ntext}' [{ns}:{ne}]\n")
                f.write(f"    noun={links_['noun']!r} verb={links_['verb']!r} "
                        f"owner={links_['owner']!r} (kind={owner_kind}, pronoun={owner_pron}) "
                        f"recipient={links_['recipient']!r}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="process only the first N pool rows (testing)")
    ap.add_argument("--wild-diagnostic", action="store_true",
                     help="ALSO run a diagnostic-only pass over the wild "
                          "holdout, written to a file no trainer may read")
    args = ap.parse_args()

    import spacy
    from tokenizers import Tokenizer

    t_start = time.time()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    hf_tok = Tokenizer.from_file(TOKENIZER_JSON)

    rows = [json.loads(l) for l in open(POOL)]
    for i, r in enumerate(rows):
        r["_orig_idx"] = i
    n_pool_rows_total = len(rows)
    if args.limit:
        rows = rows[:args.limit]

    arrays, stats, sample = process_pool(rows, nlp, hf_tok, time_budget_s=TIME_BUDGET_S,
                                          label="train-pool")

    # --- tokmask cross-check against the precompute's own gold npz -------
    tokmask_check = None
    if os.path.exists(STATES_NPZ) and not args.limit:
        try:
            z = np.load(STATES_NPZ)
            tm = z["tokmask"]
            mismatches = 0
            checked = 0
            for i in range(min(len(rows), tm.shape[0])):
                ids, offsets, trunc = _tokenize_row(hf_tok, rows[i]["text"])
                if trunc:
                    continue
                checked += 1
                if int(tm[i].sum()) != len(ids):
                    mismatches += 1
            tokmask_check = dict(checked=checked, mismatches=mismatches)
        except Exception as e:
            tokmask_check = dict(error=str(e))

    np.savez_compressed(OUT_NPZ, **arrays)

    write_sample(OUT_SAMPLE, sample,
                 header="climb_targets_sample.txt -- 10 rows from "
                        ".cache/form_mix_pm35c.jsonl, THE GOVERNOR CLIMB's "
                        "spaCy-derived targets (head piece positions shown "
                        "as token text for readability; -1/None = unresolved "
                        "or alignment-failed).")

    import spacy as _spacy_mod
    summary = dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pool=POOL,
        n_pool_rows=n_pool_rows_total,
        limited_to=args.limit,
        spacy_version=_spacy_mod.__version__,
        spacy_model=dict(nlp.meta),
        tokenizer_json=TOKENIZER_JSON,
        t_alg=T_ALG,
        max_num=MAX_NUM,
        stats=stats,
        tokmask_cross_check=tokmask_check,
        output_npz=OUT_NPZ,
        output_npz_shapes={k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape") and v.shape},
        total_wall_s=time.time() - t_start,
        notes=[
            "coreference is OUT OF SCOPE (spaCy's small model has none); "
            "owner_pronoun flags a pronoun owner rather than resolving it.",
            "verb link = the numeral's direct VERB/AUX ancestor, reported "
            "independently of whether a subject/owner was found for it.",
            "owner link reuses governor_census.find_owner verbatim, "
            "including its conj/xcomp/ccomp/advcl/acl/relcl elided-subject "
            "climb -- the owner's resolving verb may differ from the "
            "reported 'verb' link in that case (documented in the header).",
            "recipient is new (not in governor_census): dative/iobj child, "
            "or pobj of a to-headed prep, tried on 'verb' then (one step, "
            "no further climb) on find_owner's own verb_used.",
            "KNOWN FALSE-POSITIVE MODE on recipient: the to-headed-prep "
            "fallback does not distinguish dative 'to' (\"gave it to Bob\") "
            "from locative/destination 'to' (\"moved to 45 Maple Street\") "
            "-- spot-checked in the sample file (row 5: 'Street' flagged as "
            "a recipient of 'moved'). A consumer wanting only true datives "
            "should gate this link on the governing verb's lemma.",
            "noun link inherits governor_chain's own fallback (unchanged "
            "here): when no NOUN/PROPN ancestor exists at all, head_noun "
            "falls back to tok.head directly, which can be a VERB/AUX "
            "(spot-checked in the sample file, row 4: noun='is' for the "
            "numeral 'one' in 'f is eighty-one') -- this is not a bug, it "
            "is governor_census's own documented behavior, reused verbatim.",
            "numeral_candidates() does not merge multi-word spelled "
            "compounds (census's own documented limitation): "
            "'eighty-one' surfaces as two separate numeral candidates "
            "('eighty' and 'one'), each separately linked.",
        ],
    )
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in summary.items() if k != "stats"}, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))

    if args.wild_diagnostic:
        wrows = [json.loads(l) for l in open(WILD_HOLDOUT)]
        for i, r in enumerate(wrows):
            r["_orig_idx"] = i
        warrays, wstats, wsample = process_pool(wrows, nlp, hf_tok, label="wild-DIAGNOSTIC")
        np.savez_compressed(OUT_WILD_DIAG_NPZ, **warrays)
        wsummary = dict(
            HEADER="DIAGNOSTIC ONLY -- built from .cache/wild_admitted_holdout.jsonl "
                   "(THE WILD HOLDOUT, measured-only). NO TRAINER MAY READ THIS FILE "
                   "OR .cache/climb_targets_wild_DIAGNOSTIC_ONLY.npz. For inspection "
                   "of the climb's coverage on the held-out register only.",
            generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
            source=WILD_HOLDOUT,
            spacy_version=_spacy_mod.__version__,
            stats=wstats,
        )
        with open(OUT_WILD_DIAG_SUMMARY, "w") as f:
            json.dump(wsummary, f, indent=2, default=str)
        print("[climb_targets] wild DIAGNOSTIC ONLY written:", OUT_WILD_DIAG_NPZ)


if __name__ == "__main__":
    main()
