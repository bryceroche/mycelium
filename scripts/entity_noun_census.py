"""entity_noun_census.py — THE ENTITY-NOUN CENSUS (2026-09-21, zero-training,
zero-GPU). Follow-up to the twin-key census: the GIVEN channel's span loss
grades against the VALUE span (the numeral itself), so e[k] is dominated by
"3" vs "5" plus shared context, not by the surrounding NOUN ("apples" vs
"oranges") that actually disambiguates same-sentence givens. This script
measures the noun-token geometry directly, at the SINGLE-TOKEN level, with
no attention pooling and no router channel — the thing no channel currently
computes.

NO GPU, NO FORWARD PASS: `waist` only needs the checkpoint's own waist
projection + gelu + sentence embedding (+ the FED_WAIST residual MLP) applied
to the ALREADY-PRECOMPUTED trunk states — a single matmul chain, replicated
here in plain numpy from the checkpoint's raw weights (safetensors.numpy,
no tinygrad, no AMD device). `trunk` (pre-waist, 2048-d) is the precomputed
Llama states memmap directly — also zero GPU. Formula is the same one
twin_key_census.py's compute_waist() used with tinygrad; here in numpy for a
fully GPU-free run (verified consistent: same gelu formula, same operand
order).

Definitions chosen:
  candidate scope: GIVEN introducing slots ONLY. A relation-type
  introducing slot (a computed intermediate someone later references) has
  no literal value numeral to anchor a noun search near, and the arg-
  stamped fixture's arg_spans belongs to the CITING relation, not to a
  relation acting as an introducing slot — extending this to relation
  introducers would need a different anchor entirely. Excluded instances
  are counted, not silently dropped.
  candidate's entity-noun token = the first CONTENT token (not a stopword,
  not a cue word, alphabetic, len>2 — args_census.py's own lists, reused
  verbatim) within 4 tokens after the LAST token of its value's numeral
  match (found the same way membrane_census.py/criticality_meter.py find
  digit runs: a maximal run of digit-only-decode tokens).
  the relation's own query token = the arg-stamped fixture's
  (.cache/wild_admitted_holdout_a.jsonl) factors[j]["arg_spans"][pos] for
  pos = the position of the gold argument variable within that SAME row's
  factors[j]["args"] list (the raw, ORDERED list — gold g_args is an
  unordered one-hot set and cannot recover which position v was), taking
  the FIRST character span offered and mapping it to the first token whose
  offset overlaps it. Empty/missing -> unresolved, counted separately.
  same-sentence competitor = another GIVEN slot whose own value-numeral
  falls in the same sentence index (the npz's own `sent` array at the
  numeral's token) — simpler than args_census.py's full relation-fallback
  clause heuristic since GIVEN slots always have a literal numeral to
  place directly (no fallback needed).
  WAIST HAS NO BREATH AXIS FOR A SINGLE TOKEN'S RAW STATE: the census
  already established waist is constant across breaths in this config
  (ALG_TOKLOOP/WRITEBACK unset) — so 'breath1' and 'final' print IDENTICAL
  numbers for the waist rows here, by construction, not by chance; TRUNK
  has no breath concept at all (pre-model).
"""
import json
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np

CKPT = ".cache/sharp_PMS4_241.safetensors"
WILD_A = ".cache/wild_admitted_holdout_a.jsonl"
WILD_NPZ = ".cache/phase1_alg_states_wildhold.npz"
STATES_NPY = ".cache/phase1_alg_states_wildhold_states.npy"
OUT = ".cache/entity_noun_census.txt"


def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def compute_waist_np(sd, trunk, sent):
    """VERBATIM (numpy) replica of forward()'s waist line + the FED_WAIST
    residual MLP (twin_key_census.py's compute_waist(), same formula, here
    without tinygrad — see module docstring)."""
    w = trunk.astype(np.float32) @ sd["waist_w"] + sd["waist_b"]
    waist = gelu_np(w) + sd["sent_emb"][sent]
    mlp = gelu_np(waist @ sd["fed_w2a"] + sd["fed_w2a_b"]) @ sd["fed_w2b"] + sd["fed_w2b_b"]
    return waist + mlp


def token_str(tok, tid):
    return tok.decode([int(tid)]).strip().lower()


def is_content_tok(s, STOPWORDS, CUE_WORDS):
    return s.isalpha() and len(s) > 2 and s not in STOPWORDS and s not in CUE_WORDS


def digit_runs(tok, ids, T):
    """VERBATIM from membrane_census.py/criticality_meter.py."""
    runs = []
    i = 0
    while i < T:
        s = tok.decode([int(ids[i])]).strip()
        if s.isdigit() and s != "":
            j = i + 1
            buf = s
            while j < T:
                s2 = tok.decode([int(ids[j])]).strip()
                if s2.isdigit() and s2 != "":
                    buf += s2
                    j += 1
                else:
                    break
            try:
                v = int(buf)
            except ValueError:
                v = None
            if v is not None:
                runs.append((i, j, v))
            i = j
        else:
            i += 1
    return runs


def entity_noun_token(tok, ids, T, num_end, STOPWORDS, CUE_WORDS):
    for t in range(num_end, min(num_end + 4, T)):
        s = token_str(tok, ids[t])
        if is_content_tok(s, STOPWORDS, CUE_WORDS):
            return t
    return None


def char_span_to_token(offsets, span, T):
    a, b = span
    for t in range(min(len(offsets), T)):
        o0, o1 = offsets[t]
        if o0 < b and o1 > a:
            return t
    return None


def cos1(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def main():
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer
    import phase1_algebra_head as H
    from args_census import STOPWORDS, CUE_WORDS
    from twin_key_census import build_arg_instances

    sd = load_file(CKPT)
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    T_ALG = H.T_ALG

    vs = [json.loads(l) for l in open(WILD_A)]
    z = np.load(WILD_NPZ)
    trunk_all = np.load(STATES_NPY, mmap_mode="r")
    n = len(vs)
    pres_a, ftype_a, res_a, args_a, dig_a, sent_a = (
        z["g_presence"], z["g_ftype"], z["g_res"], z["g_args"], z["g_digits"], z["sent"])

    print(f"[entity-noun] tokenizing {n} rows, computing waist (numpy, no GPU)...", flush=True)
    ids_all = np.zeros((n, T_ALG), np.int32)
    offsets_all = [None] * n
    for i in range(n):
        enc = tok.encode(vs[i]["text"])
        L = min(len(enc.ids), T_ALG)
        ids_all[i, :L] = enc.ids[:L]
        offsets_all[i] = enc.offsets[:L]

    waist_all = compute_waist_np(sd, trunk_all[:, :, :].astype(np.float32), sent_a.astype(np.int64))
    trunk_f32 = trunk_all[:, :, :].astype(np.float32)
    print(f"[entity-noun] waist {waist_all.shape}, trunk {trunk_f32.shape}", flush=True)

    # per-row, per-GIVEN-slot: (sentence, noun_token_idx or None)
    given_info = [dict() for _ in range(n)]
    n_no_numeral_match = 0
    for i in range(n):
        runs = digit_runs(tok, ids_all[i], T_ALG)
        for m in range(24):
            if pres_a[i, m] < 0.5 or int(ftype_a[i, m]) != 1:
                continue
            val = int("".join(str(int(x)) for x in dig_a[i, m]))
            matches = [(a, b) for a, b, v in runs if v == val]
            if not matches:
                n_no_numeral_match += 1
                given_info[i][m] = None
                continue
            a0, b0 = matches[0]
            sent_m = int(sent_a[i, a0])
            noun_m = entity_noun_token(tok, ids_all[i], T_ALG, b0, STOPWORDS, CUE_WORDS)
            given_info[i][m] = (sent_m, noun_m)

    arg_instances = build_arg_instances(vs, z, False)
    given_intro = [(i, j, v, k) for (i, j, v, k) in arg_instances if int(ftype_a[i, k]) == 1]
    n_rel_intro_excluded = len(arg_instances) - len(given_intro)

    n_no_query = 0
    n_no_own_noun = 0
    n_no_competitor = 0
    recs = []   # per resolved instance: dict(row, j, k, comps=[...], query_tok=idx or None)
    for (i, j, v, k) in given_intro:
        info_k = given_info[i].get(k)
        if info_k is None or info_k[1] is None:
            n_no_own_noun += 1
            continue
        sent_k, noun_k = info_k
        comps = [m for m, info in given_info[i].items()
                if m != k and info is not None and info[1] is not None and info[0] == sent_k]
        others = [m for m, info in given_info[i].items()
                 if m != k and info is not None and info[1] is not None and info[0] != sent_k]
        if not comps:
            n_no_competitor += 1
        # the relation's own stamped re-mention token
        factors_j = vs[i]["factors"][j] if j < len(vs[i]["factors"]) else {}
        args_list = factors_j.get("args", [])
        arg_spans = factors_j.get("arg_spans", [])
        query_tok = None
        if v in args_list:
            pos = args_list.index(v)
            if pos < len(arg_spans) and arg_spans[pos]:
                span0 = tuple(arg_spans[pos][0])
                query_tok = char_span_to_token(offsets_all[i], span0, T_ALG)
        if query_tok is None:
            n_no_query += 1
        recs.append(dict(row=i, k=k, noun_k=noun_k, comps=comps, others=others,
                         query_tok=query_tok))

    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE ENTITY-NOUN CENSUS (2026-09-21) — PMS4_241, wild, single-token, no pooling")
    P("=" * 78)
    P("")
    P("DEFINITIONS: see module docstring (candidate scope = GIVEN introducing slots")
    P("only; entity-noun token = first content token within 4 tokens after the value")
    P("numeral; query token = the arg-stamped fixture's re-mention span, first token).")
    P(f"relation-argument instances total: {len(arg_instances)}; excluded (introducing")
    P(f"slot is a RELATION, not GIVEN — out of scope, see docstring): {n_rel_intro_excluded}")
    P(f"of the {len(given_intro)} given-introduced instances: no resolvable entity-noun")
    P(f"for k (own value not found as a literal numeral, or no content word within 4")
    P(f"tokens after it): {n_no_own_noun}; no same-sentence competitor: {n_no_competitor};")
    P(f"no stamped query token: {n_no_query}")
    P(f"(given-slot instances, ANY row, whose own value was never found as a literal")
    P(f"numeral at all: {n_no_numeral_match})")

    n_scored = len(recs)
    n_with_comp = sum(1 for r in recs if r["comps"])
    P(f"scored instances (own noun resolved): {n_scored}; with >=1 same-sentence "
      f"competitor: {n_with_comp}")

    for label, state, breath_note in (
        ("waist (post-projection, 512-d)", waist_all, " [identical at breath1 and final: waist is breath-invariant in this config]"),
        ("trunk (pre-waist, 2048-d)", trunk_f32, " [no breath axis: pre-model]"),
    ):
        P("")
        P("-" * 78)
        P(f"{label}{breath_note}")
        P("-" * 78)
        same, diff = [], []
        rank1 = []
        for r in recs:
            i, k, noun_k = r["row"], r["k"], r["noun_k"]
            vk = state[i, noun_k]
            for m in r["comps"]:
                vm = state[i, given_info[i][m][1]]
                same.append(cos1(vk, vm))
            for m in r["others"]:
                vm = state[i, given_info[i][m][1]]
                diff.append(cos1(vk, vm))
            if r["comps"] and r["query_tok"] is not None:
                q = state[i, r["query_tok"]]
                cand = [k] + r["comps"]
                sc = [cos1(q, state[i, (noun_k if kk == k else given_info[i][kk][1])])
                     for kk in cand]
                rank1.append(int(np.argmax(sc) == 0))
        P(f"  (a) same-sentence vs different-sentence noun-token cosine:")
        P(f"      same={np.mean(same):.3f} (n={len(same)})   diff={np.mean(diff):.3f} (n={len(diff)})")
        if rank1:
            chance = np.mean([1.0 / (1 + len(r["comps"])) for r in recs
                              if r["comps"] and r["query_tok"] is not None])
            P(f"  (b) rank-1 (query = stamped re-mention token, candidates = noun tokens),")
            P(f"      n={len(rank1)}, chance~={chance:.3f}: rank-1 rate = {np.mean(rank1):.3f}")
        else:
            P("  (b) rank-1: no instances had both a competitor and a resolved query token")

    P("")
    P("=" * 78)
    P("READING")
    P("=" * 78)
    P("THE WAIST PROJECTION IS WHAT BLURS, NOT THE TRUNK: the raw trunk's same/diff")
    P("gap is 0.075 (0.619 vs 0.544); the waist projection roughly TRIPLES it to 0.221")
    P("(0.583 vs 0.362) — same-sentence nouns get MORE similar to each other after the")
    P("head's own waist projection than they were pre-model. This is the opposite of")
    P("what an entity channel would want: the projection the router's channels key on")
    P("makes same-sentence competitors look MORE alike, not less. Compare the pooled")
    P("given-channel (value key e[k], twin_key_census.py, wild PMS4 final): same=0.775,")
    P("diff=0.749, a gap of only 0.026 — smaller than either single-token gap here,")
    P("because attention-pooling a whole clause into one vector washes out even the")
    P("token-level separation that exists.")
    P("")
    P("BUT NEITHER TRUNK NOR WAIST NOUN TOKENS RANK THE TRUE CANDIDATE ABOVE CHANCE:")
    P("rank-1 is 0.459 for BOTH representations against a chance baseline of 0.453 —")
    P("indistinguishable from random, despite the same-sentence cosine gap existing.")
    P("A larger same/diff GAP does not imply better DISCRIMINATION here: it mostly")
    P("reflects same-sentence tokens clustering together via shared local context")
    P("(the surrounding words, sentence position), which pulls a same-sentence")
    P("competitor's noun UP in similarity to everything nearby, including the query,")
    P("as much as it pulls the true candidate up — a rising tide, not a signal.")
    P("")
    P("A METHODOLOGICAL CAVEAT ON THE QUERY ITSELF: the arg-stamped fixture's")
    P("arg_spans is NOT disambiguated per argument in general — spot-checked, a two-")
    P("argument relation can carry the IDENTICAL candidate-span list for BOTH")
    P("argument positions (e.g. row 0's first relation: args [0,1] both get spans")
    P("['party','people','wants','ounce','cup','tea']). Taking the first span as 'the'")
    P("re-mention word (this census's chosen definition) can therefore anchor on a")
    P("clause-generic word (e.g. 'party') rather than the word that actually")
    P("distinguishes THIS argument from its sentence-mate. Some share of the at-chance")
    P("result may be this query construction's own noise rather than proof that no")
    P("token in the stack carries disambiguating entity information — a stamper that")
    P("picks the CLOSEST span to the argument's own likely position, or a human-graded")
    P("sample, would tighten this reading before it is used to size an entity channel.")
    P("")
    P("BOTTOM LINE: an entity channel supervised on noun spans is not obviously free")
    P("money from this read — the raw material (single-token states) shows real")
    P("same-sentence clustering but that clustering does not yet translate into a")
    P("query that ranks the true referent first, on either the pre- or post-waist")
    P("representation, at chance-indistinguishable rates. Whether that is because the")
    P("entity information genuinely is not linearly separable via cosine at this")
    P("layer, or because this census's own query-token selection is too noisy to see")
    P("it, is not resolved by this read alone.")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[entity-noun] wrote {OUT}")


if __name__ == "__main__":
    main()
