"""scripts/picker/jury_features.py -- THE CARICATURE (2026-10-06, zero-GPU, delegate; form 2 of
THE COURTROOM, docs/phase1_skeleton_spec.md 2026-10-06 13:35 "THE TRAINED JUROR").

Bryce's frame: "Caricatures work by comparing a real face to an average baseline and sharply
amplifying the unique differences" -- the juror reads only the AMPLIFIED DIFFERENCE between two
stories on the text span where they actually disagree, never the whole story.

REUSE, NOT REIMPLEMENTATION (every piece below is a thin wrapper over an existing, unedited module):
  - discriminating slots:        scripts/courtroom.py's discriminating_slots()
  - the text-certificate diff:   scripts/courtroom.py's juror1_cert_diff() (span-restricted where
                                  possible, row-level fallback otherwise -- the SAME granularity rule)
  - evidence-ignored / collision diffs: scripts/courtroom.py's unexplained_count()/collision_count()
  - the grounding span itself:   scripts/stamp_arg_mentions.py's clause_of()/windows_of() (the SAME
                                  function courtroom.restricted_cert() calls for its cue_agree
                                  formula) -- applied here to EVERY differing slot (given or rel),
                                  not only rel slots, since clause_of already handles both via
                                  own_value()/recursion_args().
  - sentence bounds/spans:       scripts/args_census.py's sentence_bounds/sentence_spans
  - the frozen trunk, on CPU:    mycelium/llama_loader.py's attach_llama_layers/load_llama_weights,
                                  loaded and run EXACTLY as scripts/perceiver_collect.py's
                                  _get_trunk_host()/_embed_raw_batch() do (DEV=CPU, L0-L3, rms_norm
                                  of the last layer) -- this module is a sibling port, not an edit.

THE CARICATURE EMBEDDING (per pair, shared by both orderings -- it depends on the TEXT and the
discriminating slots' union span, not on which story is "A"): span_pool (mean trunk state over the
tokens the differing slots are grounded in, widened to the enclosing clause) and span_pool -
text_pool (the span minus the "average face" = the whole text's own pooled state). Both are kept
(the task's "plus span_pool alone").

THE STRUCTURAL DIFFERENCE (per pair, SIGNED, A-minus-B -- this is what actually distinguishes the
two orderings and is swap-augmented at training time): ftype-bucket counts, op-bucket counts,
given-value-text-presence, and a "cross-clause" rate (whether a differing relation's argument now
introduces from a DIFFERENT sentence than before) -- all computed directly off the parse dicts
gen_candidates.py already produced (no new solving, no new candidate generation).

NEVER: the candidate's own log-likelihood or any score derived from a model head beyond the decoded
discrete structure (argmax ftype/op/args/value) -- the 2026-10-05 12:48 panel-picker ablation's
ruling, restated by this task's own brief.
"""
import os
import re
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

import courtroom as CT             # discriminating_slots, juror1_cert_diff, unexplained_count, collision_count
import stamp_arg_mentions as SAM   # clause_of, windows_of, build_intro_map
import args_census as AC           # sentence_bounds, sentence_spans

FTYPES = ["rel", "given", "mod", "sel", "pct", "fdiv", "macro", "frac"]
OPS = ["add", "mul", "sub", "div", None]
NUM_RE = CT.NUM_RE


# ================================================================================================
# 1. THE FROZEN TRUNK, CPU, cached per process (mycelium/llama_loader.py, perceiver_collect.py's
#    own pattern -- imported, never edited).
# ================================================================================================
_TRUNK = {}


def get_trunk():
    if "host" not in _TRUNK:
        assert os.environ.get("DEV") == "CPU", "jury_features: trunk embedding is CPU-only, always"
        from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                            LLAMA_3_2_1B_CFG, _rms_norm)
        import phase1_algebra_head as H
        from tokenizers import Tokenizer

        class _Host:
            pass
        host = _Host()
        sd = load_llama_weights(os.path.join(H._ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
        attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
        del sd
        _TRUNK["host"] = host
        _TRUNK["rms"] = _rms_norm
        _TRUNK["tok"] = Tokenizer.from_file(H.TOKENIZER_JSON)
        print("[jury-features] trunk host loaded (CPU, cached for this process)", flush=True)
    return _TRUNK["host"], _TRUNK["rms"], _TRUNK["tok"]


def embed_texts(texts, batch_size=32, log=print):
    """Returns {text: {"states": (T,2048) float32, "offsets": [(a,b),...]}} -- one trunk forward
    pass per batch, batches bucketed by token length (shortest-first) to minimize pad waste, exactly
    the embedding scripts/perceiver_collect.py's _embed_raw_batch computes (embed -> 4 layers ->
    final rms_norm), just not materialized into the dump's fixed T_ALG=256 width."""
    from tinygrad import Tensor, dtypes
    host, rms, tok = get_trunk()
    uniq = list(dict.fromkeys(texts))   # dedup, preserve first-seen order
    encs = [tok.encode(t) for t in uniq]
    order = sorted(range(len(uniq)), key=lambda i: len(encs[i].ids))
    out = {}
    t0 = __import__("time").time()
    for bstart in range(0, len(order), batch_size):
        idxs = order[bstart:bstart + batch_size]
        maxlen = max(1, max(len(encs[i].ids) for i in idxs))
        ids = np.zeros((len(idxs), maxlen), np.int32)
        for bi, i in enumerate(idxs):
            e = encs[i]
            ids[bi, :len(e.ids)] = e.ids
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        x = rms(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        c = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(c).all()
        for bi, i in enumerate(idxs):
            ntok = len(encs[i].ids)
            out[uniq[i]] = dict(states=c[bi, :ntok].astype(np.float32), offsets=list(encs[i].offsets))
        if (bstart // batch_size) % 5 == 0:
            log(f"[jury-features] embedded {min(bstart+batch_size,len(order))}/{len(order)} unique texts "
                f"({__import__('time').time()-t0:.0f}s)")
    log(f"[jury-features] embedded {len(uniq)} unique texts in {__import__('time').time()-t0:.0f}s")
    return out


def pool_states(entry, char_spans=None):
    """Mean-pool entry['states'] over tokens whose char OFFSET overlaps any (a,b) in char_spans;
    char_spans=None (or no token overlaps) -> pool the WHOLE sequence (the 'average face')."""
    states = entry["states"]
    if char_spans:
        offs = entry["offsets"]
        keep = [i for i, (a, b) in enumerate(offs) if b > a and any(not (b <= s or a >= e) for s, e in char_spans)]
        if keep:
            return states[keep].mean(axis=0)
    return states.mean(axis=0)


# ================================================================================================
# 2. THE SPAN -- the discriminating slots' grounding, widened to the enclosing clause (SAM.clause_of,
#    applied to EACH story's own version of each differing slot; union of both).
# ================================================================================================
def _clause_windows(text, parse, asg, diff_slots, bounds, sspans):
    if not parse:
        return []
    intro = SAM.build_intro_map(parse); memo = {}
    sol = list(asg) if asg is not None else []
    spans = []
    for j, f in enumerate(parse):
        if f.get("_slot") not in diff_slots:
            continue
        cl = SAM.clause_of(j, parse, text, bounds, sspans, sol, intro, memo)
        spans.extend(SAM.windows_of(cl, sspans))
    return spans


def discriminating_span(text, parse_a, asg_a, parse_b, asg_b, diff_slots):
    """Returns (char_spans, gran) -- gran='span' iff >=1 slot resolved to a real clause window on
    EITHER side, 'row' (char_spans=[], caller pools the whole text) otherwise -- the same two-tier
    naming courtroom.py's juror1_cert_diff/restricted_cert already use."""
    bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
    spans = (_clause_windows(text, parse_a, asg_a, diff_slots, bounds, sspans)
             + _clause_windows(text, parse_b, asg_b, diff_slots, bounds, sspans))
    spans = sorted(set(tuple(s) for s in spans))
    return spans, ("span" if spans else "row")


# ================================================================================================
# 3. STRUCTURAL DIFFERENCE (A minus B), per differing slot, over the role/op/numeral/clause the two
#    stories bind there.
# ================================================================================================
def _ftype_bucket_counts(parse, diff_slots):
    c = {k: 0 for k in FTYPES}
    for f in parse:
        if f.get("_slot") in diff_slots:
            c[f.get("ftype", "rel")] = c.get(f.get("ftype", "rel"), 0) + 1
    return c


def _op_bucket_counts(parse, diff_slots):
    c = {k: 0 for k in OPS}
    for f in parse:
        if f.get("_slot") in diff_slots and f.get("ftype") == "rel":
            c[f.get("op") if f.get("op") in OPS else None] += 1
    return c


def _given_value_presence(text, parse, diff_slots):
    nums = set(int(x) for x in NUM_RE.findall(text))
    gv = [f for f in parse if f.get("ftype") == "given" and f.get("_slot") in diff_slots and f.get("value") is not None]
    if not gv:
        return 0.0
    return float(np.mean([int(f["value"]) in nums for f in gv]))


def _cross_clause_rate(text, parse, diff_slots, bounds, sspans):
    """Among differing REL slots, the fraction whose arguments' introducing clauses span MORE than
    one distinct sentence (a proxy for 'this story's binding reaches across sentences') -- a cheap,
    direct read of the positional law's own concern (coreference across sentences), computed from
    the parse's own intro map, no new annotation."""
    intro = SAM.build_intro_map(parse); memo = {}
    rels = [f for f in parse if f.get("ftype") == "rel" and f.get("_slot") in diff_slots]
    if not rels:
        return 0.0
    spread = []
    for f in rels:
        sents = set()
        for a in f.get("args", []):
            k = intro.get(a)
            if k is None:
                continue
            cl = SAM.clause_of(k, parse, text, bounds, sspans, [], intro, memo)
            sents |= SAM.sent_set_of(cl, bounds, sspans)
        spread.append(1.0 if len(sents) > 1 else 0.0)
    return float(np.mean(spread))


def structural_diff(text, parse_a, parse_b, diff_slots):
    bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
    fa, fb = _ftype_bucket_counts(parse_a, diff_slots), _ftype_bucket_counts(parse_b, diff_slots)
    oa, ob = _op_bucket_counts(parse_a, diff_slots), _op_bucket_counts(parse_b, diff_slots)
    d = {}
    for k in FTYPES:
        d[f"ftype_{k}"] = float(fa[k] - fb[k])
    for k in OPS:
        d[f"op_{k}"] = float(oa[k] - ob[k])
    d["value_present_diff"] = _given_value_presence(text, parse_a, diff_slots) - _given_value_presence(text, parse_b, diff_slots)
    d["cross_clause_diff"] = _cross_clause_rate(text, parse_a, diff_slots, bounds, sspans) - _cross_clause_rate(text, parse_b, diff_slots, bounds, sspans)
    d["n_diff_slots"] = float(len(diff_slots))
    return d


STRUCT_KEYS = ([f"ftype_{k}" for k in FTYPES] + [f"op_{k}" for k in OPS]
               + ["value_present_diff", "cross_clause_diff", "n_diff_slots"])


# ================================================================================================
# 4. THE WHOLE PAIR FEATURE (structural A-minus-B + the shared caricature embedding), plus the
#    three hand-jurors' own vote (for the baseline/ablation comparisons -- reused, not recomputed).
# ================================================================================================
def pair_record(text, cA, cB, q, trunk_cache):
    pa, pb = cA["parse"], cB["parse"]
    diff_slots = CT.discriminating_slots(pa, pb)
    cert_diff, cert_gran = CT.juror1_cert_diff(text, pa, cA.get("assignment"), pb, cB.get("assignment"), q, diff_slots)
    uA, uB = CT.unexplained_count(text, pa), CT.unexplained_count(text, pb)
    colA, colB = CT.collision_count(pa), CT.collision_count(pb)
    hand_votes = [CT.sign(cert_diff), CT.sign(uB - uA), CT.sign(colB - colA)]   # courtroom's own sign convention

    sdiff = structural_diff(text, pa, pb, diff_slots)
    span_chars, span_gran = discriminating_span(text, pa, cA.get("assignment"), pb, cB.get("assignment"), diff_slots)

    entry = trunk_cache[text]
    text_pool = pool_states(entry, None)
    span_pool = pool_states(entry, span_chars) if span_chars else text_pool
    car_diff = span_pool - text_pool

    struct = dict(sdiff)
    struct["cert_diff"] = cert_diff
    struct["unexplained_diff"] = float(uA - uB)
    struct["collision_diff"] = float(colA - colB)
    struct["gran_is_span"] = 1.0 if (cert_gran == "span" or span_gran == "span") else 0.0

    return dict(struct=struct, span_pool=span_pool.astype(np.float32), car_diff=car_diff.astype(np.float32),
                hand_votes=hand_votes, span_gran=span_gran)


STRUCT_ALL_KEYS = STRUCT_KEYS + ["cert_diff", "unexplained_diff", "collision_diff", "gran_is_span"]


def struct_to_vec(struct):
    return np.asarray([struct[k] for k in STRUCT_ALL_KEYS], np.float32)


def score_pair(text, cA, cB, q, trunk_cache, model):
    """The TRAINED JUROR's own signed score for a pair (positive favors A). Returns
    (score, gran, diff_slots) -- `gran` and `diff_slots` are exactly what courtroom.py's
    cross_examine() needs from a pairwise_votes()-shaped hook."""
    rec = pair_record(text, cA, cB, q, trunk_cache)
    sv = struct_to_vec(rec["struct"])[None]
    Xs = model["pca_span"].transform(rec["span_pool"][None])
    Xc = model["pca_car"].transform(rec["car_diff"][None])
    X = np.concatenate([Xs, Xc, sv], axis=1)
    Xn = model["scaler"].transform(X)
    score = float(model["clf"].decision_function(Xn)[0])
    return score, rec["span_gran"], CT.discriminating_slots(cA["parse"], cB["parse"])
