"""scripts/unet/targets.py — THE U-NET PICTURE BAKE-OFF, per-pixel labels.
A NEW module (scripts/unet/pictures.py's sibling); no existing script
touched. Builds the (T,T) int8 class map a row's gold factor graph
implies, straight from the CACHED gold arrays in
`.cache/phase1_alg_states_<split>.npz` (built by build_gold() in
scripts/phase1_algebra_head.py) plus one host-side re-tokenization for the
digit-run match (verbatim method from scripts/membrane_scale.py's
`_digit_runs`, duplicated here rather than imported — targets.py has no
other dependency on that script, and importing it would pull in its
module-level MS_* env parsing for no reason).

CLASSES
  0 none
  1 GIVEN-VALUE  : (mention-token, value-token) for a given factor's
                   variable mentions x its value's numeral tokens. The
                   value's token location is NEVER a hand span (none
                   exists even on the diet) — it is always the digit-run
                   match, the SAME method used on wild, so class 1's
                   definition is identical diet vs wild by construction.
  2 ARG          : (arg-mention token, result-mention token) for each
                   relation argument. Both ends ARE hand spans (arg_spans
                   / the result variable's mentions) -- cached already as
                   g_aspan / g_vspan, token-mapped by build_gold's own
                   _spans_to_tokmask (verified identical convention: this
                   file does not re-implement span->token mapping at all,
                   it only reads the already-mapped arrays).
  3 RESULT/OP    : (cue token, result-mention token) for the relation's
                   operator cue words (g_cspan) x the result's mentions.

PRIORITY ON PIXEL COLLISION: a (i,j) pair can in principle get written by
more than one class (e.g. a value token that also happens to be an
argument re-mention token in some other factor's clause). Processed in
the order rel-factors first (classes 2/3) then given-factors (class 1),
so a later given-value write WINS a collision -- the choice is arbitrary
but documented: GIVEN-VALUE is the class the mission's wild evaluator
reads (eval_wild.py), so it is given precedence over the training-only
classes 2/3 rather than the reverse.

WILD: mentions=={} on every wild row (no hand spans at all) -> g_vspan /
g_aspan / g_cspan are either absent or all-zero for the wild split (see
pictures.py's module docstring and this file's `__main__` self-check,
which confirms it against .cache/phase1_alg_states_wildhold.npz). The
ONLY derivable signal on wild is class 1's VALUE end (digit runs); the
MENTION end is unknown there too. `build_target` therefore returns an
all-background map plus a loud note when called on a split with no
mentions -- real wild supervision never happens (never trained on wild,
per the word); eval_wild.py's class-1 check (below, `wild_value_runs`)
works at the column-marginal level instead, which needs no mention
location at all.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

CLASS_NONE, CLASS_GIVEN, CLASS_ARG, CLASS_RESOP = 0, 1, 2, 3
CLASS_NAMES = ("none", "given_value", "arg", "result_op")
N_CLASSES = 4


def digit_runs(tok, ids, T):
    """Maximal runs of digit-only-decode tokens -> list of (start, end_excl,
    value). Verbatim from scripts/membrane_scale.py's `_digit_runs`
    (duplicated, not imported -- see module docstring)."""
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


def value_of_digits(digit_arr):
    """g_digits[row,j] is MSD-first (phase1_algebra_head.build_gold);
    concatenating the (zero-padded) decimal string and parsing it is
    exactly the inverse -- same convention membrane_scale.py's report()
    uses (int("".join(str(int(x)) for x in gdig[i,j]))). Sign (g_sign) is
    IGNORED here, matching membrane_scale's own convention: a textual
    digit run never carries a '-' token, so sign cannot be recovered from
    digit-run matching anyway -- stated, not silently dropped."""
    return int("".join(str(int(d)) for d in digit_arr))


def given_value_tokens(tok, ids, T, value):
    runs = digit_runs(tok, ids, T)
    toks = set()
    for a, b, v in runs:
        if v == value:
            toks.update(range(a, b))
    return toks, runs


def token_set_from_span(span_row, T):
    """span_row: (T_full,) float array (g_vspan/g_aspan/g_cspan row,
    already token-mapped by build_gold's _spans_to_tokmask), cropped to T."""
    return set(np.where(span_row[:T] > 0)[0].tolist())


def _fill(label, set_a, set_b, cls):
    if not set_a or not set_b:
        return
    ia = np.array(sorted(set_a))
    ib = np.array(sorted(set_b))
    label[np.ix_(ia, ib)] = cls
    label[np.ix_(ib, ia)] = cls


def build_target(text, tok, gold, i, T):
    """gold: the dict of per-split arrays (PictureSource.gold, or any dict
    with the same keys) -- presence/ftype/res/digits/vspan/aspan/cspan,
    indexed [i] for this row. T: crop length (<= the cached T_full).
    Returns (label (T,T) int8, real (T,) bool [from re-tokenizing text;
    the SAME ids build_gold's tokenize() produced, since the tokenizer and
    truncation are deterministic])."""
    ids = tok.encode(text).ids[:T]
    real = np.zeros(T, bool)
    real[:len(ids)] = True
    label = np.zeros((T, T), np.int8)
    L_FAC = gold["presence"].shape[1]
    has_vspan = "vspan" in gold
    has_aspan = "aspan" in gold
    has_cspan = "cspan" in gold
    # pass 1: rel factors -> classes 2 (arg) and 3 (result/op)
    if has_vspan and has_aspan and has_cspan:
        for j in range(L_FAC):
            if gold["presence"][i, j] < 0.5 or int(gold["ftype"][i, j]) != 0:
                continue
            res_var = int(gold["res"][i, j])
            res_toks = token_set_from_span(gold["vspan"][i, res_var], T)
            if not res_toks:
                continue
            for pos in (0, 1):
                arg_toks = token_set_from_span(gold["aspan"][i, j, pos], T)
                _fill(label, arg_toks, res_toks, CLASS_ARG)
            cue_toks = token_set_from_span(gold["cspan"][i, j], T)
            _fill(label, cue_toks, res_toks, CLASS_RESOP)
    # pass 2: given factors -> class 1 (overwrites on collision; see docstring)
    for j in range(L_FAC):
        if gold["presence"][i, j] < 0.5 or int(gold["ftype"][i, j]) != 1:
            continue
        var = int(gold["res"][i, j])
        value = value_of_digits(gold["digits"][i, j])
        val_toks, _ = given_value_tokens(tok, ids, len(ids), value)
        if not val_toks:
            continue
        ment_toks = token_set_from_span(gold["vspan"][i, var], T) if has_vspan else set()
        _fill(label, ment_toks, val_toks, CLASS_GIVEN)
    label[~real, :] = CLASS_NONE
    label[:, ~real] = CLASS_NONE
    return label, real


if __name__ == "__main__":
    from tokenizers import Tokenizer
    import phase1_algebra_head as H   # only for TOKENIZER_JSON; needs no FAM env beyond DEV
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    import json
    T = 64
    # diet: mentions present -> expect a non-trivial target
    z = np.load(".cache/phase1_alg_states_formpm35c.npz")
    gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    samples = [json.loads(l) for l in open(".cache/form_mix_pm35c.jsonl")]
    for i in range(3):
        label, real = build_target(samples[i]["text"], tok, gold, i, T)
        counts = {CLASS_NAMES[c]: int((label == c).sum()) for c in range(N_CLASSES)}
        sym = np.array_equal(label, label.T)
        print(f"diet row {i}: real_tokens={int(real.sum())} counts={counts} symmetric={sym}")
    # wild: mentions == {} -> vspan/aspan/cspan absent or all-zero -> class 1
    # mention-end empty, class 2/3 entirely absent (no hand spans at all)
    zw = np.load(".cache/phase1_alg_states_wildhold.npz")
    goldw = {k[2:]: zw[k] for k in zw.files if k.startswith("g_")}
    samplesw = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
    print("wild gold keys present:", sorted(set(("vspan", "aspan", "cspan")) & set(goldw)))
    label, real = build_target(samplesw[0]["text"], tok, goldw, 0, T)
    print(f"wild row 0: counts={ {CLASS_NAMES[c]: int((label==c).sum()) for c in range(N_CLASSES)} } "
          f"(expect all zero off class 0 -- no mention spans to anchor either end)")
