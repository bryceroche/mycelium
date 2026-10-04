"""scripts/unet/eval_wild.py — THE U-NET PICTURE BAKE-OFF, the wild
evaluator. A NEW script; no existing script touched. NEVER trains (no
optimizer, no .backward() anywhere in this file) -- MATH-500 and the
wild holdout are MEASURED, never trained on (CLAUDE.md).

WHY WILD GETS A DIFFERENT EVALUATOR THAN THE DIET: `.cache/
wild_admitted_holdout.jsonl`'s `mentions` is `{}` on every row (no hand
mention/arg/cue spans at all -- confirmed by targets.py's own
`__main__` self-check against `.cache/phase1_alg_states_wildhold.npz`,
which carries g_vspan all-zero and no g_aspan/g_cspan key at all). The
ONLY gold signal that survives onto wild is class 1's VALUE end: for a
given slot, the digit-run tokens whose decoded value equals the slot's
gold value (scripts/membrane_scale.py's method, re-derived here via
targets.digit_runs -- see targets.py's module docstring). The MENTION
end (which tokens name the variable) is unknowable on wild, so there is
no (i,j) pixel target to score pixel-accuracy against for class 1, let
alone classes 2/3 (whose both ends are hand spans, entirely absent on
wild).

So the question this file answers is the MARGINAL one the brief poses:
for a given slot, does the trained U-Net's predicted class-1 PROBABILITY
MASS, summed over the row+column of one numeral's tokens, pick out the
GOLD numeral over every OTHER numeral mentioned in the same problem --
i.e. does the picture model LOCATE the right numeral (the same question
scripts/membrane_scale.py's membrane census asks of the trained head's
own attention, just read off this independent organ instead). Mass is
computed over BOTH the row and the column of a run's tokens (the model
is not constrained to produce a symmetric matrix at inference, unlike
its training target) at the FINAL breath (index K_B-1, matching
membrane_scale.py's "final breath = fat_all[-1]" convention).

STANDARD NUMBERS: pixel accuracy is reported for class 1 as this
location proxy; classes 0/2/3 are marked N/A on wild (no pixel target
exists for them there at all -- stated, not silently reported as a
number that would be meaningless). A companion run against the fixture
with full gold (UN_EVAL_DIET=1, any split that carries g_vspan/g_aspan/
g_cspan, e.g. formpm35c's own held-out rows) reports TRUE pixel accuracy
per class for comparison.

ENV
  UN_CKPT       checkpoint path (required unless UN_RANDOM_INIT=1)
  UN_PICTURE    Aprime | Adouble | Adouble_shuf  (must match the ckpt;
                default Adouble)
  UN_BASE / UN_BREATHS / UN_T / UN_SEED   must match training
  UN_N_EVAL     rows to evaluate (default: all 311; smoke uses a few)
  UN_RANDOM_INIT=1   skip ckpt load -- evaluate RANDOM-INIT weights (a
                pure plumbing check; the script prints a loud warning
                and the numbers are stated as meaningless)
  UN_EVAL_DIET=1   evaluate against formpm35c instead of wild (full gold
                available; reports true per-class pixel accuracy)
  UN_WILD_JSONL / UN_WILD_SPLIT   default .cache/wild_admitted_holdout.jsonl / wildhold
"""
import json
import os
import sys

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_state_dict, safe_load, load_state_dict

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/unet")

from pictures import PictureSource, fit_pca, build_picture, n_channels   # noqa: E402
from model import UNet   # noqa: E402
from targets import (build_target, digit_runs, value_of_digits,          # noqa: E402
                      N_CLASSES, CLASS_NAMES, CLASS_GIVEN)


def load_model(un_picture, base, K_B, seed, ckpt, random_init):
    n_where = 0 if un_picture == "Aprime" else 3
    Cin = n_channels(un_picture)
    model = UNet(c_content=Cin - n_where, n_where=n_where, K_B=K_B, base=base, seed=seed)
    if random_init:
        print("[unet/eval_wild] UN_RANDOM_INIT=1 -- evaluating RANDOM weights; "
              "every number below is a PLUMBING CHECK ONLY, not a result.",
              flush=True)
        return model
    assert ckpt and os.path.exists(ckpt), f"UN_CKPT={ckpt!r} not found (or set UN_RANDOM_INIT=1)"
    load_state_dict(model, safe_load(ckpt))
    print(f"[unet/eval_wild] loaded {ckpt}", flush=True)
    return model


def class1_probs(model, pic, T):
    x = Tensor(pic.reshape(1, *pic.shape), dtype=dtypes.float)
    logits = model.forward_breaths(x)              # (1,K,4,T,T)
    probs = logits.softmax(axis=2)
    final = probs[0, model.K_B - 1, CLASS_GIVEN]    # (T,T), final breath
    return final.numpy()


def run_mass(prob, real_idx, run_toks):
    """Sum of predicted class-1 probability over BOTH the row-block and
    the column-block of `run_toks` (restricted to real tokens), the
    diagonal-overlap counted once."""
    r = np.array(sorted(run_toks))
    row_sum = prob[np.ix_(r, real_idx)].sum()
    col_sum = prob[np.ix_(real_idx, r)].sum()
    overlap = prob[np.ix_(r, r)].sum()
    return float(row_sum + col_sum - overlap)


def eval_wild(model, src, tok, T, n_eval):
    n_located = 0
    n_total = 0
    n_wrong_sentence = 0
    n_wrong_same_sentence = 0
    n_no_textual_match = 0
    idxs = list(range(min(n_eval, src.n)))
    for i in idxs:
        state, tm, se = src.row(i, T=T)
        text = src.text(i)
        _, runs = given_candidates(tok, text, T)
        if not runs:
            continue
        pic = build_picture(state, tm, se, text, *PCA, mode=UN_PICTURE_GLOBAL,
                             rng=np.random.default_rng(i))
        prob = class1_probs(model, pic, T)
        real_idx = np.where(tm[:T] > 0)[0]
        L_FAC = src.gold["presence"].shape[1]
        for j in range(L_FAC):
            if src.gold["presence"][i, j] < 0.5 or int(src.gold["ftype"][i, j]) != 1:
                continue
            value = value_of_digits(src.gold["digits"][i, j])
            gold_runs = [r for r in runs if r[2] == value]
            other_runs = [r for r in runs if r[2] != value]
            if not gold_runs:
                n_no_textual_match += 1
                continue
            n_total += 1
            run_mass_of = lambda r: run_mass(prob, real_idx, set(range(r[0], r[1])))
            masses = [(run_mass_of(r), r, True) for r in gold_runs] + \
                     [(run_mass_of(r), r, False) for r in other_runs]
            masses.sort(key=lambda t_: -t_[0])
            best_mass, best_run, best_is_gold = masses[0]
            if best_is_gold:
                n_located += 1
            else:
                gold_sents = {int(se[r[0]]) for r in gold_runs}
                pred_sent = int(se[best_run[0]])
                if pred_sent in gold_sents:
                    n_wrong_same_sentence += 1
                else:
                    n_wrong_sentence += 1
    wrong = n_total - n_located
    return {
        "n_slots_with_textual_match": n_total,
        "n_no_textual_match": n_no_textual_match,
        "located": n_located,
        "located_rate": n_located / n_total if n_total else float("nan"),
        "wrong": wrong,
        "wrong_same_sentence": n_wrong_same_sentence,
        "wrong_other_sentence": n_wrong_sentence,
        "wrong_sentence_share": (n_wrong_sentence / wrong) if wrong else float("nan"),
    }


def given_candidates(tok, text, T):
    ids = tok.encode(text).ids[:T]
    runs = digit_runs(tok, ids, len(ids))
    return ids, runs


def eval_diet_pixelacc(model, src, tok, T, n_eval):
    """TRUE per-class pixel accuracy (full gold available) -- the
    comparison point for wild's located_rate proxy."""
    correct = np.zeros(N_CLASSES, np.int64)
    total = np.zeros(N_CLASSES, np.int64)
    for i in range(min(n_eval, src.n)):
        state, tm, se = src.row(i, T=T)
        text = src.text(i)
        pic = build_picture(state, tm, se, text, *PCA, mode=UN_PICTURE_GLOBAL,
                             rng=np.random.default_rng(i))
        label, real = build_target(text, tok, src.gold, i, T)
        prob = Tensor(pic.reshape(1, *pic.shape), dtype=dtypes.float)
        logits = model.forward_breaths(prob)
        pred = logits[0, model.K_B - 1].argmax(axis=0).numpy()   # (T,T)
        pixmask = real[:, None] & real[None, :]
        for c in range(N_CLASSES):
            m = (label == c) & pixmask
            total[c] += int(m.sum())
            correct[c] += int(((pred == c) & m).sum())
    return {CLASS_NAMES[c]: (correct[c] / total[c] if total[c] else float("nan"))
            for c in range(N_CLASSES)}, total


if __name__ == "__main__":
    UN_PICTURE_GLOBAL = os.environ.get("UN_PICTURE", "Adouble")
    assert UN_PICTURE_GLOBAL in ("Aprime", "Adouble", "Adouble_shuf")
    base = int(os.environ.get("UN_BASE", "16"))
    K_B = int(os.environ.get("UN_BREATHS", "7"))
    T = int(os.environ.get("UN_T", "256"))
    seed = int(os.environ.get("UN_SEED", "0"))
    n_eval = int(os.environ.get("UN_N_EVAL", "311"))
    random_init = bool(os.environ.get("UN_RANDOM_INIT"))
    ckpt = os.environ.get("UN_CKPT", f".cache/unet_{UN_PICTURE_GLOBAL}.safetensors")

    from tokenizers import Tokenizer
    import phase1_algebra_head as H
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    PCA = fit_pca()
    model = load_model(UN_PICTURE_GLOBAL, base, K_B, seed, ckpt, random_init)

    if os.environ.get("UN_EVAL_DIET"):
        split = os.environ.get("UN_SPLIT", "formpm35c")
        jsonl = os.environ.get("UN_JSONL", ".cache/form_mix_pm35c.jsonl")
        src = PictureSource(split, jsonl)
        acc, totals = eval_diet_pixelacc(model, src, tok, T, n_eval)
        print(f"[unet/eval_wild] DIET pixel accuracy per class ({min(n_eval, src.n)} rows, T={T}):")
        for c in CLASS_NAMES:
            print(f"    {c:12s} acc={acc[c]:.4f}  (n_px gold={totals[list(CLASS_NAMES).index(c)]})")
    else:
        wild_jsonl = os.environ.get("UN_WILD_JSONL", ".cache/wild_admitted_holdout.jsonl")
        wild_split = os.environ.get("UN_WILD_SPLIT", "wildhold")
        src = PictureSource(wild_split, wild_jsonl)
        print(f"[unet/eval_wild] WILD standard numbers ({min(n_eval, src.n)} rows, T={T}):")
        print(f"    class 'none'       pixel accuracy: N/A on wild (no full pixel target exists)")
        print(f"    class 'arg'        pixel accuracy: N/A on wild (no arg_spans in the wild jsonl)")
        print(f"    class 'result_op'  pixel accuracy: N/A on wild (no cue_spans in the wild jsonl)")
        res = eval_wild(model, src, tok, T, n_eval)
        print(f"    class 'given_value' GIVEN-LOCATION accuracy (the class-1 proxy): "
              f"{res['located_rate']:.4f}  ({res['located']}/{res['n_slots_with_textual_match']})")
        print(f"    no-textual-match givens (value never appears as a digit run): "
              f"{res['n_no_textual_match']}")
        print(f"    of the WRONG slots ({res['wrong']}): wrong-sentence share = "
              f"{res['wrong_sentence_share']:.4f}  "
              f"(same-sentence-wrong={res['wrong_same_sentence']}, "
              f"other-sentence={res['wrong_other_sentence']})")
