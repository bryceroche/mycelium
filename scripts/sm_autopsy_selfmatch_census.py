"""scripts/sm_autopsy_selfmatch_census.py -- THE SM_241 AUTOPSY (2026-10-09, zero training, two
CPU-only reads of banked/freshly-collected raw-logit dumps; word given after ledger entry "SM_241
... DOES NOT FIRE", 2026-10-09 12:34).

SM_241 trained the self-match term's two scalars to the WRONG sign (w_self=-0.104, w_arg=-0.024;
the build spec wanted both > 0). The entry's READING hypothesis: s_own (the args head's own
predicted membership of a slot's OWN variable among its OWN args, sigmoid of the args bilinear's
diagonal) is not a direction read; it is the SELF-LOOP ERROR -- high on FORWARD slots (where the
args head should put ~0 mass on the slot's own index but doesn't, cleanly) rather than on INVERSE
slots (where, by THE POSITIONAL LAW's reencode_ops signature, the own index IS a correct gold
arg). This script's two tasks (of three in the autopsy; task 2 lives in direction_probe.py's BODY
argument + scripts/sm_autopsy_diet_states.py):

  TASK 1 -- THE s_own CENSUS BY FORM, both bodies (PMS8_241, SM_241), on the 311-row wild holdout:
    mean s_own by form (fwd/inv), AUROC of s_own for inverse-vs-forward, the self-loop DECODE rate
    (own index among the args head's own top-2/dup decode) by form, and where the self-loop mass
    sits on forward forms (percentile rank of the own-index logit among the 24 candidates).

  TASK 3 -- THE TERM'S ACTUAL EFFECT, SM_241 only: the term's own contribution to the res logit
    (w_self*s_own at the own index; w_arg*s_own*a_j at every other candidate j) vs the res logit
    spread (std across the 24 candidates) at the SAME slot, by form -- did the term ever reach the
    scale of the decision (THE PRE/POST KNOB LAW, CLAUDE.md S4)?

Inputs (both already raw args/res LOGITS, pre-sigmoid, chain_acc.py's CA_RAWDUMP convention --
d["args"][j] / d["res"][j], (24,) per gold slot j; d["key"] unused here):
  PMS8_241: .cache/rawslots_wild_PMS8_241.pkl           (banked, unmodified, read-only)
  SM_241:   .cache/sm_autopsy_rawslots_wild_SM_241.pkl  (this autopsy's own collection,
            .cache/sm_autopsy_collect.sh step 1)
Gold factors + classify_row (THE POSITIONAL LAW's own fwd/inv/other labeler): polarity_census.py,
read-only import, not reimplemented.

usage: .venv/bin/python3 scripts/sm_autopsy_selfmatch_census.py
outputs: .cache/sm_autopsy_selfmatch_census.txt
"""
import json
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

from polarity_census import classify_row

WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
DUMPS = {"PMS8_241": ".cache/rawslots_wild_PMS8_241.pkl", "SM_241": ".cache/sm_autopsy_rawslots_wild_SM_241.pkl"}
SM_W_SELF, SM_W_ARG = -0.10384019, -0.02444939   # read off .cache/sharp_SM_241.safetensors (exact)
OUT = ".cache/sm_autopsy_selfmatch_census.txt"

LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def auroc(scores, labels):
    """labels: 1 = positive class (inverse). Mann-Whitney U / rank form, no sklearn dependency needed
    but reuse it if present for a cross-check isn't necessary -- this is exact."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average-rank correction for ties via scipy-free approach: use the standard U statistic from
    # sorting with ties handled by rankdata-equivalent (ties get average rank)
    all_scores = np.concatenate([pos, neg])
    sorter = np.argsort(all_scores, kind="mergesort")
    ranks_sorted = np.empty(len(all_scores))
    sorted_scores = all_scores[sorter]
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        ranks_sorted[i:j] = avg_rank
        r += (j - i)
        i = j
    ranks = np.empty(len(all_scores))
    ranks[sorter] = ranks_sorted
    rank_pos = ranks[:len(pos)]
    u = rank_pos.sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def load_dump(tag):
    D = pickle.load(open(DUMPS[tag], "rb"))
    return {int(d["i"]): d for d in D}


def main():
    rows = [json.loads(l) for l in open(WILD_JSONL)]
    P("THE SM_241 AUTOPSY -- s_own CENSUS + THE TERM'S ACTUAL EFFECT (2026-10-09, zero training)")
    P(f"wild fixture: {WILD_JSONL} ({len(rows)} rows)")
    P(f"sm_w_self={SM_W_SELF:.6f} sm_w_arg={SM_W_ARG:.6f} (read off .cache/sharp_SM_241.safetensors)")

    per_body = {}
    for tag in ("PMS8_241", "SM_241"):
        d = load_dump(tag)
        s_own_fwd, s_own_inv = [], []
        selfloop_fwd, selfloop_inv = [], []       # 1.0 if own index decoded among the args (top-2/dup rule)
        rank_fwd, rank_inv = [], []               # percentile rank (0=lowest,1=highest) of own-index logit among 24
        term_contrib_own = []                     # SM_241 only: |w_self*s_own| at own index
        term_contrib_arg = []                      # SM_241 only: max |w_arg*s_own*a_j| over j!=own
        res_spread = []                             # std of the res logit vector at this slot
        forms = []
        for i, r in enumerate(rows):
            if i not in d:
                continue
            factors = r["factors"]
            cls = classify_row(factors)
            dd = d[i]
            for k, f in enumerate(factors):
                c = cls[k]
                if c not in ("fwd", "inv"):
                    continue
                args_logit = dd["args"][k].astype(np.float64)   # (24,) raw logits
                s_own = float(sigmoid(args_logit[k:k+1])[0])
                rank = float((args_logit < args_logit[k]).sum()) / (len(args_logit) - 1)   # 0=lowest of 24
                if "dup" in dd and dd["dup"][k] > 0:
                    decoded = {int(np.argmax(args_logit))}
                else:
                    decoded = set(np.argsort(-args_logit)[:2].tolist())
                selfloop = float(k in decoded)
                res_logit = dd["res"][k].astype(np.float64)
                res_spread.append(float(res_logit.std()))
                forms.append(c)
                if c == "fwd":
                    s_own_fwd.append(s_own); selfloop_fwd.append(selfloop); rank_fwd.append(rank)
                else:
                    s_own_inv.append(s_own); selfloop_inv.append(selfloop); rank_inv.append(rank)
                if tag == "SM_241":
                    a_j = sigmoid(args_logit)
                    boost = SM_W_ARG * s_own * a_j
                    boost[k] = 0.0
                    term_contrib_own.append(abs(SM_W_SELF * s_own))
                    term_contrib_arg.append(float(np.max(np.abs(boost))))

        per_body[tag] = dict(s_own_fwd=np.array(s_own_fwd), s_own_inv=np.array(s_own_inv),
                              selfloop_fwd=np.array(selfloop_fwd), selfloop_inv=np.array(selfloop_inv),
                              rank_fwd=np.array(rank_fwd), rank_inv=np.array(rank_inv),
                              res_spread=np.array(res_spread), forms=np.array(forms))
        if tag == "SM_241":
            per_body[tag]["term_contrib_own"] = np.array(term_contrib_own)
            per_body[tag]["term_contrib_arg"] = np.array(term_contrib_arg)

        P(f"\n{'='*78}\nTASK 1 -- THE s_own CENSUS BY FORM -- {tag}\n{'='*78}")
        n_fwd, n_inv = len(s_own_fwd), len(s_own_inv)
        P(f"  n_fwd={n_fwd} n_inv={n_inv}")
        P(f"  mean s_own: fwd={np.mean(s_own_fwd):.4f} (sd {np.std(s_own_fwd):.4f})  "
          f"inv={np.mean(s_own_inv):.4f} (sd {np.std(s_own_inv):.4f})")
        scores = np.concatenate([per_body[tag]["s_own_fwd"], per_body[tag]["s_own_inv"]])
        labels = np.concatenate([np.zeros(n_fwd, bool), np.ones(n_inv, bool)])
        au = auroc(scores, labels)
        P(f"  AUROC(s_own; inverse=positive) = {au:.4f}  "
          f"({'s_own separates inverse FROM forward' if au > 0.6 else ('s_own separates FORWARD from inverse (INVERTED)' if au < 0.4 else 'no separation')})")
        P(f"  self-loop DECODE rate (own index in the args head's own top-2/dup decode): "
          f"fwd={np.mean(selfloop_fwd):.4f} ({int(np.sum(selfloop_fwd))}/{n_fwd})  "
          f"inv={np.mean(selfloop_inv):.4f} ({int(np.sum(selfloop_inv))}/{n_inv})")
        P(f"  own-index logit PERCENTILE RANK among the 24 candidates (0=lowest,1=highest): "
          f"fwd mean={np.mean(rank_fwd):.4f} median={np.median(rank_fwd):.4f}  "
          f"inv mean={np.mean(rank_inv):.4f} median={np.median(rank_inv):.4f}")
        # where the self-loop mass sits ON FORWARD forms specifically -- histogram of rank_fwd
        bins = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0 + 1e-9]
        hist, _ = np.histogram(rank_fwd, bins=bins)
        P(f"  forward-form own-index rank histogram (bottom half / .5-.8 / .8-.9 / .9-.95 / top 5%): "
          f"{hist.tolist()} (n={n_fwd}) -- top-5%-rank share = {hist[-1]/max(n_fwd,1):.4f}")

    P(f"\n{'='*78}\nTASK 3 -- THE TERM'S ACTUAL EFFECT -- SM_241\n{'='*78}")
    d = per_body["SM_241"]
    forms = d["forms"]
    is_fwd = forms == "fwd"
    is_inv = forms == "inv"
    for label, mask in (("fwd", is_fwd), ("inv", is_inv), ("all", np.ones_like(is_fwd, bool))):
        own = d["term_contrib_own"][mask]
        arg = d["term_contrib_arg"][mask]
        spread = d["res_spread"][mask]
        ratio_own = own / np.maximum(spread, 1e-9)
        ratio_arg = arg / np.maximum(spread, 1e-9)
        P(f"  [{label}, n={mask.sum()}] |w_self*s_own| (own idx): mean={own.mean():.5f} max={own.max():.5f} | "
          f"max_j|w_arg*s_own*a_j|: mean={arg.mean():.5f} max={arg.max():.5f} | "
          f"res-logit std (the decision's own spread): mean={spread.mean():.4f}")
        P(f"           ratio to res spread: own/spread mean={ratio_own.mean():.4f} max={ratio_own.max():.4f} "
          f"(>=1.0 would mean the term alone can flip a close decision) | "
          f"arg/spread mean={ratio_arg.mean():.4f} max={ratio_arg.max():.4f}")

    P(f"\n{'='*78}\nTHE VERDICT\n{'='*78}")
    pm = per_body["PMS8_241"]; sm = per_body["SM_241"]
    au_pm = auroc(np.concatenate([pm["s_own_fwd"], pm["s_own_inv"]]),
                  np.concatenate([np.zeros(len(pm["s_own_fwd"]), bool), np.ones(len(pm["s_own_inv"]), bool)]))
    au_sm = auroc(np.concatenate([sm["s_own_fwd"], sm["s_own_inv"]]),
                  np.concatenate([np.zeros(len(sm["s_own_fwd"]), bool), np.ones(len(sm["s_own_inv"]), bool)]))
    P(f"1. s_own AUROC for inverse-vs-forward: PMS8_241={au_pm:.4f}, SM_241={au_sm:.4f} -- "
      f"{'both READ FORWARD (s_own is higher on FORWARD slots, the self-loop-error reading)' if au_pm < 0.45 and au_sm < 0.45 else 'see the raw numbers above; not a clean confirmation either way'}.")
    P(f"2. mean s_own: PMS8_241 fwd={pm['s_own_fwd'].mean():.4f} vs inv={pm['s_own_inv'].mean():.4f}; "
      f"SM_241 fwd={sm['s_own_fwd'].mean():.4f} vs inv={sm['s_own_inv'].mean():.4f}.")
    own_ratio_mean = (d["term_contrib_own"] / np.maximum(d["res_spread"], 1e-9)).mean()
    arg_ratio_mean = (d["term_contrib_arg"] / np.maximum(d["res_spread"], 1e-9)).mean()
    P(f"3. THE TERM'S SCALE: mean own/spread ratio={own_ratio_mean:.4f}, mean max-arg/spread ratio={arg_ratio_mean:.4f} "
      f"-- {'the term is a real fraction of the decision scale (not a knob-law null-by-smallness)' if max(own_ratio_mean, arg_ratio_mean) > 0.05 else 'the term is small relative to the res spread -- a possible knob-law contributor beside the wrong sign'}.")
    P("4. READING: see the printed numbers above for the full by-form breakdown; this script makes "
      "no verdict call beyond the two lines above -- the ledger entry closes the term on the sign, "
      "this is the autopsy's supporting census.")

    with open(OUT, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print(f"\n[sm_autopsy] wrote {OUT}")


if __name__ == "__main__":
    main()
