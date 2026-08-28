"""r4_stage0_phase_roles.py — R4 STAGE 0 (2026-08-29): the marriage scout.
The six-wave carriers key slots to sentences by phase resonance
(cos(phi_slot - theta_sent), theta = (sent mod 6) * 60deg, gate sw_g open at
~0.46). The bus binds roles by phase rotation. Same physics — so BEFORE
designing role-offsets-on-carriers, MEASURE: do arg->res relations already
have CONCENTRATED sentence-phase strides? For every factor whose result var
is MENTIONED in a different sentence than an arg var (cross-sentence
wiring), collect dphase = (sent(res_site) - sent(arg_site)) mod 6.

REGISTERED PREDICTION (pinned): the dphase distribution deviates from the
within-row shuffle null (chi^2, p < 0.01) on mint (form8) AND on the wild
golds. Mint-only -> dialect artifact (carriers march because the generator
marches); both -> the carriers are proto-role-rotators and the marriage
inherits MEASURED offsets. Zero-GPU: texts + mentions + gold wiring only.
"""
import json, glob, sys, os
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter

os.environ.setdefault("ALG2", "1")
from phase1_algebra_head import TOKENIZER_JSON, sent_indices, T_ALG
from tokenizers import Tokenizer

tok = Tokenizer.from_file(TOKENIZER_JSON)
rng = np.random.default_rng(4)


def sent_of_sites(text):
    e = tok.encode(text)
    msk = np.zeros(T_ALG, np.float32)
    msk[:min(len(e.ids), T_ALG)] = 1.0
    snt = sent_indices(text, list(e.offsets), msk)
    return e, snt


def var_sent(e, snt, text, tokpos):
    return int(snt[tokpos]) if tokpos < T_ALG else None


def collect(rows, label, max_rows=4000):
    """dphase counts over cross-sentence arg->res pairs + shuffle null."""
    obs = Counter(); null = Counter()
    used = 0
    for r in rows[:max_rows]:
        text = r.get("text") or r.get("original") or ""
        men = r.get("mentions") or {}
        if not text or not men:
            continue
        e, snt = sent_of_sites(text)
        # mentions: var(str)->[charpos,...]; map char->token->sentence
        def sents_for(v):
            out = set()
            for cp in men.get(str(v), []):
                if isinstance(cp, (list, tuple)):
                    cp = cp[0]
                for t, (a, b) in enumerate(e.offsets[:T_ALG]):
                    if a <= cp < b or (a == b == cp):
                        out.add(int(snt[t])); break
            return out
        pairs = []
        for f in r.get("factors", []):
            if f.get("ftype") != "rel":
                continue
            res = f.get("result"); args = f.get("args", [])
            rs = sents_for(res)
            for a in args:
                for sa in sents_for(a):
                    for sr in rs:
                        if sa != sr:
                            pairs.append((sa, sr))
        if not pairs:
            continue
        used += 1
        allsents = sorted({s for p2 in pairs for s in p2})
        for sa, sr in pairs:
            obs[(sr - sa) % 6] += 1
            # null: shuffle sentence identities within the row
            perm = dict(zip(allsents, rng.permutation(allsents)))
            null[(perm[sr] - perm[sa]) % 6] += 1
    n_o = sum(obs.values()); n_n = sum(null.values())
    print(f"[{label}] rows-with-cross-sentence-wiring {used}, pairs {n_o}")
    if n_o < 30:
        print(f"[{label}] UNDER-POWERED (<30 pairs) — no verdict")
        return None
    exp = np.array([null.get(k, 0) for k in range(6)], float)
    exp = exp / max(exp.sum(), 1) * n_o
    o = np.array([obs.get(k, 0) for k in range(6)], float)
    chi2 = float(((o - exp) ** 2 / np.maximum(exp, 1)).sum())
    # chi2 df=5: p<0.01 at 15.09
    print(f"[{label}] dphase obs {o.astype(int).tolist()} "
          f"null-exp {exp.round(1).tolist()} chi2={chi2:.1f} "
          f"(df=5, p<0.01 at 15.09) -> "
          f"{'CONCENTRATED' if chi2 > 15.09 else 'UNIFORM-COMPATIBLE'}")
    return chi2


def main():
    mints = [json.loads(l) for i, l in
             zip(range(6000), open('.cache/form_mix8.jsonl'))]
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f):
            r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    golds = [dict(v, text=v.get("original")) for k, v in sorted(byid.items())
             if k not in sk]
    c_m = collect(mints, "mint")
    c_g = collect(golds, "gold")
    if c_m is not None and c_g is not None:
        both = c_m > 15.09 and c_g > 15.09
        print(f"[verdict] prediction (both concentrated): "
              f"{'CONFIRMED — carriers are proto-role-rotators; the marriage '
                 'inherits measured offsets' if both else 'NOT confirmed'}")


if __name__ == "__main__":
    main()
