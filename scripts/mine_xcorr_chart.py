"""mine_xcorr_chart.py — THE CORRESPONDENCE CHART's miner (2026-09-19, THE
SURFACE PIVOT, ALG_XCORR): a coordinate-free, era-free PRIOR on where each
slot's FACTOR and VALUE binding lives in the text, mined from GOLD SPANS
over the training diet — NO MODEL INVOLVED. Reads the staged npz (tokmask,
sent, g_fspan, g_vspan, g_res, g_presence) + the mix jsonl (text) DIRECTLY —
load_alg's custody gates exist for TRAINING; mining is read-only over the
same rows and needs none of them.

Per signature s (25 = sentence-count bucket {1,2,3,4,5+} x numeral-count
bucket {0-1,2,3,4,5+}), per slot j (0..L_FAC-1): a (8, 4) table over
(sentence index 0..7 clipped, token kind: numeral/number word/operator
cue/other) of where slot j's FACTOR span (fspan) tokens fall, and a second
table for the VALUE span (vspan of the slot's variable — under the
positional law, the variable index of factor j is g["res"][j]). Both
Laplace-smoothed (0.5) and stored as RAW LOG-ODDS vs uniform (1/32); the
runtime gain (ALG_XCORR / ALG_XCORR_V) multiplies and clips at consumption
(xcorr_row_bias / xcorr_build_array in phase1_algebra_head.py), so one
mined chart serves every gain including 0.0.

Usage: .venv/bin/python3 scripts/mine_xcorr_chart.py [mix.jsonl] [states.npz] [out.npz]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import phase1_algebra_head as ha

MIX = sys.argv[1] if len(sys.argv) > 1 else ".cache/form_mix_pm35.jsonl"
NPZ = sys.argv[2] if len(sys.argv) > 2 else ".cache/phase1_alg_states_formpm35.npz"
OUT = sys.argv[3] if len(sys.argv) > 3 else ".cache/xcorr_chart_formpm35.npz"


def main():
    t0 = time.time()
    samples = [json.loads(l) for l in open(MIX)]
    z = np.load(NPZ)
    tokmask = z["tokmask"]; sent = z["sent"]
    fspan = z["g_fspan"]; vspan = z["g_vspan"]
    res = z["g_res"]; presence = z["g_presence"]
    n = len(samples)
    assert n == len(tokmask) == len(fspan) == len(vspan), \
        (n, len(tokmask), len(fspan), len(vspan))
    L_FAC = ha.L_FAC; T = ha.T_ALG
    n_sig = ha.N_XCORR_SIG
    sb_c, kd_c = ha.XCORR_SENT_BUCKETS, ha.XCORR_KIND_BUCKETS

    fac_counts = np.zeros((n_sig, L_FAC, sb_c, kd_c), np.float64)
    val_counts = np.zeros_like(fac_counts)
    sig_counts = np.zeros(n_sig, np.int64)

    for i in range(n):
        kinds, sig = ha.xcorr_row_features(samples[i]["text"], tokmask[i],
                                           sent[i], T)
        sig_counts[sig] += 1
        sb_all = np.clip(sent[i, :T].astype(np.int64), 0, sb_c - 1)
        for j in range(L_FAC):
            if presence[i, j] < 0.5:
                continue
            fm = np.where(fspan[i, j, :T] > 0.5)[0]
            if len(fm):
                np.add.at(fac_counts[sig, j], (sb_all[fm], kinds[fm]), 1.0)
            v = int(res[i, j])
            if 0 <= v < vspan.shape[1]:
                # THE NUMERAL FILTER (measured, 2026-09-19): gold["vspan"] is
                # the variable's whole MENTION span (often a full clause —
                # "he can carry four cans at once", not just the numeral),
                # so counting every token in it drowns the numeral in
                # surrounding prose ("other" kind >0.9 on a first mining
                # pass). The VALUE table's job is "which sentence carries
                # this variable's number", so only the mention's NUMERAL-
                # kind tokens are counted; a mention with no numeral in it
                # (a pure cross-reference) contributes nothing (falls back
                # to the smoothed uniform for that (sig, slot)).
                vm = np.where((vspan[i, v, :T] > 0.5) & (kinds[:T] == 0))[0]
                if len(vm):
                    np.add.at(val_counts[sig, j], (sb_all[vm], kinds[vm]), 1.0)
        if i and i % 5000 == 0:
            print(f"[xcorr-mine] {i}/{n} rows ({time.time() - t0:.0f}s)",
                  flush=True)

    def normalize(counts):
        tot = counts.sum(axis=(-2, -1), keepdims=True)
        smoothed = (counts + 0.5) / (tot + 0.5 * sb_c * kd_c)
        return np.log(smoothed * (sb_c * kd_c)).astype(np.float32)

    fac_logodds = normalize(fac_counts)
    val_logodds = normalize(val_counts)
    meta = {"version": "xcorr_v1", "date": "2026-09-19", "mix": MIX,
            "states": NPZ, "n_rows": int(n),
            "sent_buckets": "index 0..7 (clipped)",
            "sent_signature_buckets": "{1,2,3,4,5+} sentences",
            "num_signature_buckets": "{0-1,2,3,4,5+} numerals",
            "kinds": ["numeral", "number_word", "operator_cue", "other"],
            "laplace": 0.5, "cells_per_dist": sb_c * kd_c,
            "signature_formula": "sent_bucket*5 + num_bucket"}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    np.savez(OUT, fac=fac_logodds, val=val_logodds,
             sig_counts=sig_counts.astype(np.int64),
             meta=json.dumps(meta))
    print(f"[xcorr-mine] wrote {OUT} in {time.time() - t0:.0f}s; "
          f"signatures populated {int((sig_counts > 0).sum())}/{n_sig}; "
          f"per-signature counts: {sig_counts.tolist()}", flush=True)

    # THE CHART SANITY summary: the 3 most common signatures' slot 0/1/2 top cells.
    order = np.argsort(-sig_counts)[:3]
    kind_names = ["num", "wrd", "cue", "oth"]
    for s in order:
        print(f"[xcorr-mine] sig={s} n={sig_counts[s]} "
              f"(sent_bucket={s // 5} num_bucket={s % 5}):", flush=True)
        for j in range(min(L_FAC, 24)):
            if j not in (0, 1, 2):
                continue
            tab = fac_logodds[s, j]
            flat = np.argsort(-tab.ravel())[:3]
            top = [(a // kd_c, a % kd_c, tab[a // kd_c, a % kd_c]) for a in flat]
            print(f"    slot {j} FACTOR top cells: " +
                  ", ".join(f"(sent={a},kind={kind_names[b]}) lo={c:.2f}"
                           for a, b, c in top), flush=True)
        # mean sentence index per slot for THE CHART SANITY gate (d)
        mean_sent = []
        for j in range(L_FAC):
            tab = fac_logodds[s, j]
            w = np.exp(tab); w = w / w.sum()
            sent_marg = w.sum(-1)  # (8,) marginal over kind
            mean_sent.append(float((sent_marg * np.arange(sb_c)).sum()))
        print(f"    mean FACTOR sentence-index per slot (this sig): " +
              ", ".join(f"{v:.2f}" for v in mean_sent[:8]) + " ...",
              flush=True)


if __name__ == "__main__":
    main()
