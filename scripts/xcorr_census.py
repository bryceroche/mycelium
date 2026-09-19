"""xcorr_census.py — THE CORRESPONDENCE CHART's pre/post knob census
(ALG_XCORR_CENSUS=1; the pre/post knob law): mean |Xb| on real tokens vs
mean |raw sc| at the grounding factor-bank call, un-JIT'd, printed once.
Runs a single eager forward() pass over one batch of the fixture named by
ALG_TRAIN/ALG_TRAIN_NAME (defaults to the family env's tiny64 fixture).
Env: the family env + ALG_XCORR=<gain> [ALG_XCORR_V=<gain>] ALG_XCORR_CENSUS=1.
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np


def main():
    import phase1_algebra_head as ha
    from tinygrad import Tensor, dtypes
    assert ha.ALG_XCORR_ON, "ALG_XCORR must be set (this censuses the road's own bias)"
    samples, states, tokmask, gold, sent = ha.load_alg("train")
    n = min(8, len(samples))
    p = ha.build_params(0)
    ts = Tensor(np.ascontiguousarray(states[:n]), dtype=dtypes.half)
    tk = Tensor(tokmask[:n].astype(np.float32), dtype=dtypes.float)
    se = Tensor(sent[:n].astype(np.int32), dtype=dtypes.int)
    chart = ha.xcorr_load_chart()
    xb = np.stack([ha.xcorr_row_bias(samples[i]["text"], tokmask[i], sent[i],
                                     chart, ha.T_ALG, ha.ALG_XCORR_GAIN,
                                     ha.ALG_XCORR_V_GAIN)
                  for i in range(n)]).astype(np.float16)
    xcorr_t = Tensor(xb, dtype=dtypes.half)
    ha._XCORR_CENSUS_SC = []
    ha._XCORR_CENSUS_XB = []
    o = ha.forward(p, ts, tk, se, xcorr=xcorr_t)
    for k in o:
        v = o[k]
        if hasattr(v, "realize"):
            v.realize()
    sc_mean = float(np.mean(ha._XCORR_CENSUS_SC)) if ha._XCORR_CENSUS_SC else float("nan")
    real = tokmask[:n].astype(np.float64)
    xb_mean = float(np.abs(xb.astype(np.float64)).sum() /
                    max(1.0, real.sum() * ha.L_FAC))
    print(f"[xcorr-census] n={n} gain_f={ha.ALG_XCORR_GAIN} "
          f"gain_v={ha.ALG_XCORR_V_GAIN} | mean|raw sc| (pre-bias, grounding "
          f"factor bank) = {sc_mean:.4f} | mean|Xb| (on real tokens) = "
          f"{xb_mean:.4f} | ratio Xb/sc = "
          f"{(xb_mean / sc_mean if sc_mean == sc_mean and sc_mean != 0 else float('nan')):.4f}")
    ha._XCORR_CENSUS_SC = None
    ha._XCORR_CENSUS_XB = None


if __name__ == "__main__":
    main()
