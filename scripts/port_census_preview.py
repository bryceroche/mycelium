"""port_census_preview.py — THE PORT CENSUS, EVERY PORT (2026-09-08).

PREVIEW: this file is the STAGED form of the port_census.py extension.
The patch that folds it into scripts/port_census.py is delivered beside
it; until the lead applies that patch, scripts/port_census.py keeps its
2026-09-01 behaviour EXACTLY and this file is the one to run.

Measures the antenna per breath: each organ's injection magnitude (vs
the base state of its OWN band), the pre/post gain ratio where both are
recorded, pairwise cosines between organ injections (alignment /
cancellation), and the port SNR. Requires the _CENSUS hook (gen-1:
apply_census_hooks.py, 2026-09-01) EXTENDED by the organ pass
(apply_census_organs.py, 2026-09-08).

WHAT CHANGED vs the 2026-09-01 reader:
  * the three organs the ledger already quotes (breath_emb, notebook,
    garage) are read EXACTLY as before — same accumulation, same rel
    column, same cosine grammar. Nothing about them moves.
  * the pass-1 forward is now given `fact_buf` the way loop_val gives it
    (ALG_ALT2). Without it the alternator injection is SKIPPED and the
    mask head's fact port is zeros — the old census was measuring a
    configuration the machine never runs. This is the instrument fix
    that makes the mask-head reading trustworthy at all.
  * BANDS are named, and each band carries ITS OWN baseline. `state` is
    the loop state (per breath) and the var-slot state (kb=0, where the
    fact injection lands); `state_slot` is the slot mixer's RAW scores at
    their birth — the sc2 the mask head and the alt bias are added to.
    The rel column always divides by the baseline of the organ's OWN
    band, and prints n/a where the band has none (the bank's token
    scores). A LOGIT rms over a STATE rms is a ratio of incommensurables
    and this reader will not print one as if it were a fraction — which
    matters, because the registered mask-head bar is stated as a
    multiple of a base and needs an in-band denominator to be judged by.
    Masked (slot-band) organs also get an rms|open column: those tensors
    are exact zeros on closed lanes, so the plain rms is diluted by the
    mask's sparsity and reads quieter than the organ actually speaks.
  * a GAIN LEDGER: rms(post)/rms(pre) per organ per breath, beside the
    gain scalar read off the checkpoint. A gain's value is not its
    organ's volume (bus_g = 0.0077 and yet the garage injects at
    0.35-0.46x, because the stamps it multiplies ride at 1e3-1e4) —
    post and pre are the two coordinates that make the product legible.

READ THE GRAMMAR AT THE BOTTOM BEFORE QUOTING A NUMBER. In particular:
an h_slot-band injection (the mixer) is further scaled by the breath
gate g = sigmoid(breath_gate[kb]) before it reaches cur, and a
score-band rms is in LOGIT units — it is not commensurate with a
state-band rms and the table never pretends otherwise.

THE INSTRUMENT'S OWN DISTURBANCE, stated: arming _CENSUS forces mid-graph
realize()s, which move kernel boundaries and therefore f32 scheduling
noise. A censused read is NOT bit-identical to an ordinary read — it is
the same read to ~1e-6 (measured; scripts/census_organs_smoke.py GATE 2
pins it against the gen-1 hook's own baseline). This has been true since
2026-09-01; the organ pass does not add a new kind of disturbance. Never
quote a census run as an accuracy read.

Zero training. Zero GPU required (DEV=CPU works; the deployed reads run
on the machine's usual DEV).
Env: PC_CKPT + the organ envs of the artifact being censused.
Optional: PC_N (rows, default 32), PC_TEST/PC_TEST_NAME are the usual
ALG_TEST/ALG_TEST_NAME.
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0)
sd = safe_load(os.environ["PC_CKPT"])
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
missing = sorted(set(p) - set(sd))
if missing: print(f"[census] fresh-init: {missing}")

# the organs that ride a named gain, and the gain each one rides
GAIN_OF = {"maskhead": "mh_gain", "alt": "alt_g", "altfact": "alt2_g",
           "mixer": "fed_mx_hg", "garage": "bus_g", "detwave": "det_g",
           "router(bank)": "r_gain", "notebook2": "fed_nb_g"}

N = int(os.environ.get("PC_N", "32"))
N = min(N, len(vs))
acc = {}          # (kb, organ) -> [vecs (H,)], [rms], [rms|open], [shape]
base_mag = {}     # baseline name -> kb -> [rms]   (each band its own)
base_shape = {}   # baseline name -> kb -> shape   (the band signature)
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, len(vs)))
    pad = 8 - len(sl)
    sl = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    H._CENSUS = []
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    fact_t = None
    if int(os.environ.get("ALG_ALT2", "0")):
        # THE INSTRUMENT FIX (2026-09-08): the deployed read path feeds
        # pass 1 the facts pass 0 parsed (loop_val's convention). Without
        # this the alternator injection never happens and the mask head's
        # fact port is zeros — a census of a configuration nobody runs.
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
        fact_t = Tensor(H.alt2_fact_buf(_oa, vse[sl].astype(np.int32),
                                        _nv, _ma), dtype=dtypes.float)
    H._CENSUS = []
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                fact_buf=fact_t)
    o["pres"].realize()
    for (kb, organ, arr) in H._CENSUS:
        v = arr.mean(axis=(0, 1)) if arr.ndim == 3 else arr.mean(axis=0)
        m = float(np.sqrt((arr ** 2).mean()))
        nz = arr[arr != 0.0]
        mo = float(np.sqrt((nz ** 2).mean())) if nz.size else 0.0
        acc.setdefault((kb, organ), ([], [], [], []))
        acc[(kb, organ)][0].append(v)
        acc[(kb, organ)][1].append(m)
        acc[(kb, organ)][2].append(mo)
        acc[(kb, organ)][3].append(arr.shape)
        if organ in ("state", "state_slot", "state_hslot"):
            base_mag.setdefault(organ, {}).setdefault(kb, []).append(m)
            base_shape.setdefault(organ, {})[kb] = arr.shape
    H._CENSUS = None

BASELINES = ("state", "state_slot", "state_hslot")
organs = sorted({o2 for (_, o2) in acc if o2 not in BASELINES})
kbs = sorted({k for (k, _) in acc})
posts = [o2 for o2 in organs if o2 + "_pre" in organs]
H_W = int(base_shape["state"][max(base_shape["state"])][-1])
L_SLOT = (int(base_shape["state_slot"][max(base_shape["state_slot"])][-1])
          if "state_slot" in base_shape else None)


def band_of(organ):
    """The organ's band, by the LAST AXIS it was recorded on. state = the
    residual space the loop state lives in (H_W); slot = the slot-mixer
    score space (L_TOT x L_TOT, what the mask head and the alt bias bias);
    tok = the bank's token scores (the router's band, which has no
    baseline recorded). Bands never cross-compare."""
    for kb in kbs:
        if (kb, organ) in acc:
            n = acc[(kb, organ)][3][0][-1]
            return ("state" if n == H_W
                    else "slot" if L_SLOT is not None and n == L_SLOT
                    else "tok")
    return "?"


BAND = {o2: band_of(o2) for o2 in organs}
BASE_OF = {"state": "state", "slot": "state_slot"}
# H_SLOT-BAND ORGANS (apply_census_organs2.py, 2026-09-08): the mixer and
# the two ALT21 stations do not add into `cur` — they add into h_slot,
# which reaches the state only through the breath gate. Their denominator
# is the h_slot they add to; dividing them by the loop state answered a
# question nobody asked. Falls back to the band default on a head that
# predates the state_hslot baseline.
BASE_PICK = {"mixer": "state_hslot", "mixer_pre": "state_hslot",
             "alt21_s3": "state_hslot", "alt21_s4": "state_hslot"}
# Organs with NO SCALAR GAIN on their path: their door is a ZERO-INIT
# OUTPUT MATRIX, not an ajar gain, so post IS the whole reading and a
# _pre form would be a fake ratio in a different space. Named here so
# their empty row in the gain ledger reads as a fact, not an omission.
NO_GAIN = ("alt21_s3", "alt21_s4")


def base_rms(kb, organ):
    """The organ's OWN baseline rms at this breath, or None. An explicit
    pick (h_slot-band) wins over the band default."""
    k = BASE_PICK.get(organ)
    if k not in base_mag:
        k = BASE_OF.get(BAND.get(organ, "?"))
    v = base_mag.get(k, {}).get(kb) if k else None
    return float(np.mean(v)) if v else None

print(f"[port census] ckpt={os.environ['PC_CKPT'].split('/')[-1]} "
      f"rows={N} organs={organs}")
print(f"[bands] " + "  ".join(f"{o2}:{BAND[o2]}" for o2 in organs))
_g = []
for o2 in organs:
    k = GAIN_OF.get(o2)
    if k and k in p:
        a = np.abs(p[k].numpy()).reshape(-1)
        _g.append(f"{k}={a.mean():.4g}" + (f"(x{a.size})" if a.size > 1 else ""))
print("[gains] " + "  ".join(_g))

# --------------------------------------------------------------- the table
print("ORGAN INJECTION TABLE — rms and (rel = rms / the SAME BAND's own "
      "baseline at that breath: `state` for state-band, the raw sc2 "
      "`state_slot` for slot-band; n/a where the band has no baseline)")
_w = max(len(o2) for o2 in organs) + 1
print(f"  {'organ':<{_w}} {'band':<6} " +
      " ".join(f"{'b' + str(kb):>16}" for kb in kbs))
for name, kbd in sorted(base_mag.items()):
    cells = [(f"{float(np.mean(kbd[kb])):>8.4g}{'':>8}" if kb in kbd
              else f"{'-':>16}") for kb in kbs]
    print(f"  {'[' + name + ']':<{_w}} {'base':<6} " + " ".join(cells))
for o2 in organs:
    cells = []
    for kb in kbs:
        if (kb, o2) not in acc:
            cells.append(f"{'-':>16}")
            continue
        m = float(np.mean(acc[(kb, o2)][1]))
        base = base_rms(kb, o2)
        cells.append(f"{m:>8.4g}({m / base:>5.2f}x)" if base
                     else f"{m:>8.4g}{'( n/a)':>8}")
    print(f"  {o2:<{_w}} {BAND[o2]:<6} " + " ".join(cells))
_masked = [o2 for o2 in organs if BAND[o2] == "slot"]
if _masked:
    print("  -- slot-band organs, rms over OPEN lanes only (closed lanes "
          "are exact zeros and dilute the plain rms):")
    for o2 in _masked:
        cells = []
        for kb in kbs:
            if (kb, o2) not in acc:
                cells.append(f"{'-':>16}")
                continue
            m = float(np.mean(acc[(kb, o2)][2]))
            base = base_rms(kb, o2)
            cells.append(f"{m:>8.4g}({m / base:>5.2f}x)" if base
                         else f"{m:>8.4g}{'( n/a)':>8}")
        print(f"  {o2:<{_w}} {'[open]':<6} " + " ".join(cells))

# ---------------------------------------------------------- the gain ledger
if posts:
    print("GAIN LEDGER — rms(post)/rms(pre) per breath; the measured effect "
          "of the gain the organ rides (gain x organ-scale is what training "
          "sets; the two are redundant coordinates)")
    for o2 in posts:
        k = GAIN_OF.get(o2)
        gtxt = ""
        if k and k in p:
            a = np.abs(p[k].numpy()).reshape(-1)
            gtxt = f" {k}={a.mean():.4g}" + (f"(x{a.size})" if a.size > 1 else "")
        cells = []
        for kb in kbs:
            if (kb, o2) not in acc or (kb, o2 + "_pre") not in acc:
                cells.append(f"{'-':>10}")
                continue
            rp = float(np.mean(acc[(kb, o2)][1]))
            rq = float(np.mean(acc[(kb, o2 + "_pre")][1]))
            cells.append(f"{(rp / rq if rq > 0 else float('nan')):>10.5g}")
        print(f"  {o2:<{_w}}{gtxt}")
        print(f"  {'':<{_w}} " + " ".join(f"b{kb}:{c}" for kb, c in
                                          zip(kbs, cells)))
    _ng = [o2 for o2 in NO_GAIN if o2 in organs]
    if _ng:
        print(f"  NO SCALAR GAIN on their path (zero-init output matrices, "
              f"not an ajar gain — post IS the whole reading; a _pre would "
              f"be a fake ratio in a different space): {', '.join(_ng)}")

# ------------------------------------------- the 2026-09-01 reading, kept
print("breath | " + " | ".join(f"{o2}: rms(rel)" for o2 in organs))
for kb in kbs:
    cells = []
    for o2 in organs:
        if (kb, o2) in acc:
            m = np.mean(acc[(kb, o2)][1])
            base = base_rms(kb, o2)
            cells.append(f"{o2}:{m:.3f}({m / base:.2f}x)" if base
                         else f"{o2}:{m:.3f}(n/a)")
        else:
            cells.append(f"{o2}:-")
    print(f"  b{kb}:  " + "  ".join(cells))

# -------------------------------------------------------- pairwise cosines
# only POST organs: interference is between what actually ENTERS, and a
# pre-gain tensor is collinear with its own post by construction
organs_c = [o2 for o2 in organs if not o2.endswith("_pre")]
print("pairwise cos (mean over breaths where both live; POST-gain only):")
for i in range(len(organs_c)):
    for j in range(i + 1, len(organs_c)):
        cs = []
        for kb in kbs:
            a = acc.get((kb, organs_c[i])); b = acc.get((kb, organs_c[j]))
            if a and b:
                va = np.mean(a[0], 0); vb = np.mean(b[0], 0)
                if va.shape != vb.shape:
                    cs = None      # different band (e.g. bank-logit space
                    break          # vs state space) — no in-band interference
                cs.append(float(va @ vb / (np.linalg.norm(va)
                                           * np.linalg.norm(vb) + 1e-9)))
        if cs is None:
            print(f"  {organs_c[i]} x {organs_c[j]}: [different band — no "
                  f"in-band interference possible]")
        elif cs:
            print(f"  {organs_c[i]} x {organs_c[j]}: {np.mean(cs):+.3f}")
print("[grammar] |cos|<0.2 & small rel-mags = clean spectrum (nulls "
      "stand); cos<-0.3 = measured cancellation (stacked verdicts get "
      "the interference asterisk)")
print("[grammar] BANDS: state = the residual space the loop state lives "
      "in; slot = the slot-mixer score space (the mask head, the alt "
      "bias), whose baseline is the RAW sc2; tok = the bank's token "
      "scores (the router), which has NO baseline recorded — its rel "
      "column reads n/a and its rms is a bare logit std. Never divide "
      "across bands.")
print("[grammar] h_slot-band organs (mixer, alt21_s3, alt21_s4) divide "
      "by state_hslot — the h_slot they add to — and are then further "
      "scaled by the breath gate g = sigmoid(breath_gate[kb]) before "
      "reaching cur; altfact is recorded at b0 against the VAR-SLOT "
      "state it modifies, not against the loop state.")
print("[grammar] UNCENSUSED still: the sync receiver oscillator "
      "(ALG_SYNC), the BEXIT commit-mass bias (ALG_BEXIT), the per-"
      "forward biases (sixwave sw_g, pmask, FED waist2), the _IMP "
      "systems-ID kick, and the state REPLACEMENTS (stellar / circle / "
      "the pressure seal), which are not injections. Absence from this "
      "table is not a claim of silence.")
