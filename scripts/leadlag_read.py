"""leadlag_read.py — THE LEAD/LAG NEEDLE (2026-09-05, the paired atlas;
ledger "REGISTERED + WORD GIVEN: THE PAIRED ATLAS", instrument (3)).
Zero-training GPU READ: loads the two-chart atlas + a checkpoint, runs
the miner's two-pass capture over a fixture, and asks the registered
question — does the READING lead the COMMITMENT?

Per item, per breath k (K_STEPS pages):
  (i)  NL contraction curve  — cosine distance of the item's breath-k
       NL state (attention-pooled waisted token read, the NL tap) to
       its class's nl page k; plus the attention ENTROPY over tokens
       per breath (the reading's spread — narrowing = unveiling).
  (ii) math contraction curve — same against the math (slot-state)
       pages, mean-pooled over the 24 slots (the miner's own pooling).
  (iii) THE NEEDLE — Pearson cross-correlation of the two DIFFERENCE
       curves (first differences: events, not levels) at lags -3..+3
       breaths, per class and overall, via
       mycelium.step_atlas.leadlag_needle (one needle, every caller).
       POSITIVE lag = reading leads commitment ("the telegraph heard
       through a stethoscope, not a wall").

Class labels come from atlas_class(row.gen) — the fixture read is a
MEASUREMENT against the mined kinds, not a retrieval test (cross_prior
has its own feed path). Items whose class is absent from the atlas (or
has any empty page) are counted and skipped, never zero-filled.

GOODHART NOTE (the diagnostic register's clause): every number this
script prints — contraction curves, attention entropy, the needle —
is DIAGNOSTIC ONLY. None of it may ever enter a loss, a training
signal, or a selection criterion for training data: a supervised
needle teaches the clock to lie (a monitored signal in the loss
teaches concealment, not cure). Read it; never train toward it.

Envs: LL_TEST = wildhold (default) | test23 | a jsonl path;
LL_CKPT (default .cache/sharp_port242.safetensors); LL_N (default 0 =
whole fixture; >0 = first N rows, deterministic); LL_ATLAS /
MH_ATLAS_MANIFEST (default the research-manifest pair). Champion env
stack via setdefault — caller/chain envs WIN (the wave_field_check
lesson). Requires ALG_MINE_BREATHS taps (apply_nl_tap.py applied) and
a PAIRED atlas (paired miner re-mine). GPU script — STAGED, NOT RUN at
build time (hold for the word); import is CPU-safe (all work under
__main__/main()).
"""
import os
import sys

_FIXTURES = {
    "wildhold": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
    "test23": (".cache/algebra_nl_test.jsonl", "test23"),
}
_ll = os.environ.get("LL_TEST", "wildhold")
_llpath, _llname = _FIXTURES.get(_ll, (_ll, os.path.basename(_ll)))

# Champion env stack — VERBATIM from the round-2 chain, via setdefault
# so a chain's explicit envs win; ALG_TEST comes from LL_TEST.
_CHAMPION_ENVS = {
    "DEV": "PCI+AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512.npz",
    "ALG_TEST": _llpath, "ALG_TEST_NAME": _llname,
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "SC_EVAL": "0", "ALG_ALT21": "1", "ALG_ALT2": "1",
    # the taps: breaths_all (math) + nl_all/nlat_all (the NL tap)
    "ALG_MINE_BREATHS": "1",
}
for _k, _v in _CHAMPION_ENVS.items():
    os.environ.setdefault(_k, _v)

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

CKPT = os.environ.get("LL_CKPT", ".cache/sharp_port242.safetensors")
ATLAS = os.environ.get("LL_ATLAS", ".cache/step_atlas_current.npz")
MANIFEST = os.environ.get("MH_ATLAS_MANIFEST",
                          ".cache/RESEARCH_MANIFEST.json")
LL_N = int(os.environ.get("LL_N", "0"))
MAX_LAG = 3


def main():
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf,
                                     K_VARS)
    from mycelium.step_atlas import (load_atlas, atlas_class,
                                     leadlag_needle, K_STEPS)

    atlas = load_atlas(ATLAS, manifest_path=MANIFEST)
    assert atlas["nl_means"] is not None, \
        ("leadlag_read needs the PAIRED atlas (nl chart) — re-mine with "
         "the paired miner (single-chart atlas refused loudly)")
    acls = {c: i for i, c in enumerate(atlas["classes"])}

    samples, states, tokmask, gold, sent = load_alg("test")
    n_all = len(samples)
    n_take = min(LL_N, n_all) if LL_N > 0 else n_all
    take = np.arange(n_take)
    print(f"[leadlag] fixture={_llname} rows={n_take}/{n_all} "
          f"ckpt={CKPT} atlas={ATLAS} (era {atlas['stamp']})", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def cosdist(a, b):
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return 1.0 - float(np.dot(a, b) / (na * nb))

    per_class = {}     # cls -> [nl_curves], [math_curves], [ent_curves]
    n_skip = 0
    for s0 in range(0, len(take), 8):
        sl = take[s0:s0 + 8]
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        # the miner's deployable two-pass cycle (loop_val's shape)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, sent[sl_p].astype(np.int32))
        _ka = (("pres", "ftype", "op", "dig")
               + (("dup",) if "dup" in o0 else ()))
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([samples[int(i)].get("n_vars", K_VARS)
                        for i in sl_p])
        _ma = np.array([samples[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, sent[sl_p], _nv, _ma)
        o = forward(p, ts, tk, se,
                    slot_mask=Tensor(mk, dtype=dtypes.float),
                    fact_buf=Tensor(fb, dtype=dtypes.float))
        br = [b.realize().numpy() for b in o["breaths_all"]]
        nl = [t.realize().numpy() for t in o["nl_all"]]
        nat = [t.realize().numpy() for t in o["nlat_all"]]
        assert len(br) == len(nl) == len(nat) == K_STEPS
        for bi, i in enumerate(sl):        # pads (bi >= len(sl)) skipped
            cls = atlas_class(samples[int(i)].get("gen"))
            ci = acls.get(cls)
            if ci is None or (atlas["counts"][:, ci] == 0).any() \
                    or (atlas["nl_counts"][:, ci] == 0).any():
                n_skip += 1
                continue
            nl_c = np.zeros(K_STEPS)
            mt_c = np.zeros(K_STEPS)
            en_c = np.zeros(K_STEPS)
            for k in range(K_STEPS):
                nl_c[k] = cosdist(nl[k][bi], atlas["nl_means"][k, ci])
                mt_c[k] = cosdist(br[k][bi].mean(0),
                                  atlas["means"][k, ci])
                w = np.clip(nat[k][bi], 1e-12, 1.0)
                en_c[k] = float(-(w * np.log(w)).sum())
            cur = per_class.setdefault(cls, ([], [], []))
            cur[0].append(nl_c)
            cur[1].append(mt_c)
            cur[2].append(en_c)
        if (s0 // 8) % 32 == 0:
            print(f"[leadlag] {s0 + len(sl)}/{len(take)}", flush=True)

    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    hdr = "  ".join(f"{int(l):+d}".rjust(6) for l in lags)
    print(f"[leadlag] skipped {n_skip} (class off-atlas/empty pages)")
    print(f"[leadlag] needle: corr(nl_diff[k], math_diff[k+lag]) — "
          f"POSITIVE lag = reading LEADS commitment")
    print(f"[leadlag] {'class':12s} {'n':>5s}  {hdr}   peak")
    all_nl, all_mt = [], []
    for cls in sorted(per_class):
        nlc = np.stack(per_class[cls][0])
        mtc = np.stack(per_class[cls][1])
        enc = np.stack(per_class[cls][2])
        all_nl.append(nlc)
        all_mt.append(mtc)
        _lg, _cr = leadlag_needle(np.diff(nlc, axis=1),
                                  np.diff(mtc, axis=1), max_lag=MAX_LAG)
        row = "  ".join(("   nan" if np.isnan(c) else f"{c:+.3f}")
                        for c in _cr)
        pk = ("nan" if np.all(np.isnan(_cr))
              else f"{int(_lg[np.nanargmax(_cr)]):+d} "
                   f"(r={np.nanmax(_cr):.3f})")
        print(f"[leadlag] {cls:12s} {len(nlc):5d}  {row}   {pk}")
        print(f"[leadlag]   {cls:10s} nl_dist/breath  = "
              + " ".join(f"{v:.4f}" for v in nlc.mean(0)))
        print(f"[leadlag]   {cls:10s} math_dist/breath= "
              + " ".join(f"{v:.4f}" for v in mtc.mean(0)))
        print(f"[leadlag]   {cls:10s} att_H/breath    = "
              + " ".join(f"{v:.3f}" for v in enc.mean(0)))
    if all_nl:
        nlc = np.concatenate(all_nl)
        mtc = np.concatenate(all_mt)
        _lg, _cr = leadlag_needle(np.diff(nlc, axis=1),
                                  np.diff(mtc, axis=1), max_lag=MAX_LAG)
        row = "  ".join(("   nan" if np.isnan(c) else f"{c:+.3f}")
                        for c in _cr)
        pk = ("nan" if np.all(np.isnan(_cr))
              else f"{int(_lg[np.nanargmax(_cr)]):+d} "
                   f"(r={np.nanmax(_cr):.3f})")
        print(f"[leadlag] {'ALL':12s} {len(nlc):5d}  {row}   {pk}")
    else:
        print("[leadlag] NO USABLE ITEMS — every row's class was "
              "off-atlas (loud, not a silent empty table)")


if __name__ == "__main__":
    main()
