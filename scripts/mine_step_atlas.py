"""mine_step_atlas.py — THE PER-STEP ATLAS MINER (2026-09-05, mask-head
round 2). Fills the registered design's store (mycelium/step_atlas.py,
ledger 2026-09-02): SEVEN per-breath_step Welford centroid banks keyed
(breath_step, class), mined on the CHAMPION's coordinates
(.cache/sharp_port242.safetensors) over a sample of the TRAIN diet
(.cache/form_mix12.jsonl via load_alg("train"), cap env MSA_N=4096).

THE TAP (the least-invasive choice): ALG_MINE_BREATHS=1 — forward()
already returns out["breaths_all"] = the raw per-breath slot states
(K_B x (B, L_FAC, H_W)), the exact read-only seam cycle_atlas_mine.py
used. NO patch to forward is needed; zero new code runs inside the
graph (the _CENSUS hook was the fallback; this is cheaper still).

POOLING CHOICE (documented per the mission): each item's per-step page
is the MEAN over all 24 factor slots -> (H_W,) per (item, breath_step).
Mean-pooling over the fixed slot bank is the mouth's own idiom (pooled
trunk reads) and keeps pages slot-permutation-stable; per-slot or
presence-weighted pooling are read-time refinements, not mined here.

CLASS LABEL: mycelium.step_atlas.atlas_class(row["gen"]) — the single-
source labeler (meter-divergence law: feed-time consumers call the SAME
organ). Ladder depth buckets 2-4/5-8/9+; wild by gen.src; else
mint_form8; teeth costume ignored.

THE CYCLE MINED = the deployable two-pass cycle (loop_val's shape):
pass-1 unmasked parse -> build_slot_masks + alt2_fact_buf (live facts,
champion ALT2 envs) -> masked pass-2 with fact_buf; breaths_all comes
from pass-2. Atlas coordinates therefore match what the round-2
trainer's forward walks through.

ERA / MANIFEST (constitutional): this is a RESEARCH-lineage artifact.
The miner writes .cache/RESEARCH_MANIFEST.json
({"parser_ckpt": ".cache/sharp_port242.safetensors"}) and stamps the
atlas against IT — research-lineage load_atlas calls must pass this
manifest path (env MH_ATLAS_MANIFEST); the deployed GENERATION.json is
NEVER touched or consulted here. A 6k gentle continuation drifts only
slightly off port242's coordinates (drift = near-pure rotation, ledger);
re-mine at any rebirth (the never-mix-generations law's standing duty).

Envs: the champion stack from .cache/mask_head_chain.sh, applied via
setdefault (the wave_field_check lesson: caller/chain envs WIN; a miner
must never clobber the declared environment). ALG_MASKHEAD is NOT set:
port242 predates the organ, and at zero-init the states are identical
anyway. GPU script — STAGED, NOT RUN at build time (hold for the word);
`import mine_step_atlas` is CPU-safe (all work under __main__/main()).
"""
import json
import os
import sys

# Champion env stack — VERBATIM from .cache/mask_head_chain.sh (the
# round-1 chain), via setdefault so a chain's explicit envs win.
_CHAMPION_ENVS = {
    "DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512.npz",
    "ALG_ALLOW_PEN_TRAIN": "1",
    "ALG_TRAIN": ".cache/form_mix12.jsonl", "ALG_TRAIN_NAME": "form12",
    "ALG_TEST": ".cache/algebra_nl_test.jsonl", "ALG_TEST_NAME": "test23",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "SC_EVAL": "0", "ALG_ALT21": "1", "ALG_ALT2": "1",
    # the tap: forward returns breaths_all (read-only, zero graph change)
    "ALG_MINE_BREATHS": "1",
}
for _k, _v in _CHAMPION_ENVS.items():
    os.environ.setdefault(_k, _v)

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

CKPT = os.environ.get("MSA_CKPT", ".cache/sharp_port242.safetensors")
ATLAS_OUT = os.environ.get("MSA_OUT", ".cache/step_atlas_current.npz")
RESEARCH_MANIFEST = os.environ.get("MH_ATLAS_MANIFEST",
                                   ".cache/RESEARCH_MANIFEST.json")
MSA_N = int(os.environ.get("MSA_N", "4096"))
MSA_SEED = int(os.environ.get("MSA_SEED", "242"))


def main():
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf,
                                     K_VARS, H_W)
    from mycelium.step_atlas import (StepWelford, save_atlas, atlas_class,
                                     K_STEPS)

    samples, states, tokmask, gold, sent = load_alg("train")
    n_all = len(samples)
    rng = np.random.default_rng(MSA_SEED)
    take = (rng.permutation(n_all)[:MSA_N] if MSA_N < n_all
            else np.arange(n_all))
    take.sort()
    print(f"[mine-atlas] {len(take)}/{n_all} train rows, ckpt={CKPT}",
          flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    cells = {}
    n_done = 0
    for s0 in range(0, len(take), 8):
        sl = take[s0:s0 + 8]
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        # pass 1: unmasked parse -> masks + live facts (loop_val's cycle)
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
        # pass 2: the breathing walk; the tap hands back raw per-breath
        # slot states (ALG_MINE_BREATHS=1 -> out["breaths_all"])
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                    fact_buf=Tensor(fb, dtype=dtypes.float))
        br = [b.realize().numpy() for b in o["breaths_all"]]
        assert len(br) == K_STEPS, \
            f"breaths_all has {len(br)} steps, atlas wants {K_STEPS}"
        for bi, i in enumerate(sl):        # pads (bi >= len(sl)) skipped
            cls = atlas_class(samples[int(i)].get("gen"))
            for s_id in range(K_STEPS):
                key = (s_id, cls)
                if key not in cells:
                    cells[key] = StepWelford(H_W)
                # POOLING (see module docstring): mean over the 24 slots
                cells[key].add(br[s_id][bi].mean(0).astype(np.float64))
        n_done += len(sl)
        if (s0 // 8) % 64 == 0:
            print(f"[mine-atlas] {n_done}/{len(take)}", flush=True)

    # research manifest FIRST (the stamp's anchor), then the stamped save
    with open(RESEARCH_MANIFEST, "w") as f:
        json.dump({"parser_ckpt": CKPT,
                   "note": "RESEARCH lineage manifest (mask-head round 2)"
                           " — era anchor for step_atlas artifacts;"
                           " deployed GENERATION.json untouched"}, f,
                  indent=1)
    path = save_atlas(cells, H_W, path=ATLAS_OUT,
                      manifest_path=RESEARCH_MANIFEST)
    classes = sorted({c for (_, c) in cells})
    print(f"[mine-atlas] saved {path} (era anchored to {RESEARCH_MANIFEST})")
    print(f"[mine-atlas] classes={classes}")
    for cls in classes:
        counts = [cells[(s, cls)].n if (s, cls) in cells else 0
                  for s in range(K_STEPS)]
        print(f"[mine-atlas]   {cls:12s} n/step={counts}")


if __name__ == "__main__":
    main()
