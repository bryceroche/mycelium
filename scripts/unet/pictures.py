"""scripts/unet/pictures.py — THE U-NET PICTURE BAKE-OFF, picture builders
(Opus's design, word given 2026-10-03). Give text the coordinate system a
spectrogram gives audio: a token x token PICTURE per problem, read by a
small 2D U-Net (model.py) that SEGMENTS the factor-graph links. This file
builds the pictures, from the CACHED trunk states only (never a trunk
call, never scripts/phase1_algebra_head.py's forward()/build_params()).

A NEW module under scripts/unet/ — scripts/phase1_algebra_head.py and every
other existing script are untouched, per the word. Import the head ONLY
for its constants/helpers (T_ALG, H_TRUNK, tree_row_ids) under the family
env; this module never trains anything and never touches the GPU path.

Run convention (every script in this directory): DEV=CPU, and the FAM env
string pinned in the mission brief, e.g.:
  DEV=CPU ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 \
  ALG_POLAR=1 ALG_TREE=t,s,c,m,t,t,t
before importing phase1_algebra_head — tree_row_ids's own module-level
constants (N_DIG, T_ALG, ...) are read from os.environ at import time.

-----------------------------------------------------------------------
THE PICTURE FAMILIES
-----------------------------------------------------------------------
  A'   (Aprime)       : 9 channels. Channel 0 = cosine similarity of the
                         raw per-token trunk states (post-L3, post-final-
                         norm; the ONLY layer cached — see the module
                         docstring note on ALG_LAYERS below). Channels 1-8
                         = "PCA band" channels: fit an 8-component PCA on
                         a sample of the diet's trunk states (once, cached
                         to disk), project every token onto each of the 8
                         components, and for band k build a (T,T) channel
                         from the OUTER PRODUCT of that one-dimensional
                         projection, squashed through tanh (normalizes to
                         [-1,1]; a direct cosine of two SCALARS is
                         degenerate (always +-1), so "band-wise cosine" is
                         read here as the band's own normalized pairwise
                         product — see the INTERPRETATION NOTE below).
  A''  (Adouble)       : A' + 3 WHERE channels (same-sentence, same-clause,
                         same-mention; binary 0/1, from tree_row_ids's unit
                         ids — a TEXT-derived structure, so it is available
                         on wild too, unlike the hand mention/arg/cue spans
                         used by targets.py). 12 channels.
  A''s (Adouble_shuf)  : A'' with the WHERE blocks' unit-id assignment
                         SHUFFLED (same unit count, same unit sizes, random
                         token-position assignment) — the placebo: a win of
                         A'' over A''s is structure, not channel count.
  A    (raw attention) : NOT built here. Stub that raises, with a message
                         (see `build_picture`'s `mode == "A"` branch) —
                         attention probabilities are not in the cached
                         states file; a new precompute would be needed.

INTERPRETATION NOTE (read before trusting the PCA-band channels): the
brief's phrase "PCA'd states' low bands ... 8 bands -> 8 channels of
band-wise cosine" is ambiguous about what "band-wise cosine" means for a
SINGLE scalar projection per token (cosine of two 1-D numbers is only
ever +-1 — a useless channel). The reading implemented here: treat each
PCA component's per-token scalar as a 1-D "signal" over the sequence
(literally the spectrogram analogy — one frequency band's amplitude
trace) and build the pairwise channel as that signal's own normalized
(by its row std, so it's scale-invariant per row) outer product, tanh-
squashed into [-1,1] per the brief's explicit normalization instruction.
Same-sign, similar-magnitude token pairs light up; opposite-sign pairs go
dark; this is the natural pairwise object for a 1-D per-token signal.
State this reading in the report; a literal per-pair cosine on the FULL
8-D PCA sub-space (one extra channel, cosine of the 8-vectors) would be a
defensible ninth content channel but is NOT what was asked (8 channels
WERE asked for, one per band) — not added.

THE LAYERS QUESTION: only ONE trunk layer's states are cached per split —
`.cache/phase1_alg_states_<split>_states.npy` is written by
do_precompute() from `x` AFTER the full L0-L3 stack and the final RMSNorm
(scripts/phase1_algebra_head.py do_precompute(), ~line 1990-2010): there is
no intermediate-layer cache. UN_LAYERS is accepted below for forward
compatibility but anything other than the default errors loudly, naming
the precompute script that would need to be added (not built here, per
the word: no changes to phase1_algebra_head.py, and a new multi-layer
precompute is out of this task's scope).
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

# phase1_algebra_head is imported lazily (see `_head()`) so that merely
# importing this module (e.g. from targets.py, which does not need
# tree_row_ids) never requires the FAM env to be set.
_H = None


def _head():
    global _H
    if _H is None:
        import phase1_algebra_head as H
        _H = H
    return _H


# ---------------------------------------------------------------------
# paths (mirrors phase1_algebra_head.py's STATES_NPZ/STATES_NPY naming —
# duplicated as plain format strings, not imported, so this module can be
# used without the FAM env when only numpy is needed, e.g. by targets.py)
# ---------------------------------------------------------------------
STATES_NPZ = ".cache/phase1_alg_states_{split}.npz"
STATES_NPY = ".cache/phase1_alg_states_{split}_states.npy"
PCA_CACHE = os.environ.get("UN_PCA_CACHE", ".cache/unet_pca8.npz")
PCA_K = int(os.environ.get("UN_PCA_K", "8"))
PCA_FIT_SPLIT = os.environ.get("UN_PCA_FIT_SPLIT", "formpm35c")
PCA_FIT_ROWS = int(os.environ.get("UN_PCA_SAMPLE", "200"))
PCA_FIT_TOK_PER_ROW = int(os.environ.get("UN_PCA_TOK_PER_ROW", "64"))
PCA_FIT_SEED = int(os.environ.get("UN_PCA_SEED", "0"))

MODES = ("Aprime", "Adouble", "Adouble_shuf", "A")
N_CONTENT = 1 + PCA_K           # cosine + PCA bands
N_WHERE = 3                      # same-sentence, same-clause, same-mention


class PictureSource:
    """One split's data, opened once (memmap states + npz tokmask/sent +
    the raw jsonl's text) — reused across many `get_picture()` calls (the
    on-the-fly builder; nothing about a picture is ever cached to disk,
    per the word: "build on the fly from the memmap")."""

    def __init__(self, split, jsonl_path, layer=None):
        if layer not in (None, "default"):
            raise NotImplementedError(
                f"UN_LAYERS={layer!r}: only the default (post-L3, post-"
                f"final-norm) trunk layer is cached in "
                f"{STATES_NPY.format(split=split)} — do_precompute() "
                f"(scripts/phase1_algebra_head.py) never wrote a per-layer "
                f"stack. A multi-layer read needs a NEW precompute script "
                f"(e.g. scripts/unet/precompute_layers.py) that reruns the "
                f"frozen trunk and dumps every layer's states; not built "
                f"here (never edit phase1_algebra_head.py; out of scope "
                f"for this bake-off).")
        self.split = split
        npz_path = STATES_NPZ.format(split=split)
        npy_path = STATES_NPY.format(split=split)
        z = np.load(npz_path)
        self.tokmask = z["tokmask"]               # (n, T_full) uint8
        self.sent = z["sent"]                     # (n, T_full) int8
        self.gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
        self.states = np.load(npy_path, mmap_mode="r")   # (n, T_full, D) f16
        self.n, self.T_full, self.D = self.states.shape
        assert self.tokmask.shape[0] == self.n, \
            f"{npz_path} has {self.tokmask.shape[0]} rows but {npy_path} has {self.n}"
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        assert len(self.samples) == self.n, \
            f"{jsonl_path} has {len(self.samples)} rows but the staged states have {self.n}"

    def text(self, idx):
        return self.samples[idx]["text"]

    def row(self, idx, T=None):
        """(state (T,D) f32, tokmask (T,) f32, sent (T,) int32) for one
        row, cropped to T (default: the full cached T_full)."""
        T = T or self.T_full
        assert T <= self.T_full
        state = np.asarray(self.states[idx, :T, :], dtype=np.float32)
        tm = self.tokmask[idx, :T].astype(np.float32)
        se = self.sent[idx, :T].astype(np.int32)
        return state, tm, se


# ---------------------------------------------------------------------
# PCA (fit once on a diet sample; cached to disk; reused frozen on wild)
# ---------------------------------------------------------------------
def fit_pca(cache_path=None, split=None, n_rows=None, tok_per_row=None, seed=None):
    cache_path = cache_path or PCA_CACHE
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["mean"].astype(np.float32), z["comps"].astype(np.float32)
    split = split or PCA_FIT_SPLIT
    n_rows = n_rows or PCA_FIT_ROWS
    tok_per_row = tok_per_row or PCA_FIT_TOK_PER_ROW
    seed = PCA_FIT_SEED if seed is None else seed
    jsonl = f".cache/form_mix_{split[2:]}.jsonl" if split.startswith("pm") else None
    # the split-name -> jsonl convention used throughout this family
    # ("formpm35c" states <- .cache/form_mix_pm35c.jsonl); PictureSource
    # does not need the jsonl here (only states+tokmask), so load those
    # two arrays directly rather than instantiating a full PictureSource.
    npz = np.load(STATES_NPZ.format(split=split))
    tokmask = npz["tokmask"]
    states = np.load(STATES_NPY.format(split=split), mmap_mode="r")
    n = states.shape[0]
    rng = np.random.RandomState(seed)
    rows = rng.choice(n, size=min(n_rows, n), replace=False)
    vecs = []
    for r in sorted(rows.tolist()):
        tm = tokmask[r] > 0
        pos = np.where(tm)[0]
        if len(pos) == 0:
            continue
        if len(pos) > tok_per_row:
            pos = rng.choice(pos, size=tok_per_row, replace=False)
        vecs.append(np.asarray(states[r, pos, :], dtype=np.float32))
    X = np.concatenate(vecs, axis=0)
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD: top-K right singular vectors == top-K PCA components
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:PCA_K].astype(np.float32)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, mean=mean.astype(np.float32), comps=comps,
              split=split, n_rows=len(vecs), seed=seed)
    print(f"[unet/pictures] fit PCA K={PCA_K} on {len(rows)} rows / "
          f"{X.shape[0]} tokens of {split} -> {cache_path}", flush=True)
    return mean.astype(np.float32), comps


# ---------------------------------------------------------------------
# channel builders (pure numpy; one row at a time)
# ---------------------------------------------------------------------
def cosine_channel(states_masked, real):
    """states_masked: (T,D) f32, already zeroed at pad rows. real: (T,) bool."""
    norm = np.linalg.norm(states_masked, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-6, norm, 1.0)
    unit = states_masked / safe
    unit[~real] = 0.0
    cos = unit @ unit.T
    cos = np.clip(cos, -1.0, 1.0)
    cos[~real, :] = 0.0
    cos[:, ~real] = 0.0
    return cos.astype(np.float32)


def pca_band_channels(states_masked, real, mean, comps):
    """8 (T,T) channels, one per PCA component -- see the INTERPRETATION
    NOTE in the module docstring for what "band-wise cosine" means here."""
    T = states_masked.shape[0]
    K = comps.shape[0]
    z = (states_masked - mean[None, :] * real[:, None].astype(np.float32)) @ comps.T   # (T,K)
    out = np.zeros((K, T, T), np.float32)
    for k in range(K):
        v = z[:, k]
        std = v[real].std() if real.any() else 0.0
        std = std if std > 1e-6 else 1.0
        outer = np.outer(v, v) / (std * std)
        outer = np.tanh(outer).astype(np.float32)
        outer[~real, :] = 0.0
        outer[:, ~real] = 0.0
        out[k] = outer
    return out


def where_channels(tree_ids, real):
    """3 (T,T) binary channels from tree_row_ids's (T,4) unit-id array:
    same-sentence / same-clause / same-mention. tree_ids columns 0,1,2;
    -1 at padding (tree_row_ids's own convention)."""
    T = tree_ids.shape[0]
    out = np.zeros((3, T, T), np.float32)
    for ci in range(3):
        col = tree_ids[:, ci]
        same = (col[:, None] == col[None, :]) & (col[:, None] >= 0) & (col[None, :] >= 0)
        out[ci] = same.astype(np.float32)
    out[:, ~real, :] = 0.0
    out[:, :, ~real] = 0.0
    return out


def shuffle_unit_ids(col, real, rng):
    """Placebo: a NEW (T,) id array with the SAME unit count and the SAME
    multiset of unit (run) sizes as `col` over the real positions, but the
    runs reassigned to random positions (the order of the runs is
    permuted; contiguity is kept so "unit" stays meaningful, only WHICH
    tokens belong to which unit is scrambled)."""
    out = col.copy()
    idx = np.where(real)[0]
    if len(idx) == 0:
        return out
    vals = col[idx]
    runs = []
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        runs.append(j - i)
        i = j
    order = rng.permutation(len(runs))
    pos = 0
    newid = 0
    for ri in order:
        length = runs[int(ri)]
        out[idx[pos:pos + length]] = newid
        newid += 1
        pos += length
    return out


def shuffled_where_channels(tree_ids, real, rng):
    shuf = tree_ids.copy()
    for ci in range(3):
        shuf[:, ci] = shuffle_unit_ids(tree_ids[:, ci], real, rng)
    return where_channels(shuf, real)


# ---------------------------------------------------------------------
# the one entry point everything else calls
# ---------------------------------------------------------------------
def build_picture(state, tokmask, sent, text, mean, comps, mode, rng=None):
    """state (T,D) f32, tokmask (T,) f32 in {0,1}, sent (T,) int32, text
    str, mean/comps from fit_pca(). Returns (C,T,T) f32, C = 9 (Aprime) or
    12 (Adouble / Adouble_shuf)."""
    if mode == "A":
        raise NotImplementedError(
            "picture family A (raw trunk attention maps) is NOT built: "
            "the cached states file (.cache/phase1_alg_states_<split>_"
            "states.npy, written by do_precompute() in "
            "scripts/phase1_algebra_head.py) stores only the POST-L3, "
            "post-final-norm per-token STATE, never attention "
            "probabilities. A new precompute (re-running the frozen "
            "trunk and dumping each layer's softmax(QK^T)) would be "
            "needed -- out of scope here (never edit "
            "phase1_algebra_head.py; CLAUDE.md), and no such cache "
            "exists today. Use 'Aprime', 'Adouble' or 'Adouble_shuf'.")
    if mode not in MODES:
        raise ValueError(f"unknown picture mode {mode!r} (want one of {MODES})")
    T = state.shape[0]
    real = tokmask > 0
    masked = state * tokmask[:, None]
    chans = [cosine_channel(masked, real)]
    chans += [c for c in pca_band_channels(masked, real, mean, comps)]
    pic = np.stack(chans, axis=0)   # (9, T, T)
    if mode == "Aprime":
        return pic
    H = _head()
    tree = H.tree_row_ids(text, tokmask, sent, T)   # (T,4) int32
    if mode == "Adouble":
        wc = where_channels(tree, real)
    else:   # Adouble_shuf
        rng = rng if rng is not None else np.random.default_rng(0)
        wc = shuffled_where_channels(tree, real, rng)
    return np.concatenate([pic, wc], axis=0)   # (12, T, T)


def n_channels(mode):
    if mode == "Aprime":
        return N_CONTENT
    if mode in ("Adouble", "Adouble_shuf"):
        return N_CONTENT + N_WHERE
    raise ValueError(mode)


if __name__ == "__main__":
    # tiny self-check: 2 real rows of the diet, T=32, all three buildable
    # modes, printed shapes + value ranges. DEV=CPU; FAM env must already
    # be set (tree_row_ids needs phase1_algebra_head's tokenizer path).
    src = PictureSource("formpm35c", ".cache/form_mix_pm35c.jsonl")
    mean, comps = fit_pca()
    for i in range(2):
        st, tm, se = src.row(i, T=32)
        txt = src.text(i)
        for mode in ("Aprime", "Adouble", "Adouble_shuf"):
            pic = build_picture(st, tm, se, txt, mean, comps, mode,
                                 rng=np.random.default_rng(i))
            print(f"row {i} {mode}: shape={pic.shape} "
                  f"range=[{pic.min():.3f},{pic.max():.3f}] "
                  f"nan={np.isnan(pic).any()}")
    try:
        build_picture(st, tm, se, txt, mean, comps, "A")
    except NotImplementedError as e:
        print("mode A stub OK:", str(e)[:60] + "...")
