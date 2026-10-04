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
  Awhich (round 2)     : A' + 6 WHICH channels (THE U-NET PICTURE BAKE-OFF
                         ROUND 2, word given 2026-10-04 — round 1's
                         WHERE channels [same-sentence/clause/mention]
                         were inert; the reconciliation (ledger, "the
                         hierarchy moves from WHERE to WHICH") said the
                         model fails at WHICH entity/event/moment a
                         number belongs to, not where it sits). All six
                         are computable on WILD — no gold mentions
                         needed, only the cached trunk states, the raw
                         text, and tree_row_ids's SENTENCE/CLAUSE ids
                         (col 0/1; never the hand-span-derived MENTION
                         col 2):
                           1. LEX-IDENTITY: 1 iff two tokens decode
                              (stripped, lowercased) to the identical
                              word string; digit-decoding tokens are
                              EXCLUDED from matching (a shared digit is
                              not a shared entity).
                           2. RETINA-COSINE: cosine of the two tokens'
                              Llama INPUT embedding, context-free (not
                              the contextual trunk state channel 0
                              already carries) — read via the SAME
                              source scripts/ident_codes_mint.py built
                              `.cache/ident_codes512.npz`'s IDT table
                              (vocab,512) f16, a centred+projected,
                              row-unit-normalized version of
                              model.embed_tokens.weight: cosine of two
                              IDT rows IS cosine in that projected space
                              (unit vectors -> dot == cosine); reusing
                              the cached table avoids re-reading the
                              2048-d safetensors embedding a second time
                              — "the same source", read literally.
                           3. EVENT: 1 iff two tokens lie in the same
                              tree_row_ids CLAUSE unit (col 1: cut at
                              ,/;/and/but, no verb cut) AND that clause
                              contains at least one token decoding into
                              membrane_scale.py's VERBISH stoplist
                              (duplicated verbatim below, same reason
                              targets.py duplicates _digit_runs rather
                              than importing a sibling script).
                           4. MOMENT: signed clause-order difference,
                              moment[i,j] = (clause[j]-clause[i]) scaled
                              to [-1,1] by the row's own max clause
                              index (before/after, continuous — NOT a
                              placebo target, see below).
                           5-6. ROLE-CUE proximity (row-wise, col-wise):
                              per-token distance (in tokens, scaled by
                              T-1 into [0,1]) to the nearest VERBISH cue
                              token in the SAME clause, broadcast along
                              the row axis and, separately, the column
                              axis (two constant-along-one-axis channels
                              — see `role_cue_channels`).
                         15 channels total (9 content + 6 WHICH); no
                         WHERE channels are included (A' is the base,
                         not A'').
  Awhich_shuf (round 2) : Awhich with ONLY the LEX-IDENTITY and EVENT
                         blocks placebo'd — named explicitly in the
                         brief ("the lex-identity and event blocks
                         SHUFFLED"); RETINA-COSINE/MOMENT/ROLE-CUE are
                         left INTACT in this arm (a genuine win needs
                         Awhich to beat a placebo that only scrambles
                         the two blocks the brief named, not every
                         WHICH channel at once — a narrower, more
                         honest placebo than one that nulls the whole
                         family). EVENT's shuffle reuses
                         `shuffle_unit_ids` (round 1's contiguous-run
                         placebo) on the CLAUSE column — same run-size
                         multiset, scrambled run order — then re-
                         derives "does this [new] clause contain a cue"
                         from the ACTUAL words now falling in each
                         scrambled block (words never move; only the
                         clause-id labelling does). LEX-IDENTITY has no
                         well-defined "runs" to preserve (same-word
                         tokens are generally SCATTERED, not contiguous),
                         so its placebo generalizes round 1's "same
                         counts/sizes, scrambled positions" rule to the
                         non-contiguous case: a full random permutation
                         of the per-token word-class id array over the
                         real positions (`shuffle_values`) — this keeps
                         every class's token COUNT exactly fixed and
                         scrambles WHICH position holds which class,
                         which is the literal content of the brief's
                         phrase once "runs" no longer applies. Stated as
                         an interpretation, not lifted verbatim from
                         round 1's code path.
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

WHICH-CHANNEL INTERPRETATION NOTES (read before trusting round 2's new
channels):
  - ROLE-CUE's "no cue in this clause" case is not derivable as a finite
    token distance; it is set to the SENTINEL value T-1 (normalizes to
    1.0, "maximally far") rather than, say, 0 or NaN — a clause with
    no verb-ish anchor is read as having no role structure to be close
    to, which is the conservative (farthest, not nearest) reading.
  - MOMENT's scale is PER-ROW (that row's own max clause index), not a
    fixed global constant — a 3-clause row and a 12-clause row both
    fill [-1,1], so the channel is about ORDER, never absolute clause
    count; stated so a reader does not mistake it for a length signal.
  - LEX-IDENTITY's placebo (`shuffle_values`) is a genuine
    generalization of round 1's `shuffle_unit_ids`, not that function
    reused unmodified — see the Awhich_shuf docstring bullet above.
  - EVENT's placebo re-tests "does this clause contain a cue" against
    the SHUFFLED clause boundaries but the ORIGINAL words at each
    position (a structural-only scramble, matching round 1's where-
    channel placebo exactly: positions/boundaries move, token content
    does not).

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
IDENT_CODES_PATH = os.environ.get("UN_IDENT_CODES", ".cache/ident_codes512.npz")

MODES = ("Aprime", "Adouble", "Adouble_shuf", "Awhich", "Awhich_shuf", "A")
N_CONTENT = 1 + PCA_K           # cosine + PCA bands
N_WHERE = 3                      # same-sentence, same-clause, same-mention
N_WHICH = 6                      # lex-identity, retina-cosine, event, moment, role-cue row/col
WHICH_NAMES = ("lex_identity", "retina_cosine", "event", "moment",
               "role_cue_row", "role_cue_col")

# ---------------------------------------------------------------------
# VERBISH -- duplicated VERBATIM from scripts/membrane_scale.py's module-
# level VERBISH set (itself byte-identical to phase1_algebra_head.py's
# private _TREE_VERBISH, used internally by tree_row_ids's MENTION column
# -- see that module's own "verbatim" comment on _tree_segment_ids). Not
# imported from either: membrane_scale.py is a standalone analyst script
# (module-level MS_* env parsing, a CKPT default path) and
# phase1_algebra_head.py is never edited and its _TREE_VERBISH is a
# private name -- duplicating a frozen stoplist is the targets.py
# convention already used in this directory for _digit_runs.
# ---------------------------------------------------------------------
VERBISH = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "will", "would", "can", "could", "costs", "cost", "spent", "spend",
    "spends", "gives", "gave", "give", "given", "bought", "buy", "buys",
    "sold", "sell", "sells", "needs", "need", "wants", "want", "makes",
    "made", "make", "paid", "pay", "pays", "receives", "receive",
    "received", "left", "leaves", "leave", "remains", "remain", "earns",
    "earn", "earned", "uses", "use", "used", "takes", "take", "took",
    "adds", "add", "added", "gets", "get", "got", "found", "finds", "find",
    "totals", "total", "equals", "equal", "contains", "contain", "starts",
    "start", "started", "ends", "end", "ended", "if", "then", "so",
    "because", "after", "before", "each", "per",
}


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
# ROUND 2: THE WHICH CHANNELS (word given 2026-10-04)
# ---------------------------------------------------------------------
_IDT_CACHE = {}   # path -> (vocab,512) f32


def load_ident_codes(path=None):
    """`.cache/ident_codes512.npz`'s IDT table (see scripts/ident_codes_
    mint.py): row-unit-normalized, centred+projected Llama input
    embedding, by token id. Cached per path (the file is 125MB; loaded
    once per process, same convention as fit_pca's PCA_CACHE)."""
    path = path or IDENT_CODES_PATH
    if path not in _IDT_CACHE:
        z = np.load(path)
        _IDT_CACHE[path] = z["IDT"].astype(np.float32)
    return _IDT_CACHE[path]


def _tok_ids_dec(text, T):
    """Re-tokenizes `text` with the head's own tokenizer path (the SAME
    deterministic ids tree_row_ids/targets.build_target produce) --
    phase1_algebra_head.py is never edited, so this duplicates its own
    ids/dec build (tree_row_ids's internal `dec`) rather than asking that
    function to also return them. Returns (ids (T,) int64, 0 at padding;
    dec (T,) list[str], stripped+lowercased, "" at padding)."""
    H = _head()
    tok = H._xcorr_tokenizer()
    raw = tok.encode(text).ids[:T]
    n = len(raw)
    ids = np.zeros(T, np.int64)
    ids[:n] = np.asarray(raw, np.int64)
    dec = [tok.decode([int(t)]).strip().lower() for t in raw] + [""] * (T - n)
    return ids, dec


def lex_class_ids(dec, real):
    """(T,) int32: a distinct small int per distinct lowercased word
    string over REAL positions, -1 for padding, empty decodes, and
    digit-only decodes (digits excluded from lex-identity matching, per
    the brief)."""
    T = len(dec)
    out = -np.ones(T, np.int32)
    seen = {}
    nxt = 0
    for t in range(T):
        if not real[t]:
            continue
        w = dec[t]
        if not w or w.isdigit():
            continue
        if w not in seen:
            seen[w] = nxt
            nxt += 1
        out[t] = seen[w]
    return out


def shuffle_values(col, real, rng):
    """THE LEX-IDENTITY PLACEBO: a full random permutation of `col`'s
    values over the REAL positions only (every class's token COUNT is
    exactly preserved; WHICH position holds which class is scrambled).
    Generalizes `shuffle_unit_ids` (round 1's contiguous-RUN-preserving
    shuffle) to classes that are not contiguous runs -- lex-identity
    classes (same word recurring) are generally SCATTERED across a row,
    so there is no run structure to preserve; a plain value permutation
    is the natural reading of "same counts/sizes, scrambled positions"
    once runs do not apply (see pictures.py's module docstring)."""
    out = col.copy()
    idx = np.where(real)[0]
    if len(idx) == 0:
        return out
    vals = col[idx].copy()
    rng.shuffle(vals)
    out[idx] = vals
    return out


def lex_identity_channel(lex, real):
    """(T,T) f32 {0,1}: same[i,j] = 1 iff token i and j share a (non-
    excluded) lex class."""
    same = (lex[:, None] == lex[None, :]) & (lex[:, None] >= 0)
    same = same.astype(np.float32)
    same[~real, :] = 0.0
    same[:, ~real] = 0.0
    return same


def retina_cosine_channel(ids, real, IDT):
    """(T,T) f32 in [-1,1]: cosine of the two tokens' context-free input
    embedding (IDT rows are already unit-normalized, so dot == cosine)."""
    V = IDT.shape[0]
    emb = IDT[np.clip(ids, 0, V - 1)].astype(np.float32)   # (T,512)
    emb = emb.copy()
    emb[~real] = 0.0
    cos = emb @ emb.T
    cos = np.clip(cos, -1.0, 1.0)
    cos[~real, :] = 0.0
    cos[:, ~real] = 0.0
    return cos.astype(np.float32)


def _clause_cue_mask(clause_col, real, dec, verbish):
    """(n_clauses,)-indexed dict: clause id -> True iff ANY real token
    assigned to that clause id decodes into `verbish`."""
    has = {}
    T = len(clause_col)
    for t in range(T):
        if not real[t]:
            continue
        cid = int(clause_col[t])
        if cid < 0:
            continue
        if dec[t] in verbish:
            has[cid] = True
        else:
            has.setdefault(cid, False)
    return has


def event_channel(clause_col, real, dec, verbish=VERBISH):
    """(T,T) f32 {0,1}: 1 iff i,j share a clause id AND that clause
    contains >=1 VERBISH cue token. `clause_col` may be the ORIGINAL
    tree_row_ids column-1 (Awhich) or a `shuffle_unit_ids`-scrambled
    version of it (Awhich_shuf's placebo) -- the cue lookup always reads
    the ACTUAL (unmoved) words at each position against whichever clause
    labelling is passed in."""
    T = len(clause_col)
    has = _clause_cue_mask(clause_col, real, dec, verbish)
    has_arr = np.array([has.get(int(c), False) for c in clause_col], dtype=bool)
    same_clause = (clause_col[:, None] == clause_col[None, :]) & (clause_col[:, None] >= 0)
    event = same_clause & has_arr[:, None]
    event = event.astype(np.float32)
    event[~real, :] = 0.0
    event[:, ~real] = 0.0
    return event


def moment_channel(clause_col, real):
    """(T,T) f32 in [-1,1]: moment[i,j] = (clause[j]-clause[i]) scaled by
    this ROW'S OWN max real clause index (order, not absolute count --
    see the module docstring's interpretation note)."""
    T = len(clause_col)
    c = clause_col.astype(np.float32)
    valid = real & (clause_col >= 0)
    max_c = float(c[valid].max()) if valid.any() else 0.0
    scale = max(1.0, max_c)
    diff = c[None, :] - c[:, None]              # diff[i,j] = c[j] - c[i]
    out = np.clip(diff / scale, -1.0, 1.0).astype(np.float32)
    out[~real, :] = 0.0
    out[:, ~real] = 0.0
    return out


def role_cue_distance(clause_col, real, dec, verbish=VERBISH):
    """(T,) f32 in [0,1]: token-distance to the nearest VERBISH cue token
    within the SAME clause, scaled by T-1. A clause with NO cue token at
    all gets the SENTINEL (scale -> normalizes to 1.0, "maximally far" --
    see the module docstring's interpretation note), never 0/NaN."""
    T = len(clause_col)
    scale = max(1.0, float(T - 1))
    cue_mask = np.zeros(T, bool)
    for t in range(T):
        cue_mask[t] = bool(real[t] and dec[t] in verbish)
    dist = np.full(T, scale, np.float32)
    clauses = set(int(c) for c, r in zip(clause_col.tolist(), real.tolist()) if r and c >= 0)
    for cid in clauses:
        members = np.where(real & (clause_col == cid))[0]
        cues = members[cue_mask[members]]
        if len(cues) == 0:
            continue
        for t in members:
            dist[t] = float(np.min(np.abs(cues - t)))
    out = np.clip(dist / scale, 0.0, 1.0).astype(np.float32)
    out[~real] = 0.0
    return out


def role_cue_channels(clause_col, real, dec, verbish=VERBISH):
    """Two (T,T) f32 channels: `dist` broadcast row-wise (row[i,j] =
    dist[i], constant across j) and column-wise (col[i,j] = dist[j],
    constant across i) -- per the brief's "broadcast along both axes:
    two channels, row-wise and column-wise"."""
    dist = role_cue_distance(clause_col, real, dec, verbish)
    T = len(dist)
    row = np.broadcast_to(dist[:, None], (T, T)).copy()
    col = np.broadcast_to(dist[None, :], (T, T)).copy()
    row[~real, :] = 0.0; row[:, ~real] = 0.0
    col[~real, :] = 0.0; col[:, ~real] = 0.0
    return row, col


def which_channels(text, real, tree, mode, rng, T):
    """(6,T,T) f32: the WHICH block, per the module docstring's ordering
    (WHICH_NAMES). `mode` in ("Awhich", "Awhich_shuf")."""
    ids, dec = _tok_ids_dec(text, T)
    clause = tree[:, 1]
    lex = lex_class_ids(dec, real)
    if mode == "Awhich_shuf":
        lex_use = shuffle_values(lex, real, rng)
        clause_event = shuffle_unit_ids(clause, real, rng)
    else:
        lex_use = lex
        clause_event = clause
    IDT = load_ident_codes()
    lexid_ch = lex_identity_channel(lex_use, real)
    retina_ch = retina_cosine_channel(ids, real, IDT)
    event_ch = event_channel(clause_event, real, dec)
    moment_ch = moment_channel(clause, real)
    row_ch, col_ch = role_cue_channels(clause, real, dec)
    return np.stack([lexid_ch, retina_ch, event_ch, moment_ch, row_ch, col_ch], axis=0)


# ---------------------------------------------------------------------
# the one entry point everything else calls
# ---------------------------------------------------------------------
def build_picture(state, tokmask, sent, text, mean, comps, mode, rng=None):
    """state (T,D) f32, tokmask (T,) f32 in {0,1}, sent (T,) int32, text
    str, mean/comps from fit_pca(). Returns (C,T,T) f32, C = 9 (Aprime),
    12 (Adouble / Adouble_shuf), or 15 (Awhich / Awhich_shuf, round 2)."""
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
            "exists today. Use 'Aprime', 'Adouble', 'Adouble_shuf', "
            "'Awhich' or 'Awhich_shuf'.")
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
    if mode in ("Adouble", "Adouble_shuf"):
        if mode == "Adouble":
            wc = where_channels(tree, real)
        else:   # Adouble_shuf
            rng = rng if rng is not None else np.random.default_rng(0)
            wc = shuffled_where_channels(tree, real, rng)
        return np.concatenate([pic, wc], axis=0)   # (12, T, T)
    # Awhich / Awhich_shuf (round 2)
    rng = rng if rng is not None else np.random.default_rng(0)
    wh = which_channels(text, real, tree, mode, rng, T)
    return np.concatenate([pic, wh], axis=0)   # (15, T, T)


def n_channels(mode):
    if mode == "Aprime":
        return N_CONTENT
    if mode in ("Adouble", "Adouble_shuf"):
        return N_CONTENT + N_WHERE
    if mode in ("Awhich", "Awhich_shuf"):
        return N_CONTENT + N_WHICH
    raise ValueError(mode)


def n_where(mode):
    """Channels the per-breath WHERE schedule (model.py's `where_sched`)
    modulates: 3 for round 1's Adouble/Adouble_shuf, 0 otherwise (Awhich/
    Awhich_shuf carry no WHERE channels at all -- the per-breath schedule
    is round 1's organ, not asked for here; the 6 WHICH channels are
    always-weight-1 content, same as the cosine/PCA-band channels)."""
    return N_WHERE if mode in ("Adouble", "Adouble_shuf") else 0


if __name__ == "__main__":
    # tiny self-check: 2 real rows of the diet, T=32, every buildable
    # mode, printed shapes + value ranges. DEV=CPU; FAM env must already
    # be set (tree_row_ids needs phase1_algebra_head's tokenizer path).
    src = PictureSource("formpm35c", ".cache/form_mix_pm35c.jsonl")
    mean, comps = fit_pca()
    for i in range(2):
        st, tm, se = src.row(i, T=32)
        txt = src.text(i)
        for mode in ("Aprime", "Adouble", "Adouble_shuf", "Awhich", "Awhich_shuf"):
            pic = build_picture(st, tm, se, txt, mean, comps, mode,
                                 rng=np.random.default_rng(i))
            assert pic.shape[0] == n_channels(mode), (mode, pic.shape)
            print(f"row {i} {mode}: shape={pic.shape} "
                  f"range=[{pic.min():.3f},{pic.max():.3f}] "
                  f"nan={np.isnan(pic).any()}")
    try:
        build_picture(st, tm, se, txt, mean, comps, "A")
    except NotImplementedError as e:
        print("mode A stub OK:", str(e)[:60] + "...")

    # THE SHUFFLE PRESERVES BLOCK COUNTS (requirement 4): verify on one
    # row that (a) the EVENT shuffle's underlying clause-run SIZE
    # multiset is unchanged (shuffle_unit_ids's own contract) and (b) the
    # LEX-IDENTITY shuffle's per-class token COUNT histogram is unchanged
    # (shuffle_values's contract) -- both over the SAME real-token mask.
    st, tm, se = src.row(0, T=64)
    txt = src.text(0)
    real0 = tm > 0
    H = _head()
    tree0 = H.tree_row_ids(txt, tm, se, 64)
    rng0 = np.random.default_rng(0)
    clause0 = tree0[:, 1]
    clause_shuf = shuffle_unit_ids(clause0, real0, rng0)
    sizes_before = sorted(np.bincount(clause0[real0 & (clause0 >= 0)]).tolist())
    sizes_after = sorted(np.bincount(clause_shuf[real0 & (clause_shuf >= 0)]).tolist())
    print(f"EVENT shuffle run-size multiset preserved: {sizes_before == sizes_after} "
          f"(before={sizes_before} after={sizes_after})")
    assert sizes_before == sizes_after

    ids0, dec0 = _tok_ids_dec(txt, 64)
    lex0 = lex_class_ids(dec0, real0)
    lex_shuf = shuffle_values(lex0, real0, rng0)
    counts_before = sorted(np.bincount(lex0[lex0 >= 0]).tolist()) if (lex0 >= 0).any() else []
    counts_after = sorted(np.bincount(lex_shuf[lex_shuf >= 0]).tolist()) if (lex_shuf >= 0).any() else []
    print(f"LEX-IDENTITY shuffle class-size histogram preserved: {counts_before == counts_after} "
          f"(before={counts_before} after={counts_after})")
    assert counts_before == counts_after
    print(f"n_real positions unchanged by either shuffle: "
          f"{int(real0.sum())} (shuffles only touch real positions)")
