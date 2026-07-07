"""phase1_delta_head.py — the first trained Phase-1 parameters: slot-based delta head
+ parse-side 512d Matryoshka waist over the FROZEN Llama-3.2-1B L0-L3 trunk.

THE OBJECT (docs/phase1_skeleton_spec.md §2-§3, single-cycle v0 — no notebook/NACK yet):
NL problem text -> frozen trunk -> per-token 2048->512 WAIST (the parser's silhouette
tap, Matryoshka prefix-masked 512->128) -> 48 learned slot queries cross-attend the
waist states -> per-slot heads emit the factor delta in the deducer's vocabulary:

  presence(1) | type(3: row/col/cage) | op(5: given,add,sub,mul,div)
  | target = 4 x 10-way DIGIT heads (exact integer, symbolic-verifier-evaluable)
  | membership = 49-way multi-hot pointer over the N_MAX grid cells

ATTENTION-BOOTSTRAP LAW COMPLIANCE (§6 — wired from the FIRST run, not added later):
both new attention pathways get DIRECT supervision from step 0:
  * the slot->token cross-attention is supervised by the gold SPAN masks (a slot must
    attend its factor's token spans) — span sets, never assumed contiguous;
  * the 49-way membership pointer is supervised by the gold multi-hot membership (BCE).

TRUNK PRECOMPUTE (--precompute, GPU once): the trunk is FROZEN, so its token states are
a pure function of text -> compute once, cache fp16 (.cache/phase1_trunk_{split}.npz),
train the 3M-param head against cached states at CPU-trivial cost. NO-TRUNCATION HARD
ASSERT at tokenize time (a truncated statement is a corrupted gold label the factor-level
round-trip gate cannot see).

EVAL (the metric ladder, strictest last):
  field accs (present slots) -> factor exact-match -> problem exact-match ->
  **SOLVE RATE**: decode predicted factors -> problem_from_kenken -> solve_symbolic ->
  grid == gold solution. The parser is graded by whether the SYMBOLIC JAW can bite on
  what it emits — the two-jaws contract as an eval metric.

MATRYOSHKA WAIST: prefix width sampled per train step from {128,256,384,512} via an
INPUT mask tensor (no python branch -> one JIT graph); eval reports per-width metrics
(evaluation at any prefix width is free — §8.6). Capture hooks from birth: --capture
dumps per-token waist states (the parse-side silhouette, capture-once schema).

USAGE:
  CPU selftest:               .venv/bin/python3 scripts/phase1_delta_head.py --selftest
  Precompute trunk states:    DEV=AMD .venv/bin/python3 scripts/phase1_delta_head.py --precompute
  Train (smoke: STEPS=30):    DEV=AMD STEPS=600 .venv/bin/python3 scripts/phase1_delta_head.py --train
  Eval a ckpt (+ solve rate): DEV=AMD .venv/bin/python3 scripts/phase1_delta_head.py --eval [--width 512]
  Capture waist silhouettes:  DEV=AMD .venv/bin/python3 scripts/phase1_delta_head.py --capture
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))

import numpy as np

T_WINDOW = 512
H_TRUNK = 2048
H_WAIST = 512
L_SLOTS = 56   # curriculum corpus: up to 40 cages + 14 row/col = 54 factors
S_CELLS = 49
N_TYPES = 3            # row / col / cage
N_OPS = 5              # given add sub mul div (OP_VOCAB order)
N_DIGITS = 4           # exact integer targets up to 9999 (corpus max 1470)
N_HEADS = 8
WIDTHS = (128, 256, 384, 512)

OPS = ["given", "add", "sub", "mul", "div"]
TYPES = ["row", "col", "cage"]
TYPE_RANK_ = {"row": 0, "col": 1, "cage": 2}
MEM_POS_WEIGHT = 5.0   # membership multi-hot is sparse (2-7 of 49) — counter the
                       # all-negative local minimum with a positive-class weight
SENT_MAX = 64          # sentence-index embedding table size (max ~45 statements)

NL_TRAIN = ".cache/kenken_nl_train.jsonl"
NL_TEST = os.environ.get("KK_TEST", ".cache/kenken_nl_test.jsonl")
KK_TEST_NAME = os.environ.get("KK_TEST_NAME", "test")
TRUNK_NPY = ".cache/phase1_trunk_{split}.npy"     # raw fp16, np.memmap (84GB at 40k)
META_NPZ = ".cache/phase1_meta_{split}.npz"       # gold in compact dtypes + tokmask + sent
CKPT_PATH = ".cache/phase1_delta_head.safetensors"
TOKENIZER_JSON = ".cache/llama-3.2-1b-weights/tokenizer.json"


# ===========================================================================
# GOLD TENSORS (CPU) — jsonl + tokenizer offsets -> slot supervision arrays
# ===========================================================================

def build_gold(samples: list, offsets_per_sample: list) -> dict:
    """Slot tensors in canonical order (generator-sorted). Returns numpy dict."""
    n = len(samples)
    g = {
        "presence": np.zeros((n, L_SLOTS), np.float32),
        "type": np.zeros((n, L_SLOTS), np.int32),
        "op": np.zeros((n, L_SLOTS), np.int32),
        "digits": np.zeros((n, L_SLOTS, N_DIGITS), np.int32),
        "members": np.zeros((n, L_SLOTS, S_CELLS), np.float32),
        "span": np.zeros((n, L_SLOTS, T_WINDOW), np.float32),
        "is_cage": np.zeros((n, L_SLOTS), np.float32),
        "N": np.zeros((n,), np.int32),
    }
    for i, (smp, offs) in enumerate(zip(samples, offsets_per_sample)):
        # TEXT-ORDER slots (2026-07-06): slot j = j-th statement in READING order, so
        # slot->span attention is nearly positional (learnable); canonical-order slots
        # made the assignment a global grid-sort (circular: cells come FROM the
        # sentence). Canonicalization moved to the decoder — factor SETS are order-free.
        facs = sorted(smp["factors"],
                      key=lambda f: (min(s for (s, _) in f["spans"]),
                                     TYPE_RANK_[f["ftype"]], f["members_flat"][0]))
        assert len(facs) <= L_SLOTS, f"{len(facs)} factors > {L_SLOTS} slots"
        g["N"][i] = smp["N"]
        for j, f in enumerate(facs):
            g["presence"][i, j] = 1.0
            g["type"][i, j] = TYPES.index(f["ftype"])
            for m in f["members_flat"]:
                g["members"][i, j, m] = 1.0
            if f["ftype"] == "cage":
                g["is_cage"][i, j] = 1.0
                g["op"][i, j] = OPS.index(f["op"])
                t = int(f["target"])
                assert t < 10 ** N_DIGITS, f"target {t} exceeds {N_DIGITS} digits"
                for d in range(N_DIGITS):          # digits MSD-first
                    g["digits"][i, j, d] = (t // 10 ** (N_DIGITS - 1 - d)) % 10
            # span mask: token overlaps any gold char span (vectorized — the python
            # double loop was O(factors x T) per sample: hours at 40k samples)
            if i not in _off_cache:
                arr = np.asarray(offs[:T_WINDOW], dtype=np.int64)
                _off_cache[i] = (arr[:, 0], arr[:, 1], arr.shape[0])
            ts_a, te_a, n_tok = _off_cache[i]
            for (cs, ce) in f["spans"]:
                hit = (ts_a < ce) & (te_a > cs)
                g["span"][i, j, :n_tok][hit] = 1.0
        _off_cache.pop(i, None)
    return g


_off_cache: dict = {}


def tokenize_corpus(path: str):
    """Returns (samples, ids (n,T) int32, tokmask (n,T) f32, offsets list).
    HARD-ERRORS on truncation — a truncated statement is a corrupted gold label."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples = [json.loads(l) for l in open(path)]
    ids = np.zeros((len(samples), T_WINDOW), np.int32)
    mask = np.zeros((len(samples), T_WINDOW), np.float32)
    offsets = []
    for i, s in enumerate(samples):
        e = tok.encode(s["text"])
        if len(e.ids) > T_WINDOW:
            raise RuntimeError(
                f"TRUNCATION at {path}:{i} — {len(e.ids)} tokens > T={T_WINDOW}. "
                f"A truncated statement is a corrupted gold label; fix the template "
                f"budget (terse forms), do NOT raise T casually (JIT graph shape).")
        ids[i, : len(e.ids)] = e.ids
        mask[i, : len(e.ids)] = 1.0
        offsets.append(list(e.offsets))
    return samples, ids, mask, offsets


# ===========================================================================
# TRUNK PRECOMPUTE (GPU, once) — frozen L0-L3 states cached fp16
# ===========================================================================

def do_precompute() -> None:
    from tinygrad import Tensor, Device, dtypes
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm)

    class _H:
        pass
    host = _H()
    sd = load_llama_weights(os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    cfg = host.llama_cfg

    from numpy.lib.format import open_memmap
    jobs = [("train", NL_TRAIN), (KK_TEST_NAME, NL_TEST)]
    if os.environ.get("PRECOMPUTE_ONLY"):
        jobs = [(n_, p_) for n_, p_ in jobs if n_ == os.environ["PRECOMPUTE_ONLY"]]
    for split, path in jobs:
        samples, ids, mask, offsets = tokenize_corpus(path)
        n = len(samples)
        out_npy = TRUNK_NPY.format(split=split)
        # STREAM to a disk memmap — at 40k samples the states are ~84GB and must never
        # exist in RAM. The frozen trunk pays this ONCE (do NOT switch to live
        # forwards: couples head training to trunk throughput forever).
        states = open_memmap(out_npy, mode="w+", dtype=np.float16,
                             shape=(n, T_WINDOW, H_TRUNK))
        B = 8
        t0 = time.time()
        for s0 in range(0, n, B):
            sl = slice(s0, min(s0 + B, n))
            x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
            for layer in host.llama_layers:
                x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            x = _rms_norm(x, host.llama_layers[-1].ffn_norm, cfg.rms_norm_eps)
            chunk = x.cast(dtypes.float).realize().numpy()
            assert np.isfinite(chunk).all(), f"non-finite trunk states at {split}:{s0}"
            states[sl] = chunk.astype(np.float16)
            if s0 % 4096 == 0:
                print(f"  [{split}] {s0}/{n} ({time.time()-t0:.0f}s)", flush=True)
        states.flush()
        del states

        # gold in compact dtypes (span/members are large but binary; digits <= 9)
        gold = build_gold(samples, offsets)
        sent = np.stack([sentence_indices(s["text"], o, mask[i])
                         for i, (s, o) in enumerate(zip(samples, offsets))])
        np.savez_compressed(
            META_NPZ.format(split=split),
            tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
            presence=gold["presence"].astype(np.uint8),
            is_cage=gold["is_cage"].astype(np.uint8),
            type=gold["type"].astype(np.int8), op=gold["op"].astype(np.int8),
            digits=gold["digits"].astype(np.int8),
            members=gold["members"].astype(np.uint8),
            span=gold["span"].astype(np.uint8), N=gold["N"])
        print(f"[precompute] {split}: ({n},{T_WINDOW},{H_TRUNK}) -> {out_npy} "
              f"({os.path.getsize(out_npy)/1e9:.1f} GB) + meta "
              f"({os.path.getsize(META_NPZ.format(split=split))/1e6:.0f} MB)", flush=True)


def sentence_indices(text: str, offs, tokmask_row) -> np.ndarray:
    """Per-token sentence index (0-based), from '. ' boundaries + tokenizer offsets.
    Deterministic function of the text — an inductive-bias feature, not leakage:
    'match a discrete sentence code' is learnable where 'count periods across 370
    tokens in one cross-attention hop' is not (the 2026-07-06 span-loss diagnosis)."""
    bounds = []
    i = text.find(". ")
    while i != -1:
        bounds.append(i + 1)
        i = text.find(". ", i + 1)
    out = np.zeros((T_WINDOW,), np.int32)
    n_tok = int(np.asarray(tokmask_row).sum())
    arr = np.asarray(offs[: min(n_tok, T_WINDOW)], dtype=np.int64)
    if len(arr):
        idx = np.searchsorted(np.asarray(bounds, dtype=np.int64), arr[:, 0],
                              side="right")
        out[: len(arr)] = np.minimum(idx, SENT_MAX - 1)
    return out


def load_split(split: str):
    """states is a READ-ONLY MEMMAP (random-access batches, ~16MB/batch off NVMe).
    gold arrays stay in compact dtypes in RAM; callers cast per batch."""
    if split == "test":
        split = KK_TEST_NAME
    samples = [json.loads(l) for l in open(NL_TRAIN if split == "train" else NL_TEST)]
    states = np.load(TRUNK_NPY.format(split=split), mmap_mode="r")
    m = np.load(META_NPZ.format(split=split))
    gold = {"presence": m["presence"], "is_cage": m["is_cage"], "type": m["type"],
            "op": m["op"], "digits": m["digits"], "members": m["members"],
            "span": m["span"], "N": m["N"].astype(np.int32)}
    return samples, states, m["tokmask"], gold, m["sent"]


# ===========================================================================
# MODEL — waist + slot cross-attention + heads (the ONLY trained params)
# ===========================================================================

def build_head_params(seed: int = 0) -> dict:
    from tinygrad import Tensor, dtypes
    rng = np.random.RandomState(seed)

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x

    def lin(i, o, scale=None):
        s = scale if scale is not None else (1.0 / math.sqrt(i))
        return t(rng.randn(i, o) * s), t(np.zeros((o,)))

    p = {}
    p["waist_w"], p["waist_b"] = lin(H_TRUNK, H_WAIST)
    p["sent_emb"] = t(rng.randn(SENT_MAX, H_WAIST) * 0.1)
    p["slot_q"] = t(rng.randn(L_SLOTS, H_WAIST) * 0.02)
    for nm in ("wq", "wk", "wv", "wo"):
        p[f"attn_{nm}"], p[f"attn_{nm}_b"] = lin(H_WAIST, H_WAIST)
    p["ffn_w1"], p["ffn_b1"] = lin(H_WAIST, 2 * H_WAIST)
    p["ffn_w2"], p["ffn_b2"] = lin(2 * H_WAIST, H_WAIST)
    p["h_pres"], p["h_pres_b"] = lin(H_WAIST, 1)
    p["h_type"], p["h_type_b"] = lin(H_WAIST, N_TYPES)
    p["h_op"], p["h_op_b"] = lin(H_WAIST, N_OPS)
    p["h_dig"], p["h_dig_b"] = lin(H_WAIST, N_DIGITS * 10)
    p["h_mem"], p["h_mem_b"] = lin(H_WAIST, S_CELLS)
    return p


def head_forward(p: dict, trunk, tokmask, wmask, sent_idx):
    """trunk (B,T,2048) f32 · tokmask (B,T) · wmask (1,1,512) Matryoshka prefix mask
    · sent_idx (B,T) int sentence indices.
    Returns dict of logits + the attention map + waist states (the silhouette tap)."""
    B = trunk.shape[0]
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu()               # (B,T,512)
    waist = (waist + p["sent_emb"][sent_idx]) * wmask                  # sentence code in-band

    q = (p["slot_q"] @ p["attn_wq"] + p["attn_wq_b"])                   # (L,512)
    k = waist @ p["attn_wk"] + p["attn_wk_b"]                           # (B,T,512)
    v = waist @ p["attn_wv"] + p["attn_wv_b"]
    hd = H_WAIST // N_HEADS
    qh = q.reshape(L_SLOTS, N_HEADS, hd).transpose(0, 1)                # (h,L,hd)
    kh = k.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)              # (B,h,T,hd)
    vh = v.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
    scores = (qh.unsqueeze(0) @ kh.transpose(-2, -1)) / math.sqrt(hd)   # (B,h,L,T)
    scores = scores.clip(-1e4, 1e4)
    scores = scores + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4       # pad tokens out
    attn = scores.softmax(-1)                                           # (B,h,L,T)
    slot = (attn @ vh).permute(0, 2, 1, 3).reshape(B, L_SLOTS, H_WAIST)
    slot = slot @ p["attn_wo"] + p["attn_wo_b"] + p["slot_q"].unsqueeze(0)
    slot = slot + ((slot @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])

    out = {
        "pres": (slot @ p["h_pres"] + p["h_pres_b"]).squeeze(-1),       # (B,L)
        "type": slot @ p["h_type"] + p["h_type_b"],                     # (B,L,3)
        "op": slot @ p["h_op"] + p["h_op_b"],                           # (B,L,5)
        "dig": (slot @ p["h_dig"] + p["h_dig_b"]).reshape(B, L_SLOTS, N_DIGITS, 10),
        "mem": slot @ p["h_mem"] + p["h_mem_b"],                        # (B,L,49)
        "attn_mean": attn.mean(1),                                      # (B,L,T)
        "waist": waist,
    }
    return out


def head_loss(out, g):
    """g: dict of gold Tensors. Masked per-field CE/BCE + the span-attention anchor."""
    pres_g = g["presence"]
    n_pres = pres_g.sum() + 1e-6
    cage_g = g["is_cage"]
    n_cage = cage_g.sum() + 1e-6

    def bce(logits, target):
        return (logits.maximum(0) - logits * target
                + (1 + (-logits.abs()).exp()).log())

    l_pres = bce(out["pres"], pres_g).mean()
    l_type = ((out["type"].log_softmax(-1) * -1)
              .gather(-1, g["type"].unsqueeze(-1)).squeeze(-1) * pres_g).sum() / n_pres
    l_op = ((out["op"].log_softmax(-1) * -1)
            .gather(-1, g["op"].unsqueeze(-1)).squeeze(-1) * cage_g).sum() / n_cage
    l_dig = ((out["dig"].log_softmax(-1) * -1)
             .gather(-1, g["digits"].unsqueeze(-1)).squeeze(-1)
             .mean(-1) * cage_g).sum() / n_cage
    mem_w = 1.0 + (MEM_POS_WEIGHT - 1.0) * g["members"]
    l_mem = ((bce(out["mem"], g["members"]) * mem_w).mean(-1) * pres_g).sum() / n_pres
    span_norm = g["span"] / (g["span"].sum(-1, keepdim=True) + 1e-6)
    l_span = ((-(out["attn_mean"] + 1e-9).log() * span_norm).sum(-1)
              * pres_g).sum() / n_pres
    total = l_pres + l_type + l_op + l_dig + 2.0 * l_mem + l_span
    return total, {"pres": l_pres, "type": l_type, "op": l_op,
                   "dig": l_dig, "mem": l_mem, "span": l_span}


# ===========================================================================
# TRAIN
# ===========================================================================

def do_train(steps: int, lr: float, batch: int, seed: int) -> None:
    from tinygrad import Tensor, Device, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW

    _, states, tokmask, gold, sent = load_split("train")
    n = states.shape[0]
    p = build_head_params(seed)
    if int(os.environ.get("RESUME", "0")) and os.path.exists(CKPT_PATH):
        from tinygrad.nn.state import safe_load
        sd = safe_load(CKPT_PATH)
        assert set(sd.keys()) == set(p.keys()), "resume ckpt key mismatch (hard error)"
        for k in p:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
        print(f"[train] RESUMED from {CKPT_PATH}", flush=True)
    params = list(p.values())
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    # fixed input buffers (assign-in-place replay — compile once)
    def fix(a):
        return Tensor(a, dtype=dtypes.float).contiguous().realize()
    b_trunk = fix(np.zeros((batch, T_WINDOW, H_TRUNK), np.float32))
    b_tok = fix(np.zeros((batch, T_WINDOW), np.float32))
    b_wmask = fix(np.ones((1, 1, H_WAIST), np.float32))
    b_sent = Tensor(np.zeros((batch, T_WINDOW), np.int32),
                    dtype=dtypes.int).contiguous().realize()
    b_gold = {
        "presence": fix(np.zeros((batch, L_SLOTS), np.float32)),
        "is_cage": fix(np.zeros((batch, L_SLOTS), np.float32)),
        "members": fix(np.zeros((batch, L_SLOTS, S_CELLS), np.float32)),
        "span": fix(np.zeros((batch, L_SLOTS, T_WINDOW), np.float32)),
        "type": Tensor(np.zeros((batch, L_SLOTS), np.int32), dtype=dtypes.int).contiguous().realize(),
        "op": Tensor(np.zeros((batch, L_SLOTS), np.int32), dtype=dtypes.int).contiguous().realize(),
        "digits": Tensor(np.zeros((batch, L_SLOTS, N_DIGITS), np.int32), dtype=dtypes.int).contiguous().realize(),
    }

    @TinyJit
    def train_step():
        Tensor.training = True
        out = head_forward(p, b_trunk, b_tok, b_wmask, b_sent)
        total, parts = head_loss(out, b_gold)
        opt.zero_grad()
        total.backward()
        opt.step()
        return total.realize(), parts["mem"].realize(), parts["span"].realize()

    print(f"[train] n={n} steps={steps} batch={batch} lr={lr} "
          f"params={sum(int(np.prod(t.shape)) for t in params)/1e6:.1f}M", flush=True)
    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(n, batch, replace=False)
        w = WIDTHS[rng.randint(len(WIDTHS))]
        wm = np.zeros((1, 1, H_WAIST), np.float32); wm[..., :w] = 1.0
        b_trunk.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_tok.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_wmask.assign(Tensor(wm, dtype=dtypes.float).contiguous()).realize()
        b_sent.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        for kk in ("presence", "is_cage", "members", "span"):
            b_gold[kk].assign(Tensor(gold[kk][idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        for kk in ("type", "op", "digits"):
            b_gold[kk].assign(Tensor(gold[kk][idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        tot, lmem, lspan = train_step()
        if step % 25 == 0 or step == steps - 1:
            tv = float(tot.numpy())
            assert np.isfinite(tv), f"non-finite loss at step {step}"
            print(f"  step {step:4d} loss={tv:.4f} mem={float(lmem.numpy()):.4f} "
                  f"span={float(lspan.numpy()):.4f} w={w} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)

    from tinygrad.nn.state import safe_save
    safe_save({k: v for k, v in p.items()}, CKPT_PATH)
    print(f"[train] saved {CKPT_PATH}", flush=True)


# ===========================================================================
# EVAL — field accs -> factor exact -> problem exact -> SOLVE RATE
# ===========================================================================

def decode_slots(out_np: dict, N: int):
    """Numpy decode of one sample's head outputs -> predicted factor list."""
    facs = []
    for j in range(L_SLOTS):
        if out_np["pres"][j] <= 0.0:      # sigmoid>0.5 <=> logit>0
            continue
        ty = TYPES[int(out_np["type"][j].argmax())]
        mem = [int(m) for m in np.where(out_np["mem"][j] > 0.0)[0]]
        f = {"ftype": ty, "members_flat": mem}
        if ty == "cage":
            f["op"] = OPS[int(out_np["op"][j].argmax())]
            digs = out_np["dig"][j].argmax(-1)
            f["target"] = int(sum(d * 10 ** (N_DIGITS - 1 - i) for i, d in enumerate(digs)))
        facs.append(f)
    return facs


def do_eval(width: int, capture: bool = False) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    samples, states, tokmask, gold, sent = load_split("test")
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = states.shape[0]
    wm = np.zeros((1, 1, H_WAIST), np.float32); wm[..., :width] = 1.0

    stats = {"type": [0, 0], "op": [0, 0], "tgt": [0, 0], "mem_f1": [],
             "fac_exact": [0, 0], "prob_exact": 0, "solved": 0}
    waists = []
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        out = head_forward(
            p, Tensor(np.asarray(states[sl], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float),
            Tensor(wm, dtype=dtypes.float),
            Tensor(sent[sl].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
        if capture:
            waists.append(out["waist"].realize().numpy().astype(np.float16))
        for bi, i in enumerate(range(s0, min(s0 + 8, n))):
            smp = samples[i]
            gfac = smp["factors"]
            prob_ok = True
            for j in range(L_SLOTS):
                present_g = gold["presence"][i, j] > 0.5
                present_p = o["pres"][bi, j] > 0.0
                if present_g != present_p:
                    prob_ok = False
                if not present_g:
                    continue
                ty_ok = int(o["type"][bi, j].argmax()) == gold["type"][i, j]
                stats["type"][0] += ty_ok; stats["type"][1] += 1
                mem_p = o["mem"][bi, j] > 0.0
                mem_g = gold["members"][i, j] > 0.5
                tp = float((mem_p & mem_g).sum())
                f1 = 2 * tp / max(mem_p.sum() + mem_g.sum(), 1e-6)
                stats["mem_f1"].append(f1)
                fac_ok = ty_ok and bool((mem_p == mem_g).all())
                if gold["is_cage"][i, j] > 0.5:
                    op_ok = int(o["op"][bi, j].argmax()) == gold["op"][i, j]
                    dig_ok = bool((o["dig"][bi, j].argmax(-1) == gold["digits"][i, j]).all())
                    stats["op"][0] += op_ok; stats["op"][1] += 1
                    stats["tgt"][0] += dig_ok; stats["tgt"][1] += 1
                    fac_ok = fac_ok and op_ok and dig_ok
                stats["fac_exact"][0] += fac_ok; stats["fac_exact"][1] += 1
                prob_ok = prob_ok and fac_ok
            stats["prob_exact"] += prob_ok
            stats["solved"] += _solve_rate_one(o, bi, smp)

    print(f"\n[eval] width={width} n={n}")
    print(f"  type acc     {stats['type'][0]/max(stats['type'][1],1):.3f}")
    print(f"  op acc       {stats['op'][0]/max(stats['op'][1],1):.3f}")
    print(f"  target acc   {stats['tgt'][0]/max(stats['tgt'][1],1):.3f}")
    print(f"  member F1    {float(np.mean(stats['mem_f1'])):.3f}")
    print(f"  factor exact {stats['fac_exact'][0]/max(stats['fac_exact'][1],1):.3f}")
    print(f"  problem exact {stats['prob_exact']}/{n}")
    print(f"  SOLVE RATE   {stats['solved']}/{n}   "
          f"(predicted parse -> symbolic solve -> gold grid)")
    if capture:
        w = np.concatenate(waists, axis=0)
        out = ".cache/phase1_waist_silhouettes_test.npz"
        np.savez_compressed(out, waist=w, tokmask=tokmask,
                            meta=np.array(json.dumps({"width": width, "ckpt": CKPT_PATH})))
        print(f"  [capture] parse-side silhouettes {w.shape} -> {out}")


def _solve_rate_one(o: dict, bi: int, smp: dict) -> int:
    """Decode predicted factors -> search tier -> exact gold grid? (0/1)"""
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    N = int(smp["N"])
    pred = decode_slots({k: o[k][bi] for k in o}, N)
    cages, clues = [], []
    for f in pred:
        if f["ftype"] != "cage":
            continue
        rc = [[m // 7, m % 7] for m in f["members_flat"]]
        if any(r >= N or c >= N for (r, c) in rc) or not rc:
            return 0                       # membership outside the N-grid: unusable parse
        cages.append(rc)
        clues.append([f["op"], f["target"]])
    try:
        res = solve_symbolic(problem_from_kenken(N, cages, clues), budget=100_000, seed=0)
    except Exception:
        return 0
    if res["status"] != "solved":
        return 0
    asg = res["assignment"]
    grid = [[int(asg[r * N + c]) for c in range(N)] for r in range(N)]
    gold_grid, gi = _gold_grid_cache(smp)
    return int(grid == gold_grid)


_GOLD_CACHE: dict = {}


def _gold_grid_cache(smp: dict):
    key = (smp["source"]["path"], smp["source"]["idx"])
    if key not in _GOLD_CACHE:
        rec = json.loads(open(key[0]).readlines()[key[1]])
        _GOLD_CACHE[key] = ([[int(v) for v in row] for row in rec["solution"]], key)
    return _GOLD_CACHE[key]


# ===========================================================================
# ERROR TAXONOMY (--errors) — detectable vs SILENT parse errors (pre-Brick-C gate)
# ===========================================================================
# The Alternator's premise: Phase 1 needn't be one-shot perfect IF errors are
# symbolically detectable (verifier UNSAT / non-uniqueness) — the NACK loop can
# retransmit those. SILENT errors (a plausible wrong graph solving UNIQUELY to a
# wrong grid) are unrecoverable by any retransmission — nobody flags them. The
# detectable fraction is the CEILING on what alternation recovers over staged
# parsing. Non-uniqueness is a GOLD-FREE flag: KenKen is unique-by-construction,
# so "my parse admits >=2 solutions" self-reports as under-constrained.
#
# HONEST SCOPE: this measures the SYMBOLIC NACK ceiling (stack tiers 1-2: exact
# verifier + uniqueness). Tier 3 (the deduce-side late-JSD field, AUC 0.687 on soft
# state) could in principle flag regions of even SILENT-wrong graphs as "felt hard,"
# adding recall above this ceiling in the full Alternator — the taxonomy is the limit
# of the exact jaws, not of the loop.

def _second_solution_exists(n, cages, clues, grid, budget=50_000) -> bool:
    """Ban each cell's found value in turn; any SAT re-solve proves multiplicity."""
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic
    for r in range(n):
        for c in range(n):
            prob = problem_from_kenken(n, cages, clues)
            dom = prob.domains0[r * n + c]
            dom.discard(grid[r][c])
            if not dom:
                continue
            try:
                res = solve_symbolic(prob, budget=budget, seed=0)
            except Exception:
                continue
            if res["status"] == "solved":
                return True
    return False


def do_error_taxonomy(width: int) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_split("test")
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = states.shape[0]
    wm = np.zeros((1, 1, H_WAIST), np.float32); wm[..., :width] = 1.0

    cats = {"CORRECT": 0, "DETECT_malformed": 0, "DETECT_unsat": 0,
            "DETECT_multi": 0, "SILENT": 0}
    FIELDS = ("phantom", "dropped", "type", "op", "target", "member")
    by_field: dict = {c: {f: 0 for f in FIELDS} for c in cats}

    def field_errors(o, bi, i) -> list:
        errs = set()
        for j in range(L_SLOTS):
            pg = gold["presence"][i, j] > 0.5
            pp = o["pres"][bi, j] > 0.0
            if pp and not pg:
                errs.add("phantom")     # emitted a factor gold doesn't have
            elif pg and not pp:
                errs.add("dropped")     # failed to emit a gold factor
            if not pg:
                continue
            if int(o["type"][bi, j].argmax()) != gold["type"][i, j]:
                errs.add("type")
            if not bool(((o["mem"][bi, j] > 0.0) == (gold["members"][i, j] > 0.5)).all()):
                errs.add("member")
            if gold["is_cage"][i, j] > 0.5:
                if int(o["op"][bi, j].argmax()) != gold["op"][i, j]:
                    errs.add("op")
                if not bool((o["dig"][bi, j].argmax(-1) == gold["digits"][i, j]).all()):
                    errs.add("target")
        return sorted(errs)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        out = head_forward(
            p, Tensor(np.asarray(states[sl], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float),
            Tensor(wm, dtype=dtypes.float),
            Tensor(sent[sl].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
        for bi, i in enumerate(range(s0, min(s0 + 8, n))):
            smp = samples[i]
            N = int(smp["N"])
            pred = decode_slots({k: o[k][bi] for k in o}, N)
            cages, clues, ok = [], [], True
            for f in pred:
                if f["ftype"] != "cage":
                    continue
                rc = [[m // 7, m % 7] for m in f["members_flat"]]
                if not rc or any(r >= N or c >= N for (r, c) in rc):
                    ok = False
                    break
                cages.append(rc); clues.append([f["op"], f["target"]])
            if not ok:
                cat = "DETECT_malformed"
            else:
                try:
                    res = solve_symbolic(problem_from_kenken(N, cages, clues),
                                         budget=100_000, seed=0)
                except Exception:
                    res = {"status": "error"}
                if res["status"] != "solved":
                    cat = "DETECT_unsat"
                else:
                    asg = res["assignment"]
                    grid = [[int(asg[r * N + c]) for c in range(N)] for r in range(N)]
                    gold_grid, _ = _gold_grid_cache(smp)
                    if grid == gold_grid:
                        cat = "CORRECT"
                    elif _second_solution_exists(N, cages, clues, grid):
                        cat = "DETECT_multi"
                    else:
                        cat = "SILENT"
            cats[cat] += 1
            for fe in field_errors(o, bi, i):
                by_field[cat][fe] += 1

    wrong = n - cats["CORRECT"]
    detect = cats["DETECT_malformed"] + cats["DETECT_unsat"] + cats["DETECT_multi"]
    print(f"\n[errors] width={width} n={n}")
    print(f"  {'category':18s} {'n':>4s}  " + "  ".join(f"{f:>8s}" for f in FIELDS))
    for k, v in cats.items():
        row = "  ".join(f"{by_field[k][f]:8d}" for f in FIELDS)
        print(f"  {k:18s} {v:4d}  {row}")
    if wrong:
        print(f"  DETECTABLE fraction of errors: {detect}/{wrong} = {detect/wrong:.2f}"
              f"   (the SYMBOLIC-jaw NACK ceiling — tier-3 late-JSD may add recall; "
              f"SILENT = unrecoverable by exact flags)")


# ===========================================================================
# BLAME SWEEP (--blame) — Brick-C localization v0, measured against gold TODAY
# ===========================================================================
# Delete-one-factor re-solve: for each UNSAT parse, remove each predicted cage factor
# in turn and re-solve; turning SAT fingers that factor as an unsat-core member.
# O(F) search-tier calls (median-zero decisions) ~= less than one deducer breath.
# Because eval has gold, blame quality is measurable NOW: a predicted factor is
# GOLD-WRONG iff no gold cage factor has the same (members, op, target). Report
# precision/recall of the blamed set vs the gold-wrong set. Overlapping errors smear
# blame (removal of ONE of several wrong factors may not restore SAT) — recall
# quantifies exactly that smear. Confidence-ordered sweeps come later (tier-0 head).

def do_blame_sweep(width: int) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_split("test")
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = states.shape[0]
    wm = np.zeros((1, 1, H_WAIST), np.float32); wm[..., :width] = 1.0

    precs, recs, n_unsat, n_restored = [], [], 0, 0
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        out = head_forward(
            p, Tensor(np.asarray(states[sl], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float),
            Tensor(wm, dtype=dtypes.float),
            Tensor(sent[sl].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
        for bi, i in enumerate(range(s0, min(s0 + 8, n))):
            smp = samples[i]
            N = int(smp["N"])
            pred = [f for f in decode_slots({k: o[k][bi] for k in o}, N)
                    if f["ftype"] == "cage"]
            cages, clues, ok = [], [], True
            for f in pred:
                rc = [[m // 7, m % 7] for m in f["members_flat"]]
                if not rc or any(r >= N or c >= N for (r, c) in rc):
                    ok = False
                    break
                cages.append(rc); clues.append([f["op"], f["target"]])
            if not ok:
                continue                        # malformed: caught pre-solve, no sweep
            try:
                res = solve_symbolic(problem_from_kenken(N, cages, clues),
                                     budget=100_000, seed=0)
            except Exception:
                continue
            if res["status"] == "solved":
                continue                        # sweep is for UNSAT parses only
            n_unsat += 1

            gold_keys = {(tuple(sorted(f["members_flat"])), f["op"], f["target"])
                         for f in smp["factors"] if f["ftype"] == "cage"}
            gold_wrong = {j for j, f in enumerate(pred)
                          if (tuple(sorted(f["members_flat"])), f["op"], f["target"])
                          not in gold_keys}
            blamed = set()
            for j in range(len(pred)):
                c2 = cages[:j] + cages[j + 1:]
                l2 = clues[:j] + clues[j + 1:]
                try:
                    r2 = solve_symbolic(problem_from_kenken(N, c2, l2),
                                        budget=100_000, seed=0)
                except Exception:
                    continue
                if r2["status"] == "solved":
                    blamed.add(j)
            if blamed:
                n_restored += 1
            tp = len(blamed & gold_wrong)
            precs.append(tp / len(blamed) if blamed else float("nan"))
            recs.append(tp / len(gold_wrong) if gold_wrong else float("nan"))

    precs_v = [x for x in precs if not math.isnan(x)]
    recs_v = [x for x in recs if not math.isnan(x)]
    print(f"\n[blame] width={width}  UNSAT parses swept: {n_unsat}")
    print(f"  sweep restored SAT (>=1 factor blamed): {n_restored}/{n_unsat}")
    if precs_v:
        print(f"  blame precision (blamed -> gold-wrong): {np.mean(precs_v):.3f}")
    if recs_v:
        print(f"  blame recall    (gold-wrong -> blamed): {np.mean(recs_v):.3f}")
    print("  (recall < 1 quantifies multi-error blame smear; "
          "confidence-ordered sweeps are the tier-0 upgrade)")


# ===========================================================================
# SELFTEST (CPU) — shapes, loss flow, span-anchor sanity, decode round trip
# ===========================================================================

def selftest() -> None:
    os.environ.setdefault("DEV", "CPU")
    from tinygrad import Tensor, dtypes

    p = build_head_params(0)
    B = 2
    trunk = Tensor(np.random.RandomState(0).randn(B, T_WINDOW, H_TRUNK).astype(np.float32) * 0.1)
    tokmask = Tensor(np.ones((B, T_WINDOW), np.float32))
    wm = Tensor(np.ones((1, 1, H_WAIST), np.float32))
    sent0 = Tensor(np.zeros((B, T_WINDOW), np.int32), dtype=dtypes.int)
    out = head_forward(p, trunk, tokmask, wm, sent0)
    assert out["pres"].shape == (B, L_SLOTS) and out["mem"].shape == (B, L_SLOTS, S_CELLS)

    g_np = {
        "presence": np.zeros((B, L_SLOTS), np.float32),
        "is_cage": np.zeros((B, L_SLOTS), np.float32),
        "members": np.zeros((B, L_SLOTS, S_CELLS), np.float32),
        "span": np.zeros((B, L_SLOTS, T_WINDOW), np.float32),
        "type": np.zeros((B, L_SLOTS), np.int32),
        "op": np.zeros((B, L_SLOTS), np.int32),
        "digits": np.zeros((B, L_SLOTS, N_DIGITS), np.int32),
    }
    g_np["presence"][:, :3] = 1
    g_np["is_cage"][:, 2] = 1
    g_np["members"][:, :3, :4] = 1
    g_np["span"][:, :3, 5:9] = 1
    g = {k: Tensor(v, dtype=dtypes.int if v.dtype == np.int32 else dtypes.float)
         for k, v in g_np.items()}
    total, parts = head_loss(out, g)
    tv = float(total.numpy())
    assert np.isfinite(tv), "loss not finite"
    total.backward()
    gnorm = float(p["waist_w"].grad.abs().max().numpy())
    assert np.isfinite(gnorm) and gnorm > 0, "no gradient into the waist"
    print(f"  [OK] forward/loss/backward finite (loss={tv:.3f}, waist |grad|max={gnorm:.2e})")

    # decode round trip: perfect logits -> exact factor recovery
    o = {"pres": np.full((L_SLOTS,), -10.0, np.float32),
         "type": np.zeros((L_SLOTS, N_TYPES), np.float32),
         "op": np.zeros((L_SLOTS, N_OPS), np.float32),
         "dig": np.zeros((L_SLOTS, N_DIGITS, 10), np.float32),
         "mem": np.full((L_SLOTS, S_CELLS), -10.0, np.float32)}
    o["pres"][0] = 10.0
    o["type"][0, 2] = 10.0
    o["op"][0, 3] = 10.0                      # mul
    for i, d in enumerate((0, 0, 1, 2)):      # target 12
        o["dig"][0, i, d] = 10.0
    o["mem"][0, [0, 1]] = 10.0
    facs = decode_slots(o, 7)
    assert facs == [{"ftype": "cage", "members_flat": [0, 1], "op": "mul", "target": 12}], facs
    print("  [OK] decode round trip (slot logits -> exact factor)")

    # gold builder: token-span overlap logic on a toy offsets list
    smp = {"N": 4, "factors": [
        {"ftype": "row", "members_flat": [0, 1, 2, 3], "op": None, "target": None,
         "spans": [[0, 10]]},
        {"ftype": "cage", "members_flat": [0, 1], "op": "add", "target": 5,
         "spans": [[12, 20], [30, 38]]}]}
    offs = [(i * 4, i * 4 + 4) for i in range(T_WINDOW)]
    g2 = build_gold([smp], [offs])
    assert g2["span"][0, 0, :3].sum() == 3 and g2["span"][0, 0, 3] == 0
    assert g2["span"][0, 1].sum() == 5      # two disjoint spans: [12,20)->2 + [30,38)->3 tokens
    assert list(g2["digits"][0, 1]) == [0, 0, 0, 5]
    print("  [OK] gold builder (span sets -> token masks, digits MSD-first)")
    print("[selftest] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--errors", action="store_true",
                    help="detectable-vs-silent parse-error taxonomy (pre-Brick-C gate)")
    ap.add_argument("--blame", action="store_true",
                    help="delete-one-factor blame sweep on UNSAT parses (Brick-C loc v0)")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--width", type=int, default=H_WAIST)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.precompute:
        do_precompute()
    elif args.train:
        do_train(steps=int(os.environ.get("STEPS", "600")),
                 lr=float(os.environ.get("LR", "3e-4")),
                 batch=int(os.environ.get("BATCH", "8")),
                 seed=int(os.environ.get("SEED", "0")))
    elif args.errors:
        do_error_taxonomy(args.width)
    elif args.blame:
        do_blame_sweep(args.width)
    elif args.eval or args.capture:
        do_eval(args.width, capture=args.capture)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
