"""distill_reader.py -- (B) THE DISTILLED READER (ledger 2026-10-26 19:27, word given). The vanilla reading half of the
system map: a small BIDIRECTIONAL transformer trained to reproduce the frozen Llama-3.2-1B L0-L3 BP states (bidirectional,
pad keys masked, final RMSNorm) from the banked teacher memmap -- no Llama forward at train time.
  embed   the FROZEN Llama embedding table (a lookup, not a transformer) -> learned 2048 -> D projection
  body    N pre-norm blocks: RMSNorm, MHA (RoPE, theta 500000, pad KEYS masked), RMSNorm, SwiGLU
  out     RMSNorm -> D -> 2048 (+ bias initialized to the teacher's per-dim mean)
  loss    on REAL tokens only: MSE / teacher variance + (1 - cos)
Modes:
  train : DR_STEPS, DR_BATCH, DR_LR, DR_D, DR_LAYERS, DR_TAG -> .cache/distill_reader_{TAG}.safetensors (+ .json, .log lines)
  --export JSONL --src SPLIT --out SPLIT2 : run the student over JSONL and write the states memmap under SPLIT2 + a copy of
          SPLIT's npz (tokmask/sent/gold are state-independent; tokmask asserted equal), so ALG_TEST_NAME=SPLIT2 reads it.
          Pad positions are written as ZEROS (the student is never trained on them).
          DR_VOCAB=fold: EMB @ w_in folded into one fp16 V x D table; DR_VOCAB=prune: + reader_vocab.py's kept set, dropped
          tokens = the mean of their fallback pieces' rows (ledger 10-27, the shrunk reader)
  --padzero SRC --out SPLIT2 : the teacher's own states with pads zeroed (the feasibility control: do pads matter?)"""
import argparse, json, math, os, queue, shutil, sys, threading, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "9"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_WIDE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--export", default=""); ap.add_argument("--src", default=""); ap.add_argument("--out", default="")
ap.add_argument("--padzero", default=""); ap.add_argument("--gsm", default=""); ap.add_argument("--table", default="")
args = ap.parse_args()
assert os.environ.get("DEV") in ("PCI+AMD", "CPU"), "DEV=PCI+AMD (never bare AMD) or CPU"
import numpy as np
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load
from phase1_algebra_head import tokenize, T_ALG, H_TRUNK, STATES_NPZ, STATES_NPY
from mycelium.llama_loader import load_llama_weights, _build_rope_freqs, _apply_rope, _rms_norm, _rotate_half
from mycelium import rotor_clock as CK

TAG = os.environ.get("DR_TAG", "s6x512")
D, NL = int(os.environ.get("DR_D", 512)), int(os.environ.get("DR_LAYERS", 6))
NH, HD = D // 64, 64
FF = int(os.environ.get("DR_FF", 1408))          # 0 = THE MLP-FREE STUDENT (ledger 10-30): no feed-forward anywhere
DBIN = os.environ.get("DR_DB") == "1"              # + the reader DB row (reader_db_build.py) as a mandatory road at the input
T6 = os.environ.get("DR_T6") in ("1", "2", "3")         # (ledger 10-31, U2; DR_T6=2 = HY, ledger 11-01: RoPE kept on planes [0:24), T6 x the SENTENCE on
# RoPE's 8 idle planes [24:32) -- they turn < 0.014 rad over the whole 256-token window at theta 500000)
HY = os.environ.get("DR_T6") == "2"
T9NS = os.environ.get("DR_T9NS") == "1"   # (ledger 11-01) T9 with its 6 sentence planes QUIET (the sentence-address test)
RSEED = int(os.environ.get("DR_SEED", "0"))   # (ledger 11-01) the replication: init seed RSEED, loader seed 2 + RSEED (0 = the banked runs)
assert (os.environ.get("DR_T6") in ("1", "3")) or os.environ.get("DR_LEGACY_ROPE") == "1", \
    "RoPE retired (ledger 11-03): the reader runs on the T9 clock (DR_T6=3); RoPE forms (no DR_T6 / DR_T6=2) need DR_LEGACY_ROPE=1"
T9 = os.environ.get("DR_T6") == "3"   # (ledger 11-01) T6 offset | T6 sentence | gears 7-9 x token | quiet planes
# (U2) NO RoPE: per head, planes [0:HD/4) turn by the T6 gears x the
# SENTENCE index, planes [HD/4:HD/2) by the T6 gears x the OFFSET in the sentence (hierarchical address on disjoint planes)
GSM = os.environ.get("DR_GSM") == "1"              # + GSM8K TRAIN text (ledger 10-31): a native head on the BANKED trunk word states
BR = int(os.environ.get("DR_BREATHS", "0"))      # (ledger 11-02) THE BREATHING READER: DR_LAYERS weight-tied layers x BR breaths
WARM = os.environ.get("DR_WARM", "")               # continue from this checkpoint (every key must match)
SAVE_AT = int(os.environ.get("DR_SAVE_AT", 0))     # an extra snapshot at this step (-> ..._{TAG}_s{step}.safetensors)
DBNPY, DBP = ".cache/readerdb_{split}.npy", 256
STEPS, BATCH, LR = int(os.environ.get("DR_STEPS", 30000)), int(os.environ.get("DR_BATCH", 32)), float(os.environ.get("DR_LR", 3e-4))
TRAIN_JSONL, TRAIN_SPLIT = ".cache/form_mix_pm35c.jsonl", "formpm35cBP"
CKPT, META = f".cache/distill_reader_{TAG}.safetensors", f".cache/distill_reader_{TAG}.json"
if (args.export or args.gsm or args.table) and os.environ.get("DR_CKPT"): CKPT = os.environ["DR_CKPT"]       # export a snapshot (DR_SAVE_AT)
EPS = 1e-5
PROF = bool(int(os.environ.get("DR_PROF", "0")))


def padzero(src, out):
    s_npy, d_npy = STATES_NPY.format(split=src), STATES_NPY.format(split=out)
    assert not os.path.exists(d_npy), f"{d_npy} exists -- refusing to overwrite"
    a = np.load(s_npy, mmap_mode="r"); tm = np.load(STATES_NPZ.format(split=src))["tokmask"]
    shutil.copyfile(STATES_NPZ.format(split=src), STATES_NPZ.format(split=out))
    st = np.lib.format.open_memmap(d_npy, mode="w+", dtype=np.float16, shape=a.shape)
    for s0 in range(0, len(a), 256):
        st[s0:s0 + 256] = np.asarray(a[s0:s0 + 256]) * tm[s0:s0 + 256, :, None].astype(np.float16)
    st.flush(); print(f"[padzero] {src} -> {out}: {len(a)} rows, pads zeroed", flush=True)


if args.padzero:
    assert args.out; padzero(args.padzero, args.out); sys.exit(0)

TABLE = os.environ.get("DR_TABLE", "")   # THE STUDENT'S OWN TABLE (ledger 10-31, Llama retired): folded + pruned, Llama's file never opened
if TABLE and (args.export or args.gsm):
    _t = np.load(TABLE); EMB = np.zeros((int(_t["V"]), int(_t["rows"].shape[1])), np.float16); EMB[_t["keep"]] = _t["rows"]
    for _j, _i in enumerate(_t["oov_ids"]):
        EMB[_i] = EMB[_t["pieces"][_t["piece_ptr"][_j]:_t["piece_ptr"][_j + 1]]].astype(np.float32).mean(0).astype(np.float16)
    print(f"[table] {TABLE}: {len(_t['keep'])} owned rows x {EMB.shape[1]} + {len(_t['oov_ids'])} composed (no Llama file)", flush=True)
else:
    sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
    EMB = sd["model.embed_tokens.weight"].to(Device.DEFAULT).cast(dtypes.half).realize().numpy()   # frozen lookup, gathered on the HOST
    # (a device-side EMB[ids] lowers to a compare-and-reduce over the whole 128k-row table: ~2e12 ops per batch, 16 s/step)
    del sd
COS, SIN = _build_rope_freqs(HD, T_ALG, 500000.0)
SPD = np.radians(np.asarray(CK.t6_plane_speeds(HD // 4), np.float32))
NRP = 24                                                                     # HY: RoPE keeps planes [0:NRP)
_ang = np.outer(np.arange(T_ALG, dtype=np.float32), 1.0 / (500000.0 ** (np.arange(0, HD // 2, dtype=np.float32) * 2 / HD)))
RC, RS = np.cos(_ang), np.sin(_ang)                                          # HY: _build_rope_freqs's angles, on the host
SPD8 = np.radians(np.asarray(CK.t6_plane_speeds(HD // 2 - NRP), np.float32))


def t6_rot(sent, tm):
    """sent, tm (B, T) -> (cos, sin) (B, 1, T, HD) in _apply_rope's half-split layout: sentence planes then offset planes"""
    T = sent.shape[1]; i = np.arange(T)[None]; chg = np.ones_like(sent, bool); chg[:, 1:] = sent[:, 1:] != sent[:, :-1]
    off = i - np.maximum.accumulate(np.where(chg, i, 0), 1)
    if T9:                                                                                            # the clock owns every plane
        G = np.radians(np.asarray(CK.T6_SPEEDS, np.float32)); S9 = 360.0 / np.asarray([192.0, 768.0, 3072.0], np.float32)
        a = np.concatenate([off[..., None] * G, sent[..., None] * G * (0.0 if T9NS else 1.0), i[..., None].repeat(sent.shape[0], 0) * np.radians(S9),
                            np.zeros(sent.shape + (HD // 2 - 15,), np.float32)], -1) * (tm[..., None] > 0)
        a = np.concatenate([a, a], -1)[:, None].astype(np.float32)
        return np.cos(a), np.sin(a)
    if HY:                                                                                            # RoPE micro | T6 macro, disjoint planes
        c, sn = np.broadcast_to(RC[None, :T], sent.shape + (HD // 2,)).copy(), np.broadcast_to(RS[None, :T], sent.shape + (HD // 2,)).copy()
        a6 = sent[..., None] * SPD8 * (0.0 if os.environ.get("DR_HY0") == "1" else 1.0)   # HY0 (ledger 11-01): speeds 0 = NMDB math
        c[..., NRP:], sn[..., NRP:] = np.cos(a6), np.sin(a6)
        c, sn = np.concatenate([c, c], -1)[:, None].astype(np.float32), np.concatenate([sn, sn], -1)[:, None].astype(np.float32)
        return c, sn
    a = np.concatenate([sent[..., None] * SPD, off[..., None] * SPD], -1) * (tm[..., None] > 0)     # (B, T, HD/2)
    a = np.concatenate([a, a], -1)[:, None].astype(np.float32)
    return np.cos(a), np.sin(a)


if BR:   # (ledger 11-02 MBR) per-breath plane masks for the coarse-to-fine clock + the query-only breath stamp, constants outside the JIT
    assert T9 and T9NS, "the breathing reader runs on the T9-NS clock (no RoPE)"
    _UNLOCK = {4: (3, 2, 1, 0), 3: (3, 1, 0), 2: (3, 0), 1: (0,)}[BR]       # first unlocked offset gear (index into T6_SPEEDS, fastest = 0)
    _GQ = np.radians(np.asarray(CK.T6_SPEEDS, np.float32))
    BRM, BRQC, BRQS = [], [], []
    for _b in range(BR):
        _m = np.ones(HD // 2, np.float32); _m[:_UNLOCK[_b]] = 0.0               # offset planes [0:6): the faster gears quiet this breath
        _qa = np.zeros(HD // 2, np.float32); _qa[15:21] = (_b + 1) * _GQ        # the breath stamp, queries only, quiet planes [15:21)
        _m, _qa = np.concatenate([_m, _m]), np.concatenate([_qa, _qa])
        BRM.append(Tensor(_m, dtype=dtypes.float).reshape(1, 1, 1, HD).contiguous().realize())
        BRQC.append(Tensor(np.cos(_qa), dtype=dtypes.float).reshape(1, 1, 1, HD).contiguous().realize())
        BRQS.append(Tensor(np.sin(_qa), dtype=dtypes.float).reshape(1, 1, 1, HD).contiguous().realize())


def init_params(mean_out):
    rng = np.random.default_rng(RSEED)
    nrm = lambda *s: Tensor(rng.normal(0, 1 / math.sqrt(s[0]), s).astype(np.float32), dtype=dtypes.float).contiguous()
    p = {"w_in": nrm(H_TRUNK, D), **({"w_db": nrm(DBP, D)} if DBIN else {}), "fnorm": Tensor.ones(D, dtype=dtypes.float).contiguous(),
         "w_out": nrm(D, H_TRUNK), "b_out": Tensor(mean_out.astype(np.float32), dtype=dtypes.float).contiguous()}
    for i in range(NL):
        p[f"an{i}"] = Tensor.ones(D, dtype=dtypes.float).contiguous(); p[f"fn{i}"] = Tensor.ones(D, dtype=dtypes.float).contiguous()
        for k in ("wq", "wk", "wv"):
            p[f"{k}{i}"] = nrm(D, D)
        p[f"wo{i}"] = Tensor.zeros(D, D, dtype=dtypes.float).contiguous()         # ReZero-style: blocks start as identity
        if FF:
            p[f"wg{i}"], p[f"wu{i}"] = nrm(D, FF), nrm(D, FF)
            p[f"wd{i}"] = Tensor.zeros(FF, D, dtype=dtypes.float).contiguous()
        else:
            del p[f"fn{i}"]
    return p


def forward(p, x0, am, db=None, rc=None):
    return head(p, body(p, x0, am, db, rc))


def head(p, x):
    return _rms_norm(x, p["fnorm"], EPS) @ p["w_out"] + p["b_out"]


def body(p, x0, am, db=None, rc=None, every=False):
    B = x0.shape[0]
    x = x0.cast(dtypes.float) @ p["w_in"]
    if db is not None:
        x = x + db.cast(dtypes.float) @ p["w_db"]
    xs = []
    for br in range(max(BR, 1)):
      if BR:   # the coarse-to-fine clock: quiet planes read cos 1 / sin 0
        kc, ks = rc[0] * BRM[br] + (1 - BRM[br]), rc[1] * BRM[br]
      for i in range(NL):
        h = _rms_norm(x, p[f"an{i}"], EPS)
        q = (h @ p[f"wq{i}"]).reshape(B, T_ALG, NH, HD).transpose(1, 2)
        k = (h @ p[f"wk{i}"]).reshape(B, T_ALG, NH, HD).transpose(1, 2)
        v = (h @ p[f"wv{i}"]).reshape(B, T_ALG, NH, HD).transpose(1, 2)
        if BR:
            q, k = q * kc + _rotate_half(q) * ks, k * kc + _rotate_half(k) * ks
            q = q * BRQC[br] + _rotate_half(q) * BRQS[br]                                            # the breath stamp (queries only)
        elif rc is None: q, k = _apply_rope(q, k, COS, SIN, T_ALG)
        else: q, k = q * rc[0] + _rotate_half(q) * rc[1], k * rc[0] + _rotate_half(k) * rc[1]   # THE T6 ADDRESS (no RoPE)
        sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(HD)) + am
        a = sc.clip(-1e4, 1e4).softmax(-1)
        x = x + (a @ v).transpose(1, 2).reshape(B, T_ALG, D) @ p[f"wo{i}"]
        if FF:
            h2 = _rms_norm(x, p[f"fn{i}"], EPS)
            x = x + ((h2 @ p[f"wg{i}"]).silu() * (h2 @ p[f"wu{i}"])) @ p[f"wd{i}"]
      xs.append(x)
    return xs if every else x


def addmask(tm):
    return Tensor(np.where(tm[:, None, None, :] > 0, 0.0, -1e4).astype(np.float32), dtype=dtypes.float)


def owned(p):
    """a loaded ckpt read through the student's own table: the table is already folded, so w_in is the identity"""
    if TABLE: p["w_in"] = Tensor.eye(D, dtype=dtypes.float).contiguous().realize()
    return p


if args.table:   # write the student's own table: EMB @ w_in folded, pruned to reader_vocab's kept rows (+ the piece map)
    assert os.path.exists(CKPT) and not os.path.exists(args.table), f"need {CKPT}; refusing to overwrite {args.table}"
    w_in = safe_load(CKPT)["w_in"].to(Device.DEFAULT).cast(dtypes.float).numpy(); z = np.load(os.environ.get("DR_VOCAB_K", ".cache/reader_vocab_k.npz"))
    rows = (EMB[z["keep"]].astype(np.float32) @ w_in).astype(np.float16)
    np.savez(args.table, V=len(EMB), keep=z["keep"], rows=rows, oov_ids=z["oov_ids"], piece_ptr=z["piece_ptr"], pieces=z["pieces"], ckpt=CKPT)
    print(f"[table] {CKPT} -> {args.table}: {rows.shape[0]} x {rows.shape[1]} owned rows ({rows.size / 1e6:.1f}M)", flush=True); sys.exit(0)

if args.export:
    assert args.src and args.out and os.path.exists(CKPT), "need --src, --out and a trained checkpoint"
    d_npy = STATES_NPY.format(split=args.out); assert not os.path.exists(d_npy), f"{d_npy} exists -- refusing to overwrite"
    samples, ids, mask, _ = tokenize(args.export)
    ztm = np.load(STATES_NPZ.format(split=args.src))["tokmask"]
    assert ztm.shape == mask.shape and (ztm.astype(np.float32) == mask).all(), "tokmask mismatch vs the source split"
    ld = safe_load(CKPT); p = owned({k: ld[k].to(Device.DEFAULT).cast(dtypes.float).contiguous().realize() for k in ld})
    VOCAB = os.environ.get("DR_VOCAB", "full"); assert not (TABLE and VOCAB != "full"), "DR_TABLE is already folded"
    if VOCAB != "full":   # THE SHRUNK READER (ledger 10-27): fold EMB @ w_in into one fp16 V x D table (exact up to fp16 storage);
        # "prune" keeps reader_vocab.py's K and gives every dropped token the MEAN of its fallback pieces' rows
        EMB = (EMB.astype(np.float32) @ p["w_in"].numpy()).astype(np.float16)
        if VOCAB == "prune":
            z = np.load(os.environ.get("DR_VOCAB_K", ".cache/reader_vocab_k.npz")); ptr, pc = z["piece_ptr"], z["pieces"]
            keep = np.zeros(len(EMB), bool); keep[z["keep"]] = True; assert keep[pc].all(), "a fallback piece was dropped"
            for j, i in enumerate(z["oov_ids"]):
                EMB[i] = EMB[pc[ptr[j]:ptr[j + 1]]].astype(np.float32).mean(0).astype(np.float16)
            print(f"[export] PRUNED: {int(keep.sum())} kept rows x {EMB.shape[1]}, {len(z['oov_ids'])} tokens composed from pieces", flush=True)
        else:
            print(f"[export] FOLDED: {EMB.shape[0]} x {EMB.shape[1]} fp16", flush=True)
        p["w_in"] = Tensor.eye(D, dtype=dtypes.float).contiguous().realize()
    shutil.copyfile(STATES_NPZ.format(split=args.src), STATES_NPZ.format(split=args.out))
    st = np.lib.format.open_memmap(d_npy, mode="w+", dtype=np.float16, shape=(len(ids), T_ALG, H_TRUNK))
    edb = np.load(DBNPY.format(split=os.environ.get("DR_DBSRC", args.src)), mmap_mode="r") if DBIN else None   # DR_DBSRC: other db rows
    if DBIN: assert edb.shape == (len(ids), T_ALG, DBP), f"db rows {edb.shape} vs {len(ids)} rows"
    esent = np.load(STATES_NPZ.format(split=args.src))["sent"] if T6 else None
    for s0 in range(0, len(ids), 8):
        sl = slice(s0, min(s0 + 8, len(ids)))
        rc = tuple(Tensor(a, dtype=dtypes.float) for a in t6_rot(esent[sl], mask[sl])) if T6 else None
        o = forward(p, Tensor(EMB[ids[sl]], dtype=dtypes.half), addmask(mask[sl]),
                    Tensor(np.asarray(edb[sl]), dtype=dtypes.half) if DBIN else None, rc).realize().numpy()
        assert np.isfinite(o).all()
        st[sl] = (o * mask[sl][:, :, None]).astype(np.float16)
    st.flush(); print(f"[export] {args.export} -> {args.out}: {len(ids)} rows (pads zero)", flush=True); sys.exit(0)

if args.gsm:   # LLAMA RETIRED (ledger 10-31): the student's word states for the GSM8K bench, laid out as gsm_trunk_states.py's
    import gsm_bench_data as GD, gsm_trunk_states as GTS
    from tokenizers import Tokenizer
    assert os.path.exists(CKPT) and not os.path.exists(args.gsm), f"need {CKPT}; refusing to overwrite {args.gsm}"
    from phase1_algebra_head import sent_indices   # (ledger 11-02 CY) the training split's own segmenter, so T6 forms export too
    Dt, Q = GTS.texts(); tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    uniq = list(dict.fromkeys(q for sp in Q.values() for q in sp)); enc = [tok.encode(q) for q in uniq]
    assert max(len(e.ids) for e in enc) <= T_ALG
    al = [GTS.align(q, e.offsets, len(e.ids)) for q, e in zip(uniq, enc)]; start = np.cumsum([0] + [len(a) for a in al])
    pos = {q: int(start[i]) for i, q in enumerate(uniq)}
    assert {sp: [pos[q] for q in Q[sp]] for sp in Q} == json.load(open(GTS.OUT_IDX)), "word index differs from the banked trunk's"
    ld = safe_load(CKPT); p = owned({k: ld[k].to(Device.DEFAULT).cast(dtypes.float).contiguous().realize() for k in ld})
    if DBIN:
        import reader_db_build as RDB
        _, _, dscale, store = RDB.load_store()
    st = np.lib.format.open_memmap(args.gsm, mode="w+", dtype=np.float16, shape=(int(start[-1]), H_TRUNK))
    for s0 in range(0, len(uniq), 8):
        sl = range(s0, min(s0 + 8, len(uniq))); ids = np.zeros((len(sl), T_ALG), np.int32); m = np.zeros((len(sl), T_ALG), np.float32)
        db = np.zeros((len(sl), T_ALG, DBP), np.float16)
        for r, i in enumerate(sl):
            n = len(enc[i].ids); ids[r, :n] = enc[i].ids; m[r, :n] = 1
            if DBIN: db[r, :n] = RDB.lookup(store, RDB.row_keys(ids[r], n)) * dscale     # as built: backoff, N >= 3
        xin = (Tensor(EMB[ids], dtype=dtypes.half), addmask(m), Tensor(db, dtype=dtypes.half) if DBIN else None)
        rc = tuple(Tensor(a, dtype=dtypes.float) for a in t6_rot(np.stack([sent_indices(uniq[i], enc[i].offsets, m[r]).astype(np.int8)
                                                                    for r, i in enumerate(sl)]), m)) if T6 else None
        if os.environ.get("DR_HEAD") == "nat":   # the GSM form's native head (trained on the banked trunk word states)
            o = (_rms_norm(body(p, *xin, rc), p["fnorm"], EPS) @ p["w_nat"] + p["b_nat"]).realize().numpy()
        else:
            o = forward(p, *xin, rc).realize().numpy()
        assert np.isfinite(o).all()
        for r, i in enumerate(sl): st[start[i]:start[i + 1]] = o[r, al[i]].astype(np.float16)
    st.flush(); print(f"[gsm] {len(uniq)} texts, {int(start[-1])} words -> {args.gsm} ({CKPT})", flush=True); sys.exit(0)

# ---- train ------------------------------------------------------------------------------------------------------
LK = os.environ.get("DR_LK") == "1"   # (ledger 11-02 LK0) read a trained MBR ckpt: the per-token best-breath ceiling
assert LK or not os.path.exists(CKPT), f"{CKPT} exists -- refusing to overwrite"
samples, ids, mask, _ = tokenize(TRAIN_JSONL)
teach = np.load(STATES_NPY.format(split=TRAIN_SPLIT), mmap_mode="r")
assert teach.shape == (len(ids), T_ALG, H_TRUNK)
tdb = np.load(DBNPY.format(split=TRAIN_SPLIT), mmap_mode="r") if DBIN else None   # TRAIN rows are leave-one-out
if DBIN: assert tdb.shape == (len(ids), T_ALG, DBP)
ztm = np.load(STATES_NPZ.format(split=TRAIN_SPLIT))["tokmask"]; assert (ztm.astype(np.float32) == mask).all(), "tokmask mismatch"
tsent = np.load(STATES_NPZ.format(split=TRAIN_SPLIT))["sent"] if T6 else None
if T6: assert DBIN and not GSM, "the T6 form is the DB student's (U2); GSM8K sentence indices are not built"
import hashlib
val = np.array([int(hashlib.sha1(s["text"].encode()).hexdigest(), 16) % 100 == 0 for s in samples])   # 1% by TEXT identity
tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0][:256]
smp = np.sort(np.random.default_rng(1).choice(tr_idx, 512, replace=False))
tsm = np.asarray(teach[smp]).astype(np.float32); msm = mask[smp] > 0
mean_out = tsm[msm].mean(0); var_t = float(tsm[msm].var(0).mean())
print(f"[distill] rows {len(ids)} train {len(tr_idx)} val {len(va_idx)} | teacher per-dim var {var_t:.4f} | D {D} layers {NL} FF {FF} "
      f"steps {STEPS} batch {BATCH} lr {LR} db {DBIN}", flush=True)
p = init_params(mean_out)
if WARM:
    ld = safe_load(WARM); assert set(ld) == set(p), f"warm keys differ: {set(ld) ^ set(p)}"
    p = {k: ld[k].to(Device.DEFAULT).cast(dtypes.float).contiguous().realize() for k in p}; print(f"[distill] WARM from {WARM}", flush=True)
if GSM:   # GSM8K TRAIN texts only (CF and TEST texts fenced); targets = the banked native word states (no Llama compute)
    import gsm_bench_data as GD, gsm_trunk_states as GTS
    from tokenizers import Tokenizer
    Dt, Qt = GTS.texts(); gidx = json.load(open(GTS.OUT_IDX)); gst = np.load(GTS.OUT_NPY, mmap_mode="r")
    fence = set(q for sp in ("CFNUM", "CFNAME", "TEST") for q in Qt[sp])
    gtx = {}
    for q, s0 in zip(Qt["TRAIN"], gidx["TRAIN"]):
        assert q not in fence, "a GSM8K TRAIN text is also a CF/TEST text"
        gtx[q] = s0
    gq = list(gtx); gtok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json"); genc = [gtok.encode(q) for q in gq]
    g_ids = np.zeros((len(gq), T_ALG), np.int32); g_mask = np.zeros((len(gq), T_ALG), np.float32); g_al = []
    for r, (q, e) in enumerate(zip(gq, genc)):
        g_ids[r, :len(e.ids)] = e.ids; g_mask[r, :len(e.ids)] = 1; g_al.append(np.array(GTS.align(q, e.offsets, len(e.ids))))
    g_s0 = np.array([gtx[q] for q in gq])
    gva = np.array([int(hashlib.sha1(q.encode()).hexdigest(), 16) % 50 == 0 for q in gq]); g_tr, g_va = np.nonzero(~gva)[0], np.nonzero(gva)[0][:64]
    gsmp = np.concatenate([np.asarray(gst[g_s0[r]:g_s0[r] + len(g_al[r])], np.float32) for r in g_tr[:400]])
    mean_nat = gsmp.mean(0); INV_VARN = 1.0 / float(gsmp.var(0).mean())
    if DBIN:
        import reader_db_build as RDB
        _, _, g_dscale, g_store = RDB.load_store()
    sd_out = p["w_out"].numpy()
    p["w_nat"] = Tensor(sd_out, dtype=dtypes.float).contiguous().realize(); p["b_nat"] = Tensor(mean_nat, dtype=dtypes.float).contiguous().realize()
    print(f"[distill] GSM: {len(gq)} TRAIN texts (train {len(g_tr)} val {len(g_va)}), fenced {len(fence)} CF/TEST texts; native var "
          f"{1 / INV_VARN:.4f}", flush=True)

    def gsm_rows(rows):
        """-> (EMB rows, mask, word mask, dense native targets, db rows) for GSM text rows"""
        tm = np.zeros((len(rows), T_ALG), np.float32); tg = np.zeros((len(rows), T_ALG, H_TRUNK), np.float16)
        db = np.zeros((len(rows), T_ALG, DBP), np.float16)
        for j, r in enumerate(rows):
            al = g_al[r]; tm[j, al] = 1; tg[j, al] = gst[g_s0[r]:g_s0[r] + len(al)]
            if DBIN:
                n = int(g_mask[r].sum()); db[j, :n] = RDB.lookup(g_store, RDB.row_keys(g_ids[r], n)) * g_dscale
        return (EMB[g_ids[rows]], g_mask[rows], tm, tg) + ((db,) if DBIN else ())
params = list(p.values())
for t in params:
    t.requires_grad = True
n_par = sum(int(np.prod(t.shape)) for t in params); print(f"[distill] trainable params {n_par:,} (+ frozen embedding)", flush=True)
opt = AdamW(params, lr=LR, weight_decay=0.01)
INV_VAR = 1.0 / var_t


def loss_fn(out, tgt, tm, inv=None):
    # .contiguous() = schedule boundaries: without them tinygrad fuses the w_out matmul's reduction under the per-row
    # mean and recomputes per-row/per-batch reductions inside the backward's elementwise kernels (13.7 s/step -> 0.28)
    # (11-02) per-row terms + weights as buffers too: the grad reaching out stays elementwise (MBR: 1.55 -> 0.25 s/step)
    out = out.contiguous(); tgt = tgt.cast(dtypes.float); wr = (tm / tm.sum()).contiguous()
    se = ((out - tgt) ** 2).mean(-1).contiguous()
    cos = ((out * tgt).sum(-1).contiguous() / ((out * out).sum(-1).sqrt().contiguous() * (tgt * tgt).sum(-1).sqrt().contiguous() + 1e-6)).contiguous()
    mse = (se * wr).sum() * (INV_VAR if inv is None else inv)
    return mse + ((1 - cos) * wr).sum(), mse


BRW = [(b + 1) / sum(range(1, BR + 1)) for b in range(BR)]   # (MBR) deep supervision: every breath distilled, later breaths weigh more
BRWT = Tensor(np.asarray(BRW or [0.0], np.float32), dtype=dtypes.float).contiguous().realize()


def loss_br(out, tgt, tm):
    """loss_fn per breath on out (BR, B, T, H), same boundaries; -> (sum_b BRW[b] * loss_b, mse of the last breath)"""
    # per-row terms and weights are buffers, so the grad reaching out is elementwise (else w_out/b_out's grad recomputes B*T sums per row)
    out = out.contiguous(); tgt = tgt.cast(dtypes.float); wr = (tm / tm.sum()).contiguous()
    se = ((out - tgt) ** 2).mean(-1).contiguous()
    cos = ((out * tgt).sum(-1).contiguous() / ((out * out).sum(-1).sqrt().contiguous() * (tgt * tgt).sum(-1).sqrt().contiguous() + 1e-6)).contiguous()
    return ((se * INV_VAR + 1 - cos) * (wr * BRWT.reshape(BR, 1, 1)).contiguous()).sum(), (se[BR - 1] * wr).sum() * INV_VAR


def _train(x0, am, tm, tgt, db, cs=None, sn=None):
    with Tensor.train():
        if BR:   # ONE head over the breaths stacked on the batch axis: four separate heads fused w_out's grad into a recomputing kernel (1.55 -> s/step)
            out = head(p, Tensor.cat(*body(p, x0, am, db, (cs, sn), every=True), dim=0)).reshape(BR, *tgt.shape)
            loss, mse = loss_br(out, tgt, tm)
        else:
            out = forward(p, x0, am, db, None if cs is None else (cs, sn)); loss, mse = loss_fn(out, tgt, tm)
        opt.zero_grad(); loss.backward()
        ok = loss.isfinite().cast(dtypes.float).contiguous()                # single-kernel guard (a boundary, or the loss is recomputed per grad element)
        for t in params:
            t.grad = t.grad * ok
        opt.step()
        return loss.realize(), mse.realize()


def _eval(x0, am, tm, tgt, db, cs=None, sn=None):
    out = forward(p, x0, am, db, None if cs is None else (cs, sn)); loss, mse = loss_fn(out, tgt, tm)
    return loss.realize(), mse.realize()


def nat(x0, am, db):
    return _rms_norm(body(p, x0, am, db), p["fnorm"], EPS) @ p["w_nat"] + p["b_nat"]


def _train_g(x0, am, tm, tgt, db, gx0, gam, gtm, gtgt, gdb):
    with Tensor.train():
        loss, mse = loss_fn(forward(p, x0, am, db), tgt, tm); gloss, gmse = loss_fn(nat(gx0, gam, gdb), gtgt, gtm, INV_VARN)
        tot = loss + gloss
        opt.zero_grad(); tot.backward()
        ok = tot.isfinite().cast(dtypes.float).contiguous()
        for t in params:
            t.grad = t.grad * ok
        opt.step()
        return loss.realize(), mse.realize(), gloss.realize(), gmse.realize()


if GSM:
    assert DBIN, "the GSM form continues the DB student"
    train_g = TinyJit(lambda *a: _train_g(*a))
    eval_g = TinyJit(lambda gx0, gam, gtm, gtgt, gdb: [t.realize() for t in loss_fn(nat(gx0, gam, gdb), gtgt, gtm, INV_VARN)])
if T6:
    train_step = TinyJit(lambda x0, am, tm, tgt, db, cs, sn: _train(x0, am, tm, tgt, db, cs, sn))
    eval_step = TinyJit(lambda x0, am, tm, tgt, db, cs, sn: _eval(x0, am, tm, tgt, db, cs, sn))
elif DBIN:
    train_step = TinyJit(lambda x0, am, tm, tgt, db: _train(x0, am, tm, tgt, db))
    eval_step = TinyJit(lambda x0, am, tm, tgt, db: _eval(x0, am, tm, tgt, db))
else:
    train_step = TinyJit(lambda x0, am, tm, tgt: _train(x0, am, tm, tgt, None))
    eval_step = TinyJit(lambda x0, am, tm, tgt: _eval(x0, am, tm, tgt, None))


def batch_of(idx):
    idx = np.sort(idx)
    return (Tensor(EMB[ids[idx]], dtype=dtypes.half), addmask(mask[idx]), Tensor(mask[idx], dtype=dtypes.float),
            Tensor(np.asarray(teach[idx]), dtype=dtypes.half)) + ((Tensor(np.asarray(tdb[idx]), dtype=dtypes.half),) if DBIN else ()) + (
            tuple(Tensor(a, dtype=dtypes.float) for a in t6_rot(tsent[idx], mask[idx])) if T6 else ())


Q = queue.Queue(maxsize=6)


def loader():
    rng = np.random.default_rng(2 + RSEED)
    while True:
        idx = rng.choice(tr_idx, BATCH, replace=False); idx = np.sort(idx)
        Q.put((EMB[ids[idx]], mask[idx], np.asarray(teach[idx])) + ((np.asarray(tdb[idx]),) if DBIN else ())
              + (gsm_rows(np.sort(rng.choice(g_tr, BATCH, replace=False))) if GSM else ()) + (t6_rot(tsent[idx], mask[idx]) if T6 else ()))


if LK:   # LK0: per-token teacher mse at each breath on the val rows; the oracle takes each token's best breath
    assert BR and os.path.exists(CKPT); ld = safe_load(CKPT); assert set(ld) == set(p), f"ckpt keys differ: {set(ld) ^ set(p)}"
    for k in p: p[k].assign(ld[k].to(Device.DEFAULT).cast(dtypes.float)).realize()
    se, wt = [], []
    for b in [va_idx[i:i + BATCH] for i in range(0, len(va_idx) - BATCH + 1, BATCH)]:
        x0, am, tm, tg, db, cs, sn = batch_of(b); tgf = tg.cast(dtypes.float)
        se.append(np.stack([((head(p, xb) - tgf) ** 2).mean(-1).numpy() for xb in body(p, x0, am, db, (cs, sn), every=True)]))   # (BR, B, T)
        wt.append(tm.numpy())
    se, wt = np.concatenate(se, 1) * INV_VAR, np.concatenate(wt, 0); m = wt > 0
    per = [float(se[k][m].mean()) for k in range(BR)]; orc = float(se.min(0)[m].mean()); best = np.bincount(se.argmin(0)[m], minlength=BR)
    print(f"[lk0] {CKPT}: per-breath val mse {' / '.join(f'{v:.4f}' for v in per)} | oracle {orc:.4f} | breath4 - oracle {per[-1] - orc:+.4f} "
          f"(bar >= +0.010) -> {'PASS' if per[-1] - orc >= 0.010 else 'FAIL'} | best-breath share {best.tolist()}", flush=True); sys.exit(0)
threading.Thread(target=loader, daemon=True).start()
VB = [va_idx[i:i + BATCH] for i in range(0, len(va_idx) - BATCH + 1, BATCH)]
t0 = time.time(); hist = []
for step in range(STEPS):
    ta = time.time(); bi, bm, bt, *bd = Q.get(); tb = time.time()
    if GSM: bd, gb = bd[:1], bd[1:]
    if T6: bd, rcs = bd[:1], bd[1:]
    xs = [Tensor(bi, dtype=dtypes.half), addmask(bm), Tensor(bm, dtype=dtypes.float), Tensor(bt, dtype=dtypes.half)] + [Tensor(d, dtype=dtypes.half) for d in bd]
    if T6: xs += [Tensor(a, dtype=dtypes.float) for a in rcs]
    if GSM:
        gx, gm, gtm, gtg, gdb = gb
        xs += [Tensor(gx, dtype=dtypes.half), addmask(gm), Tensor(gtm, dtype=dtypes.float), Tensor(gtg, dtype=dtypes.half), Tensor(gdb, dtype=dtypes.half)]
    xs = [x.contiguous().realize() for x in xs]; tc = time.time()
    if GSM: loss, mse, gloss, gmse = train_g(*xs)
    else: loss, mse = train_step(*xs)
    if PROF:
        loss.numpy(); print(f"[prof] step {step}: queue wait {tb - ta:.3f}s, inputs {tc - tb:.3f}s, train_step {time.time() - tc:.3f}s", flush=True)
    if step % 200 == 0 or step == STEPS - 1:
        lv = float(loss.numpy()); assert math.isfinite(lv), f"non-finite loss at step {step}"
        msg = f"[distill] step {step} (queue {Q.qsize()}) loss {lv:.4f} mse {float(mse.numpy()):.4f}" + (
            f" gsm mse {float(gmse.numpy()):.4f}" if GSM else "") + f" | {(time.time() - t0) / (step + 1):.3f}s/step"
        if step % 2000 == 0 or step == STEPS - 1:
            vl = [tuple(float(t.numpy()) for t in eval_step(*batch_of(b))) for b in VB]     # floats per batch: the JIT reuses its outputs
            vloss = float(np.mean([a for a, _ in vl])); vmse = float(np.mean([b for _, b in vl])); v32 = vl[-1][1]
            msg += f" | VAL loss {vloss:.4f} mse {vmse:.4f} (VAL32 {v32:.4f})"
            hist.append(dict(step=step, val_loss=vloss, val_mse=vmse, val32_mse=v32))
            if GSM:
                gv = [tuple(float(t.numpy()) for t in eval_g(*[Tensor(a, dtype=dtypes.half) if k in (0, 3, 4) else (addmask(a) if k == 1 else Tensor(a, dtype=dtypes.float))
                              for k, a in enumerate(gsm_rows(g_va[i:i + BATCH]))])) for i in range(0, len(g_va) - BATCH + 1, BATCH)]
                gvm = float(np.mean([b for _, b in gv])); msg += f" | GSM VAL native mse {gvm:.4f}"; hist[-1]["gsm_val_mse"] = gvm
        print(msg, flush=True)
    if SAVE_AT and step + 1 == SAVE_AT:
        safe_save({k: v.realize() for k, v in p.items()}, CKPT.replace(".safetensors", f"_s{SAVE_AT}.safetensors"))
        print(f"[distill] snapshot at step {SAVE_AT}", flush=True)
brv, dlt, census = [], [], []
if BR:   # (MBR, bar MB2) the per-breath val mse: does each breath sharpen the estimate?
    for b in VB:
        x0, am, tm, tg, db, cs, sn = batch_of(b)
        xbs = body(p, x0, am, db, (cs, sn), every=True); brv.append([float(loss_fn(head(p, xb), tg, tm)[1].numpy()) for xb in xbs])
        xn, mm = [xb.numpy() for xb in xbs], tm.numpy() > 0     # THE DELTA CENSUS (ledger 11-03, Bryce's NNUE point; a read, never a loss)
        dlt.append(np.stack([np.linalg.norm(xn[k] - xn[k - 1], axis=-1)[mm] / (np.linalg.norm(xn[k], axis=-1)[mm] + 1e-6) for k in range(1, BR)]))
    brv = np.mean(np.array(brv), 0).tolist(); print(f"[distill] BREATH val mse {' > '.join(f'{v:.4f}' for v in brv)}", flush=True)
    dlt = np.concatenate(dlt, 1); census = [dict(breath=k + 2, q10=float(np.quantile(d, .1)), q50=float(np.median(d)), q90=float(np.quantile(d, .9)),
                                                  q99=float(np.quantile(d, .99)), below01=float((d < 0.01).mean())) for k, d in enumerate(dlt)]
    for c in census: print(f"[distill] DELTA breath {c['breath']}: rel change q10 {c['q10']:.4f} q50 {c['q50']:.4f} q90 {c['q90']:.4f} q99 {c['q99']:.4f} | share < 0.01 {c['below01']:.3f}", flush=True)
safe_save({k: v.realize() for k, v in p.items()}, CKPT)
json.dump(dict(tag=TAG, D=D, layers=NL, breaths=BR, breath_val_mse=brv, delta_census=census, ff=FF, db=DBIN, t6=T6, gsm=GSM, warm=WARM, steps=STEPS, batch=BATCH, lr=LR, params=n_par, teacher=TRAIN_SPLIT,
               var_t=var_t, hist=hist, seconds=time.time() - t0), open(META, "w"), indent=1)
print(f"[distill] saved {CKPT} in {time.time() - t0:.0f}s", flush=True)
