"""gsm_btc.py -- THE GSM8K TIER ON THE OFFICIAL BREATHING TRANSFORMER (ledger 2026-11-07; Bryce's word for the queue).

The BT is `mycelium/breathing_transformer.BreathingTransformer`; this script is its GSM8K PORT (data -> streams/stamps/masks, outputs ->
the bench's pointer layout) and its TRAINER (tinygrad, TinyJit per question-length bucket). gsm_breath_reader stays the source of truth
for data, notebook, executor, gold and scoring: the trained BT is scored by BR.score through a facade (one scorer for every arm).

ONE BREATH of this tier = two calls of the same BT (the tier hierarchy inside a breath: decide the type, then address under it):
  A: slot self-attention + slot -> token cross-attention (the notebook's write-back re-enters every breath; consumed marks as marks)
     -> the slot silhouette z -> the step type (z against the Welford atoms, the Tree's codebook; temperature = the breath's)
  B: the pointer roads under the decided type (gold in training, the snapped prediction at inference): condition(h, type atom x p(type)),
     one full-width head per (operand a/b) x (kind token / earlier slot / constant); the type's operand kinds weigh the blocks.
The token stream is read ONCE per problem (token self-attention = the reader, breath 0's road). Composite keys: tokens carry T9 (offset
on the six gears, position on the slow gears) + token_id (a word hash; names = the problem-local entity id, so CFNAME leaves it
unchanged); slots carry slot_id (a clock over the slot index), the T6 stamp (the breath) and the tree level. Name tokens enter as their
fresh per-problem entity phases (the reader's ID chassis), never their surface states.
BTC2 (ledger 11-07, the audit's #4 + #5; both default ON, =0 restores the BTC form):
  BTC_SELFCOND=1  the radius a committed slot enters with = the model's OWN detached confidence (a first no-grad pass with the committed
                  slots' atoms hidden at radius 0; the scorer's formula p(type) x max p(a) x max p(b), p(type) alone for STOP), so the
                  confidence dial trains on the radii the scorer feeds it (BTC trained at radius 1 only).
  BTC_STAMP=1     a committed slot's T6 stamp = the breath it was committed in (the Notebook.tick contract); open slots carry the current
                  breath. Training draws a commit order per problem and stamps it on the scorer's schedule; the scorer's facade tracks it.
Modes (BTC_MODE): smoke (BTC_STEPS steps, s/step) | train (BT_STEPS, scored every 1000 on ORIG/TEST[:600], FINAL on every split,
.cache/gsmb_btc_{TAG}.json; BT_SAVE=1 -> .safetensors + atoms .npy). Env: BT_SEED, BT_TAG, BT_BATCH, BT_LR; BTC_D / BTC_NH / BTC_DB /
BTC_BLOCKS (defaults 512 / 8 / 64 / 2);
BTC_WD (AdamW weight decay, 0.01) / BTC_DROP (BTConfig.dropout, 0): the regularization knobs; BTC_TOKID=0 drops the word-hash key
from token_self; BTC_WAIST = BTConfig.waist_at (every / last / end: where the queries squeeze);
the funnel levers BTC_TIE / BTC_TURN (learned|clock) / BTC_MQA / BTC_PTRSHARE, and BTC_FUNNEL=1 = all of them + 4 blocks + 'last' (ledger 11-08).
"""
import json, math, os, sys, time, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BT_ROLE", "1")
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save

from mycelium import breathing_transformer as BTM
from mycelium import bt_organs as O

K, NC, NT, STOP, NCYC, PB = BR.K, BR.NC, BR.NT, BR.STOP, BR.NCYC, BR.PB
KINDS = BR.KINDS
_KM = np.zeros((NT, 2, 3), np.float32)
for _j, (_, _a, _b) in enumerate(BR.TYPES):
    _KM[_j, 0, KINDS.index(_a)] = 1; _KM[_j, 1, KINDS.index(_b)] = 1
_KM[STOP] = 1
FUNNEL = os.environ.get("BTC_FUNNEL", "0") == "1"   # THE FUNNEL TIER (Bryce 11-08): 4 tied layers, one squeeze (the last), clock turns,
_FD = dict(BTC_BLOCKS="4", BTC_WAIST="last", BTC_TIE="1", BTC_TURN="clock", BTC_MQA="1", BTC_PTRSHARE="1") if FUNNEL else {}   # shared KV + pointer
_env = lambda k, v: os.environ.get(k, _FD.get(k, v))
D, NH, DB, NBLK = (int(_env(k, v)) for k, v in (("BTC_D", "512"), ("BTC_NH", "8"), ("BTC_DB", "64"), ("BTC_BLOCKS", "2")))
PTR = [f"p{r}{kd}" for r in "ab" for kd in KINDS]                   # pa_t pa_s pa_c pb_t pb_s pb_c
BUCKET = 32
SELFCOND, STAMP = os.environ.get("BTC_SELFCOND", "1") == "1", os.environ.get("BTC_STAMP", "1") == "1"
WD, DROP = float(os.environ.get("BTC_WD", "0.01")), float(os.environ.get("BTC_DROP", "0"))   # regularization (BTC2 = 0.01 / 0)
WAIST = _env("BTC_WAIST", "every")      # BTConfig.waist_at: every | last | end (ledger 11-08)
TIE, TURN, MQA, PTRSHARE = _env("BTC_TIE", "0") == "1", _env("BTC_TURN", "learned"), _env("BTC_MQA", "0") == "1", _env("BTC_PTRSHARE", "0") == "1"
SLOTSELF = os.environ.get("BTC_SLOTSELF", "1") == "1"   # 0 = no slot-self road: slots see other slots only via the slot pointers (ledger 11-08 NOSS)
TOKID = os.environ.get("BTC_TOKID", "1") == "1"   # token_self matches on token_id (a word-hash key); 0 = t9 only (ledger 11-08)
# THE FIT-SPEED knobs (ledger 11-08 audit: BR's forms where BTC differs; defaults = BTC2, bit-identical)
TAU1 = os.environ.get("BTC_TAU", "")                 # a constant temperature (e.g. 1: no logit / tau anywhere, as BR) | "" = 2.0 -> 0.5
PSCALE = os.environ.get("BTC_PSCALE", "0") == "1"    # pointer q . k unscaled (BR's lt = q . kt; BTC2: / sqrt(512))
BIND = int(os.environ.get("BTC_BIND", "0") or 0)    # (ledger 11-08) multiplicative binding: 1 = the residual roads' content q / k rotate by
                                                     # their matched key angles (BR's rotated attention); 2 = + the pointer roads
BSKIP = ("pos",) if os.environ.get("BTC_BINDREL", "0") == "1" else ()   # (ledger 11-09) bind on the T9 OFFSET gears only (no absolute pos)
SLOTK = os.environ.get("BTC_SLOTK", "0") == "1"      # each slot's learned identity + level embedding in its CONTENT (BR's slotk + lev)
SAB = float(os.environ.get("BTC_SAB", "0") or 0)     # (ledger 11-09) THE SABOTAGE TRICK (NCM's coarse-code corruption): in training, each
                                                     # committed step is corrupted with a per-problem rate ~ U(0, 2 SAB) (mean SAB): its type
                                                     # -> a random other step type, its op with it (gold operands kept), so the notebook the
                                                     # breath reads carries wrong commits as the scorer's does; loss on the open slots only
MACRO = os.environ.get("BTC_MACRO", "0") == "1"   # (ledger 11-09 MACRO1, Bryce's word) THE MACRO PLAN, coarse to fine on designed codes:
                                                     # breaths 0-1 commit every slot's OP (the plan; p(op) = the type marginal), breaths 2-5
                                                     # commit the TYPE (the kind residual) + pointers; a committed op enters the slot as its
                                                     # op atom AND as a phase (the 'op' clock factor, matched + bound on slot_self = message
                                                     # passing over the plan); training draws the phase per problem by its breath
SABOP = float(os.environ.get("BTC_SABOP", "0") or 0)  # (MACRO) the plan's sabotage: each committed op -> a random other op at rate ~ U(0, 2 SABOP)
NPLAN = 2 if MACRO else 0
ZOOMN = int(os.environ.get("BTC_ZOOM", "0") or 0)    # 2 (or 1) = log p(type) = log p(op) + log p(type | op), op atoms (BR's BT_ZOOM);
ZOOM, ZOOM3 = ZOOMN >= 1, ZOOMN >= 3                 # 3 = op -> op x kind_a -> type (the chain rule, ledger 11-08)
NOP, PARENT, LEVMAX = BR.NOP, BR.PARENT, BR.LEVMAX
MID = np.array([BR.GD.OPS.index(o) * 3 + KINDS.index(a) for o, a, _ in BR.TYPES] + [len(BR.GD.OPS) * 3])  # (ZOOM3) type -> (op, kind a)
NMID = len(BR.GD.OPS) * 3 + 1 if ZOOM3 else 0                                                            # + stop
MIDP = np.array([m // 3 for m in range(len(BR.GD.OPS) * 3)] + [NOP - 1])                                # (op, kind a) -> op
assert not (MACRO and ZOOM3), "MACRO: op atoms ride the ZOOM rows (not ZOOM3)"
NA = NT + (NMID if ZOOM3 else 0) + (NOP if ZOOM or MACRO else 0)   # the atoms' rows: fine types | (ZOOM3) op x kind a | (ZOOM / MACRO) ops
BR.MACRO_PLAN = NPLAN
_MPAR = Tensor(np.eye(NOP, dtype=np.float32)[PARENT].T.copy()).realize() if MACRO else None   # (MACRO) (NOP, NT) op <- its types
OP_SPEEDS = tuple(360.0 * j / (NOP + 1) for j in (1, 2, 3))   # (MACRO) the op clock: ids 0..NOP (NOP = no plan yet) on three harmonics
_PAR = np.zeros((NOP, NT), np.float32); _PAR[PARENT, np.arange(NT)] = 1
if ZOOM3:
    _PAR = np.zeros((NMID, NT), np.float32); _PAR[MID, np.arange(NT)] = 1
    _PAR2 = np.zeros((NOP, NMID), np.float32); _PAR2[MIDP, np.arange(NMID)] = 1
_PART = Tensor(_PAR).realize() if ZOOM else None      # (ZOOM) the parent one-hot, realized once (a constant inside the JIT)
_PART2 = Tensor(_PAR2).realize() if ZOOM3 else None
_SCHED = np.array([min(c for c in range(NCYC) if math.ceil(K * (c + 1) / NCYC) > j) for j in range(K)])   # the scorer: rank j commits at


def config(d_in, d_aux):
    f = BTM.default_factors() + ((BTM.KeyFactor("op", "clock", gears=(("op", OP_SPEEDS),)),) if MACRO else ())
    roads = (BTM.RoadSpec("token_self", "token", "token", NH, ((("t9", "token_id") if TOKID else ("t9",)),) * NH, breaths=(0,), bind=BIND >= 1, bind_skip=BSKIP),
             *((BTM.RoadSpec("slot_self", "slot", "slot", NH, (("slot_id", "t6", "level") + (("op",) if MACRO else ()),) * NH, bind=BIND >= 1, bind_skip=BSKIP),) if SLOTSELF else ()),
             BTM.RoadSpec("cross", "slot", "token", NH, (("t9",),) * NH, bind=BIND >= 1, bind_skip=BSKIP)) + tuple(
        BTM.RoadSpec(n, "slot", {"t": "token", "s": "slot", "c": "const"}[n[2]], 1,
                     ((("t9",) if n[2] == "t" else ("slot_id", "level") if n[2] == "s" else ()),), residual=False, d_head=D,
                     share_q=n[1] if PTRSHARE else None, share_k=n[2] if PTRSHARE else None, q_scale=1.0 if PSCALE else None, bind=BIND >= 2, bind_skip=BSKIP) for n in PTR)
    return BTM.BTConfig(tier="gsm-step", d_in=d_in, d_aux=d_aux, d_model=D, d_waist=DB, d_head=D // NH, d_cert=3 + BR.NUF + ((K + LEVMAX + 1) if SLOTK else 0), n_atoms=NT, n_coarse=(NMID if ZOOM3 else NOP) if ZOOM else 0, n_coarse2=NOP if ZOOM3 else 0,
                        **(dict(tau_kind="constant", tau_first=float(TAU1)) if TAU1 else {}),
                        n_breaths=NCYC, blocks_per_breath=NBLK, factors=f, roads=roads, d_mark=1, n_const=NC,
                        dropout=DROP, waist_at=WAIST, tie_layers=TIE, query_turn=TURN, shared_kv=MQA)


# ---- the port: one BR batch -> host arrays at a bucketed question length -----------------------------------------------------------
def word_id(w, isn, nid):
    return 1_000_000_007 + int(nid) if isn else zlib.crc32(str(w).lower().encode())


def port_batch(t, E, its, Lb):
    """BR.batch's torch dict -> numpy host arrays for the BT at question length Lb (>= Lq)"""
    g = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in t.items()}
    B, L = g["pad"].shape; Lq = L - K; assert Lb >= Lq
    th = np.take_along_axis(g["theta"], np.clip(g["nid"], 0, None)[..., None].repeat(PB, -1), 1)      # (B, L, PB) entity phases
    isn = g["isn"][:, :Lq]
    x = np.zeros((B, Lb, g["x"].shape[-1]), np.float32); x[:, :Lq] = np.where(isn[..., None], 0.0, g["x"][:, :Lq])   # names: no surface
    aux = np.concatenate([np.cos(th[:, :Lq]) * isn[..., None], np.sin(th[:, :Lq]) * isn[..., None], isn[..., None], g["isnum"][:, :Lq, None],
                          g["g"][:, :Lq], g["role"][:, :Lq]], -1).astype(np.float32)
    A = np.zeros((B, Lb, aux.shape[-1]), np.float32); A[:, :Lq] = aux
    pad = np.ones((B, Lb), bool); pad[:, :Lq] = g["pad"][:, :Lq]
    isnum = np.zeros((B, Lb), bool); isnum[:, :Lq] = g["isnum"][:, :Lq]
    sid = np.zeros((B, Lb), np.int64); sid[:, :Lq] = g["sid"][:, :Lq]
    tid = np.zeros((B, Lb), np.int64)
    for b, it in enumerate(its):
        for i, w in enumerate(it["toks"]): tid[b, i] = word_id(w, it["isn"][i], g["nid"][b, i])
    tf = {**O.T9Clock().fields(sid), "token_id": tid}
    tf["off"] = np.where(pad, 0, tf["off"]); tf["pos"] = np.where(pad, 0, tf["pos"])
    tgt = {}
    for r, V in (("a", "va"), ("b", "vb")):
        v = g[V]; tk = np.zeros((B, K, Lb), bool); tk[..., :Lq] = v[..., :Lq]
        tgt[r] = (tk, v[..., L:L + K], v[..., L + K:L + K + NC])
    return dict(x=x, aux=A, pad=pad, isnum=isnum, tf=tf, tgt=tgt, Lq=Lq, L=L, B=B)


def host_io(bt, P, nb, tick):
    """per breath: notebook arrays (typ, r, cf, lev, cons at the batch's L) -> slot angles/presence, entry arrays, marks"""
    typ, r, cf, lev, cons = nb[:5]; B = typ.shape[0]; Lq = P["Lq"]; Lb = P["x"].shape[1]
    tick = np.asarray(tick).reshape(B, -1)                                                     # (B,) the breath, or (B, K) per-slot stamps
    sf = {"slot": np.broadcast_to(np.arange(K), (B, K)).copy(), "tick": np.broadcast_to(tick, (B, K)).copy(), "level": lev}
    com, atom, rad = typ >= 0, np.clip(typ, 0, None), r.astype(np.float32)
    if MACRO:   # an op-only slot enters as its op atom (row NT + op) at its op radius; the op field: committed op, else NOP (no plan yet)
        opk, opid, orad = nb[5:8]; po = opk & ~com
        atom = np.where(po, NT + opid, atom); rad = np.where(po, orad, rad).astype(np.float32); sf["op"] = np.where(opk | com, opid, NOP)
        com = com | opk
    sa, sp = bt.layout.angles(sf, (B, K))
    mt = np.zeros((B, Lb, 1), np.float32); mt[:, :Lq, 0] = cons[:, :Lq]
    ms = cons[:, Lq:Lq + K, None].astype(np.float32)
    if SLOTK:   # (FIT-SPEED) the slot's id and level one-hots ride the cert's Linear = BR's slotk + lev embeddings, on every slot
        cf = np.concatenate([cf, np.broadcast_to(np.eye(K, dtype=np.float32), (B, K, K)), np.eye(LEVMAX + 1, dtype=np.float32)[lev]], -1)
    return dict(sa=sa, sp=sp, mt=mt, ms=ms, com=com, atom=atom, rad=rad, cert=cf.astype(np.float32))


def valid_masks(P, B):
    Lb = P["x"].shape[1]; tok = ~P["pad"]
    return dict(token_self=np.broadcast_to(tok[:, None, :], (B, Lb, Lb)).copy(), slot_self=np.ones((B, K, K), bool),
                cross=np.broadcast_to(tok[:, None, :], (B, K, Lb)).copy(),
                pt=np.broadcast_to((tok & P["isnum"])[:, None, :], (B, K, Lb)).copy(),
                ps=np.broadcast_to(np.tril(np.ones((K, K), bool), -1)[None], (B, K, K)).copy(), pc=np.ones((B, K, NC), bool))


# ---- the network forward (pure tinygrad; shared by the trainer's JIT and the scorer) ------------------------------------------------
def forward(bt, atoms, x, aux, ta, tp, sa, sp, mt, ms, com, atom, rad, cert, vts, vss, vcr, vpt, vps, vpc, tau, cond=None):
    """-> (type logp (B, K, NT), z (B, K, DB), {road: logp (B, K, Lk)}, cond used, p(cond))"""
    coarse = atoms[NT:NT + NMID] if ZOOM3 else atoms[NT:] if ZOOM else None                         # (ZOOM) the coarse atoms ride the
    coarse2 = atoms[NT + NMID:] if ZOOM3 else None; full = atoms; atoms = atoms[:NT] if ZOOM or MACRO else atoms   # same buffer (ZOOM3: mid, then ops)
    ht = bt.enter_tokens(x, aux)
    io = BTM.BreathIO(h={"token": ht}, angles={"token": ta}, presence={"token": tp}, valid={"token_self": vts})
    ht = bt.breath(io, 0, roads=("token_self",))["h"]["token"]                                    # THE READER (once per problem)
    hs = bt.enter_slots(com, cert, atom=atom, radius=rad, atoms=full if MACRO else atoms)   # (MACRO) op atoms re-enter too
    io = BTM.BreathIO(h={"token": ht, "slot": hs}, angles={"token": ta, "slot": sa}, presence={"token": tp, "slot": sp},
                      valid={"slot_self": vss, "cross": vcr}, marks={"token": mt, "slot": ms})
    oa = bt.breath(io, None, roads=("slot_self", "cross"), tau=tau)                               # call A
    lt = (bt.silhouette_zoom_logp(oa["z"]["slot"], atoms, coarse, _PART, tau, coarse2, _PART2) if ZOOM
          else bt.silhouette_logp(oa["z"]["slot"], atoms, tau))
    c = lt.argmax(-1) if cond is None else cond
    pc_ = (lt.exp() * c.one_hot(NT).cast(dtypes.float)).sum(-1)                                  # the radius: p(the decided type)
    hs2 = bt.condition(oa["h"]["slot"], c, pc_, atoms)
    vmap = {"pat": vpt, "pbt": vpt, "pas": vps, "pbs": vps, "pac": vpc, "pbc": vpc}
    io = BTM.BreathIO(h={"token": oa["h"]["token"], "slot": hs2}, angles={"token": ta, "slot": sa}, presence={"token": tp, "slot": sp},
                      valid=vmap)
    ob = bt.breath(io, None, roads=tuple(PTR), tau=tau)                                           # call B
    return lt, oa["z"]["slot"], {n: ob["logp"][n][:, 0] for n in PTR}, c, pc_


def confidence(bt, atoms, lt, lp, c, pc_, vps):
    """the scorer's radius of each proposal: p(type) x max p(a) x max p(b) (STOP: p(type)); kinds from the DECIDED type (B, K)"""
    kw = (c.one_hot(NT).cast(dtypes.float) @ Tensor(_KM.reshape(NT, 6))).reshape(*c.shape, 2, 3)
    J = joint(lp, kw, [None, vps.any(-1), None])
    ca, cb = (Tensor.cat(*J[r], dim=-1).softmax(-1).max(-1) for r in "ab")
    return (c == STOP).where(pc_, pc_ * ca * cb)


def sabotage(rng, progs, gt, committed):
    """(SAB) the notebook's corrupted copy: each committed non-STOP step, with its problem's rate ~ U(0, 2 SAB) (capped at 1), becomes a
    random OTHER step type (never STOP) and carries that type's op over its gold operands -> (progs, types); the gold stays untouched"""
    B = gt.shape[0]; rate = np.minimum(rng.random(B) * 2 * SAB, 1.0)
    hit = committed & (gt != STOP) & (rng.random((B, K)) < rate[:, None])
    typ = gt.copy(); out = [list(p) for p in progs]
    for b, k in zip(*np.nonzero(hit)):
        j = int(rng.integers(0, STOP - 1)); j += j >= gt[b, k]                                   # uniform over the STOP - 1 others
        typ[b, k] = j; out[b][k] = (BR.TYPES[j][0], *progs[b][k][1:])
    return out, typ


def macro_stamps(rng, oponly, tcom, cyc):
    """(MACRO) (B, K) T6 stamps on the plan schedule: an op-only slot = a plan breath (0..NPLAN-1) before the current, a type-committed
    slot = a type breath (NPLAN + its rank on the type schedule) before the current, an open slot = the current breath"""
    B = tcom.shape[0]; tick = np.broadcast_to(cyc[:, None], (B, K)).copy(); ts = np.array([min(c for c in range(NCYC - NPLAN)
        if math.ceil(K * (c + 1) / (NCYC - NPLAN)) > j) for j in range(K)]) + NPLAN
    for b in range(B):
        last = max(int(cyc[b]) - 1, 0)
        ks = np.flatnonzero(oponly[b]); tick[b, ks] = np.minimum(rng.integers(0, NPLAN, len(ks)), last)
        ks = np.flatnonzero(tcom[b]); ks = ks[rng.permutation(len(ks))]; tick[b, ks] = np.minimum(ts[:len(ks)], last)
    return tick


def commit_stamps(rng, committed, cyc):
    """(B, K) T6 stamps: a committed slot = the breath it was committed in (a random commit order on the scorer's schedule, never after
    the current breath's predecessor), an open slot = the current breath"""
    B = committed.shape[0]; tick = np.broadcast_to(cyc[:, None], (B, K)).copy()
    for b in range(B):
        ks = np.flatnonzero(committed[b]); ks = ks[rng.permutation(len(ks))]
        tick[b, ks] = np.minimum(_SCHED[:len(ks)], max(int(cyc[b]) - 1, 0))
    return tick


def joint(lp, kw, open_):
    """per operand: the three kind blocks (token / earlier slot / constant), each shifted by log(its share of the type's open kinds);
    a kind the type closes, or a block with no open key (open_[j] (B, K) bool, None = always open), gets -1e4.
    lp: {name: (B, K, Lk)}, kw (B, K, 2, 3) the kind masks -> {'a': [blocks], 'b': [blocks]}"""
    out = {}
    for r, ri in (("a", 0), ("b", 1)):
        w = kw[:, :, ri]; n = w.sum(-1).maximum(1); blocks = []
        for j, kd in enumerate(KINDS):
            ok = w[..., j] > 0
            if open_[j] is not None: ok = ok & open_[j]
            blocks.append(ok.unsqueeze(-1).where(lp[f"p{r}{kd}"] + (w[..., j] / n).maximum(1e-30).log().unsqueeze(-1), -1e4))
        out[r] = blocks
    return out


# ---- training ---------------------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, bt, lr):
        self.bt = bt; self.params = bt.parameters()
        for p in self.params: p.requires_grad = True
        self.opt = AdamW(self.params, lr=lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=WD); self.jits = {}; self.cjits = {}
        self.srng = np.random.default_rng(10_007 + BR.SEED)          # the commit-order draws (own stream: the masks stay BTC's)
        self.sabrng = np.random.default_rng(20_011 + BR.SEED)        # (SAB) the corruption draws (own stream: SAB=0 stays bit-identical)

    def _conf(self, atoms, x, aux, ta, tp, sa, sp, mt, ms, com, atom, rad, cert, vts, vss, vcr, vpt, vps, vpc, tau):
        """pass 1 (no grad): the model's own confidence per slot, as the scorer computes it"""
        lt, _, lp, c, pc_ = forward(self.bt, atoms, x, aux, ta, tp, sa, sp, mt, ms, com, atom, rad, cert, vts, vss, vcr, vpt, vps, vpc, tau)
        cf = confidence(self.bt, atoms, lt, lp, c, pc_, vps)
        if MACRO: return Tensor.stack(cf, (lt.exp() @ _MPAR.T).max(-1), dim=-1).realize()   # + the plan's radius: max p(op)
        return cf.realize()

    def _step(self, atoms, x, aux, ta, tp, sa, sp, mt, ms, com, atom, rad, cert, vts, vss, vcr, vpt, vps, vpc, tau, gt, mk, am, kw,
              tat, tas, tac, tbt, tbs, tbc, *mx):
        with Tensor.train():
            self.opt.zero_grad()
            lt, z, lp, _, _ = forward(self.bt, atoms, x, aux, ta, tp, sa, sp, mt, ms, com, atom, rad, cert, vts, vss, vcr, vpt, vps, vpc,
                                      tau, cond=gt)
            ce = -(lt * gt.one_hot(NT).cast(dtypes.float)).sum(-1)
            lo = (ce * mk).sum() / mk.sum().maximum(1)
            if MACRO:   # the plan rows: -log p(gold op) = -logsumexp over its types (the op is the type marginal)
                mko, gop = mx
                lop = (lt.exp() @ _MPAR.T).maximum(1e-30).log()                                          # (B, K, NOP) log p(op)
                ceo = -(lop * gop.one_hot(NOP).cast(dtypes.float)).sum(-1)
                lo = lo + (ceo * mko).sum() / mko.sum().maximum(1)
            aw = am / am.sum().maximum(1)
            J = joint(lp, kw, [None, vps.any(-1), None])
            for r, (vt_, vs_, vc_) in (("a", (tat, tas, tac)), ("b", (tbt, tbs, tbc))):
                full = Tensor.cat(*J[r], dim=-1); val = Tensor.cat(vt_, vs_, vc_, dim=-1)
                term = full.logsumexp(-1) - val.where(full, -1e4).logsumexp(-1)
                lo = lo + (term * aw).sum()
            lo.backward()
            gp = [p for p in self.params if p.grad is not None]
            gn = Tensor.stack(*[(p.grad ** 2).sum() for p in gp]).sum().sqrt()
            coef = (1.0 / (gn + 1e-6)).minimum(1.0); fin = lo.isfinite() & gn.isfinite()
            for p in gp: p.grad = fin.where(p.grad * coef, 0.0)                                     # where-gated (NaN * 0 stays NaN)
            for p in self.params:                                                                   # a param the loss never reaches
                if p.grad is None: p.grad = p.zeros_like()                                          # (waist_at=end: the token waist)
            zz = z.detach().reshape(-1, z.shape[-1]).contiguous()
            Tensor.realize(lo, zz, fin, *[p.grad for p in gp])                                  # the silhouettes BEFORE the update
            self.opt.step()
            return lo.realize(), zz, fin.realize()

    def step(self, its, states, tree, rng, atoms):
        t, E = BR.batch(its, states, tree, rng); L = t["pad"].shape[1]
        for b, it in enumerate(its): E[b]["vals"] = it["vals"]
        progs, gt = BR.gold(E)
        m = np.array([e["m"] for e in E]); rr = rng.random(len(E))[:, None]
        masked = rng.random((len(E), K)) < rr
        masked[np.arange(len(E)), rng.integers(0, np.minimum(m + 1, K))] = True
        cyc = np.clip(np.floor(NCYC * (1 - rr[:, 0])), 0, NCYC - 1).astype(np.int64)
        Lb = max(64, int(math.ceil((L - K) / BUCKET) * BUCKET))
        P = port_batch(t, E, its, Lb); B = P["B"]
        if MACRO:   # the phase by the breath: plan rows (cyc < NPLAN) have ops committed, types open; type rows have every op committed
            plan = cyc < NPLAN; tcom = ~masked & ~plan[:, None]; opk = ~masked | ~plan[:, None]
            nprogs, ntyp = sabotage(self.sabrng, progs, gt, tcom) if SAB else (progs, gt)
            opid = np.where(tcom, PARENT[ntyp], PARENT[gt])
            po = opk & ~tcom; hit = po & (self.sabrng.random((B, K)) < np.minimum(self.sabrng.random(B) * 2 * SABOP, 1.0)[:, None])
            alt = self.sabrng.integers(0, NOP - 1, (B, K)); opid = np.where(hit, alt + (alt >= opid), opid)            # a random OTHER op
            nb = BR.notebook(E, L, nprogs, tcom, ntyp, np.zeros((B, K), np.float32) if SELFCOND else np.ones((B, K), np.float32))
            nb = (*nb, opk, opid, np.zeros((B, K), np.float32) if SELFCOND else po.astype(np.float32))
            tick = macro_stamps(self.srng, opk & ~tcom, tcom, cyc) if STAMP else cyc
        else:
            nprogs, ntyp = sabotage(self.sabrng, progs, gt, ~masked) if SAB else (progs, gt)
            nb = BR.notebook(E, L, nprogs, ~masked, ntyp, np.zeros((B, K), np.float32) if SELFCOND else np.ones((B, K), np.float32))
            tick = commit_stamps(self.srng, ~masked, cyc) if STAMP else cyc
        H_ = host_io(self.bt, P, nb, tick); V = valid_masks(P, B)
        ta, tp = self.bt.layout.angles(P["tf"], (B, Lb))
        tau = np.array([self.bt.c.tau(int(c)) for c in cyc], np.float32)
        if SELFCOND:                                                                             # pass 1: atoms hidden (radius 0)
            a1 = [atoms, P["x"], P["aux"], ta, tp, H_["sa"], H_["sp"], H_["mt"], H_["ms"], H_["com"], H_["atom"].astype(np.int32), H_["rad"],
                  H_["cert"], V["token_self"], V["slot_self"], V["cross"], V["pt"], V["ps"], V["pc"], tau]
            a1 = [a if isinstance(a, Tensor) else Tensor(np.ascontiguousarray(a)) for a in a1]
            if Lb not in self.cjits: self.cjits[Lb] = TinyJit(self._conf)
            cf1 = self.cjits[Lb](*a1).numpy()
            if MACRO: H_["rad"] = np.where(tcom, cf1[..., 0], np.where(opk, cf1[..., 1], 0)).astype(np.float32)
            else: H_["rad"] = np.where(~masked, cf1, 0).astype(np.float32)
        mkt = masked & ~plan[:, None] if MACRO else masked                                     # the type / pointer loss rows
        am = (t["am"].numpy() & mkt).astype(np.float32)
        kw = _KM[gt]
        args = [atoms, P["x"], P["aux"], ta, tp, H_["sa"], H_["sp"], H_["mt"], H_["ms"], H_["com"], H_["atom"].astype(np.int32), H_["rad"],
                H_["cert"], V["token_self"], V["slot_self"], V["cross"], V["pt"], V["ps"], V["pc"], tau, gt.astype(np.int32),
                mkt.astype(np.float32), am, kw, *P["tgt"]["a"], *P["tgt"]["b"]]
        if MACRO: args += [(masked & plan[:, None]).astype(np.float32), PARENT[gt].astype(np.int32)]
        args = [a if isinstance(a, Tensor) else Tensor(np.ascontiguousarray(a)) for a in args]
        if Lb not in self.jits: self.jits[Lb] = TinyJit(self._step)
        lo, zz, fin = self.jits[Lb](*args)
        self.bad = 0 if bool(fin.numpy()) else getattr(self, "bad", 0) + 1                    # the NaN gate zeroes a step's update;
        if self.bad >= 50: raise RuntimeError("50 consecutive non-finite steps: the NaN gate is zeroing every update")   # never silently
        return float(lo.numpy()), zz.numpy().reshape(B, K, -1), gt, masked


class Atoms:
    """The step atoms: the Tree's codebook for the silhouette (capped Welford means of gold-typed silhouettes, as BR's C / Cn)."""
    def __init__(self, n, d, cap, seed=0):
        self.C = (np.random.default_rng(seed).standard_normal((n, d)) * 0.1).astype(np.float32); self.n = np.zeros(n, np.float32); self.cap = cap

    def update(self, z, typ, mask):
        zz, tt = z[mask], typ[mask]
        if ZOOM3: zz, tt = np.concatenate([zz, zz, zz]), np.concatenate([tt, NT + MID[tt], NT + NMID + PARENT[tt]])   # + its (op, a)
        elif ZOOM or MACRO: zz, tt = np.concatenate([zz, zz]), np.concatenate([tt, NT + PARENT[tt]])        # each silhouette also feeds its op's atom
        cnt = np.bincount(tt, minlength=len(self.n)).astype(np.float32); sm = np.zeros_like(self.C); np.add.at(sm, tt, zz)
        nn_ = self.n + cnt; u = cnt > 0
        self.C[u] += (sm[u] - cnt[u, None] * self.C[u]) / nn_[u, None]; self.n = np.minimum(nn_, self.cap)

    def tensor(self):
        return Tensor(self.C).realize()


# ---- the scorer facade (BR.score drives it: one scorer for every arm) ---------------------------------------------------------------
class Facade:
    """BR.score's net: read(t) is called once per batch of 128, in order, so it advances through the items (the port needs their words)"""
    def __init__(self, bt, atoms, its):
        self.bt, self.atoms, self.its, self.s0, self.part, self.cb = bt, atoms, its, 0, None, None

    def eval(self): return self

    def read(self, t):
        B = t["pad"].shape[0]; self.part = self.its[self.s0:self.s0 + B]; self.s0 += B; self.cb = np.full((B, K), -1)
        return t                                                                                  # the reader runs inside breath

    def breath(self, t, xq, nb, cond, cyc, sp=None):
        import torch
        L = t["pad"].shape[1]; Lb = max(64, int(math.ceil((L - K) / BUCKET) * BUCKET))
        P = port_batch(t, None, self.part, Lb); B = P["B"]
        nbn = tuple(a.numpy() for a in nb[:8 if MACRO else 5]); c0 = int(cyc[0])
        if MACRO:   # a slot's stamp = the breath of its latest commit (the op, then the type)
            self.cb[(nbn[5] & (nbn[0] < 0)) & (self.cb < 0)] = c0 - 1
            nt = (nbn[0] >= 0) & (self.cb < NPLAN) & (c0 > NPLAN); self.cb[nt] = c0 - 1
            self.cb[(nbn[0] >= 0) & (self.cb < 0)] = c0 - 1
        else: self.cb[(nbn[0] >= 0) & (self.cb < 0)] = c0 - 1                                   # BR.score commits at the end of a breath
        tick = np.where(self.cb >= 0, self.cb, c0) if STAMP else np.full(B, c0)
        H_ = host_io(self.bt, P, nbn, tick); V = valid_masks(P, B)
        ta, tp = self.bt.layout.angles(P["tf"], (B, Lb))
        T = lambda a: Tensor(np.ascontiguousarray(a))
        lt, z, lp, c, _ = forward(self.bt, self.atoms, T(P["x"]), T(P["aux"]), T(ta), T(tp), T(H_["sa"]), T(H_["sp"]), T(H_["mt"]), T(H_["ms"]),
                                  T(H_["com"]), T(H_["atom"].astype(np.int32)), T(H_["rad"]), T(H_["cert"]), T(V["token_self"]), T(V["slot_self"]),
                                  T(V["cross"]), T(V["pt"]), T(V["ps"]), T(V["pc"]), self.bt.c.tau(c0),
                                  None if cond is None else Tensor(cond.numpy().astype(np.int32)))
        J = joint(lp, Tensor(_KM[c.numpy()]), [None, T(V["ps"]).any(-1), None])
        Lq = P["Lq"]; padblk = Tensor.full((B, K, L - Lq), -1e4)
        pa, pb = (Tensor.cat(J[r][0][..., :Lq], padblk, J[r][1], J[r][2], dim=-1) for r in "ab")
        lg = lt.numpy()
        return (torch.as_tensor(lg), torch.as_tensor(pa.numpy()), torch.as_tensor(pb.numpy()), torch.as_tensor(z.numpy()),
                torch.as_tensor(np.exp(lg)))


def symmetry(bt, atoms, its, states, tree):
    """THE SYMMETRY READ (the 11-07 autopsy): breath 0, empty notebook -> distinct predicted types per problem among its gold slots
    (the no-skip defect read 1.00; gold ~4.1), and the max spread of the open slots' silhouettes"""
    t, E = BR.batch(its, states, tree, np.random.default_rng(0)); L = t["pad"].shape[1]; B = len(its)
    for b, it in enumerate(its): E[b]["vals"] = it["vals"]
    progs, gt = BR.gold(E); m = np.array([e["m"] for e in E])
    Lb = max(64, int(math.ceil((L - K) / BUCKET) * BUCKET)); P = port_batch(t, E, its, Lb); V = valid_masks(P, B)
    nb = BR.notebook(E, L, progs, np.zeros((B, K), bool), gt, np.ones((B, K), np.float32))
    if MACRO: nb = (*nb, np.zeros((B, K), bool), np.zeros((B, K), np.int64), np.zeros((B, K), np.float32))   # no plan yet
    H_ = host_io(bt, P, nb, np.zeros(B, np.int64))
    ta, tp = bt.layout.angles(P["tf"], (B, Lb)); T = lambda a: Tensor(np.ascontiguousarray(a))
    lt, z, _, _, _ = forward(bt, atoms.tensor(), T(P["x"]), T(P["aux"]), T(ta), T(tp), T(H_["sa"]), T(H_["sp"]), T(H_["mt"]), T(H_["ms"]),
                             T(H_["com"]), T(H_["atom"].astype(np.int32)), T(H_["rad"]), T(H_["cert"]), T(V["token_self"]), T(V["slot_self"]),
                             T(V["cross"]), T(V["pt"]), T(V["ps"]), T(V["pc"]), bt.c.tau(0))
    ty, zz = lt.numpy().argmax(-1), z.numpy()
    return dict(pred=float(np.mean([len(set(ty[b, :m[b]])) for b in range(B)])), gold=float(np.mean([len(set(gt[b, :m[b]])) for b in range(B)])),
                spread=float(max(np.ptp(zz[b, :m[b]], 0).max() for b in range(B))))


def score(bt, atoms, its, states, tree, noise=None):
    """BR.score (its schedule loop, executor, gold and metric definitions) on the facade"""
    return BR.score(Facade(bt, atoms.tensor(), its), its, states, tree, seed=0, noise=noise)


if __name__ == "__main__":
    MODE = os.environ.get("BTC_MODE", "smoke")
    SEED, TAG = BR.SEED, os.environ.get("BT_TAG", f"BTC_s{BR.SEED}")
    Dt, states = BR.TR.load(); tree = BR.TR.load_tree(); train = Dt["TRAIN"]
    BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
    rng = np.random.default_rng(SEED); Tensor.manual_seed(SEED)
    t0_, E0 = BR.batch(train[:2], states, tree, np.random.default_rng(0)); P0 = port_batch(t0_, E0, train[:2], 64)
    bt = BTM.BreathingTransformer(config(P0["x"].shape[-1], P0["aux"].shape[-1]))
    npar = sum(p.numel() for p in bt.parameters()); atoms = Atoms(NA, DB, BR.NCAP, SEED)
    tr = Trainer(bt, BR.LR)
    print(f"[btc] {TAG} D {D} heads {NH} waist {DB} blocks {NBLK} x {NCYC} breaths | planes {bt.layout.P} | params {npar} "
          f"| d_in {P0['x'].shape[-1]} d_aux {P0['aux'].shape[-1]} | {Tensor.zeros(1).device}", flush=True)
    steps = int(os.environ.get("BTC_STEPS", "40")) if MODE == "smoke" else BR.STEPS
    t0 = time.time(); curve = []; los = []
    for step in range(1, steps + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), BR.BATCH)]
        lo, zz, gt, mk = tr.step(its, states, tree, rng, atoms.tensor())
        atoms.update(zz, gt, mk); los.append(lo)
        if step % 10 == 0 and MODE == "smoke":
            print(f"[btc] smoke step {step} loss {np.mean(los[-10:]):.4f} | {(time.time() - t0) / step:.3f}s/step (buckets {sorted(tr.jits)})", flush=True)
        if step == 20: t20 = time.time()
        if MODE == "train" and (step % 1000 == 0 or step == steps):
            r = {k: score(bt, atoms, Dt[k][:600], states, tree) for k in ("ORIG", "TEST")}
            sy = symmetry(bt, atoms, Dt["ORIG"][:64], states, tree)
            curve.append(dict(step=step, loss=float(np.mean(los[-100:])), sym=sy, **{f"{k}_{m}": float(v[m].mean()) for k, v in r.items() for m in ("ans", "step_all")}))
            print(f"[btc] {TAG} step {step} loss {np.mean(los[-100:]):.4f} " + " ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f} "
                  f"type {v['type_ok'].mean():.3f}" for k, v in r.items()) + f" | distinct types/problem {sy['pred']:.2f} (gold {sy['gold']:.2f}) "
                  f"z spread {sy['spread']:.3g} ({time.time() - t0:.0f}s)", flush=True)
    if MODE == "smoke":
        if steps > 20: print(f"[btc] steady s/step (21..{steps}): {(time.time() - t20) / (steps - 20):.3f}", flush=True)
        r = score(bt, atoms, Dt["TEST"][:128], states, tree); sy = symmetry(bt, atoms, Dt["ORIG"][:64], states, tree)
        print(f"[btc] smoke symmetry ORIG[:64]: distinct types/problem {sy['pred']:.2f} (gold {sy['gold']:.2f}) z spread {sy['spread']:.3g}", flush=True)
        print(f"[btc] smoke score TEST[:128] " + " ".join(f"{m} {r[m].mean():.3f}" for m in ("ans", "step_all", "step_ptr", "type_ok")), flush=True)
    else:
        if os.environ.get("BT_SAVE") == "1":
            safe_save(bt.state_dict(), f".cache/gsmb_btc_{TAG}.safetensors"); np.save(f".cache/gsmb_btc_{TAG}_atoms.npy", atoms.C)
        R = {k: score(bt, atoms, Dt[k], states, tree) for k in BR.SPLITS}
        out = dict(tag=TAG, arm="BTC", selfcond=SELFCOND, stamp=STAMP, wd=WD, dropout=DROP, blocks=NBLK, tokid=TOKID, slotself=SLOTSELF, tau1=TAU1, pscale=PSCALE, bind=BIND, bindrel=bool(BSKIP), sab=SAB, macro=MACRO, sabop=SABOP, slotk=SLOTK, zoom=ZOOMN, waist_at=WAIST, d_waist=DB, funnel=FUNNEL, stack=os.environ.get("BTC_STACK"), tie=TIE, turn=TURN, mqa=MQA, ptrshare=PTRSHARE, seed=SEED, d=D, params=npar, steps=steps, curve=curve, engine="tinygrad",
                   **{f"{k}_{m}": float(v[m].mean()) for k, v in R.items() for m in v if m not in ("commit_cycle", "pred", "val")},
                   **{f"{k}_per": [int(x) for x in v["ans"]] for k, v in R.items()})
        json.dump(out, open(f".cache/gsmb_btc_{TAG}.json", "w"))
        print(f"[btc] {TAG} FINAL " + " | ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f} op {v['step_op'].mean():.3f} "
              f"ptr {v['step_ptr'].mean():.3f} type {v['type_ok'].mean():.3f}" for k, v in R.items()), flush=True)
