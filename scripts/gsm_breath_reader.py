"""gsm_breath_reader.py -- THE BT ARM (ledger 2026-10-30, word given: "the diffusion reader should be a true Breathing Transformer
honoring all our ideas such as no MLP"). The GSM8K pointer-program bench (gsm_trunk_reader.py's data, splits, executor and scoring
definitions), read by a breathing transformer under every ruling:
  NO MLP          attention, LayerNorm and linear maps only (the U-Net projections are linear)
  READER          run once per problem: TICK's input chassis (trunk state, ID phases, tree gender, offset, T6 sentence clock on q/k)
  THE BREATH      ONE weight-tied set of NLB cross+self attention blocks, applied every cycle; slot rows query [question ; all slots]
                  (every step sees every step: the diffusion ruling); the T6 step tick rotates slot q/k
  U-NET           per breath, per step: D -> DB (the SILHOUETTE) -> classified against 37 ATOM centroids (op x operand kinds + stop),
                  Welford running means of gold-typed silhouettes (no gradient, count capped) -> snapped -> expanded back to D
  GRAVITY         type loss = softmax over -distance to every centroid (contrastive: own atom vs the other 36)
  RADIUS          confidence r = p(type) scales the snapped centroid's re-entry; commit order = the largest radius
  MICRO KEYS      pointer queries come ONLY from the expanded silhouette (no full-state skip: the bypass guard), bound with the micro
                  keys: slot id, T6 step tick, tree level (the executor's derivation depth)
  ROTOR PLANES    pointer space = D/2 complex planes: [0:NTK) the T6 step tick (relative rotation of q and slot keys),
                  [NTK:) the BUS: q bound to its type's role phase e^{i phi(type, operand)}, read by the conjugate (Re q e^{i phi} conj k)
  STRUCTURE       the type's operand kinds mask each pointer's domain (token / earlier slot / constant): the silhouette conditions keys
  DIFFUSION       training = masked discrete diffusion on the gold program (loss on masked slots); inference = 6 cycles from all-masked,
                  committing the most confident slots each cycle
  ALTERNATION     after every cycle the executor runs every committed slot whose operands are known and writes back certificates
                  (executed / integer / non-negative -- never values), consumed marks and tree levels; the next cycle reads them
Env: BT_SEED, BT_TAG, BT_STEPS (6000), BT_BATCH (64), BT_LR (3e-4), TOY_THREADS. Output .cache/gsmb_breath_{TAG}.json."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
os.environ["RD_ARM"] = "TICK"                     # the chassis's encoder features (ID phases, tree gender, sentence clock, step tick)
import gsm_trunk_reader as TR
import gsm_bench_data as GD
from mycelium import rotor_clock as CK
import gsm_units as GU
import gsm_dto as DTO

SEED = int(os.environ.get("BT_SEED", "0")); TAG = os.environ.get("BT_TAG", f"BT_s{SEED}")
ORACLE = os.environ.get("BT_ORACLE") == "1"          # the oracle-type read: the gold step type is the silhouette's class (inference only)
LOAD = os.environ.get("BT_LOAD", "")                 # score saved weights (.cache/gsmb_breath_{TAG}.pt) instead of training
ROLE = os.environ.get("BT_ROLE") == "1"              # THE TREE ROAD (ledger 10-31): each word += Linear(14 -> D)(its tree role feats)
CYCLOCK = os.environ.get("BT_CYCLOCK") == "1"        # (ledger 10-31) slot queries turn by the T6 gears x the CYCLE index (keys fixed)
ZOOM = os.environ.get("BT_ZOOM") == "1"              # (ledger 10-31) coarse-to-fine silhouette: p(type) = p(op) p(type | op); unsure -> op-level mask
ZOOM_TAU = 0.5
SOFT = os.environ.get("BT_SOFT") == "1"              # (ledger 11-01) the soft notebook: uncommitted slots show sum_t p(t) C_t, tau by cycle
KINDCE = os.environ.get("BT_KINDCE") == "1"          # (ledger 11-03 BT7b) + the pointer CE under the gold type with every kind open
GATE = os.environ.get("BT_GATE", "")                 # (ledger 11-02 BT6) inference: commit iff radius >= rho0 (1 - c/5); "" = the schedule
CHAIN = os.environ.get("BT_FACT") == "2"             # (ledger 11-04 BT8b) chained binding: kind keys bound to op (a) and op + kind a (b)
FACT = os.environ.get("BT_FACT") in ("1", "2")              # (ledger 11-03 BT8) the factored silhouette: op / kind a / kind b phase-bound, conjugate unbind
IDSKIP = os.environ.get("BT_IDSKIP") == "1"          # (ledger 11-03 BT9) slot id + tree level bypass the waist as phase tags (the ID skip)
SPH = os.environ.get("BT_SPH", "0")                 # (ledger 11-04 SPH) 1 = the torus silhouette; 2 = + the turning breath (rotation-only updates)
UNIT = os.environ.get("BT_UNIT", "0")                # (ledger 11-05 UNIT-BIND) 1 = units phase-bound per problem; 2 = the placebo
PU = 32; NUF = 2 * PU + 4 if UNIT != "0" else 0     # the unit block: cos | sin | kind one-hot (none, unknown, dimensionless, unit)
assert UNIT == "0" or ROLE, "UNIT-BIND rides the role road (BT_ROLE=1)"
SEAL = os.environ.get("BT_SEAL") == "1"              # (ledger 11-05) THE SEALED OP HEAD: p(type) = p_text(struct) x p_DTO(op)
assert not SEAL or not (ZOOM or FACT or SOFT or GATE), "SEAL replaces zoom's p(op); not with FACT / SOFT / GATE"
DW, DNH = 64, 4                                       # (SEAL) the DTO head: width, heads
PHPTR = os.environ.get("BT_PHPTR") == "1"            # (ledger 11-06 PP) PHASE-BOUND POINTERS: token phases, relations by the conjugate
PR = 32                                               # (PHPTR) the token phase planes
assert not PHPTR or not (SEAL or SOFT), "PHPTR: not with SEAL / SOFT"
ZQ = os.environ.get("BT_ZQ") == "1"                  # (ledger 11-06 ZQ) THE ZOOMING QUERY: log p(the token's Tree chart | q) on the pointer
NCH = 65                                              # (ZQ) 64 ST64 charts + none
CH = None                                             # (ZQ) the chart memmap, loaded on first use
CS = os.environ.get("BT_CS") == "1"                  # (ledger 11-06 CS) THE CHART-SEALED POINTER (rides on ZQ's charts + keys)
assert not CS or (ZQ and not PHPTR), "CS: needs BT_ZQ=1, not with PHPTR"


def chart_seal(F, q, zck, ch, m):
    """(CS) F (B, K, Lq) natural token logits, m (B, K, Lq) open, ch (B, Lq) charts -> the sealed token logits (natural units)"""
    import torch
    F_ = torch.where(m, F, torch.full_like(F, -1e4))
    oh = torch.nn.functional.one_hot(ch, NCH).float()                                          # (B, Lq, C)
    inc = m[..., None] & (oh[:, None] > 0.5)                                                    # (B, K, Lq, C)
    fm = torch.where(inc, F_[..., None], torch.full_like(inc, -1e4, dtype=F.dtype)).amax(2)    # (B, K, C) per-chart max
    S = torch.where(inc, F_[..., None] - fm[:, :, None], torch.full_like(inc, -1e4, dtype=F.dtype)).exp().sum(2)
    lsec = fm + S.clamp(min=1e-30).log()                                                        # lse within each chart
    lchr = torch.where(S > 0.5, torch.einsum("bkd,cd->bkc", q, zck) / math.sqrt(D), torch.full_like(S, -1e4)).log_softmax(-1)
    return F_.logsumexp(-1, keepdim=True) + torch.einsum("bkc,blc->bkl", lchr - lsec, oh) + F_
T6OFF = os.environ.get("BT_T6OFF") == "1"            # (ledger 10-31, U1) no learned offset: the offset turns its own T6 planes
STEPS, BATCH, LR = int(os.environ.get("BT_STEPS", "6000")), int(os.environ.get("BT_BATCH", "64")), float(os.environ.get("BT_LR", "3e-4"))
D, NH, DB = int(os.environ.get("BT_D", 192)), int(os.environ.get("BT_NH", 4)), int(os.environ.get("BT_DB", 32))   # (ledger 10-31: 512 / 8 / 64)
NLE, NLB, NCYC = int(os.environ.get("BT_NLE", 2)), int(os.environ.get("BT_NLB", 2)), 6   # (ledger 11-01: reader / breath depth knobs)
K, NC, H, OMAX, PB = TR.K, TR.NC, TR.H, TR.OMAX, TR.PB
KINDS = "tsc"; TYPES = [(o, a, b) for o in GD.OPS for a in KINDS for b in KINDS]; STOP = len(TYPES); NT = STOP + 1
LEVMAX, NCAP = 8, 2000
NOP = len(GD.OPS) + 1                                 # the coarse atoms: the op, + stop
PARENT = np.array([GD.OPS.index(o) for o, _, _ in TYPES] + [NOP - 1])
FKA = np.array([KINDS.index(a) for _, a, _ in TYPES] + [0]); FKB = np.array([KINDS.index(b) for _, _, b in TYPES] + [0])   # (BT8) stop: no kinds
NS = 3 * len(KINDS) + 1                               # (SEAL) the structs: operand kinds (9) + stop
SOF = np.array([3 * KINDS.index(a) + KINDS.index(b) for _, a, b in TYPES] + [NS - 1])        # type -> struct
OPI = np.array([GD.OPS.index(o) for o, _, _ in TYPES] + [0])                                  # type -> op (stop: masked by isstep)
CANON = np.array([TYPES.index((GD.OPS[0], a, b)) for a in KINDS for b in KINDS] + [STOP])    # struct -> its canonical type
assert all(j == 9 * OPI[j] + SOF[j] for j in range(STOP)), "TYPES layout: type = 9 op + struct"
SPLITS = TR.SPLITS


def rolef_name():
    """the role feats built on the same reader states (RoPE retirement ruling 11-03): a mismatch fails loudly"""
    st = os.environ.get("RD_STATES", "gsmb_rb8_states.npy")                                 # P9 (ledger 11-05): the RB8 reader
    pair = {"gsmb_rb8_states.npy": "gsmb_role_feats_rb8.npy", "gsmb_mbr8_states.npy": "gsmb_role_feats_mbr8.npy", "gsmb_t9ns_states.npy": "gsmb_role_feats_t9ns.npy", "gsmb_student_states.npy": "gsmb_role_feats.npy"}
    rf = os.environ.get("BT_ROLEF", pair.get(st, ""))
    assert rf and (st not in pair or rf == pair[st]), f"role feats {rf!r} were not built on the reader states {st!r}"
    return rf


def unit_feats(U, KD, th, rnd=None):
    """(UNIT-BIND) per row -> (n, NUF): cos | sin of the unit phase (zeros unless known) | kind one-hot; th = word -> phase vector;
    rnd (n, PU): the placebo's independent phases for unit-bearing rows"""
    f = np.zeros((len(U), NUF), np.float32)
    for i, (u, kd) in enumerate(zip(U, KD)):
        f[i, 2 * PU + kd] = 1
        if kd < 2: continue
        a = rnd[i] if rnd is not None and kd == 3 else sum((e * th[w] for w, e in u.items()), np.zeros(PU))
        f[i, :PU] = np.cos(a); f[i, PU:2 * PU] = np.sin(a)
    return f


def qnoun(it):
    """(PHPTR) the question noun: the content word after "how many / much" in the last sentence, else the token before the last "?"
    (the CHART QUERY oracle's heuristic, verbatim in effect)"""
    toks, sid = it["toks"], it["sid"]; q0 = sid.index(max(sid)); q = toks[q0:]
    for i in range(len(q) - 1):
        if q[i] == "how" and q[i + 1] in ("many", "much"):
            j, _ = GU.content(q, i + 2)
            if j is not None: return q0 + j
    last = max(i for i, t in enumerate(toks) if t == "?") if "?" in toks else len(toks) - 1
    return max(0, last - 1)


def step_type(op, a, b):
    return STOP if op == "." else TYPES.index((op, a[0], b[0]))


def notebook(E, L, progs, committed, types, radius):
    """the executor's write-back for the committed slots -> slot_in arrays: typ (K; -1 masked), r, cf (K x 3), lev, cons (L)."""
    B = len(E); typ = np.full((B, K), -1); r = np.zeros((B, K), np.float32); cf = np.zeros((B, K, 3 + NUF), np.float32)
    lev = np.zeros((B, K), np.int64); cons = np.zeros((B, L), bool)
    for b, e in enumerate(E):
        n, vals = e["n"], e["vals"]; sv = [None] * K; sl = [0] * K; su = [None] * K   # su: (UNIT) each slot's unit
        for k in range(K):
            if not committed[b, k]: continue
            typ[b, k] = types[b, k]; r[b, k] = radius[b, k]; op, A, Bo = progs[b][k]
            if op == ".": continue
            av, known, lv = [], True, 0
            for kind, i in (A, Bo):
                if kind == "t": cons[b, i] = True; av.append(vals[i] if i < n else None)
                elif kind == "s":
                    cons[b, L - K + i] = True; av.append(sv[i] if committed[b, i] else None)
                    if committed[b, i] and sl[i] > 0: lv = max(lv, sl[i])
                    else: known = False                                                    # level unknown until its operands are
                else: av.append(GD.CONSTS[i])
            if known: sl[k] = min(lv + 1, LEVMAX); lev[b, k] = sl[k]
            x, y = av
            if x is not None and y is not None and not (op == "/" and y == 0):
                try: sv[k] = x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else x / y
                except OverflowError: sv[k] = None
            cf[b, k, :3] = TR.cert(sv[k])
            if NUF:   # (UNIT-BIND) the slot's unit by the phase algebra, from its operands' units (tokens, earlier slots, constants = 1)
                ua = [(e["uU"][i] if i < n else None) if kd_ == "t" else (su[i] if committed[b, i] else None) if kd_ == "s" else {}
                      for kd_, i in (A, Bo)]
                su[k] = GU.combine(op, *ua); kd = 1 if su[k] is None else 2 if not su[k] else 3
                cf[b, k, 3:] = unit_feats([su[k]], [kd], e["uth"], e["urs"][k:k + 1] if UNIT == "2" else None)[0]
    if PHPTR:   # (ledger 11-06 PP) per slot: the token operands of the nearest committed earlier step (the prev context)
        pm = np.zeros((B, K, L), bool)
        for b, e in enumerate(E):
            for k in range(K):
                js = [j for j in range(k) if committed[b, j] and progs[b][j][0] != "."]
                if not js: continue
                for kind, i in progs[b][js[-1]][1:]:
                    if kind == "t" and i < e["n"]: pm[b, k, i] = True
        return typ, r, cf, lev, cons, pm
    return typ, r, cf, lev, cons


def gold(E):
    """per item: the gold program per slot (steps, then stops) and its step types."""
    progs, types = [], np.full((len(E), K), STOP)
    for b, e in enumerate(E):
        p = list(e["teach"]) + [(".", None, None)] * (K - len(e["teach"]))
        progs.append(p)
        for k, (op, A, Bo) in enumerate(p):
            if op != ".": types[b, k] = step_type(op, A, Bo)
    return progs, types


def make_net():
    import torch, torch.nn as nn, torch.nn.functional as F
    DH = D // NH; NPA = D // 2; NTK = NPA // 3
    w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)      # attention space: the T6 clock, per head
    wh = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 4)), dtype=torch.float32)

    def ang(sid, off):                                                                     # (B, L) x2 -> (B, L, DH/2) attention angles
        if not T6OFF: return sid[..., None].float() * w
        return torch.cat([sid[..., None].float() * wh, off[..., None].float() * wh], -1)    # sentence planes | offset planes
    wt = torch.as_tensor(np.radians(CK.t6_plane_speeds(NTK)), dtype=torch.float32)        # pointer space: the step tick's planes
    NLV = NTK // 2 if IDSKIP else 0                                                       # (BT9) the tree-level tag planes
    wl = torch.as_tensor(np.radians(CK.t6_plane_speeds(max(NLV, 1))), dtype=torch.float32)
    kmask = torch.zeros(NT, 2, 3, dtype=torch.bool)                                       # type x operand -> allowed kind (t, s, c)
    for j, (_, a, b) in enumerate(TYPES):
        kmask[j, 0, KINDS.index(a)] = True; kmask[j, 1, KINDS.index(b)] = True
    kmask[STOP] = True
    par = torch.as_tensor(PARENT); grp = torch.zeros(NOP, NT, dtype=torch.bool); grp[par, torch.arange(NT)] = True
    fka, fkb = torch.as_tensor(FKA), torch.as_tensor(FKB); isstep = (torch.arange(NT) != STOP).float()   # (BT8) the factor indices per type
    kmask_op = torch.stack([kmask[grp[g]].any(0) for g in range(NOP)])                    # the op-level mask: the union of its types' kinds
    sof, opi, canon = torch.as_tensor(SOF), torch.as_tensor(OPI), torch.as_tensor(CANON)  # (SEAL) type -> struct / op; struct -> canonical
    sgrp = torch.zeros(NS, NT, dtype=torch.bool); sgrp[sof, torch.arange(NT)] = True
    smean = sgrp.float() / sgrp.float().sum(1, keepdim=True)                              # (SEAL) the struct-mean atom weights

    def rot(z, a):                                                                         # z (..., 2P) as P planes; a (..., P)
        c, s_ = a.cos(), a.sin(); re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * s_, re * s_ + im * c], -1).flatten(-2)

    def torus(z):                                                                          # (SPH) every plane on the unit circle
        zz = z.unflatten(-1, (-1, 2)); return (zz / (zz.pow(2).sum(-1, keepdim=True) + 1e-12).sqrt()).flatten(-2)

    class Attn(nn.Module):
        def __init__(s, turn=False):
            super().__init__(); s.lnq = nn.LayerNorm(D); s.lnk = nn.LayerNorm(D)
            s.turn = turn
            if turn: s.oa = nn.Linear(D, D // 2); nn.init.zeros_(s.oa.weight); nn.init.zeros_(s.oa.bias)   # (SPH=2) the turn angles
            s.q = nn.Linear(D, D); s.kv = nn.Linear(D, 2 * D); s.o = nn.Linear(D, D)

        def forward(s, xq, xk, aq, ak, mask):
            B, Lq, _ = xq.shape; Lk = xk.shape[1]
            q = rot(s.q(s.lnq(xq)).reshape(B, Lq, NH, DH).transpose(1, 2), aq[:, None])
            k, v = (t.reshape(B, Lk, NH, DH).transpose(1, 2) for t in s.kv(s.lnk(xk)).chunk(2, -1))
            k = rot(k, ak[:, None])
            o_ = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Lq, D)
            return rot(xq, s.oa(o_)) if s.turn else xq + s.o(o_)                              # (SPH=2) a rotation, never an add

    class DAttn(nn.Module):   # (SEAL) the DTO head's plain self-attention block: no rotation, no MLP
        def __init__(s):
            super().__init__(); s.ln = nn.LayerNorm(DW); s.q = nn.Linear(DW, DW); s.kv = nn.Linear(DW, 2 * DW); s.o = nn.Linear(DW, DW)

        def forward(s, x):
            B_, Lx, _ = x.shape; y = s.ln(x); dh = DW // DNH
            q = s.q(y).reshape(B_, Lx, DNH, dh).transpose(1, 2)
            k, v = (t_.reshape(B_, Lx, DNH, dh).transpose(1, 2) for t_ in s.kv(y).chunk(2, -1))
            sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(dh))
            return x + s.o((sc.softmax(-1) @ v).transpose(1, 2).reshape(B_, Lx, DW))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.role = nn.Linear(14 + NUF, D, bias=False) if ROLE else None   # a mandatory road, no gain
            s.inp = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, D)); s.isn = nn.Embedding(2, D); s.isnum = nn.Embedding(2, D)
            s.off = None if T6OFF else nn.Embedding(OMAX, D); s.idp = nn.Linear(2 * PB, D); s.gp = nn.Linear(3, D)
            s.enc = nn.ModuleList([Attn() for _ in range(NLE)])                                # THE READER (once per problem)
            s.cons = nn.Embedding(2, D)                                                        # the notebook's consumed marks
            s.slotk = nn.Embedding(K, D); s.maskv = nn.Parameter(torch.zeros(D)); s.cfp = nn.Linear(3 + NUF, D)
            s.lev = nn.Embedding(LEVMAX + 1, D); s.reent = nn.Linear(DB, D)
            s.sre = nn.Linear(DB, D, bias=False) if SOFT else None                             # the soft notebook's road (no bias: sp=0 is the old breath)
            s.core = nn.ModuleList([Attn(SPH == "2") for _ in range(NLB)])                               # THE BREATH (weight-tied over cycles)
            s.down = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DB)); s.up = nn.Linear(DB, D); s.upc = nn.Linear(DB, D)
            s.tau = nn.Parameter(torch.zeros(1))
            s.register_buffer("C", torch.randn(NT, DB) * 0.1); s.register_buffer("Cn", torch.zeros(NT))   # the atoms (Welford)
            if ZOOM:                                                                           # the coarse atoms (Welford over the op)
                s.register_buffer("Cc", torch.randn(NOP, DB) * 0.1); s.register_buffer("Ccn", torch.zeros(NOP)); s.tauc = nn.Parameter(torch.zeros(1))
            s.phi = nn.Parameter(torch.rand(NT, 2, NPA - NTK) * 2 * math.pi)                   # type-role phases on the bus planes
            if FACT:   # (BT8) three factor codebooks (Welford) + fixed random role phases on the silhouette's DB/2 planes
                g_ = torch.Generator().manual_seed(8)
                s.register_buffer("psi", torch.rand(3, DB // 2, generator=g_) * 2 * math.pi)
                s.register_buffer("Fop", torch.randn(NOP, DB) * 0.1); s.register_buffer("Fopn", torch.zeros(NOP))
                s.register_buffer("Fa", torch.randn(3, DB) * 0.1); s.register_buffer("Fan", torch.zeros(3))
                s.register_buffer("Fb", torch.randn(3, DB) * 0.1); s.register_buffer("Fbn", torch.zeros(3))
                s.tauf = nn.Parameter(torch.zeros(3))
                if CHAIN:   # (BT8b) the fixed random key phases: op keys and kind-a keys (addresses, not cells)
                    s.register_buffer("Theta", torch.rand(NOP, DB // 2, generator=g_) * 2 * math.pi)
                    s.register_buffer("Phi", torch.rand(3, DB // 2, generator=g_) * 2 * math.pi)
            s.ln = nn.LayerNorm(D); s.qa = nn.Linear(D, D); s.qb = nn.Linear(D, D); s.kt = nn.Linear(D, D); s.ks = nn.Linear(D, D)
            s.kc = nn.Parameter(torch.randn(NC, D) * 0.02); s.pc = nn.Parameter(torch.zeros(1))
            if SEAL:   # (ledger 11-05) THE DTO HEAD: field tokens -> 2 plain attention blocks -> 4 op log-probs; never a token state
                s.dto_e = nn.Embedding(DTO.NV, DW); s.dto_f = nn.Parameter(torch.randn(DTO.NF, DW) * 0.02)
                s.dto_att = nn.ModuleList([DAttn() for _ in range(2)]); s.dto_ln = nn.LayerNorm(DW); s.dto_out = nn.Linear(DW, len(GD.OPS))
            if PHPTR:   # (ledger 11-06 PP) the token phase map, the relation phases psi (m = qn / prev / sib, role), the gains
                s.pth = nn.Linear(D, PR, bias=False); s.ppsi = nn.Parameter(torch.zeros(3, 2, PR))
                s.pg = nn.Parameter(torch.tensor([[1.6, 0.0], [0.0, 2.8], [0.0, -3.0]]))
            if ZQ:   # (ledger 11-06 ZQ) the chart keys (LAST)
                s.zck = nn.Parameter(torch.randn(NCH, D) * 0.02)

        def ophead(s, ids):
            """(SEAL) (B, K, NF) DTO field ids -> (B, K, 4) log p(op)"""
            B_, K_, _ = ids.shape; x = s.dto_e(ids.reshape(B_ * K_, -1)) + s.dto_f
            for blk in s.dto_att: x = blk(x)
            return s.dto_out(s.dto_ln(x.mean(1))).log_softmax(-1).reshape(B_, K_, -1)

        def read(s, t):
            x = s.inp(t["x"])
            th = torch.gather(t["theta"], 1, t["nid"].clamp(min=0)[..., None].expand(-1, -1, PB))
            x = torch.where(t["isn"][..., None], s.idp(torch.cat([th.cos(), th.sin()], -1)), x)
            x = x + s.isn(t["isn"].long()) + s.isnum(t["isnum"].long()) + s.gp(t["g"]) + (0 if T6OFF else s.off(t["off"]))
            if ROLE: x = x + s.role(t["role"])
            Lq = x.shape[1] - K; xq = x[:, :Lq]; aq = ang(t["sid"][:, :Lq], t["off"][:, :Lq])
            mask = torch.zeros(x.shape[0], 1, Lq, Lq).masked_fill(t["pad"][:, None, None, :Lq], -1e4)
            for blk in s.enc: xq = blk(xq, xq, aq, aq, mask)
            return xq

        def unbind(s, z):
            """(BT8) the conjugate mirror: u_r = z * conj(e^{i psi_r}) for r = op, a, b -> (3, ..., DB)"""
            return torch.stack([rot(z, -s.psi[r]) for r in range(3)])

        def keys(s):
            """(BT8b) per type: the kind-a and kind-b key phases -> (NT, DB/2) x2 (BT8: the bare role phases)"""
            if not CHAIN: return s.psi[1].expand(NT, -1), s.psi[2].expand(NT, -1)
            return s.psi[1] + s.Theta[par], s.psi[2] + s.Theta[par] + s.Phi[fka]

        def unbind_gold(s, z, tt):
            """(BT8b) the gold silhouettes unbound under the gold type's own keys -> (3, N, DB)"""
            ka_, kb_ = s.keys()
            return torch.stack([rot(z, -s.psi[0]), rot(z, -ka_[tt]), rot(z, -kb_[tt])])

        def atoms(s):
            """(BT8) the rebound composite per type: sum_r e^{i psi_r} C_r[factor_r(t)] (stop: the op factor alone) -> (NT, DB)"""
            ka_, kb_ = s.keys()
            a = rot(s.Fop[par], s.psi[0]) + (rot(s.Fa[fka], ka_) + rot(s.Fb[fkb], kb_)) * isstep[:, None]
            return a

        def breath(s, t, xq, nb, cond, cyc=None, sp=None):
            """one cycle: every slot proposed in parallel. nb = notebook tensors; cond (B, K) = the step type the keys condition on
            (gold in training, the snapped prediction at inference; None = the snapped prediction)."""
            B, Lq, _ = xq.shape; L = Lq + K
            typ, r, cf, lev, cons = nb[:5]; prevm = nb[5] if PHPTR else None
            com = typ >= 0
            CA = s.atoms() if FACT else s.C                                                   # (BT8) the rebound atoms
            re = s.reent(r[..., None] * CA[typ.clamp(min=0)])                                  # committed: atom x radius
            unc = s.maskv if sp is None else s.maskv + s.sre(sp @ CA)                        # uncommitted: the mask + what the BT believes
            h = torch.where(com[..., None], re, unc) + s.cfp(cf) + s.cons(cons[:, Lq:].long())
            if not IDSKIP: h = h + s.slotk.weight[None] + s.lev(lev)                         # (BT9: the IDs ride the skip instead)
            xk_q = xq + s.cons(cons[:, :Lq].long())
            sa = ang(t["sid"][:, Lq:], t["off"][:, Lq:]); ka = ang(t["sid"], t["off"])
            if CYCLOCK: sa = sa + cyc.float()[:, None, None] * w                               # the cycle stamp on the queries only
            mask = torch.zeros(B, 1, K, L).masked_fill(t["pad"][:, None, None, :], -1e4)
            for blk in s.core: h = blk(h, torch.cat([xk_q, h], 1), sa, ka, mask)
            z = s.down(h)                                                                      # THE SILHOUETTE
            if SPH != "0": z = torus(z)                                                        # (SPH) on the torus
            logit = -((z[:, :, None] - s.C[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tau.exp()
            if FACT:   # (BT8) unbind each factor, snap it against its own codebook, compose: log p(op) + log p(a) + log p(b)
                u = s.unbind(z)
                lf = [(-((u[r][:, :, None] - F_[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tauf[r].exp()).log_softmax(-1)
                      for r, F_ in enumerate((s.Fop, s.Fa, s.Fb))]
                lfine = (lf[1][..., fka] + lf[2][..., fkb]) * isstep
                if CHAIN:   # (BT8b) unbind each kind under every type's own key: p(ka | op), p(kb | op, ka)
                    ka_, kb_ = s.keys(); zt = z[:, :, None]
                    lca = [(-((rot(zt, -k_)[..., None, :] - F_) ** 2).sum(-1) / math.sqrt(DB) * s.tauf[r].exp()).log_softmax(-1)
                           for r, k_, F_ in ((1, ka_, s.Fa), (2, kb_, s.Fb))]                    # (B, K, NT, 3) each
                    lfine = (lca[0].gather(-1, fka.expand(*lca[0].shape[:2], -1)[..., None])[..., 0]
                             + lca[1].gather(-1, fkb.expand(*lca[1].shape[:2], -1)[..., None])[..., 0]) * isstep
                logit = lf[0][..., par] + lfine
            elif ZOOM:                                                                           # zoom: log p(op) + log p(type | op)
                lc_ = (-((z[:, :, None] - s.Cc[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tauc.exp()).log_softmax(-1)
                lse = torch.stack([logit.masked_fill(~grp[g], -1e9).logsumexp(-1) for g in range(NOP)], -1)
                lfine = logit - lse[..., par]; logit = lc_[..., par] + lfine
            if SEAL:   # (ledger 11-05) the text keeps the struct (op marginalized out); the DTO head supplies the op (training: t["dto"])
                lt_ = logit.log_softmax(-1)
                lstr = torch.stack([lt_.masked_fill(~sgrp[g], -1e9).logsumexp(-1) for g in range(NS)], -1)   # (B, K, NS) log p(struct)
                logit = lstr[..., sof]
                if "dto" in t: logit = logit + s.ophead(t["dto"])[..., opi] * isstep
            p = logit.softmax(-1)
            def ptrs(c, zoom, free=False):   # (ledger 11-02 BT7a) the pointer heads under type c, from this breath's state
                if SEAL:   # (ledger 11-05) op-blind: p(struct) as the radius, the struct-mean atom, the canonical type's phases
                    cs = sof[c]; rr = lstr.exp().gather(-1, cs[..., None]); u = s.up(z) + s.upc(rr * (smean @ CA)[cs]); c = canon[cs]
                else:
                    rr = p.gather(-1, c[..., None])                                                    # the radius: type confidence
                    u = s.up(z) + s.upc(rr * CA[c])                                                    # expand: silhouette + snapped atom x r
                hu = s.ln(u)
                tick = t["sid"][:, Lq:, None].float() * wt                                         # (B, K, NTK)

                def bind(q, role):                                                                 # bus planes: q * e^{i phi(type, role)}
                    ph = s.phi[c, role][..., NLV:]                                                 # (BT9: the level planes are not bus)
                    return torch.cat([q[..., :2 * (NTK + NLV)], rot(q[..., 2 * (NTK + NLV):], ph)], -1)

                la = lev.float()[..., None] * wl                                                   # (BT9) the level tag per slot

                def levrot(z_):                                                                    # (BT9) bind the level on its planes
                    if not IDSKIP: return z_
                    return torch.cat([z_[..., :2 * NTK], rot(z_[..., 2 * NTK:2 * (NTK + NLV)], la), z_[..., 2 * (NTK + NLV):]], -1)

                def tickrot(z_):
                    return torch.cat([rot(z_[..., :2 * NTK], tick), z_[..., 2 * NTK:]], -1)

                kt = s.kt(xk_q); ksl = levrot(tickrot(s.ks(hu))); cnt = cons.float()
                km = kmask.to(c.device)[c]                                                         # (B, K, 2, 3)
                if free: km = torch.ones_like(km)                                                  # BT7a: every kind open (structure masks stay)
                if zoom:                                                          # unsure of the fine type -> zoom out to the op
                    unsure = lfine.gather(-1, c[..., None])[..., 0].exp() < ZOOM_TAU
                    km = torch.where(unsure[..., None, None], kmask_op[par[c]], km)
                if PHPTR:   # (ledger 11-06 PP) the native phase embedding: e^{i theta_t} on PR planes of its own
                    thx = s.pth(xk_q); ez = torch.stack([thx.cos(), thx.sin()], -1)                  # (B, Lq, PR, 2)
                    cq = ez[torch.arange(B), t["qn"]][:, None].expand(-1, K, -1, -1)              # the question noun
                    pm_ = prevm[:, :, :Lq].float()
                    cp = torch.einsum("bkl,blpc->bkpc", pm_, ez) / pm_.sum(-1).clamp(min=1)[..., None, None]   # the prev step's tokens

                    def rel(ctx, m, role):   # bind the context by e^{i psi}, unbind against each token by the conjugate: mean_p Re(.)
                        a_ = s.ppsi[m, role]; c_, s_ = a_.cos(), a_.sin(); re_, im_ = ctx[..., 0], ctx[..., 1]
                        cr = torch.stack([re_ * c_ - im_ * s_, re_ * s_ + im_ * c_], -1)
                        return torch.einsum("bkpc,blpc->bkl", cr, ez) / PR
                outs = []
                for role, proj in ((0, s.qa), (1, s.qb)):
                    q = bind(proj(hu), role)
                    lt = torch.einsum("bkd,bld->bkl", q, kt) + s.pc * cnt[:, None, :Lq] * math.sqrt(D)
                    if PHPTR:   # the relations (sib: role a's soft choice, the continuous mirror)
                        ctxs = [cq, cp] + ([torch.einsum("bkl,blpc->bkpc", outs[0].softmax(-1)[..., :Lq], ez)] if role == 1 else [])
                        lt = lt + sum(s.pg[m, role] * rel(cx, m, role) for m, cx in enumerate(ctxs)) * math.sqrt(D)
                    if CS:   # (ledger 11-06 CS) the seal: chart by the zck road only, address within the chart by q . kt
                        m_ = t["isnum"][:, None, :Lq] & km[:, :, role, 0:1]
                        lt = (chart_seal(torch.einsum("bkd,bld->bkl", q, kt) / math.sqrt(D), q, s.zck, t["ch"][:, :Lq], m_)
                              + s.pc * cnt[:, None, :Lq]) * math.sqrt(D)
                    elif ZQ:   # (ledger 11-06 ZQ) the coarse level: the query searches the Tree's charts; + log p(the token's chart | q)
                        lch = (torch.einsum("bkd,cd->bkc", q, s.zck) / math.sqrt(D)).log_softmax(-1) + math.log(NCH)   # centred: uniform = 0
                        lt = lt + lch.gather(-1, t["ch"][:, None, :Lq].expand(-1, K, -1)) * math.sqrt(D)
                    lt = lt.masked_fill(~t["isnum"][:, None, :Lq] | ~km[:, :, role, 0:1], -1e4)
                    ls = torch.einsum("bkd,bjd->bkj", levrot(tickrot(q)), ksl) + s.pc * cnt[:, None, Lq:] * math.sqrt(D)
                    ls = ls.masked_fill(~torch.ones(K, K, dtype=torch.bool).tril(-1) | ~km[:, :, role, 1:2], -1e4)
                    lc = torch.einsum("bkd,cd->bkc", q, s.kc).masked_fill(~km[:, :, role, 2:3], -1e4)
                    outs.append(torch.cat([lt, ls, lc], -1) / math.sqrt(D))
                # pointer layout = TR.batch's valid-set layout: [L positions (question + pad + slots) | K slots | NC consts]
                pad = torch.full((B, K, L - Lq), -1e4)
                pa, pb = (torch.cat([o[..., :Lq], pad, o[..., Lq:]], -1) for o in outs)
                return pa, pb
            s.ptrs = lambda c_, free=False: ptrs(c_, False, free)
            c = logit.argmax(-1) if cond is None else cond
            pa, pb = ptrs(c, ZOOM and cond is None)
            return logit, pa, pb, z, p

    return Net()


MACRO_PLAN = 0   # (ledger 11-09 MACRO1; set by gsm_btc BTC_MACRO) breaths 0..MACRO_PLAN-1 commit the OP PLAN (p(op) = the type marginal),
                 # the rest commit types: a slot's committed op enters the notebook (opk, opid, orad ride after the 5 arrays); the plan
                 # conditions the type commit, never restricts it (soft evidence biases; only a proof crosses hard)
_PAR1H = np.eye(NOP, dtype=np.float32)[PARENT]        # (NT, NOP): type -> its op


def TAU(cyc):
    """the coarse-to-fine schedule: tau 2.0 at cycle 0 -> 0.5 at the last (a (B,) float tensor)"""
    return 2.0 ** (1.0 - 2.0 * cyc.float() / (NCYC - 1))


def nb_tensors(arrs):
    import torch
    return tuple(torch.as_tensor(a) for a in arrs)                                     # (PHPTR: a 6th array, the prev mask)


def batch(its, states, tree, rng):
    """TR.batch + (BT_ROLE) the tree's role feats per word (.cache/gsmb_role_feats.npy, laid out as the states memmap)"""
    import torch
    t, E = TR.batch(its, states, tree, rng)
    if ROLE:
        r = np.zeros(t["pad"].shape + (14,), np.float32)
        for b, it in enumerate(its): n = len(it["toks"]); r[b, :n] = RF[it["st0"]:it["st0"] + n]
        t["role"] = torch.as_tensor(r)
    if NUF:   # (UNIT-BIND) fresh word phases per problem from a generator seeded by the batch rng's state (read, never consumed)
        ur = np.random.default_rng((rng.bit_generator.state["state"]["state"], 1105))
        ux = np.zeros(t["pad"].shape + (NUF,), np.float32)
        for b, it in enumerate(its):
            U, KD = GU.token_units(it); n = len(it["toks"])
            th = {w: ur.random(PU) * 2 * math.pi for w in sorted({w for u in U if u for w in u})}
            rt, rs = ur.random((n, PU)) * 2 * math.pi, ur.random((K, PU)) * 2 * math.pi   # (placebo) independent phases
            ux[b, :n] = unit_feats(U, KD, th, rt if UNIT == "2" else None)
            E[b]["uU"], E[b]["uth"], E[b]["urs"] = U, th, rs
        t["role"] = torch.cat([t["role"], torch.as_tensor(ux)], -1)
    if PHPTR: t["qn"] = torch.as_tensor(np.array([qnoun(it) for it in its], np.int64))   # (PHPTR) the question noun
    if ZQ:   # (ZQ) the Tree's ST64 chart per position (NCH - 1 = none)
        global CH
        if CH is None: CH = np.load(GD.V1.C("gsmb_chart64_rb8.npy"), mmap_mode="r")
        ch = np.full(t["pad"].shape, NCH - 1, np.int64)
        for b, it in enumerate(its): n = len(it["toks"]); ch[b, :n] = CH[it["st0"]:it["st0"] + n]
        t["ch"] = torch.as_tensor(ch)
    return t, E


def train_step(net, opt, its, states, tree, rng):
    import torch, torch.nn.functional as F
    t, E = batch(its, states, tree, rng); L = t["pad"].shape[1]
    for b, it in enumerate(its): E[b]["vals"] = it["vals"]
    progs, gt = gold(E)
    m = np.array([e["m"] for e in E]); rr = rng.random(len(E))[:, None]
    masked = rng.random((len(E), K)) < rr
    masked[np.arange(len(E)), rng.integers(0, np.minimum(m + 1, K))] = True                   # at least one supervised slot masked
    if SEAL: t["dto"] = torch.as_tensor(DTO.dto_ids(its, progs, ~masked))                  # (SEAL) the gold operands' DTO; slots known iff committed
    nb = nb_tensors(notebook(E, L, progs, ~masked, gt, np.ones((len(E), K), np.float32)))
    xq = net.read(t); gtt = torch.as_tensor(gt)
    cyc = torch.as_tensor(np.clip(np.floor(NCYC * (1 - rr[:, 0])), 0, NCYC - 1).astype(np.int64))   # the cycle this mask level imitates
    sp = None
    if SOFT and rng.random() < 0.5:                                                           # self-conditioning: the model's own detached belief
        with torch.no_grad():
            l0 = net.breath(t, xq, nb, gtt, cyc)[0]
            sp = (l0 / TAU(cyc)[:, None, None]).softmax(-1)
    logit, pa, pb, z, _ = net.breath(t, xq, nb, gtt, cyc, sp)
    mk = torch.as_tensor(masked); am = t["am"] & mk
    lo = F.cross_entropy(logit[mk], gtt[mk])                                                  # gravity: own atom vs the other 36
    for p, V in ((pa, t["va"]), (pb, t["vb"])):
        lo = lo + (p.logsumexp(-1) - p.masked_fill(~V, -1e4).logsumexp(-1))[am].mean()
    if KINDCE:   # BT7b: the same multi-valid CE with every kind open, so P(kind | pointer) is learned (BP factor A reads it)
        for p, V in zip(net.ptrs(gtt, free=True), (t["va"], t["vb"])):
            lo = lo + (p.logsumexp(-1) - p.masked_fill(~V, -1e4).logsumexp(-1))[am].mean()
    opt.zero_grad(); lo.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():                                                                     # Welford: the atoms follow the silhouettes
        zz, tt = z[mk].detach(), gtt[mk]
        cnt = torch.bincount(tt, minlength=NT).float(); sm = torch.zeros(NT, DB).index_add_(0, tt, zz)
        nn_ = net.Cn + cnt; upd = cnt > 0                                                     # Chan's merge of a batch into the mean
        net.C[upd] += (sm[upd] - cnt[upd, None] * net.C[upd]) / nn_[upd, None]
        net.Cn.copy_(nn_.clamp(max=NCAP))                                                     # capped: the atom keeps following
        if FACT:   # (BT8) the factor atoms follow the UNBOUND gold silhouettes (stop slots feed the op codebook only)
            u = net.unbind(zz); st_ = tt != STOP
            if CHAIN:   # (BT8b) the kinds unbound under the gold type's own keys
                u = net.unbind_gold(zz, tt)
            for F_, Fn, ur, idx_, sel in ((net.Fop, net.Fopn, u[0], torch.as_tensor(PARENT)[tt], torch.ones_like(st_)),
                                          (net.Fa, net.Fan, u[1], torch.as_tensor(FKA)[tt], st_), (net.Fb, net.Fbn, u[2], torch.as_tensor(FKB)[tt], st_)):
                ii, uu = idx_[sel], ur[sel]; cnt = torch.bincount(ii, minlength=len(Fn)).float()
                sm = torch.zeros(len(Fn), DB).index_add_(0, ii, uu); nn_ = Fn + cnt; upd = cnt > 0
                F_[upd] += (sm[upd] - cnt[upd, None] * F_[upd]) / nn_[upd, None]; Fn.copy_(nn_.clamp(max=NCAP))
        if ZOOM:
            to = torch.as_tensor(PARENT)[tt]; cnt = torch.bincount(to, minlength=NOP).float(); sm = torch.zeros(NOP, DB).index_add_(0, to, zz)
            nn_ = net.Ccn + cnt; upd = cnt > 0
            net.Cc[upd] += (sm[upd] - cnt[upd, None] * net.Cc[upd]) / nn_[upd, None]; net.Ccn.copy_(nn_.clamp(max=NCAP))
    return float(lo.detach())


def seal_decide(net, its, progs, com, logit, pa, pb, dec):
    """(SEAL) the text's struct, then the op from the DTO of the operands the pointers chose -> type ids, confidence p(struct) p(op)"""
    import torch
    B_ = logit.shape[0]; st_ = SOF[logit.argmax(-1).numpy()]; lps = logit.max(-1).values.numpy()
    ia, ib = pa.argmax(-1).numpy(), pb.argmax(-1).numpy()
    cand = [[(dec(ia[b, k]), dec(ib[b, k])) for k in range(K)] for b in range(B_)]
    lop = net.ophead(torch.as_tensor(DTO.dto_ids(its, progs, com, cand))).numpy(); step = st_ != NS - 1
    return np.where(step, 9 * lop.argmax(-1) + st_, STOP), np.exp(lps + np.where(step, lop.max(-1), 0)).astype(np.float32)


def score(net, its, states, tree, seed=0, noise=None):
    """noise = (tau, rng): (ledger 11-10 SAMPLE) every commit choice (plan op, type, pointers) is argmax(logit / tau + Gumbel), a draw
    from the model's own distribution at tau; the commit ORDER follows the drawn choices' probabilities. None = the greedy scorer."""
    import torch
    gum = (lambda a: a) if noise is None else (lambda a: a / noise[0] - np.log(-np.log(noise[1].uniform(1e-12, 1.0, a.shape))))
    res = dict(ans=[], prog=[], step_op=[], step_ptr=[], step_all=[], first_all=[], type_ok=[], num_for_slot=[], commit_cycle=[])
    if SEAL: res["op_gold"] = []                                                             # (SEAL) the head's op on the GOLD operands
    res["pred"], res["val"], res["progp"], res["ll"] = [], [], [], []                         # (11-10 SAMPLE / SELECT) the vote's inputs
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = batch(part, states, tree, np.random.default_rng(10_000 * seed + s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            xq = net.read(t)
            com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
            IA = np.zeros((B, K), np.int64); IB = np.zeros((B, K), np.int64); cyc = np.full((B, K), -1)
            progs = [[(".", None, None)] * K for _ in range(B)]
            dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
            gt0 = gold(E)[1]
            sp = None
            if MACRO_PLAN:
                assert not (GATE or SEAL or SOFT or PHPTR or ORACLE), "MACRO_PLAN: the schedule form only"
                opk = np.zeros((B, K), bool); opid = np.zeros((B, K), np.int64); orad = np.zeros((B, K), np.float32)
            for c in range(NCYC):
                arrs = notebook(E, L, progs, com, typ, rad)
                nb = nb_tensors((*arrs, opk.copy(), opid.copy(), orad.copy()) if MACRO_PLAN else arrs)
                logit, pa, pb, _, p = net.breath(t, xq, nb, torch.as_tensor(gt0) if ORACLE else None, torch.full((B,), c), sp)
                if SOFT and c + 1 < NCYC: sp = (logit / TAU(torch.full((B,), c + 1))[:, None, None]).softmax(-1)   # next cycle reads it
                if MACRO_PLAN and c < MACRO_PLAN:                                                # THE PLAN: commit ops
                    po = p.numpy() @ _PAR1H; oy = gum(np.log(np.maximum(po, 1e-30))).argmax(-1); oc = np.take_along_axis(po, oy[..., None], -1)[..., 0]; target = math.ceil(K * (c + 1) / MACRO_PLAN)
                    for b in range(B):
                        for k in [k for k in np.argsort(-oc[b]) if not opk[b, k]][:max(target - opk[b].sum(), 0)]:
                            opk[b, k] = True; opid[b, k] = oy[b, k]; orad[b, k] = oc[b, k]; cyc[b, k] = c
                    continue
                ty = gt0 if ORACLE else gum(logit.numpy()).argmax(-1)
                pt = p.gather(-1, torch.as_tensor(ty)[..., None])[..., 0].numpy() if (ORACLE or noise is not None) else p.max(-1).values.numpy()
                if SEAL: ty, pt = seal_decide(net, part, progs, com, logit, pa, pb, dec)
                ia, ib = gum(pa.numpy()).argmax(-1), gum(pb.numpy()).argmax(-1)
                ca = np.take_along_axis(pa.softmax(-1).numpy(), ia[..., None], -1)[..., 0]; cb = np.take_along_axis(pb.softmax(-1).numpy(), ib[..., None], -1)[..., 0]
                conf = np.where(ty == STOP, pt, pt * ca * cb)                                     # the radius of each proposal
                target = math.ceil(K * (c + 1 - MACRO_PLAN) / (NCYC - MACRO_PLAN))
                for b in range(B):
                    if GATE:   # BT6: the confidence dial decides WHEN; the last cycle forces every remaining slot
                        cand = [k for k in np.argsort(-conf[b]) if not com[b, k] and conf[b, k] >= float(GATE) * (1 - c / (NCYC - 1))]
                    else:
                        need = target - com[b].sum()
                        if need <= 0: continue
                        cand = [k for k in np.argsort(-conf[b]) if not com[b, k]][:need]
                    for k in cand:
                        com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]; IA[b, k], IB[b, k] = ia[b, k], ib[b, k]; cyc[b, k] = c
                        if MACRO_PLAN: opk[b, k] = True; opid[b, k] = PARENT[ty[b, k]]
                        op = TYPES[ty[b, k]][0] if ty[b, k] != STOP else "."
                        progs[b][k] = (op, dec(ia[b, k]), dec(ib[b, k])) if op != "." else (".", None, None)
            gprogs, gt = gold(E)
            if SEAL: og = net.ophead(torch.as_tensor(DTO.dto_ids(part, gprogs, np.ones((B, K), bool)))).argmax(-1).numpy()
            for b, (it, e) in enumerate(zip(part, E)):
                prog = []
                for st in progs[b]:
                    if st[0] == ".": break
                    prog.append(st)
                v = GD.execute(prog, it["vals"] + [None] * (L - len(it["vals"])))
                res["ans"].append(v is not None and GD.V1.near(v, it["final"])); res["val"].append(v); res["progp"].append(prog)
                res["ll"].append(float(np.log(np.maximum(rad[b], 1e-30)).sum()))   # the draw's log-likelihood: sum of log commit radii (type x pointers)
                mm = e["m"]; okall = len(prog) == mm
                for k in range(mm):
                    o = progs[b][k][0] == gprogs[b][k][0]
                    ptk = bool(t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]])
                    res["step_op"].append(o); res["step_ptr"].append(ptk); res["step_all"].append(o and ptk)
                    res["pred"].append((progs[b][k][0], int(IA[b, k]), int(IB[b, k])))
                    res["type_ok"].append(typ[b, k] == gt[b, k]); res["commit_cycle"].append(int(cyc[b, k]))
                    if SEAL: res["op_gold"].append(GD.OPS[og[b, k]] == gprogs[b][k][0])
                    okall &= o and ptk
                    if k == 0: res["first_all"].append(o and ptk)
                    for V, I in ((t["va"], IA), (t["vb"], IB)):
                        Vb = V[b, k].numpy()
                        if Vb[L:L + K].any() and not Vb[:L].any(): res["num_for_slot"].append(bool(I[b, k] < L))
                res["prog"].append(bool(okall))
    return {k: (v if k in ("pred", "val", "progp", "ll") else np.asarray(v)) for k, v in res.items()}


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Dt, states = TR.load(); tree = TR.load_tree(); train = Dt["TRAIN"]
    RF = np.load(GD.V1.C(rolef_name()), mmap_mode="r") if ROLE else None   # the role feats follow the states (ledger 11-03)
    net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=LR); npar = sum(p.numel() for p in net.parameters())
    print(f"[bt] {TAG} D {D} reader {NLE} + breath {NLB} (weight-tied x {NCYC} cycles) silhouette {DB} atoms {NT} | params {npar}", flush=True)
    t0 = time.time(); curve = []
    if LOAD: net.load_state_dict(torch.load(LOAD)); STEPS = 0; print(f"[bt] {TAG} loaded {LOAD} (oracle type {ORACLE})", flush=True)
    for step in range(1, STEPS + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), BATCH)]
        lo = train_step(net, opt, its, states, tree, rng)
        if step % 1000 == 0 or step == STEPS:
            net.eval(); r = {k: score(net, Dt[k][:600], states, tree) for k in ("ORIG", "TEST")}; net.train()
            curve.append(dict(step=step, loss=lo, **{f"{k}_{m}": float(v[m].mean()) for k, v in r.items() for m in ("ans", "step_all")}))
            print(f"[bt] {TAG} step {step} loss {lo:.4f} " + " ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f} "
                  f"type {v['type_ok'].mean():.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    if os.environ.get("BT_SAVE") == "1": torch.save(net.state_dict(), f".cache/gsmb_breath_{TAG}.pt")
    net.eval(); R = {k: score(net, Dt[k], states, tree) for k in SPLITS}
    out = dict(tag=TAG, arm="BT", seed=SEED, d=D, params=npar, steps=STEPS, curve=curve, oracle=ORACLE, load=LOAD,
               **{f"{k}_{m}": float(v[m].mean()) for k, v in R.items() for m in v if m not in ("commit_cycle", "pred", "val", "progp", "ll")},
               **{f"{k}_commit_share": np.bincount(v["commit_cycle"][v["commit_cycle"] >= 0], minlength=NCYC).tolist() for k, v in R.items()},
               **{f"{k}_per": [int(x) for x in v["ans"]] for k, v in R.items()},
               atoms_C=net.C.tolist(), atoms_n=net.Cn.tolist())
    json.dump(out, open(f".cache/gsmb_breath_{TAG}.json", "w"))
    print(f"[bt] {TAG} FINAL " + " | ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f} op {v['step_op'].mean():.3f} "
          f"ptr {v['step_ptr'].mean():.3f} type {v['type_ok'].mean():.3f}" for k, v in R.items()), flush=True)
