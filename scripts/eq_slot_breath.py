"""eq_slot_breath.py -- THE EQ_SLOT TASK, the breathing transformer's first real-data task (ledger 2026-10-27; CPU torch).
The order table's job learned: read the problem (the READER's cached states, never re-encoded) and assign each equation given
slot (derivation order) one of the problem's located given mentions (text order). Graded THROUGH THE SOLVER on the wild
holdout (scripts/eq_slot_grade.py: the gold graph with the assigned values must solve to the custody key).
INPUT ENCODING (both arms): reader states X (T_ALG x 2048, pads masked) -> LN -> Linear D. Candidates = X at each mention's
first token + a T6 stamp of its TEXT rank; slots = a learned base + a T6 stamp of the DERIVATION rank (the clock as time:
text time in, plan time out). Pointer logits slot x candidate at every block; loss = CE averaged over blocks (both arms).
ARMS:
  BREATH   ONE shared block applied NB times (slot self-attn -> RE-READ cross-attn over ALL text tokens -> FF), a T6 BREATH
           stamp added each breath, and THE SNAP: a straight-through one-hot of each slot's pointer argmax re-injects the
           chosen candidate (the notebook commits; T3d: the snap's job grows with depth)
  BUS      BREATH with the RE-READ replaced by THE T256 BUS (T3e2's winner on the toy): the text is bound ONCE into a PB-plane
           phasor vector, s = sum_t exp(i k(x_t)) * rho(exp(i v(x_t))) / sqrt(n); each breath unbinds rho^-1(conj(exp(i q(S))) * s)
           (the candidates' own states still feed the pointer: the bus replaces only the context read)
  NOMLP    BREATH with the block's feed-forward REMOVED: each breath = slot self-attn -> re-read -> snap -> clock -> pointer
           (Bryce's breathing transformer is MHA without the MLP; T3d: no free MLP across planes; ledger 2026-10-27)
  VANILLA  NB DISTINCT blocks (same block form, NB x the block params), no breath stamp, no snap = the compute-matched
           standard decoder
Decode: Hungarian over the final pointer (one mention per slot). Envs: EQ_ARM, EQ_SEED, EQ_READER (BP|STU), EQ_STEPS, EQ_NB,
EQ_D, EQ_BATCH, EQ_LR, EQ_TAG. Output .cache/eq_slot_{TAG}.json + .pt
THE PROBLEM-DISJOINT RERUN (ledger 2026-10-27, word given): EQ_SPLIT=problem cuts val by PROBLEM KEY (gen.src_idx / gen.pool, else
the digit-masked text: resampled variants share it), EQ_CAP=N keeps <= N train rows per key (lowest sha1(text_sha) first),
EQ_HOLD_CURVE=1 logs the holdout exact-order beside each VAL line (a read; never used for selection). Defaults = the 10-27 runs.
EQ_FAST=1 (perf, same math): each batch trimmed to its longest text (pads are masked keys), and the shared block's re-read
projects the text's keys/values ONCE per forward instead of once per breath (BREATH/NOMLP; X is the same every breath)."""
import hashlib, json, math, os, re, sqlite3, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_grade as EG
from mycelium import rotor_clock as CK

ARM = os.environ.get("EQ_ARM", "BREATH"); SEED = int(os.environ.get("EQ_SEED", "0")); READER = os.environ.get("EQ_READER", "BP")
STEPS = int(os.environ.get("EQ_STEPS", "20000")); NB = int(os.environ.get("EQ_NB", "8")); D = int(os.environ.get("EQ_D", "256"))
BATCH = int(os.environ.get("EQ_BATCH", "32")); LR = float(os.environ.get("EQ_LR", "3e-4")); NH = 8
PB = int(os.environ.get("EQ_PB", "256"))   # BUS planes (T3e2: 64 fails held rules, 256 does not)
TAG = os.environ.get("EQ_TAG", f"{ARM}_{READER}_s{SEED}")
FAST = os.environ.get("EQ_FAST") == "1"; SPLIT = os.environ.get("EQ_SPLIT", "text"); CAP = int(os.environ.get("EQ_CAP", "0")); HOLD_CURVE = os.environ.get("EQ_HOLD_CURVE") == "1"


def problem_key(smp):   # the row's PROBLEM: the source index when the mix carries one, else the digit-masked text
    g = smp.get("gen")
    if isinstance(g, dict) and "src_idx" in g: return f"{g.get('src')}:{g['src_idx']}"
    if isinstance(g, dict) and "pool" in g: return f"{g.get('src')}:pool:{json.dumps(g['pool'], sort_keys=True)}"
    return "mask:" + re.sub(r"\d+(\.\d+)?", "#", smp["text"].strip())
T_ALG, H_TRUNK, KMAX = 256, 2048, 14
TOK = ".cache/llama-3.2-1b-weights/tokenizer.json"
NPY, NPZ = ".cache/phase1_alg_states_{}_states.npy", ".cache/phase1_alg_states_{}.npz"
SPLIT_STATES = {"train": f"formpm35c{READER}", "holdout": f"wildhold{READER}"}
CK.check_range(CK.T6_SPEEDS, max(KMAX, NB) + 1)   # THE CLOCK-RANGE RULE over the longest stamped span


def stamp(n):   # T6 stamp of integer ticks 0..n-1: cos/sin of the six gears -> (n, 12)
    ph = np.stack([CK.t6_phases(t, 6) for t in range(n)])
    return np.concatenate([np.cos(ph), np.sin(ph)], -1).astype(np.float32)


def build(split):
    """Task rows with token-level candidates. Rows: located givens >= 1, <= KMAX; the train row sharing text with the holdout
    is fenced out."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK); c = sqlite3.connect(EG.DB)
    fence = set(r[0] for r in c.execute("select text_sha from order_rows where split='holdout'")) if split == "train" else set()
    by = {}
    for r, eq, cs, loc in c.execute("select row,eq_slot,char_start,locate from order_map where split=? order by row,eq_slot", (split,)):
        by.setdefault(r, []).append((eq, None if loc == "unstated" else cs))
    sha = {r: s for r, s in c.execute("select row,text_sha from order_rows where split=?", (split,))}
    tm = np.load(NPZ.format(SPLIT_STATES[split]))["tokmask"]
    out = []
    for i, line in enumerate(open(EG.MIX[split])):
        if i not in by or sha[i] in fence: continue
        st = [s for s in by[i] if s[1] is not None]
        if not 1 <= len(st) <= KMAX: continue
        smp = json.loads(line); e = tok.encode(smp["text"]); assert int(tm[i].sum()) == len(e.ids), (split, i)
        offs = np.asarray(e.offsets)
        cands = sorted(s[1] for s in st)
        ctok = [int(np.nonzero(offs[:, 1] > cs)[0][0]) for cs in cands]
        tgt = [cands.index(s[1]) for s in st]                          # stated slot rank r -> candidate index (text rank)
        vkey = problem_key(smp) if SPLIT == "problem" else sha[i]
        out.append(dict(row=i, n=len(st), ctok=ctok, tgt=tgt, eq=[s[0] for s in st], cands=cands, pkey=problem_key(smp),
                        hsh=hashlib.sha1(sha[i].encode()).hexdigest(), val=int(hashlib.sha1(vkey.encode()).hexdigest(), 16) % 100 == 0))
    if CAP and split == "train":   # <= CAP train rows per problem (val rows are whole problems under EQ_SPLIT=problem; kept)
        seen = {}; keep = []
        for r in sorted(out, key=lambda r: r["hsh"]):
            if r["val"] or seen.get(r["pkey"], 0) < CAP:
                keep.append(r); seen[r["pkey"]] = seen.get(r["pkey"], 0) + (not r["val"])
        out = sorted(keep, key=lambda r: r["row"])
    return out, tm


def make_net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
            self.sa = nn.MultiheadAttention(D, NH, batch_first=True); self.ca = nn.MultiheadAttention(D, NH, batch_first=True)
            if ARM != "NOMLP":   # NOMLP: Bryce's breathing transformer = MHA WITHOUT THE MLP (10-24 notes; the T3d build rule)
                self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
            self.pq, self.pk = nn.Linear(D, D), nn.Linear(D, D)
            if ARM == "BUS":
                self.bq, self.bo = nn.Linear(D, PB), nn.Linear(2 * PB, D)
                gp = torch.Generator().manual_seed(7); pr = torch.randperm(PB, generator=gp)
                self.register_buffer("rho_inv", pr.argsort())

        def text_kv(self, X):   # EQ_FAST: the re-read's keys/values, (B, NH, T, D/NH), projected once per forward
            B, T, _ = X.shape; _, wk, wv = self.ca.in_proj_weight.chunk(3); _, bk, bv = self.ca.in_proj_bias.chunk(3)
            return tuple(F.linear(X, w, b).view(B, T, NH, D // NH).transpose(1, 2) for w, b in ((wk, bk), (wv, bv)))

        def forward(self, S, smask, X, xmask, C, cmask, bus=None, kv=None):
            h = self.n1(S); S = S + self.sa(h, h, h, key_padding_mask=~smask, need_weights=False)[0]
            if kv is not None:   # THE RE-READ, the cached-projection form (same math as self.ca)
                h = self.n2(S); B, K, _ = h.shape; wq = self.ca.in_proj_weight[:D]; bq = self.ca.in_proj_bias[:D]
                q = F.linear(h, wq, bq).view(B, K, NH, D // NH).transpose(1, 2)
                o = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=xmask[:, None, None, :])
                S = S + self.ca.out_proj(o.transpose(1, 2).reshape(B, K, D))
            elif bus is None:
                h = self.n2(S); S = S + self.ca(h, X, X, key_padding_mask=~xmask, need_weights=False)[0]   # THE RE-READ
            else:   # THE BUS: unbind the one-shot text vector with this breath's question, rho^-1(conj(q) * s)
                u = (torch.polar(torch.ones(()), -self.bq(self.n2(S))) * bus[:, None])[..., self.rho_inv]
                S = S + self.bo(torch.cat([u.real, u.imag], -1))
            if ARM != "NOMLP":
                S = S + self.ff(self.n3(S))
            lg = torch.einsum("bkd,bjd->bkj", self.pq(S), self.pk(C)) / math.sqrt(D)
            return S, lg.masked_fill(~cmask[:, None, :], -1e4)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.xn = nn.LayerNorm(H_TRUNK); self.xp = nn.Linear(H_TRUNK, D)
            self.register_buffer("stk", torch.from_numpy(stamp(max(KMAX, NB))))
            self.cst, self.sst = nn.Linear(12, D), nn.Linear(12, D); self.s0 = nn.Parameter(torch.zeros(D))
            self.blocks = nn.ModuleList([Block() for _ in range(1 if ARM in ("BREATH", "BUS", "NOMLP") else NB)])
            if ARM in ("BREATH", "BUS", "NOMLP"):
                self.bst = nn.Linear(12, D); self.snap = nn.Linear(D, D)
            if ARM == "BUS":
                self.bk, self.bv = nn.Linear(D, PB), nn.Linear(D, PB)
                gp = torch.Generator().manual_seed(7); self.register_buffer("rho", torch.randperm(PB, generator=gp))

        def forward(self, X, xmask, ctok, cmask, smask):
            X = self.xp(self.xn(X)); B = X.shape[0]
            C = torch.gather(X, 1, ctok[..., None].expand(-1, -1, D)) + self.cst(self.stk[:KMAX])[None]
            S = (self.s0 + self.sst(self.stk[:KMAX]))[None].expand(B, -1, -1)
            lgs = []; bus = None
            kv = self.blocks[0].text_kv(X) if FAST and ARM in ("BREATH", "NOMLP") else None
            if ARM == "BUS":   # THE HANDOFF, once: every text token bound key (x) rho(value) into ONE PB-plane phasor vector
                one = torch.ones(()); m = xmask[..., None].float()
                bus = ((torch.polar(one, self.bk(X)) * torch.polar(one, self.bv(X))[..., self.rho]) * m).sum(1) / m.sum(1).sqrt()
            for b in range(NB):
                if ARM in ("BREATH", "BUS", "NOMLP"):
                    S = S + self.bst(self.stk[b])
                    S, lg = self.blocks[0](S, smask, X, xmask, C, cmask, bus, kv)
                    p = lg.softmax(-1); hard = F.one_hot(p.argmax(-1), KMAX).float() + p - p.detach()   # THE SNAP
                    S = S + self.snap(hard @ C)
                else:
                    S, lg = self.blocks[b](S, smask, X, xmask, C, cmask)
                lgs.append(lg)
            return torch.stack(lgs, 1)   # (B, NB, KMAX slots, KMAX cands)

    return Net()


def batch(rows, idx, mm):
    import torch
    rs = [rows[i] for i in idx]; order = np.argsort([r["row"] for r in rs]); rs = [rs[i] for i in order]
    mm, tm = mm; ri = [r["row"] for r in rs]
    X = torch.from_numpy(np.asarray(mm[ri], dtype=np.float32))
    xmask = torch.from_numpy(tm[ri] > 0)                               # the tokenizer's mask (teacher pads are not zero)
    ctok = torch.zeros(len(rs), KMAX, dtype=torch.long); cmask = torch.zeros(len(rs), KMAX, dtype=torch.bool)
    tgt = torch.full((len(rs), KMAX), -100, dtype=torch.long)
    for i, r in enumerate(rs):
        ctok[i, :r["n"]] = torch.tensor(r["ctok"]); cmask[i, :r["n"]] = True; tgt[i, :r["n"]] = torch.tensor(r["tgt"])
    if FAST:   # trim to the batch's longest text: every key beyond it is a masked pad; every mention token lies inside it
        L = int(xmask.sum(1).max()); X, xmask = X[:, :L].contiguous(), xmask[:, :L].contiguous()
    return rs, X, xmask, ctok, cmask, cmask.clone(), tgt


def decode(lg, n):
    from scipy.optimize import linear_sum_assignment
    _, col = linear_sum_assignment(-lg[:n, :n]); return list(col)


def evaluate(net, rows, mm, tasks=None):
    """exact-order accuracy per block (the breath curve) + solver-graded accuracy on the final block (if tasks given)."""
    import torch
    net.eval(); ex = np.zeros(NB); solv = 0; per_row = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 64):
            rs, X, xm, ct, cm, sm, tg = batch(rows, range(s0, min(s0 + 64, len(rows))), mm)
            lg = net(X, xm, ct, cm, sm).numpy()
            for i, r in enumerate(rs):
                for b in range(NB):
                    ex[b] += decode(lg[i, b], r["n"]) == r["tgt"]
                if tasks is not None:
                    col = decode(lg[i, -1], r["n"]); ok = EG.grade(tasks[r["row"]], {r["eq"][k]: r["cands"][col[k]] for k in range(r["n"])})
                    solv += ok; per_row.append((r["row"], int(ok)))
    net.train()
    return ex / len(rows), (solv / len(rows) if tasks is not None else None), per_row


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    t0 = time.time()
    tr, tm_tr = build("train"); ho, tm_ho = build("holdout")
    mm_tr = (np.load(NPY.format(SPLIT_STATES["train"]), mmap_mode="r"), tm_tr)
    mm_ho = (np.load(NPY.format(SPLIT_STATES["holdout"]), mmap_mode="r"), tm_ho)
    trn = [r for r in tr if not r["val"]]; val = [r for r in tr if r["val"]]
    T_ho, _ = EG.rows("holdout"); T_ho = {t["row"]: t for t in T_ho}
    ho = [r for r in ho if r["row"] in T_ho]
    net = make_net(); npar = sum(p.numel() for p in net.parameters())
    print(f"[eq_slot] {TAG} arm {ARM} reader {READER} NB {NB} D {D} params {npar:,} | train {len(trn)} val {len(val)} holdout {len(ho)} "
          f"| built in {time.time() - t0:.0f}s", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    t1 = time.time()
    for step in range(STEPS):
        _, X, xm, ct, cm, sm, tg = batch(trn, rng.integers(0, len(trn), BATCH), mm_tr)
        lg = net(X, xm, ct, cm, sm)
        loss = F.cross_entropy(lg.reshape(-1, KMAX), tg[:, None].expand(-1, NB, -1).reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == STEPS - 1:
            msg = f"[eq_slot] step {step} loss {loss.item():.4f} | {(time.time() - t1) / (step + 1):.3f}s/step"
            if step % 5000 == 0 or step == STEPS - 1:
                ex, _, _ = evaluate(net, val, mm_tr); msg += f" | VAL exact-order final {ex[-1]:.3f} (per block {' '.join(f'{x:.2f}' for x in ex)})"
                if HOLD_CURVE:
                    eh, _, _ = evaluate(net, ho, mm_ho); msg += f" | HOLD exact-order final {eh[-1]:.3f} (block1 {eh[0]:.3f})"
            print(msg, flush=True)
    ex, sv, per_row = evaluate(net, ho, mm_ho, T_ho)
    print(f"[eq_slot] HOLDOUT {TAG}: solver {sv:.4f} | exact-order final {ex[-1]:.3f} | per block {' '.join(f'{x:.3f}' for x in ex)} "
          f"| n {len(ho)} | {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(tag=TAG, arm=ARM, reader=READER, seed=SEED, nb=NB, d=D, steps=STEPS, params=npar, split=SPLIT, cap=CAP, n_train=len(trn), holdout_solver=sv,
                   holdout_exact=ex.tolist(), per_row=per_row), open(f".cache/eq_slot_{TAG}.json", "w"))
    torch.save(net.state_dict(), f".cache/eq_slot_{TAG}.pt")
