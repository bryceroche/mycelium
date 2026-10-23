"""clock_coref_toy.py -- T2m, THE T6 CLOCK AS THE TIME KEEPER, GEARS BUILT IN (2026-10-23; Bryce's word "word given for T6
clock which should have the gears built in"; ledger 16:47 "T2m"). CPU, PyTorch, no text encoder: the READING is given (slots,
their tree type node, which mention is a pronoun); only the TIME read is under test.
The register is GSM8K-shaped (the 16:27 check: protagonist 0.753, recency 0.680; the previous-subject rule 0.327): a pronoun's
antecedent is THE PROTAGONIST (the earliest-created slot) if type-compatible, else the MOST RECENTLY MENTIONED compatible slot.
Both are TIME reads: protagonist = the oldest creation stamp, recency = the youngest last-mention stamp. Each slot carries two
clock stamps (creation tick, last-mention tick; tick = sentence index, offset by 0-2 entity-free preamble sentences) and the
resolver reads them RELATIVE TO NOW (stamp unbound by the current tick: clock(s) * conj(clock(now)) = clock(s - now)).
Arms (one resolver for all: per-slot MLP -> one self-attention layer across the live slots -> score, + learned type-match weight):
  NOCLOCK  type match only (floor)
  HAND     hand relational flags: is-earliest-created, recency rank one-hot (ceiling)
  GEARS    the six waves as GEARS: periods 2,4,8,16,32,64 ticks (ratio 2; fast gears separate ticks, slow gears order them)
  FLAT6    the deployed six-wave as-is: one period-6 wheel read at six 60-degree phases (phase1_algebra_head's SW form)
  BAND     six full-band frequencies (uniform in (0, pi], fixed draw): the 10-16 'hash' clock
  GEARS_BOUND  GEARS with each slot's stamps stored VOLATILE in the slot's own register (theta-gamma: slot = register):
           r_j = role_c * gear(c_j) + role_l * gear(l_j), PB 64, last-mention UPDATED in place (unbind old, bind new) and
           read back by unbind + cleanup against the gear codebook; the recovered ticks feed the GEARS features.
Splits: in-dist (6-16 content sentences) and LONG (20-32; ages never seen in training)."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from mycelium import complex_tensor as CT

mode = "smoke" if "--smoke" in sys.argv else "full"
STEPS = int(os.environ.get("BM_STEPS", "300" if mode == "smoke" else "3000"))
BATCH, LR, D = 256, 2e-3, 64
N_EVAL = 300 if mode == "smoke" else 4000                                  # episodes per eval split
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
ARMS = os.environ.get("CK_ARMS", "NOCLOCK,HAND,GEARS,FLAT6,BAND,GEARS_BOUND").split(",")
OUT_TXT = ".cache/clock_coref{}.txt"
S, NTYPE, TMAX, PB = 6, 4, 40, 64                                          # slots; he/she/they/it; codebook ticks; phasor dims
OMEGA = {"GEARS": 2 * np.pi / np.array([2., 4., 8., 16., 32., 64.]),
         "BAND": np.random.default_rng(616).uniform(1e-3, np.pi, 6)}
_pg = np.random.default_rng(4051)
GEAR_OF = np.arange(PB) % 6                                                # each phasor dim rides one gear
GEAR_CB = CT.phasor(OMEGA["GEARS"][GEAR_OF][None] * np.arange(TMAX + 1)[:, None] + _pg.uniform(0, 2 * np.pi, PB)[None])
ROLE_C, ROLE_L = (CT.phasor(_pg.uniform(0, 2 * np.pi, PB)) for _ in range(2))
RECOV = dict(n=0, ok=0)


def episode(rng, split):
    """-> list of pronoun records: dict(now, c (S,), l (S,), live (S,), tm (S,) type match, tgt, kind, ncomp, reg (S,PB) or None)."""
    T = int(rng.integers(6, 17)) if split == "indist" else int(rng.integers(20, 33))
    pre = int(rng.integers(0, 3)); n = int(rng.integers(3, S + 1))
    types = [int(rng.integers(2))]                                         # the protagonist is a person
    for _ in range(n - 1):
        u = rng.random()
        types.append(types[0] if u < 0.3 else (int(rng.choice(types[1:])) if len(types) > 1 and u < 0.65 else int(rng.integers(NTYPE))))
    birth = [pre] + sorted(int(x) for x in rng.integers(pre + 1, pre + T, n - 1))
    c = np.full(S, -1); l = np.full(S, -1); reg = np.zeros((S, PB), np.complex64); recs = []

    def rule(ty, live):
        comp = [j for j in live if types[j] == ty]
        if not comp:
            return None, None, 0
        if 0 in comp:
            return 0, "protag", len(comp)
        best = max(l[j] for j in comp); top = [j for j in comp if l[j] == best]
        return (top[0], "recency", len(comp)) if len(top) == 1 else (None, None, len(comp))

    def touch(j, t):
        if c[j] < 0:
            c[j] = t; reg[j] = CT.bind(GEAR_CB[t], ROLE_C) + CT.bind(GEAR_CB[t], ROLE_L)
        else:
            reg[j] = reg[j] - CT.bind(GEAR_CB[l[j]], ROLE_L) + CT.bind(GEAR_CB[t], ROLE_L)   # the volatile update
        l[j] = t

    for t in range(pre, pre + T):
        live = [j for j in range(n) if 0 <= c[j] < t]                      # the state BEFORE this sentence
        new = [j for j in range(n) if birth[j] == t and c[j] < 0]
        subj = new[0] if new else (0 if (rng.random() < 0.55 or len(live) < 2) else int(rng.choice(live)))
        obj = None
        if rng.random() < 0.5 and len(live) + len(new) > 1:
            obj = int(rng.choice([j for j in live + new[1:] if j != subj]))
        rest = [j for j in new if j not in (subj, obj)]
        snap = dict(c=c.copy(), l=l.copy(), reg=reg.copy())
        ptypes = []
        for j in (subj, obj):
            if j is None or j not in live:
                continue
            tgt, kind, nc = rule(types[j], live)
            if tgt == j and types[j] not in ptypes and rng.random() < 0.7:
                ptypes.append(types[j])
                recs.append(dict(now=t, c=snap["c"], l=snap["l"], reg=snap["reg"], live=np.array([k in live for k in range(S)]),
                                 tm=np.array([k in live and types[k] == types[j] for k in range(S)]), tgt=j, kind=kind, ncomp=nc))
        for j in (subj, obj, *rest):
            if j is not None:
                touch(j, t)
    return recs


def feats(arm, recs):
    """-> (B,S,F) float features per live slot."""
    B = len(recs); now = np.array([r["now"] for r in recs])[:, None]
    c = np.stack([r["c"] for r in recs]); l = np.stack([r["l"] for r in recs]); live = np.stack([r["live"] for r in recs])
    if arm == "GEARS_BOUND":
        reg = np.stack([r["reg"] for r in recs])
        for st, role in ((c, ROLE_C), (l, ROLE_L)):
            rec = CT.cleanup(CT.unbind(reg, role).reshape(-1, PB), GEAR_CB).reshape(B, S)
            RECOV["n"] += int(live.sum()); RECOV["ok"] += int((rec == st)[live].sum())
            st[live] = rec[live]
    if arm == "NOCLOCK":
        return np.zeros((B, S, 1), np.float32)
    if arm == "HAND":
        BIG = 10 ** 6
        cm = np.where(live, c, BIG); earliest = (cm == cm.min(1, keepdims=True)) & live
        lm = np.where(live, l, -BIG); rank = (lm[:, None, :] > lm[:, :, None]).sum(-1)       # 0 = most recent
        return np.concatenate([earliest[..., None], np.eye(S)[np.clip(rank, 0, S - 1)] * live[..., None]], -1).astype(np.float32)
    om = OMEGA["GEARS" if arm.startswith("GEARS") else arm] if arm != "FLAT6" else None
    out = []
    for st in (c, l):
        age = (now - st).astype(np.float64)                                # = phase of clock(st) * conj(clock(now)), unbound
        if arm == "FLAT6":
            ang = 2 * np.pi * age[..., None] / 6 + np.arange(6) * np.pi / 3   # one period-6 wheel, six 60-degree teeth
            out += [np.cos(ang)]
        else:
            ang = age[..., None] * om
            out += [np.cos(ang), np.sin(ang)]
    return (np.concatenate(out, -1) * live[..., None]).astype(np.float32)


def batch(rng, nep, split, arm):
    recs = [r for _ in range(nep) for r in episode(rng, split)]
    t = lambda v: torch.from_numpy(np.asarray(v))
    x, live, tm = feats(arm, recs), np.stack([r["live"] for r in recs]), np.stack([r["tm"] for r in recs]).astype(np.float32)
    perm = np.argsort(rng.random((len(recs), S)), 1); ar = np.arange(len(recs))[:, None]   # slot order is NOT creation order
    tgt = np.argsort(perm, 1)[ar[:, 0], [r["tgt"] for r in recs]]
    return dict(x=t(x[ar, perm]), live=t(live[ar, perm]), tm=t(tm[ar, perm]),
                tgt=t(tgt), kind=[r["kind"] for r in recs], ncomp=t(np.array([r["ncomp"] for r in recs])))


class Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf + 1, D), nn.GELU(), nn.Linear(D, D))
        self.att = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True)
        self.out = nn.Linear(D, 1); self.wt = nn.Parameter(torch.tensor(3.0))

    def forward(self, bt):
        h = self.att(self.inp(torch.cat([bt["x"], bt["tm"][..., None]], -1)), src_key_padding_mask=~bt["live"])
        return (self.out(h)[..., 0] + self.wt * bt["tm"]).masked_fill(~bt["live"], -1e9)


def evaluate(net, ev):
    out = {}; net.eval()
    with torch.no_grad():
        for sp, bt in ev.items():
            ok = net(bt).argmax(-1) == bt["tgt"]
            out[f"{sp}_acc"] = float(ok.float().mean()); out[f"{sp}_n"] = len(ok)
            for k in ("protag", "recency"):
                m = torch.tensor([x == k for x in bt["kind"]])
                out[f"{sp}_{k}"] = float(ok[m].float().mean()); out[f"{sp}_share_{k}"] = float(m.float().mean())
            m = bt["ncomp"] >= 2; out[f"{sp}_contested"] = float(ok[m].float().mean()); out[f"{sp}_share_contested"] = float(m.float().mean())
    net.train(); return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {sp: batch(erng, N_EVAL, sp, arm) for sp in ("indist", "long")}
    torch.manual_seed(seed); net = Net(ev["indist"]["x"].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = batch(rng, BATCH // 4, "indist", arm)
        loss = F.cross_entropy(net(bt), bt["tgt"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return evaluate(net, ev)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    OUT = OUT_TXT.format("_smoke" if mode == "smoke" else "")
    assert mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} seeds {SEEDS}; BAND periods {np.round(2 * np.pi / OMEGA['BAND'], 2).tolist()}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        a = {kk: float(np.nanmean([x[kk] for x in rs])) for kk in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        for sp in ("indist", "long"):
            say(f"{arm:12s} {sp:6s} | acc {a[f'{sp}_acc']:.3f} | protag {a[f'{sp}_protag']:.3f} (share {a[f'{sp}_share_protag']:.2f}) "
                f"recency {a[f'{sp}_recency']:.3f} | contested {a[f'{sp}_contested']:.3f} (share {a[f'{sp}_share_contested']:.2f}) "
                f"| n {a[f'{sp}_n']:.0f} | seeds {[round(x[f'{sp}_acc'], 3) for x in rs]}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    say(f"GEARS_BOUND stamp recovery by unbind+cleanup: {RECOV['ok']}/{RECOV['n']} = {RECOV['ok'] / max(1, RECOV['n']):.5f}")
    if mode == "full" and all(x in A for x in ("HAND", "GEARS", "FLAT6", "BAND", "GEARS_BOUND")):   # PINNED (ledger 16:47, T2m)
        h, g, f6, bd, gb = (A[k] for k in ("HAND", "GEARS", "FLAT6", "BAND", "GEARS_BOUND"))
        say("VERDICTS (3-seed means)")
        m1 = g["indist_acc"] >= h["indist_acc"] - 0.03
        say(f"M1 the geared clock keeps time (in range): {'PASS' if m1 else 'FAIL'} (in-dist {g['indist_acc']:.3f} vs HAND {h['indist_acc']:.3f})")
        m1l = g["long_acc"] >= h["long_acc"] - 0.03
        say(f"M1L ...and past the trained ages (prediction FAIL): {'PASS' if m1l else 'FAIL'} (long {g['long_acc']:.3f} vs HAND {h['long_acc']:.3f})")
        say(f"M2 predictions: FLAT6 in-dist {f6['indist_acc']:.3f} <= GEARS - 0.05 -> {'HELD' if f6['indist_acc'] <= g['indist_acc'] - 0.05 else 'MISSED'}; "
            f"BAND in-dist {bd['indist_acc']:.3f} within 0.01 of GEARS -> {'HELD' if abs(bd['indist_acc'] - g['indist_acc']) <= 0.01 else 'MISSED'}; "
            f"(read) BAND long {bd['long_acc']:.3f} vs GEARS long {g['long_acc']:.3f}")
        m3 = RECOV["ok"] == RECOV["n"] and abs(gb["indist_acc"] - g["indist_acc"]) <= 0.01 and abs(gb["long_acc"] - g["long_acc"]) <= 0.01
        say(f"M3 lossless volatile stamps: {'PASS' if m3 else 'FAIL'} (recovery {RECOV['ok']}/{RECOV['n']}; in-dist {gb['indist_acc']:.3f} vs "
            f"{g['indist_acc']:.3f}; long {gb['long_acc']:.3f} vs {g['long_acc']:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
