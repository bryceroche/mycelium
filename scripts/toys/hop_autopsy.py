"""hop_autopsy.py -- T2h autopsy (2026-10-23): retrain HOP2_I seed 0 (T2h recipe) and slice the OUT record's owner decode
by how the giver is written (first / name / desc / pronoun) and by verb (GIVES / LOSES); for GIVES, how often the decode
lands on the RECIPIENT. Diagnostic; no bar."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import hop_query_toy as HQ
BQ, SS = HQ.BQ, HQ.SS
torch.set_num_threads(16)
orig = BQ.episode
def episode(rng):                                       # same draws; also keep the OUT record's form / verb / recipient
    ex = orig(rng); o = ex["ops"][1]; ex["meta"] = (o[5], o[6]); return ex
BQ.episode = episode
_b = BQ.batch
def batch(rng, B, erase=False):
    st = rng.bit_generator.state; bt = _b(rng, B, erase); rng.bit_generator.state = st
    bt["meta"] = [episode(rng)["meta"] for _ in range(B)]; return bt
BQ.batch = batch
net = None
def run_keep(seed=0):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(90_000 + seed)
    ev = BQ.batch(erng, 6000)
    torch.manual_seed(seed); n = HQ.HopNet("I")
    opt = torch.optim.AdamW(n.parameters(), lr=SS.LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SS.LR, total_steps=SS.STEPS, pct_start=0.1)
    for _ in range(SS.STEPS):
        bt = BQ.batch(rng, SS.BATCH); lp, aux = n(bt); tg = bt["tgt"]; m = tg >= 0
        loss = torch.nn.functional.nll_loss(lp[m], tg[m]) + aux
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0); opt.step(); sch.step()
    return n, ev
n, ev = run_keep(0); n.eval()
with torch.no_grad():
    lp, _ = n(ev); dec = n._dl.argmax(-1); tg = ev["tgt"]; ar = torch.arange(len(tg))
    own = ev["rfeat"][ar, tg[:, 1], 0]
c = collections.defaultdict(lambda: [0, 0, 0])
for b, (form, rec) in enumerate(ev["meta"]):
    verb = "GIVES" if rec >= 0 else "LOSES"
    for key in (("form", form), ("verb", verb), ("both", form + "/" + verb), ("all", "")):
        c[key][0] += 1; c[key][1] += int(dec[b] == own[b]); c[key][2] += int(rec >= 0 and dec[b] == rec)
print("OUT-record owner decode (seed 0, 6000 eval episodes): n / right / lands-on-recipient")
for k in sorted(c):
    n_, r, rc = c[k]; print(f"  {k[0]:5s} {k[1]:16s} n {n_:5d}  right {r / n_:.3f}  recipient {rc / n_:.3f}")
