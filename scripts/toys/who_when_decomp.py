"""who_when_decomp.py -- WHERE THE ROUTER MISSES AT N 128 (2026-10-21; diagnostic of the (who, when) rung, Opus's split)
Retrains HYB and CACHE at seed 0 with the full run's swept lr (3e-3; reproduces the banked seed-0 path exactly: same rng
seeds, same steps), saves the state_dicts, and splits the router's anchor hits at N 128 (4000 fresh sequences) by:
  DIST  query-to-anchor distance (ns - anc): <= 64 (seen in training) vs > 64 (never seen), in bins of 16
  C     one sale (content unambiguous) vs two (first/last needed); WHICH first vs last within two-sale sequences
Reading (Opus): misses piling up at DIST > 64 even with ONE sale = position extrapolation (RoPE); misses concentrated in
two-sale sequences at every distance = the first/last judgment. Output .cache/who_when_decomp.txt (mode 'x').
"""
import os, sys
sys.argv = [sys.argv[0], "--feas"]                                           # 4000 steps, seed 0 (the full run's seed-0 path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
import who_when_rung as W  # noqa: E402

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
out = open(".cache/who_when_decomp.txt", "x")
def say(s): print(s, flush=True); out.write(s + "\n"); out.flush()

for arm in ("HYB", "CACHE"):
    seed, lr = 0, 3e-3
    rng = np.random.default_rng(seed * 11 + 3)
    net = W.build(arm, seed)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=W.STEPS, pct_start=0.1)
    for step in range(W.STEPS):
        bt = W.batch(W.BATCH, rng); loss = F.cross_entropy(net(bt), bt["y"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    torch.save(net.state_dict(), f".cache/who_when_decomp_{arm}_s0.pt")
    net.eval(); er = np.random.default_rng(77_000)
    hits, corr, dist, cnt, which = [], [], [], [], []
    with torch.no_grad():
        for _ in range(4):
            bt = W.batch(1000, er, 128); pr = net(bt).argmax(-1)
            hits.append(net.router_hit.numpy()); corr.append((pr == bt["y"]).float().numpy())
            n = bt["ns"].numpy(); dist.append(n - bt["anc"].numpy())
            A = bt["x1"].numpy()[np.arange(1000), n] - W.V_QA
            cnt.append((bt["x2"].numpy()[:, :128] == W.V_ACT + A[:, None]).sum(1))
            which.append(bt["x2"].numpy()[np.arange(1000), n] - W.V_WH)
    h, c, d, k, wh = (np.concatenate(z) for z in (hits, corr, dist, cnt, which))
    say(f"== {arm} at N 128 (4000 sequences): router {h.mean():.3f}  accuracy {c.mean():.3f}")
    def row(lbl, m): say(f"  {lbl:<34} n {m.sum():>5}  router {h[m].mean():.3f}  acc {c[m].mean():.3f}")
    for lo in range(0, 128, 16):
        row(f"dist {lo + 1:>3}-{lo + 16:<3}", (d > lo) & (d <= lo + 16))
    for one in (True, False):
        for far in (False, True):
            row(f"{'one sale' if one else 'two sales'}, dist {'> 64' if far else '<= 64'}", ((k == 1) == one) & ((d > 64) == far))
    for w in (0, 1):
        for far in (False, True):
            row(f"two sales, {'first' if w == 0 else 'last'}, dist {'> 64' if far else '<= 64'}", (k == 2) & (wh == w) & ((d > 64) == far))
