"""gsm_pred_dump.py -- B-CEIL's picks (ledger 2026-10-29 17:07): re-train an arm of gsm_trunk_reader.py exactly as the step-0 grid
(same seed / steps / batch / threads; the training loop is the reader's own, imported) and dump every predicted slot on the requested
splits. The grid saves no picks; this is the one re-run that supplies them. Non-NOTEBOOK arms only (single-pass decoding).
Env: RD_ARM / RD_SEED as the reader; DUMP_SPLITS (default TEST). Output .cache/gsmb_picks_{TAG}.json:
{split: [{"slots": [[op, [kind, idx], [kind, idx]] x K], "m": gold steps}]} + the step_all read for the reproduction check."""
import json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_trunk_reader as R

SPLITS = os.environ.get("DUMP_SPLITS", "TEST").split(",")


def picks(net, its, states, tree):
    import torch
    out, ok_all = [], []
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = R.batch(part, states, tree, np.random.default_rng(10_000 * 0 + s0))
            op, pa, pb = net(t); L = t["pad"].shape[1]
            po, ia, ib = op.argmax(-1).numpy(), pa.argmax(-1).numpy(), pb.argmax(-1).numpy()
            dec = lambda i: ["t", int(i)] if i < L else (["s", int(i - L)] if i < L + R.K else ["c", int(i - L - R.K)])
            for b, e in enumerate(E):
                out.append(dict(slots=[[R.OPS[po[b, k]], dec(ia[b, k]), dec(ib[b, k])] for k in range(R.K)], m=e["m"]))
                ok_all += [bool(po[b, k] == e["op"][k] and t["va"][b, k, ia[b, k]] and t["vb"][b, k, ib[b, k]]) for k in range(e["m"])]
    return out, float(np.mean(ok_all))


if __name__ == "__main__":
    import torch
    assert not R.NOTE, "single-pass arms only"
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8"))); torch.manual_seed(R.SEED); rng = np.random.default_rng(R.SEED)
    Dt, states = R.load(); tree = R.load_tree(); train = Dt["TRAIN"]
    net = R.make_net(); opt = torch.optim.AdamW(net.parameters(), lr=R.LR); t0 = time.time()
    for step in range(1, R.STEPS + 1):                                  # the reader's loop, minus the interim reads (they draw no rng)
        its = [train[int(i)] for i in rng.integers(0, len(train), R.BATCH)]
        t, _ = R.batch(its, states, tree, rng); loss = R.loss_fn(net(t), t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 1000 == 0: print(f"[dump] {R.TAG} step {step} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); res = {}
    for sp in SPLITS:
        res[sp], sa = picks(net, Dt[sp], states, tree); res[f"{sp}_step_all"] = sa
        print(f"[dump] {R.TAG} {sp} step_all {sa:.4f} ({len(res[sp])} items)", flush=True)
    json.dump(res, open(f".cache/gsmb_picks_{R.TAG}.json", "w"))
