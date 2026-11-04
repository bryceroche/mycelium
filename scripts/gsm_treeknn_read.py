"""A1 THE TREE-KNN READ (ledger 2026-11-03, word given: "lets move towards our long term vision"): zero-training, banked BT weights.

"Overfit the facts to the Tree, generalize the reader": every TRAIN slot's silhouette at its commit cycle is stored as a LEAF with its
gold step type (memorize aggressively; real steps + the first stop slot). A TEST slot is classified by majority vote of its k nearest
leaves (Euclidean, the centroid classifier's own metric; ties -> the nearer class) and compared with the model's own centroid type,
on the same commit trajectories (the BT6 gate, rho0 from the banked BEAM read). k fit on TRAIN[:600] leave-one-PROBLEM-out.
Beside: the same vote on h (the 512-d slot state before the silhouette's down-projection).
-> .cache/gsmb_treeknn_{TAG}.json
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR

KS = (1, 5, 15, 45)
TAG = os.path.basename(BR.LOAD).replace("gsmb_breath_", "").replace(".pt", "")
assert BR.LOAD and os.path.exists(BR.LOAD) and not BR.GATE and not BR.ORACLE
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "4")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
if BR.ROLE: BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
net = BR.make_net(); net.load_state_dict(torch.load(BR.LOAD)); net.eval()
K, STOP, NCYC = BR.K, BR.STOP, BR.NCYC
RHO = json.load(open(f".cache/gsmb_beam_{TAG}.json"))["rho0"]
HCAP = {}
net.down.register_forward_hook(lambda m, i, o: HCAP.__setitem__("h", i[0].detach()))


def collect(its):
    """the BT6-gate loop -> per gold slot (real steps + the first stop): z, h at its commit cycle, committed type, gold type, item id"""
    Z, Hs, ty_c, ty_g, item = [], [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            xq = net.read(t)
            com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
            zc = np.zeros((B, K, BR.DB), np.float32); hc = np.zeros((B, K, BR.D), np.float32)
            progs = [[(".", None, None)] * K for _ in range(B)]
            dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
            for c in range(NCYC):
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, typ, rad))
                logit, pa, pb, z, p = net.breath(t, xq, nb, None, torch.full((B,), c), None)
                h = HCAP["h"]
                ty = logit.argmax(-1).numpy(); pt = p.max(-1).values.numpy()
                ia, ib = pa.argmax(-1).numpy(), pb.argmax(-1).numpy()
                ca = pa.softmax(-1).max(-1).values.numpy(); cb = pb.softmax(-1).max(-1).values.numpy()
                conf = np.where(ty == STOP, pt, pt * ca * cb)
                for b in range(B):
                    for k in [k for k in np.argsort(-conf[b]) if not com[b, k] and conf[b, k] >= RHO * (1 - c / (NCYC - 1))]:
                        com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]; zc[b, k] = z[b, k].numpy(); hc[b, k] = h[b, k].numpy()
                        op = BR.TYPES[ty[b, k]][0] if ty[b, k] != STOP else "."
                        progs[b][k] = (op, dec(ia[b, k]), dec(ib[b, k])) if op != "." else (".", None, None)
            _, gt = BR.gold(E)
            for b, e in enumerate(E):
                for k in range(min(e["m"] + 1, K)):
                    Z.append(zc[b, k]); Hs.append(hc[b, k]); ty_c.append(typ[b, k]); ty_g.append(gt[b, k]); item.append(s0 + b)
    return dict(z=np.array(Z), h=np.array(Hs), c=np.array(ty_c), g=np.array(ty_g), item=np.array(item))


def knn(Q, Lv, Ly, k, excl=None, Qi=None, Li=None):
    """majority vote of the k nearest leaves (ties -> the class of the nearer member); excl: leave-one-problem-out"""
    out = np.zeros(len(Q), np.int64)
    for s0 in range(0, len(Q), 512):
        q = Q[s0:s0 + 512]
        d = (q ** 2).sum(1)[:, None] - 2 * q @ Lv.T + (Lv ** 2).sum(1)[None]
        if excl: d[Qi[s0:s0 + 512, None] == Li[None]] = np.inf
        nn_ = np.argpartition(d, k, 1)[:, :k] if k < Lv.shape[0] else np.argsort(d, 1)[:, :k]
        for r in range(len(q)):
            idx = nn_[r][np.argsort(d[r, nn_[r]])]; lab = Ly[idx]
            cnt = np.bincount(lab, minlength=BR.NT); best = np.flatnonzero(cnt == cnt.max())
            out[s0 + r] = next(l for l in lab if l in best)
    return out


TR = collect(Dt["TRAIN"]); print(f"[tk] {TAG} leaves {len(TR['g'])} (TRAIN slots, real steps + first stop)", flush=True)
real = TR["g"] != STOP
q = np.flatnonzero(TR["item"] < 600)
out = dict(tag=TAG, rho0=RHO, leaves=int(len(TR["g"])), fit={})
for space in ("z", "h"):
    acc = {k: float((knn(TR[space][q], TR[space], TR["g"], k, True, TR["item"][q], TR["item"]) == TR["g"][q])[real[q]].mean()) for k in KS}
    kb = max(acc, key=lambda k: acc[k]); out["fit"][space] = dict(acc=acc, k=kb)
    print(f"[tk] {TAG} fit {space} (TRAIN[:600] leave-one-problem-out, real steps) " + " ".join(f"k{k}:{v:.3f}" for k, v in acc.items()) + f" -> k {kb}", flush=True)
for split in ("ORIG", "TEST"):
    S = collect(Dt[split]); rs = S["g"] != STOP
    cen = (S["c"] == S["g"])[rs]
    o = out[split] = dict(n=int(rs.sum()), centroid=float(cen.mean()))
    for space in ("z", "h"):
        pk = knn(S[space], TR[space], TR["g"], out["fit"][space]["k"])
        kk = (pk == S["g"])[rs]; d = kk.astype(float) - cen.astype(float)
        o[space] = dict(knn=float(kk.mean()), diff=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))),
                        knn_right_cen_wrong=float((kk & ~cen).mean()), cen_right_knn_wrong=float((~kk & cen).mean()))
    print(f"[tk] {TAG} {split} n {o['n']} type: centroid {o['centroid']:.4f} | kNN z {o['z']['knn']:.4f} ({o['z']['diff']:+.4f}, SE {o['z']['se']:.4f}; "
          f"+{o['z']['knn_right_cen_wrong']:.3f}/-{o['z']['cen_right_knn_wrong']:.3f}) | kNN h {o['h']['knn']:.4f} ({o['h']['diff']:+.4f}, SE {o['h']['se']:.4f})", flush=True)
json.dump(out, open(f".cache/gsmb_treeknn_{TAG}.json", "w"), indent=1)
