"""rung2b.py — THE LEAN-LoRA TAGGER (2026-08-27, word given; the
board's last untried square): token-grain cue supervision THROUGH
trunk adapters (r8, wq/wo/wdown, L0-L3, runtime copy — base safetensors
never touched, research lineage) on the backfilled canonical diet.
Seven instruments proved heads-on-frozen-states are sub-constant; this
is G58's actual lesson at the op grain: make the states BECOME
cue-bearing. Head: 2048->256->6 token classifier; joint train with
adapters; read = span-count on the 143 golds through the adapters.
BARS (pinned): canonical rock — exact > 17/143 AND F1 > 0.499. KILL
below -> the mint road is exhausted; the books road (wild annotation)
inherits the offensive.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, load_alg
from beacon_closing_arm import _trunk_host
from mycelium.llama_loader import _rms_norm
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import AdamW

CLS = ["none", "addf", "mul", "sq", "opa", "fr"]
NB = 8
STEPS = int(os.environ.get("R2B_STEPS", "4000"))
R = 8; SCALE = 8.0; NONE_W = 0.1
_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
rng = np.random.default_rng(0)

def main():
    Y = np.load('.cache/cue_y2_form8.npy').astype(np.int32)
    Y2 = Y.copy(); Y2[Y == 2] = 1
    for old, new in ((3, 2), (4, 3), (5, 4), (6, 5)):
        Y2[Y == old] = new
    ok = np.load('.cache/cue_rows2_form8.npy')
    rows_ok = np.where(ok)[0]
    texts = []
    for l in open('.cache/form_mix8.jsonl'):
        r = json.loads(l); texts.append(r.get("text") or r.get("original", ""))
    IDS = np.zeros((len(texts), T_ALG), np.int32)
    MSK = np.zeros((len(texts), T_ALG), np.float32)
    for i in rows_ok:
        e = tok.encode(texts[i])
        tid = e.ids[:T_ALG]
        IDS[i, :len(tid)] = tid; MSK[i, :len(tid)] = 1.0
    print(f"[2b] tokenized {len(rows_ok)} training rows", flush=True)
    host = _trunk_host()
    nrng = np.random.RandomState(7)
    P = {}
    for li in range(4):
        for nm, din in (("wq", 2048), ("wo", 2048), ("wdown", 8192)):
            P[f"lora{li}_{nm}_A"] = Tensor(
                (nrng.randn(din, R) * 0.01).astype(np.float32))
            P[f"lora{li}_{nm}_B"] = Tensor(np.zeros((R, 2048), np.float32))
    P["W1"] = Tensor(nrng.randn(2048, 256).astype(np.float32) * 0.02)
    P["b1"] = Tensor(np.zeros(256, np.float32))
    P["W2"] = Tensor(nrng.randn(256, 6).astype(np.float32) * 0.02)
    P["b2"] = Tensor(np.zeros(6, np.float32))
    for v in P.values(): v.requires_grad = True
    opt = AdamW(list(P.values()), lr=1e-4, weight_decay=0.0)

    def trunk(ids_t):
        x = host.llama_embed[ids_t].detach()
        for li, layer in enumerate(host.llama_layers):
            ld = {f"{nm}_{ab}": (P[f"lora{li}_{nm}_{ab}"]
                                 * (SCALE if ab == "B" else 1.0))
                  for nm in ("wq", "wo", "wdown") for ab in ("A", "B")}
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin, lora=ld)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm,
                      host.llama_cfg.rms_norm_eps)
        return x.cast(dtypes.float)

    Tensor.training = True
    for s in range(STEPS):
        idx = np.sort(rng.choice(rows_ok, 8, replace=False))
        X = trunk(Tensor(IDS[idx], dtype=dtypes.int))
        y = Y2[idx]; m = MSK[idx]
        w = m * np.where(y > 0, 1.0, NONE_W).astype(np.float32)
        lg = ((X @ P["W1"] + P["b1"]).relu() @ P["W2"] + P["b2"]).log_softmax(-1)
        nll = -lg.gather(-1, Tensor(y[..., None])).squeeze(-1)
        loss = (nll * Tensor(w)).sum() / (float(w.sum()) + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"[2b] step {s} loss {float(loss.numpy()):.4f}", flush=True)
    Tensor.training = False
    from tinygrad.nn.state import safe_save
    safe_save({k: v for k, v in P.items()}, '.cache/rung2b_tagger.safetensors')

    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    def canon(r):
        c = Counter()
        for f in r["factors"]:
            if f["ftype"] == "rel":
                if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                    c["sq"] += 1
                elif f.get("op") in ("add", "sub"): c["addf"] += 1
                elif f.get("op") == "mul": c["mul"] += 1
                elif f.get("op") == "div": c["fr"] += 1
            elif f["ftype"] == "macro":
                c["opa" if f.get("name") == "OP_APPLY" else "fr"] += 1
            elif f["ftype"] == "frac": c["fr"] += 1
        return Counter({k: min(v, NB - 1) for k, v in c.items() if v > 0})
    golds = [canon(r) for r in rows]
    def f1(a, b):
        i = sum((a & b).values())
        return 2 * i / max(sum(a.values()) + sum(b.values()), 1)
    keys = Counter(tuple(sorted(g.items())) for g in golds)
    cex = keys.most_common(1)[0][1]
    cf1 = max(np.mean([f1(Counter(dict(k)), g) for g in golds]) for k in keys)
    preds = []
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            tid = e.ids[:T_ALG]
            ids[li, :len(tid)] = tid; msk[li, :len(tid)] = 1.0
        X = trunk(Tensor(ids, dtype=dtypes.int))
        lg = ((X @ P["W1"] + P["b1"]).relu() @ P["W2"] + P["b2"])
        lab = lg.argmax(-1).numpy() * (msk > 0)
        for li in range(len(sl)):
            c = Counter(); prev = 0
            for t in range(T_ALG):
                v = int(lab[li, t])
                if v > 0 and v != prev: c[CLS[v]] += 1
                prev = v
            preds.append(Counter({k: min(v, NB - 1) for k, v in c.items()}))
    exo = 0; f1s = []; distinct = set()
    for d, g in zip(preds, golds):
        distinct.add(tuple(sorted(d.items())))
        if d == g: exo += 1
        f1s.append(f1(d, g))
    print(f"[2b] GOLD143 CANON exact {exo}/143 (rock {cex})  "
          f"F1 {np.mean(f1s):.3f} (rock {cf1:.3f})  distinct {len(distinct)}",
          flush=True)
    print("[2b] VERDICT: " + ("ABOVE THE ROCK — the trunk learned to see "
          "cues; first above-constant op skill in campaign history"
          if exo > cex and np.mean(f1s) > cf1 else
          "at or below the rock — the mint road is exhausted; the books "
          "road (wild annotation) inherits the offensive"), flush=True)

if __name__ == "__main__":
    main()
