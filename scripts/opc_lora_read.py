"""opc_lora_read.py — LEG A PILOT READ (2026-08-26): the count head on a
REGISTER-REPAIRED trunk. States computed THROUGH the candidate's LoRA
adapters (the trainer's host pass mirrored: rope-off honored, per-layer
lora dicts from whatever keys exist). Grade: op-multiset exact + F1 on
the 143 wild golds. BAR (pinned): op-only exact > 14/143 (the frozen-
trunk count head's best) on ANY pilot = the register-repair hypothesis
confirmed at the op grain -> fleet word. Env: OPCL_CKPT + the
candidate's own head/LoRA envs set by the caller.
"""
import os, sys, json, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import (build_params, forward, T_ALG, TOKENIZER_JSON,
                                 sent_indices, load_alg, OPC_CLASSES, _opc_meta)
from beacon_closing_arm import _trunk_host
from mycelium.llama_loader import _rms_norm
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
ROPE_OFF = int(os.environ.get("ALG_ROPE_OFF", "0"))
SCALE = float(os.environ.get("ALG_LORA_SCALE", "8.0"))

def corpus143():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f):
            r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    return [v for k, v in sorted(byid.items()) if k not in sk]

def main():
    ck = os.environ["OPCL_CKPT"]
    p = build_params(0)
    sd = safe_load(f'.cache/{ck}.safetensors')
    assert set(sd.keys()) == set(p.keys()), \
        f"{ck} key mismatch: {sorted(set(sd)-set(p))[:3]}/{sorted(set(p)-set(sd))[:3]}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    host = _trunk_host()
    rc = host.llama_rope_cos[ROPE_OFF:] if ROPE_OFF else host.llama_rope_cos
    rs = host.llama_rope_sin[ROPE_OFF:] if ROPE_OFF else host.llama_rope_sin
    LD = []
    for li in range(4):
        d = {f"{nm}_{ab}": (p[f"lora{li}_{nm}_{ab}"] * (SCALE if ab == "B" else 1.0))
             for nm in ("wq", "wo", "wdown") for ab in ("A", "B")
             if f"lora{li}_{nm}_A" in p}
        LD.append(d if d else None)
    gold = corpus143()
    dec = []
    for s0 in range(0, len(gold), 8):
        sl = gold[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32)
        msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        for li2, layer in enumerate(host.llama_layers):
            x = layer(x, rc, rs, lora=LD[li2])
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        sts = x.cast(dtypes.float)
        o = forward(p, sts, Tensor(msk, dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int))
        opc = o["opc"].realize().numpy()
        for li in range(len(sl)):
            cnt = opc[li].argmax(-1)
            dec.append(Counter({c: int(k) for c, k in zip(OPC_CLASSES, cnt)
                                if k > 0}))
    OPS = ("add", "sub", "mul", "div", "sq", "opa", "fr")
    exf = exo = 0; f1s = []
    for d, r in zip(dec, gold):
        g = Counter({c: k for c, k in zip(OPC_CLASSES, _opc_meta(r)) if k > 0})
        if d == g: exf += 1
        if Counter({c: v for c, v in d.items() if c in OPS}) == \
           Counter({c: v for c, v in g.items() if c in OPS}): exo += 1
        inter = sum((d & g).values())
        f1s.append(2 * inter / max(sum(d.values()) + sum(g.values()), 1))
    print(f"[opcl {ck}] FULL exact {exf}/143  OP-ONLY exact {exo}/143 "
          f"(frozen-trunk best 14; bar >14)  F1 {np.mean(f1s):.3f}/"
          f"{np.median(f1s):.3f}", flush=True)

if __name__ == "__main__":
    main()
