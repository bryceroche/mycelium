"""reader_vote_one.py — one reader's root-grain votes (2026-08-29):
parse the 183 fixture raws through THIS reader's LoRA trunk, solve,
write answers. Env: RV_ID + the candidate's head/LoRA envs (set by the
driver before import)."""
import json, os, sys, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg)
from beacon_closing_arm import _trunk_host
from mycelium.llama_loader import _rms_norm
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
RID = os.environ["RV_ID"]
ROPE_OFF = int(os.environ.get("ALG_ROPE_OFF", "0"))
SCALE = float(os.environ.get("ALG_LORA_SCALE", "8.0"))

def fixtures():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [{"original": v["original"], "answer": v["answer"], "tag": "gold"}
            for k, v in sorted(byid.items()) if k not in sk]
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    rows += [{"original": r["original"], "answer": r["answer"], "tag": "wv"} for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(x["src_idx"] for x in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        rows += [{"original": h[i]["problem"], "answer": int(str(h[i]["answer"]).strip()),
                  "tag": "held"} for i in rg.permutation(len(h)) if i not in drafted
                 and str(h[i]["answer"]).strip().isdigit()][:10]
    return rows

def main():
    p = build_params(0)
    sd = safe_load(f'.cache/pool_{RID}.safetensors')
    assert set(sd.keys()) == set(p.keys()), f"{RID} env/ckpt mismatch"
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
    rows = fixtures()
    K = ("pres","ftype","op","islit","dig","args","res","query")
    out = []
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0+8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for i, r in enumerate(sl):
            e = tok.encode(r["original"])
            Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
            snt[i] = sent_indices(r["original"], list(e.offsets), msk[i])
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        for li2, layer in enumerate(host.llama_layers):
            x = layer(x, rc, rs, lora=LD[li2])
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        o = forward(p, x.cast(dtypes.float), Tensor(msk, dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int))
        keys = K + tuple(k2 for k2 in ("sel","dup","sgn") if k2 in o)
        onp = {k2: o[k2].realize().numpy() for k2 in keys}
        for i in range(len(sl)):
            facs, q = decode({k2: onp[k2][i] for k2 in onp})
            try:
                a = solve2(facs, q, {"n_vars": 24, "m": 300})
            except Exception:
                a = None
            out.append(a)
    json.dump({"id": RID, "answers": out},
              open(f'.cache/rootvote_{RID}.json', 'w'))
    ok = sum(1 for a in out if a is not None)
    print(f"[rv {RID}] solved {ok}/{len(out)}", flush=True)

if __name__ == "__main__":
    main()
