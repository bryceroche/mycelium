"""gsm_trunk_states.py -- STEP 0 of the GSM8K reader bench (ledger 2026-10-29, Bryce "word given for both"): the frozen Llama-3.2-1B
L0-L3 states for every question of every split in .cache/gsmb_data.json. GPU, tinygrad, DEV=PCI+AMD. Run once.
Regime: NATIVE (causal + pad keys masked: Llama's own regime; no length axis), 256 frame, final RMSNorm (precompute_masked.py's run()).
Numerals are VISIBLE to the trunk (the reader sees all surface; the bench's <num> masking was for the scratch reader) -- CFNUM is
therefore a real read here. Word alignment: each bench word (gsm_bench_data.tokenize) takes the state of the LAST Llama token whose
char span overlaps it (causal: the last sub-token has read the whole word). The text of every item is re-derived from its source
(parquet by gid / the CF fixture by cf id) and re-tokenized; the word sequence must equal the banked one (asserted).
Output: .cache/gsmb_trunk_states.npy (fp16 [n_words, 2048], one block per DISTINCT text) + .cache/gsmb_trunk_index.json
{split: [start per item]} aligned with gsmb_data.json's item order."""
import glob, json, os, re, sys, time

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import gsm_bench_data as GD

OUT_NPY, OUT_IDX = GD.V1.C("gsmb_trunk_states.npy"), GD.V1.C("gsmb_trunk_index.json")
T_ALG, H = 256, 2048


def texts():
    import pyarrow.parquet as pq
    rd = lambda sp: pq.read_table(glob.glob(GD.V1.C(f"gsm8k/datasets--openai--gsm8k/snapshots/*/main/{sp}-*.parquet"))[0]).to_pydict()
    tr, te = rd("train")["question"], rd("test")["question"]
    cf = {}                                                             # a CF id is shared by its num and num+name rows: key by kind
    for l in open(GD.V1.C("gsm8k_cf2_train.jsonl")):
        r = json.loads(l); cf[("CFNUM" if r["kind"] == "num" else "CFNAME", r["id"])] = r["question_cf"]
    Dt = json.load(open(GD.OUT)); out = {}
    for sp, its in Dt.items():
        src = {"TRAIN": tr, "ORIG": tr, "TEST": te}.get(sp)
        out[sp] = [src[it["gid"]] if src is not None else cf[(sp, it["cf"])] for it in its]
        for q, it in zip(out[sp], its):
            assert GD.tokenize(q, set(it["names"]))[0] == it["toks"], f"{sp} word sequence mismatch"
    return Dt, out


def align(q, offs, ntok):
    """-> index of the Llama token holding each bench word's state (last overlapping sub-token)."""
    spans = [m.span() for m in GD.TOK.finditer(q)]; out = []
    for a, b in spans:
        hit = [j for j in range(1, ntok) if offs[j][0] < b and offs[j][1] > a]
        assert hit, f"word {q[a:b]!r} has no Llama token"
        out.append(hit[-1])
    return out


if __name__ == "__main__":
    assert os.environ.get("DEV") in ("PCI+AMD", "CPU"), "DEV=PCI+AMD (never bare AMD) or CPU"
    assert not os.path.exists(OUT_NPY), f"refusing to overwrite {OUT_NPY}"
    from tokenizers import Tokenizer
    Dt, Q = texts()
    tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    uniq = list(dict.fromkeys(q for sp in Q.values() for q in sp))
    enc = [tok.encode(q) for q in uniq]
    assert max(len(e.ids) for e in enc) <= T_ALG, "a question exceeds the 256 frame"
    al = [align(q, e.offsets, len(e.ids)) for q, e in zip(uniq, enc)]
    start = np.cumsum([0] + [len(a) for a in al]); nw = int(start[-1])
    print(f"[trunk] {len(uniq)} distinct texts, {nw} words, max Llama len {max(len(e.ids) for e in enc)}", flush=True)
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm

    class _H: pass
    host = _H(); sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG); del sd
    st = np.lib.format.open_memmap(OUT_NPY, mode="w+", dtype=np.float16, shape=(nw, H))
    causal = np.triu(np.full((T_ALG, T_ALG), -1e4, np.float32), 1)[None, None]
    t0 = time.time(); BS = 8
    for s0 in range(0, len(uniq), BS):
        sl = range(s0, min(s0 + BS, len(uniq))); ids = np.zeros((BS, T_ALG), np.int32); m = np.zeros((BS, T_ALG), np.float32)
        for r, i in enumerate(sl): ids[r, :len(enc[i].ids)] = enc[i].ids; m[r, :len(enc[i].ids)] = 1
        am = Tensor((np.where(m[:, None, None, :] > 0, 0.0, -1e4).astype(np.float32) + causal).astype(np.float32))
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        for layer in host.llama_layers: x = layer(x, host.llama_rope_cos, host.llama_rope_sin, attn_mask=am)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps).cast(dtypes.float).realize().numpy()
        assert np.isfinite(x).all()
        for r, i in enumerate(sl): st[start[i]:start[i + 1]] = x[r, al[i]].astype(np.float16)
        if (s0 // BS) % 200 == 0: print(f"[trunk] batch {s0 // BS}/{(len(uniq) + BS - 1) // BS} {time.time() - t0:.0f}s", flush=True)
    st.flush()
    pos = {q: int(start[i]) for i, q in enumerate(uniq)}
    json.dump({sp: [pos[q] for q in Q[sp]] for sp in Q}, open(OUT_IDX, "w"))
    print(f"[trunk] DONE {nw} word states in {time.time() - t0:.0f}s", flush=True)
