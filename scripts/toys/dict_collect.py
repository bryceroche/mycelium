"""dict_collect.py -- THE DICTIONARY READ's trunk pass (2026-10-20; Bryce's word for the queue; Opus: train on D4 AND ~D7).
Every real token of N rows through the frozen Llama-3.2-1B trunk in the BP regime (bidirectional, pad keys masked = the
regime of the chassis's {split}BP states), taps D4 (the deployed tap, 'L0-L3') and D7 (the depth probe's mid-stack peak).
Rows: the climb targets' eligible pool (form_mix_pm35c, not truncated, >= 2 numerals with an owner link), EXCLUDING the
depth probe's 1500 rows (its fixture stays its own); seeded sample.
Labels per stored token (from .cache/climb_targets_formpm35c.npz): numeral span, the numeral's noun, verb, OWNER and
RECIPIENT positions; plus the token id (lexical control for the role read).
Out: .cache/dict_states_{D4,D7}.npy (Ntok, 2048) fp16 + .cache/dict_index.npz.
Run on the card: flock .cache/gpu.lock, DEV=PCI+AMD. --selftest (DEV=CPU): padded+masked == unpadded on 2 rows.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n_rows", type=int, default=12000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()
assert os.environ.get("DEV") == ("CPU" if args.selftest else "PCI+AMD"), "selftest DEV=CPU; the pass DEV=PCI+AMD"
from tokenizers import Tokenizer  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG  # noqa: E402

T, B, H, TAPS = 256, 8, 2048, (4, 7)
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
assert tok.id_to_token(0) == "!"


def load_host():
    class _H: pass
    h = _H(); sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
    attach_llama_layers(h, n_layers=max(TAPS), sd=sd, cfg=LLAMA_3_2_1B_CFG); del sd
    return h


def run_taps(h, ids, msk):
    """BP regime: bidirectional, pad keys masked. Returns {tap: (b, S, H) float32}."""
    b, S = msk.shape
    mask = None
    if msk is not None:
        m = np.where(msk[:, None, None, :] > 0, 0.0, -1e4).astype(np.float32)
        mask = Tensor(np.broadcast_to(m, (b, 1, S, S)).copy())
    x = h.llama_embed[Tensor(ids, dtype=dtypes.int)]; out = {}
    for k, L in enumerate(h.llama_layers, 1):
        x = L(x, h.llama_rope_cos, h.llama_rope_sin, attn_mask=mask).realize()
        if k in TAPS: out[k] = x.numpy()
    return out


if args.selftest:
    h = load_host(); worst = 0.0
    for l in [json.loads(l) for _, l in zip(range(2), open(".cache/form_mix_pm35c.jsonl"))]:
        ids = np.array(tok.encode(l["text"]).ids, np.int32); n = len(ids)
        pad = np.zeros((1, T), np.int32); pad[0, :n] = ids; msk = np.zeros((1, T), np.float32); msk[0, :n] = 1
        got = run_taps(h, pad, msk)
        x = h.llama_embed[Tensor(ids[None], dtype=dtypes.int)]; ref = {}
        for k, L in enumerate(h.llama_layers, 1):
            x = L(x, h.llama_rope_cos, h.llama_rope_sin, attn_mask=None).realize()
            if k in TAPS: ref[k] = x.numpy()
        for k in TAPS:
            rel = float(np.abs(got[k][0, :n] - ref[k][0]).max() / (np.abs(ref[k][0]).max() + 1e-9)); worst = max(worst, rel)
            assert rel < 1e-3, f"selftest FAIL D{k} rel {rel:.2e}"
    print(f"[selftest] PASS: BP padded+masked == unpadded bidirectional, 2 rows x taps {TAPS} (worst rel {worst:.1e})")
    sys.exit(0)

ct = np.load(".cache/climb_targets_formpm35c.npz", allow_pickle=True)
dp = set(np.load(".cache/depthprobe2_index.npz", allow_pickle=True)["pool_idx"].tolist())
elig = np.where((ct["truncated"] == 0) & (ct["n_num"] >= 2) & (ct["owner_valid"].sum(1) >= 2))[0]
elig = np.array([c for c in elig if int(ct["row_idx"][c]) not in dp])
pick = np.sort(np.random.default_rng(args.seed).choice(elig, size=args.n_rows, replace=False))
need = {int(ct["row_idx"][c]) for c in pick}; texts = {}
for i, l in enumerate(open(".cache/form_mix_pm35c.jsonl")):
    if i in need: texts[i] = json.loads(l)["text"]
assert len(texts) == len(need)
row_ids, lab = [], {k: [] for k in ("row", "pos", "tok", "numeral", "noun", "verb", "owner", "recipient")}
for ri, c in enumerate(pick):
    ids = np.array(tok.encode(texts[int(ct["row_idx"][c])]).ids, np.int32); n = len(ids); assert n <= T
    role = {k: np.zeros(n, np.int8) for k in ("numeral", "noun", "verb", "owner", "recipient")}
    for k in range(int(ct["n_num"][c])):
        s, e = ct["numeral_span"][c, k]
        if s < 0: continue
        role["numeral"][s:e] = 1
        for name, v in (("noun", "noun"), ("verb", "verb"), ("owner", "owner"), ("recipient", "recipient")):
            if ct[f"{v}_valid"][c, k]:
                ps, pe = ct[f"{v}_span"][c, k]; role[name][ps:pe] = 1
    row_ids.append(ids)
    lab["row"].append(np.full(n, ri, np.int32)); lab["pos"].append(np.arange(n, dtype=np.int32)); lab["tok"].append(ids)
    for k in role: lab[k].append(role[k])
lab = {k: np.concatenate(v) for k, v in lab.items()}; N = len(lab["row"])
print(f"[dict-collect] {len(elig)} eligible (depth-probe rows excluded); {len(pick)} rows, {N} tokens -> "
      f"{len(TAPS) * N * H * 2 / 1e9:.1f} GB", flush=True)
np.savez(".cache/dict_index.npz", pool_idx=np.array([int(ct["row_idx"][c]) for c in pick]), taps=np.array(TAPS), **lab)
h = load_host()
mm = {k: np.lib.format.open_memmap(f".cache/dict_states_D{k}.npy", mode="w+", dtype=np.float16, shape=(N, H)) for k in TAPS}
starts = np.concatenate([[0], np.cumsum([len(r) for r in row_ids])]); t0 = time.time()
for b0 in range(0, len(row_ids), B):
    sl = range(b0, min(b0 + B, len(row_ids)))
    ids = np.zeros((B, T), np.int32); msk = np.zeros((B, T), np.float32)
    for j, ri in enumerate(sl): ids[j, :len(row_ids[ri])] = row_ids[ri]; msk[j, :len(row_ids[ri])] = 1
    out = run_taps(h, ids, msk)
    for k, a in out.items():
        assert np.isfinite(a[:len(sl)]).all(), f"non-finite D{k}"
        for j, ri in enumerate(sl): mm[k][starts[ri]:starts[ri + 1]] = a[j, :len(row_ids[ri])].astype(np.float16)
    if (b0 // B) % 100 == 0: print(f"[dict-collect] batch {b0 // B}/{(len(row_ids) + B - 1) // B}  {time.time() - t0:.0f}s", flush=True)
for m in mm.values(): m.flush()
print(f"[dict-collect] DONE {len(row_ids)} rows, {N} tokens, taps {TAPS} in {time.time() - t0:.0f}s", flush=True)
