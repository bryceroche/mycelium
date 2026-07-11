"""lexical_implicit_probe.py — the retrained examiner (2026-07-10).
Synthetic lexical-implicit corpus, PHRASE-SPLIT PINNED: train and test
evokers disjoint, so a passing probe reads ENCODING GEOMETRY, not vocabulary
trivia. Reads: (i) within-phrase held-out, (ii) CROSS-PHRASE (the verdict),
(iii) MATH-500's 16 real lexical implicits (the target). Two depths.
OUTCOMES: cross-phrase decodes -> implicit values live in a learnable
coordinate system (pairs-cure world for lexical); within-pass/cross-fail ->
dictionary-lookup verdict (mixed, sharper); both fail -> strongly stage-ward.
"""
import json, os, random, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, N_DIG, TOKENIZER_JSON, sent_indices, _spans_to_tokmask
from survivor_depth_probe import train_probe, eval_probe
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

TRAIN_EVOKERS = [("a dozen", 12), ("half a dozen", 6), ("a score", 20),
                 ("a pair", 2), ("a trio", 3), ("a quartet", 4),
                 ("a fortnight of days", 14), ("a century", 100),
                 ("eleven", 11), ("fifteen", 15), ("forty", 40),
                 ("seventy", 70)]
TEST_EVOKERS = [("a baker's dozen", 13), ("a gross", 144), ("a decade", 10),
                ("the days in February", 28), ("the minutes in an hour", 60),
                ("the inches in a foot", 12), ("the sides of an octagon", 8),
                ("the sides of a hexagon", 6), ("thirty", 30), ("ninety", 90)]
TEMPLATES = ["Consider the numbers {A}, {B}, {C}. The value of {B} is {PH}. "
             "{A} plus {B} equals {C}. What is {C}?",
             "Let {A}, {B}, {C} be whole numbers. {B} equals {PH}. "
             "{C} is {A} multiplied by {B}. What is {A}?",
             "The following holds about {A}, {B}. It is known that {A} is "
             "{PH}. {B} exceeds {A} by four. What is {B}?"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def mint(evokers, n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        ph, val = rng.choice(evokers)
        ls = rng.sample(LETTERS, 3)
        t = rng.choice(TEMPLATES).format(A=ls[0], B=ls[1], C=ls[2], PH=ph)
        pos = t.find(ph)
        rows.append((t, pos, pos + len(ph), val))
    return rows

def featurize(rows, depth_states, offs_list):
    X, Y = [], []
    for r_, (t, a, b, val) in enumerate(rows):
        m = np.zeros((T_ALG,), np.float32)
        _spans_to_tokmask([[a, b]], offs_list[r_], m)
        if m.sum() == 0: continue
        m /= m.sum()
        X.append((depth_states[r_].astype(np.float32) * m[:, None]).sum(0))
        Y.append([(val // 10 ** (N_DIG-1-d)) % 10 for d in range(N_DIG)])
    return np.stack(X), np.array(Y)

tok = Tokenizer.from_file(TOKENIZER_JSON)
def encode(rows):
    ids = np.zeros((len(rows), T_ALG), np.int32)
    offs = []
    for i, (t, *_ ) in enumerate(rows):
        e = tok.encode(t)
        ids[i, :len(e.ids)] = e.ids[:T_ALG]
        offs.append(list(e.offsets))
    return ids, offs

tr = mint(TRAIN_EVOKERS, 2000, 1)
wi = mint(TRAIN_EVOKERS, 300, 2)      # within-phrase held-out
cr = mint(TEST_EVOKERS, 300, 3)       # cross-phrase (the verdict)
LEX = [(27,"three",3),(40,"octagon",8),(40,"hexagon",6),(49,"Half",2),
       (128,"third",3),(133,"twice",2),(133,"right",90),(212,"one-hundred",100),
       (246,"Twelve",12),(329,"feet",12),(329,"minute",60),(341,"twice",2),
       (341,"one less",1),(385,"half",2),(411,"Six",6),(411,"ten",10)]
mrows = [json.loads(l) for l in open(".cache/math500_test.jsonl")]
math_rows = []
for i, ph, v in LEX:
    t = mrows[i]["problem"]
    p = t.find(ph)
    if p >= 0 and len(tok.encode(t).ids) <= T_ALG:
        math_rows.append((t, p, p + len(ph), v))

for depth in (4, 8):
    os.environ["_DEPTH"] = str(depth)
    def states_for(rows):
        ids, offs = encode(rows)
        if depth == 4:
            return recompute_states(ids), offs
        import mycelium.llama_loader as LL
        from tinygrad import Tensor, dtypes
        class _H: pass
        host = _H()
        sd = LL.load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
        LL.attach_llama_layers(host, n_layers=8, sd=sd, cfg=LL.LLAMA_3_2_1B_CFG)
        del sd
        out = np.zeros((len(rows), T_ALG, 2048), np.float16)
        for s0 in range(0, len(rows), 8):
            sl = slice(s0, min(s0+8, len(rows)))
            x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
            for l in host.llama_layers:
                x = l(x, host.llama_rope_cos, host.llama_rope_sin)
            x = LL._rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
            out[sl] = x.cast(dtypes.float).realize().numpy().astype(np.float16)
        return out, offs
    st_tr, o_tr = states_for(tr)
    Xtr, Ytr = featurize(tr, st_tr, o_tr)
    mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
    _, fwd = train_probe((Xtr - mu) / sd_, Ytr, seed=1)
    res = {}
    for nm, rows in (("within", wi), ("CROSS", cr), ("MATH-lex", math_rows)):
        st, offs = states_for(rows)
        X, Y = featurize(rows, st, offs)
        res[nm] = eval_probe(fwd, (X - mu) / sd_, Y)
    print(f"[L0-{depth-1}] within {res['within']:.2f} | "
          f"CROSS-PHRASE {res['CROSS']:.2f} | MATH-lex {res['MATH-lex']:.2f} "
          f"(n={len(math_rows)})")
print("VERDICT KEY: CROSS>=0.7 = learnable implicit geometry (pairs-cure); "
      "within-pass/cross-fail = dictionary lookup (mixed, sharper); "
      "both-fail = strongly stage-ward.")
