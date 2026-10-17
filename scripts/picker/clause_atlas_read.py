"""clause_atlas_read.py -- THE CLAUSE ATLAS CEILING (2026-10-16, Bryce: "memorize only the atomic pieces and mycelium builds the
factor graph from those"; zero GPU: banked BP trunk states + gold arrays of formpm35cBP).
WILD CARRIES NO SPANS, so the read runs on the diet's GSM8K rows with a PROBLEM-LEVEL hold-out: problems are grouped by
digit-masked skeleton (a number-swapped sibling = the same problem); HOLD problems are queried by their verbatim row,
the atlas holds every row of every OTHER problem (no sibling of a query is ever in the atlas).
CLAUSE KEY  per relation slot: the mean BP trunk state over its OP-CUE tokens (g_cspan), L2-normalized. Slots without a
            cue are skipped (coverage reported).
FRAGMENT    (op, arg kinds, given-arg mention offsets): arg kinds GIVEN / PRODUCED / SEEN-ARG / FIRST-HERE from the slot
            order; a GIVEN arg's offset = its mention start (g_vspan) - the cue start, in tokens.
THE ATLAS COMPILES a query relation from its nearest atlas clause (k=1, cosine): op; kinds; and each GIVEN arg is placed at
            the query's given whose mention start is nearest cue + offset (no reuse).
BASELINES   MAJORITY op/kinds; NEAREST-GIVENS (the givens whose mentions sit nearest the cue, count from gold kinds);
            RANDOM-NN (a random atlas clause's fragment, same apply rule) = how much the KEY matters.
Out: .cache/clause_atlas_read.txt"""
import collections, json, re, sys
import numpy as np

SPLIT = "formpm35cBP"; N_HOLD = 400; RNG = np.random.default_rng(0)
_z = np.load(f".cache/phase1_alg_states_{SPLIT}.npz")
# load each gold array ONCE (an NpzFile re-reads the whole member on every z[k] access)
z = {k: _z[k] for k in ("g_presence", "g_ftype", "g_op", "g_args", "g_res", "g_cspan", "g_vspan")}
S = np.load(f".cache/phase1_alg_states_{SPLIT}_states.npy", mmap_mode="r")
rows = [json.loads(l) for l in open(".cache/form_mix_pm35c_RO.jsonl")]
assert len(rows) == S.shape[0]
skel = lambda s: re.sub(r"\d[\d,\.]*", "#", re.sub(r"\s+", " ", s).strip())
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


gsm = [i for i, r in enumerate(rows) if isinstance(r.get("gen"), dict) and r["gen"].get("src") == "gsm8k"]
grp = collections.defaultdict(list)
for i in gsm:
    grp[skel(rows[i]["text"])].append(i)
keys = sorted(grp); RNG.shuffle(keys)
hold = []
for k in keys:
    verb = [i for i in grp[k] if not rows[i]["gen"].get("resample")]
    if verb:
        hold.append((k, verb[0]))
    if len(hold) == N_HOLD:
        break
hold_keys = {k for k, _ in hold}
atlas_rows = [i for k in keys if k not in hold_keys for i in grp[k]]


def first(mask):
    w = np.nonzero(mask)[0]
    return int(w[0]) if len(w) else None


def fragments(i):
    """per relation slot of row i: (key vector, op, kinds tuple, given offsets list, gold given slots list)."""
    pres, ft, op, args = z["g_presence"][i], z["g_ftype"][i], z["g_op"][i], z["g_args"][i]
    res, cs, vs = z["g_res"][i], z["g_cspan"][i], z["g_vspan"][i]
    intro = {}   # var -> (slot, how)
    out, nocue = [], 0
    st = None
    for j in range(24):
        if not pres[j]:
            continue
        if ft[j] == 1:
            intro.setdefault(j, (j, "G")); continue
        if ft[j] != 0:
            return None, 0
        a = [int(x) for x in np.nonzero(args[j] > 0.5)[0]]
        if len(a) == 1:
            a = a * 2
        kinds, offs, gslots = [], [], []
        c0 = first(cs[j])
        for v in a:
            if v in intro:
                s, how = intro[v]
                kinds.append("GIVEN" if how == "G" else "PRODUCED" if how == "R" else "SEEN")
                if how == "G":
                    m0 = first(vs[s])
                    offs.append(None if (m0 is None or c0 is None) else m0 - c0); gslots.append(s)
            else:
                kinds.append("FIRST")
        for v in a:
            intro.setdefault(v, (j, "A"))
        intro.setdefault(int(res[j]), (j, "R"))
        if c0 is None or None in offs:
            nocue += 1; continue
        if st is None:
            st = np.asarray(S[i], dtype=np.float32)
        kv = st[cs[j] > 0].mean(0); kv /= np.linalg.norm(kv) + 1e-8
        out.append((kv, int(op[j]), tuple(sorted(kinds)), offs, gslots, c0))
    return out, nocue


def given_starts(i):
    pres, ft, vs = z["g_presence"][i], z["g_ftype"][i], z["g_vspan"][i]
    return {j: first(vs[j]) for j in range(24) if pres[j] and ft[j] == 1 and first(vs[j]) is not None}


def place(offs, c0, gs):
    used, got = set(), []
    for o in offs:
        c = [(abs(m - (c0 + o)), s) for s, m in gs.items() if s not in used]
        if not c:
            got.append(None); continue
        s = min(c)[1]; used.add(s); got.append(s)
    return got


say(f"GSM8K diet rows {len(gsm)} | problems {len(keys)} | HOLD {len(hold)} problems (verbatim row) | atlas rows {len(atlas_rows)}")
A_K, A_F = [], []
skipped = 0
for n, i in enumerate(atlas_rows):
    fr, nc = fragments(i)
    if fr is None:
        skipped += 1; continue
    for kv, o, kd, offs, gsl, c0 in fr:
        A_K.append(kv.astype(np.float16)); A_F.append((o, kd, offs))
    if n % 5000 == 0:
        print(f"  atlas {n}/{len(atlas_rows)} clauses {len(A_K)}", flush=True)
A_K = np.stack(A_K).astype(np.float32)
say(f"atlas clauses {len(A_K)} (rows skipped for non-rel/given ftypes {skipped})")
opmaj = collections.Counter(f[0] for f in A_F).most_common(1)[0][0]
kdmaj = collections.Counter(f[1] for f in A_F).most_common(1)[0][0]
C = collections.Counter()
for k, i in hold:
    fr, nc = fragments(i)
    if fr is None:
        continue
    C["nocue"] += nc
    gs = given_starts(i)
    for kv, o, kd, offs, gsl, c0 in fr:
        C["q"] += 1
        sims = A_K @ kv; nn = int(sims.argmax()); rn = int(RNG.integers(len(A_F)))
        for tag, f in (("NN", A_F[nn]), ("RND", A_F[rn])):
            C[tag + "_op"] += f[0] == o; C[tag + "_kd"] += f[1] == kd
        C["MAJ_op"] += opmaj == o; C["MAJ_kd"] += kdmaj == kd
        if gsl:
            C["g"] += len(gsl)
            for tag, f in (("NN", A_F[nn]), ("RND", A_F[rn])):
                pl = place(f[2], c0, gs)
                C[tag + "_g"] += sum(1 for s in gsl if s in pl)
                C[tag + "_gall"] += set(gsl) <= set(pl)
            near = sorted(gs, key=lambda s: abs(gs[s] - c0))[:len(gsl)]
            C["NEAR_g"] += sum(1 for s in gsl if s in near); C["NEAR_gall"] += set(gsl) <= set(near)
            C["gq"] += 1
q = max(C["q"], 1); g = max(C["g"], 1); gq = max(C["gq"], 1)
say(f"query relations {C['q']} (skipped: no cue / unplaceable {C['nocue']})")
say(f"  OP            atlas-NN {C['NN_op'] / q:.3f} | random clause {C['RND_op'] / q:.3f} | majority {C['MAJ_op'] / q:.3f}")
say(f"  ARG KINDS     atlas-NN {C['NN_kd'] / q:.3f} | random clause {C['RND_kd'] / q:.3f} | majority {C['MAJ_kd'] / q:.3f}")
say(f"  GIVEN ARGS ({C['g']} in {C['gq']} relations), per arg / whole relation's given set:")
say(f"    atlas-NN offsets {C['NN_g'] / g:.3f} / {C['NN_gall'] / gq:.3f} | random clause {C['RND_g'] / g:.3f} / {C['RND_gall'] / gq:.3f} | "
    f"nearest-givens (count from gold) {C['NEAR_g'] / g:.3f} / {C['NEAR_gall'] / gq:.3f}")
open(".cache/clause_atlas_read.txt", "w").write("\n".join(OUT) + "\n")
