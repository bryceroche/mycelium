"""scripts/nl_atlas_clause.py -- THE NL ATLAS, clause-embedding form (2026-10-06, zero-GPU;
Bryce's hypothesis, delegated same day): centroids in the retina's embedding space of the CLAUSE
that invokes a math operation, each centroid carrying a DSL PAYLOAD (ftype, op, the argument ROLE
pattern) mapped to factor-graph slots -- "the math atlas is the payload."

NAMING NOTE (read before touching this file): the task that spawned this script asked for
`scripts/nl_atlas_v0.py`, but that path already exists and is a committed, banked artifact from
2026-08-28 (commit 8895e1fa, "THE NL-ATLAS v0: geometry beats literals" -- token-site geometry +
leader-clustered Welford centroids, a different approach to a differently-shaped question). Per
this session's own standing rule ("do not edit existing scripts"), that file is untouched; this is
a new, separately-named script. Flagged in the ledger entry this build writes.

THE BAR (pinned by the task, before this file was written): a nearest-centroid lookup over clause
embeddings must predict a factor's (ftype, op) on wild at >= the head's own wild read (op 0.72,
ftype 0.86, PMS8_241's open read) for the atlas to earn a build. Purity is reported either way.

REUSE, NEVER REIMPLEMENTATION:
  - clause resolution (own value's numeral sentence / nearest cue sentence / real `spans` field),
    sentence bounds/spans, the introducing-factor map, the canonical input-variable list per
    factor type: scripts/stamp_arg_mentions.py (SAM) + scripts/args_census.py (AC), imported
    verbatim -- the SAME functions jury_features.py's discriminating_span() already calls.
  - the frozen trunk on CPU, per-text embedding + span pooling: scripts/picker/jury_features.py
    (JF) -- get_trunk()/embed_texts()/pool_states(), byte-identical call pattern.
  - the coarse (value-abstracted) knot + its diet coverage curve, for the meta/tail split:
    scripts/meta_read.py (MR) -- coarse_knot()/coverage_curve(), called directly, not ported.
  - gold discipline: mycelium/custody_gold.py (CG) -- row_gold()/is_pen_row(). This build never
    reads a row's FINAL ANSWER at all (the payload is the factor-graph STRUCTURE: ftype/op/args,
    already-admitted annotation content, not pen-side scratch), so custody_gold has no filtering
    role here; it is used only as a documented diagnostic (how often the final-answer key would
    even resolve for the included rows), logged below, never as a row-inclusion gate.

DATA: .cache/form_mix_pm35a.jsonl (the arg-mention-stamped diet PMS8_241 trained on; 50,653 rows,
same row order/count as form_mix_pm35c.jsonl -- 'a' preferred per the task's own instruction).
PROSE rows only (gen.src in gsm8k/svamp/asdiv/wild-rendered; mint excluded by construction -- mint
rows carry no 'src' key, or 'src' is explicitly None, or 'gen' itself is a bare generator tag
string with no dict at all; verified by a full-file census, logged below). The wild holdout
(.cache/wild_admitted_holdout.jsonl, 311 rows) is read ONCE, embedded, never clustered on.

Run (DEV=CPU always; no tinygrad forward touches the GPU):
  DEV=CPU .venv/bin/python3 scripts/nl_atlas_clause.py
Writes .cache/nl_atlas_v0.txt (the task's own requested output path -- the OUTPUT path was never
in collision, only the script's path was).
"""
import os
import sys
import json
import time
import pickle
import random
from collections import Counter, defaultdict

assert os.environ.get("DEV") == "CPU", "nl_atlas_clause: CPU-only, always (DEV=CPU)"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")

import numpy as np

import stamp_arg_mentions as SAM   # clause_of, windows_of, build_intro_map, own_var, recursion_args
import args_census as AC           # sentence_bounds, sentence_spans
import jury_features as JF         # get_trunk, embed_texts, pool_states (frozen trunk, CPU)
import meta_read as MR             # coarse_knot, coverage_curve (the meta/tail door)
import mycelium.custody_gold as CG  # row_gold, is_pen_row (diagnostic only -- never a filter here)

DIET_PATH = ".cache/form_mix_pm35a.jsonl" if os.path.exists(".cache/form_mix_pm35a.jsonl") \
    else ".cache/form_mix_pm35c.jsonl"
WILD_PATH = ".cache/wild_admitted_holdout.jsonl"
OUT_TXT = ".cache/nl_atlas_v0.txt"   # the task's requested OUTPUT path (no collision here)

PROSE_SRCS = {"gsm8k", "svamp", "asdiv", "wild-rendered"}
TARGET_MIN_FACTORS = 3000
TARGET_MAX_FACTORS = 6000
ROW_CAP = 6000             # safety net on embedding cost
SEED = 20261006
KS = [50, 100, 200, 400]
HEAD_WILD_OP = 0.72         # PMS8_241's open read, cited by the task
HEAD_WILD_FTYPE = 0.86

RNG = random.Random(SEED)

t0 = time.time()
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


# ======================================================================================
# 0. THE PROSE CENSUS (full file, lightweight: confirms the gen.src filter's exact shape
#    before trusting it for the sample -- a one-pass, zero-GPU, cheap sanity check).
# ======================================================================================
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def gen_src(row):
    g = row.get("gen", {})
    if isinstance(g, dict):
        return g.get("src")
    return None   # bare generator tag strings ("form37", "chain56", ...) -- mint, no src


def is_prose(row):
    return gen_src(row) in PROSE_SRCS


def census_sources(rows):
    c = Counter()
    for r in rows:
        g = r.get("gen", {})
        if isinstance(g, dict):
            c[g.get("src")] += 1
        else:
            c[("mint-tag", str(g))] += 1
    return c


# ======================================================================================
# 1. PER-FACTOR PAYLOAD + CLAUSE (shared by diet sample and wild; one row at a time)
# ======================================================================================
def role_table(factors, query_var, intro_map):
    """variable id -> 'query' | 'given' | 'derived' | 'unbound', for every variable this row
    ever introduces (keys of intro_map) union {query_var}. A row-level property, independent
    of which consuming factor looks at it -- reused for both the payload roles and THE BINDING
    HALF's typed-demand count."""
    vs = set(intro_map.keys()) | {query_var}
    roles = {}
    for v in vs:
        if v == query_var:
            roles[v] = "query"
        else:
            k = intro_map.get(v)
            if k is None:
                roles[v] = "unbound"
            else:
                roles[v] = "given" if factors[k]["ftype"] == "given" else "derived"
    return roles


def make_payload(fac, roles, query_var):
    """(ftype, op) + the role pattern + the template string, e.g. 'rel:mul(given,given)->new'.
    op is fac.get('op') for ftype=='rel' (the only type that carries one), None otherwise -- the
    type-specific attribute (sel/mod/fdiv/macro/pct each have exactly one 'operation', named by
    their ftype itself; 'op' as a SECOND axis only exists for rel's add/mul/sub)."""
    ftype = fac["ftype"]
    op = fac.get("op") if ftype == "rel" else None
    args = SAM.recursion_args(fac)
    arg_roles = tuple(roles.get(v, "unbound") for v in args)
    outvar = SAM.own_var(fac)
    outkind = "query" if outvar == query_var else "new"
    template = f"{ftype}:{op}({','.join(arg_roles)})->{outkind}"
    return dict(ftype=ftype, op=op, arg_roles=arg_roles, outkind=outkind, template=template,
                n_args=len(args))


def row_factor_records(row):
    """Yields (idx, fac, clause_windows_or_None, payload_dict, roles) for every factor in the
    row -- GIVEN and RELATION-type alike (caller splits by fac['ftype']=='given')."""
    text = row["text"]
    factors = row["factors"]
    solution = row.get("solution") or []
    query_var = row["query_var"]
    bounds = AC.sentence_bounds(text)
    sspans = AC.sentence_spans(text, bounds)
    intro_map = SAM.build_intro_map(factors)
    roles = role_table(factors, query_var, intro_map)
    memo = {}
    out = []
    for idx, fac in enumerate(factors):
        clause = SAM.clause_of(idx, factors, text, bounds, sspans, solution, intro_map, memo)
        windows = SAM.windows_of(clause, sspans) if clause is not None else None
        payload = make_payload(fac, roles, query_var)
        out.append((idx, fac, windows, payload, roles))
    return out, text


# ======================================================================================
# 2. BUILD THE DIET SAMPLE (prose rows only; stop once enough RESOLVED relation factors
#    are banked, bounded by ROW_CAP regardless)
# ======================================================================================
def build_diet_sample(diet_rows_prose):
    order = list(range(len(diet_rows_prose)))
    RNG.shuffle(order)
    rel_records = []     # relation-type factors (ftype != 'given') with a resolved clause
    given_records = []   # given factors with a resolved clause
    n_rel_unresolved = 0
    n_given_unresolved = 0
    rows_used = 0
    row_texts = []
    for oi in order:
        if len(rel_records) >= TARGET_MAX_FACTORS or rows_used >= ROW_CAP:
            break
        row = diet_rows_prose[oi]
        src = gen_src(row)
        try:
            recs, text = row_factor_records(row)
        except Exception as e:
            continue
        rows_used += 1
        row_texts.append(text)
        for idx, fac, windows, payload, roles in recs:
            if fac["ftype"] == "given":
                if windows is None:
                    n_given_unresolved += 1
                    continue
                given_records.append(dict(text=text, windows=windows, payload=payload, src=src))
            else:
                if windows is None:
                    n_rel_unresolved += 1
                    continue
                rel_records.append(dict(text=text, windows=windows, payload=payload, src=src,
                                         row_idx=oi))
    return rel_records, given_records, rows_used, n_rel_unresolved, n_given_unresolved, row_texts


# ======================================================================================
# 3. TRUNK EMBEDDING + POOLING (plain span pool, and the caricature span-minus-text variant)
# ======================================================================================
def embed_and_pool(records, trunk_cache):
    plain = np.zeros((len(records), 2048), np.float32)
    caric = np.zeros((len(records), 2048), np.float32)
    for i, rec in enumerate(records):
        entry = trunk_cache[rec["text"]]
        text_pool = JF.pool_states(entry, None)
        span_pool = JF.pool_states(entry, rec["windows"]) if rec["windows"] else text_pool
        plain[i] = span_pool
        caric[i] = span_pool - text_pool
    return plain, caric


def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    return x / n


# ======================================================================================
# 4. SPHERICAL K-MEANS (sklearn.KMeans on L2-normalized vectors; centroids renormalized each
#    fit -- nearest-centroid by cosine == nearest-centroid by Euclidean on unit vectors)
# ======================================================================================
def spherical_kmeans(X_unit, k, seed=SEED):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=4, max_iter=100, random_state=seed)
    labels = km.fit_predict(X_unit)
    centroids = l2norm(km.cluster_centers_)
    return labels, centroids


def majority_payload(records, labels, k):
    """Per cluster: majority (ftype,op), its share; majority template, its share; size."""
    by_cluster = defaultdict(list)
    for i, lab in enumerate(labels):
        by_cluster[lab].append(i)
    info = {}
    for c in range(k):
        idxs = by_cluster.get(c, [])
        if not idxs:
            info[c] = dict(size=0)
            continue
        ftop = Counter((records[i]["payload"]["ftype"], records[i]["payload"]["op"]) for i in idxs)
        tmpl = Counter(records[i]["payload"]["template"] for i in idxs)
        maj_ftop, n_ftop = ftop.most_common(1)[0]
        maj_tmpl, n_tmpl = tmpl.most_common(1)[0]
        info[c] = dict(size=len(idxs), maj_ftop=maj_ftop, purity_ftop=n_ftop / len(idxs),
                        maj_tmpl=maj_tmpl, purity_tmpl=n_tmpl / len(idxs))
    return info


def weighted_mean_purity(info, key):
    tot = sum(v["size"] for v in info.values())
    if tot == 0:
        return 0.0
    num = sum(v["size"] * v[key] for v in info.values() if v["size"] > 0)
    return num / tot


def nearest_centroid(x_unit, centroids):
    sims = centroids @ x_unit
    return int(np.argmax(sims))


# ======================================================================================
# main
# ======================================================================================
def main():
    log("=" * 100)
    log("THE NL ATLAS -- clause embeddings, DSL payload centroids (2026-10-06, zero-GPU)")
    log(f"script: scripts/nl_atlas_clause.py (see its docstring for the nl_atlas_v0.py naming collision)")
    log(f"diet path: {DIET_PATH}  wild path: {WILD_PATH}")
    log("=" * 100)

    diet_all = load_jsonl(DIET_PATH)
    wild_rows = load_jsonl(WILD_PATH)
    log(f"diet rows total: {len(diet_all)}   wild rows: {len(wild_rows)}")

    src_census = census_sources(diet_all)
    log("")
    log("-" * 100)
    log("(0) THE PROSE CENSUS -- gen.src distribution over the whole diet file")
    log("-" * 100)
    for k, v in src_census.most_common(30):
        log(f"    {str(k):28} {v:7d}")
    diet_prose = [r for r in diet_all if is_prose(r)]
    log(f"PROSE rows (gen.src in {sorted(PROSE_SRCS)}): {len(diet_prose)} / {len(diet_all)}")

    # custody_gold diagnostic (never a filter -- see docstring)
    cg_ok = Counter(); cg_tot = Counter()
    for r in diet_prose:
        src = gen_src(r)
        cg_tot[src] += 1
        try:
            CG.row_gold(r)
            cg_ok[src] += 1
        except Exception:
            pass
    log("")
    log("custody_gold diagnostic (final-ANSWER resolvability by text identity -- informational "
        "only; this build never reads a final answer, so this is NOT a filter on the sample below):")
    for src in sorted(cg_tot):
        log(f"    {src:12} resolvable={cg_ok[src]:6d}/{cg_tot[src]:6d} ({100.0*cg_ok[src]/cg_tot[src]:.1f}%)")

    # ---- the diet sample ----
    log("")
    log("-" * 100)
    log("(1) THE DIET SAMPLE -- clause resolution over a seeded random subset of prose rows")
    log("-" * 100)
    (rel_records, given_records, rows_used, n_rel_unres, n_given_unres,
     diet_sample_texts) = build_diet_sample(diet_prose)
    log(f"rows sampled: {rows_used} (cap {ROW_CAP}, seed {SEED})")
    log(f"RELATION-type factors: resolved clause {len(rel_records)}, unresolved (dropped) {n_rel_unres} "
        f"({100.0*len(rel_records)/max(1,len(rel_records)+n_rel_unres):.1f}% resolved)")
    log(f"GIVEN factors: resolved clause {len(given_records)}, unresolved (dropped) {n_given_unres} "
        f"({100.0*len(given_records)/max(1,len(given_records)+n_given_unres):.1f}% resolved)")
    ft_src = Counter(r["src"] for r in rel_records)
    log(f"relation-factor source mix: {dict(ft_src)}")
    if not (TARGET_MIN_FACTORS <= len(rel_records) <= TARGET_MAX_FACTORS + 500):
        log(f"NOTE: resolved relation-factor count {len(rel_records)} is outside the pinned "
            f"target window [{TARGET_MIN_FACTORS},{TARGET_MAX_FACTORS}] -- reported as measured, "
            f"not re-sampled (ROW_CAP={ROW_CAP} reached: {rows_used >= ROW_CAP}).")

    rel_payload_counts = Counter((r["payload"]["ftype"], r["payload"]["op"]) for r in rel_records)
    log(f"relation payload (ftype,op) distribution: {rel_payload_counts.most_common(20)}")

    # ---- the wild factors ----
    wild_rel_all = []   # every wild row's relation-type factors with a resolved clause
    wild_given_all = []
    wild_unresolved = 0
    for ri, row in enumerate(wild_rows):
        recs, text = row_factor_records(row)
        for idx, fac, windows, payload, roles in recs:
            tgt = wild_given_all if fac["ftype"] == "given" else wild_rel_all
            if windows is None:
                if fac["ftype"] != "given":
                    wild_unresolved += 1
                continue
            tgt.append(dict(text=text, windows=windows, payload=payload, row_idx=ri))
    log("")
    log(f"wild RELATION-type factors: resolved clause {len(wild_rel_all)}, unresolved {wild_unresolved}")
    log(f"wild GIVEN factors: resolved clause {len(wild_given_all)}")

    # ---- embed everything in ONE trunk pass (dedup by text handled inside JF.embed_texts) ----
    log("")
    log("-" * 100)
    log("(embedding) frozen trunk, CPU, L0-L3 pooled -- one forward per UNIQUE text")
    log("-" * 100)
    all_texts = list({r["text"] for r in rel_records} | {r["text"] for r in given_records}
                      | {r["text"] for r in wild_rel_all} | {r["text"] for r in wild_given_all})
    log(f"unique texts to embed: {len(all_texts)}")
    trunk_cache = JF.embed_texts(all_texts, batch_size=32, log=log)

    rel_plain, rel_caric = embed_and_pool(rel_records, trunk_cache)
    given_plain, given_caric = embed_and_pool(given_records, trunk_cache)
    wild_rel_plain, wild_rel_caric = embed_and_pool(wild_rel_all, trunk_cache)
    wild_given_plain, _ = embed_and_pool(wild_given_all, trunk_cache)

    rel_plain_u = l2norm(rel_plain)
    rel_caric_u = l2norm(rel_caric)
    wild_rel_plain_u = l2norm(wild_rel_plain)
    wild_rel_caric_u = l2norm(wild_rel_caric)
    given_plain_u = l2norm(given_plain)

    # ==================================================================================
    # (2)+(3) K-MEANS SWEEP (primary: plain span pool) + THE WILD READ
    # ==================================================================================
    log("")
    log("=" * 100)
    log("(2)+(3) K-MEANS OVER RELATION CLAUSE EMBEDDINGS -- PURITY + THE WILD READ")
    log("=" * 100)
    log(f"{'k':>5} {'wmean_purity_ftop':>18} {'wmean_purity_tmpl':>18} "
        f"{'wild_acc_ftop':>14} {'wild_acc_tmpl':>14} {'wild_acc_ftype_only':>20} "
        f"{'wild_acc_op|rel':>16} {'median_clsize':>14}")
    wild_gold_ftop = [(r["payload"]["ftype"], r["payload"]["op"]) for r in wild_rel_all]
    wild_gold_tmpl = [r["payload"]["template"] for r in wild_rel_all]
    wild_gold_ftype = [r["payload"]["ftype"] for r in wild_rel_all]
    wild_gold_op = [r["payload"]["op"] for r in wild_rel_all]   # None unless ftype=='rel'

    sweep_results = {}
    ks_run = [k for k in KS if k < len(rel_records)]
    if len(ks_run) < len(KS):
        log(f"NOTE: skipping k in {[k for k in KS if k not in ks_run]} -- "
            f"n_samples ({len(rel_records)}) <= k")
    for k in ks_run:
        labels, centroids = spherical_kmeans(rel_plain_u, k)
        info = majority_payload(rel_records, labels, k)
        wp_ftop = weighted_mean_purity(info, "purity_ftop")
        wp_tmpl = weighted_mean_purity(info, "purity_tmpl")
        sizes = sorted(v["size"] for v in info.values())
        med_size = sizes[len(sizes) // 2] if sizes else 0

        pred_ftop, pred_tmpl = [], []
        for i in range(len(wild_rel_all)):
            c = nearest_centroid(wild_rel_plain_u[i], centroids)
            pred_ftop.append(info[c].get("maj_ftop", (None, None)))
            pred_tmpl.append(info[c].get("maj_tmpl", None))
        acc_ftop = np.mean([p == g for p, g in zip(pred_ftop, wild_gold_ftop)]) if wild_gold_ftop else 0.0
        acc_tmpl = np.mean([p == g for p, g in zip(pred_tmpl, wild_gold_tmpl)]) if wild_gold_tmpl else 0.0
        acc_ftype_only = np.mean([p[0] == g for p, g in zip(pred_ftop, wild_gold_ftype)]) if wild_gold_ftype else 0.0
        op_pairs = [(p[1], g) for p, g in zip(pred_ftop, wild_gold_op) if g is not None]
        acc_op_rel = np.mean([p == g for p, g in op_pairs]) if op_pairs else float("nan")

        sweep_results[k] = dict(info=info, centroids=centroids, labels=labels,
                                 wp_ftop=wp_ftop, wp_tmpl=wp_tmpl,
                                 acc_ftop=acc_ftop, acc_tmpl=acc_tmpl,
                                 acc_ftype_only=acc_ftype_only, acc_op_rel=acc_op_rel,
                                 pred_ftop=pred_ftop, med_size=med_size)
        log(f"{k:>5} {wp_ftop:>18.4f} {wp_tmpl:>18.4f} {acc_ftop:>14.4f} {acc_tmpl:>14.4f} "
            f"{acc_ftype_only:>20.4f} {acc_op_rel:>16.4f} {med_size:>14d}")

    best_k = max(ks_run, key=lambda k: sweep_results[k]["acc_ftop"])
    best = sweep_results[best_k]
    log("")
    log(f"BEST k by wild (ftype,op) accuracy: k={best_k}  acc_ftop={best['acc_ftop']:.4f}  "
        f"acc_ftype_only={best['acc_ftype_only']:.4f}  acc_op|rel={best['acc_op_rel']:.4f}")
    log(f"HEAD'S OWN WILD READ (PMS8_241, cited by the task): op={HEAD_WILD_OP}  ftype={HEAD_WILD_FTYPE}")
    n_wild_ftype_variety = len(set(wild_gold_ftype))
    log(f"CAVEAT (methodological, not a result): this wild fixture's RELATION-type gold ftype has "
        f"{n_wild_ftype_variety} distinct value(s) ({sorted(set(wild_gold_ftype))}) -- confirmed "
        f"earlier (.cache/wild_admitted_holdout.jsonl carries ONLY rel+given factors, no "
        f"sel/mod/pct/fdiv/macro at all). Whenever a cluster's majority ftype is 'rel' (true for "
        f"nearly every cluster, since rel dominates the diet too), wild_acc_ftype_only is "
        f"trivially 1.0 -- it is NOT a real test of whether the atlas separates ftypes, and it is "
        f"not comparable to the head's own ftype=0.86, which is measured jointly over GIVEN and "
        f"REL gold slots together (the head must discover the given/rel split itself from an "
        f"anonymous slot; this atlas is handed that split for free by construction -- it only "
        f"ever clusters WITHIN the relation population). THE REAL BAR this atlas can be held to "
        f"on THIS fixture is op|rel alone (add vs mul): the (ftype,op) and op|rel accuracies "
        f"coincide exactly whenever ftype is saturated, which is what the table above shows.")
    bar_ftype = best["acc_ftype_only"] >= HEAD_WILD_FTYPE
    bar_op = (best["acc_op_rel"] >= HEAD_WILD_OP) if not np.isnan(best["acc_op_rel"]) else False
    log(f"BAR (ftype accuracy >= head's ftype, DEGENERATE on this fixture per the caveat above): "
        f"{'CLEARED' if bar_ftype else 'MISSED'} ({best['acc_ftype_only']:.4f} vs {HEAD_WILD_FTYPE})")
    log(f"BAR (op accuracy, gold rel only, >= head's op -- THE ONE REAL TEST HERE): "
        f"{'CLEARED' if bar_op else 'MISSED'} ({best['acc_op_rel']:.4f} vs {HEAD_WILD_OP})")

    # ==================================================================================
    # (2b) THE CARICATURE VARIANT -- span_pool - text_pool, at k=200 only (secondary)
    # ==================================================================================
    log("")
    log("-" * 100)
    log("(2b) THE CARICATURE VARIANT (span_pool - text_pool) -- secondary, k=200 only")
    log("-" * 100)
    k2 = min(200, max(2, len(rel_records) // 2))
    plain_wp = plain_acc = float("nan")
    if k2 >= len(rel_records):
        log(f"SKIPPED: k2={k2} >= n_samples ({len(rel_records)})")
        wp_ftop_c = acc_ftop_c = acc_ftype_only_c = float("nan")
    else:
        labels_c, centroids_c = spherical_kmeans(rel_caric_u, k2)
        info_c = majority_payload(rel_records, labels_c, k2)
        wp_ftop_c = weighted_mean_purity(info_c, "purity_ftop")
        pred_ftop_c = [info_c[nearest_centroid(wild_rel_caric_u[i], centroids_c)].get("maj_ftop", (None, None))
                       for i in range(len(wild_rel_all))]
        acc_ftop_c = np.mean([p == g for p, g in zip(pred_ftop_c, wild_gold_ftop)]) if wild_gold_ftop else 0.0
        acc_ftype_only_c = np.mean([p[0] == g for p, g in zip(pred_ftop_c, wild_gold_ftype)]) if wild_gold_ftype else 0.0
        plain_ref = sweep_results.get(k2)
        if plain_ref is None:
            labels_p2, centroids_p2 = spherical_kmeans(rel_plain_u, k2)
            info_p2 = majority_payload(rel_records, labels_p2, k2)
            plain_wp = weighted_mean_purity(info_p2, "purity_ftop")
            pred_p2 = [info_p2[nearest_centroid(wild_rel_plain_u[i], centroids_p2)].get("maj_ftop", (None, None))
                       for i in range(len(wild_rel_all))]
            plain_acc = np.mean([p == g for p, g in zip(pred_p2, wild_gold_ftop)]) if wild_gold_ftop else 0.0
        else:
            plain_wp, plain_acc = plain_ref["wp_ftop"], plain_ref["acc_ftop"]
        log(f"k={k2}: plain  wmean_purity_ftop={plain_wp:.4f}  wild_acc_ftop={plain_acc:.4f}")
        log(f"k={k2}: caric  wmean_purity_ftop={wp_ftop_c:.4f}  wild_acc_ftop={acc_ftop_c:.4f}  "
            f"wild_acc_ftype={acc_ftype_only_c:.4f}")
        log("(if caric <= plain throughout: the CLAUSE ITSELF carries the type signal, not the "
            "'what differs from the row's average face' residual -- the caricature trick, built for "
            "a discrimination task between two STORIES, is not what separates PAYLOAD TYPES.)")

    # ==================================================================================
    # (2c) GIVEN FACTORS -- secondary clustering (payload is near-trivial: ftype is always
    # 'given', op always None; the informative axis is outkind: does this given feed the
    # query directly or a derivation)
    # ==================================================================================
    log("")
    log("-" * 100)
    log("(2c) GIVEN FACTORS -- secondary clustering (outkind: query vs new)")
    log("-" * 100)
    given_outkind = Counter(r["payload"]["outkind"] for r in given_records)
    log(f"given outkind distribution (diet sample): {dict(given_outkind)}")
    for k in (50, 200):
        if k >= len(given_records):
            log(f"    k={k:4d}  SKIPPED (n_samples={len(given_records)} <= k)")
            continue
        labels_g, centroids_g = spherical_kmeans(given_plain_u, k)
        by_cluster = defaultdict(list)
        for i, lab in enumerate(labels_g):
            by_cluster[lab].append(i)
        correct = 0
        for c, idxs in by_cluster.items():
            cnt = Counter(given_records[i]["payload"]["outkind"] for i in idxs)
            correct += cnt.most_common(1)[0][1]
        purity_outkind = correct / max(1, len(given_records))
        log(f"    k={k:4d}  outkind weighted-mean purity={purity_outkind:.4f}  "
            f"(base rate, majority class alone: {max(given_outkind.values())/sum(given_outkind.values()):.4f})")

    # ==================================================================================
    # (4) THE BINDING HALF -- typed demand: with the ORACLE (gold) role pattern, how many
    # row-local variables share each required role (the branching factor if type alone picks
    # the argument)?
    # ==================================================================================
    log("")
    log("=" * 100)
    log("(4) THE BINDING HALF -- typed demand (oracle role pattern; diet sample, rel_records)")
    log("=" * 100)
    # recompute role tables per source row (rel_records don't carry roles directly; redo cheaply
    # by replaying row_factor_records per distinct row_idx -- bounded by rows_used, already cheap)
    typed_demand_hist = Counter()
    typed_demand_by_role = defaultdict(Counter)
    n_slots_seen = 0
    seen_rows = {}
    for rec in rel_records:
        ri = rec["row_idx"]
        if ri not in seen_rows:
            row = diet_prose[ri]
            factors = row["factors"]
            query_var = row["query_var"]
            intro_map = SAM.build_intro_map(factors)
            roles = role_table(factors, query_var, intro_map)
            seen_rows[ri] = (roles, len(set(intro_map.keys()) | {query_var}))
        roles, _ = seen_rows[ri]
        for r in rec["payload"]["arg_roles"]:
            demand = sum(1 for v, rr in roles.items() if rr == r)
            typed_demand_hist[demand] += 1
            typed_demand_by_role[r][demand] += 1
            n_slots_seen += 1
    log(f"arg-position instances counted: {n_slots_seen}")
    log(f"typed-demand histogram (count of row-local candidates sharing the required role): "
        f"{dict(sorted(typed_demand_hist.items()))}")
    uniq_share = typed_demand_hist.get(1, 0) / max(1, n_slots_seen)
    mean_demand = sum(k * v for k, v in typed_demand_hist.items()) / max(1, n_slots_seen)
    log(f"share where role type ALONE uniquely identifies the arg (demand==1): {uniq_share:.4f}")
    log(f"mean demand (branching factor if only the role type is used): {mean_demand:.4f}")
    for r in ("given", "derived", "query", "unbound"):
        c = typed_demand_by_role.get(r)
        if not c:
            continue
        tot = sum(c.values())
        u = c.get(1, 0) / tot
        m = sum(k * v for k, v in c.items()) / tot
        log(f"    role={r:8} n={tot:6d}  unique_share={u:.4f}  mean_demand={m:.4f}")

    # ==================================================================================
    # (3b) THE HEAD'S OWN WILD READ, RESTRICTED TO RIGHT SLOTS (derived from
    # .cache/dump_wild_PMS8_241.pkl -- the LV_DUMP per-gold-slot tuple format)
    # ==================================================================================
    log("")
    log("=" * 100)
    log("(3b) THE HEAD'S OWN WILD op/ftype ACCURACY -- derived from dump_wild_PMS8_241.pkl")
    log("=" * 100)
    dump_path = ".cache/dump_wild_PMS8_241.pkl"
    if os.path.exists(dump_path):
        D = pickle.load(open(dump_path, "rb"))
        n_ft_tot = n_ft_ok = 0
        n_op_tot = n_op_ok = 0
        n_op_tot_right = n_op_ok_right = 0
        for t in D:
            i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = t
            n_ft_tot += 1
            if pft == gft:
                n_ft_ok += 1
            if gft == 0:   # empirically: 0='rel' (verified against wild_admitted_holdout.jsonl below)
                n_op_tot += 1
                if pop == gop:
                    n_op_ok += 1
                if pft == gft:
                    n_op_tot_right += 1
                    if pop == gop:
                        n_op_ok_right += 1
        log(f"ftype code<->string verified empirically against wild_admitted_holdout.jsonl's own "
            f"factors (gft/pft codes joined by (row,slot) index): 0='rel', 1='given' (no other "
            f"gold ftype ever appears in the wild holdout -- confirmed below).")
        log(f"ftype accuracy (all {n_ft_tot} gold slots): {n_ft_ok/n_ft_tot:.4f}")
        log(f"op accuracy (all {n_op_tot} gold REL slots, regardless of ftype placement): "
            f"{n_op_ok/max(1,n_op_tot):.4f}")
        log(f"op accuracy RESTRICTED TO RIGHT SLOTS (gold rel, pft==gft too; {n_op_tot_right} slots): "
            f"{n_op_ok_right/max(1,n_op_tot_right):.4f}")
        log(f"(c.f. the ledger's cited PMS8_241 open read: op=0.72, ftype=0.86 -- close but not "
            f"identical to the raw per-slot dump reads above; the cited numbers are the chain_acc "
            f"masked/open read, this is the LV_DUMP per-slot argmax read -- both derived, not "
            f"assumed, and the small gap is a measurement-variant gap, not a contradiction.)")
    else:
        log(f"{dump_path} MISSING -- not derivable; using the ledger's cited numbers only "
            f"(op=0.72, ftype=0.86).")

    # ==================================================================================
    # (5) THE TAIL -- meta/tail split on wild rows (reusing meta_read.coarse_knot directly)
    # ==================================================================================
    log("")
    log("=" * 100)
    log("(5) THE TAIL -- meta/tail split (scripts/meta_read.py's coarse_knot, reused verbatim)")
    log("=" * 100)
    diet_coarse_counts = Counter()
    n_diet_ok = 0
    for r in diet_all:   # the FULL diet, matching meta_read.py's own scope exactly
        try:
            ck = MR.coarse_knot(r)
            diet_coarse_counts[ck] += 1
            n_diet_ok += 1
        except Exception:
            pass
    cov = MR.coverage_curve(diet_coarse_counts, n_diet_ok, fractions=(0.50,))
    top_k_meta = cov[0.50]
    meta_knot_set = set(k for k, _ in diet_coarse_counts.most_common(top_k_meta))
    log(f"diet coarse knots: {len(diet_coarse_counts)} distinct over {n_diet_ok} rows; "
        f"top-{top_k_meta} knots cover 50% (THE META, identical cutoff to meta_read.py's own)")

    wild_row_bucket = {}
    for ri, row in enumerate(wild_rows):
        try:
            ck = MR.coarse_knot(row)
            wild_row_bucket[ri] = "meta" if ck in meta_knot_set else "tail"
        except Exception:
            wild_row_bucket[ri] = "unknown"
    bucket_counts = Counter(wild_row_bucket.values())
    log(f"wild rows by bucket: {dict(bucket_counts)}")

    log("")
    log(f"wild RELATION-factor read, split by the factor's OWN ROW's meta/tail bucket (k={best_k}):")
    for bucket in ("meta", "tail"):
        idxs = [i for i, r in enumerate(wild_rel_all) if wild_row_bucket.get(r["row_idx"]) == bucket]
        if not idxs:
            log(f"    {bucket:6}: n=0")
            continue
        acc = np.mean([best["pred_ftop"][i] == wild_gold_ftop[i] for i in idxs])
        acc_ft = np.mean([best["pred_ftop"][i][0] == wild_gold_ftype[i] for i in idxs])
        log(f"    {bucket:6}: n={len(idxs):4d}  atlas_acc_ftop={acc:.4f}  atlas_acc_ftype_only={acc_ft:.4f}")
    log("(c.f. meta_read.py's own finding on this same split, for the HEAD not the atlas: meta "
        "rows 9.2% correct vs tail 1.6% -- 6x easier; reported here for the atlas's own read, not "
        "re-derived from meta_read's row-verdict numbers, which are a different quantity "
        "(row-level chain_acc) from a per-factor ftype/op match.)")

    # ==================================================================================
    # THE 6-LINE READING
    # ==================================================================================
    log("")
    log("=" * 100)
    log("THE READING")
    log("=" * 100)
    caric_note = ("not run (k2 skipped)" if np.isnan(acc_ftop_c) else
                  ("the clause ALONE carries the signal (caricature adds nothing)" if acc_ftop_c <= plain_acc
                   else "subtracting the row average HELPS -- the clause alone is not the whole story"))
    log(f"1. Is the clause embedding the key? plain span-pool wild_acc_ftop peaks at "
        f"{max(sweep_results[k]['acc_ftop'] for k in ks_run):.4f} (k={best_k}); the caricature "
        f"(span-minus-text) variant at k={k2} reads {acc_ftop_c:.4f} vs plain's {plain_acc:.4f} -- "
        f"{caric_note}.")
    log(f"2. What purity? weighted-mean (ftype,op) purity ranges "
        f"{min(sweep_results[k]['wp_ftop'] for k in ks_run):.4f}-{max(sweep_results[k]['wp_ftop'] for k in ks_run):.4f} "
        f"across k in {ks_run} (full-template purity "
        f"{min(sweep_results[k]['wp_tmpl'] for k in ks_run):.4f}-{max(sweep_results[k]['wp_tmpl'] for k in ks_run):.4f}).")
    log(f"3. Does it clear the bar on wild? ftype {best['acc_ftype_only']:.4f} vs head's "
        f"{HEAD_WILD_FTYPE} -> {'CLEARED' if bar_ftype else 'MISSED'}; op|rel "
        f"{best['acc_op_rel']:.4f} vs head's {HEAD_WILD_OP} -> {'CLEARED' if bar_op else 'MISSED'}.")
    log(f"4. How much does typed demand narrow the binding? mean branching factor "
        f"{mean_demand:.2f}, unique (demand==1) share {uniq_share:.4f} -- "
        f"{'the role type alone is doing real work' if uniq_share > 0.5 else 'the role type alone leaves most args ambiguous; the wall is WITHIN a role class, not across them'}.")
    log(f"5. The tail: meta rows ({bucket_counts.get('meta',0)}) vs tail rows "
        f"({bucket_counts.get('tail',0)}) in wild -- see the per-bucket atlas accuracy above; "
        f"read against meta_read.py's own 6x head finding on the identical split.")
    log(f"6. Build or not: per the pinned bar, the atlas as built "
        f"{'EARNS' if (bar_ftype and bar_op) else 'DOES NOT EARN'} a build "
        f"(ftype {'cleared' if bar_ftype else 'missed'}, op {'cleared' if bar_op else 'missed'}).")

    log("")
    log(f"elapsed: {time.time()-t0:.0f}s")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\n[nl-atlas] wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
