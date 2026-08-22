"""macro_propose.py — PORT-CRITICALITY RUNG 1 (2026-08-21, word given):
the marriage of the two organs. The miner finds what RECURS (sink-rooted
subgraph motifs, size 2-3, over the 268 wild gold graphs); the wall's
deletion probe prices which ports are LOAD-BEARING (per-instance
mandatory checks); rank = compression-gain x port-coherence. Candidates
land in campaign.db macro_proposals as PROPOSED — RANK-NEVER-ADMIT:
admission is Bryce's word + the hand-quota, and macros always expand
before the solver sees anything (the key grades primitives).
Hand-habit fence: occurrences counted separately in hand corpus /
machine deeds / dialect mint sample; hand-only candidates flagged.
Zero GPU.
"""
import sys, os, json, glob, sqlite3, itertools, datetime
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from repair_replace_swap import solve_forced
from mycelium.campaign_db import conn

def sig_of(facs, sink, members, prod):
    f = facs[sink]
    if f["ftype"] == "rel":
        parts = []
        for a in f["args"]:
            p = prod.get(a)
            if p is None: parts.append("?")
            elif p in members and facs[p]["ftype"] != "given":
                parts.append(sig_of(facs, p, members, prod))
            elif facs[p]["ftype"] == "given": parts.append("G")
            else: parts.append("X")
        return f'{f["op"]}({",".join(sorted(parts))})'
    if f["ftype"] == "fdiv":
        p = prod.get(f["var"])
        inner = ("G" if p is not None and facs[p]["ftype"] == "given"
                 else sig_of(facs, p, members, prod)
                 if p in members else "X")
        return f'fdiv{f["k"]}({inner})'
    return "?"

def motifs(facs):
    prod = {}
    for i, f in enumerate(facs):
        if f["ftype"] == "given": prod[f["var"]] = i
        elif f["ftype"] == "rel": prod[f["result"]] = i
        elif f["ftype"] == "fdiv": prod[f["result"]] = i
    kids = {i: [] for i in range(len(facs))}
    for i, f in enumerate(facs):
        srcs = (f.get("args", []) if f["ftype"] == "rel"
                else [f["var"]] if f["ftype"] == "fdiv" else [])
        for v in srcs:
            p = prod.get(v)
            if p is not None and facs[p]["ftype"] != "given":
                kids[i].append(p)
    out = []
    ops = [i for i, f in enumerate(facs) if f["ftype"] in ("rel", "fdiv")]
    for sink in ops:
        direct = kids[sink]
        for r in (1, 2):
            for combo in itertools.combinations(direct, min(r, len(direct))):
                members = set(combo) | {sink}
                if len(members) < 2: continue
                out.append((sig_of(facs, sink, members, prod),
                            frozenset(members)))
        for c1 in direct:                       # 3-chains through one child
            for c2 in kids[c1]:
                members = {sink, c1, c2}
                out.append((sig_of(facs, sink, members, prod),
                            frozenset(members)))
    return prod, set(out)

def harvest_counts(rows, probe):
    stats = {}
    for r in rows:
        facs, q, ans = r["factors"], r["query"], r["answer"]
        _, ms = motifs(facs)
        for sig, members in ms:
            st = stats.setdefault(sig, {"occ": 0, "mand_ok": 0, "size": len(members),
                                        "shas": []})
            st["occ"] += 1
            st["shas"].append(r.get("src_idx", -1))
            if probe:
                ok = all(
                    solve_forced([f for j, f in enumerate(facs) if j != m],
                                 q, {"n_vars": 24, "m": 300}) != ans
                    for m in members)
                st["mand_ok"] += int(ok)
    return stats

corpus = [json.loads(l) for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl'))
          for l in open(f) if l.strip()]
deeds = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
hand = harvest_counts(corpus, probe=True)
mach = harvest_counts(deeds, probe=True)
dial_rows = []
with open('.cache/form_mix8.jsonl') as f:
    for i, l in enumerate(f):
        if i >= 2000: break
        r = json.loads(l)
        dial_rows.append({"factors": r["factors"], "query": r["query_var"],
                          "answer": r["solution"][r["query_var"]]})
dial = harvest_counts(dial_rows, probe=False)

c = conn()
c.executescript("""CREATE TABLE IF NOT EXISTS macro_proposals(
  signature TEXT PRIMARY KEY, size INTEGER, occ_hand INTEGER,
  occ_deeds INTEGER, occ_dialect INTEGER, compression INTEGER,
  port_coherence REAL, hand_only INTEGER, example_srcs TEXT,
  status TEXT, proposed TEXT);""")
rows_out = []
for sig in set(hand) | set(mach):
    h, m = hand.get(sig), mach.get(sig)
    occ_h = h["occ"] if h else 0
    occ_m = m["occ"] if m else 0
    size = (h or m)["size"]
    occ = occ_h + occ_m
    if occ < 3: continue
    mand = ((h["mand_ok"] if h else 0) + (m["mand_ok"] if m else 0)) / occ
    comp = occ * (size - 1)
    d = dial.get(sig, {"occ": 0})["occ"]
    hand_only = int(occ_m == 0 and d == 0)
    ex = ((h["shas"] if h else []) + (m["shas"] if m else []))[:6]
    rows_out.append((sig, size, occ_h, occ_m, d, comp, round(mand, 3),
                     hand_only, json.dumps(ex), "PROPOSED",
                     "2026-08-21"))
for t in rows_out:
    c.execute("INSERT OR REPLACE INTO macro_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?)", t)
c.commit()
rows_out.sort(key=lambda t: -(t[5] * t[6]))
print(f"[macro] mined: hand {len(corpus)} graphs, deeds {len(deeds)}, dialect sample 2000")
print(f"[macro] {len(rows_out)} candidates (occ>=3) -> campaign.db macro_proposals (PROPOSED)")
print(f"{'signature':44s} {'sz':2s} {'hand':4s} {'deed':4s} {'dial':4s} {'comp':4s} {'coh':5s} {'FLAG'}")
for t in rows_out[:14]:
    print(f"{t[0][:44]:44s} {t[1]:2d} {t[2]:4d} {t[3]:4d} {t[4]:4d} {t[5]:4d} {t[6]:5.2f} {'HAND-ONLY' if t[7] else ''}")
c.close()
