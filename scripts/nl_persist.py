"""nl_persist.py — THE PERSISTENCE CHORE (2026-08-29, word given): the
NL side earns its tables. nl_registry (entries: class, flavor, pattern,
kind lexical|symbolic, guards, provenance, stats) + nl_atlas (semantic
families: class, cycle [NULL until R4], Welford count/mean/M2) +
nl_anchors (provenance pointers: family -> source row/span/cue — audit
trail, never payloads; raw vectors recomputable from frozen forward).
Keys per the binding theorem: (class, family, cycle) — slot_id is
provenance only, never an aggregation key.
"""
import json, re, os, sys, sqlite3, hashlib
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, load_alg
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
OPS = ("addf", "mul", "sq", "fr")
db = sqlite3.connect('.cache/campaign.db')
db.execute("""CREATE TABLE IF NOT EXISTS nl_registry(
  entry_id INTEGER PRIMARY KEY, class TEXT, flavor TEXT, pattern TEXT,
  kind TEXT, guards TEXT, provenance TEXT, n_seen INTEGER)""")
db.execute("""CREATE TABLE IF NOT EXISTS nl_atlas(
  family_id INTEGER PRIMARY KEY, class TEXT, cycle INTEGER,
  count INTEGER, mean BLOB, m2 BLOB)""")
db.execute("""CREATE TABLE IF NOT EXISTS nl_anchors(
  anchor_id INTEGER PRIMARY KEY, family_id INTEGER, source TEXT,
  span_a INTEGER, span_b INTEGER, cue TEXT)""")
db.execute("DELETE FROM nl_registry"); db.execute("DELETE FROM nl_atlas")
db.execute("DELETE FROM nl_anchors")

# ---- registry: wild entries (provenance = book3 row) + mint-mined ----
NUMRE = re.compile(r"\d+")
eid = 0
wild = []
for li, l in enumerate(open('.cache/book3.jsonl')):
    r = json.loads(l)
    for s in (r.get('op_spans') or []):
        if s['op'] not in OPS or not (2 <= len(s['cue']) <= 60): continue
        eid += 1
        prov = json.dumps({"book": 3, "lane_idx": r.get('lane_idx'),
                           "tranche": r.get('tranche'), "source": s.get('source', 'tranche')})
        db.execute("INSERT INTO nl_registry VALUES(?,?,?,?,?,?,?,?)",
                   (eid, s['op'], s['op'], s['cue'].lower(), 'lexical', '', prov, 1))
        wild.append((r['raw'], s['span'][0], s['span'][1], s['op'], li))
def opg(f):
    if f["ftype"] == "rel":
        if f.get("op") == "mul" and len(set(f.get("args", []))) == 1: return "sq"
        return {"add": "addf", "sub": "addf", "mul": "mul"}.get(f.get("op"))
    if f["ftype"] == "macro": return None if f.get("name") == "OP_APPLY" else "fr"
    if f["ftype"] == "frac": return "fr"
    return None
mint = {c: Counter() for c in OPS}
mint_rows = []
n = 0
for i, l in enumerate(open('.cache/form_mix8.jsonl')):
    if n >= 1500: break
    r = json.loads(l); txt = r.get('text') or ''
    got = False
    for f in r.get('factors', []):
        c = opg(f)
        if c is None: continue
        for (a, b) in (f.get('spans') or []):
            ph = NUMRE.sub('#', txt[a:b].lower()).strip()
            if 3 <= len(ph) <= 60:
                mint[c][ph] += 1
                mint_rows.append((txt, a, b, c, i)); got = True
    if got: n += 1
for c in OPS:
    for ph, cnt in mint[c].items():
        if cnt >= 5 and max(mint[c2][ph] for c2 in OPS if c2 != c) * 3 <= cnt:
            eid += 1
            db.execute("INSERT INTO nl_registry VALUES(?,?,?,?,?,?,?,?)",
                       (eid, c, c, ph, 'lexical',
                        '', json.dumps({"source": "mint-mined", "corpus": "form_mix8"}), cnt))
# symbolic register entries (the scanner's rules, as first-class rows)
for cls, pat, guard in [("addf", r"\+", ""), ("addf", r"binary-minus", "not-unary"),
                        ("mul", r"\\cdot|\\times", ""), ("mul", r"digit-letter-juxtapose", "no-func-names"),
                        ("sq", r"\^\{?2\}?", "not-^2x"), ("fr", r"\\d?frac", "numeric-frac-is-value"),
                        ("fr", r"\\div", "")]:
    eid += 1
    db.execute("INSERT INTO nl_registry VALUES(?,?,?,?,?,?,?,?)",
               (eid, cls, cls, pat, 'symbolic', guard,
                json.dumps({"source": "scanner-v2"}), 0))
db.commit()
print(f"[persist] nl_registry: {eid} entries", flush=True)

# ---- atlas: recompute anchors -> leader families -> Welford persist ----
sites = {}
for (txt, a, b, c, ref) in wild + mint_rows:
    sites.setdefault(txt, []).append((a, b, c, ref))
texts = list(sites.keys())
fam = {c: [] for c in OPS}      # per class: list of [count, mean, M2, anchors]
def welford_add(F, v):
    F[0] += 1
    d = v - F[1]
    F[1] += d / F[0]
    F[2] += d * (v - F[1])
for s0 in range(0, len(texts), 8):
    sl = texts[s0:s0+8]
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    offs = []
    for i, t in enumerate(sl):
        e = tok.encode(t)
        tid = e.ids[:T_ALG]
        ids[i, :len(tid)] = tid; msk[i, :len(tid)] = 1.0
        offs.append(list(e.offsets)[:T_ALG])
    st = np.asarray(recompute_states(ids)).astype(np.float32)
    for i, t in enumerate(sl):
        for (a, b, c, ref) in sites[t]:
            idxs = [k for k, o in enumerate(offs[i]) if o[1] > a and o[0] < b]
            if not idxs: continue
            v = st[i][idxs].mean(0)
            v = v / (np.linalg.norm(v) + 1e-9)
            best, bj = 1.0, -1
            for j, F in enumerate(fam[c]):
                m = F[1] / (np.linalg.norm(F[1]) + 1e-9)
                d = 1 - float(m @ v)
                if d < best: best, bj = d, j
            if bj >= 0 and best < 0.15:
                welford_add(fam[c][bj], v)
                fam[c][bj][3].append((str(ref), a, b, t[a:b][:60]))
            else:
                fam[c].append([1, v.copy(), np.zeros_like(v), [(str(ref), a, b, t[a:b][:60])]])
fid = 0; aid = 0
for c in OPS:
    for F in fam[c]:
        fid += 1
        db.execute("INSERT INTO nl_atlas VALUES(?,?,?,?,?,?)",
                   (fid, c, None, F[0],
                    F[1].astype(np.float32).tobytes(), F[2].astype(np.float32).tobytes()))
        for (src, a, b, cue) in F[3][:20]:
            aid += 1
            db.execute("INSERT INTO nl_anchors VALUES(?,?,?,?,?,?)",
                       (aid, fid, src, a, b, cue))
db.commit()
nf = db.execute("SELECT COUNT(*) FROM nl_atlas").fetchone()[0]
na = db.execute("SELECT COUNT(*) FROM nl_anchors").fetchone()[0]
per = dict(db.execute("SELECT class, COUNT(*) FROM nl_atlas GROUP BY class"))
print(f"[persist] nl_atlas: {nf} families {per}; nl_anchors: {na}", flush=True)
db.close()
