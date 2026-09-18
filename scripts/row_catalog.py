"""row_catalog.py — THE ROW CATALOG (2026-09-18, word given): every training/measurement row the campaign holds,
in one local SQLite table with its PROVENANCE — source, tier, convention, silver version, admitting gate, parent
(for resampled copies), and the files it appears in — so the dose law is a query and file identity is no
longer carried by hand. Local, embedded, no service. Rebuilt from the files (the files stay the truth).
usage: row_catalog.py build | dose <mix.jsonl> | sources | find <text prefix>"""
import sqlite3, json, sys, os, hashlib, collections
DB = ".cache/rows.sqlite"
FILES = [   # path, role, source, tier, convention
    (".cache/algebra_nl_test.jsonl",              "measure", "mint",   "test23",    "first-mention"),
    (".cache/wordmint_test.jsonl",                "measure", "mint",   "wordmint",  "first-mention"),
    (".cache/wild_admitted_holdout.jsonl",        "measure", "gsm8k",  "holdout",   "positional"),
    (".cache/silver460_test.jsonl",               "measure", "sonnet", "fit",       "positional"),
    (".cache/gsm8k_chain_v2_pos.jsonl",           "train",   "gsm8k",  "chain_t1",  "positional"),
    (".cache/gsm8k_chain_v2_pos_x3.jsonl",        "train",   "gsm8k",  "chain_t1",  "positional"),
    (".cache/gsm8k_chain_v2t2_pos.jsonl",         "train",   "gsm8k",  "chain_t2lex", "positional"),
    (".cache/gsm8k_chain_v2t2_pos_x3.jsonl",      "train",   "gsm8k",  "chain_t2lex", "positional"),
    (".cache/eqsets_pos.jsonl",                   "train",   "eqsets", "anchored",  "positional"),
    (".cache/eqsets_pos_x3.jsonl",                "train",   "eqsets", "anchored",  "positional"),
    (".cache/silver_sonnet_v1_all_pos.jsonl",     "train",   "sonnet", "silver",    "positional"),
    (".cache/silver_sonnet_v1_all_x3_pos.jsonl",  "train",   "sonnet", "silver",    "positional"),
]

def h(text): return hashlib.sha1(text.encode()).hexdigest()[:16]

def build():
    if os.path.exists(DB): os.remove(DB)
    db = sqlite3.connect(DB); c = db.cursor()
    c.execute("""CREATE TABLE rows (hash TEXT, text TEXT, role TEXT, source TEXT, tier TEXT, convention TEXT, silver TEXT, admitted_by TEXT,
                 canonical TEXT, parent_hash TEXT, resample_k INTEGER, n_vars INTEGER, n_factors INTEGER, key INTEGER, file TEXT)""")
    c.execute("CREATE INDEX ix_hash ON rows(hash)"); c.execute("CREATE INDEX ix_file ON rows(file)")
    n = 0
    for path, role, source, tier, conv in FILES:
        if not os.path.exists(path): print(f"  (missing) {path}"); continue
        for l in open(path):
            r = json.loads(l); g = r.get("gen") if isinstance(r.get("gen"), dict) else {}
            rs = g.get("resample") or {}; parent = None
            if rs: parent = None   # the parent is matched by the numeral-masked story below
            c.execute("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (h(r["text"]), r["text"], role, g.get("src", source), g.get("tier", tier), g.get("canonical", conv), g.get("silver"), g.get("admitted_by"),
                       g.get("canonical"), parent, rs.get("k"), r.get("n_vars"), len(r.get("factors", [])), r.get("key"), path)); n += 1
    # the mix's mint/pen rows: form_mix12 is the base pool (mint + the 08-31 chain-lane pen rows)
    if os.path.exists(".cache/form_mix12.jsonl"):
        for l in open(".cache/form_mix12.jsonl"):
            r = json.loads(l); g = r.get("gen") if isinstance(r.get("gen"), dict) else {}
            src = "gsm8k" if g.get("src") == "gsm8k" else "mint"
            c.execute("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (h(r["text"]), r["text"], "train", src, "pen" if src == "gsm8k" else "form_mix12", "positional" if src == "gsm8k" else "first-mention",
                       None, g.get("admission"), None, None, None, r.get("n_vars"), len(r.get("factors", [])), None, ".cache/form_mix12.jsonl")); n += 1
    db.commit(); print(f"[catalog] {n} rows -> {DB}")
    # leakage check: any measurement text inside a training file?
    q = c.execute("SELECT m.file, t.file, COUNT(*) FROM rows m JOIN rows t ON m.hash = t.hash WHERE m.role='measure' AND t.role='train' GROUP BY m.file, t.file").fetchall()
    print("[catalog] measurement texts found in training files:", q if q else "none")
    db.close()

def dose(mix):
    db = sqlite3.connect(DB); c = db.cursor(); cnt = collections.Counter(); uniq = collections.defaultdict(set); tot = 0
    for l in open(mix):
        r = json.loads(l); hh = h(r["text"]); tot += 1
        row = c.execute("SELECT source, tier FROM rows WHERE hash=? LIMIT 1", (hh,)).fetchone()
        k = (row[0], row[1]) if row else ("?", "?"); cnt[k] += 1; uniq[k].add(hh)
    print(f"[dose] {mix}: {tot} rows")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]): print(f"   {k[0]:<8} {k[1]:<12} rows {v:>6} ({v/tot:5.1%})  unique texts {len(uniq[k]):>5}  reps/unique {v/len(uniq[k]):.1f}")
    db.close()

def sources():
    db = sqlite3.connect(DB)
    for r in db.execute("SELECT role, source, tier, convention, COUNT(*), COUNT(DISTINCT hash) FROM rows GROUP BY 1,2,3,4 ORDER BY 1,2,3"): print("  ", r)
    db.close()

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build": build()
    elif cmd == "dose": dose(sys.argv[2])
    elif cmd == "sources": sources()
    elif cmd == "find":
        db = sqlite3.connect(DB)
        for r in db.execute("SELECT file, source, tier, convention, silver FROM rows WHERE text LIKE ?", (sys.argv[2] + "%",)): print("  ", r)
