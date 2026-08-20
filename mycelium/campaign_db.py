"""campaign_db.py — THE TWO ENABLERS (2026-08-20, the word given):
(1) THE REGISTRY DB — the ftype alphabet + macro catalog as a typed,
queryable store (certificates, regime tags, expirations; the
rank-never-admit door's ledger). (2) THE CONFLICT STORE — the refutation
tree's cores persisted (text-sha keyed per custody law): which
constructions make the parser contradict itself = the books campaign's
targeting signal. SQLite, embedded, .cache-resident. No daemons."""
import sqlite3, hashlib, json, os

DB = os.path.join(os.path.dirname(__file__), "..", ".cache", "campaign.db")

def conn():
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS ftypes(
      name TEXT PRIMARY KEY, code INTEGER, since TEXT, status TEXT,
      certificate TEXT, regime TEXT, expiration TEXT);
    CREATE TABLE IF NOT EXISTS macros(
      name TEXT PRIMARY KEY, expansion TEXT, admitted TEXT,
      receipts TEXT, grammar_version TEXT);
    CREATE TABLE IF NOT EXISTS residents(
      name TEXT PRIMARY KEY, filed TEXT, evidence TEXT, status TEXT);
    CREATE TABLE IF NOT EXISTS waist_patterns(
      cluster_id INTEGER PRIMARY KEY, count INTEGER,
      mean BLOB, m2 BLOB,               -- Welford running stats (512-d)
      register TEXT, first_seen TEXT, last_seen TEXT,
      macro_candidate TEXT DEFAULT NULL);  -- rank-never-admit fills this
    CREATE TABLE IF NOT EXISTS conflict_cores(
      text_sha TEXT, register TEXT, diagnosis TEXT,
      core_factor TEXT, all_factors TEXT, recorded TEXT,
      PRIMARY KEY(text_sha, core_factor));
    """)
    return c

def seed():
    c = conn()
    ft = [("rel",0),("given",1),("mod",2),("sel",3),("pct",4),
          ("fdiv",5),("macro",6),("frac",7),("chain",8)]
    for n,k in ft:
        c.execute("INSERT OR IGNORE INTO ftypes VALUES(?,?,?,?,?,?,?)",
                  (n,k,"pre-2026-08","LIVE","deployed in g41 lineage",
                   "gen41+","none"))
    c.execute("INSERT OR IGNORE INTO macros VALUES(?,?,?,?,?)",
              ("CHAIN_MUL","left-fold rel-muls through temps (mg3)",
               "2026-08 (tower)","200/200 exact through key; sight 100/100",
               "mg3"))
    c.execute("INSERT OR IGNORE INTO residents VALUES(?,?,?,?)",
              ("rate-family","2026-07","trunk frame-entanglement z=-2.05",
               "STANDING"))
    c.execute("INSERT OR IGNORE INTO residents VALUES(?,?,?,?)",
              ("chained-fdiv","2026-07","autopsy staged: derived-value digit path",
               "STANDING"))
    c.commit(); c.close()

def record_core(text, register, diagnosis, core_factor, all_factors, when):
    c = conn()
    c.execute("INSERT OR IGNORE INTO conflict_cores VALUES(?,?,?,?,?,?)",
              (hashlib.sha256(text.encode()).hexdigest()[:16], register,
               diagnosis, json.dumps(core_factor), json.dumps(all_factors), when))
    c.commit(); c.close()

def core_census():
    c = conn()
    rows = c.execute("SELECT diagnosis, count(*) FROM conflict_cores GROUP BY diagnosis").fetchall()
    c.close(); return rows
