"""reads.py — THE READS TABLE (2026-09-18, word given): every number the ledger quotes, as a row in
.cache/rows.sqlite, ingested from the ARTIFACTS (read logs, per-slot npz files, dumps) — never from prose —
with the artifact's path and mtime, the git sha at ingest, and the per-field values. Paired reads are computed
from the per-slot files on demand and stored. The conviction index's assert-on-read becomes `assert`.
usage:
  reads.py ingest                                  scan .cache for read artifacts, upsert
  reads.py table <fixture> [kind]                  the table (kind: open|masked|matched|chain|paired)
  reads.py pair <A> <B> <fixture> [open|legal]     McNemar A vs B from the per-slot files -> a paired row
  reads.py assert <ckpt> <fixture> <kind> <value>  exit 1 if the stored value differs by > 1e-4
  reads.py sha                                     the git sha stored with the latest ingest"""
import os, re, sys, glob, json, sqlite3, subprocess, math, time
import numpy as np
DB = ".cache/rows.sqlite"

def _db():
    db = sqlite3.connect(DB); db.execute("""CREATE TABLE IF NOT EXISTS reads (
        ckpt TEXT, fixture TEXT, kind TEXT, mask TEXT, value REAL, n INTEGER, se REAL,
        vs TEXT, diff REAL, z REAL, gained INTEGER, lost INTEGER,
        pres REAL, ftype REAL, op REAL, args REAL, res REAL, dig REAL,
        artifact TEXT, artifact_mtime TEXT, git_sha TEXT, ingested TEXT, note TEXT,
        PRIMARY KEY (ckpt, fixture, kind, mask, vs))"""); return db

def _sha():
    try: return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: return None

def _fields(line):
    f = {}
    for k in ("pres", "ftype", "op", "args", "res", "dig"):
        m = re.search(rf"\b{k}=([0-9.]+)\(", line)
        if m: f[k] = float(m.group(1))
    return f

def _mtime(p): return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))

def ingest():
    db = _db(); sha = _sha(); now = time.strftime("%Y-%m-%d %H:%M"); n = 0
    def put(row):
        nonlocal n
        db.execute("INSERT OR REPLACE INTO reads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row); n += 1
    # open / masked / fields reads: read_open_<fixture>_<ckpt>.log, read_legal_wild_<ckpt>.log, read_fields_<fixture>_<ckpt>.log
    for p in glob.glob(".cache/read_open_*_*.log") + glob.glob(".cache/read_legal_wild_*.log") + glob.glob(".cache/read_fields_*_*.log"):
        base = os.path.basename(p)[:-4]
        if base.startswith("read_legal_wild_"): fixture, ckpt, kind, mask = "wildhold", base[len("read_legal_wild_"):], "masked", "num"
        else:
            m = re.match(r"read_(open|fields)_([a-z0-9]+)_(.+)$", base)
            if not m: continue
            fixture, ckpt, kind, mask = m.group(2), m.group(3), "open", "none"
            if fixture == "wild": fixture = "wildhold"
        txt = open(p, errors="ignore").read(); mm = re.findall(r"fac-exact=([0-9.]+) \(n=(\d+)\)", txt)
        if not mm: continue
        val, cnt = float(mm[-1][0]), int(mm[-1][1]); fl = re.findall(r"^\[fields\].*$", txt, re.M); f = _fields(fl[-1]) if fl else {}
        se = math.sqrt(val * (1 - val) / max(cnt, 1))
        put((ckpt, fixture, kind, mask, val, cnt, se, "", None, None, None, None, f.get("pres"), f.get("ftype"), f.get("op"), f.get("args"), f.get("res"), f.get("dig"), p, _mtime(p), sha, now, None))
    # chain accuracy: any chain_acc*.log
    for p in glob.glob(".cache/chain_acc*.log"):
        for line in open(p, errors="ignore"):
            m = re.match(r"\[chain-acc\] sharp_(.+?)\.safetensors on (\w+) mask=(\d): rows (\d+) \| CORRECT (\d+) \(([0-9.]+)\) \| refused (\d+).*\| wrong (\d+)", line)
            if not m: continue
            ckpt, fixture, mask, rows, correct, val, refused, wrong = m.group(1), m.group(2), "num" if m.group(3) == "1" else "none", int(m.group(4)), int(m.group(5)), float(m.group(6)), int(m.group(7)), int(m.group(8))
            put((ckpt, fixture, "chain", mask, val, rows, math.sqrt(val * (1 - val) / rows), "", None, None, None, None, None, None, None, None, None, None, p, _mtime(p), sha, now, f"correct {correct} refused {refused} wrong {wrong}"))
    # matched reads: recompute from the dumps (cheap, deterministic)
    sys.path.insert(0, "scripts")
    try:
        import matched_read as MR, pickle, collections
        for p in glob.glob(".cache/dump_wild_*.pkl") + glob.glob(".cache/dump_wildhold_*.pkl"):
            ckpt = re.match(r".*dump_wild(?:hold)?_(.+)\.pkl$", p).group(1); D = pickle.load(open(p, "rb")); rows = collections.defaultdict(list)
            for t in D: rows[t[0]].append(t[1:])
            pos = mat = tot = 0
            for i, slots in rows.items():
                slots.sort(key=lambda s: s[0]); tot += len(slots); pos += MR.positional(slots); ok, _ = MR.match_row(slots); mat += sum(ok.values())
            put((ckpt, "wildhold", "matched", "none", mat / tot, tot, math.sqrt((mat / tot) * (1 - mat / tot) / tot), "", None, None, None, None, None, None, None, None, None, None, p, _mtime(p), sha, now, f"positional {pos/tot:.4f}"))
    except Exception as e: print("matched ingest skipped:", e)
    db.commit(); print(f"[reads] ingested/updated {n} rows (git {sha}) -> {DB}"); db.close()

def pair(A, B, fixture, mode="open"):
    fx = "wild" if fixture == "wildhold" else fixture; pre = "ps_legal_wild" if mode == "legal" else f"ps_open_{fx}"
    da, dbb = np.load(f".cache/{pre}_{A}.npz"), np.load(f".cache/{pre}_{B}.npz")
    ka = {(int(r), int(s)): bool(o) for r, s, o in zip(da["rows"], da["slots"], da["ok"])}; kb = {(int(r), int(s)): bool(o) for r, s, o in zip(dbb["rows"], dbb["slots"], dbb["ok"])}
    keys = sorted(set(ka) & set(kb)); a = np.array([ka[k] for k in keys]); b = np.array([kb[k] for k in keys])
    g = int((b & ~a).sum()); l = int((a & ~b).sum()); z = (g - l) / math.sqrt(max(g + l, 1)); diff = b.mean() - a.mean(); se = math.sqrt(g + l) / len(keys)
    db = _db(); db.execute("INSERT OR REPLACE INTO reads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (B, fixture, "paired", "num" if mode == "legal" else "none", float(b.mean()), len(keys), se, A, float(diff), z, g, l, None, None, None, None, None, None, f".cache/{pre}_{B}.npz", _mtime(f".cache/{pre}_{B}.npz"), _sha(), time.strftime("%Y-%m-%d %H:%M"), f"{A} {a.mean():.4f} -> {B} {b.mean():.4f}"))
    db.commit(); db.close(); print(f"[paired] {fixture} {mode}: {A} {a.mean():.4f} vs {B} {b.mean():.4f} | diff {diff:+.4f} | gained {g} lost {l} | z {z:+.2f} | n {len(keys)}")

def table(fixture, kind=None):
    db = _db(); q = "SELECT ckpt, kind, mask, value, n, se, vs, diff, z, ftype, res, args, dig, artifact FROM reads WHERE fixture=?" + (" AND kind=?" if kind else "") + " ORDER BY kind, value DESC"
    for r in db.execute(q, (fixture, kind) if kind else (fixture,)):
        ckpt, k, mask, v, n, se, vs, diff, z, ft, res, args, dig, art = r
        extra = f" vs {vs} diff {diff:+.4f} z {z:+.2f}" if vs else (f" | ftype {ft:.3f} res {res:.3f} args {args:.3f} dig {dig:.3f}" if ft is not None else "")
        print(f"  {ckpt:<18} {k:<8} {mask:<5} {v:.4f} (n {n}, se {se:.4f}){extra}   <- {os.path.basename(art)}")
    db.close()

def assert_read(ckpt, fixture, kind, value):
    db = _db(); r = db.execute("SELECT value, artifact FROM reads WHERE ckpt=? AND fixture=? AND kind=? ORDER BY ingested DESC LIMIT 1", (ckpt, fixture, kind)).fetchone(); db.close()
    if r is None: print(f"[assert] NO READ on record for {ckpt} {fixture} {kind} — the number has no artifact"); sys.exit(1)
    ok = abs(r[0] - float(value)) <= 1e-4; print(f"[assert] {ckpt} {fixture} {kind}: quoted {float(value):.4f} vs artifact {r[0]:.4f} ({os.path.basename(r[1])}) -> {'OK' if ok else 'MISMATCH'}"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "ingest": ingest()
    elif c == "table": table(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif c == "pair": pair(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "open")
    elif c == "assert": assert_read(*sys.argv[2:6])
    elif c == "sha": print(_sha())
