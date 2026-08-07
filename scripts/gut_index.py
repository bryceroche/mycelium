"""gut_index.py — regenerates docs/GUT_INDEX.md from THE LEDGER
(two-home law: the index is DERIVED, never edited by hand; re-run
after any registration. The ledger remains the sole authority)."""
import re
src = open("docs/phase1_skeleton_spec.md").read()
pat = re.compile(r'\*\*GUT #(\d+)(?:\s*\(([^)]*)\))?\s*(?:REGISTERED)?\s*[—:+]\s*[“"]?([^”"(\n]+)', )
lines = []
for m in pat.finditer(src):
    num, paren, name = m.group(1), m.group(2) or "", m.group(3).strip().rstrip('"” ')
    # date + first content clause from the entry's opening
    tail = src[m.end():m.end()+400]
    dm = re.search(r'\((\d{4}-\d{2}-\d{2})', tail) or re.search(r'(\d{4}-\d{2}-\d{2})', tail)
    date = dm.group(1) if dm else ""
    hook = re.sub(r'\s+', ' ', tail.split(':**')[0]).strip()
    hook = re.sub(r'^[”"\s]*\((\d{4}-\d{2}-\d{2})[;,]?\s*', '', hook).rstrip(') ')[:90]
    lines.append((int(num), name, date, hook))
lines.sort()
seen = set(); out = []
for num, name, date, hook in lines:
    if num in seen: continue
    seen.add(num)
    out.append(f"- **#{num}** {name} ({date}) — {hook}")
hdr = ("# GUT INDEX — derived from THE LEDGER (do not edit; regenerate via "
       "scripts/gut_index.py)\n\nOne line per registration; full forms, rents, "
       "verdicts, and countersigns live in docs/phase1_skeleton_spec.md at the "
       "entry. The ledger is the sole authority.\n\n")
open("docs/GUT_INDEX.md", "w").write(hdr + "\n".join(out) + "\n")
print(f"index: {len(out)} guts (#{out and min(seen)}–#{max(seen)})")
