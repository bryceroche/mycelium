"""resolve_refused_merge.py -- THE LOOPED TRANSFORMER's queue (2026-10-09): a narrow, mechanical
resolver for EXACTLY ONE conflict shape in scripts/step_trainer.py's REFUSED tuple -- two
worktree branches (this one and a sibling, e.g. waistskip) each append their own new quoted
ALG_* flag(s) on their own new line at the END of the tuple, per the standing instruction both
builds were given ("own new line at the END of the list"). A real git conflict on the tuple's
tail is therefore a UNION, never a real divergence: take "ours" (this branch's own tuple,
already correct and complete for ITS flags) and splice in any NEW quoted ALG_* tokens "theirs"
added that are not already present, immediately before ours's own closing paren.

Usage: resolve_refused_merge.py <path-with-conflict-markers>
Exits nonzero (loudly, no silent fallback) if the file has no conflict markers, more than one
conflict region, or the reconstructed file fails to parse as Python -- a human/agent then
resolves it by hand.
"""
import ast
import re
import sys

path = sys.argv[1]
with open(path) as f:
    text = f.read()

OURS_RE = re.compile(r"<<<<<<< [^\n]*\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]*\n", re.DOTALL)
matches = list(OURS_RE.finditer(text))
if not matches:
    print(f"ABORT: no conflict markers found in {path}")
    sys.exit(1)
if len(matches) > 1:
    print(f"ABORT: {len(matches)} conflict regions in {path} -- this resolver handles exactly one")
    sys.exit(1)

m = matches[0]
ours, theirs = m.group(1), m.group(2)

TOKEN_RE = re.compile(r'"([A-Z][A-Z0-9_]*)"')
ours_tokens = set(TOKEN_RE.findall(ours))
theirs_tokens = set(TOKEN_RE.findall(theirs))
new_tokens = sorted(theirs_tokens - ours_tokens)

if not new_tokens:
    print("[resolve-refused] theirs added no new ALG_* tokens beyond ours -- keeping ours verbatim")
    resolved_region = ours
else:
    # find ours's own closing paren of the tuple -- the line containing a lone ")" at the end
    # of a tuple literal, i.e. the last line of `ours` that ends with ")" (optionally followed
    # by a trailing comment, which there should not be on this specific line by house style).
    lines = ours.split("\n")
    CLOSE_RE = re.compile(r"^(?P<pre>\s*\S.*?)\)(?P<comment>\s*#.*)?$")
    close_idx = None
    close_m = None
    for i in range(len(lines) - 1, -1, -1):
        cm = CLOSE_RE.match(lines[i])
        if cm:
            close_idx = i
            close_m = cm
            break
    if close_idx is None:
        print("ABORT: could not find ours's own tuple-closing line to splice into")
        sys.exit(1)
    indent = re.match(r"^(\s*)", lines[close_idx]).group(1)
    # change ours's closing ")" to "," (dropping its own trailing comment, re-attached to the
    # NEW closing line below so nothing is silently lost) and append a new line with theirs's
    # new tokens + ")"
    lines[close_idx] = close_m.group("pre") + ","
    comment = close_m.group("comment") or ""
    new_line = indent + ", ".join(f'"{t}"' for t in new_tokens) + ")" + comment
    lines.insert(close_idx + 1, new_line)
    resolved_region = "\n".join(lines)
    print(f"[resolve-refused] spliced {len(new_tokens)} new token(s) from theirs into ours: {new_tokens}")

resolved_text = text[:m.start()] + resolved_region + text[m.end():]

try:
    ast.parse(resolved_text)
except SyntaxError as e:
    print(f"ABORT: resolved {path} does not parse as Python: {e}")
    sys.exit(1)

with open(path, "w") as f:
    f.write(resolved_text)
print(f"[resolve-refused] wrote resolved {path}")
