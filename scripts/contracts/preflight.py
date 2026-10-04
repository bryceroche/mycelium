"""preflight.py — THE PRE-FLIGHT CONTRACT (hill 5, ledger 2026-10-04 08:09).

Before any chain fires, assert every arm's and baseline's .cache inputs
exist — a missing baseline must crash at step 0, not three hours in (the
caric chain died on a missing control dump; the tree chain's collect
overwrote a baseline). Call at a chain's top, right after `set -eo
pipefail`:

    .venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1

(`$0` resolves to the chain's own path under both direct invocation and
`systemd-run --user ... /usr/bin/bash <chain>` — bash sets $0 to the
script path given to it either way.)

WHAT THIS DOES: parses the chain's own shell text (never sources or runs
it — a chain fires real GPU training) and recovers, as closely as a
static read can, which .cache paths it REQURES already present and which
it PRODUCES itself (and therefore need not pre-exist). It is a reader of
THIS REPO'S chain idiom (the four chains docs/phase1_skeleton_spec.md's
hill-5 paragraph names: tree2_chain.sh, certloop_chain.sh, caric_chain.sh,
unet_bakeoff_chain.sh), not a general bash interpreter — the supported
shapes are:
  - top-level `NAME="literal with $OTHER refs"` env-string assignments
    (FAM, TR, SURF8, W/M/S, RS8, TRB, LOCK, ...), inlined once, in order,
    before anything else is parsed;
  - `for VAR in "a:b" "c:d" ...; do BODY done` loops (single-line or
    spanning lines), including `NAME=${VAR%%:*}` / `NAME=${VAR#*:}`
    colon-split idiom right after the loop header; expanded by cross
    product (sequential sibling loops multiply, nested loops multiply) —
    small item counts in practice (<=6 total combinations per chain);
  - `until grep -aq "PATTERN" FILE; do sleep N; done` gate waits and the
    standalone `grep -a[q] "PATTERN" FILE ... || { echo ABORT; exit 1; }`
    guards that follow — the referenced _gate.log is a required input,
    and the PATTERN is actually re-grepped against it here (so a stale
    gate shows as a fresh FAIL before the chain would hang 10 hours on
    `until`);
  - known script flags (env KEY=value: LV_CKPT/LV_DUMP/LV_PER_SLOT,
    CA_CKPT/CA_RAWDUMP/CA_ROWS, MS_CKPT, WARM_FROM, UN_CKPT, BIND_CODES,
    ALG_POLAR_D_INIT, ALG_TRAIN(_NAME)/ALG_TEST(_NAME), BO_DUMP) and
    known scripts' positional CLI args (paired_read.py, matched_read.py,
    busreg_cell_compare.py, twin_gap.py / certifier_signal_census.py's
    TAG -> rawslots_wild_<TAG>.pkl);
  - `flock -w N LOCKFILE`, `cd DIR;` (the working directory), and a
    `[ -f PATH ] || PRODUCING_CMD` conditional-produce idiom (caric's
    rawslots fallback).

A path is PRODUCED-BY-CHAIN if some earlier line (in execution order,
across every expanded iteration) writes it (ALG_CKPT/LV_DUMP/LV_PER_SLOT/
CA_RAWDUMP/CA_ROWS/UN_CKPT-from-train.py/a `> .cache/x.log` redirect/
membrane_scale.py's MS_MODE=collect outputs derived from MS_CKPT's TAG);
otherwise it is REQUIRED and checked against the filesystem now.

Exit code: 0 iff every REQUIRED input exists and every re-checked gate
log's pattern is present; non-zero otherwise (printed table says which).
systemctl's failed pc-* units and an absent working directory are WARNED,
never fail the exit code on their own (per the brief: "warn").
"""
import os
import re
import shlex
import subprocess
import sys

ROOT = "/home/bryce/mycelium"

# ---------------------------------------------------------------------------
# env keys this reader understands, and what they mean
# ---------------------------------------------------------------------------
# keys that are ALWAYS a write (the named script always produces this path)
ALWAYS_WRITE_KEYS = {"ALG_CKPT", "LV_DUMP", "LV_PER_SLOT", "CA_RAWDUMP", "CA_ROWS"}
# keys that are ALWAYS a read of a pre-existing, never-chain-produced file
ALWAYS_READ_KEYS = {"BIND_CODES", "ALG_POLAR_D_INIT", "ALG_TRAIN", "ALG_TEST"}
# keys that are a read of a checkpoint/dump that MAY have been produced
# earlier in this same chain (classified read-or-produced by the running
# `produced` set at the point they're encountered)
MAYBE_PRODUCED_READ_KEYS = {"LV_CKPT", "CA_CKPT", "MS_CKPT", "WARM_FROM", "BO_DUMP"}
# split-name keys -> derive the cached-states files they require
SPLIT_NAME_KEYS = {"ALG_TRAIN_NAME", "ALG_TEST_NAME"}
# UN_CKPT is a write when the command on its line is unet/train.py, else a read
AMBIGUOUS_KEYS = {"UN_CKPT"}

POSITIONAL_SPECS = {
    # script basename -> how many leading whitespace-delimited, path-shaped
    # tokens after it are read args, and a function turning a token into
    # the actual .cache path(s) it names (default: identity)
    "paired_read.py": (2, None),
    "matched_read.py": (1, None),   # at least one; we also grab a 2nd if path-shaped
    "busreg_cell_compare.py": (5, None),
    "twin_gap.py": (1, lambda tag: [f".cache/rawslots_wild_{tag}.pkl"]),
    "certifier_signal_census.py": (1, lambda tag: [f".cache/rawslots_wild_{tag}.pkl",
                                                     f".cache/beam_oracle_{tag}.log"]),
}

TOKEN_RE = re.compile(r"^[.\w/+=-]+$")


def strip_comments(text):
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def join_continuations(text):
    """Fold `cmd \\\n  more` backslash line-continuations into one logical
    line, so a flag on one physical line (e.g. UN_CKPT=$CKPT) and the
    script name that would classify it (scripts/unet/train.py) two lines
    later are visible to the same per-line scan."""
    return re.sub(r"\\\s*\n\s*", " ", text)


def extract_top_vars(text):
    """NAME="..." (possibly several `;`-separated on one line) anywhere
    BEFORE the chain's first `for VAR in ...` loop — these chains always
    define FAM/TR/W/M/S/SURF8/... up there, and loop-local derived vars
    (`C=${ARM%%:*}`) only ever appear INSIDE a loop body, so splitting on
    the first `for` keyword keeps the two apart. Later defs may reference
    earlier ones via $NAME/${NAME}. Returns (vars, text_with_the_head_
    portion's_assignment_lines_removed + the untouched tail)."""
    fm = re.search(r"\bfor\s+\w+\s+in\s+", text)
    head, tail = (text[:fm.start()], text[fm.start():]) if fm else (text, "")
    vars_ = {}
    out_lines = []
    for line in head.splitlines():
        if line.lstrip().startswith(("if ", "elif", "until ", "grep ")):
            out_lines.append(line)
            continue
        segs = line.split(";")
        leftover = []
        for seg in segs:
            m = re.match(r'^\s*([A-Z][A-Z0-9_]*)=("(?:[^"\\]|\\.)*"|\S+)\s*$', seg)
            if m:
                name, val = m.group(1), m.group(2)
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                val = subst(val, vars_)
                vars_[name] = val
            else:
                leftover.append(seg)
        if leftover:
            out_lines.append(";".join(leftover))
    return vars_, "\n".join(out_lines) + "\n" + tail


def subst(text, vars_):
    def repl_braced(m):
        name, default = m.group(1), m.group(2)
        if name in vars_:
            return vars_[name]
        return default if default is not None else m.group(0)
    text = re.sub(r"\$\{(\w+):-([^}]*)\}", repl_braced, text)
    for name, val in vars_.items():
        text = re.sub(r"\$\{%s\}" % re.escape(name), val, text)
        text = re.sub(r"\$%s\b" % re.escape(name), val, text)
    return text


def find_first_for(text):
    """Find the first `for VAR in ITEMS; do BODY done`, matched by a
    do/done depth count (every bash loop keyword — for/while/until — closes
    with `done`, so counting bare do/done is construct-agnostic). Returns
    (prefix, var, items_text, body, suffix) or None."""
    m = re.search(r"\bfor\s+(\w+)\s+in\s+(.*?);\s*do\b", text, re.S)
    if not m:
        return None
    prefix = text[:m.start()]
    var, items_text = m.group(1), m.group(2)
    rest = text[m.end():]
    depth = 1
    pos = 0
    for tok in re.finditer(r"\b(do|done)\b", rest):
        if tok.group(1) == "do":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                pos = tok.start()
                break
    else:
        raise ValueError("unbalanced do/done while scanning a for-loop body")
    body = rest[:pos]
    suffix = rest[pos + len("done"):]
    return prefix, var, items_text, body, suffix


def absorb_assignments(text, vars_):
    """A second, narrower pass beside extract_top_vars: catches a plain
    `NAME="...${loopvar}..."` assignment written FRESH inside a loop body
    (e.g. unet_bakeoff_chain.sh's `CKPT=".cache/unet_${ARM}_241.safetensors"`)
    once the loop var is already substituted in and the value is fully
    resolved (no `$` left) — so a line like `C=${ARM%%:*}` belonging to the
    colon-split idiom (handled elsewhere, deliberately not here) is never
    touched, since it still contains a literal `$` at this point regardless
    of whether ARM is bound (the %%:* modifier means subst's plain-${NAME}
    regex does not match it). Returns an updated vars_ (new dict)."""
    vars_ = dict(vars_)
    for line in text.splitlines():
        if re.search(r"\bfor\s+\w+\s+in\b", line):
            continue
        for seg in line.split(";"):
            m = re.match(r'^\s*([A-Z][A-Z0-9_]*)=("(?:[^"\\]|\\.)*"|\S+)\s*$', seg)
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if "$" in val:
                continue   # still has an unbound reference — leave for later/elsewhere
            vars_[name] = val
    return vars_


def expand_loops(text, vars_):
    text = subst(text, vars_)
    vars_ = absorb_assignments(text, vars_)
    text = subst(text, vars_)
    found = find_first_for(text)
    if found is None:
        return [text]
    prefix, var, items_text, body, suffix = found
    items = shlex.split(items_text)
    chunks = []
    for item in items:
        local = dict(vars_)
        local[var] = item
        # the colon-split idiom: NAME=${VAR%%:*}  /  NAME=${VAR#*:}
        b = body
        if ":" in item:
            head, _, tail = item.partition(":")
        else:
            head, tail = item, item

        def _sub_head(m):
            local[m.group(1)] = head
            return f"{m.group(1)}={head}"

        def _sub_tail(m):
            local[m.group(1)] = tail
            return f"{m.group(1)}={tail}"

        b = re.sub(r"(\w+)=\$\{%s%%%%:\*\}" % re.escape(var), _sub_head, b)
        b = re.sub(r"(\w+)=\$\{%s#\*:\}" % re.escape(var), _sub_tail, b)
        for body_chunk in expand_loops(b, local):
            for suffix_chunk in expand_loops(suffix, vars_):
                chunks.append(prefix + body_chunk + suffix_chunk)
    return chunks


def classify_line_tokens(line, produced, rows, cwd):
    tokens = [t.strip('"\';') for t in line.split()]
    for tok in tokens:
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.+)$", tok)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        all_keys = (SPLIT_NAME_KEYS | ALWAYS_WRITE_KEYS | ALWAYS_READ_KEYS
                    | MAYBE_PRODUCED_READ_KEYS | AMBIGUOUS_KEYS)
        if key not in all_keys:
            continue   # not one of the flags this contract tracks (e.g. a bash scalar like BOSG=)
        if "$" in val:   # one of OUR keys, but left an unresolved variable — report, don't crash
            rows.append(("unresolved", "env", f"{key}={val}", "could not fully resolve — read manually"))
            continue
        if key in SPLIT_NAME_KEYS:
            for suffix in (f".cache/phase1_alg_states_{val}.npz",
                           f".cache/phase1_alg_states_{val}_states.npy"):
                _check(suffix, produced, rows, "cached states")
            continue
        if key in ALWAYS_WRITE_KEYS:
            path = val if val.startswith(".cache/") or val.startswith("/") else os.path.join(cwd, val)
            produced.add(os.path.normpath(path))
            rows.append(("produced-by-chain", "output", val, f"written by {key}"))
            continue
        if key in ALWAYS_READ_KEYS:
            _check(val, produced, rows, "fixture/codebook" if key != "ALG_TRAIN" and key != "ALG_TEST" else "fixture")
            continue
        if key in MAYBE_PRODUCED_READ_KEYS:
            _check(val, produced, rows, "checkpoint/dump")
            continue
        if key in AMBIGUOUS_KEYS:
            if "unet/train.py" in line:
                produced.add(os.path.normpath(val))
                rows.append(("produced-by-chain", "output", val, f"written by {key} (train.py)"))
            else:
                _check(val, produced, rows, "checkpoint (unet)")
            continue
    # positional-arg scripts — a line may call the same script more than
    # once (e.g. three paired_read.py calls inside one echo-studded line)
    for script, (nargs, deriver) in POSITIONAL_SPECS.items():
        for m in re.finditer(re.escape(f"scripts/{script}"), line):
            rest = line[m.end():]
            toks = []
            for t in rest.split():
                t = t.strip('"\')')
                if not TOKEN_RE.match(t):
                    break
                toks.append(t)
            args = toks[:max(nargs, 1)]
            for a in args:
                if deriver:
                    for p in deriver(a):
                        _check(p, produced, rows, f"{script} dependency")
                elif a.startswith(".cache/"):
                    _check(a, produced, rows, f"{script} arg")
    # redirection targets -> logs this chain writes
    for m in re.finditer(r">\s*(\.cache/\S+\.log)\b", line):
        produced.add(os.path.normpath(m.group(1)))
    # flock lock files
    for m in re.finditer(r"flock\s+-w\s+\d+\s+(\S+)", line):
        _check(m.group(1), produced, rows, "gpu lock", warn_only=True)
    # conditional-produce idiom: [ -f PATH ] || ...
    for m in re.finditer(r"\[\s*-f\s+(\.cache/\S+)\s*\]\s*\|\|", line):
        p = os.path.normpath(m.group(1))
        if os.path.exists(os.path.join(ROOT, p)):
            rows.append(("exists", "conditional", m.group(1), "already present — the || producer will not fire"))
        else:
            rows.append(("produced-by-chain", "conditional", m.group(1), "absent now — the chain's own `|| ...` branch produces it"))
        produced.add(p)   # either way, it's guaranteed to exist after this line


def _check(val, produced, rows, kind, warn_only=False):
    p = os.path.normpath(val)
    if p in produced:
        rows.append(("produced-by-chain", kind, val, "written earlier in this chain"))
        return
    full = val if os.path.isabs(val) else os.path.join(ROOT, val)
    if os.path.exists(full):
        rows.append(("exists", kind, val, ""))
    else:
        rows.append(("missing" if not warn_only else "missing(warn)", kind, val, ""))


def check_gates(raw_text, rows):
    """Re-run the `until grep -aq "PAT" FILE; do sleep ...; done` waits and the
    standalone `grep -a[q] "PAT" FILE ... || ABORT` guards against the FILE's
    current content — catches a stale/incomplete gate before the chain would
    hang or abort for real."""
    for m in re.finditer(r'until\s+grep\s+-a[q]?\s+"([^"]*)"\s+(\.cache/\S+)\s*;\s*do\s+sleep', raw_text):
        pat, path = m.group(1), m.group(2)
        _gate_check(pat, path, rows, blocking=True)
    for m in re.finditer(r'grep\s+-a[q]?\s+"([^"]*)"\s+(\.cache/\S+)(?:\s*\|\s*tail[^|]*\|\s*grep\s+-a[q]?\s+"([^"]*)")?[^\n]*\|\|\s*\{', raw_text):
        pat, path, pat2 = m.group(1), m.group(2), m.group(3)
        _gate_check(pat2 or pat, path, rows, blocking=False)


def _gate_check(pat, path, rows, blocking):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        rows.append(("missing", "gate-log", path, f"chain {'blocks forever' if blocking else 'aborts'} without it"))
        return
    try:
        hit = subprocess.run(["grep", "-aq", pat, full]).returncode == 0
    except Exception:
        hit = False
    if hit:
        rows.append(("exists", "gate-log", path, f'pattern "{pat[:60]}" found'))
    else:
        rows.append(("missing", "gate-log", path, f'pattern "{pat[:60]}" NOT found — {"would hang" if blocking else "would ABORT"}'))


def check_cwd_and_units(raw_text, rows):
    m = re.search(r"\bcd\s+(\S+)\s*;", raw_text)
    if m:
        d = m.group(1)
        if os.path.isdir(d):
            rows.append(("exists", "workdir", d, ""))
        else:
            rows.append(("missing", "workdir", d, "the chain's `cd` target does not exist"))
    try:
        out = subprocess.run(["systemctl", "--user", "list-units", "pc-*", "--all", "--no-legend"],
                              capture_output=True, text=True, timeout=10).stdout
        failed = [l for l in out.splitlines() if "failed" in l]
        if failed:
            for l in failed:
                rows.append(("warn", "systemd", l.strip(), "failed pc-* unit"))
        else:
            rows.append(("exists", "systemd", "pc-* units", "none failed"))
    except Exception as e:
        rows.append(("warn", "systemd", "pc-*", f"could not query: {e}"))
    lockp = os.path.join(ROOT, ".cache/gpu.lock")
    rows.append(("exists" if os.path.exists(lockp) else "missing", "gpu-lock", ".cache/gpu.lock", ""))


def main():
    if len(sys.argv) < 2:
        print("usage: preflight.py <chain.sh>", file=sys.stderr)
        return 2
    chain_path = sys.argv[1]
    if not os.path.isabs(chain_path):
        chain_path = os.path.join(ROOT, chain_path)
    raw = open(chain_path).read()
    text = join_continuations(strip_comments(raw))
    top_vars, text = extract_top_vars(text)
    chunks = expand_loops(text, top_vars)

    rows = []
    produced = set()
    for chunk in chunks:
        local_produced = set(produced)   # each chunk starts from what prior real lines have produced
        for line in chunk.splitlines():
            if not line.strip():
                continue
            classify_line_tokens(line, local_produced, rows, os.path.dirname(chain_path))
        produced |= local_produced

    check_gates(raw, rows)
    check_cwd_and_units(raw, rows)

    # dedupe, preserving first-seen status per path (status order favors
    # the most informative: produced-by-chain/exists over a later restate)
    seen = {}
    order = []
    for status, kind, path, note in rows:
        key = (kind, path)
        if key not in seen:
            seen[key] = (status, kind, path, note)
            order.append(key)
        else:
            old = seen[key]
            if old[0] == "exists" and status == "produced-by-chain":
                seen[key] = (status, kind, path, note)

    print(f"[preflight] {os.path.relpath(chain_path, ROOT)} — {len(order)} distinct inputs/outputs tracked, "
          f"{len(chunks)} loop-expansion(s)")
    print(f"{'STATUS':<22} {'KIND':<18} {'PATH':<55} NOTE")
    bad = 0
    for key in order:
        status, kind, path, note = seen[key]
        if status in ("missing",):
            bad += 1
        print(f"{status:<22} {kind:<18} {path:<55} {note}")
    print()
    if bad:
        print(f"[preflight] FAIL — {bad} required input(s) missing")
        return 1
    print("[preflight] PASS — every required input accounted for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
