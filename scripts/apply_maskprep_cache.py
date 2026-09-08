"""apply_maskprep_cache.py — THE MASK-PREP CACHE, a staged patch
(2026-09-08). Perf item, zero science: do_train's mask-prep pass
recomputes, for EVERY arm, the same (n, L_FAC, L_FAC) mask tensor (and,
under the alternator/mask-head/xprior doors, FACTS / MASSB / NL0) from
the WARM head's own breath-0 parses. It costs ~75 minutes at form-scale
and every arm this week paid it for arrays that were bit-identical.
This patch banks them, keyed by everything they depend on, behind
ALG_MASKPREP_CACHE=1. UNSET = the old pass, verbatim (the two call-site
anchors are the only lines that move, and both fold to the old code
when the door is dark: _maskprep_lookup returns (None, None, None) and
the loop's iterator is `range(0, n, 8)` — the literal old expression).

WHAT THE PASS FILLS (enumerated from the head, all of them cached):
  MASKS (n, L_FAC, L_FAC) f32   build_slot_masks on breath-0 fat/args/res
                                — always, whenever K_B > 1
  FACTS (n, K_VARS, 4)  f32     alt2_fact_buf pass-1 commit; door ALG_ALT2
  MASSB (n, K_VARS)     f32     domain mass, /301; doors ALG_ALT2+ALG_MH_MASS
  NL0   (n, H_W)        f32     breath-0 NL state; doors ALG_MH_ATLAS+
                                ALG_MH_XPRIOR (feeds cross_prior AFTER the
                                loop — that retrieval still runs, on the
                                restored NL0, so ATLAS_IDX is rebuilt from
                                cached input rather than cached itself)
An array whose door is dark is not written to the npz, and a cache
whose array set disagrees with the live doors is a HARD ERROR.

WHAT THEY DEPEND ON (the key; sha256 over a sorted JSON record):
  1. THE CODE. sha of the head source itself (mix_sha16 — the ledger's
     one file-hash organ, mycelium/era.py, never a second implementation)
     plus the sha of every mycelium/*.py module the head imports (found
     by regex over the source, so the list maintains itself).
  2. THE PARAMETERS, twice over: (a) the DIRECT identity — a sha over
     every realized tensor in p, in sorted key order, AFTER RESUME /
     WARM_FROM / PAD-WARM / the door-#12 seeds have run, which is
     literally the thing the pass consumes (~1s of host transfer); and
     (b) the declared provenance beside it — the warm source's path and
     file sha, plus SEED. (a) alone would be enough; (b) is in the key
     because the task pinned it and because a provenance change with
     identical weights should not silently reuse a bucket.
  3. THE TRAIN ARRAYS, identified the way load_alg identifies them and
     not by a private re-derivation: ALG_TRAIN_NAME, ALG_TRAIN and its
     mix_sha16 (THE SHA-FENCE's own key), the staged npz's mix_sha
     STAMP (the declared key, read as one lazy npz member) and its file
     fingerprint, and the states memmap's file fingerprint. NEITHER BIG
     FILE IS HASHED BY BYTES: the npz is 14 GB and the memmap 140 GB at
     form-scale — a full read would cost more than the pass this cache
     skips, and would OOM a box already holding a training run. They are
     keyed by (size, mtime_ns) plus the declared stamp, and their
     CONTENT is covered by the verification below, which re-runs the
     forward on real rows of the memmap. Which mode each file got is
     recorded in the banked fingerprint (_MP_HASH_CAP = 256 MB).
  4. THE ENVIRONMENT: every ALG_* variable present, plus every env name
     the head source itself reads (mechanically extracted from the
     source, so SC_EVAL, SC_KB, BIND_CODES, NB_PERSLOT, MH_ATLAS,
     MH_ATLAS_MANIFEST, WARM_FROM, RESUME, SEED, BREATH_* and any door
     added tomorrow are all in the key without anyone remembering to
     list them), plus DEV/BEAM/JIT/NOOPT (the kernel-selection dials —
     a different device is a different arithmetic). Every value that
     names an existing file is hashed too (BIND_CODES, ALG_POLAR_BANDS,
     ALG_POLAR_D_INIT, MH_ATLAS, ...): a rebuilt side file must not
     ride an old bucket.
  Excluded by construction: the cache's own three doors, and the
  TRAINER-ONLY dials (STEPS, LR, BATCH, VAL_EVERY, SNAP_EVERY,
  PRECOMPUTE_ONLY) — so two arms that share a diet and differ only in
  length share a bucket. That exclusion is not a claim in prose: on every
  key build _maskprep_trainer_only re-proves from the source that none of
  those names is read at module scope, inside any function the pass
  calls, or in do_train ahead of the pass. The day one of them moves into
  that region, the key build RAISES.

THE VERIFICATION (why this is not a trust-me cache). On a HIT the pass
does not simply skip: it RE-RUNS the first two batches exactly as it
would have — same forward, same build_slot_masks/alt2_fact_buf calls,
same slices — and np.array_equal's the result against the cached rows,
for every array. A mismatch RAISES, naming the array and the row; there
is no silent fallback to recompute (a cache that is wrong about rows
0-15 is wrong about the key, and the key is the thing to fix). Cost:
2 batches out of n/8.

ALG_MASKPREP_IGNORE=NAME[,NAME...] (declared, printed, recorded in the
npz) drops named env vars from the key — the lawful way to let two arms
that differ only in a TRAINING door (ALG_PC_MIX, say, whose _PCV is not
armed until after this pass) share one bucket. It is safe exactly
because the verification stands behind it: an ignored door that DOES
move breath-0 output fails loudly on the first two batches. When the
ignore list is non-empty the verification adds the LAST batch as well
(a third, tail sample), and says so in the print.

DOORS
  ALG_MASKPREP_CACHE=1     arm the cache (unset = byte-identical)
  ALG_MASKPREP_DIR=path    where buckets live (default .cache)
  ALG_MASKPREP_IGNORE=...  declared key exclusions (see above)

--check: asserts every anchor present and unique, builds the would-be
result, ast-parses it, runs the structural asserts and the symtable
free-variable audit, and writes NOTHING. MP_TARGET may point at a copy
(rehearsal); default scripts/phase1_algebra_head.py. The CPU proofs
live in scripts/maskprep_cache_smoke.py.
"""
import ast
import builtins
import os
import symtable
import sys

fn = os.environ.get("MP_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert 'ALG_MASKPREP_CACHE' not in s and '_maskprep_' not in s, \
    "the mask-prep cache is already present — patch was applied; refuse (idempotence)"
assert 'print("[breath] mask-prep pass ...", flush=True)' in s, \
    "the mask-prep pass is missing — wrong vintage of the head"
assert 'from mycelium.era import mix_sha16' in s, \
    "the sha-fence organ is not imported anywhere in this head — wrong vintage"
assert 'STATES_NPZ = ".cache/phase1_alg_states_{split}.npz"' in s and \
    'STATES_NPY = ".cache/phase1_alg_states_{split}_states.npy"' in s, \
    "the staged-array paths moved — the train-identity block must be re-read"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# 1. the organ: key, lookup, verify, bank. Module level, immediately
#    above do_train (its only caller).
patch(1, "module: the mask-prep cache organ (key/lookup/verify/bank)",
      '''def do_train(steps, lr, batch, seed):''',
      '''# ===========================================================================
# THE MASK-PREP CACHE (apply_maskprep_cache.py, 2026-09-08) — a perf
# organ, no science. Everything here is dark unless ALG_MASKPREP_CACHE
# is set; see the apply script's docstring for the full contract.
# ===========================================================================
_MP_CTRL = ("ALG_MASKPREP_CACHE", "ALG_MASKPREP_DIR", "ALG_MASKPREP_IGNORE")
_MP_EXTRA_ENV = ("DEV", "BEAM", "JIT", "NOOPT")
_MP_ARRAYS = ("MASKS", "FACTS", "MASSB", "NL0")
# Files at or under this size are keyed by their BYTES; bigger ones by
# (size, mtime_ns) plus whatever DECLARED stamp they carry. The staged
# train npz is 14 GB at form-scale and the states memmap 140 GB — hashing
# them would cost more than the pass this cache exists to skip, and on a
# box whose RAM is already holding a training run it would be an OOM, not
# a slowdown. Which mode was used is RECORDED in the fingerprint, so a
# bucket can always be autopsied for what it actually checked.
_MP_HASH_CAP = 256 << 20
# the dials read ONLY after the pass has run — the difference between two
# arms that share a diet and differ in LENGTH. Excluded from the key so
# those arms share a bucket; the exclusion is CHECKED against the source
# on every key build (_maskprep_trainer_only), never asserted in prose.
_MP_TRAINER_ONLY = ("STEPS", "LR", "BATCH", "VAL_EVERY", "SNAP_EVERY",
                    "PRECOMPUTE_ONLY")
# everything the mask-prep pass actually calls (the reachability set the
# exclusion check is run against)
_MP_PASS_FUNCS = ("forward", "build_params", "build_slot_masks",
                  "alt2_fact_buf", "_alt2_fact_buf_v0", "_alt2_fact_buf_v1",
                  "load_alg", "tokenize", "build_gold", "sent_indices",
                  "cross_prior")


def _maskprep_env_names(src):
    """Every env var name the head source itself reads. Mechanical, so a
    door added tomorrow enters the key without anyone remembering to add
    it here (the hand-maintained list is the thing that goes stale)."""
    import re
    return set(re.findall(
        r'os\\.environ(?:\\.get)?[\\(\\[]\\s*"([A-Za-z0-9_]+)"', src))


def _maskprep_file_fp(path):
    """A file's fingerprint: its bytes when that is affordable, else its
    (size, mtime_ns). Never silent about which — the record says so."""
    from mycelium.era import mix_sha16
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    if st.st_size <= _MP_HASH_CAP:
        return {"sha": mix_sha16(path), "size": int(st.st_size)}
    return {"sha": None, "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "why": "over the hash cap"}


def _maskprep_trainer_only(src):
    """Prove, from the source, that every name in _MP_TRAINER_ONLY is read
    only AFTER the mask-prep pass: never at module scope, never inside a
    function the pass calls, and never in do_train ahead of the pass. If
    one of them ever moves into that region this raises and the name goes
    back into the key — the exclusion cannot rot silently."""
    import ast
    tree = ast.parse(src)
    seg = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in _MP_PASS_FUNCS:
                seg.append(ast.get_source_segment(src, node) or "")
        elif not isinstance(node, ast.ClassDef):
            seg.append(ast.get_source_segment(src, node) or "")   # module scope
    _dt = src.index("def do_train(")
    seg.append(src[_dt:src.index('print(f"[breath] masks ready', _dt)])
    read = _maskprep_env_names(chr(10).join(seg))   # the READ syntax, not
                                                   # a prose mention
    for nm in _MP_TRAINER_ONLY:
        assert nm not in read, (
            f"[maskprep] {nm} is read by code the mask-prep pass runs — it "
            f"can no longer be excluded from the cache key. Remove it from "
            f"_MP_TRAINER_ONLY (the arms that differ in it will re-key).")
    return set(_MP_TRAINER_ONLY)


def _maskprep_ignored():
    return sorted({x.strip() for x in
                   os.environ.get("ALG_MASKPREP_IGNORE", "").split(",")
                   if x.strip()})


def _maskprep_fingerprints(p, n, seed):
    """The full dependency record of the mask-prep pass, JSON-able. The
    cache key is a sha over this; the record itself is banked beside the
    arrays so a stale bucket can be autopsied instead of guessed at."""
    import hashlib
    import mycelium
    from mycelium.era import mix_sha16   # the ledger's ONE file-hash
                                         # organ (meter-divergence law)
    src_path = os.path.abspath(__file__)
    src = open(src_path).read()
    fp = {"v": 1, "head": mix_sha16(src_path)}
    # the mycelium modules this head imports — a code change one level
    # out must invalidate too
    import re as _re
    _mroot = os.path.dirname(os.path.abspath(mycelium.__file__))
    fp["mods"] = {}
    for _m in sorted(set(_re.findall(
            r'mycelium\\.([a-z0-9_]+)', src))):
        _mp = os.path.join(_mroot, _m + ".py")
        if os.path.exists(_mp):
            fp["mods"][_m] = mix_sha16(_mp)
    # the parameters the pass actually runs (post RESUME/WARM/PAD-WARM)
    _hp = hashlib.sha256()
    for k in sorted(p):
        _a = np.ascontiguousarray(p[k].detach().numpy())
        _hp.update(k.encode())
        _hp.update(f"{_a.shape}{_a.dtype}".encode())
        _hp.update(_a.tobytes())
    fp["params"] = _hp.hexdigest()[:16]
    fp["n_params"] = len(p)
    fp["seed"] = int(seed)
    _wf = (ALG_CKPT if (int(os.environ.get("RESUME", "0"))
                        and os.path.exists(ALG_CKPT))
           else os.environ.get("WARM_FROM", ""))
    fp["warm"] = {"path": _wf, "file": _maskprep_file_fp(_wf)}
    # the train arrays, identified as load_alg identifies them
    _npz = STATES_NPZ.format(split=TRAIN_NAME)
    _npy = STATES_NPY.format(split=TRAIN_NAME)
    _z = np.load(_npz)     # lazy: reading one member, not the archive
    fp["train"] = {
        "name": TRAIN_NAME, "mix": ALG_TRAIN,
        "mix_sha16": mix_sha16(ALG_TRAIN),          # THE SHA-FENCE's key
        "npz": _npz, "npz_file": _maskprep_file_fp(_npz),
        "npz_stamp": (str(_z["mix_sha"]) if "mix_sha" in _z.files
                      else None),                   # the fence's stamp
        "npy": _npy, "npy_file": _maskprep_file_fp(_npy),
        "n": int(n)}
    # the environment
    _ign = set(_maskprep_ignored())
    _names = ((_maskprep_env_names(src)
               | {k for k in os.environ if k.startswith("ALG_")}
               | set(_MP_EXTRA_ENV))
              - set(_MP_CTRL) - _ign - _maskprep_trainer_only(src))
    fp["env"] = {k: os.environ.get(k) for k in sorted(_names)}
    fp["env_files"] = {k: _maskprep_file_fp(v)
                       for k, v in sorted(fp["env"].items())
                       if v and os.path.isfile(v)}
    fp["ignored"] = sorted(_ign)
    fp["trainer_only"] = sorted(_MP_TRAINER_ONLY)
    fp["pass"] = {"batch": 8}
    return fp


def _maskprep_path(key):
    d = os.environ.get("ALG_MASKPREP_DIR", ".cache")
    return os.path.join(d, f"maskprep_{key}.npz")


def _maskprep_ver_starts(n):
    """Batch starts the pass RE-RUNS on a hit: the first two, plus the
    tail batch when a declared ignore list is live (the ignore door's
    second wall)."""
    st = [s0 for s0 in (0, 8) if s0 < n]
    if _maskprep_ignored():
        _last = ((max(n, 1) - 1) // 8) * 8
        if _last not in st:
            st.append(_last)
    return st


def _maskprep_lookup(p, n, seed):
    """(key, fingerprints, cached-or-None). Door unset -> all None, and
    the call site's iterator stays range(0, n, 8)."""
    if not int(os.environ.get("ALG_MASKPREP_CACHE", "0")):
        return None, None, None
    import hashlib
    import json
    import time
    _t0 = time.time()
    fp = _maskprep_fingerprints(p, n, seed)
    key = hashlib.sha256(
        json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    path = _maskprep_path(key)
    _ig = _maskprep_ignored()
    if _ig:
        print(f"[maskprep] key EXCLUSIONS declared: {','.join(_ig)} "
              f"(verification adds the tail batch)", flush=True)
    if not os.path.exists(path):
        print(f"[maskprep] cache MISS key={key} ({time.time() - _t0:.1f}s "
              f"to key) -> running the pass, will bank {path}", flush=True)
        return key, fp, None
    z = np.load(path, allow_pickle=False)
    cached = {k: z[k] for k in z.files if k in _MP_ARRAYS}
    print(f"[maskprep] cache candidate {path} "
          f"({', '.join(f'{k}{cached[k].shape}' for k in sorted(cached))}) "
          f"— verifying", flush=True)
    return key, fp, cached


def _maskprep_finish(key, fp, cached, n, starts, arrays):
    """HIT: assert the recomputed batches equal the cached rows and hand
    back the cached arrays. MISS: bank what the pass just built. Either
    way the caller's arrays are replaced by the returned ones."""
    import json
    if cached is None:
        out = {k: v for k, v in arrays.items() if v is not None}
        path = _maskprep_path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + f".tmp{os.getpid()}.npz"   # np.savez appends .npz
                                                # to any other suffix
        np.savez_compressed(
            tmp, _key=np.array(key),
            _fingerprints=np.array(json.dumps(fp, sort_keys=True, indent=1)),
            **out)
        os.replace(tmp, path)   # atomic: a killed run never leaves half
        print(f"[maskprep] cached -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB, arrays "
              f"{'+'.join(sorted(out))})", flush=True)
        return arrays
    rows = np.concatenate([np.arange(s0, min(s0 + 8, n)) for s0 in starts])
    live = {k for k, v in arrays.items() if v is not None}
    assert live == set(cached), (
        f"[maskprep] STALE CACHE {_maskprep_path(key)}: it holds "
        f"{sorted(cached)} but this run's doors fill {sorted(live)} — "
        f"the key is incomplete; delete the bucket and re-key")
    for nm in sorted(live):
        cur, c = arrays[nm], cached[nm]
        assert c.shape == cur.shape and c.dtype == cur.dtype, (
            f"[maskprep] STALE CACHE {_maskprep_path(key)}: {nm} is "
            f"{c.shape}/{c.dtype}, this run wants {cur.shape}/{cur.dtype}")
        if not np.array_equal(c[rows], cur[rows]):
            _bad = rows[[not np.array_equal(c[r], cur[r]) for r in rows]]
            raise RuntimeError(
                f"[maskprep] CACHE VERIFICATION FAILED on {nm}: "
                f"{len(_bad)}/{len(rows)} recomputed rows differ from the "
                f"cached ones (first: row {int(_bad[0])}) — bucket "
                f"{_maskprep_path(key)} is stale or the key is missing a "
                f"dependency. NOT falling back silently: fix the key (or "
                f"delete the bucket) and re-run.")
    _tail = "" if len(starts) <= 2 else " + tail (ignore-list live)"
    print(f"[maskprep] CACHE HIT key={key} verified on "
          f"{min(len(starts), 2)} batches{_tail}", flush=True)
    return {k: (cached[k] if k in cached else None) for k in arrays}


def do_train(steps, lr, batch, seed):''')

# 2. do_train: the loop's iterator. On a HIT it is the verification
#    starts; door unset it is the literal old expression.
patch(2, "do_train: mask-prep loop rides the cache lookup",
      '''        print("[breath] mask-prep pass ...", flush=True)
        MASKS = np.zeros((n, L_FAC, L_FAC), np.float32)
        for s0 in range(0, n, 8):''',
      '''        print("[breath] mask-prep pass ...", flush=True)
        MASKS = np.zeros((n, L_FAC, L_FAC), np.float32)
        # THE MASK-PREP CACHE (apply_maskprep_cache.py, 2026-09-08).
        # Door unset: _maskprep_lookup returns (None, None, None), the
        # iterator is the literal old range, and _maskprep_finish is
        # never called — this pass is the old pass. HIT: only the
        # verification batches run, and the recomputed rows are
        # asserted against the cache below before it is trusted.
        _mp_key, _mp_fp, _mp_cached = _maskprep_lookup(p, n, seed)
        _mp_starts = (range(0, n, 8) if _mp_cached is None
                      else _maskprep_ver_starts(n))
        for s0 in _mp_starts:''')

# 3. do_train: verify-and-restore (hit) / bank (miss), before the
#    summary print so the printed mean degree is the DEPLOYED array's.
patch(3, "do_train: verify/restore or bank, ahead of the ready print",
      '''        print(f"[breath] masks ready (mean degree "
              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)''',
      '''        if _mp_key is not None:
            _mp_out = _maskprep_finish(
                _mp_key, _mp_fp, _mp_cached, n, _mp_starts,
                {"MASKS": MASKS, "FACTS": FACTS, "MASSB": MASSB,
                 "NL0": (NL0 if ATLAS_TAB is not None else None)})
            MASKS, FACTS, MASSB = (_mp_out["MASKS"], _mp_out["FACTS"],
                                   _mp_out["MASSB"])
            if ATLAS_TAB is not None:
                NL0 = _mp_out["NL0"]
        print(f"[breath] masks ready (mean degree "
              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ------------------------------------------------- structural asserts
assert s.count('def do_train(steps, lr, batch, seed):') == 1
assert s.count('_mp_key, _mp_fp, _mp_cached = _maskprep_lookup') == 1
assert s.count('for s0 in _mp_starts:') == 1
assert s.count('_maskprep_finish(') == 2, "one def, one call"
_DT = chr(10) + 'def do_train(steps, lr, batch, seed):'
assert s.index('def _maskprep_lookup') < s.index(_DT), \
    "the organ must be defined before its caller's module-level def"
assert s.index('_mp_key, _mp_fp, _mp_cached') < s.index('for s0 in _mp_starts:'), \
    "lookup must precede the loop"
assert s.index('for s0 in _mp_starts:') < s.index('_mp_out = _maskprep_finish('), \
    "the loop must precede the verify/bank"
assert s.index('_mp_out = _maskprep_finish(') \
    < s.rindex('print(f"[breath] masks ready (mean degree '), \
    "verify/restore must precede the summary print (it reports the "\
    "array the trainer will actually feed)"
# the door-unset fold: the old iterator expression must survive verbatim
assert 'range(0, n, 8) if _mp_cached is None' in s, \
    "the dark path must be the literal old range expression"
# every array the pass fills is enumerated in the organ AND handed to it
_decl = s.split('_MP_ARRAYS = ')[1].split('\n')[0]
for _nm in ("MASKS", "FACTS", "MASSB", "NL0"):
    assert f'"{_nm}"' in _decl, f"{_nm} missing from _MP_ARRAYS"
assert '"MASKS": MASKS, "FACTS": FACTS, "MASSB": MASSB' in s
assert '"NL0": (NL0 if ATLAS_TAB is not None else None)' in s, \
    "NL0 is only BOUND under ALG_MH_ATLAS — it must never be named "\
    "unguarded (NameError otherwise)"
# no silent fallback anywhere in the organ
_organ = s.split('# THE MASK-PREP CACHE (apply_maskprep_cache.py')[1] \
    .split(_DT)[0]
assert 'except' not in _organ and 'try:' not in _organ, \
    "no swallowed exceptions in the cache organ (no-silent-fallbacks law)"
assert 'os.replace(' in _organ, "the bank must be atomic"
assert '_maskprep_trainer_only(src)' in _organ, \
    "the trainer-only exclusion must be re-proved at key time, not trusted"

# --------------------------- the symtable free-variable audit (idiom)
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC', '_PCV'}
BUILTIN = set(dir(builtins))


def audit(tbl, fname):
    bad = set()
    for sym in tbl.get_symbols():
        n_ = sym.get_name()
        if sym.is_global() and n_ not in module_names \
                and n_ not in DYNAMIC_OK and n_ not in BUILTIN:
            bad.add(n_)
    for ch in tbl.get_children():
        bad |= audit(ch, fname)
    assert not bad, f"{fname}: unresolved free variables {sorted(bad)}"
    return set()


for child in mod_tbl.get_children():
    if child.get_name() in ('do_train', 'forward', 'build_params',
                            '_maskprep_fingerprints', '_maskprep_lookup',
                            '_maskprep_finish', '_maskprep_ver_starts',
                            '_maskprep_env_names', '_maskprep_path',
                            '_maskprep_ignored', '_maskprep_trainer_only'):
        audit(child, child.get_name())

print(f"[maskprep cache] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[maskprep cache] symtable free-var audit PASS (do_train, forward, "
      "build_params + the six organ functions)")
print("[maskprep cache] NEW params: 0. ALG_MASKPREP_CACHE unset = the old "
      "pass verbatim (the iterator folds to `range(0, n, 8)`)")
print("[maskprep cache] contract: HIT re-runs the first two batches and "
      "np.array_equal's them against the cached rows — a stale bucket "
      "RAISES; there is no silent recompute-fallback")
if CHECK:
    print("[maskprep cache] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[maskprep cache] APPLIED ({fn}); ast OK — run "
          "scripts/maskprep_cache_smoke.py (CPU) before trusting")
