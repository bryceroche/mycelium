"""maskprep_cache_smoke.py — THE MASK-PREP CACHE PROOFS (CPU, zero GPU,
zero training, zero gradient descent; 2026-09-08).

Proves scripts/apply_maskprep_cache.py's contract against the STAGED
source — the patch is built in memory (the apply script forced to
--check, the polar_sink_smoke.py idiom) and written to a scratch copy of
the head that the subprocesses import; the real head is NEVER written.

FIXTURE (built here, never in .cache): 32 rows sliced out of
.cache/form_mix16.jsonl + its staged arrays into a temp dir whose
.cache/ holds the slice plus symlinks to the val fixture and side files.
Every run is `do_train(steps=0)` — the mask-prep pass is the whole
subject; the step loop never runs. DEV=CPU, hard, everywhere.
The env stack is the champion's MINUS the bindbus doors: form16's npz
predates g_bindvec/g_bind_ids and the head's feed-door fence rightly
refuses to run that terminal without its gold. ALG_ALT2 / ALG_MH_MASS /
ALG_MH_ATLAS+ALG_MH_XPRIOR are all ON, so all four cached arrays
(MASKS, FACTS, MASSB, NL0) are live in every gate.

  GATE 0 — THE APPLY SCRIPT: --check on a pristine copy (anchors present
     and unique, ast OK, structural asserts + symtable audit), a second
     application refused by the idempotence guard, and NOTHING written.
  GATE 1 — DOOR UNSET (the byte-identity claim, two ways):
     1a  SOURCE: the diff of do_train pristine->staged is exactly the
         guarded cache lines; the one REPLACED line keeps the literal
         old expression `range(0, n, 8)` as its dark branch.
     1b  RUNTIME: pristine head vs staged head, same fixture, same env,
         ALG_MASKPREP_CACHE unset -> np.array_equal on MASKS, FACTS,
         MASSB and NL0, and the staged run prints no [maskprep] line at
         all (the organ is never entered).
  GATE 2 — DOOR SET: run 1 MISSES and banks the bucket; run 2 HITS,
     passes the built-in 2-batch verification, and the arrays it hands
     do_train are equal to the pristine run's on every element.
  GATE 3 — TAMPER: one flipped mask entry inside the verified rows makes
     run 3 DIE loudly (RuntimeError naming the array and the row, no
     silent recompute). Reported beside it, measured not assumed: a flip
     OUTSIDE the verified rows is not caught — the fence's declared
     scope is the first two batches (plus the tail batch when a key
     exclusion is declared).
  GATE 4 — THE KEY: key-only runs (no forward at all) that vary one
     input at a time and diff the banked fingerprint records — env door,
     side file bytes, head source bytes, seed, a perturbed parameter,
     the declared-exclusion round trip, and the negative control
     (STEPS/LR are NOT in the key, so two arms that differ only in
     training length share a bucket).

Run from the repo root:
  .venv/bin/python3 scripts/maskprep_cache_smoke.py
Post-apply, point MP_TARGET at a pristine pre-apply copy:
  git show <rev>:scripts/phase1_algebra_head.py > /tmp/head_pre.py
  MP_TARGET=/tmp/head_pre.py .venv/bin/python3 scripts/maskprep_cache_smoke.py
Keep the scratch dir for forensics with MP_KEEP=1.
"""
import difflib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402

HEAD = os.environ.get("MP_TARGET",
                      os.path.join(_ROOT, "scripts", "phase1_algebra_head.py"))
PY = os.path.join(_ROOT, ".venv", "bin", "python3")
N_ROWS = 32
SRC_MIX = os.path.join(_ROOT, ".cache", "form_mix16.jsonl")
SRC_NPZ = os.path.join(_ROOT, ".cache", "phase1_alg_states_form16.npz")
SRC_NPY = os.path.join(_ROOT, ".cache", "phase1_alg_states_form16_states.npy")
CKPT = os.path.join(_ROOT, ".cache", "sharp_fedon242.safetensors")
ARRAYS = ("masks", "facts", "massb", "nl0")
FINALS = {"masks": "final_MASKS", "facts": "final_FACTS",
          "massb": "final_MASSB", "nl0": "final_NL0"}

DRIVER = r'''
"""mp_driver — import a head module BY PATH, record every array the
mask-prep pass fills, run do_train(steps=0). MP_KEYONLY=1 stops after
the fingerprint/key (no forward at all)."""
import hashlib, importlib.util, json, os, sys
import numpy as np
sys.path.insert(0, os.environ["MP_ROOT"])
sys.path.insert(0, os.path.join(os.environ["MP_ROOT"], "scripts"))
spec = importlib.util.spec_from_file_location("mp_head", os.environ["MP_HEAD"])
mod = importlib.util.module_from_spec(spec)
sys.modules["mp_head"] = mod
spec.loader.exec_module(mod)
SEED = int(os.environ.get("MP_SEED", "0"))

if os.environ.get("MP_KEYONLY"):
    samples, states, tokmask, gold, sent = mod.load_alg("train")
    p = mod.build_params(SEED)
    if os.environ.get("MP_PERTURB"):
        from tinygrad import Tensor
        k = sorted(p)[0]
        a = p[k].detach().numpy().copy()
        a.flat[0] += np.float32(1e-3)
        p[k].assign(Tensor(a, dtype=p[k].dtype)).realize()
    fp = mod._maskprep_fingerprints(p, states.shape[0], SEED)
    key = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    open(os.environ["MP_OUT"], "w").write(json.dumps({"key": key, "fp": fp}))
    print("[driver] key=" + key, flush=True)
    raise SystemExit(0)

REC = {"masks": [], "facts": []}
_bsm = mod.build_slot_masks
def _rec_masks(o_np, sent_rows):
    m = _bsm(o_np, sent_rows)
    REC["masks"].append(np.array(m, copy=True))
    return m
mod.build_slot_masks = _rec_masks
_afb = mod.alt2_fact_buf
def _rec_facts(*a, **kw):
    f = _afb(*a, **kw)
    REC["facts"].append(np.array(f, copy=True))
    if kw.get("mass_out") is not None:
        REC.setdefault("massb", []).append(np.array(kw["mass_out"], copy=True))
    return f
mod.alt2_fact_buf = _rec_facts
import mycelium.step_atlas as _sa        # do_train's LOCAL import reads the
_cp = _sa.cross_prior                    # module attribute -> the wrapper
def _rec_cp(atl, nl0, *a, **kw):
    REC["nl0"] = np.array(nl0, copy=True)
    return _cp(atl, nl0, *a, **kw)
_sa.cross_prior = _rec_cp
if hasattr(mod, "_maskprep_finish"):
    _fin = mod._maskprep_finish
    def _rec_fin(key, fp, cached, n, starts, arrays):
        out = _fin(key, fp, cached, n, starts, arrays)
        REC["final"] = {k: (None if v is None else np.array(v, copy=True))
                        for k, v in out.items()}
        REC["key"] = key
        REC["fp"] = json.dumps(fp, sort_keys=True)
        return out
    mod._maskprep_finish = _rec_fin

mod.do_train(0, 1e-4, 8, SEED)

out = {}
for k in ("masks", "facts", "massb"):
    if REC.get(k):
        out[k] = np.concatenate(REC[k], 0)
if REC.get("nl0") is not None:
    out["nl0"] = REC["nl0"]
for k, v in REC.get("final", {}).items():
    if v is not None:
        out["final_" + k] = v
if "key" in REC:
    out["key"] = np.array(REC["key"])
    out["fp"] = np.array(REC["fp"])
np.savez(os.environ["MP_OUT"], **out)
print("[driver] wrote " + os.environ["MP_OUT"] + " " + str(sorted(out)), flush=True)
'''


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    import contextlib
    import io
    argv0 = sys.argv
    sys.argv = ["apply_maskprep_cache.py", "--check"]
    env0 = os.environ.get("MP_TARGET")
    os.environ["MP_TARGET"] = HEAD
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # GATE 0 prints it
            ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                             "apply_maskprep_cache.py"),
                                run_name="_maskprep_staging")
    finally:
        sys.argv = argv0
        if env0 is None:
            os.environ.pop("MP_TARGET", None)
        else:
            os.environ["MP_TARGET"] = env0
    assert ns["CHECK"], "staging must run under --check (nothing written)"
    return ns["s"]


def func_lines(src, name):
    import ast
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return src.split("\n")[node.lineno - 1:node.end_lineno]
    raise AssertionError(f"{name} not found")


# ===========================================================================
# FIXTURE
# ===========================================================================
def build_fixture(tmp):
    from mycelium.era import mix_sha16
    cache = os.path.join(tmp, "fix", ".cache")
    os.makedirs(cache, exist_ok=True)
    rows = open(SRC_MIX).readlines()[:N_ROWS]
    jp = os.path.join(cache, "mp_smoke_mix.jsonl")
    open(jp, "w").writelines(rows)
    z = np.load(SRC_NPZ)
    out = {}
    for k in z.files:
        if k == "mix_sha":
            continue
        a = z[k]
        out[k] = a[:N_ROWS] if (a.ndim and a.shape[0] == len(
            np.load(SRC_NPZ)["tokmask"])) else a
    out["mix_sha"] = np.array(mix_sha16(jp))     # the sha-fence, honestly
    np.savez(os.path.join(cache, "phase1_alg_states_mpsmoke.npz"), **out)
    src = np.load(SRC_NPY, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        os.path.join(cache, "phase1_alg_states_mpsmoke_states.npy"), mode="w+",
        dtype=src.dtype, shape=(N_ROWS,) + src.shape[1:])
    dst[:] = src[:N_ROWS]
    dst.flush()
    del dst
    for f in ("phase1_alg_states_test23.npz",
              "phase1_alg_states_test23_states.npy", "polar_bands.json",
              "bindbus_codes512.npz", "step_atlas_current.npz",
              "RESEARCH_MANIFEST.json", "algebra_nl_test.jsonl"):
        d = os.path.join(cache, f)
        if not os.path.exists(d):
            os.symlink(os.path.join(_ROOT, ".cache", f), d)
    return os.path.dirname(cache)


def env_for(tmp, fixdir, **over):
    e = dict(os.environ)
    e.update({
        "DEV": "CPU", "MP_ROOT": _ROOT,
        "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
        "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
        "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
        "BIND_CODES": os.path.join(_ROOT, ".cache", "bindbus_codes512.npz"),
        "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
        "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
        "ALG_MH_MASS": "1", "ALG_MH_ATLAS": "1", "ALG_MH_XPRIOR": "2",
        "MH_ATLAS": os.path.join(_ROOT, ".cache", "step_atlas_current.npz"),
        "MH_ATLAS_MANIFEST": os.path.join(_ROOT, ".cache",
                                          "RESEARCH_MANIFEST.json"),
        "ALG_FED": "1", "SC_EVAL": "0", "WARM_FROM": CKPT,
        "ALG_TRAIN": os.path.join(fixdir, ".cache", "mp_smoke_mix.jsonl"),
        "ALG_TRAIN_NAME": "mpsmoke",
        "ALG_TEST": os.path.join(_ROOT, ".cache", "algebra_nl_test.jsonl"),
        "ALG_TEST_NAME": "test23",
        "ALG_CKPT": os.path.join(tmp, "mp_ckpt.safetensors"),
        "VAL_EVERY": "999999", "STEPS": "0",
        "ALG_MASKPREP_DIR": os.path.join(tmp, "buckets"),
    })
    for k in ("ALG_MASKPREP_CACHE", "ALG_MASKPREP_IGNORE", "ALG_POLAR",
              "ALG_PC_MIX", "ALG_TRUNK_LORA", "RESUME"):
        e.pop(k, None)
    for k, v in over.items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = str(v)
    return e


def run_case(tmp, fixdir, head, out, tag, must_fail=False, **over):
    e = env_for(tmp, fixdir, **over)
    e["MP_HEAD"] = head
    e["MP_OUT"] = out
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(tmp, "mp_driver.py")], env=e,
                       cwd=fixdir, capture_output=True, text=True)
    dt = time.time() - t0
    log = r.stdout + r.stderr
    open(os.path.join(tmp, f"log_{tag}.txt"), "w").write(log)
    if must_fail:
        assert r.returncode != 0, \
            f"[{tag}] expected a LOUD failure, got returncode 0:\n{log[-2000:]}"
    else:
        assert r.returncode == 0, f"[{tag}] FAILED:\n{log[-4000:]}"
    return log, dt


def load(out):
    z = np.load(out, allow_pickle=False)
    return {k: z[k] for k in z.files}


def eq(a, b, nm, tag):
    assert a.shape == b.shape, f"[{tag}] {nm} shape {a.shape} != {b.shape}"
    assert np.array_equal(a, b), (
        f"[{tag}] {nm} DIFFERS: {int((a != b).sum())}/{a.size} entries")


def keyrun(tmp, fixdir, head, tag, **over):
    out = os.path.join(tmp, f"key_{tag}.json")
    run_case(tmp, fixdir, head, out, "key_" + tag, MP_KEYONLY="1", **over)
    d = json.load(open(out))
    return d["key"], d["fp"]


def fp_diff(a, b):
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


# ===========================================================================
def main():
    print("=" * 74)
    print("MASK-PREP CACHE SMOKE — CPU, zero GPU, zero training")
    print("=" * 74)
    for f in (SRC_MIX, SRC_NPZ, SRC_NPY, CKPT):
        assert os.path.exists(f), f"fixture source missing: {f}"
    tmp = tempfile.mkdtemp(prefix="maskprep_smoke_")
    print(f"scratch: {tmp}")
    ok = False
    try:
        # ---------------------------------------------------- GATE 0
        print("\n-- GATE 0: the apply script --")
        pristine = os.path.join(tmp, "head_pristine.py")
        shutil.copy2(HEAD, pristine)
        staged = os.path.join(tmp, "head_staged.py")
        s_new = staged_source()
        open(staged, "w").write(s_new)
        r = subprocess.run([PY, os.path.join(_ROOT, "scripts",
                                             "apply_maskprep_cache.py"),
                            "--check"],
                           env=dict(os.environ, MP_TARGET=pristine),
                           capture_output=True, text=True, cwd=_ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        for line in r.stdout.strip().split("\n"):
            print("   " + line)
        assert open(pristine).read() == open(HEAD).read(), \
            "--check WROTE to the target"
        r2 = subprocess.run([PY, os.path.join(_ROOT, "scripts",
                                              "apply_maskprep_cache.py"),
                             "--check"],
                            env=dict(os.environ, MP_TARGET=staged),
                            capture_output=True, text=True, cwd=_ROOT)
        assert r2.returncode != 0 and "idempotence" in (r2.stdout + r2.stderr), \
            "the idempotence guard did not refuse a second application"
        print("   idempotence guard: second application REFUSED  [OK]")

        # ---------------------------------------------------- GATE 1a
        print("\n-- GATE 1a: door unset, the SOURCE diff of do_train --")
        a = func_lines(open(pristine).read(), "do_train")
        b = func_lines(s_new, "do_train")
        hunks, rep = [], []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, a, b, autojunk=False).get_opcodes():
            if op == "insert":
                hunks.append(b[j1:j2])
            elif op in ("replace", "delete"):
                rep.append((a[i1:i2], b[j1:j2]))
        # the ONE replaced line: the loop's iterator. Its dark branch is
        # the literal old expression, so door-unset is the old loop.
        assert len(rep) == 1, f"more than one replaced hunk: {rep}"
        old_h, new_h = rep[0]
        assert [x.strip() for x in old_h] == ["for s0 in range(0, n, 8):"], old_h
        assert new_h[-1].strip() == "for s0 in _mp_starts:", new_h
        assert "range(0, n, 8) if _mp_cached is None" in "\n".join(new_h), \
            "the dark branch is not the literal old expression"
        hunks.append(new_h[:-1])          # the lookup + iterator statements
        code = lambda h: [x for x in h if x.strip()
                          and not x.strip().startswith("#")]
        assert len(hunks) == 2, f"{len(hunks)} inserted hunks, want 2"
        finish, lookup = sorted(hunks, key=lambda h: "_maskprep_lookup"
                                in "\n".join(h))
        # hunk 1 (lookup + iterator): binds _mp_* names and nothing else
        for line in code(lookup):
            assert "_mp_" in line or "_maskprep_" in line, \
                f"lookup hunk line unguarded: {line!r}"
        # hunk 2 (verify/restore): every line lives INSIDE `if _mp_key is
        # not None:` — with the door unset _mp_key is None and not one of
        # these lines executes
        fc = code(finish)
        assert fc[0].strip() == "if _mp_key is not None:", fc[:1]
        ind0 = len(fc[0]) - len(fc[0].lstrip())
        for line in fc[1:]:
            assert (len(line) - len(line.lstrip())) > ind0, \
                f"line escapes the `if _mp_key is not None:` guard: {line!r}"
        n_code = len(code(lookup)) + len(fc)
        n_all = len(lookup) + len(finish)
        print(f"   1 replaced line (the loop iterator; its dark branch is "
              f"the literal old `range(0, n, 8)`), {n_code} inserted code "
              f"lines, {n_all - n_code} comment lines, 2 hunks")
        for line in lookup + finish:
            print("     | " + line.rstrip())
        print("   hunk 1 binds only _mp_* names; every line of hunk 2 is "
              "inside `if _mp_key is not None:`  [OK]")

        # ---------------------------------------------------- fixture
        open(os.path.join(tmp, "mp_driver.py"), "w").write(DRIVER)
        fixdir = build_fixture(tmp)
        print(f"\n-- fixture: {N_ROWS} rows -> {fixdir}/.cache "
              f"(sliced from form_mix16; sha-fence stamped) --")

        # ---------------------------------------------------- GATE 1b
        print("\n-- GATE 1b: door unset, RUNTIME identity --")
        log_p, dt_p = run_case(tmp, fixdir, pristine,
                               os.path.join(tmp, "pristine.npz"), "pristine")
        log_s, dt_s = run_case(tmp, fixdir, staged,
                               os.path.join(tmp, "staged_dark.npz"),
                               "staged_dark")
        P = load(os.path.join(tmp, "pristine.npz"))
        S = load(os.path.join(tmp, "staged_dark.npz"))
        assert set(P) == set(S) == set(ARRAYS), (sorted(P), sorted(S))
        for nm in ARRAYS:
            eq(P[nm], S[nm], nm, "1b")
        assert "[maskprep]" not in log_s, "the dark run entered the organ"
        print(f"   arrays recorded: " + ", ".join(
            f"{nm}{P[nm].shape}" for nm in ARRAYS))
        print(f"   pristine {dt_p:.1f}s vs staged-dark {dt_s:.1f}s; "
              f"all four arrays np.array_equal; no [maskprep] line  [OK]")

        # ---------------------------------------------------- GATE 2
        print("\n-- GATE 2: door set, MISS then HIT --")
        log_m, dt_m = run_case(tmp, fixdir, staged,
                               os.path.join(tmp, "miss.npz"), "miss",
                               ALG_MASKPREP_CACHE="1")
        assert "cache MISS" in log_m, log_m[-1500:]
        assert "[maskprep] cached ->" in log_m, log_m[-1500:]
        bucket = [x for x in os.listdir(os.path.join(tmp, "buckets"))]
        assert len(bucket) == 1, bucket
        bpath = os.path.join(tmp, "buckets", bucket[0])
        M = load(os.path.join(tmp, "miss.npz"))
        for nm in ARRAYS:
            eq(P[nm], M[nm], nm, "2-miss-recorded")
        # MASSB is the /301-normalized copy of the recorded raw mass, so
        # the raw recording is compared run-to-run above and the banked
        # array is compared organ-to-organ (MISS vs HIT) below — never
        # re-derived here (a checker that recomputes the head's formula
        # is a second meter; the head's own verification is the fence).
        for nm in ("masks", "facts", "nl0"):
            eq(P[nm], M[FINALS[nm]], nm, "2-miss-final")
        log_h, dt_h = run_case(tmp, fixdir, staged,
                               os.path.join(tmp, "hit.npz"), "hit",
                               ALG_MASKPREP_CACHE="1")
        assert "CACHE HIT" in log_h and "verified on 2 batches" in log_h, \
            log_h[-1500:]
        H = load(os.path.join(tmp, "hit.npz"))
        assert H["masks"].shape[0] == 16, \
            f"the hit run recomputed {H['masks'].shape[0]} rows, want 16"
        for nm in ARRAYS:
            eq(M[FINALS[nm]], H[FINALS[nm]], FINALS[nm], "2-hit-vs-miss")
        for nm in ("masks", "facts", "nl0"):
            eq(P[nm], H[FINALS[nm]], nm, "2-hit-final")
        eq(P["masks"][:16], H["masks"], "masks[:16]", "2-hit-recompute")
        assert str(M["key"]) == str(H["key"]), "the key moved between runs"
        print(f"   bucket {bucket[0]} "
              f"({os.path.getsize(bpath) / 1e6:.2f} MB) key={str(M['key'])}")
        print(f"   MISS {dt_m:.1f}s -> HIT {dt_h:.1f}s "
              f"({N_ROWS // 8} batches -> 2); every array the HIT run hands "
              f"do_train equals the MISS run's, and MASKS/FACTS/NL0 "
              f"equal the pristine run's  [OK]")
        print("   " + [x for x in log_h.split("\n") if "CACHE HIT" in x][0])

        # ---------------------------------------------------- GATE 3
        print("\n-- GATE 3: tamper --")
        good = load(bpath)
        raw = np.load(bpath, allow_pickle=False)
        meta = {k: raw[k] for k in raw.files if k.startswith("_")}
        for row, expect_fail in ((0, True), (20, False)):
            t = {k: v.copy() for k, v in good.items() if not k.startswith("_")}
            t["MASKS"][row, 0, 0] = 1.0 - t["MASKS"][row, 0, 0]
            np.savez_compressed(bpath, **meta, **t)
            log_t, _ = run_case(tmp, fixdir, staged,
                                os.path.join(tmp, f"tamper{row}.npz"),
                                f"tamper{row}", must_fail=expect_fail,
                                ALG_MASKPREP_CACHE="1")
            if expect_fail:
                assert "CACHE VERIFICATION FAILED" in log_t, log_t[-2000:]
                line = [x for x in log_t.split("\n")
                        if "CACHE VERIFICATION FAILED" in x][0]
                print(f"   row {row} (inside the verified batches): DIED — "
                      f"{line.strip()[:110]}...  [OK]")
            else:
                assert "CACHE HIT" in log_t
                T = load(os.path.join(tmp, f"tamper{row}.npz"))
                assert not np.array_equal(P["masks"], T[FINALS["masks"]])
                print(f"   row {row} (outside the verified batches): NOT "
                      f"caught, and the tampered row was served — the "
                      f"fence's DECLARED scope is the sampled batches")
        np.savez_compressed(bpath, **meta,
                            **{k: v for k, v in good.items()
                               if not k.startswith("_")})

        # ---------------------------------------------------- GATE 4
        print("\n-- GATE 4: the key --")
        base_k, base_fp = keyrun(tmp, fixdir, staged, "base")
        cases = [
            ("ALG_POLAR=1 (an env door)", dict(ALG_POLAR="1"), False, {"env"}),
            ("SC_EVAL unset", dict(SC_EVAL=None), False, {"env"}),
            ("NB_PERSLOT=0", dict(NB_PERSLOT="0"), False, {"env"}),
            ("MP_SEED=1 (build_params seed)", dict(MP_SEED="1"), False,
             {"params", "seed"}),
            ("STEPS/LR/VAL_EVERY/SNAP_EVERY (trainer-only)",
             dict(STEPS="99999", LR="1", VAL_EVERY="7", SNAP_EVERY="3"),
             True, set()),
        ]
        for name, over, same, want in cases:
            k, fp = keyrun(tmp, fixdir, staged,
                           name.split()[0].replace("=", "").replace("/", ""),
                           **over)
            d = set(fp_diff(base_fp, fp))
            if same:
                assert k == base_k, f"{name}: key MOVED (fields {sorted(d)})"
                print(f"   {name:44s} key SAME  [OK]")
            else:
                assert k != base_k, f"{name}: key did not move"
                assert d == want, f"{name}: fields moved {sorted(d)} != {sorted(want)}"
                print(f"   {name:44s} key MOVED (fields {sorted(d)})  [OK]")
        # a side file's BYTES
        bc = os.path.join(tmp, "bindbus_copy.npz")
        shutil.copy2(os.path.join(_ROOT, ".cache", "bindbus_codes512.npz"), bc)
        k_c, fp_c = keyrun(tmp, fixdir, staged, "bindcopy", BIND_CODES=bc)
        # same bytes at a new path: only the recorded path moves
        assert k_c != base_k and fp_diff(base_fp, fp_c) == ["env"]
        with open(bc, "ab") as f:
            f.write(b"\0")
        k_c2, fp_c2 = keyrun(tmp, fixdir, staged, "bindcopy2", BIND_CODES=bc)
        assert k_c2 != k_c and fp_diff(fp_c, fp_c2) == ["env_files"]
        print(f"   {'BIND_CODES bytes changed (same path)':44s} key MOVED "
              f"(fields ['env_files'])  [OK]")
        # the head source's own bytes
        head2 = os.path.join(tmp, "head_staged2.py")
        open(head2, "w").write(s_new + "\n# one comment\n")
        k_h, fp_h = keyrun(tmp, fixdir, head2, "headbytes")
        assert k_h != base_k and fp_diff(base_fp, fp_h) == ["head"]
        print(f"   {'head source bytes changed':44s} key MOVED "
              f"(fields ['head'])  [OK]")
        # one perturbed parameter
        k_p, fp_p = keyrun(tmp, fixdir, staged, "perturb", MP_PERTURB="1")
        assert k_p != base_k and fp_diff(base_fp, fp_p) == ["params"]
        print(f"   {'one parameter entry nudged by 1e-3':44s} key MOVED "
              f"(fields ['params'])  [OK]")
        # the declared exclusion: both arms declare it, one sets the door
        k_i1, _ = keyrun(tmp, fixdir, staged, "ign1",
                         ALG_MASKPREP_IGNORE="ALG_POLAR")
        k_i2, _ = keyrun(tmp, fixdir, staged, "ign2",
                         ALG_MASKPREP_IGNORE="ALG_POLAR", ALG_POLAR="1")
        assert k_i1 == k_i2 and k_i1 != base_k
        print(f"   {'ALG_MASKPREP_IGNORE=ALG_POLAR, door on vs off':44s} key "
              f"SAME (declared exclusion)  [OK]")

        print("\n" + "=" * 74)
        print("ALL GATES PASS — the mask-prep cache is byte-inert when dark, "
              "verified when live,\nand dies loudly when the bucket lies.")
        print("=" * 74)
        ok = True
    finally:
        if os.environ.get("MP_KEEP") or not ok:
            print(f"[smoke] scratch KEPT: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
