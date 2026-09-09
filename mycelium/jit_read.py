"""jit_read.py — THE JIT'D READ FORWARD (2026-09-08; door ALG_JIT_READ).

THE MEASUREMENT THAT ASKED FOR IT (ledger 2026-09-08, "PERF APPLIED"):
the read path's forward() is EAGER — the only @TinyJit in
phase1_algebra_head.py lives inside do_train() — so a read pays full
schedule+dispatch per batch (~1.3 s per two-pass batch at B=8 on the
card) while the TRAINED step does fwd+bwd in 0.5 s under the JIT. This
module gives the read path the trainer's own idiom, and nothing else:

  fixed pre-allocated input buffers -> assign in place -> ONE captured
  graph per (pass shape, requested outputs, env, param dict) -> replay.

IT ADDS NO ARITHMETIC. Every number still comes out of
phase1_algebra_head.forward(); this module only decides which tensors
are handed to it and when the graph is re-used. With ALG_JIT_READ unset
read_forward() is a bare call to forward() with the caller's own
arguments — byte-inert by construction (the `keys` argument is dropped
on the floor).

------------------------------------------------------------------ THE KEY

A captured graph BAKES every Python-level branch forward() took while it
was building — and forward()/breath_step() read os.environ AT BUILD TIME.
That is exactly THE UNLIT STOVE (ledger 2026-09-06): SC_EVAL exported
process-wide collapsed the mode-2 blend to open identity at the
TRAINER's capture and the seal never existed again. One captured graph
serves ONE seal mode, forever.

So the cache key carries, in full:

  ("pass"): which optional ports are fed (slot_mask / fact_buf /
      mh_mass / mh_atlas_traj), with their shapes and dtypes. The open
      pass (no slot_mask -> no breath loop) and the masked pass are
      DIFFERENT GRAPHS; feeding fact_buf changes the graph again
      (_fact_inject). Absence is part of the signature: forward()'s own
      guards are `is not None` tests.
  ("B"): the batch dimension, and every input's trailing shape.
  ("outs"): the requested output keys — the captured `ret` is fixed at
      capture time, so a different key set is a different graph.
  ("env"): (name, value) for EVERY environment variable name that
      appears in the head's source, read fresh on every call. The name
      list is mined from the source with a regex (the self-maintaining
      idiom the mask-prep cache already uses: a door added tomorrow
      lands in the key without anyone remembering to add it). The list
      is deliberately WIDER than the read path — a name that cannot
      change during a read costs one tuple entry and nothing else,
      while a name left out costs a wrong meter.
  ("params"): id() of the param dict, with the dict itself held in the
      slot and identity-asserted on every call. Weights are swapped by
      p[k].assign(...).realize() — an IN-PLACE write to the very
      buffers the graph captured — so a checkpoint swap does NOT
      invalidate the graph (proved: see the checkpoint-swap gate in
      the scratchpad proofs). A DIFFERENT dict object would not be
      seen by the captured graph at all, so it gets its own slot.

Nothing keys on the checkpoint path: the weights are data in captured
buffers, not structure.

------------------------------------------------------------ WHAT IS NOT SAFE

- The returned tensors at replay are THE CAPTURE'S OWN tensors; their
  buffers are overwritten by the next replay. read_forward() therefore
  copies every requested output to a fresh numpy array before it
  returns, and hands back a view whose members answer
  `.realize().numpy()` with that copy. Callers written against the
  eager path keep working, and a caller that holds an output across
  another forward() gets its own data, not a moving buffer.
- A key that forward() did not emit is silently absent from the view
  (the eager `"dup" in o0` / `"fst_s" in o` idioms keep working). Ask
  for a superset; you get what exists.
- Slots are never evicted. ALG_JIT_READ_MAX (default 32) is a loud
  ceiling: if the key thrashes, this raises instead of quietly
  re-capturing a graph per batch.

ENV
  ALG_JIT_READ        1 = arm the door (unset/0 = eager, byte-inert)
  ALG_JIT_READ_MAX    slot ceiling (default 32), loud on overflow
  ALG_JIT_READ_DEBUG  1 = print a line per capture and per key change

CPU selftest (no GPU, no checkpoint, no head):
  .venv/bin/python3 -m mycelium.jit_read --selftest
"""
import os
import re
import sys

__all__ = ["read_forward", "enabled", "env_names", "slot_count", "reset"]

_ENV_RE = re.compile(
    r"os\.environ(?:\.get)?[\(\[]\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']")

# Names the head does not read itself but tinygrad does, and which change
# what a captured graph IS (kernel selection / whether the JIT runs).
_EXTRA_ENV = ("DEV", "JIT", "BEAM", "JITBEAM", "NOOPT",
              "IGNORE_JIT_FIRST_BEAM", "ALG_JIT_READ")

_ENV_NAMES_CACHE = {}
_SLOTS = {}
_STATS = {"captures": 0, "calls": 0}


def enabled():
    """The door. Read per call: read_batch's env_scope swaps the whole
    environment between meters, and a door that latched at import would
    outlive the meter that opened it."""
    try:
        return bool(int(os.environ.get("ALG_JIT_READ", "0") or 0))
    except ValueError:
        raise AssertionError(
            f"ALG_JIT_READ={os.environ.get('ALG_JIT_READ')!r} is not an "
            f"integer — a mistyped door must never read as OFF")


def _debug():
    return bool(int(os.environ.get("ALG_JIT_READ_DEBUG", "0") or 0))


def env_names(source_path):
    """Every environment variable name that appears in the head's source,
    plus the tinygrad dials. Mined from the text so a door added
    tomorrow is in the key tomorrow (the mask-prep cache's idiom)."""
    names = _ENV_NAMES_CACHE.get(source_path)
    if names is None:
        with open(source_path) as fh:
            src = fh.read()
        names = tuple(sorted(set(_ENV_RE.findall(src)) | set(_EXTRA_ENV)))
        _ENV_NAMES_CACHE[source_path] = names
    return names


def _source_of(fwd):
    mod = sys.modules.get(getattr(fwd, "__module__", None))
    path = getattr(mod, "__file__", None)
    assert path, (
        "read_forward could not locate the source of the forward it was "
        "handed — the env key is mined from that source and a key built "
        "from nothing is a baked graph waiting to happen")
    return os.path.abspath(path)


def _env_key(fwd):
    return tuple((n, os.environ.get(n)) for n in env_names(_source_of(fwd)))


def _is_tensor(x):
    return x.__class__.__name__ == "Tensor" and hasattr(x, "uop")


def _spec(x):
    """(shape, dtype-name) of a Tensor or ndarray input."""
    if _is_tensor(x):
        return (tuple(int(s) for s in x.shape), str(x.dtype))
    return (tuple(int(s) for s in x.shape), str(x.dtype))


def _dtype_of(x):
    from tinygrad import dtypes
    import numpy as np
    if _is_tensor(x):
        return x.dtype
    a = np.asarray(x)
    if a.dtype == np.int32 or a.dtype == np.int64:
        return dtypes.int
    return dtypes.float


def _alloc(x, B):
    """A fixed, realized, contiguous buffer shaped (B,) + x.shape[1:]."""
    import numpy as np
    from tinygrad import Tensor
    dt = _dtype_of(x)
    shape = (B,) + tuple(int(s) for s in x.shape[1:])
    npdt = np.int32 if "int" in str(dt) else np.float32
    return Tensor(np.zeros(shape, npdt), dtype=dt).contiguous().realize()


def _feed(buf, x, n_real, B):
    """Assign x into the fixed buffer, padding rows [n_real, B) with row 0
    — the readers' own `sl[:1].repeat(pad)` idiom, applied one layer down
    so a short tail batch cannot capture a second graph. Returns the
    unrealized assign tensor; the caller realizes them all at once (one
    schedule, not N dispatches — the trainer's perf audit #2)."""
    import numpy as np
    from tinygrad import Tensor
    if _is_tensor(x):
        t = x
        if n_real < B:
            t = Tensor.cat(t[:n_real], t[:1].repeat((B - n_real,) +
                                                    (1,) * (len(t.shape) - 1)),
                           dim=0)
        return buf.assign(t.contiguous())
    a = np.asarray(x)
    if n_real < B:
        a = np.concatenate([a[:n_real], a[:1].repeat(B - n_real, axis=0)])
    npdt = np.int32 if "int" in str(buf.dtype) else np.float32
    return buf.assign(Tensor(a.astype(npdt), dtype=buf.dtype).contiguous())


class _Arr:
    """An output that answers the eager path's `.realize().numpy()`.
    Holds a private copy: the captured graph re-writes its own output
    buffers on the next replay."""
    __slots__ = ("_a",)

    def __init__(self, a):
        self._a = a

    def realize(self):
        return self

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape

    def __repr__(self):
        return f"_Arr{self._a.shape}"


class OutView(dict):
    """dict of key -> _Arr (or list of _Arr). `k in o`, `o[k]`, and
    `o[k].realize().numpy()` behave as the eager output dict does."""
    available = ()


class _Slot:
    __slots__ = ("key", "jit", "bufs", "opt_names", "p", "B", "meta")

    def __init__(self, key, jit, bufs, opt_names, p, B, meta):
        self.key = key
        self.jit = jit
        self.bufs = bufs
        self.opt_names = opt_names
        self.p = p
        self.B = B
        self.meta = meta


_OPT_PORTS = ("slot_mask", "revoke", "tail", "drop", "anchor", "amask",
              "gmod", "pmask", "lsent", "reg", "fact_buf", "mh_mass",
              "mh_atlas_traj")


def _make_slot(fwd, p, key, B, ts, tk, se, opts, keys):
    from tinygrad.engine.jit import TinyJit
    bufs = {"trunk": _alloc(ts, B), "tokmask": _alloc(tk, B),
            "sent": _alloc(se, B)}
    opt_names = tuple(sorted(opts))
    for nm in opt_names:
        bufs[nm] = _alloc(opts[nm], B)
    meta = {"avail": (), "struct": None}

    @TinyJit
    def _f():
        o = fwd(p, bufs["trunk"], bufs["tokmask"], bufs["sent"],
                **{nm: bufs[nm] for nm in opt_names})
        meta["avail"] = tuple(o.keys())
        flat = {}
        struct = []
        for k in keys:
            if k not in o:
                continue
            v = o[k]
            if isinstance(v, (list, tuple)):
                struct.append((k, len(v)))
                for i, t in enumerate(v):
                    flat[f"{k}\x00{i}"] = t
            else:
                struct.append((k, None))
                flat[k] = v
        meta["struct"] = tuple(struct)
        return flat

    return _Slot(key, _f, bufs, opt_names, p, B, meta)


def read_forward(fwd, p, trunk, tokmask, sent, keys=None, n_real=None,
                 **kw):
    """Drop-in for `forward(p, trunk, tokmask, sent, **kw)` on the READ
    path.

    ALG_JIT_READ unset -> `fwd(p, trunk, tokmask, sent, **kw)`, verbatim;
    `keys` and `n_real` are ignored and nothing in this module runs. The
    caller gets the head's own output dict of live Tensors, exactly as
    before.

    ALG_JIT_READ=1 -> the batch is assigned into fixed buffers and a
    captured graph is replayed; the return is an OutView whose members
    answer `.realize().numpy()` with numpy copies of the requested keys.

    keys    the output keys this call site consumes (required when the
            door is open — the captured return is fixed at capture, so
            it cannot be discovered later). Keys forward() does not
            emit are absent from the view, as they are absent from the
            eager dict.
    n_real  rows of the batch that are real (default: all). Rows beyond
            it are padded with row 0 before the graph runs and the
            returned arrays are sliced back to n_real. The readers
            already pad to their own batch size, so this is a second
            safety net, not the primary one.
    """
    if not enabled():
        return fwd(p, trunk, tokmask, sent, **kw)

    import numpy as np
    from tinygrad import Tensor

    assert keys, (
        "read_forward with ALG_JIT_READ=1 needs an explicit `keys` tuple: "
        "a captured graph's return value is fixed at capture time, so the "
        "outputs a call site consumes must be declared before the graph "
        "exists. (Eager mode ignores `keys`; that is the only difference.)")
    keys = tuple(keys)
    opts = {k: v for k, v in kw.items() if v is not None}
    bad = [k for k in opts if k not in _OPT_PORTS]
    assert not bad, (
        f"read_forward: unknown forward() port(s) {bad} — add them to "
        f"_OPT_PORTS after checking they are Tensors, not flags")
    for k, v in opts.items():
        assert _is_tensor(v) or hasattr(v, "shape"), (
            f"read_forward: port {k!r} is neither a Tensor nor an array")

    n_in = int(trunk.shape[0])
    n_real = n_in if n_real is None else int(n_real)
    assert 0 < n_real <= n_in, (n_real, n_in)
    B_env = os.environ.get("ALG_JIT_READ_B")
    B = int(B_env) if B_env else n_in
    assert n_in <= B, (
        f"read_forward: batch of {n_in} rows exceeds the fixed JIT batch "
        f"B={B} (ALG_JIT_READ_B). One graph serves one batch shape; raise "
        f"ALG_JIT_READ_B or feed the reader's own batch size.")

    key = (("outs", keys),
           ("B", B),
           ("in", _spec(trunk), _spec(tokmask), _spec(sent)),
           ("opts", tuple((k, _spec(v)) for k, v in sorted(opts.items()))),
           ("params", id(p)),
           ("env", _env_key(fwd)))

    _STATS["calls"] += 1
    slot = _SLOTS.get(key)
    if slot is None:
        cap = int(os.environ.get("ALG_JIT_READ_MAX", "32") or 32)
        assert len(_SLOTS) < cap, (
            f"read_forward: {len(_SLOTS)} captured graphs already and a "
            f"{len(_SLOTS) + 1}th key arrived — the key is THRASHING (an "
            f"env var changing per call, or a param dict rebuilt per "
            f"batch). Refusing to capture: a graph per batch is slower "
            f"than eager and hides a real bug. Raise ALG_JIT_READ_MAX "
            f"only when the extra graphs are intended.")
        slot = _make_slot(fwd, p, key, B, trunk, tokmask, sent, opts, keys)
        _SLOTS[key] = slot
        _STATS["captures"] += 1
        if _debug():
            print(f"[jit-read] new slot #{len(_SLOTS)}: B={B} "
                  f"ports={tuple(sorted(opts))} keys={keys} "
                  f"SC_EVAL={os.environ.get('SC_EVAL', '(unset)')}",
                  flush=True)
    assert slot.p is p, (
        "read_forward: the param dict changed identity under a key that "
        "matched — the captured graph holds the OLD tensors and would "
        "read stale weights. (Swap weights with p[k].assign(...); never "
        "rebuild the dict.)")

    feeds = [_feed(slot.bufs["trunk"], trunk, n_real, B),
             _feed(slot.bufs["tokmask"], tokmask, n_real, B),
             _feed(slot.bufs["sent"], sent, n_real, B)]
    for nm in slot.opt_names:
        feeds.append(_feed(slot.bufs[nm], opts[nm], n_real, B))
    Tensor.realize(*feeds)

    flat = slot.jit()

    out = OutView()
    out.available = slot.meta["avail"]
    for k, n in (slot.meta["struct"] or ()):
        if n is None:
            a = np.array(flat[k].numpy(), copy=True)
            out[k] = _Arr(a[:n_real] if n_real < B else a)
        else:
            lst = []
            for i in range(n):
                a = np.array(flat[f"{k}\x00{i}"].numpy(), copy=True)
                lst.append(_Arr(a[:n_real] if n_real < B else a))
            out[k] = lst
    return out


def slot_count():
    return len(_SLOTS)


def stats():
    return dict(_STATS, slots=len(_SLOTS))


def reset():
    """Drop every captured graph (tests only; a read never needs this)."""
    _SLOTS.clear()
    _STATS["captures"] = 0
    _STATS["calls"] = 0


# --------------------------------------------------------------- selftest
def selftest():
    """CPU-only, no head, no checkpoint, no GPU: the door's inertness,
    the key's composition, the padding/slicing contract, and the
    thrash ceiling — against a fake `forward` that records its inputs."""
    import numpy as np
    import types
    from tinygrad import Tensor, dtypes

    reset()
    fake_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_jit_read_fake.py")
    with open(fake_src, "w") as fh:
        fh.write("import os\n"
                 "os.environ.get('FAKE_DOOR', '0')\n"
                 "os.environ['FAKE_HARD']\n")
    mod = types.ModuleType("_jit_read_fake")
    mod.__file__ = fake_src
    sys.modules["_jit_read_fake"] = mod

    calls = {"n": 0}

    def fwd(p, trunk, tokmask, sent, slot_mask=None, fact_buf=None, **kw):
        calls["n"] += 1
        s = trunk.sum(axis=1) * p["w"]                 # (B, D)
        o = {"a": s, "b": s * 2.0, "list": [s, s + 1.0]}
        if slot_mask is not None:
            o["masked"] = s + slot_mask.sum(axis=1, keepdim=True)
        return o
    fwd.__module__ = "_jit_read_fake"

    B, T, D = 4, 3, 2
    p = {"w": Tensor(np.ones((D,), np.float32),
                     dtype=dtypes.float).contiguous().realize()}
    x = np.arange(B * T * D, dtype=np.float32).reshape(B, T, D)
    tr = Tensor(x, dtype=dtypes.float)
    tk = Tensor(np.ones((B, T), np.float32), dtype=dtypes.float)
    se = Tensor(np.zeros((B, T), np.int32), dtype=dtypes.int)

    # --- door OFF: byte-inert, `keys` dropped, head's own dict returned
    os.environ.pop("ALG_JIT_READ", None)
    assert not enabled()
    o = read_forward(fwd, p, tr, tk, se, keys=("a",))
    assert set(o) == {"a", "b", "list"} and calls["n"] == 1
    assert slot_count() == 0, "the door was shut and a slot was built"
    ref_a = o["a"].realize().numpy()

    # --- door ON: same numbers, three calls (eager / capture / replay)
    os.environ["ALG_JIT_READ"] = "1"
    os.environ["FAKE_HARD"] = "x"
    for i in range(3):
        v = read_forward(fwd, p, tr, tk, se, keys=("a", "list", "nope"))
        assert np.array_equal(v["a"].realize().numpy(), ref_a), i
        assert "nope" not in v and "b" not in v
        assert [q.realize().numpy().tolist() for q in v["list"]] == \
            [ref_a.tolist(), (ref_a + 1.0).tolist()], i
        assert "b" in v.available
    assert slot_count() == 1, slot_count()

    # --- the ports are part of the key (open pass vs masked pass)
    sm = Tensor(np.ones((B, T), np.float32), dtype=dtypes.float)
    v = read_forward(fwd, p, tr, tk, se, keys=("masked",), slot_mask=sm)
    assert slot_count() == 2 and "masked" in v

    # --- the env is part of the key: a name the source reads moves it
    os.environ.pop("FAKE_DOOR", None)
    a_open = read_forward(fwd, p, tr, tk, se, keys=("a",))["a"] \
        .realize().numpy()
    assert slot_count() == 3, slot_count()           # keys=("a",), no door
    os.environ["FAKE_DOOR"] = "7"
    read_forward(fwd, p, tr, tk, se, keys=("a",))
    assert slot_count() == 4, "FAKE_DOOR=7 reused the unset graph"
    read_forward(fwd, p, tr, tk, se, keys=("a",))
    assert slot_count() == 4, "same env re-captured"
    os.environ.pop("FAKE_DOOR")
    v = read_forward(fwd, p, tr, tk, se, keys=("a",))
    assert slot_count() == 4, "unset did not return to its own graph"
    assert np.array_equal(v["a"].realize().numpy(), a_open)
    # a name the source does NOT read must not move the key
    os.environ["NOT_IN_SOURCE_XYZ"] = "1"
    read_forward(fwd, p, tr, tk, se, keys=("a",))
    assert slot_count() == 4, "an unrelated env name moved the key"
    os.environ.pop("NOT_IN_SOURCE_XYZ")
    assert "SC_EVAL" not in env_names(fake_src)          # fake source
    # the real head's names: SC_EVAL and the seal dials MUST be there
    head = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "phase1_algebra_head.py")
    if os.path.exists(head):
        nm = env_names(head)
        for req in ("SC_EVAL", "SC_KB", "ALG_SHELF_CIRCLE", "ALG_PC_MIX",
                    "ALG_BREATH", "ALG_ALT2", "ALG_MASKHEAD", "ALG_INV",
                    "ALG_MINE_BREATHS", "DEV"):
            assert req in nm, f"env key misses {req}"

    # --- the outs are part of the key
    read_forward(fwd, p, tr, tk, se, keys=("a", "b"))
    assert slot_count() == 5, slot_count()

    # --- a rebuilt param dict gets its OWN graph (never served stale):
    # id(p) is in the key, and the slot holds a reference to p so that id
    # cannot be recycled underneath it. The identity assert on the hit
    # path is the belt to that braces.
    n5 = slot_count()
    p2 = dict(p)
    read_forward(fwd, p2, tr, tk, se, keys=("a",))
    assert slot_count() == n5 + 1, "a rebuilt param dict reused a graph"
    assert any(sl.p is p2 for sl in _SLOTS.values())

    # --- weights swapped IN PLACE are seen by the replayed graph
    p["w"].assign(Tensor(np.full((D,), 3.0, np.float32),
                         dtype=dtypes.float)).realize()
    v = read_forward(fwd, p, tr, tk, se, keys=("a",))
    assert np.allclose(v["a"].realize().numpy(), ref_a * 3.0), \
        "an in-place weight swap was invisible to the captured graph"
    p["w"].assign(Tensor(np.ones((D,), np.float32),
                         dtype=dtypes.float)).realize()

    # --- a short tail batch pads to B, runs one graph, slices back
    n_slots = slot_count()
    v = read_forward(fwd, p, tr, tk, se, keys=("a",), n_real=2)
    assert v["a"].realize().numpy().shape[0] == 2
    assert np.array_equal(v["a"].realize().numpy(), ref_a[:2])
    assert slot_count() == n_slots, "a tail batch captured a second graph"

    # --- the thrash ceiling is loud
    os.environ["ALG_JIT_READ_MAX"] = str(slot_count())
    try:
        read_forward(fwd, p, tr, tk, se, keys=("a", "b", "list"))
        raise SystemExit("slot ceiling did not fire")
    except AssertionError as e:
        assert "THRASHING" in str(e)
    os.environ.pop("ALG_JIT_READ_MAX")

    os.environ.pop("ALG_JIT_READ", None)
    os.environ.pop("FAKE_HARD", None)
    os.remove(fake_src)
    reset()
    print("[jit-read] selftest PASS: door inert when unset; key spans "
          "ports/outs/batch/env/param-identity; unrelated env names do "
          "not move it; in-place weight swap visible to the replayed "
          "graph; tail batch padded+sliced on one graph; thrash ceiling "
          "loud. No GPU, no head, no checkpoint.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
