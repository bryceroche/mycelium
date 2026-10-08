"""polar_latent_hook.py -- 2026-10-08, Bryce: "is the silhouette 128 dims?" A pure RUNTIME
monkeypatch (no file on disk is read-modified-written -- scripts/phase1_algebra_head.py, a chain-
imported file, is never edited; the precedent is scripts/eyes_autopsy.py's own `H._make_bank`
monkeypatch) that taps the POLAR WAIST's 128-d bottleneck -- the content-plane squeeze INSIDE the
loop (_polar_waist: 384 content dims -> ALG_POLAR_D (128) -> 384, scripts/phase1_algebra_head.py
:1569) -- and appends it to the module's EXISTING H._CENSUS hook under tag "polar_latent", in the
SAME (kb, tag, array) convention every other census tap already uses (so welford_atlas.py's
existing harvesting code -- `got = {kb: arr for (kb, tag, arr) in H._CENSUS if tag == TAG}` --
reads it identically to "state").

WHY TWO FUNCTIONS, NOT ONE: `_polar_waist(u, p, state)` carries no `kb` argument (its one call
site, breath_step :6203, has `kb` in scope but does not pass it in) -- so the latent itself cannot
be labelled by breath from inside a patch on `_polar_waist` alone. `breath_step(p, state, kb, ctx)`
DOES receive `kb`, and `state` is the SAME dict (a fresh dict per forward, shared across breaths,
per `_polar_waist`'s own docstring) passed on into `_polar_waist` -- so the breath_step patch
stashes kb into `state` right before calling the real breath_step (popped after, so no other
reader of `state` ever sees it), and the `_polar_waist` patch reads it back out. Both patches call
the REAL function FIRST and return its result UNCHANGED -- this is a READ-ONLY tap: forward()'s
output is bit-identical with or without `install()`, and the 128-d latent (`u @ state["polar_wd_eff"]`)
is recomputed AFTER the real call from the SAME cached `polar_wd_eff` the real call itself just
populated (deterministic given `p`; recomputing it changes nothing).

Only called from kb in breath_step's own loop range (1..K_B-1 -- "breath 0 is outside time", the
module's own phrase; breath 0's grounding pass does not go through breath_step/_polar_waist at
all), and only when "waist" is not severed (ALG_POLAR_D on, the "waist" sever absent) -- both
conditions hold for every body this hook has been run against so far (PMS8_241, plain family env).

usage:
    import phase1_algebra_head as H
    import polar_latent_hook
    polar_latent_hook.install(H)
    # ... H._CENSUS = []; H.forward(...); harvest (kb, "polar_latent", arr) entries as usual ...
"""

TAG = "polar_latent"


def install(H):
    """Idempotent: a second call on the same module is a no-op (checked via a sentinel attribute
    set on the module itself, never on a copy -- a fresh process always starts unpatched)."""
    if getattr(H, "_polar_latent_hook_installed", False):
        return
    assert H.ALG_POLAR and H.POLAR_D > 0, (
        "polar_latent_hook.install: ALG_POLAR_D is not set in this process's env -- "
        "nothing to tap (the family env must be built BEFORE importing phase1_algebra_head)")

    _orig_breath_step = H.breath_step
    _orig_polar_waist = H._polar_waist

    def _patched_breath_step(p, state, kb, ctx):
        state["_census_cur_kb"] = kb
        try:
            return _orig_breath_step(p, state, kb, ctx)
        finally:
            state.pop("_census_cur_kb", None)

    def _patched_polar_waist(u, p, state):
        out = _orig_polar_waist(u, p, state)          # unchanged behavior, called first
        cs = getattr(H, "_CENSUS", None)
        if cs is not None:
            wd = state.get("polar_wd_eff")             # (H_W, POLAR_D); populated by the call above
            kb = state.get("_census_cur_kb")
            if wd is not None and kb is not None:
                latent = u @ wd                         # (B, L_TOT, POLAR_D) -- the 128-d squeeze itself
                cs.append((kb, TAG, latent.realize().numpy()))
        return out

    H.breath_step = _patched_breath_step
    H._polar_waist = _patched_polar_waist
    H._polar_latent_hook_installed = True
