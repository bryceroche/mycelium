# THE T6 CLOCK AUDIT (2026-10-05)

Read-only audit. No files modified, no GPU work run. CPU-only check run:
`DEV=CPU .venv/bin/python3 mycelium/rotor_clock.py` — self-tests pass,
wheel table confirmed (below). All other evidence is static-read from
the repo (`scripts/phase1_algebra_head.py`, `mycelium/rotor_clock.py`,
`.cache/*_chain.sh`, `docs/phase1_skeleton_spec.md`).

Scope note up front: the deployed manifest (`.cache/GENERATION.json`,
gen-41, 2026-08-10) does NOT carry any of this — the T6 clock lives only
in the research lineage (PM35/PMS* family, `ALG_BREATH=7 ALG_POLAR=1
ALG_CLOCK_CANON=1 ALG_SIXWAVE=1`, confirmed live in `.cache/family_chain
.sh:6`, `pms8_chain.sh:6`, `pm35_241_chain.sh:7`, `clock_reads_chain.sh
:22`, `ctrls242_chain.sh:5`). "Promoted to conductor" is a design
decision recorded in the ledger (2026-10-04 12:38), not yet a build.

---

## 0. VERIFIED: the clock itself runs as specified

```
$ DEV=CPU .venv/bin/python3 mycelium/rotor_clock.py
[rotor_clock] SEXTET: 6 loop breaths @ 60deg, breath-0 outside time
[rotor_clock] wheel table (deg):
[[  0.   0.   0.] [ 60. 120.   0.] [120. 240.   0.] [180.   0.   0.]
 [240. 120.   0.] [300. 240.   0.]]
[rotor_clock] odometer diag mean: 1.000 (off-diag max 0.000)
[rotor_clock] ALL SELF-TESTS PASS
```
`K_BREATH=7`, `N_LOOP=6`, `QUANTUM=60deg` (`mycelium/rotor_clock.py:44-47`);
`is_clocked`/`phase_of` refuse breath 0 by raising (`:62-76`), matching
the "breath-0 is outside time" law.

---

## 1. INVENTORY — where time enters

### (a) Through the CLOCK (wheel tables / phase_of / canon frame / Q-rotation / E&B / six-wave)

| Site | file:line | Gates | Road/Gain | Train == Read? |
|---|---|---|---|---|
| Band allocation | `.cache/polar_bands.json` (64/256 planes: 32 breath-hand / 16 parity / 16 pass) | which planes turn at all | road (frozen allocation, `"frozen": true`) | identical — read once, cached (`_polar_tables`, `scripts/phase1_algebra_head.py:972`) |
| `_polar_tables()` | `:972-1030` | builds `(dc,ds,ac,as,wheel_of)` from `rotor_clock.wheel_table()` | road; asserts the compounding contract (`:1012-1016`: accumulated increments must reproduce the absolute table, `np.allclose` — VERIFIED, this assert runs on every process that imports the module) | identical by construction (same table object, no RNG) |
| State turn (sextet) | `:5381-5385` (`_pol_u = _rot2(_pol_u, _pdc[kb-1], _pds[kb-1])`) | the per-slot content state's clock planes, every breath | **road**, gated only by `1 <= kb <= _RC_N_LOOP` — no learnable gain anywhere in the rotation itself | identical: `_rot2`/`breath_step` run the same way at train and at `loop_val.py`/`chain_acc.py` read (same `forward()`/`breath_step()` code path; readers differ only in which `ctx` ports are populated, e.g. `stop_after`) |
| Q-side rotation, mixer main path | `:4895-4909` (`POLAR_QROT>=2`, builds `_bq2`, uses `mac`/`mas` = absolute phase) | the slot-mixer's own query (`sc2`) | road, unconditional ("no gains, no learnable rate" — comment at `:4898-4903`) | identical |
| Q-side rotation, attention mixer | `:5095-5112` (`POLAR_QROT>=1`, `_mx_q`) | the typed-mail mixer's query | road, unconditional; **K stays unrotated** both places (the "v109pi relative-phase" precedent, so attention is a function of phase *differences* only) | identical |
| E&B coupling | `apply_polar_sink.py`'s `_polar_em`, called `:5386` area | clock-plane pairs, along slot-mask lanes | **gain**, but `POLAR_EM` is a **fixed float** read from env (`ALG_POLAR_EM=0.1` in FAM), not a trained parameter — so it is a fixed gain, not a learnable one | identical |
| Canon frame (`_clock_frame`) | def `:745-755`; call sites `:4197-4198` (notebook read, `kb-1,+1`), `:4307-4308` (garage read, `kb-1,+1`), `:4494-4495`/`:5427-5428`/`:5441-5442` (notebook ink / garage write, `kb,-1`) | any cross-breath shelf (notebook, garage) | road, gated `if ALG_CLOCK_CANON and ALG_POLAR` | identical — same function call in training's `forward()` and in `loop_val.py`'s read-time call into the same `forward()`/`breath_step()` |
| Six-wave (static) | `:54` (door), `:5658-5676` (the term) | the grounding bank-read bias only (breath-independent) | **learnable gain** `p["sw_g"]` (`:2561-2562`), trained value 0.13 (ledger 09-24 17:30) | identical, but see §4 — it is NOT a per-breath clock signal at all |
| Six-wave tick (built, not in FAM) | `:63-64` (door, asserts `ALG_SIXWAVE` set), `:5677-5682` (`_mk_swtick`) | would advance the wave's phase `(kb-1)*60deg` in lockstep with the hand | road when armed, but **off by default** — `ALG_SW_TICK` is not in any FAM string | N/A — experimental arm only (`swtick_chain.sh`), read NULL-ish per the ledger (see §4) |

### (b) Through the RAW BREATH INDEX `kb` (schedules, not clock content)

| Organ | file:line | In FAM/PMS8 chassis? |
|---|---|---|
| Tree descent levels `_TREE_LEVELS[kb]` | `:3242-3246` (def), `:3886`,`:3907` (reads) | NO — `ALG_TREE` unset in `family_chain.sh`/`pms8_chain.sh`; only in `tree_chain.sh`/`tree2_chain.sh`/`treecode_chain.sh` arms |
| Three consults (`stop_after=2/4`, `facts3`/`facts5` from breath 3/5) | `:246-272` (doors), `:5881-5896` (loop) | NO — `ALG_ALT3` unset in FAM; only in `alt3_chain.sh`/`cert2_chain.sh`/`certloop_chain.sh` |
| Certifier lines `cert3`/`cert5` (`kb>=3`/`kb>=5`) | `:4835-4844` | NO — rides `ALG_CERT`/`ALG_ALT3`, same arms as above |
| Form-2 fresh read `ALG_CERT2` (`kb in (3,5)`) | `:4861` | NO |
| Hierarchical freeze `_HIER_DAMP[kb]` | `:3299-3300` (def), `:5552-5567` (apply) | NO — only `hier_chain.sh` (`HS_241`, NULL, see §5) |
| Plane unlock `_UNLOCK_K[kb-1]` | `:321-323` (def), `:363` (`_unlock_planes`), `:5425` (call, unconditional — no-op unless `ALG_UNLOCK` set) | NO — only `unlock_chain.sh` |
| Matryoshka readout `_MATRY_K` | `:335-336` | NO — only `matry_chain.sh` |
| Stellarator handoff weight `cos²(kb·π/(2(K_B-1)))` | `:4277` | **YES** — `ALG_STELLAR=2` is in FAM (`family_chain.sh:6`); this is a hand-written per-breath cosine schedule, keyed on raw `kb` and `K_B`, entirely independent of `rotor_clock`'s tables |
| Idle breaths `kb in _IDLE` | `:742` (def, env `ALG_IDLE_BREATHS`), `:4278`,`:5315` (use) | unset by default (empty frozenset) — read-only escape hatch |
| Consult/tree `stop_after` partial passes at read | `loop_val.py`/`chain_acc.py` partial-pass ports (same `forward(..., stop_after=...)` signature) | same code path as training — see 1(a) train==read column |

### (c) Through LEARNED per-breath parameters

| Param | file:line | Shape/indexing | Note |
|---|---|---|---|
| `p["breath_emb"]` | born `:2383`/`:2385`; read `:4181`,`:4287`,`:4291`,`:4325` | `(K_B, H_W)`, read at `[kb]` for `kb` in `range(1,K_B)` only | **row 0 is never read** — see §2 |
| `p["breath_gate"]` | born `:2386` | `(K_B,)` | init `-2.0` (closed); not traced further in this audit — flagged for a follow-up read |
| `p["sw_g"]` | born `:2561-2562` | scalar, not per-breath | the six-wave carrier gate — see §4 |
| `_treecode_table` per-level phasor codes | `:3252` area | fixed (seeded FHRR), not learned, not per-breath (per-*level*) | out of scope for FAM (ALG_TREE unset) |

**Reading for the inventory**: of everything genuinely live in the deployed research chassis (FAM/PMS8), only three mechanisms touch breath index at all: the sextet clock (state turn + Q-rotation + E&B, all through `rotor_clock`'s one table), the six-wave's static bias (no breath term — see §4), and the stellarator's hand-written `cos²(kb·...)` schedule (§5). Every other `kb`-schedule inventoried above (tree, consults, certifier lines, hier-damp, unlock, matryoshka) is an **experimental arm, not in the trained body** — this matters for Q5 below: the "conductor gap" is mostly hypothetical against the *shipped* chassis today, real against the roadmap's next builds.

---

## 2. CONSISTENCY — breath numbering

VERIFIED consistent:
- `breaths = [fst]` (`:5719`) then `for kb in range(1, _kb_stop)` appends one `cur` per loop breath (`:5881-5882`) → `breaths[0]` = grounding read (outside time), `breaths[1..K_B-1]` = clocked. `K_B = len(o["breaths"])` is asserted implicitly used downstream at `:6362`.
- `rotor_clock.wheel_table()`'s `t = k - 1` (`:72` in `rotor_clock.py`) and the head's `_pdc[kb-1]`/`_pac[kb-1]` everywhere (`:5385`, `:4909`, `:5112`) agree: loop breath `kb` (1-indexed) reads row `kb-1` (0-indexed) of the 6-row table. No off-by-one found in the clock-table indexing itself.
- `_TREE_LEVELS[kb]` (direct index, breath 0 = index 0) is a **different indexing convention** from the clock tables (breath `kb` = index `kb-1`) — not a bug (both are internally consistent and documented: `:3221-3222` "level per breath 0..K_B-1", asserted `_TREE_LEVELS[0]==3` at `:3246`), but a reader of this codebase must not assume one indexing rule for all `[kb]`-shaped lists.

ONE GENUINE OFF-BY-ONE FOUND (minor, inert today):
- `p["breath_emb"]` is born with shape `(K_B, H_W)` and its phase formula at birth uses `_bk = np.arange(K_B)` (`:2365`, 0..K_B-1) — i.e. row 0 is given a real phase at init — but every read site (`:4181`, `:4287`, `:4291`, `:4325`) indexes `p["breath_emb"][kb]` only for `kb` in `range(1, K_B)`. **Row 0 of `breath_emb` is a trained parameter that is never read by anything.** Harmless (wasted gradient/row, not a correctness bug), but worth a one-line fix (either skip row 0 at birth or start the loop's read at `kb-1`) next time the head is touched.

NOT A BUG, but UNGUARDED (latent risk, INFERRED): nothing in the file asserts `len(_TREE_LEVELS) == K_B`, `len(_UNLOCK_K) == K_B-1`, or `len(_MATRY_K) == K_B-1` against the live `ALG_BREATH`. Today `ALG_BREATH=7` is hard-pinned in every FAM string and `rotor_clock.N_LOOP=6` is hard-coded, so `kb` never exceeds 6 in practice and the `else 3` / `else default` fallbacks (`:3886`,`:3907`) never actually trigger silently-wrong behavior. But if `ALG_BREATH` were ever changed without updating these comma-lists, breaths beyond the list length would silently fall back to defaults (tree: leaf; unlock/matry: no-op) with no error. Recommend an assert at parse time (`_TREE_LEVELS is None or len(_TREE_LEVELS) == K_B`, etc.) — currently there is only the clock-side guard `1 <= kb <= _RC_N_LOOP` (`:4895`,`:5095`,`:5382`,`:5552`,`:5566`), which protects the clock itself but not the raw-`kb` schedules.

Readers' partial passes (`stop_after`): `_kb_stop = min(K_B, stop_after+1) if stop_after else K_B` (`:5881`) is the SAME expression used by training's own consult capture (`_consult_into`, `:9187-9188`) and by any reader passing `stop_after` — one definition, not duplicated. VERIFIED identical.

---

## 3. TWO CLOCKS — the compounding assertion and the canon-frame direction

**The compounding assertion holds and is checked at runtime, not just in prose.** `_polar_tables()` (`:1007-1016`) builds the delta table `_dlt` from `rotor_clock.wheel_table()`'s absolute table `_abs`, then asserts
```
np.cos(np.cumsum(_dlt,0)) == np.cos(_abs)  and  np.sin(...) == np.sin(_abs)   (atol=1e-6)
```
i.e. the STATE's per-breath increment table and the Q-SIDE's absolute-phase table are proven, at import/construction time, to be two views of the *same* six numbers from `rotor_clock.wheel_table()` — both indexed by the same `kb-1` row. Both `dc/ds` (state increments, compounding) and `ac/as` (Q-side absolute, rebuilt fresh each breath) are returned from the one `_polar_tables()` call (`:972-1030`), so there is structurally only one table, read two ways — VERIFIED, not inferred.

**Canon-frame direction**, confirmed by reading both write and read call sites:
- Writes (notebook ink `:5427-5428`, garage drop `:5441-5442`): `_clock_frame(cur, kb, -1, _rot2)` — subtracts the CURRENT breath's absolute phase (`ac[kb-1]`/`as[kb-1]`) from the just-turned state, i.e. stores content in the phase-0 (canonical) frame.
- Reads (notebook shelf-read `:4197-4198`, garage shelf-read `:4307-4308`): `_clock_frame(_rd, kb-1, +1, _rot2)` — adds back the PREVIOUS breath's absolute phase. This is correct, not an off-by-one: at the point in `breath_step` where the shelf is read, `cur` (this breath's input) still carries the phase set by the *previous* breath's turn (`kb-1`), because this breath's own turn (`:5381-5385`) has not run yet. Re-phasing the canonical content to `kb-1` makes it addable to `cur` in the frame `cur` is actually in at that point. VERIFIED self-consistent.

**ALG_CLOCK_CANON in the family env chains**: confirmed `ALG_CLOCK_CANON=1` present in every FAM string checked — `family_chain.sh:6`, `pms8_chain.sh:6`, `pms9_chain.sh:6`, `pm35_241_chain.sh:7`, `clock_probe_chain.sh:5`, `clock_reads_chain.sh:22`, `swtick_chain.sh:10`, `ctrls242_chain.sh:5`. It is always paired with `ALG_POLAR=1` in these strings, which matters because of the next finding.

**GAP FOUND (minor)**: every use site gates on `if ALG_CLOCK_CANON and ALG_POLAR` (`:4197`,`:4307`,`:4495`,`:5428`,`:5442`) but there is no `assert` tying the two flags together the way `ALG_SW_TICK` is tied to `ALG_SIXWAVE` (`:64`) or `ALG_UNLOCK` to `ALG_POLAR` (`:323`). Setting `ALG_CLOCK_CANON=1` with `ALG_POLAR=0` is silently accepted and silently does nothing. Not triggered in any shipped chain (both always set together), but it is an unguarded combination — INFERRED risk, not an observed failure.

---

## 4. WHAT DOES NOT TICK

**Six-wave — confirmed static.** `:5658-5676`: `_th` (token phase) comes from `sent mod 6`, `_phi` (slot phase) from `slot index mod 6` — both fixed at construction, no `kb` term anywhere in the base `_sw_term`. The ledger's own audit (`docs/phase1_skeleton_spec.md`, 2026-09-24 17:30, "THE AUDIT… THE TWO CLOCKS ARE NOT IN SYNC BECAUSE ONE DOES NOT TICK") independently reaches the same conclusion and records the trained gate value: `sw_g` trained to 0.13 on PMS8_241/242. The tick fix (`ALG_SW_TICK`, `:5677-5682`) exists, is gated bit-identical when unset, and was fired once as arm `SW_241` (`swtick_chain.sh`) — **not folded into the FAM chassis**, i.e. still not ticking in the body currently called "the research lineage."

**`breath_emb` — learned, not a clock.** Confirmed: it is a free `(K_B,H_W)` parameter with no tie to `rotor_clock`'s angles (its own sinusoidal *initialization* at `:2378-2382` is a convenience, not a constraint — gradient is free to move every row away from that init). It is additive into `q_extra` alongside the clock-turned `cur`, so it is a second, independent, per-breath-indexed, *learned* signal living beside the one true clock. This is exactly the shape the mandatory-road law would flag if `breath_emb` ever grew a gain — it currently doesn't have one (it's added unconditionally, not behind a scalar), so it is a road, not a gain, but it is NOT the clock and nothing forces its content to respect the clock's structure.

**Stellarator's `cos²(kb·π/(2(K_B-1)))` — its own private schedule.** (`:4277`) This is a hand-derived cosine taper, correctly shaped to be 1 at breath 1 and 0 at the last loop breath, but computed directly from `kb` and `K_B`, not from `rotor_clock.phase_of`/`wheel_table`. It happens to be *compatible* with "one tempo" in spirit (deterministic, monotonic, no learnable rate) but it is a second, independently-authored clock-like function, not a read of the one table. It is live in FAM (`ALG_STELLAR=2`).

**Nothing else found ticking outside the clock in the live FAM chassis.** Tree/consults/certifier-lines/hier-damp/unlock/matryoshka are all `kb`-scheduled but are experimental arms not in FAM/PMS8 (confirmed absent from every FAM string grepped).

---

## 5. THE CONDUCTOR GAP

Ledger's own framing (2026-10-04 12:38): *"the breath index already conducts the tree schedule and the consults; promotion = one tempo for the levels, the temperature, the per-band damping, the consults and the U-Net's looks, entering as a FIXED MANDATORY signal (the tick's 0.13 gate is the counter-example)."* Reading that against the code:

1. **Schedules that read `kb` directly** (tree levels, consult breaths, certifier lines, hier-damp, unlock, matryoshka) — Bryce's own argument ("kb is the clock's own index") is defensible as far as it goes: `kb` literally comes from `rotor_clock.is_clocked`'s domain (1..6), so reading `kb` is reading *an* authoritative fact about the clock. But it is reading the clock's **index**, not its **content** — none of these schedules consult `phase_of(kb)` or the band tables, so a change to the clock's *shape* (e.g. a future pass-wheel engaging, or a different quantum) would not propagate to them automatically. They satisfy "one tempo" only in the weak sense of "one counter"; they do not satisfy it in the sense the roadmap entry asks for ("one tempo… entering as a FIXED MANDATORY signal") because each organ re-derives its own threshold logic (`kb>=3`, `kb in (3,5)`, `kb-1 < len(...)`) independently, with no shared table.
2. **Organs that read the clock's actual content** — the state turn, the two Q-rotations, the E&B field, the canon frame. These are the only organs that are unambiguously "one tempo, one source" today: all four pull from the single `_polar_tables()` cache, which is itself built once from `rotor_clock.wheel_table()` (VERIFIED, §0/§3).
3. **Organs with their own per-breath parameter, outside the clock's table** — `breath_emb` (learned) and the stellarator's `cos²` taper (fixed-but-bespoke). These are the clearest violations of "one tempo": they are both per-breath and both independent of `rotor_clock`.

**Minimal change recommended** (file:line specific): add one function, e.g. `mycelium/rotor_clock.py::conductor(kb)` (or a thin wrapper in the head, since the schedules are head-specific, e.g. `_conductor(kb)` near `:742` beside `_IDLE`), returning a small namedtuple/dict — `{open_levels, damp_lambda_by_band, consult_flag, mask_temperature, unet_look}` — all derived from `phase_of(kb)` or at minimum from one shared `kb`-indexed table built alongside `_polar_tables()`. Then:
   - Replace the independent literals `kb>=3`/`kb>=5` (`:4836`,`:4841`), `kb in (3,5)` (`:4861`), `_HIER_DAMP[kb]`-style settle breaths (`:3299`), `_UNLOCK_K[kb-1]` (`:373`), `_TREE_LEVELS[kb]` (`:3886`,`:3907`) with reads of this one structure.
   - Fold the stellarator's `cos²(kb·π/(2(K_B-1)))` (`:4277`) into the same table (it is already clock-compatible in shape — just author it there instead of inline).
   - `breath_emb` is the one item that resists this cleanly: it is *learned content*, not a schedule, so "read one table" doesn't apply to it directly. The mandatory-road law's relevant question is narrower: does `breath_emb` have a *learnable gain* gating it? **No** — it enters unconditionally added, not behind a scalar multiply — so it does not trip the law's "gated organs are voted down from birth" clause. But it is the one place where the design intent ("the breath index conducts… as a FIXED MANDATORY signal") and the implementation (a free-floating learned embedding that may drift away from any rotational structure) are least aligned. Worth a registered question, not a law violation.
   - The organ the mandatory-road law WOULD refuse outright if it existed is the six-wave's `sw_g` (`:2561-2562`) — a *learnable gain* in front of a structural signal, trained to 0.13, i.e. the canonical "gated organ voted down from birth" the law describes (and the ledger says exactly this: "the tick's 0.13 gate is the counter-example," 2026-10-04 12:38). Confirms the law's prediction rather than contradicting it.

---

## 6. RISKS — readers and the clock's turn

- **`clock_band_probe.py` deliberately does NOT undo the turn** — confirmed by its own docstring (`scripts/clock_band_probe.py:1-20`): it banks "the state ENTERING each loop breath kb=1..6" in its native (rotated) frame, because the whole point of probe A ("breath <- band") is to measure whether the rotation is legible. This is correct by design, not a bug — flagging only so a future reader doesn't "fix" it.
- **`membrane_scale.py` reads attention weights, not raw clock-plane state** (`grep` found only an `ALG_CLOCK_CANON: "1"` entry in its env-config dict, `:121` — it runs the model under the same training env, it does not itself call `_clock_frame`). Attention score/weight tensors are not clock-rotated objects (the rotation is on Q/K *before* the softmax, not on the resulting distribution), so no frame bug is possible here. VERIFIED by absence of any raw-state read in the script, not by a positive check of every line.
- **`scripts/unet/pictures.py` — confirmed, reads the frozen trunk, never the head's state.** Its own docstring (`:5-6`): "builds the pictures, from the CACHED trunk states only (never a trunk call, never scripts/phase1_algebra_head.py's forward()/build_params())." Since the trunk is frozen and never touched by `rotor_clock`, there is no rotation to undo here — this is why the ledger's 2026-10-05 10:38 entry ("THE EYES, specified") explicitly calls out that the *next* U-Net form must read the head's state "in the CANON FRAME (the clock's turn undone, the notebook's trick)" — i.e. the audit's concern is correctly anticipated in the design doc for work not yet built, and is moot for the U-Net work already done (round 1/round 2, CLOSED per ledger 2026-10-05 03:55).
- **`dump_wild_*.pkl` (via `loop_val.py`'s `LV_DUMP`)** — confirmed these dumps hold **decoded per-slot field predictions** (`ftype, op, args, res, digits` — `scripts/loop_val.py:460-461`), i.e. already-read-out categorical/argmax values, not raw `cur` state vectors. The clock-plane frame question does not apply to this artifact: each readout head (`pres/ftype/op/args/res/dig`) consumes `cur` within the single breath it's computed at, is trained end-to-end on whatever frame `cur` is naturally in at that breath, and never crosses a breath boundary itself — unlike the notebook/garage shelves, which explicitly need (and get) `_clock_frame`. No risk found here; flagged as INFERRED-but-well-supported since the heads' weight matrices were not individually re-derived in this audit.
- **One miner path not currently consumed**: `state["u_all"]/state["r_all"]` (the polar direction/radius tap, `scripts/apply_polar_waist.py:355-356`, gated by `ALG_MINE_BREATHS`) stores `_pol_u` **after** that breath's own turn but with no frame normalization across breaths. No current script (`schema_miner.py` was checked — zero references) consumes this tap, so there is no live reader to audit for a missed `_clock_frame` call. Flagged as a latent risk ONLY if/when a cross-breath miner is built on this tap: it would need to canonicalize each breath's `u` before comparing across breaths, exactly as the notebook/garage shelves do.

---

## Summary of findings by status

**VERIFIED** (read code + ran one CPU self-test): the clock's own self-tests pass; the compounding contract between state-increment and Q-side-absolute tables is asserted at runtime and is the same six numbers; canon-frame write/read signs are self-consistent; breath-0/loop-breath indexing is consistent everywhere except one dead parameter row; six-wave is static in the deployed chassis and the tick exists only as an unmerged arm; the stellarator and consult/tree/certifier/hier-damp/unlock/matryoshka schedules are raw-`kb`, not clock-content, reads; `pictures.py` reads the trunk not the state; `dump_wild_*.pkl` holds decoded fields, not raw state.

**INFERRED** (reasoned from code structure, not independently executed): the `ALG_CLOCK_CANON` without `ALG_POLAR` silent-no-op gap; the unguarded list-length-vs-`K_B` assumption in tree/unlock/matryoshka; the `u_all`/`r_all` tap's cross-breath-frame risk if a miner is ever built on it; the practical harmlessness of the dead `breath_emb[0]` row.

**ONE ACTIONABLE BUG** (cosmetic, zero behavioral impact today): `p["breath_emb"]` row 0 is trained but never read (`scripts/phase1_algebra_head.py:2383`/`:4181` etc.) — the init formula includes `kb=0`'s phase but no read site ever indexes `[0]`.

**ONE DESIGN GAP matching Bryce's own framing**: "T6 promoted to conductor" is a 2026-10-04 design decision, not yet a build — today's `kb`-schedules (tree/consults/certifier-lines/hier-damp/unlock/matryoshka) each re-derive their own breath thresholds independently and are, in any case, experimental arms absent from the live FAM/PMS8 chassis; the two organs that are both live AND independent of the one clock table are `breath_emb` (learned, ungated) and the stellarator's bespoke `cos²` taper (fixed, ungated) — neither violates the mandatory-road law (neither has a learnable gain), but neither is "one tempo" either. The six-wave's `sw_g` IS the law's predicted counter-example, already identified as such in the ledger.
