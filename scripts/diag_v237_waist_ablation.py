"""#237 waist-ablation eval (§1A.E.12, pinned before the step-1000 read).

The C4-C5 unification test: per-breath CE on the eval set with the waist
contribution silenced at inference. No retrain.

Three conditions per checkpoint:
  baseline   — model as trained (gate = learned value)
  gate0      — fg_v200_waist_gate forced to -30 (sigmoid ~ 1e-13 ~ 0):
               norm_commit KEPT, directional contribution removed.
               THE BINDING ABLATION per E.12.
  waist_off  — stage2a_waist=False: norm_commit AND contribution removed.
               Reported alongside as the norm/contribution split.

Pre-committed readings (E.12):
  anti-ladder flattens/inverts under gate0  -> C4 and C5 are one thread
    (fixed-direction waist masks per-breath differentiation); #239 promoted
    with measured mechanism (still behind §1A.E.2 wiring checks in order).
  nothing changes -> waist exonerated on C4; branch table proceeds clean.

Uses the #237 driver's own _per_breath_ce_at_eval so the CE computation is
identical to the in-run checkpoint reads. CE eval uses the same val loader
construction as the driver (seed SEED+2, EVAL_BATCH=8, EVAL_BATCHES=8).

Usage:
  DEV=CPU STEPS_LIST=1000 .venv/bin/python scripts/diag_v237_waist_ablation.py
  STEPS_LIST=1000,2000 .venv/bin/python scripts/diag_v237_waist_ablation.py   # GPU after run
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

from mycelium.llama_loader import (
    attach_llama_layers, load_llama_weights,
    LLAMA_3_2_1B_CFG, SMOLLM2_1_7B_CFG,
)
from mycelium.factor_graph_v200 import (
    attach_fg_params_v200, fg_v200_state_dict,
    V200_K_MAX, V200_N_MAX, V200_F_MAX, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_STAGE2A_WAIST, V200_WAIST_DIM,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107
from mycelium.provenance import make_provenance

# Reuse the driver's CE-at-eval implementation verbatim (method identity)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "r237", os.path.join(_SCRIPT_DIR, "v200_resmoke_237.py"))
_r237 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_r237)
_per_breath_ce_at_eval = _r237._per_breath_ce_at_eval

RUN_TAG   = os.environ.get("RUN_TAG", "237")   # "237" (frozen control) or "237_5"
CKPT_STEM = {"237": "v200_perceiver_237_mask1a", "237_5": "v200_perceiver_237_5_substrate", "238": "v200_perceiver_238_write8"}[RUN_TAG]
CKPT_GLOB = f".cache/v200_perceiver_ckpts/{CKPT_STEM}_step*.safetensors"
OUT_DIR   = ".cache/v200_smoke"
SEED      = 42
EVAL_BATCHES = 8


class _Obj:
    pass


def _slope(pb_ce: list, K: int) -> float:
    ys = np.array(pb_ce, dtype=np.float64)
    if np.all(np.isfinite(ys)) and K >= 2:
        return float(np.polyfit(np.arange(K, dtype=np.float64), ys, 1)[0])
    return float("nan")


def main() -> None:
    K = V200_K_MAX
    steps_env = os.environ.get("STEPS_LIST", "1000")
    wanted = {int(s) for s in steps_env.split(",")}
    ckpts = sorted(
        (c for c in glob.glob(CKPT_GLOB)
         if int(re.search(r"step(\d+)\.safetensors", c).group(1)) in wanted),
        key=lambda p: int(re.search(r"step(\d+)\.safetensors", p).group(1)))
    assert ckpts, f"no checkpoints matched {CKPT_GLOB} with STEPS_LIST={steps_env}"
    print(f"[ablate-237] {len(ckpts)} ckpts; DEV={os.environ.get('DEV', '')}")

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    llama32_path = ".cache/llama-3.2-1b-weights/model.safetensors"
    if os.path.exists(llama32_path):
        sd = safe_load(llama32_path)
        cfg = LLAMA_3_2_1B_CFG
    else:
        sd = load_llama_weights()
        cfg = SMOLLM2_1_7B_CFG
    model = _Obj()
    attach_llama_layers(model, n_layers=4, sd=sd, cfg=cfg)
    del sd
    attach_fg_params_v200(
        model, n_latents=32, n_var_lat=V200_N_VAR_LAT, k_max=K,
        n_digits=V200_N_DIGITS, n_max=V200_N_MAX, f_max=V200_F_MAX,
        stage2a_waist=True, waist_dim=V200_WAIST_DIM,   # #237 trained with the waist ON — explicit, not env-inherited (sweep-crash fix)
    )
    for layer in model.llama_layers:
        for p in layer.parameters():
            if p.dtype != dtypes.float:
                p.assign(p.cast(dtypes.float)).realize()

    val_loader = FactorGraphLoaderV107(
        ".cache/factor_graph_test.jsonl", batch_size=8,
        difficulty_filter=None, curriculum=False,
        n_max=V200_N_MAX, f_max=V200_F_MAX, k_max=K, n_heads=16, seed=SEED + 2,
    )

    targets = fg_v200_state_dict(model)
    results = []
    for ck in ckpts:
        step = int(re.search(r"step(\d+)\.safetensors", ck).group(1))
        sd_ck = safe_load(ck)
        for name, dst in targets.items():
            if name in sd_ck:
                src = sd_ck[name].to(dst.device).realize()
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
        del sd_ck

        learned_gate = float(model.fg_v200_waist_gate.cast(dtypes.float)
                             .realize().numpy().item())

        conds = {}
        # baseline — as trained
        pb = _per_breath_ce_at_eval(model, val_loader, K=K, n_max=V200_N_MAX,
                                    f_max=V200_F_MAX, n_var_lat=V200_N_VAR_LAT,
                                    n_digits=V200_N_DIGITS,
                                    max_batches=EVAL_BATCHES,
                                    stage2a_waist=True)
        conds["baseline"] = {"per_breath_ce": pb, "slope": _slope(pb, K)}

        # gate0 — BINDING ablation: contribution silenced, norm_commit kept
        model.fg_v200_waist_gate.assign(
            Tensor(np.full((1,), -30.0, dtype=np.float32),
                   dtype=dtypes.float)).realize()
        pb = _per_breath_ce_at_eval(model, val_loader, K=K, n_max=V200_N_MAX,
                                    f_max=V200_F_MAX, n_var_lat=V200_N_VAR_LAT,
                                    n_digits=V200_N_DIGITS,
                                    max_batches=EVAL_BATCHES,
                                    stage2a_waist=True)
        conds["gate0"] = {"per_breath_ce": pb, "slope": _slope(pb, K)}

        # restore learned gate, then waist_off — norm/contribution split
        model.fg_v200_waist_gate.assign(
            Tensor(np.full((1,), learned_gate, dtype=np.float32),
                   dtype=dtypes.float)).realize()
        pb = _per_breath_ce_at_eval(model, val_loader, K=K, n_max=V200_N_MAX,
                                    f_max=V200_F_MAX, n_var_lat=V200_N_VAR_LAT,
                                    n_digits=V200_N_DIGITS,
                                    max_batches=EVAL_BATCHES,
                                    stage2a_waist=False)
        conds["waist_off"] = {"per_breath_ce": pb, "slope": _slope(pb, K)}

        # carrier_proj — §1A.E.13 DIRECT BLACKBOARD TEST: project the shared
        # carrier dims (from specialization_237.json, step-matched) out of z
        # at init + every breath boundary. gate-0 tests the waist's
        # contribution; THIS tests the carrier itself (latent_init common
        # mode, waist co-aligned-not-created). Pre-committed reading:
        # early-breath CE degrades or the U collapses -> carrier LOAD-BEARING
        # -> blackboard supported; unchanged -> carrier decorative ->
        # blackboard weakened, #238 path cleaner.
        spec_path = os.path.join(OUT_DIR, f"specialization_{RUN_TAG}.json")
        carrier_dims = None
        if os.path.exists(spec_path):
            with open(spec_path) as f_sp:
                spec = json.load(f_sp)
            for e in spec.get("checkpoints", []):
                if e["step"] == step:
                    carrier_dims = e.get("top10_overlap", {}).get(
                        "modal_shared_top10_dims_final_breath")
                    break
        if carrier_dims:
            mask_np_c = np.ones(2048, dtype=np.float32)
            mask_np_c[np.array(carrier_dims, dtype=np.int64)] = 0.0
            cmask = Tensor(mask_np_c, dtype=dtypes.float).contiguous().realize()
            # E.13 four-cell design: the projection SITE embeds which blackboard
            # reading is tested — boundary = cross-breath MEMORY; post_think =
            # within-breath BUS; both = both. DECORATIVE may only be declared
            # from the full set.
            for cond_key, site in (("carrier_proj", "boundary"),
                                   ("carrier_proj_bus", "post_think"),
                                   ("carrier_proj_both", "both")):
                pb = _per_breath_ce_at_eval(model, val_loader, K=K, n_max=V200_N_MAX,
                                            f_max=V200_F_MAX, n_var_lat=V200_N_VAR_LAT,
                                            n_digits=V200_N_DIGITS,
                                            max_batches=EVAL_BATCHES,
                                            stage2a_waist=True,
                                            carrier_dim_mask=cmask,
                                            carrier_mask_site=site)
                conds[cond_key] = {"per_breath_ce": pb, "slope": _slope(pb, K),
                                   "site": site, "carrier_dims": carrier_dims}
        else:
            for cond_key in ("carrier_proj", "carrier_proj_bus", "carrier_proj_both"):
                conds[cond_key] = {
                    "skipped": ("no modal_shared_top10_dims_final_breath for this step in "
                                "specialization_237.json — run diag_v237_specialization first"),
                }

        base_s, gate0_s = conds["baseline"]["slope"], conds["gate0"]["slope"]
        # E.12 pre-committed reading: anti-ladder flattens (slope moves toward
        # 0 from positive) or inverts (goes negative) under gate0
        if np.isfinite(base_s) and np.isfinite(gate0_s):
            if gate0_s <= -0.05:
                reading = "INVERTS — C4+C5 ONE THREAD (waist masks differentiation); #239 promoted w/ mechanism"
            elif base_s > 0 and gate0_s < 0.5 * base_s:
                reading = "FLATTENS — C4+C5 ONE THREAD (waist masks differentiation); #239 promoted w/ mechanism"
            else:
                reading = "UNCHANGED — waist exonerated on C4; branch table proceeds clean"
        else:
            reading = "INDETERMINATE (non-finite slope)"

        # §1A.E.13 four-cell carrier read. Per-condition HURT criterion
        # (pre-committed, uniform across sites): mean per-breath dCE > +0.01
        # nats OR early-breath (k=0..2) dCE > +0.01 OR U-range halved.
        # Cells: both-hurt -> MEMORY+BUS; boundary-only -> MEMORY;
        # post_think-only -> BUS; neither -> DECORATIVE (declared from a test
        # that covered the hypothesis space; only valid with all conditions run).
        carrier_reading = "SKIPPED"
        if "per_breath_ce" in conds.get("carrier_proj", {}):
            base_ce = np.array(conds["baseline"]["per_breath_ce"])

            def _hurt(cond_key: str) -> tuple:
                ce = np.array(conds[cond_key]["per_breath_ce"])
                mean_d = float(ce.mean() - base_ce.mean())
                early_d = float(ce[:3].mean() - base_ce[:3].mean())
                u_collapsed = bool(np.ptp(ce) < 0.5 * np.ptp(base_ce))
                return (mean_d > 0.01 or early_d > 0.01 or u_collapsed,
                        mean_d, early_d, int(ce.argmin()))

            mem_hurt, mem_md, mem_ed, mem_umin = _hurt("carrier_proj")
            bus_hurt, bus_md, bus_ed, bus_umin = _hurt("carrier_proj_bus")
            both_hurt, both_md, both_ed, both_umin = _hurt("carrier_proj_both")
            umin_base = int(base_ce.argmin())
            detail = (f"memory(dCE={mem_md:+.4f}/early{mem_ed:+.4f}/min k{umin_base}->{mem_umin}) "
                      f"bus(dCE={bus_md:+.4f}/early{bus_ed:+.4f}/min k{umin_base}->{bus_umin}) "
                      f"both(dCE={both_md:+.4f})")
            if mem_hurt and bus_hurt:
                carrier_reading = f"MEMORY+BUS — carrier is both; blackboard SUPPORTED strongly. {detail}"
            elif mem_hurt:
                carrier_reading = f"MEMORY — cross-breath persistence load-bearing; blackboard (workspace reading) SUPPORTED. {detail}"
            elif bus_hurt:
                carrier_reading = f"BUS — within-breath broadcast load-bearing; blackboard (bus reading) SUPPORTED. {detail}"
            else:
                carrier_reading = (f"DECORATIVE (full-scope: both persistence and bus tested) — "
                                   f"blackboard weakened, #238 path cleaner. {detail}")

        entry = {"step": step, "ckpt": ck, "learned_gate": learned_gate,
                 "learned_gate_sigmoid": float(1.0 / (1.0 + np.exp(-learned_gate))),
                 "conditions": conds, "e12_reading": reading,
                 "e13_carrier_reading": carrier_reading}
        results.append(entry)
        print(f"[step {step:5d}] gate={learned_gate:+.3f} "
              f"| slope base={base_s:+.5f} gate0={gate0_s:+.5f} "
              f"waist_off={conds['waist_off']['slope']:+.5f} | {reading}",
              flush=True)
        print(f"    carrier: {carrier_reading}")
        for cond_name, c in conds.items():
            if "per_breath_ce" in c:
                print(f"    {cond_name:12s} ce={['%.4f' % v for v in c['per_breath_ce']]}")
            else:
                print(f"    {cond_name:12s} {c.get('skipped', '(no data)')}")

    # Cross-step dim-identity check: the sweep's conclusions hang on the
    # carrier dims being the same object across checkpoints. 9.99/10 stability
    # says they are; verify rather than assume (overlap >= 8/10), downgrade
    # loudly instead of crashing — drift would itself be a finding.
    dims_by_step = {e["step"]: set(e["conditions"]["carrier_proj"].get("carrier_dims") or [])
                    for e in results if "carrier_dims" in e["conditions"].get("carrier_proj", {})}
    dims_drifted = False
    steps_sorted = sorted(dims_by_step)
    for a, b in zip(steps_sorted, steps_sorted[1:]):
        ov = len(dims_by_step[a] & dims_by_step[b])
        if ov < 8:
            dims_drifted = True
            print(f"WARNING: carrier dims DRIFTED between step {a} and {b} "
                  f"(overlap {ov}/10) — cross-step carrier conclusions are "
                  f"scoped per-step, and the drift is itself a finding")

    out_path = os.path.join(OUT_DIR, f"waist_ablation_{RUN_TAG}.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_id": RUN_TAG,
            "measured_config": {"stage2a_waist_baseline": True, "K": K,
                                "n_latents": 32, "topology_mask": "1a"},
            "e12_pre_commit": ("flattens/inverts under gate0 -> C4+C5 one thread, "
                               "#239 promoted w/ mechanism (behind §1A.E.2 checks); "
                               "unchanged -> waist exonerated; flatten threshold: "
                               "gate0 slope < 0.5x baseline positive slope"),
            "e13_carrier_four_cell": ("hurt = mean dCE>+0.01 | early dCE>+0.01 | "
                                      "U-range halved; boundary=memory, post_think=bus, "
                                      "DECORATIVE only from full set"),
            "carrier_dims_drifted_across_steps": dims_drifted,
            "checkpoints": results,
        }, f, indent=2)
    try:
        sha8 = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_PROJECT_ROOT, timeout=5).stdout.strip()[:8]
        if subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                          text=True, cwd=_PROJECT_ROOT, timeout=5).stdout.strip():
            sha8 += "-dirty"
        msha = subprocess.run(
            ["git", "hash-object", os.path.join(_PROJECT_ROOT, "mycelium", "factor_graph_v200.py")],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5).stdout.strip()
    except Exception:
        sha8, msha = "unknown", "unknown"
    prov = make_provenance(
        metric="waist_ablation_per_breath_ce_e12", units="CE (nats) / slope",
        shape=[len(results), 3], ckpt="dense-ckpt-sweep",
        split="smoke-eval", seed=SEED, step=results[-1]["step"],
        env_vars={"STEPS_LIST": steps_env, "EVAL_BATCHES": str(EVAL_BATCHES),
                  "DEV": os.environ.get("DEV", ""),
                  "measured_stage2a_waist_baseline": "True"},
        output_path=os.path.abspath(out_path), key="checkpoints",
        arch_version=f"v200-{sha8}-K8_L32_prenorm5_seamthree_gate-2_mask1a",
        metric_sha=msha,
    )
    with open(out_path.replace(".json", ".provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
