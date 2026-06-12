"""Provenance metadata helpers for Mycelium v200 (per docs/v200_brief.md §6).

Every persisted artifact carries a four-axis provenance dict, written next to
the data file as `<artifact_stem>.provenance.json`.  An artifact without a
provenance dict is not a v200 artifact.

Four-axis schema (locked):
  "what":      metric, units, shape, head_group
  "where":     file path, key inside file
  "when":      timestamp_iso, git_sha, config_diff, step
  "with_what": ckpt, split, seed, env (tinygrad_sha, device, env_vars)

Usage:
  from mycelium.provenance import make_provenance, write_with_provenance
  import numpy as np

  prov = make_provenance(
      metric="latent_z_per_breath",
      units="bf16",
      shape=[8, 2, 32, 2048],  # [K, B_sample, L, H]
      ckpt="cold-start",
      split="smoke",
      seed=42,
      step=0,
      env_vars={"K_MAX": "8", "FIXED_LEN": "24"},
  )
  write_with_provenance(data_array, "/abs/path/latent_z.npz", prov)
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_git_sha(project_root: Optional[str] = None) -> str:
    """Return the HEAD git SHA of the project (or tinygrad) repo."""
    if project_root is None:
        # Walk up from this file's location to find the repo root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=project_root, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_tinygrad_sha() -> str:
    """Return the HEAD git SHA of the tinygrad installation."""
    try:
        import tinygrad
        tg_dir = os.path.dirname(os.path.dirname(tinygrad.__file__))
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=tg_dir, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "editable-no-git"
    except Exception:
        return "unknown"


def _get_device() -> str:
    """Return a human-readable device descriptor."""
    try:
        from tinygrad import Device
        return f"AM driver/AMD 7900 XTX ({Device.DEFAULT})"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_provenance(
    metric: str,
    units: str,
    shape: list,
    ckpt: str,
    split: str,
    seed: int,
    step: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None,
    head_group: Optional[str] = None,
    config_diff: Optional[str] = None,
    output_path: Optional[str] = None,
    key: Optional[str] = None,
    project_root: Optional[str] = None,
    arch_version: Optional[str] = None,
    metric_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the four-axis provenance dict for a v200 artifact.

    Args:
      metric:      Human-readable name of the metric/data type.
                   E.g. "latent_z_per_breath", "attn_jsd_mean".
      units:       Units / encoding. E.g. "bf16", "nats", "raw".
      shape:       List of ints or descriptive string for shape.
      ckpt:        Checkpoint identifier. E.g. "cold-start", "v200_step500".
      split:       Data split. E.g. "train", "val", "smoke".
      seed:        Random seed used in the run.
      step:        Training step this artifact was produced at (None = unknown).
      env_vars:    Dict of relevant env vars (K_MAX, FIXED_LEN, BATCH, etc.).
      head_group:  Attention head group identifier, if applicable.
      config_diff: Textual description of how this run differs from the base
                   config. None = no diff (base config).
      output_path: Absolute path where the artifact will be written.
                   Used to fill "where.file". May be None at construction time.
      key:         Key inside the output file (e.g. for npz arrays). None = N/A.
      project_root: Repo root for git SHA lookup. Auto-detected if None.

    Returns:
      Dict matching the four-axis schema in docs/v200_brief.md §6.

    Raises:
      ValueError if a required field would produce a silent misalignment
      (e.g. empty metric string, non-absolute output_path).
    """
    if not metric:
        raise ValueError("provenance: metric must be non-empty")
    if output_path is not None and not os.path.isabs(output_path):
        raise ValueError(
            f"provenance: output_path must be absolute, got: {output_path!r}"
        )

    return {
        "what": {
            "metric":     metric,
            "units":      units,
            "shape":      shape,
            "head_group": head_group,
            "metric_sha": metric_sha,   # §7 method identity (git SHA of metric function)
        },
        "where": {
            "file": output_path,
            "key":  key,
        },
        "when": {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "git_sha":       _get_git_sha(project_root),
            "config_diff":   config_diff,
            "step":          step,
        },
        "with_what": {
            "ckpt":         ckpt,
            "split":        split,
            "seed":         seed,
            "arch_version": arch_version,  # §6 required field (added Jun 11)
            "env": {
                "tinygrad_sha": _get_tinygrad_sha(),
                "device":       _get_device(),
                "env_vars":     env_vars or {},
            },
        },
    }


def write_with_provenance(
    data: Any,
    output_path: str,
    provenance_dict: Dict[str, Any],
) -> str:
    """Write data to output_path and a sidecar provenance JSON.

    The provenance sidecar is written to `<stem>.provenance.json` in the
    same directory as output_path.

    Supports:
      - np.ndarray / dict-of-arrays (written as .npz via numpy.savez)
      - str / bytes (written as-is)
      - anything else with a .numpy() method (converted then written as npz)
      - dict without arrays (written as JSON)

    The "where.file" field in provenance_dict is filled in with output_path
    if it is currently None.

    Returns the path to the provenance JSON sidecar.
    """
    import numpy as np

    if not os.path.isabs(output_path):
        raise ValueError(f"write_with_provenance: output_path must be absolute, got {output_path!r}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write data
    if isinstance(data, np.ndarray):
        np.savez(output_path, data=data)
    elif isinstance(data, dict) and any(isinstance(v, np.ndarray) for v in data.values()):
        np.savez(output_path, **data)
    elif isinstance(data, (str, bytes)):
        mode = "w" if isinstance(data, str) else "wb"
        with open(output_path, mode) as f:
            f.write(data)
    elif hasattr(data, "numpy"):
        np.savez(output_path, data=data.numpy())
    elif isinstance(data, dict):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        # Fallback: repr
        with open(output_path, "w") as f:
            f.write(repr(data))

    # Fill in where.file if not set
    prov = dict(provenance_dict)
    if prov.get("where", {}).get("file") is None:
        prov["where"] = dict(prov.get("where", {}))
        prov["where"]["file"] = output_path

    # Write provenance sidecar
    stem = Path(output_path).stem
    # Handle multi-suffix stems (e.g. latent_z.npz.provenance.json)
    sidecar_path = str(Path(output_path).parent / f"{stem}.provenance.json")
    with open(sidecar_path, "w") as f:
        json.dump(prov, f, indent=2)

    return sidecar_path


def validate_provenance(prov: Dict[str, Any]) -> None:
    """Raise ValueError if provenance dict is missing required fields.

    Called by plotting/comparison scripts to fail loudly on mismatched
    provenance axes, per brief §6 discipline.
    """
    required_top = ["what", "where", "when", "with_what"]
    for k in required_top:
        if k not in prov:
            raise ValueError(f"provenance missing top-level key: {k!r}")

    what_keys = ["metric", "units", "shape", "head_group"]
    for k in what_keys:
        if k not in prov["what"]:
            raise ValueError(f"provenance['what'] missing key: {k!r}")

    when_keys = ["timestamp_iso", "git_sha", "config_diff", "step"]
    for k in when_keys:
        if k not in prov["when"]:
            raise ValueError(f"provenance['when'] missing key: {k!r}")

    ww_keys = ["ckpt", "split", "seed", "env"]
    for k in ww_keys:
        if k not in prov["with_what"]:
            raise ValueError(f"provenance['with_what'] missing key: {k!r}")

    env_keys = ["tinygrad_sha", "device", "env_vars"]
    for k in env_keys:
        if k not in prov["with_what"]["env"]:
            raise ValueError(f"provenance['with_what']['env'] missing key: {k!r}")


def assert_axes_match(prov_a: Dict[str, Any], prov_b: Dict[str, Any],
                      axes: Optional[list] = None) -> None:
    """Raise ValueError if two provenance dicts have mismatched with_what axes.

    Default axes checked: ["ckpt", "split", "seed"].
    Pass axes=[] to skip (useful when comparing across splits intentionally).
    """
    if axes is None:
        axes = ["ckpt", "split", "seed"]

    ww_a = prov_a.get("with_what", {})
    ww_b = prov_b.get("with_what", {})

    mismatches = []
    for ax in axes:
        va = ww_a.get(ax)
        vb = ww_b.get(ax)
        if va != vb:
            mismatches.append(f"{ax}: {va!r} vs {vb!r}")

    if mismatches:
        raise ValueError(
            "provenance axis mismatch (cross-ckpt/split/seed comparison silently "
            f"blocked): {'; '.join(mismatches)}"
        )
