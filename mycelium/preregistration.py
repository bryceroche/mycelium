"""preregistration.py — THE PRE-REGISTRATION DOOR (#159, built
2026-08-06). The campaign's most load-bearing prose law becomes a
lane: bars pin BEFORE measurement, mechanically. The door enforces
ORDERING, never blindness (the fence on the fence, stated at
registration) — the honest path becomes the default; the dishonest
one requires an act. Confession-list failures were omissions, not
acts; this covers the measured failure mode entirely.

Usage (registration side, BEFORE the read exists):
    from mycelium.preregistration import register
    register("my_read", bars={"primary": "AUC >= 0.70", ...})
Usage (read side, first line before any verdict prints):
    from mycelium.preregistration import require
    require("my_read", artifact=".cache/my_read.json")  # artifact optional
The read REFUSES (hard error) unless a registration exists and
predates the artifact's mtime (when the artifact already exists —
the re-read case) or now (the fresh case)."""
import os, json, time, subprocess

REG_DIR = ".cache/registrations"

def register(name, bars, notes=""):
    os.makedirs(REG_DIR, exist_ok=True)
    path = os.path.join(REG_DIR, f"{name}.json")
    if os.path.exists(path):
        raise RuntimeError(f"registration '{name}' already exists — "
                           f"bars never bend after pinning; register a new name")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()[:12]
    except Exception:
        head = "no-git"
    rec = {"name": name, "bars": bars, "notes": notes,
           "pinned_at": time.time(),
           "pinned_at_h": time.strftime("%Y-%m-%d %H:%M:%S"),
           "commit": head}
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return rec

def require(name, artifact=None):
    path = os.path.join(REG_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"NO REGISTRATION for '{name}' — bars pin BEFORE "
                           f"measurement; call register() first (#159's door)")
    rec = json.load(open(path))
    if artifact and os.path.exists(artifact):
        if rec["pinned_at"] >= os.path.getmtime(artifact):
            raise RuntimeError(
                f"registration '{name}' POSTDATES its artifact {artifact} — "
                f"a bar pinned after seeing data is unrecoverable; the read "
                f"refuses (#159: ordering enforced, blindness not claimed)")
    return rec["bars"]
