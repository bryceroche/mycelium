"""The figure style contract — every paper figure imports this.

One place for fonts, palette, and the freeze stamp. Every saved figure
carries the freeze tag + manifest hashes in a visible footer AND in the
file's embedded metadata (PDF Subject/Keywords, PNG text chunks), so a
figure detached from the paper still cites its fixtures.
"""
import json
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
MANIFEST = REPO / ".cache" / "GENERATION.json"

# Okabe-Ito (colorblind-safe), named for the roles figures actually use.
PALETTE = {
    "ink":   "#1a1a1a",   # text, axes
    "faint": "#8b8b8b",   # gridlines, ghost annotations
    "gate":  "#0072B2",   # structural elements (gates, bars)
    "ok":    "#009E73",   # the certified / passing channel
    "kill":  "#D55E00",   # deaths, refusals, kill bars
    "wild":  "#E69F00",   # the wild register / OOD
    "alt":   "#CC79A7",   # secondary channel (answer-not-certify, arms)
    "cool":  "#56B4E9",   # tertiary series
}


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
    })


def freeze_meta():
    """Tag + manifest identity. Fails loudly if the manifest is missing."""
    desc = subprocess.run(
        ["git", "describe", "--tags", "--always"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    m = json.loads(MANIFEST.read_text())
    return {
        "tag": desc,
        "gen": m["gen_id"],
        "hashes": m["hashes"],
        "parser": m["hashes"]["parser"],
    }


def stamp(fig, meta=None, fixtures=None):
    """Visible footer: freeze tag, generation, parser hash, fixtures read."""
    meta = meta or freeze_meta()
    line = f"{meta['tag']} · gen-{meta['gen']} · parser {meta['parser']}"
    if fixtures:
        line += " · " + ", ".join(fixtures)
    fig.text(0.995, 0.005, line, ha="right", va="bottom",
             fontsize=6, color=PALETTE["faint"], family="DejaVu Sans Mono")
    return meta


def save(fig, name, meta=None, fixtures=None):
    """Write PDF + PNG to paper/figures/out with self-citing metadata."""
    meta = meta or freeze_meta()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = "; ".join(f"{k}={v}" for k, v in meta["hashes"].items())
    subject = f"Mycelium paper 1 · {meta['tag']} · gen-{meta['gen']}"
    keywords = hashes + (
        " · fixtures: " + ", ".join(fixtures) if fixtures else ""
    )
    fig.savefig(OUT / f"{name}.pdf",
                metadata={"Subject": subject, "Keywords": keywords})
    fig.savefig(OUT / f"{name}.png",
                metadata={"Subject": subject, "Keywords": keywords})
    print(f"[figstyle] {name}: pdf+png -> {OUT}  ({meta['tag']})")
