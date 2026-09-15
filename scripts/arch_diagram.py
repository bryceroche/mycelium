"""arch_diagram.py — THE ARCHITECTURE DRAWN FROM A CONFIG (2026-09-15, word
given): docs/architecture.json describes the system (groups, organs, edges,
positions); this tool COUNTS the parameters of every organ from the
checkpoint by key regex (rounded to 100k) — nothing typed by hand — and
renders paper/figures/out/architecture.{png,svg}. Unassigned checkpoint
keys are printed so the config can be completed. Re-run after any change:
  .venv/bin/python3 scripts/arch_diagram.py [docs/architecture.json]"""
import json, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "docs/architecture.json"))

def load_keys(path):
    from tinygrad.nn.state import safe_load
    return {k: int(np.prod(v.shape)) for k, v in safe_load(path).items()}

main_keys = load_keys(cfg["ckpt"]); extra = {n: load_keys(p) for n, p in cfg.get("ckpt_extra", {}).items()}
trunk = load_keys(cfg["trunk_weights"]); L = cfg["trunk_layers"]
assigned = set(); counts = {}
for nd in cfg["nodes"]:
    p = nd["params"]
    if "regex" in p:
        keys = extra[p["ckpt"]] if "ckpt" in p else main_keys
        hit = [k for k in keys if any(re.search(r, k) for r in p["regex"])]
        counts[nd["id"]] = sum(keys[k] for k in hit)
        if "ckpt" not in p: assigned.update(hit)
    elif p.get("llama_embed"):
        counts[nd["id"]] = sum(n for k, n in trunk.items() if "embed" in k)
    elif p.get("llama_layers"):
        counts[nd["id"]] = sum(n for k, n in trunk.items() if (m := re.search(r"layers\.(\d+)\.", k)) and int(m.group(1)) < L)
    else:
        counts[nd["id"]] = 0
unassigned = {k: v for k, v in main_keys.items() if k not in assigned}
head_total = sum(main_keys.values())
print(f"[arch] head params {head_total / 1e6:.2f}M; assigned {sum(main_keys[k] for k in assigned) / 1e6:.2f}M; UNASSIGNED {sum(unassigned.values()) / 1e6:.2f}M in {len(unassigned)} keys: {sorted(unassigned)[:12]}")

def fmt(n, symbolic=False):
    if symbolic: return "0 params (symbolic)"
    if n >= 1e6: return f"{round(n / 1e5) / 10:.1f}M params"
    if n >= 5e4: return f"{round(n / 1e5) / 10:.1f}M params" if n >= 5e4 else "<0.1M"
    return "~0 params"

fig, ax = plt.subplots(figsize=(31, 17)); ax.set_xlim(0, 126); ax.set_ylim(0, 100); ax.axis("off")
gpos = {}
for g in cfg["groups"]:
    ax.add_patch(FancyBboxPatch((g["x"], g["y"]), g["w"], g["h"], boxstyle="round,pad=0.4,rounding_size=1.5", fc=g["color"], ec="#999", lw=1.2, zorder=0))
    ax.text(g["x"] + 0.8, g["y"] + g["h"] - 1.8, g["label"], fontsize=11, weight="bold", color="#333", va="top", zorder=1)
NW, NH = 14.5, 9.5; centers = {}
for nd in cfg["nodes"]:
    x, y = nd["x"], nd["y"]; sym = nd["params"].get("symbolic", False)
    ax.add_patch(FancyBboxPatch((x, y), NW, NH, boxstyle="round,pad=0.3,rounding_size=0.8", fc="white", ec="#444" if not sym else "#8a6d3b", lw=1.4, ls="-" if not sym else "--", zorder=2))
    ax.text(x + 0.5, y + NH - 0.9, nd["label"], fontsize=9.6, weight="bold", va="top", zorder=3)
    ax.text(x + 0.5, y + NH - 3.0, nd["detail"], fontsize=6.9, va="top", wrap=True, zorder=3, color="#333", linespacing=1.15) if False else None
    # wrap the detail by hand (matplotlib's wrap ignores the box width)
    words = nd["detail"].split(); lines = []; cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > 36: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    lines.append(cur)
    ax.text(x + 0.5, y + NH - 3.0, "\n".join(lines[:4]), fontsize=6.9, va="top", zorder=3, color="#333", linespacing=1.15)
    ax.text(x + NW - 0.5, y + 0.6, fmt(counts[nd["id"]], sym), fontsize=7.4, ha="right", va="bottom", color="#1a4d8f" if not sym else "#8a6d3b", weight="bold", zorder=3)
    centers[nd["id"]] = (x + NW / 2, y + NH / 2, x, y)
for a, b, lab in cfg["edges"]:
    (ax0, ay0, x0, y0), (bx0, by0, x1, y1) = centers[a], centers[b]
    # attach at box edges: leave from the right/top/bottom, arrive at the left, by relative position
    if abs(bx0 - ax0) > NW: sa = (x0 + NW if bx0 > ax0 else x0, ay0); sb = (x1 if bx0 > ax0 else x1 + NW, by0)
    else: sa = (ax0, y0 + NH if by0 > ay0 else y0); sb = (bx0, y1 if by0 > ay0 else y1 + NH)
    ax.add_patch(FancyArrowPatch(sa, sb, arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#555", connectionstyle="arc3,rad=0.12", zorder=1.5, alpha=0.85))
    if lab: ax.text((sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2 + 0.8, lab, fontsize=6.4, color="#555", ha="center", zorder=3, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.8))
tr = counts["trunk"] + counts["embed"]
ax.text(1, 97, cfg["title"], fontsize=18, weight="bold", va="top")
ax.text(1, 93.6, f"{cfg['subtitle']}   |   trained head {head_total / 1e6:.1f}M params (from {cfg['ckpt'].split('/')[-1]})   |   frozen trunk {tr / 1e6:.0f}M (embedding {counts["embed"] / 1e6:.0f}M + L0-L3 {counts["trunk"] / 1e6:.0f}M)   |   dashed = symbolic organ (0 params)", fontsize=10, va="top", color="#333")
for ext in ("png", "svg"):
    out = f"paper/figures/out/architecture.{ext}"; fig.savefig(out, dpi=110 if ext == "png" else None, bbox_inches="tight"); print(f"[arch] wrote {out}")
