"""Assemble paper/draft/*.md into one document with final numbering.

- Section renumber (the 5+6 merge): 7->6, 8->7, 9->8, 10->9, 11->10,
  applied to headers (## N. / ### N.M) and inline §N / §N.M refs.
- Figure map (draft names -> final sequential numbers):
    f7c chain of custody  -> Figure 1
    f7b entropy basins    -> Figure 2
    f3a binding specimens -> Figure 3
    f7a frontier          -> Figure 4
    f8a register map      -> Figure 5
    f8b length warp       -> Figure 6
    f9b survivor cascade  -> Figure 7
    f5a rotation          -> Figure 8
    f9a saturation        -> Figure 9
- Tables: census = Table 1, laws = Table 2.
- HTML comment blocks (working notes) are stripped.

Figure numbers finalize again at camera-ready if section edits reorder
first citations; this pass makes the draft read as one paper.
"""
import re
from pathlib import Path

DRAFT = Path(__file__).parent / "draft"
OUT = Path(__file__).parent / "paper1_assembled.md"

ORDER = [
    "s00_front_door.md",
    "s02_related_work.md",
    "s03_architecture.md",
    "s04_corpus_discipline.md",
    "s05_repair_stack.md",
    "s07_certification_lattice.md",
    "s08_external_anchor.md",
    "s09_method.md",
    "s10_reading_campaign.md",
    "s11_honest_limitations.md",
    "contributions.md",
]

SECMAP = {2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10}

FIGMAP = [  # (old, new) — longest/most-specific first
    ("Fig. 3-a's", "Figure 3's"),
    ("Fig. 3-a", "Figure 3"),
    ("Fig. 5-a", "Figure 8"),
    ("Fig. 7-b", "Figure 2"),
    ("Figure 7-a", "Figure 4"),
    ("Figure N", "Figure 4"),
    ("Fig. 8-a", "Figure 5"),
    ("Fig. 8-b", "Figure 6"),
    ("Fig. 9-a's", "Figure 9's"),
    ("Fig. 9-a", "Figure 9"),
    ("Fig. 9-b", "Figure 7"),
    ("Fig. 1", "Figure 1"),
    ("Table 9-1", "Table 2"),
]


def renumber_sections(text):
    def sub_ref(m):
        major = int(m.group(1))
        rest = m.group(2) or ""
        return f"§{SECMAP.get(major, major)}{rest}"
    text = re.sub(r"§(\d+)(\.\d+)?", sub_ref, text)

    def sub_h2(m):
        return f"## {SECMAP.get(int(m.group(1)), int(m.group(1)))}."
    text = re.sub(r"^## (\d+)\.", sub_h2, text, flags=re.M)

    def sub_h3(m):
        return f"### {SECMAP.get(int(m.group(1)), int(m.group(1)))}.{m.group(2)}"
    text = re.sub(r"^### (\d+)\.(\d+)", sub_h3, text, flags=re.M)
    return text


def strip_comments(text):
    return re.sub(r"<!--.*?-->\n?", "", text, flags=re.S)


def strip_merge_note(text):
    return text.replace(
        "*(§5 and §6 of the original skeleton, merged: the stack's measured\n"
        "capabilities, ending on the boundary they stop at. The survivor arc's\n"
        "narrative belongs to §9.2 and is not retold here.)*\n\n", "")


parts = []
for name in ORDER:
    t = (DRAFT / name).read_text()
    t = strip_comments(t)
    if name == "s05_repair_stack.md":
        t = strip_merge_note(t)
    for old, new in FIGMAP:
        t = t.replace(old, new)
    t = renumber_sections(t)
    parts.append(t.strip() + "\n")

doc = "\n\n".join(parts)

# Figures: embed each image after the paragraph that first anchors it.
FIGS = [
    ("certification lattice** (Figure 1)", "f7c_chain_of_custody",
     "**Figure 1.** The chain of custody: four gates, four invariances "
     "(register, rendering, lineage, truth), five real trajectories — "
     "every gate annotated with the failure it provably catches."),
    ("Figure 3 shows the four specimens", "f3a_binding_theorem",
     "**Figure 3.** The binding theorem by specimen: same frame, "
     "different knots (top); same knot, different surfaces (bottom). "
     "All four items and both graphs are real, from the banked fixtures."),
    ("That is the epigraph in numbers", "f7b_entropy_basins",
     "**Figure 2.** Vote entropy by outcome class (n = 36 pilot): "
     "entropy separates shallow from deep and cannot separate deep-wrong "
     "from deep-correct — temperature is orthogonal to truth."),
    ("Figure 4 (the precision–coverage frontier) plots every",
     "f7a_precision_coverage",
     "**Figure 4.** The precision–coverage frontier: the vote-threshold "
     "ladder at first measurement, and the certification channel "
     "widening across generations to the freeze point (912/1500 at "
     "measured 1.0000)."),
    ("The constellation's shape survives; its coordinates do not.",
     "f5a_rotation_not_decay",
     "**Figure 8.** Drift is rotation, not decay: raw constellations "
     "(mean cos 0.59), Procrustes-aligned (0.988), and the per-kind "
     "residuals reported honestly. Computed from the actual gen-5 and "
     "gen-9b centroids; the script asserts reproduction."),
    ("Figure 5 maps every population", "f8a_register_map",
     "**Figure 5.** The register map: the native fixture, the census "
     "pool's hundred per-item reads, and the foreign benchmark's banked "
     "band on one ruler (raw vintage; corrected reads in Figure 6)."),
    ("both panels from the banked per-item reads", "f8b_length_warp",
     "**Figure 6.** The length warp, before and after: raw distance "
     "correlates with length at r = −0.825; the 1/length correction "
     "takes it to −0.024, and the foreign refusal holds at 100% on the "
     "straightened ruler."),
    ("refutations ran in order (Figure 7)", "f9b_survivor_cascade",
     "**Figure 7.** The survivor cascade: nine registered refutations — "
     "five kills, the mechanism named at step six, three repairs priced "
     "and declined — ending in a priced population, not a mystery."),
    ("measured its own end: Figure 9's", "f9a_saturation_curve",
     "**Figure 9.** The saturation curve, with its scope drawn on the "
     "plot: the campaign measured its own completion for this "
     "distribution; the library did not close."),
]
paras = doc.split("\n\n")
out_paras = []
for p in paras:
    out_paras.append(p)
    flat = " ".join(p.split())
    for anchor, fname, caption in FIGS:
        if " ".join(anchor.split()) in flat:
            out_paras.append(f"![{caption}](figures/out/{fname}.png)")
doc = "\n\n".join(out_paras)
n_embedded = doc.count("](figures/out/")
assert n_embedded == len(FIGS), f"embedded {n_embedded} != {len(FIGS)} figures"

# References: entries only (the per-entry "cited for" ledger notes stay
# in bibliography.md for the stranger read's use-matches-source check).
bib = (Path(__file__).parent / "bibliography.md").read_text()
cited_block = bib.split("## Cited\n", 1)[1].split("## Verified, held", 1)[0]
entries = [ln for ln in cited_block.splitlines() if ln.startswith("- ")]
doc += ("\n\n## References\n\n*(Every entry verified against its source; "
        "per-entry \"cited for\" notes in bibliography.md.)*\n\n"
        + "\n".join(entries) + "\n")
# label the census table where it is introduced
doc = doc.replace(
    "The parameter census, re-run at the freeze tag against the deployed\n"
    "checkpoints rather than quoted from memory:",
    "The parameter census (Table 1), re-run at the freeze tag against the\n"
    "deployed checkpoints rather than quoted from memory:")
doc = doc.replace("Table 2 lists", "Table 2 lists")

assert "§11" not in doc and "Figure N" not in doc and "Fig. " not in doc, \
    "unmapped reference survived assembly"
OUT.write_text(doc)
print(f"[assemble] {len(ORDER)} sections -> {OUT} ({len(doc.splitlines())} lines)")
for tok in ["§6", "§7", "§8", "§9", "§10", "Figure 1", "Figure 9", "Table 2"]:
    print(f"  {tok}: {doc.count(tok)} refs")
