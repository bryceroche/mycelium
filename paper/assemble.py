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
