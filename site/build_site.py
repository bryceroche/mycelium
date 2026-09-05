"""Build theshapeofthought.ai into site/dist/ (static, self-contained).

Pages: / (landing = the paper's cover), /paper/ (full HTML paper),
/paper1.pdf, /figures/*, /ledger.md (supplementary).
Rebuild any time with: .venv/bin/python site/build_site.py
"""
import re
import shutil
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DIST = ROOT / "site" / "dist"

TITLE = "The Shape of Thought: Notes on Building a Reasoning Machine"
BYLINE = "Bryce Roche · Claude (Anthropic)"
TAG = subprocess.run(["git", "describe", "--tags", "--always"], cwd=ROOT,
                     capture_output=True, text=True).stdout.strip()
STAMP = f"September 2026 · {TAG}"

CSS = """
:root {
  --ground: #fbfbf9; --ink: #1c2422; --faint: #6c7672;
  --accent: #00795c; --link: #0b6aa8; --rule: #d8dcd9;
  --card: #f3f5f2; --mono: ui-monospace, 'Cascadia Mono', Menlo, monospace;
}
:root[data-theme="dark"] {
  --ground: #121715; --ink: #e6eae7; --faint: #939d98;
  --accent: #35c496; --link: #6db6e8; --rule: #2b332f;
  --card: #1a211e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #121715; --ink: #e6eae7; --faint: #939d98;
    --accent: #35c496; --link: #6db6e8; --rule: #2b332f;
    --card: #1a211e;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink);
  font: 17px/1.65 Georgia, 'Iowan Old Style', 'Times New Roman', serif; }
a { color: var(--link); text-decoration-thickness: 1px; }
.wrap { max-width: 46rem; margin: 0 auto; padding: 0 1.2rem 4rem; }
header.masthead { border-bottom: 1px solid var(--rule); }
.masthead .wrap { display: flex; justify-content: space-between;
  align-items: baseline; padding: 1rem 1.2rem; }
.brand { font-variant: small-caps; letter-spacing: 0.14em;
  font-size: 0.95rem; color: var(--faint); text-decoration: none; }
.theme-note { font-size: 0.75rem; color: var(--faint); }
h1.paper-title { font-size: 2.0rem; line-height: 1.25; margin: 2.6rem 0 0.8rem;
  text-wrap: balance; }
.byline { font-size: 1.05rem; margin: 0 0 0.15rem; }
.stamp { font-family: var(--mono); font-size: 0.72rem; color: var(--faint);
  margin: 0 0 1.6rem; }
.lede { font-size: 1.18rem; font-style: italic; color: var(--accent);
  border-left: 3px solid var(--accent); padding-left: 0.9rem;
  margin: 1.6rem 0; text-wrap: balance; }
.cardlist a, .cardlist a:visited { color: inherit; text-decoration: none; display: block; }
.cardlist li:hover { border-color: var(--accent); }
.topnav { display: flex; gap: 1.1rem; align-items: baseline; }
.topnav a { color: var(--ink-dim); text-decoration: none; font-size: 0.95rem; }
.topnav a:hover { color: var(--accent); }
.masthead .wrap { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; }
.actions { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1.4rem 0 2rem; }
.actions a { border: 1.5px solid var(--accent); color: var(--accent);
  text-decoration: none; padding: 0.45rem 0.95rem; border-radius: 3px;
  font-size: 0.92rem; }
.actions a.primary { background: var(--accent); color: var(--ground); }
h2 { font-size: 1.25rem; margin: 2.2rem 0 0.6rem;
  border-bottom: 1px solid var(--rule); padding-bottom: 0.25rem; }
h3 { font-size: 1.02rem; margin: 1.5rem 0 0.4rem; }
blockquote { margin: 1.2rem 1.4rem; font-style: italic; color: var(--faint); }
figure { margin: 1.6rem 0; }
figure img { max-width: 100%; border: 1px solid var(--rule);
  border-radius: 3px; background: #fff; }
figcaption { font-size: 0.82rem; color: var(--faint); margin-top: 0.4rem; }
table { border-collapse: collapse; font-size: 0.82rem; margin: 1rem 0;
  display: block; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: 0.3rem 0.55rem;
  text-align: left; }
th { background: var(--card); }
code { font-family: var(--mono); font-size: 0.85em; }
.cardlist { list-style: none; padding: 0; }
.cardlist li { background: var(--card); border: 1px solid var(--rule);
  border-radius: 4px; padding: 0.8rem 1rem; margin: 0.6rem 0; }
.cardlist .k { font-family: var(--mono); font-size: 0.72rem;
  color: var(--accent); letter-spacing: 0.06em; }
footer { border-top: 1px solid var(--rule); margin-top: 3rem; }
footer .wrap { padding: 1.2rem; font-size: 0.8rem; color: var(--faint); }
img { max-width: 100%; }
.paper-body p { text-align: justify; }
.paper-body img { border: 1px solid var(--rule); border-radius: 3px;
  background: #fff; display: block; margin: 1.4rem auto 0.3rem; }
.paper-body img + p em:first-child { font-size: 0.82rem; }
"""

THEME_JS = """<script>
const t = localStorage.getItem('theme');
if (t) document.documentElement.dataset.theme = t;
function flip() {
  const cur = document.documentElement.dataset.theme ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
}
</script>"""


def page(title, body, depth=0):
    p = "../" * depth
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{TITLE} — {BYLINE}">
<link rel="icon" type="image/png" sizes="64x64" href="/favicon-64.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>{CSS}</style>{THEME_JS}</head><body>
<header class="masthead"><div class="wrap">
<a class="brand" href="{p if depth else './'}">The Shape of Thought</a>
<nav class="topnav">
<a href="{p if depth else './'}paper/">Core ideas</a>
<a href="{p if depth else './'}blog/">Blog</a>
<a href="{p if depth else './'}bio/">About</a>
<a class="theme-note" href="javascript:flip()">light / dark</a>
</nav>
</div></header>
<div class="wrap">
{body}
</div>
<footer><div class="wrap">
{BYLINE} · {STAMP} · built on one card, between two jaws ·
<a href="https://github.com/bryceroche/mycelium">code</a>
</div></footer></body></html>"""


# ---------------------------------------------------------------- paper page
src = (PAPER / "shape_of_thought.md").read_text()
_, rest = src.split("\n", 1)
body = markdown.markdown(rest, extensions=["tables", "smarty"])
body = body.replace('src="figures/out/', 'src="../figures/')
# image alt text becomes a visible caption
def cap(m):
    c = markdown.markdown(m.group(1))[3:-4]
    return (f'<figure><img src="{m.group(2)}" loading="lazy">'
            f'<figcaption>{c}</figcaption></figure>')
body = re.sub(r'<img alt="([^"]*)" src="([^"]*)"\s*/?>', cap, body)
paper_html = page(TITLE, f"""
<h1 class="paper-title">{TITLE}</h1>
<p class="byline">{BYLINE}</p>
<p class="stamp">{STAMP}</p>
<h2>The two observations that seeded everything</h2>
<ul class="cardlist">
<li><a href="../blog/2026-09-03-the-solver-inside-technical/"><span class="k">1 · NEURAL NETS BUILD SOLVERS INSIDE THEIR WEIGHTS</span><br>
Frozen layers of a small language model, given hand-crafted attention
masks, solved Sudoku, map coloring, and KenKen — the constraint-
propagation machinery was already latent in the pretrained weights.
They need help with the masking; the solving is in there. (Full
story: The solver inside.)</a></li>
<li><a href="../blog/2026-09-03-the-solver-inside-technical/"><span class="k">2 · THE TELEGRAPH IN THE ATTENTION</span><br>
Our archived attention analyses (stored in our S3 buckets) carry a
two-state hidden Markov signal — attention snapping between local
focus and global survey, dwelling in each. We read this as the
alternation between the neural network and its internal, improvised
implementation of a solver — the rhythm this whole architecture
makes explicit.</a></li>
</ul>
<div class="paper-body">{body}</div>
""", depth=1)

# ---------------------------------------------------------------- landing
abstract = src.split("## Abstract\n", 1)[1].split("\n## ", 1)[0].strip()
abstract_html = markdown.markdown(abstract, extensions=["smarty"])
landing = page("The Shape of Thought", f"""
<h1 class="paper-title">{TITLE}</h1>
<p class="byline">{BYLINE}</p>
<p class="stamp">{STAMP}</p>
<p class="lede">A neuro-symbolic dance: neural proposes, symbolic
disposes &mdash; two jaws taking turns until the answer is certain, or
the silence is.</p>
<div class="actions">
<a href="paper/">Core ideas</a>
<a href="blog/">Blog</a>
<a href="bio/">About</a>
<a href="https://github.com/bryceroche/mycelium">Code</a>
</div>
<h2>Abstract</h2>
{abstract_html}
<h2>What this is</h2>
<ul class="cardlist">
<li><a href="blog/2026-09-01-what-is-the-shape-of-thought/"><span class="k">WHAT IS THE SHAPE OF THOUGHT?</span><br>The site's title question,
answered with measured geometry: a braid that tightens through nested
rings, breaking like the wave at Nazar&eacute; into a crystal of nodes
and edges.</a></li>
<li><a href="blog/2026-09-01-the-paper-in-plain-language/"><span class="k">THE PAPER, IN PLAIN LANGUAGE</span><br>The full technical paper,
rewritten for the largest possible audience. Every number verified;
every difficult idea explained rather than glossed.</a></li>
<li><a href="blog/2026-09-03-the-solver-inside-technical/"><span class="k">THE SOLVER INSIDE</span><br>A hidden two-state
switch found in a hundred attention traces, and the frozen model
that already knew how to solve Sudoku — the observations that seeded
the architecture.</a></li>
</ul>
""")

# ---------------------------------------------------------------- blog + bio
BLOG = ROOT / "site" / "blog"
blog_pages = []
for md in sorted(BLOG.glob("*.md"), reverse=True):
    lines = md.read_text().split("\n")
    meta = {}
    while lines and ":" in lines[0] and not lines[0].startswith("#"):
        k, v = lines.pop(0).split(":", 1)
        meta[k.strip()] = v.strip()
    slug = md.stem
    bhtml = markdown.markdown("\n".join(lines), extensions=["tables", "smarty"])
    blog_pages.append((slug, meta.get("title", slug), meta.get("date", ""), bhtml))
blog_index = ("<h1 class=\"paper-title\">Blog</h1>\n<ul class=\"cardlist\">"
    + '<li><a href="../paper/"><strong>Core ideas</strong> · pinned — '
      'the full essay: the shape of the machine, from the two jaws to the wall of witnesses</a></li>'
    + "".join(
    f'<li><a href="{slug}/"><strong>{t}</strong> · {d}</a></li>'
    for slug, t, d, _ in blog_pages) + "</ul>")
bio_src = (ROOT / "site" / "bio.md").read_text()
bio_html = markdown.markdown(bio_src, extensions=["smarty"])

# ---------------------------------------------------------------- write dist
if DIST.exists():
    shutil.rmtree(DIST)
(DIST / "paper").mkdir(parents=True)
(DIST / "figures").mkdir()
(DIST / "index.html").write_text(landing)
for ic in ("favicon-64.png", "favicon-32.png", "apple-touch-icon.png"):
    shutil.copy(ROOT / "site" / ic, DIST / ic)
(DIST / "paper" / "index.html").write_text(paper_html)
shutil.copy(PAPER / "paper1.pdf", DIST / "paper1.pdf")
shutil.copy(ROOT / "docs" / "phase1_skeleton_spec.md", DIST / "ledger.md")
for png in (PAPER / "figures" / "out").glob("*.png"):
    shutil.copy(png, DIST / "figures" / png.name)
n = sum(1 for _ in DIST.rglob("*") if _.is_file())
print(f"[site] {n} files -> {DIST}")

(DIST / "blog").mkdir(exist_ok=True)
for img in BLOG.glob("*.jpg"):
    shutil.copy(img, DIST / "blog" / img.name)
(DIST / "blog" / "index.html").write_text(page("Blog — The Shape of Thought", blog_index, depth=1))
for slug, t, d, bhtml in blog_pages:
    (DIST / "blog" / slug).mkdir(parents=True, exist_ok=True)
    (DIST / "blog" / slug / "index.html").write_text(
        page(f"{t} — The Shape of Thought",
             f'<p class="stamp">{d}</p><div class="paper-body">{bhtml}</div>'
             f'<p><a href="../">&larr; all posts</a></p>', depth=2))
(DIST / "bio").mkdir(exist_ok=True)
(DIST / "bio" / "index.html").write_text(
    page("Bryce Roche — The Shape of Thought",
         f'<div class="paper-body">{bio_html}</div>', depth=1))
print(f"[site] blog: {len(blog_pages)} posts + bio page")
