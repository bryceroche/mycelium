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

TITLE = "Certify, Answer, Flag, Abstain: A Chain of Custody for Machine-Read Mathematics"
BYLINE = "Bryce Roche · Claude (Anthropic)"
TAG = subprocess.run(["git", "describe", "--tags", "--always"], cwd=ROOT,
                     capture_output=True, text=True).stdout.strip()
STAMP = f"July 2026 · {TAG}"

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
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='0.9em' font-size='90'>✓</text></svg>">
<style>{CSS}</style>{THEME_JS}</head><body>
<header class="masthead"><div class="wrap">
<a class="brand" href="{p if depth else './'}">The Shape of Thought</a>
<a class="theme-note" href="javascript:flip()">light / dark</a>
</div></header>
<div class="wrap">
{body}
</div>
<footer><div class="wrap">
{BYLINE} · {STAMP} · every number in the paper traces to a pinned
fixture or the public ledger · <a href="https://github.com/bryceroche/mycelium">code &amp; ledger</a>
</div></footer></body></html>"""


# ---------------------------------------------------------------- paper page
src = (PAPER / "paper1_assembled.md").read_text()
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
<p class="stamp">{STAMP} · <a href="../paper1.pdf">download PDF</a></p>
<div class="paper-body">{body}</div>
""", depth=1)

# ---------------------------------------------------------------- landing
abstract = src.split("## Abstract\n", 1)[1].split("\n## ", 1)[0].strip()
abstract_html = markdown.markdown(abstract, extensions=["smarty"])
landing = page("The Shape of Thought", f"""
<h1 class="paper-title">{TITLE}</h1>
<p class="byline">{BYLINE}</p>
<p class="stamp">{STAMP}</p>
<p class="lede">A deployed reasoning system&rsquo;s output should not be an
answer; it should be a decision.</p>
<div class="actions">
<a class="primary" href="paper/">Read the paper</a>
<a href="paper1.pdf">PDF</a>
<a href="ledger.md">The ledger (supplementary)</a>
<a href="https://github.com/bryceroche/mycelium">Code</a>
</div>
<h2>Abstract</h2>
{abstract_html}
<h2>What this is</h2>
<ul class="cardlist">
<li><span class="k">THE ARTIFACT</span><br>A certification lattice: a
small trained parser over a frozen trunk, an exact symbolic solver, and
zero-parameter decision machinery that certifies 912 of 1,500 held-out
problems at measured 1.0000 precision — with the boundary of that claim
measured and published at the same strength.</li>
<li><span class="k">THE METHOD</span><br>Fourteen model generations under
registered predictions with mechanized verdicts. The complete
chronological ledger — every registration, kill bar, verdict, and law —
ships as supplementary material: offered for audit, not trust.</li>
<li><span class="k">THE BET</span><br>The paper closes with a
falsifiable, pre-registered prediction that its own certification
instrument will age — published with the succession plan for its
replacement.</li>
</ul>
<figure><img src="figures/f7c_chain_of_custody.png" alt="The chain of custody"
loading="lazy"><figcaption><strong>Figure 1.</strong> The chain of
custody: four gates, four invariances, five real trajectories — every
gate annotated with the failure it provably catches.</figcaption></figure>
<h2>Coming</h2>
<p><em>Guided by Primes</em> (Paper II — the abstraction ladder) and
<em>The Shadow of Intelligence</em> (essay). This site is the canonical
home; the byline is the byline.</p>
""")

# ---------------------------------------------------------------- write dist
if DIST.exists():
    shutil.rmtree(DIST)
(DIST / "paper").mkdir(parents=True)
(DIST / "figures").mkdir()
(DIST / "index.html").write_text(landing)
(DIST / "paper" / "index.html").write_text(paper_html)
shutil.copy(PAPER / "paper1.pdf", DIST / "paper1.pdf")
shutil.copy(ROOT / "docs" / "phase1_skeleton_spec.md", DIST / "ledger.md")
for png in (PAPER / "figures" / "out").glob("*.png"):
    shutil.copy(png, DIST / "figures" / png.name)
n = sum(1 for _ in DIST.rglob("*") if _.is_file())
print(f"[site] {n} files -> {DIST}")
