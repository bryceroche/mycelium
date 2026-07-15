"""Render paper1_assembled.md to paper1.pdf (markdown -> HTML -> weasyprint).

Byline per the declared authorship policy; the venue version's byline
is decided at venue selection.
"""
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).parent
SRC = HERE / "paper1_assembled.md"
OUT = HERE / "paper1.pdf"

BYLINE = "Bryce Roche · Claude (Anthropic)"
DATE = "July 2026 · paper-1-freeze"

CSS = """
@page { size: A4; margin: 22mm 20mm;
  @bottom-center { content: counter(page); font-size: 8pt; color: #888; } }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 17pt; text-align: center; line-height: 1.3;
     margin: 0 0 4pt 0; }
.byline { text-align: center; font-size: 10.5pt; margin: 2pt 0; }
.date { text-align: center; font-size: 9pt; color: #666;
        margin: 0 0 18pt 0; }
h2 { font-size: 12.5pt; margin: 18pt 0 6pt; border-bottom: 0.6pt solid #bbb;
     padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 12pt 0 4pt; }
p { margin: 5pt 0; text-align: justify; }
blockquote { margin: 8pt 24pt; font-style: italic; color: #444; }
img { max-width: 100%; margin: 8pt auto 2pt; display: block; }
p > img + em, .caption { font-size: 8.5pt; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 8pt 0;
        width: 100%; }
th, td { border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left; }
th { background: #f0f0f0; }
li { margin: 3pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
"""

text = SRC.read_text()
title, rest = text.split("\n", 1)
title = title.lstrip("# ").strip()
body = markdown.markdown(rest, extensions=["tables", "smarty"])
# figure captions: alt text renders below the image
body = body.replace("<img alt=", "<img data-caption=")
import re
def fig_caption(m):
    cap = m.group(1)
    cap_html = markdown.markdown(cap)[3:-4]  # strip <p></p>
    return (f'<img src="{m.group(2)}"/>'
            f'<p style="font-size:8.5pt;color:#333;margin:2pt 12pt 10pt;">'
            f'{cap_html}</p>')
body = re.sub(r'<img data-caption="([^"]*)" src="([^"]*)"\s*/?>',
              fig_caption, body)

html = f"""<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<h1>{title}</h1>
<p class="byline">{BYLINE}</p>
<p class="date">{DATE}</p>
{body}
</body></html>"""

HTML(string=html, base_url=str(HERE)).write_pdf(OUT)
print(f"[render] {OUT} written")
