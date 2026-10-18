"""Render a Markdown report as a print-friendly HTML document."""

import re
from pathlib import Path


def render(source, destination=None, title=None):
    import markdown

    source = Path(source)
    destination = Path(destination) if destination else source.with_suffix(".html")
    src = source.read_text()
    lines, out = src.split("\n"), []
    item = re.compile(r"^(\s*)([-*]|\d+\.) ")
    in_list = False
    for ln in lines:
        m = item.match(ln)
        if m:
            ind = len(m.group(1))
            ln = ("    " if ind else "") + ln[ind:]
            ln = re.sub(r"^(\s*(?:[-*]|\d+\.) )#", r"\1\\#", ln)
            last = next((x for x in reversed(out) if x.strip()), "")
            if out and out[-1].strip() and (not item.match(out[-1]) or (not ind and last.startswith(" "))):
                out.append("")
            in_list = True
        elif not ln.strip():
            in_list = False
        elif in_list and ln.startswith(" ") and not ln.lstrip().startswith("|"):
            ln = "    " + ln.lstrip()
        out.append(ln)
    body = markdown.markdown("\n".join(out), extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r'<p>(<img [^>]+>)</p>', r'<figure>\1</figure>', body)
    css = """
@page { size: A4; margin: 16mm 13mm 16mm 13mm;
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt sans-serif; color: #777; } }
body { font-family: "DejaVu Sans", "Helvetica", sans-serif; font-size: 9.6pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt; border-bottom: 2px solid #2a78d6; padding-bottom: 5pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; color: #1b4f8f; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 16pt; margin: 4pt 0; }
li { margin: 2pt 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f1f1ee; padding: 0 2px; border-radius: 2px; word-break: break-word; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 7.4pt; line-height: 1.3; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #d6d5cf; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #eef3fa; }
tbody tr:nth-child(even) td { background: #fafaf8; }
figure { margin: 8pt 0 12pt; break-inside: avoid; }
figure img, img { width: 100%; height: auto; border: 1px solid #e6e5e0; }
strong { color: #000; }
"""
    page_title = title or source.stem.replace("_", " ")
    destination.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{page_title}</title><style>{css}</style></head><body>{body}</body></html>")
    return destination
