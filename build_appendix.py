"""Rebuild site/appendix.html from docs/APPENDIX_prompt_log_and_limits.md.
Run after every new prompt is added to the log:  python3 site/build_appendix.py
Uses only the Python standard library.
"""
import html, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "..", "docs", "APPENDIX_prompt_log_and_limits.md")
OUT = os.path.join(HERE, "appendix.html")


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s


def convert(md):
    # The preamble (title, placeholder links, tool line) is replaced by the page's own intro.
    md = md[md.index("## A."):]
    out, para, rows = [], [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if rows:
            head, *body = rows
            out.append('<div class="tblwrap"><table class="log"><thead><tr>'
                       + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table></div>")
            rows.clear()

    for line in md.splitlines():
        t = line.strip()
        if t.startswith("|"):
            if para:
                flush()
            cells = [c.strip() for c in t.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        elif t.startswith("## "):
            flush(); out.append(f"<h2>{inline(t[3:])}</h2>")
        elif not t:
            flush()
        else:
            if rows:
                flush()
            para.append(t)
    flush()
    return "\n".join(out)


TEMPLATE = open(os.path.join(HERE, "appendix_template.html"), encoding="utf-8").read()
body = convert(open(MD, encoding="utf-8").read())
open(OUT, "w", encoding="utf-8").write(TEMPLATE.replace("<!--LOG-->", body))
print("wrote", OUT)
