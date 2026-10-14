"""Check relative links and #anchors in Markdown files (GitHub slugs).

Usage: python check_links.py FILE.md [...]
"""

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def slug(text):
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in " -":
            out.append("-" if ch == " " else ch)
        elif ch == "_" or cat[0] in "LN":
            out.append(ch)
    return "".join(out)


def anchors(path):
    seen = {}
    res = set()
    in_code = False
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = HEAD.match(line)
        if m:
            s = slug(m.group(2))
            n = seen.get(s, 0)
            res.add(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
        # An HTML anchor (``<a id="x">`` / ``<a name="x">``) is a target too.
        res.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', line))
    return res


bad = 0
for name in sys.argv[1:]:
    p = Path(name)
    text = p.read_text()
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)  # a link in inline code is text
    for m in list(LINK.finditer(text)) + list(IMG.finditer(text)):
        target = m.group(1)
        if re.match(r"[a-z]+:", target) or target.startswith("mailto"):
            continue
        path_part, _, frag = target.partition("#")
        dest = (p.parent / path_part).resolve() if path_part else p.resolve()
        if path_part and not dest.exists():
            print(f"{name}: missing path {target}")
            bad += 1
            continue
        if frag and dest.suffix == ".md" and frag not in anchors(dest):
            print(f"{name}: missing anchor {target}")
            bad += 1
print(f"{bad} problem(s)")
