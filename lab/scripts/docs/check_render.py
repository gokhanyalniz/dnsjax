"""Render local Markdown through GitHub's API and compare its math.

Usage: python check_render.py <repo-relative .md> ...

POSTs each file's text to https://api.github.com/markdown (mode gfm,
context gokhanyalniz/dnsjax), extracts every <math-renderer> payload and
reports any that is not, verbatim, a math span of the source, and any
'$' left in rendered text outside math and code.
"""

import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root


def render(text):
    body = json.dumps(
        {"text": text, "mode": "gfm", "context": "gokhanyalniz/dnsjax"}
    ).encode()
    req = urllib.request.Request(
        "https://api.github.com/markdown",
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


def source_math(md):
    segs = []
    for m in re.finditer(r"```math\n(.*?)```", md, re.S):
        segs.append(m.group(1).strip())
    body = re.sub(r"```.*?```", "", md, flags=re.S)
    body = re.sub(r"`[^`\n]*`", "", body)
    for m in re.finditer(r"(?<![\\$])\$([^$\n]+?)\$(?!\$)", body):
        segs.append(m.group(1).strip())
    return segs


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


total = 0
for path in sys.argv[1:]:
    md = (ROOT / path).read_text()
    src = {norm(s) for s in source_math(md)}
    h = render(md)
    time.sleep(1.5)
    rendered = [
        html.unescape(m.group(1))
        for m in re.finditer(
            r"<math-renderer[^>]*>(.*?)</math-renderer>", h, re.S
        )
    ]
    problems = []
    for r in rendered:
        t = r.strip()
        for a, b in (("$$", "$$"), ("$`", "`$"), ("$", "$")):
            if t.startswith(a) and t.endswith(b) and len(t) > len(a) + len(b):
                t = t[len(a) : -len(b)]
                break
        if "<" in t or "&" in t:
            problems.append(("HTML/escape inside math", r[:160]))
        elif norm(t) not in src:
            problems.append(("altered/unmatched", r[:160]))
    outside = re.sub(r"<math-renderer.*?</math-renderer>", "", h, flags=re.S)
    outside = re.sub(r"<pre.*?</pre>", "", outside, flags=re.S)
    outside = re.sub(r"<code.*?</code>", "", outside, flags=re.S)
    outside = re.sub(r"<[^>]+>", "", outside)
    for line in html.unescape(outside).splitlines():
        if "$" in line:
            problems.append(("raw $ in text", line.strip()[:160]))
    total += len(problems)
    print(
        f"== {path}: {len(src)} source spans, {len(rendered)} rendered, "
        f"{len(problems)} problems"
    )
    for kind, what in problems:
        print(f"   [{kind}] {what}")
print("TOTAL problems:", total)
