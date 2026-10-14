"""Apply Oxford spelling inside documentation ranges of the Python tree.

-ise/-isation -> -ize/-ization (exempt words keep -ise), modeling ->
modelling, artifact -> artefact.  Only comments, docstrings and
description=/help= strings change (see doc_ranges.py).  Prints a count
per file; ``--dry`` prints the planned replacements instead.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_ranges import doc_ranges  # noqa: E402

ROOT = str(Path(__file__).resolve().parents[3])  # the repository root
DRY = "--dry" in sys.argv
SUFFIXES = ["ations", "ation", "ers", "er", "ing", "es", "ed", "e", "able"]
PAT = re.compile(r"\b[A-Za-z]+is(?:" + "|".join(SUFFIXES) + r")\b")
EXEMPT = re.compile(
    r"(?i)^(otherwise|\w*wise|precise|concise|rise|rises|rising|arise|"
    r"arises|arising|raise[sd]?|raising|raiser|noise|noises|denoise|"
    r"exercise[sd]?|exercising|advise[sd]?|advising|advertise[sd]?|"
    r"advertising|comprise[sd]?|comprising|compromise[sd]?|devise[sd]?|"
    r"devising|revise[sd]?|revising|supervise[sd]?|surprise[sd]?|"
    r"surprising|premise|premises|promise[sd]?|enterprise|expertise|"
    r"franchise|merchandise|kleiser|praise|poise|cruise|bruise|guise|"
    r"disguise[sd]?|paradise|treatise|incise|excise|precis\w*)$"
)
OTHER = [
    (re.compile(r"\bmodeling\b"), "modelling"),
    (re.compile(r"\bModeling\b"), "Modelling"),
    (re.compile(r"\bmodeled\b"), "modelled"),
    (re.compile(r"\bartifacts\b"), "artefacts"),
    (re.compile(r"\bartifact\b"), "artefact"),
]


def oxford(word):
    for suf in SUFFIXES:
        tail = "is" + suf
        if word.endswith(tail):
            return word[: -len(tail)] + "iz" + suf
    return word


files = subprocess.run(
    ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True
).stdout.split()
# This script's own docstring names the spellings it replaces.
SELF = Path(__file__).resolve()
files = [f for f in files if Path(ROOT, f).resolve() != SELF]
total = 0
for f in files:
    path = f"{ROOT}/{f}"
    src = Path(path).read_text()
    out = []
    last = 0
    cnt = [0]
    for a, b, _kind in doc_ranges(src):
        if a < last:
            continue  # nested range already handled
        out.append(src[last:a])
        seg = src[a:b]

        def repl(m, *, f=f, cnt=cnt):
            w = m.group(0)
            if EXEMPT.match(w):
                return w
            new = oxford(w)
            if new != w:
                cnt[0] += 1
                if DRY:
                    print(f"{f}: {w} -> {new}")
            return new

        seg = PAT.sub(repl, seg)
        for pat, rep in OTHER:
            seg, k = pat.subn(rep, seg)
            if k and DRY:
                print(f"{f}: {pat.pattern} -> {rep} x{k}")
            cnt[0] += k
        out.append(seg)
        last = b
    out.append(src[last:])
    new_src = "".join(out)
    n = cnt[0]
    if n:
        total += n
        if not DRY:
            Path(path).write_text(new_src)
        print(f"{n:4d} {f}")
print("total", total)
