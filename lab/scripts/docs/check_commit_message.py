"""Check commit messages against the repository's commit rules.

Usage:
    python check_commit_message.py FILE
    python check_commit_message.py --commits REV [REV ...]

The first form is the commit-msg hook that `prek.toml` installs: FILE
is the message git is about to record. The second checks the recorded
messages of the commits `git rev-list REV ...` lists, for example
`--commits origin/main..HEAD` before a push or `--commits HEAD` for the
whole history. Each problem is printed on one line; the exit status is
1 if any message breaks a rule.

The rules are the mechanical part of CONTRIBUTING.md's "Commits":

- the subject is at most 72 characters, starts with an upper-case
  letter, carries no `type:` or `type(scope):` prefix, does not end in
  a full stop, is not a merge's subject (`main` stays linear), and
  does not open with a vague verb (`VAGUE`);
- a blank line separates the subject from the body, and no body line
  exceeds 72 columns;
- no trailer block: no line that `git interpret-trailers` parses as a
  trailer, and no `Co-authored-by:` or `Signed-off-by:` line anywhere;
- no commit hash: no 7- to 40-digit hexadecimal token that names a
  commit of this repository (cite a commit by its subject instead);
- no banned word of `banned_words.TONE` or `banned_words.WORDS`
  outside a backticked code span, where it may name the code it
  describes;
- no email address (the pattern of the email step of CI's lint job,
  `.github/workflows/ci.yml`: change the two together).

Imperative mood, accuracy against the diff and the reason a body gives
are left to the writer and the reviewer.

In hook mode the message is checked as git records it after an editor
session: everything from the scissors line of `git commit -v` down and
every comment line (`core.commentChar`) is dropped first. A message
given with `-m` or `-F` keeps such lines verbatim, and they go
unchecked. A `fixup!`, `squash!` or `amend!` subject, which `git rebase
--autosquash` folds into its target before a push, is exempt from the
subject rules; its body is checked.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from banned_words import TONE, WORDS  # noqa: E402

WIDTH = 72
VAGUE = re.compile(
    r"(update|improve|fix stuff|cleanup|clean up|various|misc)\b", re.I
)
PREFIX = re.compile(r"[A-Za-z]+(\([^)]*\))?!?: ")
AUTOSQUASH = ("fixup! ", "squash! ", "amend! ")
TRAILER_LINE = re.compile(r"(?im)^(co-authored-by|signed-off-by)\s*:")
HEX = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")
CODE_SPAN = re.compile(r"`[^`\n]*`")
EMAIL = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)


def git(*args, stdin=None):
    """Run git with `args` and return its stdout; raise on failure."""
    return subprocess.run(
        ["git", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def as_recorded(raw):
    """Return the message git records from editor text `raw`.

    Drops the scissors line and everything below it, then the comment
    lines and surplus blank lines (`git stripspace --strip-comments`,
    which reads `core.commentChar`).
    """
    char = git("config", "--default", "#", "core.commentChar").strip()
    if len(char) != 1:  # "auto" picks a character per message
        char = "#"
    scissors = f"{char} {'-' * 24} >8 {'-' * 24}"
    lines = raw.split("\n")
    if scissors in lines:
        lines = lines[: lines.index(scissors)]
    return git("stripspace", "--strip-comments", stdin="\n".join(lines))


def problems(msg):
    """Return the rules the recorded message `msg` breaks.

    One string per problem; an empty list means the message passes.
    """
    out = []
    lines = msg.rstrip("\n").split("\n")
    subject = lines[0]
    if not subject.startswith(AUTOSQUASH):
        if len(subject) > WIDTH:
            out.append(f"subject is {len(subject)} characters (> {WIDTH})")
        if not subject[:1].isupper():
            out.append("subject does not start with an upper-case letter")
        if PREFIX.match(subject):
            out.append("subject carries a type prefix")
        if subject.endswith("."):
            out.append("subject ends in a full stop")
        if subject.startswith("Merge"):
            out.append("a merge commit's subject (main stays linear)")
        if VAGUE.match(subject):
            out.append(f"subject opens with a vague verb ({subject[:20]!r})")
    if len(lines) > 1 and lines[1]:
        out.append("no blank line after the subject")
    for k, line in enumerate(lines[1:], 2):
        if len(line) > WIDTH:
            out.append(f"line {k} is {len(line)} columns (> {WIDTH})")
    trailers = git("interpret-trailers", "--parse", "--no-divider", stdin=msg)
    if trailers.strip() or TRAILER_LINE.search(msg):
        out.append("a trailer block (none is allowed)")
    tokens = sorted(set(HEX.findall(msg)))
    if tokens:
        found = git(
            "cat-file",
            "--batch-check",
            stdin="".join(f"{t}^{{commit}}\n" for t in tokens),
        ).splitlines()
        for tok, res in zip(tokens, found, strict=True):
            if not res.endswith("missing"):
                out.append(f"commit hash {tok} (cite the subject instead)")
    prose = CODE_SPAN.sub(" ", msg)
    words = {m.group(1).lower() for m in TONE.finditer(prose)}
    words |= {m.group(1).lower() for m in WORDS.finditer(prose)}
    for w in sorted(words):
        out.append(f"banned word {w!r} outside a code span")
    if EMAIL.search(msg):
        out.append("an email address")
    return out


def main(argv):
    """Check the hook's message file or the listed commits.

    Returns the exit status: 0 clean, 1 a rule broken, 2 a usage error.
    """
    if argv[:1] == ["--commits"] and len(argv) > 1:
        log = git("log", "-z", "--format=%H%n%B", *argv[1:])
        bad = 0
        for rec in filter(None, log.split("\0")):
            sha, _, msg = rec.partition("\n")
            for p in problems(msg):
                print(f"{sha[:10]} {msg.splitlines()[0][:40]!r}: {p}")
                bad = 1
        return bad
    if len(argv) != 1 or argv[0].startswith("-"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    msg = as_recorded(Path(argv[0]).read_text(encoding="utf-8"))
    if not msg.strip():
        return 0  # git refuses an empty message itself
    found = problems(msg)
    for p in found:
        print(f"commit message: {p}", file=sys.stderr)
    if found:
        print(
            f"The message is kept in {argv[0]}; edit it and retry with "
            f"`git commit -e -F {argv[0]}`.",
            file=sys.stderr,
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
