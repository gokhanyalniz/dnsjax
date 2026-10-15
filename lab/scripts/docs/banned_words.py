"""The banned-word patterns the prose checks share.

`TONE` is the tone and filler vocabulary `check_docstrings.py` counts
over the tracked tree's docstrings and comments; `WORDS` the filler,
temporal and marker words `lint_docs.py` flags in named files.
`check_commit_message.py` applies both to commit messages, outside
backticked code spans: a banned word is allowed only where it is the
name of the code it describes, as an argument's name can be. Each
pattern has one capturing group, the matched word.
"""

import re

TONE = re.compile(
    r"\b(flagship|oracle|anchor(?:s|ed|ing)?|hand-written|handwritten|"
    r"honest(?:ly)?|crucial(?:ly)?|notably|importantly|seamless(?:ly)?|"
    r"leverag(?:e|es|ed|ing)|robust(?:ly|ness)?|comprehensive|"
    r"cutting-edge|state-of-the-art|delv(?:e|es|ing)|simply|obviously|"
    r"clearly|trivially|elegant(?:ly)?|powerful|magic(?:al)?|"
    r"essentially|basically|literally|very|really|actually|just)\b",
    re.I,
)
WORDS = re.compile(
    r"\b(actually|simply|honest(?:ly)?|oracle|anchor(?:ed|s)?|today|"
    r"currently|in the future|future work|hand-written|this box|"
    r"dev box|for now|not yet implemented|TODO|FIXME|XXX)\b",
    re.I,
)
