---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "scripts/**/*.py"
---

# Docstrings and comments

- A function, method or class docstring states its contract first:
  what it does, its arguments and return, and what a caller relies on
  (donation, sharding, units, invariants). The why takes a couple of
  sentences, more only where an exception is warranted. A derivation
  stays with the function whose math it is: tighten its wording, never
  drop a step or a result.
- Measurements (timings, A/B results, tables, the machine), rejected
  alternatives and history go in the module docstring's closing
  "Design notes" section, one titled entry per topic, which the
  docstring relying on it names. An entry drops repetition, never
  information: every number a conclusion rests on stays, with how it
  was measured. Keep a date only where a reader needs it to tell
  whether their data predates a change.
- A comment says what the next lines do, or why, in a sentence or two;
  longer reasoning is a Design notes entry.
- Describe dnsjax on its own terms: credit a borrowed method to its
  published source, and never describe the code it came from.
- Language, in docstrings, comments and the `description=` / `help=`
  strings that `--help` renders, is that of the human docs: Oxford
  spelling (-ize; British -our, -re, -lling), plain words (no "oracle",
  "flagship" or "anchor" for a reference value; the Pallas kernel is
  "custom"), and no reference to a development machine ("this box").
- Docstring math is LaTeX: inline `` `$...$` ``, display `.. math::`.
  A docstring containing a backslash is raw (`r"""`): in a plain one
  `\t` becomes a TAB and a trailing `\` eats its newline.
