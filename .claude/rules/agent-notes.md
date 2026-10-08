---
paths:
  - "**/CLAUDE.md"
  - ".claude/rules/**/*.md"
---

# Writing the agent notes

- A line stays only if an agent needs it before, or instead of, reading
  the code that holds the answer: a command, a rule for the agent's own
  conduct, a convention for code not yet written, an invariant whose
  two ends live in different modules, or a pointer to a `module.symbol`
  docstring. How and why one module works, measured numbers, history
  and per-function behaviour belong in its docstring; a note never
  restates one.
- Put a note in the narrowest file whose load scope covers every file
  where its mistake could be made: the root `CLAUDE.md` (always
  loaded), a directory's `CLAUDE.md`, or a `.claude/rules/` file whose
  `paths:` globs name the files it governs (each glob must match a
  tracked file). The last two load when a file they cover is read,
  edited or written. Before adding a line, check that no note loaded
  alongside already says it. The indexes (the root's package map,
  `tests/`, `scripts/`) are one line per file.
- Sizes are guides, counted at the 79-column wrap. The root aims at
  about 200 lines; every line costs every session, so grow it only with
  the utmost care, after asking whether a directory note or a rule file
  covers the files where the mistake would be made. Those load only
  with their files and have more room (about 120 lines is comfortable,
  not a limit), but the line test still decides what goes in.
- State a rule as what to do, with its reason or a pointer to it: the
  reason is what carries a rule to a case it does not name.
- A note's general statement that code contradicts is a case for the
  root's Conduct rule: fix the code, and narrow or edit the note only
  where the note is what is wrong.
- Emphasis (bold, capitals) goes on one line per file at most; on many
  lines it marks none.
- Outside the indexes and the root's opening description, name the
  symbol that owns a set or a value (`flows.registry`,
  `parameters.OFF_CADENCES`), never a hand list, a count or a version
  number that changes with the code: those drift silently.
