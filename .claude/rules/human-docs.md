---
paths:
  - "README.md"
  - "CONTRIBUTING.md"
  - "docs/**/*.md"
  - "examples/**/README.md"
  - "scripts/README.md"
  - "src/**/README.md"
  - "tests/README.md"
---

# Human-facing docs

- These docs are pointer-first: they name the docstring that holds the
  detail. Where one disagrees with a docstring, fix the docstring
  first.
- Spelling is Oxford: -ize (matching identifiers such as
  `check_laminarization`) with British -our, -re and -lling
  (centreline, modelling, travelling).
- Plain words, not fancy ones: never "flagship", "oracle" or "anchor".
  The Pallas kernel is "custom", not "hand-written": the README's "Use
  of AI" section keeps "by hand" for the AI-free first version. Make
  only mathematical claims, verified against the code, never
  phenomenological ones.
- Display math goes in ```` ```math ```` fences: GitHub's double-dollar
  blocks break on a line that starts with `-` or `+`. An inline `$...$`
  never spans a line break (GitHub renders it raw).
- GitHub's Markdown parser runs before the math renderer. In inline
  math `\,` and `\{` lose their backslash, `<` arrives double-escaped
  and `[a](b)` becomes a link: write `\ `, `\lbrace` / `\rbrace` and
  `\lt`, or use a math block. A `$` directly after a hyphen or an
  en-dash does not open math.
- Em-dashes stay: a spaced hyphen ` - ` used as a dash becomes ` — `.
