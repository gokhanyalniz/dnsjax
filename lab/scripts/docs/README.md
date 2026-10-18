# Documentation checks

Mechanical and read-only unless the row says otherwise. Each script's
module docstring is its manual.

| Script | What it does |
|---|---|
| `check_links.py` | relative links and in-page `#` targets of the given Markdown files, by GitHub's slug rules |
| `check_refs.py` | every backticked module, symbol, class member, parameter field, flag and path in docstrings, comments and docs resolves in the tree; `check_snake.py` reuses its index to list the backticked `snake_case` tokens that name nothing |
| `check_docstrings.py` | escape, math-form, doubled-word, spaced-hyphen-dash and tone-word checks over the tracked Python tree; `lint_docs.py`, banned words, multi-line summaries and split Sphinx roles in named files |
| `banned_words.py` | the banned-word patterns the docstring, Markdown and commit-message checks share |
| `check_commit_message.py` | a commit message against the rules of the root `CONTRIBUTING.md` ("Commits"), as the commit-msg hook of `prek.toml`; `--commits` checks recorded commits, such as what a push would publish |
| `check_returns.py` | return-tuple arity and documented parameters against each signature, and a public function's `Parameters` section that omits one |
| `docs_only.py` | proves a change against a base ref touches docstrings, comments and `description=` strings only |
| `prose_only.py` | the same proof file by file and stricter: also the token stream, directive comments, shebangs, `description=` / `help=` strings and the docstrings read at run time; TOML data, comment-only YAML and shell edits, notebook code cells |
| `prose_atoms.py` | what a documentation change drops: the numbers, code spans, math, citations and quoted titles whose count fell, each traced to where it now lives, and the qualifier words lost per paragraph |
| `doc_pointers.py` | headings, titles, banners and numbering a change removes, with every line still quoting each; `--inbound`, the lines elsewhere that point into given files |
| `dup_prose.py` | pairs of repeated prose passages across the tree, most shared first |
| `md_lint.py` | the Markdown lines a change adds: width, Oxford spelling, banned words, inline math across a line break |
| `parse_cmds.py` | parse-validates one documented command line or TOML file without JAX |
| `check_render.py` | renders a Markdown file through GitHub's API (a request per file) and compares its math with the source |
| `oxford.py`, `reflow.py`, `underline.py` | rewrite documentation in place: Oxford spelling in the ranges `doc_ranges.py` finds (`--dry` previews), one reflowed paragraph, section underlines |
| `doc_inventory.py` | per-file dumps of docstrings and comments, written under the current directory, and the missing docstrings, printed |
| `crossref.py` | bibliographic metadata for a DOI or a free-text query, from Crossref |
