# Documentation checks

Mechanical and read-only unless the row says otherwise. Each script's
module docstring is its manual.

| Script | What it does |
|---|---|
| `check_links.py` | relative links and in-page `#` targets of the given Markdown files, by GitHub's slug rules |
| `check_refs.py` | every backticked module, symbol, class member, parameter field, flag and path in docstrings, comments and docs resolves in the tree; `check_snake.py` reuses its index to list the backticked `snake_case` tokens that name nothing |
| `check_docstrings.py` | escape, math-form, doubled-word and tone-word checks over the tracked Python tree; `lint_docs.py` the same for named files, plus multi-line summaries and split Sphinx roles |
| `banned_words.py` | the banned-word patterns the docstring checks and the commit-message check share |
| `check_commit_message.py` | a commit message against the rules of the root `CONTRIBUTING.md` ("Commits"), as the commit-msg hook of `prek.toml`; `--commits` checks recorded commits, such as what a push would publish |
| `check_returns.py` | return-tuple arity and documented parameters against each signature |
| `docs_only.py` | proves a change against a base ref touches docstrings, comments and `description=` strings only |
| `parse_cmds.py` | parse-validates one documented command line or TOML file without JAX |
| `check_render.py` | renders a Markdown file through GitHub's API (a request per file) and compares its math with the source |
| `oxford.py`, `reflow.py`, `underline.py` | rewrite documentation ranges in place (`doc_ranges.py` finds them): Oxford spelling (`--dry` previews), one reflowed paragraph, section underlines |
| `doc_inventory.py` | per-file dumps of docstrings and comments and the public docstrings missing, written under the current directory |
| `crossref.py` | bibliographic metadata for a DOI or a free-text query, from Crossref |
