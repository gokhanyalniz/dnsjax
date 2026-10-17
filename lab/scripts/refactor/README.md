# Refactor regression check

A regression check for behaviour-preserving refactors. Each script's
module docstring is its manual.

| Script | What it does |
|---|---|
| `verify.py` | checks a commit out into a throwaway worktree, captures it, and compares the capture with a reference |
| `capture.py` | the behavioural fingerprint of a tree, through its public command lines only: every flow's `--help` and `--sample-toml`, short fixed-seed runs of every flow, scheme and start mode with variants, resumes, streams and transient growth, their outputs and HLO fingerprints |
| `compare.py` | diffs two captures, bitwise for arrays and streams, structurally for metadata; reviewed differences are allowed by pattern |
| `refingerprint.py` | recomputes a capture's HLO fingerprints from its kept dumps |
| `analysis_fp.py` | fingerprints the JAX-free analysis API on a capture's snapshots, and compares two fingerprints bitwise |
| `check_imports.py` | checks, without importing, that every `dnsjax` / `dnsjax_lab` import in the repository resolves |
| `prose_scan.py` | lists the lab's vocabulary in the core's prose, for review |
