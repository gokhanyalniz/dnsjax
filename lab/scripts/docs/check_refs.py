"""Resolve backticked references in docstrings, comments and docs.

Flags references to modules, symbols, class members, parameter fields,
CLI flags and repo paths that do not exist in the current tree: every
tracked ``.py`` and ``.md`` file but the agent notes and this directory,
from the repository root this file sits in.  Importing the module builds the
index (``pyfiles``, ``mdfiles``, ``fields``, ``all_names``, ``top``,
``TICK``), which ``check_snake.py`` reuses.  Known false positives: a
plugin field the section models hold only with the lab installed, an
external name, an attribute written like a module path.

Usage::

    uv run python lab/scripts/docs/check_refs.py
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
tracked = subprocess.run(
    ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
).stdout.split()
# The checkers quote the patterns they look for: index them, never
# scan them.
SELF = "lab/scripts/docs/"
pyfiles = [f for f in tracked if f.endswith(".py")]
mdfiles = [
    f
    for f in tracked
    if f.endswith(".md")
    and not f.endswith("CLAUDE.md")
    and not f.startswith(".claude/")
]

# ---------------------------------------------------------------- index
modules = {}  # dotted -> file
for f in pyfiles:
    p = Path(f)
    if f.startswith("src/"):
        parts = list(p.with_suffix("").parts[1:])
    else:
        parts = list(p.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    modules[".".join(parts)] = f

top = defaultdict(set)  # file -> top-level names
members = defaultdict(set)  # (file, class) -> member names
all_names = set()
classes_anywhere = defaultdict(set)  # class name -> member names
for f in pyfiles:
    tree = ast.parse((ROOT / f).read_text())
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            top[f].add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        top[f].add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            top[f].add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                top[f].add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(
                    sub, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
                ):
                    top[f].add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            top[f].add(t.id)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        top[f].add((a.asname or a.name).split(".")[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            mem = set()
            for sub in node.body:
                if isinstance(
                    sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    mem.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            mem.add(t.id)
                elif isinstance(sub, ast.AnnAssign) and isinstance(
                    sub.target, ast.Name
                ):
                    mem.add(sub.target.id)
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in ("self", "cls")
                    and isinstance(sub.ctx, ast.Store)
                ):
                    mem.add(sub.attr)
            members[(f, node.name)] |= mem
            classes_anywhere[node.name] |= mem
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            all_names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            all_names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx, ast.Store
        ):
            all_names.add(node.attr)
        elif (
            isinstance(node, ast.arg)
            or isinstance(node, ast.keyword)
            and node.arg
        ):
            all_names.add(node.arg)

# parameter sections -> fields
SECTION_CLASS = {
    "dist": ("src/dnsjax/parameters.py", "Distribution"),
    "phys": ("src/dnsjax/parameters.py", "Physics"),
    "geo": ("src/dnsjax/parameters.py", "Geometry"),
    "res": ("src/dnsjax/parameters.py", "Resolution"),
    "init": ("src/dnsjax/parameters.py", "Initiation"),
    "outs": ("src/dnsjax/parameters.py", "Outputs"),
    "lowres": ("src/dnsjax/parameters.py", "Lowres"),
    "step": ("src/dnsjax/parameters.py", "TimeStepping"),
    "stop": ("src/dnsjax/parameters.py", "Termination"),
    "solver": ("src/dnsjax/parameters.py", "Solver"),
    "probes": ("lab/src/dnsjax_lab/extensions/__init__.py", "ProbesParams"),
    "force": ("lab/src/dnsjax_lab/extensions/__init__.py", "ForceParams"),
    "twin": ("lab/src/dnsjax_lab/twin/driver.py", "TwinParams"),
    "tg": ("src/dnsjax/analysis/transient_growth.py", "TGParams"),
    "recon": ("lab/scripts/twin_postprocess.py", "ReconParams"),
    "perturb": ("scripts/snapshot_perturb.py", "PerturbParams"),
    "calib": ("lab/scripts/random_ic_calibrate.py", "CalibParams"),
}
fields = {}
for sec, (f, cls) in SECTION_CLASS.items():
    fields[sec] = set(members[(f, cls)])
fields["res"] |= {"nr", "ntheta"}
fields["geo"] |= {"lz"}
fields["lowres"] |= {"nr", "ntheta"}
DERIVED = members[("src/dnsjax/parameters.py", "DerivedParameters")]
PADDED = members[("src/dnsjax/parameters.py", "PaddedResolution")]

EXTERNAL = {
    "jax",
    "jnp",
    "np",
    "numpy",
    "lax",
    "scipy",
    "pl",
    "plgpu",
    "pltpu",
    "pydantic",
    "ts",
    "tensorstore",
    "zarr",
    "plt",
    "matplotlib",
    "os",
    "sys",
    "math",
    "subprocess",
    "tarfile",
    "json",
    "time",
    "signal",
    "functools",
    "dataclasses",
    "typing",
    "collections",
    "itertools",
    "MPI",
    "mpi4py",
    "xla",
    "jaxlib",
    "triton",
    "tomllib",
    "pathlib",
    "Path",
    "argparse",
    "shutil",
    "tempfile",
    "struct",
    "hashlib",
    "datetime",
    "timedelta",
    "re",
    "io",
    "contextlib",
    "threading",
    "multiprocessing",
    "socket",
    "uv",
    "pytest",
    "ruff",
    "self",
    "cls",
    "mpl",
    "string",
    "random",
    "warnings",
    "traceback",
    "inspect",
    "importlib",
    "resource",
    "psutil",
    "gc",
    "atexit",
    "enum",
    "abc",
    "operator",
    "builtins",
    "ctypes",
    "platform",
    "logging",
    "csv",
    "glob",
    "fnmatch",
    "copy",
    "textwrap",
    "difflib",
    "numbers",
    "array",
    "mmap",
    "select",
    "errno",
    "stat",
    "weakref",
    "types",
    "heapq",
    "bisect",
    "statistics",
    "fractions",
    "decimal",
    "cmath",
    "pickle",
    "base64",
    "zlib",
    "gzip",
    "zipfile",
    "urllib",
    "http",
    "email",
    "xml",
    "html",
    "unittest",
    "doctest",
    "pdb",
    "cProfile",
    "pstats",
    "timeit",
    "sched",
    "queue",
    "asyncio",
    "concurrent",
    "fcntl",
    "termios",
    "tty",
    "pty",
    "pwd",
    "grp",
    "posix",
    "nt",
    "msvcrt",
    "winreg",
    "getpass",
    "locale",
    "gettext",
    "codecs",
    "unicodedata",
    "stringprep",
    "readline",
    "rlcompleter",
    "sqlite3",
    "dbm",
    "shelve",
    "marshal",
    "copyreg",
    "configparser",
    "netrc",
    "plistlib",
    "secrets",
    "hmac",
    "uuid",
    "ipaddress",
    "ssl",
    "selectors",
    "asyncore",
    "asynchat",
    "mailbox",
    "mimetypes",
    "binascii",
    "quopri",
    "uu",
    "webbrowser",
    "cgi",
    "wsgiref",
    "ftplib",
    "poplib",
    "imaplib",
    "smtplib",
    "socketserver",
    "xmlrpc",
    "audioop",
    "wave",
    "colorsys",
    "imghdr",
    "sndhdr",
    "curses",
    "turtle",
    "cmd",
    "shlex",
    "tkinter",
    "typing_extensions",
    "pyvista",
    "vtk",
    "dask",
    "xarray",
    "h5py",
    "flax",
    "optax",
    "equinox",
    "chex",
    "numba",
    "torch",
    "tf",
    "cupy",
    "jax_triton",
    "mosaic",
    "Mosaic",
    "ml_dtypes",
    "opt_einsum",
    "fsspec",
    "srun",
    "sbatch",
    "mpirun",
    "mpiexec",
    "nvidia",
    "cuda",
    "nccl",
    "gloo",
    "PRRTE",
    "OMPI",
    "SLURM",
    "PMIX",
    "PMI",
}


def resolve_dotted(ref, here):
    """Return None if resolved, else a reason string."""
    ref = ref.rstrip("()").rstrip(".")
    parts = ref.split(".")
    explicit = parts[0] == "dnsjax"
    if explicit:
        parts = parts[1:]
        if not parts:
            return None
    if (
        len(parts) >= 2
        and parts[0] in fields
        and (
            explicit
            or any(
                m == parts[0] + "." + parts[1]
                or m.endswith("." + parts[0] + "." + parts[1])
                for m in modules
            )
        )
    ):
        # a module path such as twin.driver
        pass
    elif len(parts) == 2 and parts[0] in fields:
        return None if parts[1] in fields[parts[0]] else "param field"
    if parts[0] in EXTERNAL or parts[0].startswith("__"):
        return None
    # params.* / derived_params.* / padded_res.*
    if parts[0] in ("params", "parameters") and len(parts) >= 2:
        if parts[1] in fields:
            if len(parts) == 2:
                return None
            return (
                None
                if parts[2] in fields[parts[1]]
                else f"param {parts[1]}.{parts[2]}"
            )
        if parts[0] == "params":
            return None
    if parts[0] == "derived_params" and len(parts) == 2:
        return None if parts[1] in DERIVED else "derived_params field"
    if parts[0] == "padded_res" and len(parts) == 2:
        return None if parts[1] in PADDED else "padded_res field"
    # module resolution: longest dotted prefix matching a module suffix
    for k in range(len(parts), 0, -1):
        prefix = ".".join(parts[:k])
        cands = [
            m
            for m in modules
            if m == prefix
            or m.endswith("." + prefix)
            or m.endswith("dnsjax." + prefix)
        ]
        if cands:
            rest = parts[k:]
            if not rest:
                return None
            for m in cands:
                f = modules[m]
                if (
                    f == "src/dnsjax/sharding.py"
                    and rest[0] in members[(f, "Sharding")]
                ):
                    return None
                if rest[0] in top[f]:
                    if len(rest) == 1:
                        return None
                    mem = members.get((f, rest[0]))
                    if mem is None:
                        return None  # attr of a non-class: can't check
                    if rest[1] in mem or rest[1] in all_names:
                        return None
                    return f"member {rest[0]}.{rest[1]} not in {f}"
                # submodule?
                sub = m + "." + rest[0]
                if sub in modules:
                    return resolve_dotted(".".join([sub] + rest[1:]), here)
            return f"name {rest[0]} not in {[modules[m] for m in cands]}"
    # class anywhere
    if parts[0] in classes_anywhere:
        if len(parts) == 1:
            return None
        if parts[1] in classes_anywhere[parts[0]] or parts[1] in all_names:
            return None
        return f"member {parts[1]} not on class {parts[0]}"
    # weak: last segment exists somewhere
    if parts[-1] in all_names or parts[-1] in classes_anywhere:
        return None
    return "unresolved"


PATH_RE = re.compile(
    r"^(?:\.\./)*[\w./{}*,-]+\.(?:py|md|toml|cff|yml|yaml|csv|svg|webp)$"
)


def resolve_path(ref, here):
    if "*" in ref or "{" in ref or "<" in ref:
        return None
    cands = [
        ROOT / ref,
        ROOT / "src" / ref,
        ROOT / "src/dnsjax" / ref,
        (ROOT / here).parent / ref,
        ROOT / "scripts" / ref,
        ROOT / "tests" / ref,
        ROOT / "docs" / ref,
        ROOT / "src/dnsjax/geometries" / ref,
        ROOT / "src/dnsjax/geometries/wall_bounded" / ref,
        ROOT / "src/dnsjax/flows" / ref,
        ROOT / "src/dnsjax/analysis" / ref,
        ROOT / "tests/response" / ref,
        ROOT / "lab/tests/response" / ref,
        ROOT / "lab/tests" / ref,
        ROOT / "lab/scripts" / ref,
        ROOT / "lab/src/dnsjax_lab" / ref,
        ROOT / "examples" / ref,
    ]
    if any(c.exists() for c in cands):
        return None
    # basename anywhere
    base = Path(ref).name
    if any(Path(t).name == base for t in tracked):
        return None
    return "missing path"


TICK = re.compile(r"(?::[a-z]+:)?``?([^`\n]+?)``?")
FLAG = re.compile(r"--([a-z]+)\.([a-z_0-9]+)")
DOTTED = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\(\))?$")
CALL = re.compile(r"^[A-Za-z_]\w*\(\)$")


def scan_text(text, here, lineno, out):
    for m in TICK.finditer(text):
        tok = m.group(1).strip()
        if tok.startswith("~"):
            tok = tok[1:]
        if " " in tok:
            # still check flags inside
            for fm in FLAG.finditer(tok):
                sec, fld = fm.groups()
                if sec in fields and fld not in fields[sec]:
                    out.append((here, lineno, tok, "flag field"))
            continue
        if re.search(
            r"\.(dat|bin|tar|json|npz|npy|txt|partial|so|dylib|h5|ipynb|svg|webp|csv|log)$",
            tok,
        ):
            continue
        if re.search(r"\.(py|md|toml|cff|yml|yaml)$", tok):
            r = resolve_path(tok, here)
        elif DOTTED.match(tok):
            r = resolve_dotted(tok, here)
        elif CALL.match(tok):
            r = (
                None
                if tok[:-2] in all_names or tok[:-2] in EXTERNAL
                else "call unresolved"
            )
        else:
            continue
        if r:
            out.append((here, lineno, tok, r))
    for fm in FLAG.finditer(text):
        sec, fld = fm.groups()
        if sec in fields and fld not in fields[sec]:
            out.append((here, lineno, fm.group(0), "flag field"))


def docstrings_and_comments(f):
    """``(text, line)`` of every docstring line and comment of file *f*."""
    src = (ROOT / f).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            b = node.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                for i, line in enumerate(b[0].value.value.splitlines()):
                    yield line, b[0].lineno + i
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.string, tok.start[0]


def main() -> int:
    out = []
    for f in pyfiles:
        if f.startswith(SELF):
            continue
        for text, lineno in docstrings_and_comments(f):
            scan_text(text, f, lineno, out)
    for f in mdfiles:
        if f.startswith(SELF):
            continue
        lines = (ROOT / f).read_text().splitlines()
        for i, line in enumerate(lines, 1):
            scan_text(line, f, i, out)
            # markdown links
            for m in re.finditer(r"\]\(([^)#\s]+)(#[^)]*)?\)", line):
                target = m.group(1)
                if target.startswith("http"):
                    continue
                if not ((ROOT / f).parent / target).exists():
                    out.append((f, i, target, "broken md link"))
    seen = set()
    for here, ln, tok, why in out:
        key = (here, tok, why)
        if key in seen:
            continue
        seen.add(key)
        print(f"{here}:{ln}: `{tok}` -- {why}")
    print(len(seen), "findings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
