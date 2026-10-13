"""Shared pieces of the parameter-surface tests, core and lab.

Each script's cases record their verdicts through :func:`check` and
restore the ``params`` singleton through :func:`reset`; :func:`finish`
reports and gives the exit code.
"""

from __future__ import annotations

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: object = "") -> None:
    """Record one verdict, printing it (and *detail* on a failure)."""
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILURES.append(label)


def reset() -> None:
    """Fresh section models, no explicit-set tracking, extension defaults."""
    import dnsjax.parameters as P
    from dnsjax.extensions import reset_extensions

    fresh = P.Parameters()
    for section in P.Parameters.model_fields:
        setattr(P.params, section, getattr(fresh, section))
    P._user_set_fields.clear()
    P._materialized_defaults.clear()
    reset_extensions()


def finish(name: str) -> int:
    """Print the summary; the exit code."""
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"\nAll {name} tests passed.")
    return 0


def core_systems() -> set[str]:
    """The systems the core registers itself (no plugin's)."""
    from dnsjax.flows.triply_periodic.specs import SPECS as PERIODIC
    from dnsjax.flows.wall_bounded.specs import SPECS as WALL

    return {s.system for s in (*PERIODIC, *WALL)}
