r"""Random-seed resolution: OS entropy, provenance, and agreement.

Every random draw in the solver is seeded from a user-facing parameter
-- ``init.random_seed`` (the random initial condition,
:mod:`dnsjax.ic.random_field`), ``twin.seed`` (the ``dnsjax-twin``
partner perturbation) and ``force.seed`` (the ``[force]`` stochastic
kicks, :mod:`dnsjax.extensions.forcing`).  All three default to
**unset** (``None``), and this module defines what that means:

1. **Unset** -- a seed is drawn from the local operating-system entropy
   pool, printed with its source, and recorded in the run's metadata
   (the snapshot ``params`` dump, ``twin.json``, ``forcing.json``), so
   the run stays reproducible by passing the printed value back.
2. **Unset and no entropy source** -- the run refuses rather than
   silently falling back to a fixed value.  The refusal fires only for
   a seed the run would actually *draw* with: a laminar, localized-rolls
   or snapshot-resume start needs no entropy at all.
3. **Set** (command line, ``parameters.toml``, or inherited from a
   snapshot / ``twin.json``) -- used unchanged.

**Why a hard-coded default is wrong here.**  With one, an ensemble of
"independent" runs launched without an explicit seed is not an ensemble:
every member draws the same initial condition, the same partner and the
same kick sequence.  Reproducibility is not what the fixed default was
buying -- recording the drawn seed buys the same thing without pinning
every user to one realisation.

**Why the sentinel is ``None`` and not a pydantic ``default_factory``.**
Several sites compare a live value against a freshly constructed model
default -- the "stray knob" checks of the ``[force]`` and ``[twin]``
sections, and the absent-key fallback of
``parameters.trajectory_defining_changes`` -- and
``param_surface.render_sample_toml`` reads ``field_info.default``
directly.  A factory would draw a *new* seed at each of those, producing
spurious "set without the enabling ..." errors, spurious trajectory
changes on resume, and a ``PydanticUndefined`` in ``--sample-toml``.  A
plain ``None`` default compares equal to itself everywhere; the draw is
an explicit startup step (``bootstrap.resolve_run_seeds``) instead.

**Width and cross-process transport.**  A seed is
:data:`SEED_BITS`-bit, drawn with :func:`draw_seed`.  Under
``mpirun`` every rank must hold the **same** seed: the random-field
generator keys each mode's draw on ``(seed, global mode index)`` so the
field is identical at any ``(np0, np1)``, and per-rank draws would
assemble one field out of several unrelated streams -- divergence-free
and correctly normalised, but reproducible from no recorded seed at
all.  Process 0 therefore draws and the value is broadcast
(``bootstrap.resolve_seed``).  :func:`split_seed` / :func:`join_seed`
carry it as two 31-bit words so the payload is exact in ``int32``,
whether or not ``jax_enable_x64`` is on -- that follows
``res.double_precision``, and an ``int64`` payload would be silently
truncated in single precision.

This module is stdlib-only and JAX-free, so it is importable from the
parameter layer, the entry points and the offline scripts alike.
"""

import os

#: Width of a drawn seed.  Two 31-bit transport words (:func:`split_seed`)
#: give 62 usable bits -- enough that a campaign never sees a repeat,
#: while every intermediate stays a non-negative ``int32``.
SEED_BITS: int = 62

_WORD_BITS: int = SEED_BITS // 2
_WORD_MASK: int = (1 << _WORD_BITS) - 1
_SEED_MASK: int = (1 << SEED_BITS) - 1

#: Provenance labels, as printed after the resolved value.  A run's
#: banner carries one line per resolved seed, e.g.
#: ``init.random_seed = 3907558269744012891 (drawn from system entropy)``.
SOURCE_CLI = "set on the command line"
SOURCE_TOML = "set in parameters.toml"
SOURCE_SNAPSHOT = "inherited from the snapshot"
SOURCE_SIDECAR = "inherited from twin.json"
SOURCE_DRAWN = "drawn from system entropy"


class NoEntropySource(RuntimeError):
    """The platform offers no entropy pool to draw a seed from.

    Raised by :func:`draw_seed`; the entry points turn it into a
    ``SystemExit`` carrying :func:`missing_entropy_message`.
    """


def draw_seed() -> int:
    r"""A fresh :data:`SEED_BITS`-bit seed from the OS entropy pool.

    Returns
    -------
    :
        A non-negative integer below `$2^{62}$`.

    Raises
    ------
    NoEntropySource
        The platform has no entropy source.  ``os.urandom`` reports
        this as ``NotImplementedError`` when the interpreter was built
        without one and as ``OSError`` when the source exists but
        cannot be read (a sandbox denying ``/dev/urandom``).
        ``os.getrandom`` is deliberately not used: it is absent on some
        supported builds, and ``os.urandom`` blocks rather than failing
        while the pool initialises.
    """
    try:
        raw = os.urandom(8)
    except (NotImplementedError, OSError) as exc:
        raise NoEntropySource(str(exc) or type(exc).__name__) from exc
    return int.from_bytes(raw, "big") & _SEED_MASK


def split_seed(seed: int) -> tuple[int, int]:
    r"""Split *seed* into two 31-bit words for an ``int32`` payload.

    Parameters
    ----------
    seed:
        A seed in `$[0, 2^{62})$`, as returned by :func:`draw_seed`.

    Returns
    -------
    :
        ``(high, low)``, both non-negative and below `$2^{31}$`.
    """
    if not 0 <= seed <= _SEED_MASK:
        raise ValueError(
            f"seed {seed} is outside the {SEED_BITS}-bit transport "
            f"range [0, {_SEED_MASK}]."
        )
    return seed >> _WORD_BITS, seed & _WORD_MASK


def join_seed(high: int, low: int) -> int:
    """Reassemble the two words of :func:`split_seed`."""
    return (int(high) << _WORD_BITS) | int(low)


def seed_note(label: str, value: int | None, source: str) -> str:
    """One banner line reporting a resolved seed and where it came from.

    *label* is the user-facing parameter name (``"init.random_seed"``),
    *source* one of the ``SOURCE_*`` constants.
    """
    return f"{label} = {value} ({source})"


def missing_entropy_message(label: str, flag: str, reason: str) -> str:
    """The refusal text for an unset seed with no entropy available.

    *label* names the parameter (``"init.random_seed"``), *flag* the
    command-line form that fixes it, *reason* what the platform said.
    """
    section, _, key = label.partition(".")
    return (
        f"{label} is unset and this run needs it for a random draw, "
        f"but no system entropy source is available ({reason}).  Pass "
        f"{flag} <int> on the command line, or set '{key}' under "
        f"[{section}] in parameters.toml."
    )
