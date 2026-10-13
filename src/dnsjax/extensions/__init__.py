r"""Extension parameter sections: the JAX-free registry and its plugins.

An *extension* owns a parameter section outside
:mod:`dnsjax.parameters`: the section is parsed from the ``dnsjax``
CLI (``--<name>.<field>``) and from ``parameters.toml``
(``[<name>]``), appears in ``--help`` for the flows it applies to,
is validated strictly (an extension section on a flow it does not
apply to is an error, like any other irrelevant parameter), is
recorded in snapshot metadata / sidecar JSON, and can be
trajectory-defining.  Modules and scripts register their sections via
:func:`register_extension`; the resolved values live on the
extension's ``values`` singleton (analogous to the global ``params``).
An entry point owns its section (the transient-growth CLI's ``[tg]``,
``scripts/snapshot_perturb.py``'s ``[perturb]``); an installed plugin
package registers its sections from a module it names under the
``dnsjax.extensions`` entry-point group, which this module imports
last, once every name such a module imports from it exists.

A section can take part in the solver's run as well:
:attr:`ParamExtension.runtime` names a factory of :class:`RunHooks`,
which :mod:`dnsjax.__main__` builds once the section is configured and
calls at fixed points of every iteration.

Import direction: this module never imports :mod:`dnsjax.parameters`
(the validate hooks receive the live ``params`` as an argument), so
``parameters -> flows.registry`` and ``bootstrap -> extensions`` stay
cycle-free.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points

from pydantic import BaseModel


@dataclass
class ParamExtension:
    """One registered extension parameter section.

    ``name`` is the TOML section, the CLI prefix (``--name.field``),
    and the metadata key; ``relevant(system)`` decides whether the
    section appears on a flow's surface (an irrelevant section is a
    strict error, like any other irrelevant parameter), except for the
    entry point that owns it, whose surfaces carry it on every flow
    (``own_extensions`` of :func:`dnsjax.bootstrap.resolve_parameters`);
    ``validate(values, params)`` runs with the global checks after
    the final configuration layer -- unconditionally, so it must
    itself reject a *configured* section on an unsupported system
    (see :func:`validate_extensions`); ``trajectory_defining`` folds
    the section into the resume trajectory comparison;
    ``record_in_metadata`` includes the resolved section in snapshot
    metadata / sidecar params dumps.  ``off_cadences`` names the
    section's optional cadences that ``0`` switches off -- turned back
    into unset by :func:`apply_extension_layer`, as
    :data:`dnsjax.parameters.OFF_CADENCES` are by the core merge --
    and ``reset_when_off`` makes a ``0`` there reset the **whole**
    section to its defaults, for an all-or-none section whose other
    knobs, inherited and impossible to unset, would otherwise fail its
    "set together" check.  ``values`` is the live merged singleton
    (analogous to the global ``params``).

    ``configured(values)`` says whether the section is switched on;
    ``None`` means it only parameterizes the entry point that owns it.
    Once it is, :func:`dnsjax.bootstrap.resolve_run_seeds` draws each
    unset field of ``seed_fields`` (and records it), unless
    ``seeds_at_start`` is false, when the owner resolves them itself
    (after a decision of its own, such as whether a resume is paired)
    and startup only records where a given value came from.  ``runtime``
    (``"module:attr"``) names a factory ``(state, t) -> RunHooks``,
    the section's part in the solver's run loop (:class:`RunHooks`),
    in a module imported only once the section is configured, so it
    may import JAX.
    """

    name: str
    model: type[BaseModel]
    relevant: Callable[[str], bool]
    #: One line for the ``--help`` section header (the model docstring
    #: is reference documentation, too long for help output).
    summary: str = ""
    validate: Callable[[BaseModel, object], None] | None = None
    trajectory_defining: bool = False
    record_in_metadata: bool = True
    off_cadences: tuple[str, ...] = ()
    reset_when_off: bool = False
    configured: Callable[[BaseModel], bool] | None = None
    seed_fields: tuple[str, ...] = ()
    seeds_at_start: bool = True
    runtime: str | None = None
    values: BaseModel = field(init=False)

    def __post_init__(self) -> None:
        self.values = self.model()


#: Registered extensions by section name (insertion order = surface /
#: help / metadata order).
EXTENSIONS: dict[str, ParamExtension] = {}

#: Section names owned by the core ``Parameters`` model; extensions
#: may not claim them.
_RESERVED_SECTIONS = frozenset(
    {
        "dist",
        "phys",
        "geo",
        "res",
        "init",
        "outs",
        "lowres",
        "step",
        "stop",
        "solver",
    }
)


def register_extension(ext: ParamExtension) -> ParamExtension:
    """Register *ext* (idempotent per name; a name clash is an error)."""
    if ext.name in _RESERVED_SECTIONS:
        raise ValueError(
            f"extension section {ext.name!r} clashes with a core "
            "parameter section"
        )
    existing = EXTENSIONS.get(ext.name)
    if existing is not None:
        if existing.model is not ext.model:
            raise ValueError(
                f"extension section {ext.name!r} is already registered "
                f"with a different model ({existing.model.__name__})"
            )
        return existing
    EXTENSIONS[ext.name] = ext
    return ext


def relevant_extensions(system: str) -> dict[str, ParamExtension]:
    """The registered extensions applying to *system*."""
    return {
        name: ext for name, ext in EXTENSIONS.items() if ext.relevant(system)
    }


def apply_extension_layer(overlays: dict[str, dict]) -> None:
    """Merge one configuration layer's extension sections.

    *overlays* maps section name -> ``exclude_unset``-style dict of
    explicitly-set fields; ``None`` values are skipped (mirroring the
    core merge loop: a layer cannot unset a field).  A ``0`` in one of
    a section's ``off_cadences`` is how a layer switches it off: it
    becomes unset, and under ``reset_when_off`` the whole section
    returns to its defaults.
    """
    for name, overlay in overlays.items():
        ext = EXTENSIONS[name]
        for key, value in overlay.items():
            if value is not None:
                setattr(ext.values, key, value)
        off = [
            key for key in ext.off_cadences if getattr(ext.values, key) == 0
        ]
        if off and ext.reset_when_off:
            _reset_values(ext)
        else:
            for key in off:
                setattr(ext.values, key, None)


def validate_extensions(params) -> None:
    """Run every extension's validate hook on the final configuration.

    Dispatched unconditionally -- not gated on ``relevant`` -- so a
    hook must tolerate an irrelevant system itself: return early when
    the section is unconfigured, and *reject* a configured section on
    a system the feature does not support.  (The CLI/toml surfaces
    never expose an irrelevant section, but direct assignment to the
    ``values`` singleton bypasses them, and silently skipping the
    check would let the runtime consume the stale config.)
    """
    for ext in EXTENSIONS.values():
        if ext.validate is not None:
            ext.validate(ext.values, params)


def extension_metadata(system: str) -> dict[str, dict]:
    """The recordable extension sections for *system*, resolved, as JSON.

    Merged into the ``params`` dump of snapshot metadata and sidecar
    JSON (:func:`dnsjax.param_surface.recorded_params_dump`).
    """
    return {
        name: ext.values.model_dump(mode="json")
        for name, ext in relevant_extensions(system).items()
        if ext.record_in_metadata
    }


def _reset_values(ext: ParamExtension) -> None:
    """Return one extension's ``values`` to its defaults, in place.

    In place because consumers hold references to the singleton (e.g.
    ``from dnsjax.extensions import force_params``), so it is never
    replaced.
    """
    fresh = ext.model()
    for name in ext.model.model_fields:
        setattr(ext.values, name, getattr(fresh, name))


def reset_extensions() -> None:
    """Reset every extension's ``values`` to defaults (test helper)."""
    for ext in EXTENSIONS.values():
        _reset_values(ext)


class RunHooks:
    r"""A configured extension's part in the solver's run loop.

    :mod:`dnsjax.__main__` builds one per relevant, configured
    extension that has a :attr:`ParamExtension.runtime` factory, as
    soon as the initial state is in place, and calls it at fixed points
    of the run, the extensions in registration order:

    1. :meth:`start` -- once, on the initial state (``it0``, ``t0``),
       right after the factory;
    2. :meth:`sample` -- at the top of each iteration, after the stats
       row and before the periodic snapshot, so on the state that
       snapshot records;
    3. :meth:`modify` -- after the periodic snapshot and the lowres
       write, immediately before the step: the one hook that may
       change the state, so every saved state precedes it;
    4. :meth:`finish` -- once after the loop, before the final
       snapshot;
    5. :meth:`flush` -- with every flush of the diagnostic streams
       (before each snapshot, after the first step, at shutdown and on
       a termination signal).

    Every default does nothing.  ``state`` is the solver-basis state
    the steppers carry.  A ``str`` returned by :meth:`start`,
    :meth:`sample`, :meth:`finish` or :meth:`flush` reports a
    non-finite value, and the run aborts with it (exit code 3), as on a
    non-finite core diagnostic.
    """

    def start(self, state, t: float) -> str | None:
        """Record on the initial state; a non-finite message or None."""
        return None

    def sample(self, it: int, state, t: float) -> str | None:
        """Record on the pre-step state; a non-finite message or None."""
        return None

    def modify(self, it: int, state, t: float):
        """The state to step from (the default: *state* unchanged)."""
        return state

    def finish(self, it: int, state, t: float) -> str | None:
        """Record on the final state; a non-finite message or None."""
        return None

    def flush(self, check: bool) -> str | None:
        """Write buffered records; checked, a non-finite message."""
        return None


def runtime_extensions(system: str) -> list[ParamExtension]:
    """The relevant, configured extensions of *system* with a runtime."""
    return [
        ext
        for ext in relevant_extensions(system).values()
        if ext.runtime is not None
        and ext.configured is not None
        and ext.configured(ext.values)
    ]


def build_run_hooks(ext: ParamExtension, state, t: float) -> RunHooks:
    """Import *ext*'s runtime factory and build its hooks on *state*."""
    module, _, attr = ext.runtime.partition(":")
    return getattr(import_module(module), attr)(state, t)


#: The entry-point group under which plugin packages register
#: extension sections: each names a module whose import registers them.
EXTENSIONS_ENTRY_POINT_GROUP = "dnsjax.extensions"


def _load_plugins() -> None:
    """Import the modules registered under the entry-point group.

    Last in this module, so a plugin module importing from it finds
    every name it needs; an import already under way (the plugin
    module imported first, importing this one) is left to finish.
    """
    found = entry_points(group=EXTENSIONS_ENTRY_POINT_GROUP)
    for ep in sorted(found, key=lambda e: e.name):
        try:
            ep.load()
        except Exception as exc:
            raise RuntimeError(
                f"extension plugin {ep.name!r} ({ep.value}) failed to "
                f"load: {exc}"
            ) from exc


_load_plugins()
