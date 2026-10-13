r"""The ``[hookcheck]`` section of ``test_run_hooks.py``, and its hooks.

The core's own sections take no part in the run loop, so the test
registers this one: the solver child it launches imports this module
before the parameters resolve, which registers the section, and the
section's ``runtime`` names :class:`Recorder` back here.  Once
``hookcheck.on`` is set, each hook appends one JSON line to
:data:`LOG` in the run directory: the hook's name, ``it``, ``t`` and
the squared norm of the state it was handed.  ``hookcheck.scale_at``
makes :meth:`Recorder.modify` halve the state at that iteration, and
``hookcheck.fail_at`` makes :meth:`Recorder.sample` report a
non-finite value there.

JAX is imported inside the hooks only: the module is imported before
the solver configures JAX.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from dnsjax.extensions import ParamExtension, RunHooks, register_extension

#: The hook log, one JSON object per line, in the run directory.
LOG = "hooks.jsonl"


class HookCheck(BaseModel):
    """The test section: on, and where to modify or fail."""

    on: bool = False
    scale_at: int | None = None
    fail_at: int | None = None


EXTENSION = register_extension(
    ParamExtension(
        name="hookcheck",
        model=HookCheck,
        relevant=lambda system: True,
        summary="The run-hook test's section.",
        configured=lambda values: values.on,
        runtime="_run_hooks:Recorder",
    )
)


class Recorder(RunHooks):
    """Log every call; modify or fail where the section says."""

    def __init__(self, state, t: float) -> None:
        self.values = EXTENSION.values
        self._log("init", None, t, state)

    def _log(self, hook: str, it: int | None, t, state=None) -> None:
        row = {"hook": hook, "it": it, "t": None if t is None else float(t)}
        if state is not None:
            import jax.numpy as jnp

            row["norm2"] = float(jnp.vdot(state, state).real)
        with open(LOG, "a") as f:
            f.write(json.dumps(row) + "\n")

    def start(self, state, t: float) -> str | None:
        self._log("start", None, t, state)
        return None

    def sample(self, it: int, state, t: float) -> str | None:
        self._log("sample", it, t, state)
        if it == self.values.fail_at:
            return f"hookcheck: a non-finite test value at it = {it}"
        return None

    def modify(self, it: int, state, t: float):
        self._log("modify", it, t, state)
        if it == self.values.scale_at:
            return 0.5 * state
        return state

    def finish(self, it: int, state, t: float) -> str | None:
        self._log("finish", it, t, state)
        return None

    def flush(self, check: bool) -> str | None:
        self._log("flush", None, None)
        return None
