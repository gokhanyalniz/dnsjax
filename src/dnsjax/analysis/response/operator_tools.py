r"""Per-mode linear-operator tools on transient-growth exports.

Post-processing for the ``--tg.save_operator`` bundles written by the
transient-growth CLI (``<stem>_tg_op.npz``; storage layout and
coordinate contract: the ``_write_operator_npz`` docstring in
:mod:`dnsjax.analysis.transient_growth`): controllability Gramians
and modes, energy growth curves of **arbitrary** operator matrices
(exported, restricted, or identified), Galerkin restriction, and the
full-state lift used to inject basis vectors with
``scripts/snapshot_perturb.py``.

Coordinates
===========
Every operator here lives in the export's **energy-orthonormal**
coordinates: a state ``a`` (length ``r_res``) satisfies
`$\lVert a\rVert_2^2 = q^H \mathrm{diag}(w)\, q$` for the full state
`$q = T_\mathrm{lift}\, a$`; conversely `$a = T_\mathrm{proj}\, q$`
(:class:`OperatorData` precomputes both maps).  Hence the plain
matrix 2-norm *is* the energy norm: `$G(t) = \lVert e^{tA}
\rVert_2^2$` (:func:`growth_curve`), a Galerkin restriction onto
orthonormal columns preserves the norm (:func:`restrict`), and the
controllability Gramian with unit-covariance forcing in the energy
inner product is simply the Lyapunov solution of `$(A, I)$`
(:func:`controllability_gramian`).

JAX and SciPy
=============
The dense time sweeps (:func:`growth_curve`,
:func:`input_response_curve`) run batched ``expm`` + SVD on the JAX
default device -- GPU-capable; enable float64 first
(``bootstrap.configure_jax_platform(..., double_precision=True)``, or
``jax.config.update("jax_enable_x64", True)``); both raise otherwise.
JAX is imported inside those functions only, so importing this module
(and the Gramian path, which is NumPy/SciPy) stays JAX-free.  SciPy is
a core dependency but is likewise imported lazily; the Lyapunov solve
keeps an eigendecomposition closed form as a fallback.

CLI
===
Compute and save leading controllability modes (the injection basis
for ensemble response experiments)::

    python -m dnsjax.analysis.response.operator_tools \
        --operator U_mean_tg_op.npz --n-modes 30 --out U_mean_cont.npz

writes ``profiles_{i2}_{i3}`` (``(m, C, Ny)`` full-state profiles,
unit energy norm, consumable by ``snapshot_perturb.py
--modes-npz``) and ``gram_eigvals_{i2}_{i3}`` per mode present in the
operator bundle (or a ``--modes "i2,i3;..."`` subset).  NumPy/SciPy
only -- no JAX, no device.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "OperatorData",
    "available_modes",
    "load_operator",
    "controllability_gramian",
    "controllability_modes",
    "lift_modes",
    "load_modes_npz",
    "recover_basis",
    "save_modes_npz",
    "growth_curve",
    "input_response_curve",
    "restrict",
]


# ── Loading ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class OperatorData:
    """One mode's exported operator, with the coordinate maps.

    ``A`` (``r_res x r_res``) is the reduced generator in
    energy-orthonormal coordinates; ``lam = eig(A)``.  ``T_proj``
    (``r_res x n``) maps a full component-major state vector
    (``c*Ny + j``) to those coordinates; ``T_lift`` (``n x r_res``)
    is its right inverse onto the resolved subspace.  ``meta`` holds
    the bundle's global keys (system, grids, provenance JSON).
    """

    i2: int
    i3: int
    A: np.ndarray
    lam: np.ndarray
    Q: np.ndarray
    F: np.ndarray
    V: np.ndarray
    T_proj: np.ndarray
    T_lift: np.ndarray
    w_diag: np.ndarray
    y: np.ndarray
    system: str
    family: str
    k_metric: float
    volume_fac: float
    meta: dict


def available_modes(path: str | Path) -> list[tuple[int, int]]:
    """The ``(i2, i3)`` modes present in an operator bundle."""
    with np.load(path) as npz:
        return [
            (int(a), int(b))
            for a, b in zip(npz["mode_i2"], npz["mode_i3"], strict=True)
        ]


def load_operator(path: str | Path, i2: int, i3: int) -> OperatorData:
    """Load one mode's operator from a ``<stem>_tg_op.npz`` bundle."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        modes = list(
            zip(npz["mode_i2"].tolist(), npz["mode_i3"].tolist(), strict=True)
        )
        if (i2, i3) not in modes:
            raise KeyError(
                f"mode ({i2},{i3}) is not in {path} (modes: {modes})"
            )
        row = modes.index((i2, i3))
        sfx = f"_{i2}_{i3}"
        a_mat = np.asarray(npz["A" + sfx])
        q_mat = np.asarray(npz["Q" + sfx])
        f_mat = np.asarray(npz["F" + sfx])
        v_mat = np.asarray(npz["V" + sfx])
        lam = np.asarray(npz["lam" + sfx])
        meta = {
            k: npz[k]
            for k in (
                "readme",
                "system",
                "family",
                "params_json",
                "tg_config_json",
                "profile_file",
                "component_labels",
                "code_grid",
                "profile_on_grid",
                "energy_weights",
                "tg_dt",
                "volume_fac",
                "t_grid",
            )
        }
        k_metric = float(npz["mode_k_metric"][row])
    t_proj = q_mat.conj().T @ f_mat @ v_mat.conj().T
    t_lift = v_mat @ np.linalg.solve(f_mat, q_mat)
    return OperatorData(
        i2=i2,
        i3=i3,
        A=a_mat,
        lam=lam,
        Q=q_mat,
        F=f_mat,
        V=v_mat,
        T_proj=t_proj,
        T_lift=t_lift,
        w_diag=np.asarray(meta["energy_weights"], dtype=float),
        y=np.asarray(meta["code_grid"], dtype=float),
        system=str(meta["system"]),
        family=str(meta["family"]),
        k_metric=k_metric,
        volume_fac=float(meta["volume_fac"]),
        meta=meta,
    )


# ── Controllability ──────────────────────────────────────────────


def _gramian_eig_closed_form(a: np.ndarray) -> np.ndarray:
    r"""Lyapunov solution via the eigendecomposition closed form.

    With `$A = E\Lambda E^{-1}$` and `$\tilde{X} = E^{-1} X E^{-H}$`,
    `$AX + XA^H = -I$` becomes `$\tilde{X}_{ij} = -(E^{-1}E^{-H})_{ij}
    / (\lambda_i + \bar{\lambda}_j)$`.  SciPy-free fallback; less
    robust than Bartels-Stewart for near-defective `$A$`.
    """
    e_val, e_vec = np.linalg.eig(a)
    g = np.linalg.inv(e_vec)
    x_t = -(g @ g.conj().T) / (e_val[:, None] + np.conj(e_val)[None, :])
    return e_vec @ x_t @ e_vec.conj().T


def controllability_gramian(a: np.ndarray) -> np.ndarray:
    r"""Controllability Gramian `$X$`: `$AX + XA^H + I = 0$`.

    In energy-orthonormal coordinates, with white forcing of unit
    covariance in the energy inner product, `$X$` is both the
    (infinite-horizon) controllability Gramian and the steady forced
    covariance.  Requires a stable `$A$`.  Uses SciPy's
    Bartels-Stewart solver when available, else the eigendecomposition
    closed form; the result is Hermitian-symmetrised.
    """
    a = np.asarray(a)
    abscissa = float(np.max(np.linalg.eigvals(a).real))
    if abscissa >= 0.0:
        raise ValueError(
            f"A is not stable (spectral abscissa {abscissa:+.3e}); "
            "the infinite-horizon Gramian does not exist"
        )
    try:
        from scipy.linalg import solve_continuous_lyapunov
    except ImportError:
        x = _gramian_eig_closed_form(a)
    else:
        x = solve_continuous_lyapunov(a, -np.eye(a.shape[0], dtype=complex))
    return 0.5 * (x + x.conj().T)


def controllability_modes(
    a: np.ndarray, m: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Leading controllability modes of the generator ``a``.

    Returns ``(gram_eigvals, P)``: all Gramian eigenvalues in
    descending order, and the leading ``m`` orthonormal eigenvectors
    as columns of ``P`` (``r_res x m``; ``m = r_res`` when ``None``).
    Orthonormal in the energy inner product by construction.

    Pick ``m`` from the eigenvalue decay: the discarded tail's share
    of ``sum(gram_eigvals)`` is the neglected fraction of the
    white-forced steady variance, and downstream results (restricted
    growth curves, identified operators) should be insensitive to
    raising ``m`` further.  Each kept mode costs one ensemble tree in
    the direct-identification pipeline, so keep ``m`` as small as
    that insensitivity allows.
    """
    x = controllability_gramian(a)
    vals, vecs = np.linalg.eigh(x)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    if m is not None:
        if not (1 <= m <= vecs.shape[1]):
            raise ValueError(f"m = {m} out of range (1..{vecs.shape[1]})")
        vecs = vecs[:, :m]
    return vals, vecs


def lift_modes(op: OperatorData, p: np.ndarray) -> np.ndarray:
    """Lift basis columns to full-state profiles ``(m, C, Ny)``.

    Each column of *p* (energy-orthonormal coordinates) becomes a
    ``(C, Ny)`` complex profile in the stored component basis with
    unit energy norm -- directly consumable by
    ``snapshot_perturb.py`` (which rescales by its amplitude
    convention anyway).
    """
    ny = op.y.shape[0]
    full = (op.T_lift @ p).T  # (m, n)
    return full.reshape(full.shape[0], -1, ny)


def load_modes_npz(
    path: str | Path, i2: int, i3: int, n: int | None = None
) -> np.ndarray:
    """Lifted mode profiles from a :func:`save_modes_npz` bundle.

    Returns the ``profiles_{i2}_{i3}`` array (``(m, C, Ny)``
    full-state profiles), truncated to the leading *n* when given.
    Shared loader for every consumer of an injection basis (the
    ensemble / LIM / SSI identification and the runtime forcing).
    """
    path = Path(path)
    key = f"profiles_{i2}_{i3}"
    with np.load(path, allow_pickle=False) as npz:
        if key not in npz:
            have = [k for k in npz.files if k.startswith("profiles")]
            raise KeyError(f"{path} has no {key!r} (available: {have})")
        arr = np.asarray(npz[key])
    if n is not None:
        if not (1 <= n <= arr.shape[0]):
            raise ValueError(
                f"n = {n} out of range ({key} holds {arr.shape[0]} modes)"
            )
        arr = arr[:n]
    return arr


def recover_basis(
    op: OperatorData, lifted: np.ndarray, atol: float = 1e-8
) -> np.ndarray:
    r"""Recover the orthonormal basis `$P$` from lifted profiles.

    *lifted* is ``(m, C, Ny)`` full-state profiles that were produced
    as `$T_\mathrm{lift} p_j$` (e.g. :func:`lift_modes` output stored
    by :func:`save_modes_npz`); since `$T_\mathrm{proj}
    T_\mathrm{lift} = I$`, projecting them recovers the coordinate
    columns exactly.  Returns `$P$` (``r_res x m``) and raises when
    `$P^H P$` deviates from the identity beyond *atol* -- the
    telltale of pairing the profiles with the wrong operator bundle
    (or a regridded / hand-altered basis).
    """
    lifted = np.asarray(lifted)
    m = lifted.shape[0]
    p = (lifted.reshape(m, -1) @ op.T_proj.T).T  # (r_res, m)
    gram_err = float(np.max(np.abs(p.conj().T @ p - np.eye(m))))
    if gram_err > atol:
        raise ValueError(
            f"basis recovery failed (max |P^H P - I| = {gram_err:.2e}); "
            "the profiles do not match this operator bundle"
        )
    return p


def save_modes_npz(
    path: str | Path,
    ops: list[OperatorData],
    n_modes: int,
    operator_file: str | Path,
) -> None:
    """Compute + save leading controllability modes for each *op*.

    Writes ``profiles_{i2}_{i3}`` (``(m, C, Ny)`` lifted profiles)
    and ``gram_eigvals_{i2}_{i3}`` per operator, with the grid /
    system keys ``snapshot_perturb.py --perturb.modes_npz`` checks.
    """
    if not ops:
        raise ValueError("no operators given")
    out: dict[str, Any] = {
        "readme": (
            "dnsjax controllability modes. profiles_{i2}_{i3}: "
            "(m, C, Ny) full-state profiles (stored component "
            "basis, unit energy norm), the leading eigenvectors of "
            "the controllability Gramian of the exported reduced "
            "generator; gram_eigvals_{i2}_{i3}: all Gramian "
            "eigenvalues, descending."
        ),
        "system": ops[0].system,
        "family": ops[0].family,
        "code_grid": ops[0].y,
        "component_labels": ops[0].meta["component_labels"],
        "operator_file": str(operator_file),
        "params_json": ops[0].meta["params_json"],
        "n_modes": int(n_modes),
        "mode_i2": np.asarray([op.i2 for op in ops]),
        "mode_i3": np.asarray([op.i3 for op in ops]),
    }
    for op in ops:
        vals, p = controllability_modes(op.A, n_modes)
        sfx = f"_{op.i2}_{op.i3}"
        out["profiles" + sfx] = lift_modes(op, p)
        out["gram_eigvals" + sfx] = vals
    np.savez(path, **out)
    print(f"[operator_tools] wrote {path}")


# ── Growth curves (JAX; batched expm + SVD) ──────────────────────


def _require_x64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "growth curves need float64: call bootstrap."
            "configure_jax_platform(..., double_precision=True) or "
            'jax.config.update("jax_enable_x64", True) first'
        )


def growth_curve(
    a: np.ndarray, ts: np.ndarray, t_chunk: int = 16
) -> np.ndarray:
    r"""Optimal energy growth `$G(t) = \lVert e^{tA}\rVert_2^2$`.

    For **any** generator matrix in energy-orthonormal coordinates
    (an exported ``A``, a :func:`restrict` restriction, or an
    identified operator).  Batched ``expm`` + SVD on the JAX default
    device, ``t_chunk`` horizons at a time -- a batch-size/memory
    knob only (device memory scales with ``t_chunk * r_res**2``):
    raise it for throughput on long time grids with small operators,
    lower it if a large ``r_res`` runs out of device memory.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm

    _require_x64()
    a_dev = jnp.asarray(np.asarray(a, dtype=complex))

    @jax.jit
    def _chunk(ts_dev: Any) -> Any:
        mats = jax.vmap(lambda t: expm(t * a_dev))(ts_dev)
        return jnp.linalg.svd(mats, compute_uv=False)[:, 0] ** 2

    ts = np.asarray(ts, dtype=float)
    out = [
        np.asarray(_chunk(jnp.asarray(ts[lo : lo + t_chunk])))
        for lo in range(0, len(ts), t_chunk)
    ]
    return np.concatenate(out) if out else np.zeros(0, dtype=float)


def input_response_curve(
    a: np.ndarray, a0: np.ndarray, ts: np.ndarray, t_chunk: int = 16
) -> np.ndarray:
    r"""Energy growth of one input: `$\lVert e^{tA}a_0\rVert_2^2 /
    \lVert a_0\rVert_2^2$`.

    The *actual* predicted growth of a specific injected input --
    :func:`growth_curve` is only its envelope, and the two agree at
    the optimal input/horizon pair alone.  Same ``t_chunk`` batching
    knob as :func:`growth_curve`.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm

    _require_x64()
    a_dev = jnp.asarray(np.asarray(a, dtype=complex))
    a0_dev = jnp.asarray(np.asarray(a0, dtype=complex))

    @jax.jit
    def _chunk(ts_dev: Any) -> Any:
        vecs = jax.vmap(lambda t: expm(t * a_dev) @ a0_dev)(ts_dev)
        return jnp.sum(jnp.abs(vecs) ** 2, axis=1)

    ts = np.asarray(ts, dtype=float)
    norm0 = float(np.sum(np.abs(a0) ** 2))
    if norm0 == 0.0:
        raise ValueError("a0 is zero")
    out = [
        np.asarray(_chunk(jnp.asarray(ts[lo : lo + t_chunk])))
        for lo in range(0, len(ts), t_chunk)
    ]
    return np.concatenate(out) / norm0 if out else np.zeros(0, dtype=float)


def restrict(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    r"""Galerkin restriction `$P^H A P$` onto orthonormal columns.

    With *p* orthonormal (e.g. :func:`controllability_modes` output)
    the restricted operator lives in the same energy-orthonormal
    convention, so :func:`growth_curve` applies unchanged.
    """
    p = np.asarray(p)
    gram_err = float(np.max(np.abs(p.conj().T @ p - np.eye(p.shape[1]))))
    if gram_err > 1e-10:
        raise ValueError(
            f"restriction columns are not orthonormal (max "
            f"|P^H P - I| = {gram_err:.2e})"
        )
    return p.conj().T @ np.asarray(a) @ p


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Controllability-mode export CLI (see the module docstring)."""
    p = argparse.ArgumentParser(
        prog="python -m dnsjax.analysis.response.operator_tools",
        description="Controllability modes from a transient-growth "
        "operator bundle.",
        allow_abbrev=False,
    )
    p.add_argument("--operator", required=True, help="<stem>_tg_op.npz bundle")
    p.add_argument(
        "--modes",
        default="all",
        help='"all" (every mode in the bundle) or "i2,i3;..."',
    )
    p.add_argument(
        "--n-modes",
        type=int,
        default=30,
        help="leading controllability modes kept per mode (pick from "
        "the gram_eigvals decay; see controllability_modes)",
    )
    p.add_argument("--out", required=True, help="output npz path")
    args = p.parse_args(argv)

    have = available_modes(args.operator)
    if args.modes.strip() == "all":
        wanted = have
    else:
        from ...harmonics import parse_mode_pairs

        wanted = parse_mode_pairs(args.modes)
        missing = [m for m in wanted if m not in have]
        if missing:
            raise SystemExit(
                f"modes {missing} are not in {args.operator} "
                f"(available: {have})"
            )
    ops = [load_operator(args.operator, i2, i3) for i2, i3 in wanted]
    n_modes = min(args.n_modes, min(op.A.shape[0] for op in ops))
    if n_modes != args.n_modes:
        print(
            f"[operator_tools] n-modes reduced to {n_modes} (the "
            "smallest resolved rank)."
        )
    save_modes_npz(args.out, ops, n_modes, args.operator)
    for op in ops:
        print(
            f"  mode ({op.i2},{op.i3}): r_res = {op.A.shape[0]}, "
            f"spectral abscissa {float(np.max(op.lam.real)):+.4e}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(None))
