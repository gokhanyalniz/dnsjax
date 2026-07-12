r"""JAX-free probe-reader tests (``dnsjax.analysis.response.probes``).

Runs entirely on synthetic ``probes.bin``/``probes.json`` pairs
written by hand with NumPy (no solver, no JAX -- the import guard is
the first assertion), covering:

- the JAX-free import guarantee of the reader module;
- ``read_probes`` reconstruction (dtype from the sidecar, complex128
  upcast from ``<f4`` values, path forms: directory / ``.bin`` /
  ``.json``), the truncated-trailing-record drop, and the
  non-monotonic-timestamp warning path;
- ``mean_profile``: laminar-profile addition, tilt projection, and
  the ``t_min`` transient cut;
- ``re_tau`` against a closed-form total profile (Fornberg stencils
  are exact for the quadratic);
- ``write_profile_file`` round-trip and ascending-grid flip.

Usage::

    uv run python tests/response/test_probes_reader.py
"""

from __future__ import annotations

import sys

from dnsjax.analysis.response import probes as rp

assert "jax" not in sys.modules, (
    "importing dnsjax.analysis.response.probes must not import JAX"
)

import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

NY = 9
Y = np.cos(np.pi * np.arange(NY) / (NY - 1))  # CGL, descending
MODES = [[0, 0], [1, 0]]
RE = 100.0
#: Mean-mode streamwise perturbation: total = 1.25 (1 - y^2).
DELTA_U = 0.25 * (1.0 - Y**2)


def _write_stream(
    directory: str,
    t: np.ndarray,
    value_dtype: str = "<f8",
    tilt_degree: float = 0.0,
    truncate_bytes: int = 0,
) -> None:
    """Write a synthetic sidecar + binary with a known mean mode.

    Sample ``i`` holds ``(1 + i) * DELTA_U`` on the mean mode's
    streamwise component (so the ``t_min`` cut is observable) and a
    fixed complex profile on mode (1,0).
    """
    sidecar = {
        "format_version": 1,
        "modes": MODES,
        "wavenumbers": [[0, 0], [1, 0]],
        "n_components": 3,
        "component_labels": ["u_x", "u_y", "u_z"],
        "ny": NY,
        "wall_normal_grid": [float(v) for v in Y],
        "value_dtype": value_dtype,
        "it_probes": 1,
        "dt": 0.01,
        "system": "plane-poiseuille",
        "double_precision": value_dtype == "<f8",
        "git_hash": "synthetic",
        "params": {
            "phys": {"re": RE},
            "geo": {"tilt_degree": tilt_degree},
            "res": {"fd_order": 4},
        },
    }
    rec_dtype = np.dtype(
        [("t", "<f8"), ("u", value_dtype, (len(MODES), 3, NY, 2))]
    )
    rec = np.zeros(len(t), dtype=rec_dtype)
    rec["t"] = t
    rad = math.radians(tilt_degree)
    for i in range(len(t)):
        scale = 1.0 + i
        # Tilted-frame mean mode: (u_x, u_z) = U_s (cos, sin).
        rec["u"][i, 0, 0, :, 0] = scale * DELTA_U * math.cos(rad)
        rec["u"][i, 0, 2, :, 0] = scale * DELTA_U * math.sin(rad)
        rec["u"][i, 1, :, :, 0] = 0.25
        rec["u"][i, 1, :, :, 1] = -0.5
    with open(Path(directory) / "probes.json", "w") as f:
        json.dump(sidecar, f)
    raw = rec.tobytes()
    if truncate_bytes:
        raw += raw[:truncate_bytes]  # partial trailing record
    (Path(directory) / "probes.bin").write_bytes(raw)


def test_read_probes_paths_and_values() -> None:
    """Directory / .bin / .json path forms; values and dtypes."""
    t = np.array([0.0, 0.01, 0.02])
    with tempfile.TemporaryDirectory() as tmp:
        _write_stream(tmp, t, value_dtype="<f4")
        for path in (tmp, f"{tmp}/probes.bin", f"{tmp}/probes.json"):
            data = rp.read_probes(path)
            assert data.u.dtype == np.complex128
            assert data.u.shape == (3, 2, 3, NY)
            assert_allclose(data.t, t)
        assert_allclose(
            data.u[0, 0, 0].real, DELTA_U.astype(np.float32), rtol=0
        )
        assert_allclose(data.u[:, 1], 0.25 - 0.5j, rtol=1e-6)
        assert data.mode_index(1, 0) == 1
        try:
            data.mode_index(2, 0)
        except KeyError:
            pass
        else:
            raise AssertionError("missing mode was found")


def test_read_probes_truncated_and_nonmonotonic() -> None:
    """A partial trailing record is dropped; overlapping timestamps
    (a resumed re-run) only warn."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_stream(tmp, np.array([0.0, 0.01, 0.005]), truncate_bytes=100)
        data = rp.read_probes(tmp)  # prints two warnings
        assert data.t.shape[0] == 3


def test_read_probes_drops_resume_seams() -> None:
    """Exact-duplicate consecutive timestamps (continuation seams) are
    dropped keeping the last, restoring the uniform grid."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_stream(tmp, np.array([0.0, 0.01, 0.01, 0.02, 0.02]))
        data = rp.read_probes(tmp)  # prints the seam-drop note
        assert_allclose(data.t, [0.0, 0.01, 0.02], rtol=0)
        # "Keep the last": sample values scale with the write index,
        # so the survivors are writes 0, 2, 4.
        assert_allclose(
            data.u[:, 0, 0, 0].real, np.array([1.0, 3.0, 5.0]) * DELTA_U[0]
        )


def test_mean_profile_and_t_min() -> None:
    """Laminar addition, the transient cut, and the tilt projection."""
    t = np.array([0.0, 0.01, 0.02])
    with tempfile.TemporaryDirectory() as tmp:
        _write_stream(tmp, t, tilt_degree=30.0)
        data = rp.read_probes(tmp)
        # All samples: mean scale = (1 + 2 + 3)/3 = 2.
        y, u_s = rp.mean_profile(data)
        assert_allclose(y, Y)
        assert_allclose(u_s, (1.0 - Y**2) + 2.0 * DELTA_U, atol=1e-12)
        # t_min cuts the first two samples: scale = 3.
        _, u_s3 = rp.mean_profile(data, t_min=0.015)
        assert_allclose(u_s3, (1.0 - Y**2) + 3.0 * DELTA_U, atol=1e-12)
        try:
            rp.mean_profile(data, t_min=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("empty t_min window was accepted")


def test_re_tau_closed_form() -> None:
    r"""Closed-form check: the sample mean of scales (1, 2) is 1.5,
    so the total profile is `$(1 + 1.5/4)(1 - y^2)$` with wall slope
    `$|dU/dy|_w = 2.75$`, resolved exactly by the FD stencils on the
    quadratic; `$Re_\tau = \sqrt{Re \cdot 2.75}$`."""
    t = np.array([0.0, 0.01])
    with tempfile.TemporaryDirectory() as tmp:
        _write_stream(tmp, t)
        data = rp.read_probes(tmp)
        # mean scale 1.5 -> U = (1 + 1.5/4)(1 - y^2); |dU/dy|_w = 2.75.
        assert_allclose(rp.re_tau(data), math.sqrt(RE * 2.75), rtol=1e-12)


def test_write_profile_file() -> None:
    """Two-column output, descending grid, ascending input flipped."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "U.txt"
        rp.write_profile_file(path, Y, 1.0 - Y**2)
        out = np.loadtxt(path)
        assert_allclose(out[:, 0], Y)
        assert_allclose(out[:, 1], 1.0 - Y**2)

        rp.write_profile_file(path, Y[::-1], (1.0 - Y**2)[::-1])
        out2 = np.loadtxt(path)
        assert_allclose(out2, out)  # flipped back to descending

        try:
            rp.write_profile_file(path, Y, np.zeros((2, 2)))
        except ValueError:
            pass
        else:
            raise AssertionError("shape mismatch was accepted")


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    assert "jax" not in sys.modules, "a test pulled JAX in"
    print(f"\nAll {len(tests)} tests passed.")
