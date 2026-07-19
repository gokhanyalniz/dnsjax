r"""Input-output / response analysis tools for dnsjax runs.

Submodules (import them explicitly; this ``__init__`` stays empty so
``import dnsjax.analysis`` never pulls JAX or SciPy):

- :mod:`dnsjax.analysis.response.probes` -- JAX-free (NumPy) reader
  for the runtime spectral-mode probe stream
  (``probes.bin``/``probes.json``; written by :mod:`dnsjax.probes`),
  plus mean-profile / friction-Reynolds-number helpers.
- :mod:`dnsjax.analysis.response.operator_tools` -- per-mode linear
  operators exported by the transient-growth CLI
  (``--tg.save_operator``): controllability Gramian / modes, growth
  curves, subspace restriction.  JAX-based (GPU-capable), SciPy
  imported lazily.
- :mod:`dnsjax.analysis.response.ensemble` -- ensemble-response
  aggregation and direct operator identification from injected-mode
  response data.  JAX-based, SciPy imported lazily.
- :mod:`dnsjax.analysis.response.lim` -- linear inverse modeling:
  the same operator identified from lagged covariances of a plain
  *unforced* probe stream (no extra runs; assumes white-in-time
  turbulent forcing).  JAX-based, SciPy imported lazily.
- :mod:`dnsjax.analysis.response.ssi` -- stochastic-forcing
  identification: reader for the runtime kick log
  (``forcing.bin``, :mod:`dnsjax.forcing`) and the kick/response
  cross-covariance fit (no whiteness hypothesis on the background;
  needs a ``[force]``-enabled run).  JAX-based, SciPy imported
  lazily.

Pipeline
========
The full workflow from a turbulent run to a data-driven linear
operator, in order (per-step detail and knob guidance: the named
docstrings):

1. **Probe the run.**  Add ``--probes.modes "0,0;3,0"
   --probes.it_probes 10`` to a DNS run; the listed modes'
   wall-normal profiles `$\hat{u}(y, t)$` stream to ``probes.bin``
   (:mod:`dnsjax.probes`).
2. **Turbulent mean.**  :func:`~dnsjax.analysis.response.probes.
   mean_profile` + ``write_profile_file`` turn the ``(0,0)`` probe
   into a total mean-profile file (cut the transient with ``t_min``;
   sanity-check ``re_tau``).
3. **Linear operator about the mean.**  ``python -m
   dnsjax.analysis.transient_growth --tg.profile mean.txt
   --tg.modes ... --tg.save_operator True``: optimal energy growth
   `$G(t)$` per mode plus the reduced-generator bundle
   ``<stem>_tg_op.npz``.
4. **Injection basis.**  The :mod:`.operator_tools` CLI computes the
   leading controllability modes of each exported generator -- the
   most excitable directions, the natural basis for response
   experiments -- as full-state profiles ready for injection.
5. **Ensemble impulse responses.**  ``scripts/ensemble_setup.py``
   ``harvest``/``build`` seed antithetic member pairs from harvested
   snapshots (via ``scripts/snapshot_perturb.py``); run the members;
   the :mod:`.ensemble` CLI ``aggregate`` yields the
   ensemble-averaged response `$\langle\hat{u}\rangle(t)$` and
   compares its energy with the linear prediction and the `$G(t)$`
   envelope.
6. **Direct identification.**  Repeat step 5 once per basis index,
   then ``ensemble identify`` assembles propagator samples
   `$M(\tau) \approx e^{\tau L}$` from the responses and fits the
   generator `$L$` (matrix logarithm), reporting its spectrum and
   growth curve against the restricted reference operator.

Steps 1-3 stand alone (turbulent statistics; optimal growth about a
measured mean); steps 4-6 need the earlier outputs.  Two
**alternative identification routes** replace steps 5-6 on the same
basis, coordinates, and output convention (so all three operators
are directly comparable): :mod:`.lim` needs only the step-1 probe
stream of the plain run (cheapest; adds the whiteness hypothesis),
and :mod:`.ssi` needs one run re-run with the ``[force]`` stochastic
kicks (one experiment instead of an ensemble; hypothesis-free on the
background).  Cost/assumption trade-offs: the two module docstrings.

Unlike the rest of :mod:`dnsjax.analysis`, modules here **may** use
JAX (they are never imported from ``analysis/__init__.py``, so the
package-level JAX-free import guarantee is unaffected; the
``transient_growth`` precedent).  CLIs select the platform via
``bootstrap.configure_jax_platform`` before any JAX import.

The JAX/NumPy line is drawn by profitability, not habit: the dense
time sweeps (batched ``expm``/SVD growth curves) run in JAX and are
GPU-capable; matrix factorisations (``logm``, the Lyapunov solves,
non-symmetric ``eig``) stay SciPy/LAPACK, which JAX has no (GPU)
kernels for; and the stream projections / covariance estimators
stay NumPy BLAS deliberately -- the identified coordinates are
small (`$m \lesssim 30$`) and the wall time is dominated by reading
the multi-GB probe streams.  Revisit that last choice only if
streams reach `$O(10^7)$` samples.
"""
