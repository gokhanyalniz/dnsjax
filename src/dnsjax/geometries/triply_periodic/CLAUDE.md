# Triply-periodic geometry

- One module, `triply_periodic.py` (its docstring is the manual), and
  one flow, `flows/triply_periodic/monochromatic.py` (Kolmogorov).
- Helmholtz inversion is algebraic: `dnsjax.solvers` is never involved
  and `solver.backend` is absent from the periodic surfaces.
  Divergence is removed by projection (`correct_divergence`), not an
  influence matrix.
- `ly = 4` is the fixed length reference. `flows/triply_periodic/` and
  the IC builders import it, and `analysis/_core.LY_PERIODIC` mirrors
  it JAX-free: change them together.
- `y` is a Fourier axis here, oversampled like the others; `np0` needs
  `ny_padded` divisible, which a 7-smooth round-up provides
  (`parameters.round_up_padded_smooth`).
