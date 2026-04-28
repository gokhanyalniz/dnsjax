#!/usr/bin/env python3
"""Display SPIKE block-partition trade-offs for a given resolution.

Shows every valid block partition (P, m) for the banded solver,
together with per-operator memory, arithmetic intensity of the
per-block LU, and a qualitative GPU-efficiency rating.  The
memory-optimal row (the automatic default) is marked with ``>>>``.

Usage::

    python scripts/spike_partition_info.py --nx 128 --ny 128 --nz 128
    python scripts/spike_partition_info.py \\
        --ny 256 --nz 256 --nx 512 --fd-order 6
"""

from __future__ import annotations

import argparse
import sys


def _divisors_of(n: int) -> list[int]:
    divs: list[int] = []
    for i in range(1, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs


def _format_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.1f} KB"


def _gpu_rating(ai: float) -> str:
    """Qualitative GPU rating from arithmetic intensity."""
    if ai < 0.5:
        return "poor"
    if ai < 2.0:
        return "fair"
    if ai < 8.0:
        return "good"
    return "excellent"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SPIKE block-partition trade-off table.",
    )
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--fd-order", type=int, default=4, dest="fd_order")
    ap.add_argument(
        "--precision",
        choices=["single", "double"],
        default="double",
    )
    args = ap.parse_args()

    Ny = args.ny
    p = args.fd_order
    Nkz = args.nz - 1
    Nkx = args.nx // 2
    n_modes = Nkz * Nkx
    bpe = 16 if args.precision == "double" else 8  # bytes per element
    min_m = max(2 * p, 1)

    print(f"\nResolution: nx={args.nx}, ny={Ny}, nz={args.nz}")
    print(f"Fourier modes: Nkz={Nkz}, Nkx={Nkx} ({n_modes} total)")
    print(f"FD order (half-bandwidth): p={p}")
    print(f"Precision: {args.precision} ({bpe} bytes/element)")
    print(f"Minimum block size: m >= 2p = {min_m}")
    print()
    print(
        "Per-mode SPIKE storage = Ny^2/P + 4*P^2*p^2"
        "  (block LU + reduced system)"
    )
    print(
        "Arithmetic intensity of per-block LU solve"
        " ~ (2/3)*m / bytes_per_element"
    )
    print()

    # Gather valid partitions.
    rows: list[dict] = []
    for P in _divisors_of(Ny):
        m = Ny // P
        if m < min_m and P > 1:
            continue
        cost_per_mode = Ny * Ny / P + 4.0 * P * P * p * p
        block_lu_bytes = n_modes * P * m * m * bpe * 2
        reduced_n = 2 * P * p
        reduced_bytes = n_modes * reduced_n * reduced_n * bpe * 2
        total_bytes = block_lu_bytes + reduced_bytes

        # V, W spike matrices: 2 * n_modes * P * m * p * bpe * 2 ops
        spike_bytes = 2 * n_modes * P * m * p * bpe * 2
        total_bytes += spike_bytes

        ai = (2.0 / 3.0) * m / bpe
        rows.append(
            {
                "P": P,
                "m": m,
                "cost": cost_per_mode,
                "block_lu": block_lu_bytes,
                "reduced": reduced_bytes,
                "spike": spike_bytes,
                "total": total_bytes,
                "ai": ai,
            }
        )

    if not rows:
        print("No valid SPIKE partitions for these parameters.")
        sys.exit(1)

    # Find memory-optimal (excluding P=1 unless it's the only option).
    valid_spike = [r for r in rows if r["P"] >= 2]
    if valid_spike:
        best_cost = min(r["cost"] for r in valid_spike)
        best_P = next(r["P"] for r in valid_spike if r["cost"] == best_cost)
    else:
        best_P = 1

    # Dense backend row for comparison (LU factors only; original
    # matrices are discarded after factorisation, matvecs use D1/D2).
    dense_lu_bytes = n_modes * Ny * Ny * bpe * 2
    dense_total = dense_lu_bytes

    # Print table.
    hdr = (
        f"{'':>3}  {'P':>4}  {'m':>4}  "
        f"{'Block LU':>10}  {'Reduced':>10}  "
        f"{'Spikes':>10}  {'Total':>10}  "
        f"{'AI':>7}  {'GPU':<10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        marker = ">>>" if r["P"] == best_P else "   "
        print(
            f"{marker}  {r['P']:>4}  {r['m']:>4}  "
            f"{_format_bytes(r['block_lu']):>10}  "
            f"{_format_bytes(r['reduced']):>10}  "
            f"{_format_bytes(r['spike']):>10}  "
            f"{_format_bytes(r['total']):>10}  "
            f"{r['ai']:>5.2f}  "
            f"{_gpu_rating(r['ai']):<10}"
        )

    print("-" * len(hdr))
    print(
        f"     {'dense':>9}  "
        f"{_format_bytes(dense_lu_bytes):>10}  "
        f"{'--':>10}  {'--':>10}  "
        f"{_format_bytes(dense_total):>10}  "
        f"{(2.0 / 3.0) * Ny / bpe:>5.2f}  "
        f"{_gpu_rating((2.0 / 3.0) * Ny / bpe):<10}"
    )

    print()
    if best_P >= 2:
        print(f">>> = memory-optimal default (P={best_P})")
    else:
        print("    Only P=1 (single block) is valid for this Ny and p.")
    print(
        "    Total column = both Lk + Hk operators"
        " (block LU + reduced system + spike matrices)"
    )
    print(
        "    AI = arithmetic intensity of per-block LU"
        " (FLOP/byte; higher is better for GPU)"
    )
    print()


if __name__ == "__main__":
    main()
