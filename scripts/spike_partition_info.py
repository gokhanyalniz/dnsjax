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
    python scripts/spike_partition_info.py \\
        --ny 128 --nz 128 --nx 128 --block-thomas
    python scripts/spike_partition_info.py \\
        --ny 128 --nz 128 --nx 128 --n-operators 4
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
    ap.add_argument(
        "--block-thomas",
        action="store_true",
        dest="block_thomas",
        help="Use the block-Thomas reduced system "
        "(matches --solver.block_thomas True).",
    )
    ap.add_argument(
        "--n-operators",
        type=int,
        default=2,
        dest="n_operators",
        help="Number of SPIKE operator equivalents: "
        "2 for Cartesian (1 Lk + 1 Hk, default), "
        "4 for cylindrical (1 Lk + 3 stacked Hk).",
    )
    args = ap.parse_args()

    Ny = args.ny
    p = args.fd_order
    Nkz = args.nz - 1
    Nkx = args.nx // 2
    n_modes = Nkz * Nkx
    # Factors are stored and solved in real arithmetic (the
    # operators are real; a complex RHS is split into re/im
    # columns at solve time), so bytes/element is the real size.
    bpe = 8 if args.precision == "double" else 4
    min_m = max(2 * p, 1)
    block_thomas = args.block_thomas
    n_ops = args.n_operators

    print(f"\nResolution: nx={args.nx}, ny={Ny}, nz={args.nz}")
    print(f"Fourier modes: Nkz={Nkz}, Nkx={Nkx} ({n_modes} total)")
    print(f"FD order (half-bandwidth): p={p}")
    print(f"Precision: {args.precision} ({bpe} bytes/element, real factors)")
    print(f"Minimum block size: m >= 2p = {min_m}")
    print(f"Operators: {n_ops} (Cartesian=2, cylindrical=4)")
    bt_label = (
        "block-Thomas (--block-thomas)" if block_thomas else "dense (default)"
    )
    print(f"Reduced system: {bt_label}")
    print()
    if block_thomas:
        print(
            "Per-mode cost = Ny^2/P + (3P-2)*4*p^2"
            "  (block LU + block-Thomas reduced)"
        )
    else:
        print("Per-mode cost = Ny^2/P + 4*P^2*p^2  (block LU + dense reduced)")
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

        # Block LU: P dense (m, m) blocks per mode per operator.
        block_bytes = n_modes * P * m * m * bpe * n_ops

        # Reduced system (per mode per operator).
        if block_thomas:
            # P diag + (P-1) super + (P-1) sub, each (2p, 2p).
            red_entries = (3 * P - 2) * (2 * p) ** 2
        else:
            # Full (2Pp, 2Pp) dense LU.
            red_entries = (2 * P * p) ** 2
        reduced_bytes = n_modes * red_entries * bpe * n_ops

        # Spike matrices V, W: 2 * P * m * p per mode.
        spike_bytes = 2 * n_modes * P * m * p * bpe * n_ops
        total_bytes = block_bytes + reduced_bytes + spike_bytes

        # Per-mode cost for optimisation (spike cost 2*Ny*p
        # is P-independent and cancels out).
        cost_bt = Ny * Ny / P + (3 * P - 2) * 4.0 * p * p
        cost_dense = Ny * Ny / P + 4.0 * P * P * p * p
        cost = cost_bt if block_thomas else cost_dense
        cost_alt = cost_dense if block_thomas else cost_bt
        ai = (2.0 / 3.0) * m / bpe

        rows.append(
            {
                "P": P,
                "m": m,
                "cost": cost,
                "cost_alt": cost_alt,
                "block": block_bytes,
                "reduced": reduced_bytes,
                "spike": spike_bytes,
                "total": total_bytes,
                "ai": ai,
            }
        )

    if not rows:
        print("No valid SPIKE partitions for these parameters.")
        sys.exit(1)

    # Find memory-optimal (excluding P=1).
    valid_spike = [r for r in rows if r["P"] >= 2]
    if valid_spike:
        best_cost = min(r["cost"] for r in valid_spike)
        best_P = next(r["P"] for r in valid_spike if r["cost"] == best_cost)
    else:
        best_P = 1

    # Find optimal under alternative cost model.
    alt_P = None
    if valid_spike:
        best_cost_alt = min(r["cost_alt"] for r in valid_spike)
        alt_P = next(
            r["P"] for r in valid_spike if r["cost_alt"] == best_cost_alt
        )

    # Dense backend row for comparison.
    dense_lu_bytes = n_modes * Ny * Ny * bpe * n_ops

    # Print table.
    hdr = (
        f"{'':>3}  {'P':>4}  {'m':>4}  "
        f"{'Block':>10}  {'Reduced':>10}  "
        f"{'Spikes':>10}  {'Total':>10}  "
        f"{'AI':>7}  {'GPU':<10}"
    )
    print(hdr)
    print("-" * len(hdr))

    alt_marker = "dn>" if block_thomas else "bt>"
    for r in rows:
        if r["P"] == best_P:
            marker = ">>>"
        elif alt_P and r["P"] == alt_P and alt_P != best_P:
            marker = alt_marker
        else:
            marker = "   "
        print(
            f"{marker}  {r['P']:>4}  {r['m']:>4}  "
            f"{_format_bytes(r['block']):>10}  "
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
        f"{_format_bytes(dense_lu_bytes):>10}  "
        f"{(2.0 / 3.0) * Ny / bpe:>5.2f}  "
        f"{_gpu_rating((2.0 / 3.0) * Ny / bpe):<10}"
    )

    print()
    mode = "block-Thomas" if block_thomas else "dense reduced"
    if best_P >= 2:
        default = "" if block_thomas else " (code default)"
        print(f">>> = {mode} optimal (P={best_P}){default}")
    else:
        print("    Only P=1 (single block) is valid for this Ny and p.")
    if alt_P and alt_P != best_P:
        alt_mode = "dense reduced" if block_thomas else "block-Thomas"
        print(f"{alt_marker} = {alt_mode} optimal (P={alt_P})")
    print(
        f"    Total = all SPIKE operators"
        f" ({n_ops}x: block + reduced + spike matrices)"
    )
    print(
        "    AI = arithmetic intensity of per-block LU"
        " (FLOP/byte; higher is better for GPU)"
    )
    if block_thomas:
        print(
            "    Latency: block-Thomas runs 2(P-1) sequential scan"
            " steps per solve;\n    prefer a larger m"
            " (--solver.spike_block_size) or the dense reduced"
            " system\n    (the code default) when kernel-launch"
            " latency dominates."
        )
    print()


if __name__ == "__main__":
    main()
