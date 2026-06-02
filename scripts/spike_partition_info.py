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


def _serial_depth(m: int, p: int, P: int, banded: bool) -> str:
    """Total sequential scan depth (for --banded mode)."""
    if not banded:
        return "--"
    import math

    blk_depth = math.ceil(m / p)
    return f"{blk_depth}+{P}"


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
        "--banded",
        action="store_true",
        help="Show post-item-B banded-block cost model.",
    )
    args = ap.parse_args()

    Ny = args.ny
    p = args.fd_order
    Nkz = args.nz - 1
    Nkx = args.nx // 2
    n_modes = Nkz * Nkx
    bpe = 16 if args.precision == "double" else 8  # bytes per element
    min_m = max(2 * p, 1)

    banded = args.banded

    print(f"\nResolution: nx={args.nx}, ny={Ny}, nz={args.nz}")
    print(f"Fourier modes: Nkz={Nkz}, Nkx={Nkx} ({n_modes} total)")
    print(f"FD order (half-bandwidth): p={p}")
    print(f"Precision: {args.precision} ({bpe} bytes/element)")
    print(f"Minimum block size: m >= 2p = {min_m}")
    if banded:
        print(f"Banded-block threshold: m > 3p+1 = {3 * p + 1}")
    print()
    if banded:
        print(
            "Block storage = ~3*Ny*p (banded, m > 3p+1)"
            " or Ny^2/P (dense, m <= 3p+1)"
        )
        print("Reduced storage = ~12*P*p^2 (block-tridiagonal)")
    else:
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

        banded_ok = m > 3 * p + 1

        if banded and banded_ok:
            # Banded blocks: ~3*m*p per block.
            block_bytes = n_modes * P * 3 * m * p * bpe * 2
            # Block-tridiagonal reduced: ~(3P-2)*(2p)^2 entries
            red_entries = (3 * P - 2) * (2 * p) ** 2
        else:
            block_bytes = n_modes * P * m * m * bpe * 2
            red_entries = (2 * P * p) ** 2

        reduced_bytes = n_modes * red_entries * bpe * 2
        spike_bytes = 2 * n_modes * P * m * p * bpe * 2
        total_bytes = block_bytes + reduced_bytes + spike_bytes

        cost_per_mode = Ny * Ny / P + 4.0 * P * P * p * p
        ai = (2.0 / 3.0) * m / bpe

        import math

        rows.append(
            {
                "P": P,
                "m": m,
                "cost": cost_per_mode,
                "block": block_bytes,
                "reduced": reduced_bytes,
                "spike": spike_bytes,
                "total": total_bytes,
                "ai": ai,
                "m_over_p": m / p,
                "banded_ok": banded_ok,
                "serial": math.ceil(m / p) + P,
            }
        )

    if not rows:
        print("No valid SPIKE partitions for these parameters.")
        sys.exit(1)

    # Find memory-optimal (excluding P=1).
    valid_spike = [r for r in rows if r["P"] >= 2]
    if valid_spike:
        best_total = min(r["total"] for r in valid_spike)
        best_P = next(r["P"] for r in valid_spike if r["total"] == best_total)
    else:
        best_P = 1

    # Speed-optimal: minimise serial depth ceil(m/p) + P.
    speed_P = None
    if banded and valid_spike:
        best_serial = min(r["serial"] for r in valid_spike)
        speed_P = next(
            r["P"] for r in valid_spike if r["serial"] == best_serial
        )

    # Dense backend row for comparison.
    dense_lu_bytes = n_modes * Ny * Ny * bpe * 2
    dense_total = dense_lu_bytes

    # Print table.
    extra = ""
    if banded:
        extra = f"  {'m/p':>5}  {'banded':>6}  {'depth':>7}"
    hdr = (
        f"{'':>3}  {'P':>4}  {'m':>4}  "
        f"{'Block':>10}  {'Reduced':>10}  "
        f"{'Spikes':>10}  {'Total':>10}  "
        f"{'AI':>7}  {'GPU':<10}" + extra
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        if r["P"] == best_P:
            marker = ">>>"
        elif banded and speed_P and r["P"] == speed_P:
            marker = " v "
        else:
            marker = "   "
        extra_cols = ""
        if banded:
            extra_cols = (
                f"  {r['m_over_p']:>5.1f}  "
                f"{'yes' if r['banded_ok'] else 'no':>6}  "
                f"{_serial_depth(r['m'], p, r['P'], banded):>7}"
            )
        print(
            f"{marker}  {r['P']:>4}  {r['m']:>4}  "
            f"{_format_bytes(r['block']):>10}  "
            f"{_format_bytes(r['reduced']):>10}  "
            f"{_format_bytes(r['spike']):>10}  "
            f"{_format_bytes(r['total']):>10}  "
            f"{r['ai']:>5.2f}  "
            f"{_gpu_rating(r['ai']):<10}" + extra_cols
        )

    print("-" * len(hdr))
    dense_extra = ""
    if banded:
        dense_extra = f"  {'':>5}  {'':>6}  {'':>7}"
    print(
        f"     {'dense':>9}  "
        f"{_format_bytes(dense_lu_bytes):>10}  "
        f"{'--':>10}  {'--':>10}  "
        f"{_format_bytes(dense_total):>10}  "
        f"{(2.0 / 3.0) * Ny / bpe:>5.2f}  "
        f"{_gpu_rating((2.0 / 3.0) * Ny / bpe):<10}" + dense_extra
    )

    print()
    if best_P >= 2:
        print(f">>> = memory-optimal default (P={best_P})")
    else:
        print("    Only P=1 (single block) is valid for this Ny and p.")
    if banded and speed_P and speed_P != best_P:
        print(
            f" v  = speed-optimal (P={speed_P},"
            f" serial depth"
            f" {_serial_depth(Ny // speed_P, p, speed_P, True)})"
        )
    print(
        "    Total column = both Lk + Hk operators"
        " (block + reduced + spike matrices)"
    )
    if banded:
        print("    depth = ceil(m/p) + P (per-block + reduced scan steps)")
    print(
        "    AI = arithmetic intensity of per-block LU"
        " (FLOP/byte; higher is better for GPU)"
    )
    print()


if __name__ == "__main__":
    main()
