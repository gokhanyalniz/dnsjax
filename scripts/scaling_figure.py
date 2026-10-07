#!/usr/bin/env python3
r"""Strong-scaling figure and tables of the docs, from measured runs.

Reads a CSV with one row per measured run -- ``nodes, ranks, np0, np1,
nx, s_per_t``, the last the solver's ``s/t`` (wall seconds per unit of
simulated time) -- and

- draws two panels against the fastest one-node layout at the largest
  ``nx`` in the file: (a) the speed-up, log-log, beside the ideal
  line, and (b) the parallel efficiency `$T_1 / (N\,T_N)$`.  Each
  point is the mean of a layout's runs at that node count, its bar
  their range;
- writes the figure as two SVGs, one per colour scheme (the README's
  ``<picture>`` picks one), the text left as text in the README's
  system font stack;
- prints the two markdown tables of ``docs/scaling.md``: the strong
  scaling, and the one-node runs at a reduced ``nx`` -- each rank then
  holds the block of an ``N``-node run, ``N = nx_full / nx`` -- against
  those ``N``-node runs.

A layout is a family ``(np0, k N)``: its ``np0`` fixed, its ``np1``
growing with the node count ``N``.  Rows at the largest ``nx`` are the
scaling runs; the others are the reduced one-node runs.  The defaults
read and write the ARCHER2 files under ``docs/figures/`` (``--csv``,
``--out``), and a re-run on the same data rewrites the SVGs byte for
byte.

Usage (from the repository root; matplotlib comes with the ``plots``
group)::

    uv run --group plots python scripts/scaling_figure.py
    # PNG previews beside the SVGs' names, to look at both schemes
    uv run --group plots python scripts/scaling_figure.py --png /tmp/fig
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "figures"

#: The README's font stack, as ``docs/figures/step-pipeline-*.svg`` set
#: it.  The layout is computed with matplotlib's own sans-serif font;
#: the SVG text is anchored, so a narrower system font only tightens it.
FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, "
    "sans-serif"
)

#: Per colour scheme: the page behind the figure (GitHub's, which the
#: marker rings take), the step-pipeline figures' ink and rules, and
#: two categorical hues whose pair clears the colour-vision-deficiency
#: separation checks against that page.
SCHEMES = {
    "light": {
        "page": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "axis": "#8c959f",
        "grid": "#e6eaef",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "page": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "axis": "#7d8590",
        "grid": "#262c36",
        "series": ("#3987e5", "#d95926"),
    },
}


def read_runs(path: Path) -> list[dict]:
    """The CSV's rows, integers but for ``s_per_t``."""
    with open(path, newline="") as fh:
        return [
            {k: float(v) if k == "s_per_t" else int(v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def families(runs: list[dict], nx: int) -> dict[tuple[int, int], dict]:
    """``{(np0, k): {N: [s_per_t, ...]}}`` of the runs at *nx*.

    ``k = np1 / N``: the family ``(np0, k N)``, ordered by ``np0``
    descending (the first is the docs' recommended grid).
    """
    out: dict[tuple[int, int], dict] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["nx"] == nx:
            key = (r["np0"], r["np1"] // r["nodes"])
            out[key][r["nodes"]].append(r["s_per_t"])
    return dict(sorted(out.items(), key=lambda kv: -kv[0][0]))


def family_label(key: tuple[int, int]) -> str:
    np0, k = key
    return f"({np0}, {'' if k == 1 else k}n)"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def reference_time(fams: dict) -> float:
    """`$T_1$`: the fastest one-node layout's mean ``s/t``."""
    return min(mean(nodes[1]) for nodes in fams.values() if 1 in nodes)


def _seconds(x: float) -> str:
    """An ``s/t`` value: one decimal from 100 up, two below."""
    return f"{x:.1f}" if x >= 100 else f"{x:.2f}"


def _sig(x: float, digits: int = 3) -> str:
    """*x* to *digits* significant figures, trailing zeros kept."""
    return f"{x:#.{digits}g}".rstrip(".")


def scaling_table(fams: dict, t1: float) -> str:
    """The strong-scaling table: the first family in full, the second's
    ``s/t`` beside it."""
    keys = list(fams)
    first, rest = fams[keys[0]], keys[1:]
    head = (
        f"| nodes | ranks | `{family_label(keys[0])}`: s per time unit "
        "| speed-up | efficiency | node hours per time unit |"
    )
    rule = "|---:|---:|---:|---:|---:|---:|"
    for key in rest:
        head += f" `{family_label(key)}`: s per time unit |"
        rule += "---:|"
    lines = [head, rule]
    for n in sorted(first):
        t = mean(first[n])
        line = (
            f"| {n} | {n * keys[0][0] * keys[0][1]} | {_seconds(t)} "
            f"| {_sig(t1 / t)} | {t1 / (n * t):.2f} "
            f"| {n * t / 3600:.2f} |"
        )
        for key in rest:
            vals = fams[key].get(n)
            line += f" {_seconds(mean(vals)) if vals else '—'} |"
        lines.append(line)
    return "\n".join(lines)


def per_rank_table(runs: list[dict], fams: dict, t1: float) -> str:
    """The one-node runs at a reduced ``nx`` against the ``N``-node
    runs whose per-rank block they hold, for the first family."""
    nx_full = max(r["nx"] for r in runs)
    key = next(iter(fams))
    np0, k = key
    lines = [
        "| N | one node, `nx = "
        f"{nx_full}/N`: s per time unit | per-rank factor "
        "| N nodes: s per time unit | off-node factor |",
        "|---:|---:|---:|---:|---:|",
    ]
    reduced = {
        nx_full // r["nx"]: r["s_per_t"]
        for r in runs
        if r["nx"] < nx_full
        and r["nodes"] == 1
        and r["np0"] == np0
        and r["np1"] == k
    }
    for n in sorted(reduced):
        if n not in fams[key]:
            continue
        one, many = reduced[n], mean(fams[key][n])
        lines.append(
            f"| {n} | {_seconds(one)} | {one * n / t1:.2f} "
            f"| {_seconds(many)} "
            f"| {many / one:.2f} |"
        )
    return "\n".join(lines)


def _restyle_svg(path: Path) -> None:
    """Set every text run's font family to the README's stack."""
    text = path.read_text()
    text = re.sub(r"font-family: [^;\"]+", f"font-family: {FONT_STACK}", text)
    path.write_text(text)


def draw(
    fams: dict,
    t1: float,
    xlabel: str,
    scheme: str,
    out: Path,
    png: Path | None,
) -> None:
    """Both panels in one colour *scheme*, to *out* (and a PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    c = SCHEMES[scheme]
    rc = {
        "svg.fonttype": "none",
        "svg.hashsalt": "dnsjax-scaling",
        "font.size": 9.5,
        "text.color": c["ink"],
        "axes.labelcolor": c["ink"],
        "axes.edgecolor": c["axis"],
        "axes.linewidth": 0.8,
        "xtick.color": c["axis"],
        "ytick.color": c["axis"],
        "xtick.labelcolor": c["muted"],
        "ytick.labelcolor": c["muted"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.labelcolor": c["ink"],
    }
    with plt.rc_context(rc):
        fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(9.0, 3.3))
        fig.subplots_adjust(
            left=0.065, right=0.99, bottom=0.155, top=0.97, wspace=0.22
        )
        nodes = sorted({n for fam in fams.values() for n in fam})
        lo, hi = nodes[0], nodes[-1]
        for ax in (ax_s, ax_e):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(nodes))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([str(n) for n in nodes])
            ax.set_xlim(lo / 1.35, hi * 1.35)
            ax.set_xlabel(xlabel)
            ax.grid(True, color=c["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.patch.set_alpha(0)

        ax_s.set_yscale("log", base=2)
        ax_s.yaxis.set_major_locator(FixedLocator(nodes))
        ax_s.yaxis.set_minor_locator(NullLocator())
        ax_s.set_yticklabels([str(n) for n in nodes])
        ax_s.set_ylim(lo / 1.35, hi * 1.35)
        ax_s.set_ylabel("speed-up over one node")
        ideal = [lo / 1.35, hi * 1.35]
        ax_s.plot(ideal, ideal, color=c["axis"], linewidth=1.0, zorder=1)

        ax_e.set_ylim(0.6, 1.1)
        ax_e.yaxis.set_major_locator(FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0]))
        ax_e.set_ylabel("parallel efficiency")
        ax_e.axhline(1.0, color=c["axis"], linewidth=1.0, zorder=1)

        handles = []
        # Two validated hues: at most two layouts (main() checks).
        for (key, fam), colour in zip(fams.items(), c["series"], strict=False):
            ns = sorted(fam)
            # Each point's mean, slowest and fastest run, as speed-ups.
            speed, slow, fast = (
                [t1 / agg(fam[n]) for n in ns] for agg in (mean, max, min)
            )
            eff, eff_slow, eff_fast = (
                [s / n for s, n in zip(seq, ns, strict=True)]
                for seq in (speed, slow, fast)
            )
            for ax, y, ylo, yhi in (
                (ax_s, speed, slow, fast),
                (ax_e, eff, eff_slow, eff_fast),
            ):
                ax.vlines(ns, ylo, yhi, color=colour, linewidth=1.2, zorder=3)
                ax.plot(
                    ns,
                    y,
                    color=colour,
                    linewidth=1.5,
                    marker="o",
                    markersize=6.5,
                    markeredgecolor=c["page"],
                    markeredgewidth=1.4,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=4,
                )
            # A direct label beside the middle point, the first family's
            # above its line and the second's below, in the text ink.
            mid = ns[len(ns) // 2]
            above = not handles
            ax_e.annotate(
                family_label(key),
                xy=(mid, eff[ns.index(mid)]),
                xytext=(5, 7 if above else -7),
                textcoords="offset points",
                ha="left",
                va="bottom" if above else "top",
                color=c["ink"],
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    color=colour,
                    linewidth=1.5,
                    marker="o",
                    markersize=6.5,
                    markeredgecolor=c["page"],
                    markeredgewidth=1.4,
                    label=family_label(key),
                )
            )
        handles.append(
            Line2D([], [], color=c["axis"], linewidth=1.0, label="ideal")
        )
        ax_s.legend(
            handles=handles,
            title="device grid (np0, np1)",
            title_fontsize=9.5,
            loc="upper left",
            frameon=False,
            borderaxespad=0.2,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out, format="svg", transparent=True, metadata={"Date": None}
        )
        if png is not None:
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=150, facecolor=c["page"])
        plt.close(fig)
    _restyle_svg(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=FIGURES / "archer2-scaling.csv",
        help="the measured runs",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=FIGURES / "archer2-scaling",
        help="SVG path prefix: <out>-light.svg and <out>-dark.svg",
    )
    ap.add_argument(
        "--png",
        type=Path,
        default=None,
        help="also write <png>-light.png and <png>-dark.png previews",
    )
    a = ap.parse_args()

    runs = read_runs(a.csv)
    nx_full = max(r["nx"] for r in runs)
    fams = families(runs, nx_full)
    if len(fams) > min(len(c["series"]) for c in SCHEMES.values()):
        raise SystemExit(f"{len(fams)} layouts: the figure has two hues")
    t1 = reference_time(fams)
    per_node = {r["ranks"] // r["nodes"] for r in runs if r["nx"] == nx_full}
    xlabel = "nodes"
    if len(per_node) == 1:
        xlabel += f" ({per_node.pop()} MPI ranks each)"
    for scheme in SCHEMES:
        out = a.out.with_name(f"{a.out.name}-{scheme}.svg")
        png = a.png.with_name(f"{a.png.name}-{scheme}.png") if a.png else None
        draw(fams, t1, xlabel, scheme, out, png)
        print(f"wrote {out}")
    print(f"\nT_1 = {_seconds(t1)} s per time unit\n")
    print(scaling_table(fams, t1))
    print()
    print(per_rank_table(runs, fams, t1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
