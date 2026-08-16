"""Step 017 — the food-cost distribution chart.

`scripts/validate_foodcost.py` prints the numbers; this draws them. Nothing here
runs the pipeline and nothing here calls an LLM: it reads only the cache files
that validation already wrote to `data/output/<slug>-costed.json`. If they are
missing it says so and stops, because silently plotting a subset of the corpus
would put a chart on screen that nobody could reproduce.

What the chart claims, and what it does not:

  It plots the COSTING. Menu price is transcribed from the restaurant's own
  menu; plate cost is computed bottom-up from decomposed quantities, canonical
  SKUs and catalog prices. The two arrive from opposite ends of the pipeline and
  nothing in the code ties them together.

  On the current corpus the distribution lands well BELOW the industry's 28-33%
  band, and the title says so rather than claiming otherwise. The band is
  *actual* food cost -- total purchases over total revenue, weighted by what
  sells and inclusive of waste, spoilage and staff meals. What is plotted is an
  unweighted median of *theoretical* plate cost over every priced menu line,
  including $8 rice sides and $12 juice. Theoretical always sits under actual,
  and the unweighted median over cheap sides widens the gap further, so the
  comparison is not like-for-like. The band is still drawn, because the gap is
  the finding; see `data/output/findings.md`.

  It says NOTHING about the demand forecast. There is no POS history in this
  build to validate a forecast against, and the chart is captioned to say so.

The band and the coverage floor are imported from `scripts.validate_foodcost`
rather than restated, so the shading on the chart is by construction the same
band the printed statistics use. A second copy of `0.28` in this file would be a
place for the two to silently disagree.

Usage:
    uv run python -m scripts.plot_foodcost
    uv run python -m scripts.plot_foodcost --tier all --out data/output/fc.png
Out:
    data/output/foodcost-distribution.png
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this runs over SSH and in CI, never in a window
import matplotlib.pyplot as plt

from scripts.validate_foodcost import BAND_HIGH, BAND_LOW, COVERAGE_FLOOR, _plates
from src.tools.snapshot import list_corpus

OUTPUT_PATH = Path("data/output/foodcost-distribution.png")
CACHE_DIR = Path("data/output")

# 2 percentage points per bar. Narrow enough that a bimodal distribution (the
# failure mode where half the plates are missing their protein) would show as two
# humps rather than one wide one; wide enough not to look like noise at n ~ 400.
BIN_WIDTH = 0.02

# The tail is clipped at this percentile so the bulk of the distribution is
# readable instead of squeezed against the y axis by three plates at 300%.
# Whatever gets clipped is counted and stated on the chart.
CLIP_PERCENTILE = 0.98

# The percentile alone is not enough: a corpus with 3% wild outliers puts p98 at
# 160% and squashes the whole distribution into the left tenth of the axis. The
# axis is therefore also capped at this multiple of the median, which scales with
# the data instead of hard-coding a range.
CLIP_MEDIAN_MULTIPLE = 2.0

# Never clip below this: the industry band must always be fully on screen with
# room to its right, or the shading loses its meaning as a reference.
MIN_X_MAX = 0.60

# --- palette (dataviz skill, light surface; the reference instance's first slots)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_BARS = "#2a78d6"     # slot 1, blue — one series, one color, never a value ramp
C_MEDIAN = "#eb6834"   # slot 2, orange — the one annotation that must not read as data
BAND_FILL = "#e1e0d9"  # neutral: the band is a reference region, not a verdict


def cache_path(slug: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{slug}-costed.json"


def load_records(slugs: list[str], cache_dir: Path = CACHE_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every cache file that exists. Returns (records, missing slugs)."""
    records, missing = [], []
    for slug in slugs:
        path = cache_path(slug, cache_dir)
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(slug)
    return records, missing


def select(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The chart's data selection, and the only place it happens.

    Two exclusions, both of which have to be visible on the chart:

    * `food_cost_pct is None` — nothing could be costed at all. `cost_plates`
      writes None rather than 0.0 there precisely so an uncosted plate is never
      mistaken for a free one. These never had a ratio, so they are not part of
      any denominator here.
    * `coverage < COVERAGE_FLOOR` — costed from under half its ingredients, so
      the ratio describes the catalog's gaps rather than the plate. Counted,
      excluded, and stated in the annotation.
    """
    rated = _plates(records)  # already drops food_cost_pct is None
    included = [p for p in rated if float(p.get("coverage") or 0.0) >= COVERAGE_FLOOR]
    ratios = sorted(float(p["food_cost_pct"]) for p in included)

    return {
        "ratios": ratios,
        "n_included": len(ratios),
        "n_excluded": len(rated) - len(included),
        # Restaurants that contributed at least one plotted plate — the honest
        # denominator for "across N restaurants", not the number of files read.
        "n_restaurants": len({p["slug"] for p in included}),
        "median": statistics.median(ratios) if ratios else 0.0,
    }


def annotation_text(sel: dict[str, Any]) -> str:
    """The line drawn on the chart and printed to the console, ASCII only.

    Kept identical in both places on purpose: the chart gets screenshotted and
    passed around without its narration, and the console line is what ends up
    pasted into a message. They must not be able to drift.
    """
    return (
        f"n = {sel['n_included']} plates - {sel['n_restaurants']} restaurants - "
        f"plates below {COVERAGE_FLOOR:.0%} ingredient coverage excluded "
        f"(n = {sel['n_excluded']})"
    )


def x_limit(ratios: list[float]) -> tuple[float, int]:
    """Right-hand x limit and how many plates fall beyond it.

    Nearest-rank percentile rounded up to a whole bin, so the last bar is a full
    bar rather than a sliver.
    """
    if not ratios:
        return MIN_X_MAX, 0

    ordered = sorted(ratios)
    idx = min(len(ordered) - 1, max(0, int(round(CLIP_PERCENTILE * (len(ordered) - 1)))))
    cap = max(MIN_X_MAX, CLIP_MEDIAN_MULTIPLE * statistics.median(ordered))
    raw = min(max(ordered[idx], MIN_X_MAX), cap)
    hi = BIN_WIDTH * (int(raw / BIN_WIDTH) + 1)
    return hi, sum(1 for r in ordered if r > hi)


def draw(sel: dict[str, Any], out_path: Path) -> None:
    """One series, one color; band and median as recessive reference marks."""
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    ratios = sel["ratios"]
    median = sel["median"]
    hi, clipped = x_limit(ratios)
    shown = [r for r in ratios if r <= hi]

    edges, edge = [], 0.0
    while edge < hi - 1e-9:
        edges.append(round(edge, 6))
        edge += BIN_WIDTH
    edges.append(round(hi, 6))

    # Font sizes are set for on-screen viewing at reduced scale, not for a print
    # figure: every tick label has to stay legible after downscaling.
    fig, ax = plt.subplots(figsize=(12.0, 6.8), dpi=200)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=13, length=0)

    # The band goes down first: it is the backdrop the bars are read against.
    ax.axvspan(BAND_LOW, BAND_HIGH, color=BAND_FILL, zorder=0)

    counts, _, _ = ax.hist(
        shown,
        bins=edges,
        color=C_BARS,
        edgecolor=SURFACE,  # 2px surface gap between adjacent bars, not a border
        linewidth=2.0,
        zorder=2,
    )
    top = max(counts) if len(counts) else 1.0
    # Headroom for the two reference labels, which live in separate horizontal
    # lanes: band on top, median below it. They sit within a few points of each
    # other on the x axis by construction, so stacking them is the only way they
    # never collide.
    ax.set_ylim(0, top * 1.42)
    ax.set_xlim(0, hi)

    # The band label is wider than the band, so it is centred on the band and
    # given the full width above the plot. Muted ink: the shading carries no
    # verdict on its own and the words have to say what it is.
    ax.text(
        (BAND_LOW + BAND_HIGH) / 2, top * 1.40,
        f"Industry band {BAND_LOW:.0%}-{BAND_HIGH:.0%}",
        color=INK_SECONDARY, fontsize=13, ha="center", va="top", zorder=3,
    )

    # Stops below its own label rather than spanning the axes: a full-height rule
    # runs straight through the band label sitting above the plot.
    ax.vlines(median, 0, top * 1.16, color=C_MEDIAN, linewidth=2.5, zorder=4)
    # Label flips to the left of the line when the median sits in the right half,
    # so it can never run off the axes.
    right_side = median < hi * 0.65
    ax.text(
        median + (hi * 0.012 if right_side else -hi * 0.012), top * 1.24,
        f"Median {median:.1%}",
        color=C_MEDIAN, fontsize=15, fontweight="bold",
        ha="left" if right_side else "right", va="center", zorder=4,
    )

    ax.set_xticks([round(BIN_WIDTH * 5 * i, 4) for i in range(int(hi / (BIN_WIDTH * 5)) + 1)])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(
        "Implied food cost (plate cost / menu price)",
        color=INK_SECONDARY, fontsize=15, labelpad=11,
    )
    ax.set_ylabel("Plates", color=INK_SECONDARY, fontsize=15, labelpad=11)

    # The title is computed from the median, never asserted. A hard-coded "lands
    # on the band" survived one corpus and would have lied on the next.
    if median < BAND_LOW:
        verdict = f"Implied food cost lands well below the industry band: median {median:.1%}"
    elif median > BAND_HIGH:
        verdict = f"Implied food cost lands above the industry band: median {median:.1%}"
    else:
        verdict = f"Implied food cost lands on the industry band: median {median:.1%}"
    ax.set_title(
        verdict,
        color=INK, fontsize=19, fontweight="bold", loc="left", pad=42,
    )
    ax.text(
        0.0, 1.055, annotation_text(sel),
        transform=ax.transAxes, color=INK_SECONDARY, fontsize=13, va="bottom", ha="left",
    )

    # Three short footer lines rather than two long ones: at this figure width a
    # single sentence of this length runs off the right edge.
    # Hard-wrapped by hand, ~135 characters a line. At this figure width and
    # font size a longer line runs off the right edge of the PNG.
    lines = [
        "Menu price is transcribed from the menu; plate cost is computed bottom-up from "
        "decomposed quantities and catalog prices.",
        "The 28-33% band is ACTUAL food cost: purchases over revenue, weighted by what "
        "sells, including waste, spoilage and staff meals.",
        "Plotted here is an unweighted median of THEORETICAL plate cost over every priced "
        "line, high-margin sides and drinks included.",
        "The two are not like-for-like and the gap is the finding, not a failing grade. This "
        "says nothing about the demand forecast.",
    ]
    if clipped:
        lines.append(
            f"Tail clipped at {hi:.0%}: {clipped} plate{'s' if clipped != 1 else ''} above it "
            f"(highest {max(ratios):.0%}) are not drawn."
        )
    # Laid out from the bottom of the figure upward, so adding a footer line
    # pushes the block up instead of running it off the bottom edge.
    for i, line in enumerate(lines):
        fig.text(
            0.012, 0.014 + (len(lines) - 1 - i) * 0.024, line,
            color=INK_MUTED, fontsize=11, va="bottom", ha="left",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.024 + len(lines) * 0.024, 1.0, 1.0))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="plot_foodcost",
        description="Plot the implied food cost distribution from cached costed runs.",
    )
    parser.add_argument(
        "--tier", default="B", choices=["A", "B", "all"],
        help="corpus tier to plot (default: B, matching validate_foodcost)",
    )
    parser.add_argument(
        "--out", default=str(OUTPUT_PATH), metavar="PATH",
        help=f"where to write the PNG (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--cache-dir", default=str(CACHE_DIR), metavar="DIR",
        help=f"directory holding <slug>-costed.json (default: {CACHE_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    slugs = list_corpus(None if args.tier == "all" else args.tier)
    records, missing = load_records(slugs, Path(args.cache_dir))

    if missing:
        print(
            f"No cached run for {len(missing)} of {len(slugs)} tier-{args.tier} "
            f"restaurants: {', '.join(missing[:5])}"
            + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""),
            file=sys.stderr,
        )
        print(
            "This script only reads caches; it never runs the pipeline. Run:\n"
            "    uv run python -m scripts.validate_foodcost"
            + ("" if args.tier == "B" else f" --tier {args.tier}"),
            file=sys.stderr,
        )
        return 2

    sel = select(records)
    if not sel["ratios"]:
        print(
            "The cached runs contain no plate with both a food cost ratio and "
            f"at least {COVERAGE_FLOOR:.0%} ingredient coverage — nothing to plot.",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out)
    draw(sel, out_path)

    hi, clipped = x_limit(sel["ratios"])
    print(annotation_text(sel))
    print(f"median {sel['median']:.1%} - band {BAND_LOW:.0%}-{BAND_HIGH:.0%} shaded")
    if clipped:
        print(f"tail clipped at {hi:.0%}: {clipped} plates above it are not drawn")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
