"""Prior-to-actuals convergence curve: what happens to the cold-start forecast.

`src/nodes/forecast.py` emits a prior (`method: "cold_start_prior"`) because a
restaurant on day one has no POS history. The obvious objection is that the
prior is guesswork. It partly is — and it was always meant as a bridge, not a
destination. This script draws the bridge: a flat prior, real sales arriving
daily, and `src.tools.blend.blend` walking the estimate from one to the other
over six weeks.

THE ACTUALS ON THIS CHART ARE SYNTHETIC. There is no POS feed in this build,
and presenting invented sales as measured ones would be the single most
dishonest thing this repo could do, so the actuals are generated from
`random.Random(42)` and labelled as synthetic on the chart itself — not only
here, because a chart gets read on its own, detached from the text around it.
What is being demonstrated is the *mechanism*,
which is deterministic arithmetic and does not depend on the numbers being real.

Run:  uv run python -m scripts.plot_convergence
Out:  data/output/convergence.png
"""
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI and over SSH, never in a window
import matplotlib.pyplot as plt

from src.tools.blend import blend
from src.tools.snapshot import list_corpus, load_restaurant

OUTPUT_PATH = Path("data/output/convergence.png")
CACHE_DIR = Path("data/output")

DAYS = 42                 # six whole weeks: long enough that the weekly shape reads
PRIOR_STRENGTH_K = 14.0   # blend()'s k, in days of data. Two weeks of trade beats the prior.
SEED = 42

# The synthetic truth is deliberately offset from the prior so there is visible
# convergence to watch. 18% sits inside the 15-20% band the step asks for and is
# roughly the size of miss a metadata-only estimate should be expected to make.
TRUTH_OFFSET = 1.18

# Modest compounding growth: a new restaurant builds trade. 0.35%/day is ~16%
# across the six weeks — a ramp, not a hockey stick.
DAILY_TREND = 0.0035

# Day-to-day noise as a fraction of that day's expected covers. Real covers are
# noisy; a smooth synthetic series would make the blend look better than it is.
NOISE_SD = 0.09

# Fallback weekly shape (Mon..Sun) if the snapshot carries none. Mean is not 1.0;
# it is normalised at use.
DEFAULT_WEEK_SHAPE = [0.80, 0.85, 0.90, 1.00, 1.20, 1.30, 0.95]

# --- palette (dataviz skill, light surface; validated all-pairs, worst CVD dE 9.2)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_PRIOR = "#eb6834"     # slot 2, orange
C_ACTUALS = "#1baf7a"   # slot 3, aqua — sub-3:1 on this surface, so it is direct-labelled
C_BLEND = "#2a78d6"     # slot 1, blue — the hero series


def _prior_from_cache(slug: str) -> float | None:
    """Mean daily covers from `data/output/<slug>-costed.json`, if step 017 cached one."""
    path = CACHE_DIR / f"{slug}-costed.json"
    if not path.is_file():
        return None

    state = json.loads(path.read_text(encoding="utf-8"))
    forecast = state.get("demand_forecast") or {}
    per_week = forecast.get("covers_per_week")
    if per_week:
        return float(per_week) / 7.0

    per_day = forecast.get("covers_per_day") or []
    if len(per_day) == 7:
        return sum(float(v) for v in per_day) / 7.0
    return None


def _prior_from_snapshot(slug: str) -> float:
    """Fallback: the deterministic half of the step 013 prior, straight from metadata.

    `_covers_per_day` is arithmetic over seats / service style / price tier /
    popular-times shape — no LLM call and no API key, so this script stays
    runnable before the corpus has been swept. Imported lazily because this is a
    plotting script and should not drag langchain in when a cache exists.
    """
    from src.nodes.forecast import _covers_per_day  # noqa: PLC0415 — see docstring

    covers = _covers_per_day(load_restaurant(slug))
    return sum(covers) / 7.0


def resolve_prior() -> tuple[str, float, str]:
    """(slug, mean daily covers, provenance) for the first Tier A restaurant we can price."""
    slugs = list_corpus("A")
    if not slugs:
        raise RuntimeError("No Tier A restaurants in data/restaurants/index.json")

    for slug in slugs:
        cached = _prior_from_cache(slug)
        if cached:
            return slug, cached, "cached run"

    slug = slugs[0]
    return slug, _prior_from_snapshot(slug), "snapshot metadata (no cached run yet)"


def week_shape(slug: str) -> list[float]:
    """Mon..Sun multipliers with mean exactly 1.0, from the snapshot if it has one."""
    try:
        index = [float(v) for v in load_restaurant(slug).get("popular_times_index") or []]
    except Exception:
        index = []

    if len(index) != 7 or sum(index) <= 0:
        index = list(DEFAULT_WEEK_SHAPE)

    mean = sum(index) / 7.0
    return [v / mean for v in index]


def synthetic_actuals(prior: float, shape: list[float], days: int = DAYS) -> list[float]:
    """`days` of SYNTHETIC daily covers: offset truth x weekly shape x trend + noise."""
    rng = random.Random(SEED)
    level = prior * TRUTH_OFFSET

    series = []
    for day in range(days):
        expected = level * shape[day % 7] * (1.0 + DAILY_TREND) ** day
        series.append(max(expected + rng.gauss(0.0, NOISE_SD * expected), 0.0))
    return series


def walk(prior: float, actuals: list[float], k: float = PRIOR_STRENGTH_K) -> list[float]:
    """Blended estimate after each day. Index 0 is day zero: no data, pure prior."""
    estimates = [blend(prior, prior, 0, k=k)]
    running = 0.0
    for n, value in enumerate(actuals, start=1):
        running += value
        estimates.append(blend(prior, running / n, n, k=k))
    return estimates


def draw(slug: str, name: str, prior: float, actuals: list[float], estimates: list[float]) -> None:
    """Three series on one axis: flat prior, synthetic actuals, blended estimate."""
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    days_actual = list(range(1, len(actuals) + 1))
    days_estimate = list(range(len(estimates)))
    right = len(actuals) + 10.5

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=200)

    # Grid and frame recede: they orient, they do not compete with the marks.
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=10, length=0)

    # Bounds set before any annotation is placed, so labels can be positioned in
    # data coordinates without matplotlib re-fitting the view around them.
    low = min(min(actuals), prior) * 0.88
    high = max(actuals) * 1.06
    ax.set_ylim(low, high)

    # n == k: the 50/50 point. Marking it makes `k` readable off the chart itself.
    ax.axvline(PRIOR_STRENGTH_K, color=AXIS, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    ax.text(
        PRIOR_STRENGTH_K + 0.5, low + (high - low) * 0.015,
        "n = k = 14 days\n50/50 blend",
        color=INK_MUTED, fontsize=9, va="bottom", ha="left", linespacing=1.3, zorder=1,
    )

    ax.plot(
        [0, len(actuals)], [prior, prior],
        color=C_PRIOR, linewidth=2.0, linestyle=(0, (5, 3)), zorder=3,
    )
    ax.plot(
        days_actual, actuals,
        color=C_ACTUALS, linewidth=1.2, marker="o", markersize=3.2,
        markeredgecolor=SURFACE, markeredgewidth=0.6, alpha=0.9, zorder=2,
    )
    # Reference line, not a fourth series: without it the eye cannot tell where a
    # series that swings 45-117 covers actually averages, and "the blend lands on
    # the actuals" is exactly the claim the chart has to let you check.
    actual_mean = sum(actuals) / len(actuals)
    # Stops at day 42 rather than spanning the axes: the right margin belongs to
    # the direct labels, and a rule running through them is the classic collision.
    ax.plot(
        [0, len(actuals)], [actual_mean, actual_mean],
        color=C_ACTUALS, linewidth=1.0, linestyle=(0, (1, 3)), alpha=0.75, zorder=1,
    )
    ax.text(
        0.4, actual_mean + (high - low) * 0.012,
        f"mean of the synthetic actuals, {actual_mean:.0f}/day",
        color=C_ACTUALS, fontsize=9, va="bottom", ha="left", zorder=1,
    )

    ax.plot(
        days_estimate, estimates,
        color=C_BLEND, linewidth=2.8, solid_capstyle="round", zorder=4,
    )

    # Direct labels: identity never rests on color alone, and the aqua sits below
    # 3:1 on this surface so the dataviz relief rule requires a visible label.
    label_x = len(actuals) + 1.2
    ax.text(
        label_x, prior - (high - low) * 0.045, "Cold-start prior\n(metadata only)",
        color=C_PRIOR, fontsize=10, va="top", ha="left", linespacing=1.35,
    )
    ax.text(
        label_x, high - (high - low) * 0.06, "Synthetic actuals\n(seeded — no POS feed)",
        color=C_ACTUALS, fontsize=10, va="top", ha="left", linespacing=1.35,
    )
    ax.text(
        label_x, estimates[-1] + (high - low) * 0.035,
        "Blended estimate\nblend(prior, actuals, n)",
        color=C_BLEND, fontsize=10, fontweight="bold", va="bottom", ha="left", linespacing=1.35,
    )

    ax.set_xlim(0, right)
    ax.set_xlabel("Days of real sales observed", color=INK_SECONDARY, fontsize=11, labelpad=9)
    ax.set_ylabel("Covers per day", color=INK_SECONDARY, fontsize=11, labelpad=9)
    ax.set_xticks(list(range(0, len(actuals) + 1, 7)))

    ax.set_title(
        f"The cold-start prior is replaced by actuals within about two weeks — {name}",
        color=INK, fontsize=15, fontweight="bold", loc="left", pad=34,
    )
    # The word "synthetic" has to survive a screenshot with no narration attached.
    ax.text(
        0.0, 1.045,
        "SYNTHETIC actuals — mechanism demonstration, not a measured result",
        transform=ax.transAxes, color=INK_SECONDARY, fontsize=12, va="bottom", ha="left",
    )

    fig.text(
        0.012, 0.055,
        f"blend(prior, actual_mean, n) = (1−w)·prior + w·actual_mean,  w = n/(n+k),  "
        f"k = {PRIOR_STRENGTH_K:.0f} days  ·  "
        f"prior = {prior:.0f} covers/day from {slug} public metadata, no sales history",
        color=INK_MUTED, fontsize=8.5, va="bottom", ha="left",
    )
    fig.text(
        0.012, 0.018,
        f"Synthetic series: truth offset +{(TRUTH_OFFSET - 1) * 100:.0f}% from the prior, "
        f"weekly shape from popular times, +{DAILY_TREND * 100:.2f}%/day trend, "
        f"{NOISE_SD * 100:.0f}% daily noise, random.Random({SEED}) — no real POS data exists in this build",
        color=INK_MUTED, fontsize=8.5, va="bottom", ha="left",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    fig.savefig(OUTPUT_PATH, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    slug, prior, provenance = resolve_prior()
    name = load_restaurant(slug).get("name", slug)

    actuals = synthetic_actuals(prior, week_shape(slug))
    estimates = walk(prior, actuals)
    actual_mean = sum(actuals) / len(actuals)

    draw(slug, name, prior, actuals, estimates)

    print(f"Prior source        : {slug} ({provenance})")
    print(f"Prior               : {prior:.1f} covers/day")
    print(f"Synthetic actuals   : mean {actual_mean:.1f} covers/day over {len(actuals)} days "
          f"(SEEDED — no POS data exists)")
    print(f"Blended day 0       : {estimates[0]:.1f}  (exactly the prior)")
    print(f"Blended day 14      : {estimates[14]:.1f}  (n = k, halfway)")
    print(f"Blended day {len(actuals):<2}      : {estimates[-1]:.1f}  "
          f"({abs(estimates[-1] - actual_mean) / actual_mean * 100:.1f}% from the actuals mean)")
    print(f"Wrote               : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
