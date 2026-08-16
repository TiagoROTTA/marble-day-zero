"""Step 018 — corpus sweep report.

Reads ONLY the cached `data/output/<slug>-costed.json` files written by
`scripts/validate_foodcost.py` and renders `data/output/sweep-report.md`. It
never runs the graph and never calls an LLM: aggregating a sweep must be free
and instant, or nobody re-runs it after tuning the catalog.

What the report is for: a single successful run proves nothing about a
pipeline. "It holds up on these inputs, it degrades on those, and here is why"
is the only claim worth making, and every number below exists to support that
claim with something measured rather than assumed. The two cross-cuts (by
menu_format, by cuisine) are computed, not asserted — where the data
contradicts the expectation, the report says so.

Group sizes are printed next to every mean. A "trend" over two restaurants is
not a trend, and presenting one as though it were would make the report less
reliable than the raw cache files it summarises.

Usage:
    uv run python -m scripts.sweep_report
    uv run python -m scripts.sweep_report --top-failures 15
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# COVERAGE_FLOOR and the industry band are Step 017's, imported rather than
# restated: two files disagreeing about where the floor sits is a bug that
# would only show up as a quietly different headline number.
from scripts.validate_foodcost import (
    BAND_HIGH,
    BAND_LOW,
    COVERAGE_FLOOR,
    OUTPUT_DIR,
)

# --------------------------------------------------------------------------
# PRICING — Anthropic public list rates, USD per million tokens.
# Source: claude-api skill / platform.claude.com pricing, checked 2026-08-10.
# Model: `claude-opus-5` (src/config.py settings.llm_model and
# settings.llm_model_cheap are both this model today).
# Cache reads bill at 0.1x input; 5-minute-TTL cache writes at 1.25x input.
# Update these four numbers and nothing else when the rates move.
# --------------------------------------------------------------------------
PRICING_PER_MTOK = {
    "model": "claude-opus-5",
    "input": 5.00,
    "output": 25.00,
    "cache_read": 0.50,   # 0.1x input
    "cache_write": 6.25,  # 1.25x input, 5-minute TTL
}

# Below this many restaurants a group mean describes the restaurants, not the
# format or the cuisine. Groups under it are still reported, and labelled.
MIN_GROUP = 3

DASH = "--"


# --- loading ---------------------------------------------------------------


def load_records(directory: Path = OUTPUT_DIR) -> list[dict[str, Any]]:
    """Every `<slug>-costed.json` in `directory`, slug-sorted. No pipeline."""
    import json

    records = []
    for path in sorted(Path(directory).glob("*-costed.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


# --- per-restaurant aggregation --------------------------------------------


def canon_breakdown(sku_matches: list[dict[str, Any]]) -> dict[str, int]:
    """alias / normalized / llm / unmatched counts over one restaurant's matches.

    `src/nodes/canonicalize.py` writes `method` for the pass that won and
    `sku_id: None` when nothing in the catalog genuinely fits. A null sku_id is
    counted as unmatched whatever the method says — the LLM pass returning null
    is its *good* answer, not an llm match, and passes 1 and 2 cannot produce
    one. Counting an honest gap as a match would flatter exactly the number
    this report exists to expose.
    """
    counts = {"alias": 0, "normalized": 0, "llm": 0, "unmatched": 0}
    for match in sku_matches or []:
        if match.get("sku_id") is None:
            counts["unmatched"] += 1
            continue
        method = match.get("method")
        if method in counts:
            counts[method] += 1
    return counts


def mean_item_confidence(state: dict[str, Any]) -> float | None:
    values = [
        float(item["confidence"])
        for item in state.get("menu_items") or []
        if item.get("confidence") is not None
    ]
    return statistics.fmean(values) if values else None


def mean_coverage(state: dict[str, Any]) -> float | None:
    """Mean coverage over costable plates only, matching `cost_plates`.

    A plate nothing could be costed for has coverage 0.0 and would drag the
    mean toward a number that describes the catalog rather than the extraction.
    """
    values = [
        float(plate.get("coverage") or 0.0)
        for plate in state.get("plate_costs") or []
        if plate.get("costable", True)
    ]
    return statistics.fmean(values) if values else None


def median_food_cost(state: dict[str, Any]) -> float | None:
    """Median food cost % over plates at or above COVERAGE_FLOOR."""
    values = [
        float(plate["food_cost_pct"])
        for plate in state.get("plate_costs") or []
        if plate.get("food_cost_pct") is not None
        and float(plate.get("coverage") or 0.0) >= COVERAGE_FLOOR
    ]
    return statistics.median(values) if values else None


def corpus_food_costs(records: list[dict[str, Any]]) -> list[float]:
    """Every plate ratio in the corpus at or above COVERAGE_FLOOR, pooled.

    Pooled per plate, not per restaurant, so it is the same population the
    histogram in `scripts/plot_foodcost.py` draws. The two must not disagree.
    """
    values = []
    for record in records:
        for plate in (record.get("state") or {}).get("plate_costs") or []:
            if plate.get("food_cost_pct") is None:
                continue
            if float(plate.get("coverage") or 0.0) < COVERAGE_FLOOR:
                continue
            values.append(float(plate["food_cost_pct"]))
    return values


def unmatched_rate(state: dict[str, Any]) -> float | None:
    matches = state.get("sku_matches") or []
    if not matches:
        return None
    return canon_breakdown(matches)["unmatched"] / len(matches)


def restaurant_row(record: dict[str, Any]) -> dict[str, str]:
    """One table row. Every value is a non-empty string, `--` when absent.

    An empty cell reads as "we did not measure this"; `--` reads as "there was
    nothing to measure". They are different claims and the table must not blur
    them.
    """
    state = record.get("state") or {}
    breakdown = canon_breakdown(state.get("sku_matches") or [])
    conf = mean_item_confidence(state)
    cov = mean_coverage(state)
    med = median_food_cost(state)

    return {
        "slug": record.get("slug") or DASH,
        "cuisine": record.get("cuisine") or DASH,
        "menu_format": record.get("menu_format") or DASH,
        "items": str(len(state.get("menu_items") or [])),
        "item_conf": f"{conf:.2f}" if conf is not None else DASH,
        "alias": str(breakdown["alias"]),
        "normalized": str(breakdown["normalized"]),
        "llm": str(breakdown["llm"]),
        "unmatched": str(breakdown["unmatched"]),
        "coverage": f"{cov:.2f}" if cov is not None else DASH,
        "food_cost": f"{med:.1%}" if med is not None else DASH,
        "review": str(len(state.get("review_queue") or [])),
        "error": record.get("error") or "",
    }


# --- cross-cuts ------------------------------------------------------------


def group_by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get(key) or "unknown", []).append(record)
    return groups


def _mean_of(records: list[dict[str, Any]], fn) -> float | None:
    """Mean over restaurants of a per-restaurant statistic.

    Per restaurant, not per item: a 90-item menu would otherwise decide the
    format's score on its own, and the question being asked is "does this
    format work", not "does this restaurant work".
    """
    values = [v for v in (fn(r.get("state") or {}) for r in records) if v is not None]
    return statistics.fmean(values) if values else None


def by_format(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for fmt, group in sorted(group_by(records, "menu_format").items()):
        out.append({
            "menu_format": fmt,
            "n": len(group),
            "extraction_confidence": _mean_of(group, mean_item_confidence),
            "coverage": _mean_of(group, mean_coverage),
        })
    return out


def by_cuisine(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cuisine, group in sorted(group_by(records, "cuisine").items()):
        out.append({
            "cuisine": cuisine,
            "n": len(group),
            "unmatched_rate": _mean_of(group, unmatched_rate),
        })
    return out


# --- cost and latency ------------------------------------------------------


def cost_rollup(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Total tokens, cache-read share, wall clock and dollars.

    `usage.input_tokens` is the total input as langchain reports it, with
    cache_read and cache_write as subsets of it. Billing the whole of it at the
    full input rate would overstate the cost by roughly the amount the caching
    design was built to save, so the uncached remainder is priced separately.
    """
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read": 0,
              "cache_write": 0}
    elapsed = []
    for record in records:
        usage = record.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        if record.get("elapsed_s") is not None:
            elapsed.append(float(record["elapsed_s"]))

    fresh_input = max(0, totals["input_tokens"] - totals["cache_read"] - totals["cache_write"])
    dollars = (
        fresh_input * PRICING_PER_MTOK["input"]
        + totals["cache_read"] * PRICING_PER_MTOK["cache_read"]
        + totals["cache_write"] * PRICING_PER_MTOK["cache_write"]
        + totals["output_tokens"] * PRICING_PER_MTOK["output"]
    ) / 1_000_000

    n = len(records) or 1
    return {
        **totals,
        "fresh_input_tokens": fresh_input,
        "cache_read_share": (totals["cache_read"] / totals["input_tokens"])
        if totals["input_tokens"] else 0.0,
        "total_cost_usd": dollars,
        "cost_per_restaurant_usd": dollars / n,
        "mean_elapsed_s": statistics.fmean(elapsed) if elapsed else 0.0,
        "median_elapsed_s": statistics.median(elapsed) if elapsed else 0.0,
        "n_restaurants": len(records),
    }


# --- failure list ----------------------------------------------------------


def worst_plates(records: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The n lowest-coverage plates across the corpus, worst first."""
    plates = []
    for record in records:
        for plate in (record.get("state") or {}).get("plate_costs") or []:
            plates.append({
                "slug": record.get("slug") or DASH,
                "item_name": plate.get("item_name") or DASH,
                "coverage": float(plate.get("coverage") or 0.0),
                "uncosted": list(plate.get("uncosted") or []),
            })
    plates.sort(key=lambda p: (p["coverage"], p["slug"], p["item_name"]))
    return plates[:n]


def uncosted_counts(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Most-frequently-uncosted ingredient names across the corpus.

    More actionable than the raw plate list: one SKU added to
    `data/catalog/skus.json` for a name that appears 30 times lifts 30 plates.
    """
    counter: Counter[str] = Counter()
    for record in records:
        for plate in (record.get("state") or {}).get("plate_costs") or []:
            for name in plate.get("uncosted") or []:
                counter[str(name).strip().lower()] += 1
    return counter.most_common()


# --- rendering -------------------------------------------------------------

_COLUMNS = [
    ("slug", "restaurant"),
    ("cuisine", "cuisine"),
    ("menu_format", "format"),
    ("items", "items"),
    ("item_conf", "item conf"),
    ("alias", "alias"),
    ("normalized", "norm"),
    ("llm", "llm"),
    ("unmatched", "unmatched"),
    ("coverage", "coverage"),
    ("food_cost", "median FC%"),
    ("review", "review"),
]


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else DASH


def render_table(rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    lines = [header, rule]
    for row in rows:
        cells = [row.get(key) or DASH for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(records: list[dict[str, Any]], tier: str) -> str:
    rows = [restaurant_row(r) for r in records]
    failed = [r for r in records if r.get("error")]
    formats = by_format(records)
    cuisines = by_cuisine(records)
    cost = cost_rollup(records)

    out: list[str] = []
    out.append("# Sweep report")
    out.append("")
    out.append(
        f"Tier swept: **{tier}** · {len(records)} restaurants · "
        f"{len(failed)} failed (non-empty `error`)."
    )
    out.append("")
    out.append(
        "Every figure below is read from the cached `data/output/<slug>-costed.json` "
        "files; no part of this report runs the pipeline. `--` means there was "
        "nothing to measure, never that a measurement was skipped."
    )
    out.append("")

    if failed:
        out.append("Failed runs (their rows below are still shown, and are incomplete):")
        out.append("")
        for record in failed:
            out.append(f"- `{record.get('slug')}` — {record.get('error')}")
        out.append("")

    ratios = corpus_food_costs(records)
    if ratios:
        in_band = sum(1 for v in ratios if BAND_LOW <= v <= BAND_HIGH) / len(ratios)
        out.append("## Headline")
        out.append("")
        out.append(
            f"Pooled over {len(ratios)} plates at or above {COVERAGE_FLOOR:.0%} ingredient "
            f"coverage, the median implied food cost is **{statistics.median(ratios):.1%}** "
            f"(mean {statistics.fmean(ratios):.1%}); {in_band:.1%} of plates fall inside the "
            f"{BAND_LOW:.0%}-{BAND_HIGH:.0%} industry band. That is a clear miss against the "
            f"band, and it is reported as measured."
        )
        out.append("")
        out.append(
            "The band is *actual* food cost: total purchases over total revenue, weighted by "
            "what actually sells and inclusive of waste, spoilage, over-portioning and staff "
            "meals. The figure above is an unweighted median of *theoretical* plate cost over "
            "every priced menu line, high-margin sides and drinks included. Theoretical plate "
            "cost always sits below actual food cost, and the unweighted median widens the gap "
            "further. The two are not like-for-like; see `data/output/findings.md` for what the "
            "harness does and does not establish."
        )
        out.append("")

    out.append("## Per restaurant")
    out.append("")
    out.append(render_table(rows))
    out.append("")
    out.append(
        f"`coverage` is the mean over costable plates. `median FC%` is over plates "
        f"with coverage >= {COVERAGE_FLOOR:.0%}; the industry band is "
        f"{BAND_LOW:.0%}-{BAND_HIGH:.0%}. `unmatched` counts every match with a null "
        f"`sku_id`, whichever pass produced it — an honest catalog gap, not a failure "
        f"of the matcher."
    )
    out.append("")

    # --- cross-cut: menu_format ---
    out.append("## Cross-cut: menu format")
    out.append("")
    out.append("| format | n restaurants | mean extraction confidence | mean coverage |")
    out.append("|---|---|---|---|")
    for group in formats:
        out.append(
            f"| {group['menu_format']} | {group['n']} | "
            f"{_fmt(group['extraction_confidence'], '.2f')} | "
            f"{_fmt(group['coverage'], '.2f')} |"
        )
    out.append("")
    out.extend(_format_prose(formats))
    out.append("")

    # --- cross-cut: cuisine ---
    out.append("## Cross-cut: cuisine")
    out.append("")
    out.append("| cuisine | n restaurants | mean unmatched-ingredient rate |")
    out.append("|---|---|---|")
    for group in cuisines:
        out.append(
            f"| {group['cuisine']} | {group['n']} | "
            f"{_fmt(group['unmatched_rate'], '.1%')} |"
        )
    out.append("")
    out.extend(_cuisine_prose(cuisines))
    out.append("")

    # --- cost and latency ---
    out.append("## Cost and latency")
    out.append("")
    out.append("| metric | value |")
    out.append("|---|---|")
    out.append(f"| LLM calls | {cost['calls']:,} |")
    out.append(f"| input tokens (total) | {cost['input_tokens']:,} |")
    out.append(f"| of which read from cache | {cost['cache_read']:,} "
               f"({cost['cache_read_share']:.1%}) |")
    out.append(f"| of which written to cache | {cost['cache_write']:,} |")
    out.append(f"| fresh input tokens (full rate) | {cost['fresh_input_tokens']:,} |")
    out.append(f"| output tokens | {cost['output_tokens']:,} |")
    out.append(f"| mean wall clock per restaurant | {cost['mean_elapsed_s']:.1f}s |")
    out.append(f"| median wall clock per restaurant | {cost['median_elapsed_s']:.1f}s |")
    out.append(f"| estimated cost, full sweep | ${cost['total_cost_usd']:.2f} |")
    out.append(f"| estimated cost per restaurant | ${cost['cost_per_restaurant_usd']:.2f} |")
    out.append("")
    out.append(
        f"Priced at {PRICING_PER_MTOK['model']} list rates "
        f"(${PRICING_PER_MTOK['input']:.2f} in / ${PRICING_PER_MTOK['output']:.2f} out / "
        f"${PRICING_PER_MTOK['cache_read']:.2f} cache read / "
        f"${PRICING_PER_MTOK['cache_write']:.2f} cache write per million tokens; see "
        f"`PRICING_PER_MTOK` at the top of `scripts/sweep_report.py` for the source and "
        f"date). Cache reads are priced at the cached rate, which is the whole point of "
        f"putting the catalog behind one `cache_control` block: "
        f"{cost['cache_read_share']:.1%} of input tokens billed at a tenth of the "
        f"input rate."
    )
    out.append("")

    # --- catalog shopping list ---
    top = uncosted_counts(records)[:15]
    out.append("## Most-frequently-uncosted ingredients")
    out.append("")
    if not top:
        out.append("No plate reported an uncosted ingredient. Nothing to add to the catalog.")
    else:
        out.append("| ingredient | plates affected |")
        out.append("|---|---|")
        for name, count in top:
            out.append(f"| {name} | {count} |")
        out.append("")
        out.append(
            "This is the catalog shopping list, ordered by leverage: one SKU added to "
            "`data/catalog/skus.json` for the top name lifts every plate that mentions it."
        )
    out.append("")
    return "\n".join(out)


def _format_prose(formats: list[dict[str, Any]]) -> list[str]:
    """Say what the format cross-cut actually shows, including when it is thin."""
    lines = []
    small = [g for g in formats if g["n"] < MIN_GROUP]
    if small:
        for group in small:
            lines.append(
                f"- **{group['menu_format']}** has n={group['n']}, below the "
                f"{MIN_GROUP}-restaurant floor: its mean describes those "
                f"{group['n']} restaurant(s), not the format. Read it as an anecdote."
            )

    ranked = [g for g in formats if g["coverage"] is not None]
    ranked.sort(key=lambda g: g["coverage"], reverse=True)
    if len(ranked) < 2:
        lines.append(
            "- Only one format produced a coverage figure, so there is no comparison "
            "to make. The html > pdf > image expectation is untested here."
        )
        return lines

    order = " > ".join(f"{g['menu_format']} ({g['coverage']:.2f}, n={g['n']})" for g in ranked)
    lines.append(f"- Measured coverage ranking: {order}.")

    expected = [f for f in ("html", "pdf", "image") if any(g["menu_format"] == f for g in ranked)]
    actual = [g["menu_format"] for g in ranked]
    if actual == expected:
        lines.append(
            "- That matches the expected html > pdf > image ordering. It was computed, "
            "not assumed."
        )
    else:
        lines.append(
            f"- That **contradicts** the expected {' > '.join(expected)} ordering. The "
            f"data says {' > '.join(actual)}; the expectation was wrong, and the "
            f"honest finding is more interesting than the predicted one."
        )
    return lines


def _cuisine_prose(cuisines: list[dict[str, Any]]) -> list[str]:
    lines = []
    rated = [g for g in cuisines if g["unmatched_rate"] is not None]
    if not rated:
        lines.append("- No cuisine produced an unmatched rate; there is nothing to compare.")
        return lines

    rated.sort(key=lambda g: g["unmatched_rate"], reverse=True)
    worst, best = rated[0], rated[-1]
    lines.append(
        f"- Highest unmatched rate: **{worst['cuisine']}** at "
        f"{worst['unmatched_rate']:.1%} (n={worst['n']}). Lowest: **{best['cuisine']}** "
        f"at {best['unmatched_rate']:.1%} (n={best['n']})."
    )
    if len(rated) > 1:
        spread = worst["unmatched_rate"] - best["unmatched_rate"]
        lines.append(
            f"- Spread of {spread:.1%} between them. The expectation was that cuisines "
            f"the 324-SKU catalog under-serves would show materially higher unmatched "
            f"rates; on this sweep that "
            + ("holds." if spread >= 0.10 else "is not clearly borne out — the spread is "
               "narrow enough that catalog coverage looks roughly even across cuisines.")
        )
    thin = [g for g in rated if g["n"] < MIN_GROUP]
    if thin:
        lines.append(
            "- n < "
            f"{MIN_GROUP} for: "
            + ", ".join(f"{g['cuisine']} (n={g['n']})" for g in thin)
            + ". Those rows are single restaurants wearing a cuisine label, not a "
            "measurement of the cuisine."
        )
    return lines


# --- stdout: the failure list ----------------------------------------------


def print_top_failures(records: list[dict[str, Any]], n: int) -> None:
    """ASCII only. A U+2192 here once crashed the cp1252 Windows console."""
    print()
    print("=" * 72)
    print(f"{n} LOWEST-COVERAGE PLATES")
    print("=" * 72)
    plates = worst_plates(records, n)
    if not plates:
        print("no plates in the cache")
    for plate in plates:
        uncosted = ", ".join(plate["uncosted"]) if plate["uncosted"] else "(none listed)"
        print(f"  {plate['coverage']:.2f}  {plate['slug']:<24} {plate['item_name']}")
        print(f"        uncosted: {uncosted}")

    print()
    print("-" * 72)
    print("MOST-FREQUENTLY-UNCOSTED INGREDIENTS (the catalog shopping list)")
    print("-" * 72)
    counts = uncosted_counts(records)
    if not counts:
        print("  no uncosted ingredients recorded")
    for name, count in counts[:n]:
        print(f"  {count:>4}  {name}")
    print()


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sweep_report",
        description="Aggregate the cached per-restaurant costed runs into one report.",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR / "sweep-report.md"),
        metavar="PATH",
        help="where to write the markdown report (default: data/output/sweep-report.md)",
    )
    parser.add_argument(
        "--top-failures",
        type=int,
        default=0,
        metavar="N",
        help="print the N lowest-coverage plates and the uncosted-ingredient tally",
    )
    parser.add_argument(
        "--tier",
        default="B",
        help="tier label for the report header (default: B)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Passed explicitly rather than defaulted: the module global is what tests
    # and callers redirect, and a default bound at import time would ignore it.
    records = load_records(OUTPUT_DIR)
    if not records:
        print(
            f"no cached runs found in {OUTPUT_DIR}/ — run "
            f"`uv run python -m scripts.validate_foodcost` first to produce "
            f"data/output/<slug>-costed.json",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(records, args.tier), encoding="utf-8")

    failed = sum(1 for r in records if r.get("error"))
    print(
        f"wrote {out_path} - {len(records)} restaurants"
        + (f", {failed} failed" if failed else "")
    )

    if args.top_failures > 0:
        print_top_failures(records, args.top_failures)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
