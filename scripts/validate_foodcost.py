"""Step 017 — food-cost validation harness.

The value of this harness comes entirely from what it refuses to claim. The
demand forecast cannot be validated: there is no POS history to check it
against, and any claim otherwise would be fabricated. The costing *can* be
validated, and doing so exercises the whole chain underneath it — extraction, recipe decomposition,
canonicalization, unit conversion, catalog pricing.

The logic: menu price is transcribed from the restaurant's own menu, plate cost
is computed bottom-up from decomposed quantities and catalog prices. Nothing
ties the two together in the code — they arrive from opposite ends of the
pipeline. Their ratio is the implied food cost percentage, and the restaurant
industry runs food cost at roughly 28-33% of revenue (National Restaurant
Association, 2026). So if the computed distribution across unrelated
restaurants lands in that band, the maths underneath is probably sound. If it
lands at 12% or 60%, something upstream is broken and this found it.

  THE BAND IS AN EXTERNAL BENCHMARK, FIXED BEFORE ANY NUMBER HERE WAS
  COMPUTED. It is sourced from the National Restaurant Association's 2026
  figures and is not derived from this corpus. A benchmark adjusted after
  seeing the result it is meant to test measures nothing.

Low-coverage plates are excluded from the headline statistic and the exclusion
is counted and printed. A plate costed from three of its eight ingredients is
always artificially cheap, and including it silently would flatter the result.
"Median 31%, excluding 37 plates where under half the ingredients could be
costed" is a stronger sentence than a better-looking number.

Each restaurant's costed state is cached to `data/output/<slug>-costed.json`
and reused unless --force is passed. This gets run repeatedly while tuning the
catalog, and re-paying for a corpus of extraction each time is slow and
expensive. The cache file is also the input to Step 018's sweep report, so its
shape is a contract: see `_write_cache`.

Usage:
    uv run python -m scripts.validate_foodcost
    uv run python -m scripts.validate_foodcost --tier all --force
    uv run python -m scripts.validate_foodcost --slug madame-vo --slug hanoi-house
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from langgraph.errors import GraphInterrupt

from scripts.run_dayzero import DRY_RUN_STOP_STAGE, _UsageCollector
from src.graph import get_compiled_graph
from src.state import initial_dayzero_state
from src.tools.snapshot import list_corpus, load_restaurant

OUTPUT_DIR = Path("data/output")

# A plate costed from under half its ingredients is excluded from the headline
# statistic. Not a tuning knob — it is the threshold below which the ratio stops
# describing the plate and starts describing the catalog's gaps.
COVERAGE_FLOOR = 0.5

# National Restaurant Association 2026: food and labour each run about 33c of
# every sales dollar. Fixed before any number below was computed.
BAND_LOW, BAND_HIGH = 0.28, 0.33

# The wider band that still describes a real restaurant rather than a bug.
PLAUSIBLE_LOW, PLAUSIBLE_HIGH = 0.20, 0.45

# The graph runs 8 nodes and each may retry up to MAX_RETRIES; LangGraph's
# default recursion limit of 25 supersteps can be reached legitimately on a
# restaurant that retries twice. The CLI does not raise it because a demo run
# that spirals should stop; an unattended sweep should not lose a restaurant to
# a limit that has nothing to do with the restaurant.
RECURSION_LIMIT = 60


def _cache_path(slug: str) -> Path:
    return OUTPUT_DIR / f"{slug}-costed.json"


def _run_one(slug: str, stamp: int) -> dict[str, Any]:
    """Run one restaurant to `costed` and return the cache record.

    Mirrors `scripts/run_dayzero.py --dry-run` deliberately rather than shelling
    out to it: the sweep needs the token usage and wall clock back, and the CLI
    prints those rather than returning them. The stop condition, the Slack
    disabling and the usage callback are all the CLI's, reused as imports.
    """
    # A dry run must never reach Slack. The WebClient is bound once at import
    # time from settings, so the degraded path is forced by unbinding it.
    import src.slack.client as slack_client

    slack_client._client = None

    # A fresh thread per run, every run. `messages` carries the `add_messages`
    # reducer (src/state.py:20), so streaming a fresh input into a thread that
    # has already run does NOT clear the history — it appends, and the next
    # restaurant's nodes would see the previous attempt's error messages.
    thread_id = f"sweep-{slug}-{stamp}"
    usage = _UsageCollector()
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [usage],
        "recursion_limit": RECURSION_LIMIT,
    }

    graph = get_compiled_graph()
    started = time.monotonic()
    final: dict[str, Any] = {}
    error = ""

    try:
        for values in graph.stream(
            initial_dayzero_state(slug), config=config, stream_mode="values"
        ):
            final = values
            if values.get("stage") == DRY_RUN_STOP_STAGE:
                break
    except GraphInterrupt:
        # Reached a human gate before `costed` — the circuit breaker tripped.
        # That is a result about this restaurant, not a crash of the sweep.
        error = "halted at a human gate before costing"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.monotonic() - started
    if not error and final.get("stage") != DRY_RUN_STOP_STAGE:
        error = f"ended at stage {final.get('stage') or '(none)'} instead of {DRY_RUN_STOP_STAGE}"

    meta = load_restaurant(slug)
    return {
        "slug": slug,
        "tier": meta.get("tier"),
        "cuisine": meta.get("cuisine"),
        "menu_format": meta.get("menu_format"),
        "elapsed_s": round(elapsed, 1),
        "usage": {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read": usage.cache_read,
            "cache_write": usage.cache_write,
        },
        "error": error,
        "state": {k: v for k, v in final.items() if k not in ("messages", "__interrupt__")},
    }


def _write_cache(record: dict[str, Any]) -> None:
    """Persist one restaurant's costed run.

    Shape is a contract with `scripts/sweep_report.py`:
        slug, tier, cuisine, menu_format, elapsed_s,
        usage {calls, input_tokens, output_tokens, cache_read, cache_write},
        error (empty string when the run reached `costed`),
        state {menu_items, recipes, sku_matches, plate_costs, review_queue, ...}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(record["slug"])
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def recost(slugs: list[str]) -> int:
    """Recompute plate costs on the cached states, without re-running the LLM.

    `cost_plates_node` is pure arithmetic over the catalog, the units table and
    the state's own `recipes` / `sku_matches` — no LLM call, no network. So a
    change to catalog prices or to a SKU's `conversions` map does NOT require
    re-paying for extraction: the expensive half of the pipeline is already on
    disk. This runs the cheap half again, in seconds rather than an hour.

    `review_queue` is rebuilt rather than appended to. `cost_plates_node` adds
    its newly-flagged plates to whatever queue it is handed, so feeding it the
    cached queue would duplicate every flag on every recost.
    """
    from src.nodes.cost_plates import cost_plates_node

    touched = 0
    for slug in slugs:
        path = _cache_path(slug)
        if not path.is_file():
            print(f"  {slug:<28} no cache - skipped", flush=True)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        state = dict(record.get("state") or {})
        if not state.get("recipes"):
            print(f"  {slug:<28} no recipes in cache - skipped", flush=True)
            continue

        before = [p for p in state.get("plate_costs") or [] if p.get("costable", True)]
        mean_before = (
            sum(float(p.get("coverage") or 0.0) for p in before) / len(before)
            if before
            else 0.0
        )

        state["review_queue"] = []
        state.update(cost_plates_node(state))
        record["state"] = state
        _write_cache(record)

        after = [p for p in state.get("plate_costs") or [] if p.get("costable", True)]
        mean_after = (
            sum(float(p.get("coverage") or 0.0) for p in after) / len(after)
            if after
            else 0.0
        )
        print(
            f"  {slug:<28} coverage {mean_before:.2f} -> {mean_after:.2f}"
            f"  ({mean_after - mean_before:+.2f})",
            flush=True,
        )
        touched += 1
    return touched


def collect(slugs: list[str], force: bool) -> list[dict[str, Any]]:
    """Load or produce one cache record per slug, narrating as it goes."""
    stamp = int(time.time())
    records = []
    for i, slug in enumerate(slugs, 1):
        path = _cache_path(slug)
        cached = None
        if path.is_file() and not force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            # A run that failed is not a result worth keeping. The first sweep
            # lost 15 restaurants to an exhausted API balance and wrote a cache
            # file for every one of them; treating those as "done" would make
            # the failure permanent and silent, and the report would quietly
            # describe 2 restaurants while claiming to describe 20.
            if cached.get("error"):
                cached = None

        if cached is not None:
            record = cached
            plates = len(record.get("state", {}).get("plate_costs") or [])
            print(f"[{i:2}/{len(slugs)}] {slug:<28} cached  · {plates} plates", flush=True)
        else:
            print(f"[{i:2}/{len(slugs)}] {slug:<28} running ...", end=" ", flush=True)
            record = _run_one(slug, stamp)
            _write_cache(record)
            plates = len(record.get("state", {}).get("plate_costs") or [])
            note = record["error"] or f"{plates} plates"
            print(f"{record['elapsed_s']:.0f}s · {note}", flush=True)
        records.append(record)
    return records


def _plates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every plate that produced a ratio, tagged with its restaurant.

    `food_cost_pct is not None` already excludes plates nothing could be costed
    for — `cost_plates` writes None rather than 0.0 there precisely so that a
    plate we could not cost is never mistaken for a plate that costs nothing.
    """
    out = []
    for r in records:
        for p in r.get("state", {}).get("plate_costs") or []:
            if p.get("food_cost_pct") is None:
                continue
            out.append({**p, "slug": r["slug"], "cuisine": r.get("cuisine")})
    return out


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. n is small enough that interpolation is noise."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _share(values: list[float], low: float, high: float) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if low <= v <= high) / len(values)


def report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Print the summary and the two breakdowns. Returns the headline stats."""
    all_plates = _plates(records)
    included = [p for p in all_plates if float(p.get("coverage") or 0.0) >= COVERAGE_FLOOR]
    excluded = len(all_plates) - len(included)
    ratios = [float(p["food_cost_pct"]) for p in included]

    ok = [r for r in records if not r.get("error")]
    failed = [r for r in records if r.get("error")]

    print()
    print("=" * 72)
    print(f"FOOD COST VALIDATION · {len(ok)} restaurants costed"
          + (f" · {len(failed)} failed" if failed else ""))
    print("=" * 72)

    if not ratios:
        print("no plates produced a food cost ratio — nothing to validate")
        return {}

    median = statistics.median(ratios)
    mean = statistics.fmean(ratios)
    print(f"  plates with a ratio  : {len(all_plates)}")
    print(f"  excluded (coverage<{COVERAGE_FLOOR:.0%}) : {excluded}")
    print(f"  included in headline : {len(ratios)}")
    print()
    print(f"  median food cost     : {median:.1%}   <- the headline number")
    print(f"  mean                 : {mean:.1%}")
    print(f"  p10 / p90            : {_pct(ratios, 0.10):.1%} / {_pct(ratios, 0.90):.1%}")
    print()
    print(f"  inside {BAND_LOW:.0%}-{BAND_HIGH:.0%} industry band : "
          f"{_share(ratios, BAND_LOW, BAND_HIGH):.1%}")
    print(f"  inside {PLAUSIBLE_LOW:.0%}-{PLAUSIBLE_HIGH:.0%} plausible band: "
          f"{_share(ratios, PLAUSIBLE_LOW, PLAUSIBLE_HIGH):.1%}")

    verdict = (
        "PASS — lands where the industry actually sits"
        if 0.25 <= median <= 0.38
        else "MISS — fix the upstream node and rerun with --force; do NOT move the band"
    )
    print(f"\n  verdict              : {verdict}")

    # Per restaurant. Coverage is over costable plates only, matching the node.
    print("\n" + "-" * 72)
    print(f"  {'restaurant':<28} {'cuisine':<18} {'n':>4} {'median':>8} {'coverage':>9}")
    print("-" * 72)
    for r in records:
        plates = r.get("state", {}).get("plate_costs") or []
        costable = [p for p in plates if p.get("costable", True)]
        rs = [
            float(p["food_cost_pct"])
            for p in plates
            if p.get("food_cost_pct") is not None
            and float(p.get("coverage") or 0.0) >= COVERAGE_FLOOR
        ]
        cov = statistics.fmean([float(p.get("coverage") or 0.0) for p in costable]) if costable else 0.0
        med = f"{statistics.median(rs):.1%}" if rs else "--"
        note = r.get("error") or ""
        print(f"  {r['slug']:<28} {(r.get('cuisine') or '?'):<18} {len(rs):>4} "
              f"{med:>8} {cov:>9.2f}  {note}")

    # Per cuisine. This is where the interesting finding lives: coverage
    # degrades on cuisines the catalog under-serves, and naming that precisely
    # is worth more than a clean aggregate.
    print("\n" + "-" * 72)
    print(f"  {'cuisine':<20} {'restaurants':>12} {'n plates':>9} {'median':>8} {'coverage':>9}")
    print("-" * 72)
    cuisines: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        cuisines.setdefault(r.get("cuisine") or "?", []).append(r)
    for cuisine, group in sorted(cuisines.items()):
        rs, covs = [], []
        for r in group:
            for p in r.get("state", {}).get("plate_costs") or []:
                if p.get("costable", True):
                    covs.append(float(p.get("coverage") or 0.0))
                if (
                    p.get("food_cost_pct") is not None
                    and float(p.get("coverage") or 0.0) >= COVERAGE_FLOOR
                ):
                    rs.append(float(p["food_cost_pct"]))
        med = f"{statistics.median(rs):.1%}" if rs else "--"
        cov = statistics.fmean(covs) if covs else 0.0
        print(f"  {cuisine:<20} {len(group):>12} {len(rs):>9} {med:>8} {cov:>9.2f}")

    print()
    return {
        "n_plates_total": len(all_plates),
        "n_included": len(ratios),
        "n_excluded": excluded,
        "n_restaurants": len(ok),
        "median": median,
        "mean": mean,
        "p10": _pct(ratios, 0.10),
        "p90": _pct(ratios, 0.90),
        "share_band": _share(ratios, BAND_LOW, BAND_HIGH),
        "share_plausible": _share(ratios, PLAUSIBLE_LOW, PLAUSIBLE_HIGH),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_foodcost",
        description="Run the corpus to `costed` and check the implied food cost distribution.",
    )
    parser.add_argument(
        "--tier",
        default="B",
        choices=["A", "B", "all"],
        help="corpus tier to sweep (default: B)",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        metavar="SLUG",
        help="run only this slug; repeatable, overrides --tier",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when data/output/<slug>-costed.json exists",
    )
    parser.add_argument(
        "--recost",
        action="store_true",
        help=(
            "re-run costing on the cached states only (no LLM, no extraction). "
            "Use after editing catalog prices or a SKU's conversions map."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.slug:
        corpus = list_corpus()
        unknown = [s for s in args.slug if s not in corpus]
        if unknown:
            print(f"unknown slug(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        slugs = args.slug
    else:
        slugs = list_corpus(None if args.tier == "all" else args.tier)

    print(f"=== food cost validation · {len(slugs)} restaurants "
          f"(tier {args.tier}{', forced' if args.force else ''}) ===\n")

    if args.recost:
        print("recosting cached states (no LLM calls)\n")
        touched = recost(slugs)
        print(f"\n{touched} restaurant(s) recosted\n")

    records = collect(slugs, args.force)
    stats = report(records)
    return 0 if stats else 1


if __name__ == "__main__":
    raise SystemExit(main())
