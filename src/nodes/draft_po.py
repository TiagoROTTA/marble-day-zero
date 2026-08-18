"""Par levels and the draft opening purchase order.

**This node makes no LLM call.** It is arithmetic joining the forecast to the
recipes to the catalog, and it has to stay deterministic because the number it
produces — a dollar total a restaurant is being asked to approve — must be
defensible line by line. Every line can be traced back:

    item mix share x covers_per_week      -> orders of that menu item per week
    orders x the recipe's qty (converted)  -> weekly consumption in the SKU's uom
    weekly / 7 x days_cover x (1 + safety) -> par level
    ceil(par / pack_qty)                   -> packs
    packs x price_per_pack                 -> line cost

A component the decomposer is not sure about never reaches that arithmetic: a
component confidence below `CONF_REVIEW_FLOOR` is withheld, and the withholding
is written into `purchase_order["excluded_skus"]` with the dish, the confidence
and the money not spent. Withheld, not dropped — an invisible removal would be
the same class of bug as the invisible purchase it replaces.

`src/nodes/cost_plates.py` deliberately does NOT withhold these any more, and
the two nodes are meant to disagree here. Costing produces a statistic: dropping
a hedged-but-expensive line there made the food-cost distribution biased low
(the Adda lamb shank costed $2.97 instead of $10.37, 6% instead of 22%), so
costing includes the component and folds its confidence into the plate's
instead. A purchase order spends money on goods, where the same uncertainty
argues the other way — buying 20 lb of lamb on a 0.50-confidence quantity is a
real cost to a real restaurant. Conservative points in opposite directions for a
number you report and a number you spend. Do not "fix" this by re-syncing them.

`_convert()` is imported from `src.nodes.cost_plates` rather than reimplemented.
That is reuse of an existing function, not a new abstraction: two divergent
copies of unit conversion is exactly the bug that produces an indefensible
total.

Failure handling: any exception is caught; retry_count++ ; the error is fed back
through `last_error`.
"""
import json
import math
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.nodes.cost_plates import _convert
from src.state import CONF_REVIEW_FLOOR, AgentState

CATALOG_PATH = str(Path(__file__).resolve().parents[2] / "data" / "catalog" / "skus.json")

# How many days of stock a category should carry. Perishables get short cover
# because the alternative is a walk-in full of spoilage; dry goods get long cover
# because the only cost of holding them is shelf space and a delivery avoided.
# These are policy, stated here so a chef can argue with the number rather than
# discover it buried in a literal.
DAYS_COVER = {
    "produce": 2,
    "dairy": 3,
    "protein": 3,
    "bakery": 2,
    "dry_goods": 14,
    "spices": 30,
    "oils": 14,
    "beverage": 7,
    "bar": 14,
    # The two entries above split differently in the actual catalog, which ships
    # `oils_condiments` and `beverage_bar` as single categories. Both spellings are
    # kept: dropping the policy names would lose the intent, and dropping the
    # catalog names would silently send every oil and every soda to the fallback.
    "oils_condiments": 14,
    "beverage_bar": 7,
}

# Fallback for a category the policy above has no opinion on. A week is the
# midpoint of the table: an unknown category should get an ordinary order, not
# block the purchase order and not silently get a month of cover.
DEFAULT_DAYS_COVER = 7

# Flat buffer on top of projected consumption. Day one has no history, so the
# forecast is a prior rather than a forecast; 15% is the cheapest insurance
# against running out of an ingredient in week one, and it is flat on purpose —
# a per-category safety factor would be false precision over a prior.
SAFETY_FACTOR = 0.15

# A week's food order should land near 28-33% of projected weekly revenue. This
# node prints the ratio rather than enforcing it: a reading of 5% or 90% means
# the fault is upstream (bad item mix, wrong unit conversion, wrong SKU match)
# and this is the cheapest place in the pipeline to notice.
SANE_RATIO_LOW = 0.28
SANE_RATIO_HIGH = 0.33


def _load_catalog() -> dict[str, dict]:
    """Return the SKU catalog keyed by sku_id."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return {sku["sku_id"]: sku for sku in catalog}


def _consumption(
    state: AgentState, skus: dict[str, dict]
) -> tuple[dict[str, float], list[str], dict[str, dict]]:
    """Pass 1 — weekly consumption per SKU, in that SKU's own uom.

    Returns `(consumption, gaps, withheld)`. `gaps` is the visible record of what
    could not be joined: a missing recipe, an unmatched component, an impossible
    unit conversion. Those quantities are absent from the order, and an absence
    nobody can see is how a purchase order ends up short on a Friday.

    `withheld` is the record of what *could* have been joined and was deliberately
    not bought, keyed by sku_id: `{sku_id, reasons, dishes, confidence, qty}`. Two
    things land there:

      * `skipped_refs` (set by `review_wait_node` in `src/nodes/hitl.py` from the
        reviewer's Slack answers): a component is dropped when
        its `sku_id` or its `raw_name` appears in that list. Skipping has to
        genuinely remove the quantity, otherwise "skip" is a silent approve of a
        number nobody reviewed.
      * A component whose own `confidence` is below `CONF_REVIEW_FLOOR`. The
        decomposer flagged that quantity as a guess it does not stand behind; the
        confidence band in `src/state.py` says a guess that weak is a gap, not a
        question, and a gap must not turn into a line on an invoice. This is the
        $388 wheel of Parmigiano Reggiano the pipeline bought for a Vietnamese
        restaurant off a 0.40-confidence hallucination.

    A component with **no** stated confidence is kept. Absent evidence is not
    evidence of a bad guess, and dropping it would be the same silent-loss bug in
    the other direction.
    """
    forecast = state.get("demand_forecast") or {}
    item_mix = forecast.get("item_mix") or {}
    covers_per_week = float(forecast.get("covers_per_week") or 0.0)

    skipped = {str(ref) for ref in (state.get("skipped_refs") or [])}

    matches: dict[str, dict] = {}
    for m in state.get("sku_matches") or []:
        matches.setdefault(m.get("raw_name", ""), m)

    recipes: dict[str, dict] = {}
    for r in state.get("recipes") or []:
        recipes.setdefault(r.get("item_name", ""), r)

    # Only used to word the gap below. A name that is not on the menu at all did
    # not lose its recipe — it never named a dish, and blaming the decomposer for
    # it sends the next reader to the wrong node.
    menu_names = {i.get("name", "") for i in state.get("menu_items") or []}

    consumption: dict[str, float] = {}
    gaps: list[str] = []
    withheld: dict[str, dict] = {}

    for item_name, share in item_mix.items():
        servings = float(share or 0.0) * covers_per_week
        if servings <= 0.0:
            continue

        recipe = recipes.get(item_name)
        if recipe is None:
            why = (
                "no decomposed recipe"
                if item_name in menu_names
                else "not a menu item name; matched no recipe"
            )
            gaps.append(f"{item_name} ({why}; not ordered for)")
            continue

        # `servings` above counts ORDERS of this menu item, not diners:
        # `src/nodes/forecast.py` defines share as "the fraction of covers that
        # order that item". And a recipe describes one SOLD UNIT — the thing an
        # order buys — because that is what the decomposer was asked for ("still
        # describe one sold unit and say so via yield_qty / yield_uom").
        #
        # So one order consumes the recipe's components as written, undivided.
        # This used to divide them by yield_qty, which under-ordered every
        # multi-serving line by exactly its yield: an order of Joe's "Classic
        # Cheese Pie 8 slices" would buy 1.75 oz of flour for a pie that takes
        # 14. The same off-by-the-yield bug lived in cost_plates.py, and the two
        # are fixed together so plate cost and order quantity still agree.
        for component in recipe.get("components") or []:
            raw = component.get("raw_name", "")
            match = matches.get(raw)
            if match is None:
                gaps.append(f"{raw} in {item_name} (no SKU match record)")
                continue
            sku_id = match.get("sku_id")
            if not sku_id:
                gaps.append(f"{raw} in {item_name} (no catalog SKU matched)")
                continue
            sku = skus.get(sku_id)
            if sku is None:
                gaps.append(f"{raw} in {item_name} (sku_id '{sku_id}' is not in the catalog)")
                continue

            per_order = float(component.get("qty") or 0.0)
            converted = _convert(
                per_order, component.get("uom", ""), sku.get("uom", ""), sku
            )
            if converted is None:
                gaps.append(
                    f"{raw} in {item_name} (no conversion from "
                    f"{component.get('uom', '')} to {sku.get('uom', '')})"
                )
                continue

            weekly = converted * servings

            # The confidence the decomposer put on this very component, not the
            # confidence of the SKU match. The canonicalizer matching
            # "Parmigiano Reggiano" to the Parmigiano SKU at 1.0 is correct and
            # says nothing about whether the ingredient belongs in the dish.
            try:
                conf = component.get("confidence")
                conf = None if conf is None else float(conf)
            except (TypeError, ValueError):
                conf = None

            # A human's explicit skip is named first: they made that call, and
            # attributing it to the model's uncertainty would misreport why.
            reason = ""
            if sku_id in skipped or raw in skipped:
                reason = "skipped_in_review"
            elif conf is not None and conf < CONF_REVIEW_FLOOR:
                reason = "confidence_below_floor"

            if reason:
                record = withheld.setdefault(sku_id, {
                    "sku_id": sku_id,
                    "display_name": sku.get("display_name", sku_id),
                    "uom": str(sku.get("uom") or ""),
                    "reasons": [],
                    "dishes": [],
                    "confidence": None,
                    "qty": 0.0,
                })
                if reason not in record["reasons"]:
                    record["reasons"].append(reason)
                if item_name not in record["dishes"]:
                    record["dishes"].append(item_name)
                # Lowest confidence seen wins: it is the one that has to be
                # defended when somebody asks why this was not bought.
                if conf is not None and (
                    record["confidence"] is None or conf < record["confidence"]
                ):
                    record["confidence"] = conf
                record["qty"] += weekly
                continue

            consumption[sku_id] = consumption.get(sku_id, 0.0) + weekly

    return consumption, gaps, withheld


def _par_levels(consumption: dict[str, float], skus: dict[str, dict]) -> list[dict]:
    """Pass 2 — par level per SKU, each carrying the sentence that explains it.

    `par_qty = daily_consumption x days_cover x (1 + SAFETY_FACTOR)`. The
    rationale names all three inputs because every number in this system should
    be able to explain itself without someone re-deriving it from source.
    """
    levels: list[dict] = []

    for sku_id, weekly in consumption.items():
        sku = skus.get(sku_id) or {}
        category = str(sku.get("category") or "")
        uom = str(sku.get("uom") or "")

        days_cover = DAYS_COVER.get(category, DEFAULT_DAYS_COVER)
        daily = weekly / 7.0
        par_qty = daily * days_cover * (1.0 + SAFETY_FACTOR)

        levels.append({
            "sku_id": sku_id,
            "par_qty": round(par_qty, 4),
            "uom": uom,
            "days_cover": days_cover,
            "daily_consumption": round(daily, 4),
            "rationale": (
                f"{days_cover} days cover on {category or 'uncategorised'}, "
                f"{daily:.1f} {uom}/day projected, +{SAFETY_FACTOR:.0%} safety"
            ),
        })

    return levels


def _to_packs(par_levels: list[dict], skus: dict[str, dict]) -> list[dict]:
    """Pass 3 — round each par level up to whole purchasable packs.

    A restaurant cannot order 4.2 lb of a 25 lb case, so `packs = ceil(par_qty /
    pack_qty)`. Both figures survive into the line: the gap between the par level
    and what actually arrives is real week-one overstock, and it is worth being
    able to talk about rather than hiding.

    A SKU whose packs round to 0 is omitted entirely rather than ordered at zero —
    a zero-quantity line on a purchase order is noise a chef has to read past.
    """
    lines: list[dict] = []

    for level in par_levels:
        sku = skus.get(level["sku_id"]) or {}
        pack_qty = float(sku.get("pack_qty") or 0.0)
        if pack_qty <= 0.0:
            continue

        packs = math.ceil(level["par_qty"] / pack_qty)
        if packs <= 0:
            continue

        lines.append({
            "sku_id": level["sku_id"],
            "display_name": sku.get("display_name", level["sku_id"]),
            "packs": packs,
            "pack_unit": sku.get("pack_unit", ""),
            "pack_qty": pack_qty,
            "pack_uom": sku.get("pack_uom", ""),
            "line_cost": round(packs * float(sku.get("price_per_pack") or 0.0), 2),
            "par_qty": level["par_qty"],
        })

    return lines


def _projected_weekly_revenue(state: AgentState) -> float:
    """Covers x average item price, weighted by the same item mix the order used.

    The shares in `item_mix` sum to roughly items-per-cover rather than to 1.0, so
    summing `share x covers x menu_price` across the menu is exactly "covers times
    average item price" with the average taken over what people actually order.
    Items with no printed price contribute nothing, which understates revenue
    slightly and therefore overstates the ratio — the safe direction for a sanity
    check whose whole job is to notice a total that is too large.
    """
    forecast = state.get("demand_forecast") or {}
    item_mix = forecast.get("item_mix") or {}
    covers_per_week = float(forecast.get("covers_per_week") or 0.0)

    prices: dict[str, float] = {}
    for item in state.get("menu_items") or []:
        price = item.get("price")
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            continue
        prices.setdefault(item.get("name", ""), float(price))

    return sum(
        float(share or 0.0) * covers_per_week * prices.get(name, 0.0)
        for name, share in item_mix.items()
    )


def draft_po_node(state: AgentState) -> dict:
    """Forecast + recipes + catalog -> par levels and a draft opening order."""
    try:
        skus = _load_catalog()

        consumption, gaps, withheld = _consumption(state, skus)
        par_levels = _par_levels(consumption, skus)
        lines = _to_packs(par_levels, skus)

        # What the order would have cost had nothing been withheld. The same two
        # pure passes are run a second time over consumption + withheld, so the
        # money not spent is a real difference between two orders rather than an
        # estimate: pack rounding is included on both sides, and a SKU still
        # bought for a confident dish correctly reports only the delta.
        if withheld:
            unfiltered = dict(consumption)
            for sku_id, record in withheld.items():
                unfiltered[sku_id] = unfiltered.get(sku_id, 0.0) + record["qty"]
            would_have_cost = {
                line["sku_id"]: line["line_cost"]
                for line in _to_packs(_par_levels(unfiltered, skus), skus)
            }
        else:
            would_have_cost = {}
        actual_cost = {line["sku_id"]: line["line_cost"] for line in lines}

        excluded_skus = []
        for sku_id, record in withheld.items():
            excluded_skus.append({
                "sku_id": sku_id,
                "display_name": record["display_name"],
                "reason": " + ".join(record["reasons"]),
                "confidence": record["confidence"],
                "dishes": record["dishes"],
                "qty_withheld": round(record["qty"], 4),
                "uom": record["uom"],
                "cost_not_spent": round(
                    would_have_cost.get(sku_id, 0.0) - actual_cost.get(sku_id, 0.0), 2
                ),
            })
        excluded_skus.sort(key=lambda e: e["cost_not_spent"], reverse=True)
        excluded_cost_total = round(sum(e["cost_not_spent"] for e in excluded_skus), 2)

        # Pass 4 — group by vendor, money at the top of each group.
        vendor_lines: dict[str, list[dict]] = {}
        for line in lines:
            vendor = str((skus.get(line["sku_id"]) or {}).get("vendor") or "Unassigned vendor")
            vendor_lines.setdefault(vendor, []).append(line)
        for group in vendor_lines.values():
            group.sort(key=lambda line: line["line_cost"], reverse=True)

        total_cost = round(sum(line["line_cost"] for line in lines), 2)

        # The cover policy is per category, so there is no single number to
        # report. State the range actually used by the lines on THIS order: a
        # card claiming "7 days cover" over an order that mixes 2-day produce
        # with 30-day spices states something the arithmetic never did.
        cover_by_sku = {level["sku_id"]: level["days_cover"] for level in par_levels}
        covers_used = sorted({cover_by_sku[line["sku_id"]] for line in lines})
        if not covers_used:
            days_cover_label = ""
        elif len(covers_used) == 1:
            days_cover_label = f"{covers_used[0]} days cover"
        else:
            days_cover_label = (
                f"{covers_used[0]}-{covers_used[-1]} days cover by category"
            )

        forecast = state.get("demand_forecast") or {}
        covers_per_week = float(forecast.get("covers_per_week") or 0.0)

        assumptions = list(forecast.get("assumptions") or [])
        assumptions.append(
            "Days of cover by category: "
            + ", ".join(f"{c} {d}d" for c, d in DAYS_COVER.items())
            + f" (anything else {DEFAULT_DAYS_COVER}d)."
        )
        assumptions.append(
            f"A flat {SAFETY_FACTOR:.0%} safety buffer is added on top of projected "
            f"consumption, because day one has a prior rather than a forecast."
        )
        assumptions.append(
            "Quantities are rounded UP to whole purchasable packs, so week one carries "
            "deliberate overstock wherever a par level falls below one case."
        )
        assumptions.append(
            "Prices are pack prices from the hand-curated SKU catalog, not live vendor "
            "quotes; the real invoice will differ."
        )
        if excluded_skus:
            assumptions.append(
                f"{len(excluded_skus)} item(s) worth ${excluded_cost_total:,.2f} are "
                f"deliberately NOT on this order — a component confidence below "
                f"{CONF_REVIEW_FLOOR:.0%} or an explicit skip in review: "
                + ", ".join(
                    f"{e['display_name']} (${e['cost_not_spent']:,.2f}, {e['reason']})"
                    for e in excluded_skus[:10]
                )
            )
        if gaps:
            assumptions.append(
                f"{len(gaps)} ingredient(s) could not be joined to a catalog SKU and are "
                f"NOT on this order: " + "; ".join(gaps[:10])
            )

        purchase_order = {
            "vendor_lines": vendor_lines,
            "total_cost": total_cost,
            "covers_per_week": covers_per_week,
            "days_cover_label": days_cover_label,
            "assumptions": assumptions,
            "excluded_skus": excluded_skus,
            "excluded_cost_total": excluded_cost_total,
            "generated_at_stage": "po_drafted",
        }

        # Louder than the total, on purpose. "I did not buy these because I was
        # not sure" is the product; a withheld line nobody is told about is the
        # same silent-loss bug as an unreviewed line nobody was asked about.
        if excluded_skus:
            below_floor = sum(
                1 for e in excluded_skus if "confidence_below_floor" in e["reason"]
            )
            print(
                f"draft_po: WITHHELD {len(excluded_skus)} SKU(s) worth "
                f"${excluded_cost_total:,.2f} - {below_floor} below the "
                f"{CONF_REVIEW_FLOOR:.0%} confidence floor, "
                f"{len(excluded_skus) - below_floor} skipped in review"
            )

        # The cheapest upstream sanity check in the pipeline. Printed, never
        # enforced: a wrong ratio is evidence about the forecast or the unit
        # conversions, not a reason to refuse to produce the order.
        revenue = _projected_weekly_revenue(state)
        ratio = total_cost / revenue if revenue > 0 else None
        if ratio is None:
            print(
                f"draft_po: {len(lines)} lines across {len(vendor_lines)} vendors, "
                f"total ${total_cost:,.2f} - no projected revenue, food-cost ratio n/a"
            )
        else:
            verdict = (
                "in the 28-33% band"
                if SANE_RATIO_LOW <= ratio <= SANE_RATIO_HIGH
                # ASCII only: this string is printed, and a non-ASCII dash raises
                # UnicodeEncodeError on the cp1252 Windows console from inside
                # this node's own try/except, which trips the circuit breaker.
                else "OUTSIDE the 28-33% band - check item mix, unit conversions, SKU matches"
            )
            print(
                f"draft_po: {len(lines)} lines across {len(vendor_lines)} vendors, "
                f"total ${total_cost:,.2f} vs projected weekly revenue ${revenue:,.2f} "
                f"- food-cost ratio {ratio:.1%} ({verdict})"
            )

        return {
            "par_levels": par_levels,
            "purchase_order": purchase_order,
            "stage": "po_drafted",
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "purchase_order": {},
            "messages": [HumanMessage(f"Purchase order generation failed: {e}")],
        }
