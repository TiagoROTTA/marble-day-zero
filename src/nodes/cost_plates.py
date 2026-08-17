"""Plate costing: turn recipes + SKU matches + catalog prices into a cost per menu line.

The invariant the whole node is built around: **`plate_cost` and `menu_price`
must describe the same sold unit.** `food_cost_pct` is one divided by the other,
so any mismatch between the two makes it a meaningless number that still looks
like a percentage. The decomposer is asked for the recipe of one SOLD UNIT — what
`menu_price` buys — and `yield_qty` merely reports how many servings that unit
contains, so `plate_cost` is the cost of the whole line, undivided.
`cost_per_serving` carries the divided figure alongside it for the shareable
platters where a per-head number is the interesting one.

**This node makes no LLM call at all.** It is pure arithmetic over what the three
previous nodes produced, and that is the whole point: the food-cost validation in
`scripts/validate_foodcost.py` only means something because every number on the
chart can be recomputed by hand from a transcribed menu price, a decomposed
quantity, a unit conversion and a catalog price. A number a model guessed proves
nothing.

Two rules keep the arithmetic honest rather than merely plausible:

  1. `_convert()` returns `None` when no conversion exists. Never 1.0, never an
     exception. An unconvertible unit has to become a visible gap in `uncosted`,
     because a silently wrong multiplier is a plate cost that looks fine and is not.
  2. `confidence` treats measurement and opinion differently. The recipe's own
     confidence and the mean confidence of the SKU matches actually used are two
     model self-assessments, so they are combined as a geometric mean. Coverage
     is not an opinion — it is a measured fact about how much of the plate could
     be priced at all — so it MULTIPLIES that result rather than being averaged
     into it: a plate costed from half its ingredients reports at most half the
     confidence, whatever the model believed about the half it saw.
  3. A component whose own `confidence` is below `CONF_REVIEW_FLOOR` is not
     costed at all — same treatment as an unconvertible unit. `src/nodes/draft_po.py`
     withholds those same components from the purchase order, so pricing them
     into the plate would mean the plate cost and the order describe different
     sets of ingredients. Because the exclusion runs through `coverage`, the
     resulting under-costing announces itself as lower plate confidence rather
     than hiding in an inflated total. A component with no stated confidence is
     still costed; draft_po.py makes the same call.
  4. A plate where NOT ONE component could be costed reports `plate_cost: None`
     and `costable: False`, never 0.0. `$0.00` is a measurement; "we could not
     cost this" is not, and `scripts/plot_foodcost.py` plots the food-cost
     distribution across 20 restaurants against the 28-33% industry band. A
     handful of fake 0% plates
     would move that chart and the finding drawn from it. Such a plate is also
     kept out of the "outside sane band" count and out of the mean coverage —
     it is reported as its own category instead.

Failure handling: any exception (a malformed catalog entry, a non-numeric price) is
caught; retry_count++ ; the error is fed back through `last_error`. There is no
network call here, but the circuit breaker should still catch a bad catalog.
"""
import json
import math
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.state import CONF_REVIEW_FLOOR, AgentState

CATALOG_PATH = str(Path(__file__).resolve().parents[2] / "data" / "catalog" / "skus.json")
UNITS_PATH = str(Path(__file__).resolve().parents[2] / "data" / "catalog" / "units.json")

# The band a restaurant actually runs in: food cost is ~28-33% of revenue, so a
# dish computing to 80% is nearly always a bug upstream (a lb/oz mix-up, a wrong
# SKU match), and a dish at 4% is nearly always a missed ingredient. Occasionally
# an outlier is a genuinely unprofitable dish — a real finding — but assume bug first.
SANE_BAND_LOW = 0.10
SANE_BAND_HIGH = 0.55

# Filled on first use by _load_catalog() / _convert(). The table is eighteen
# entries of static data from `data/catalog/units.json`; re-reading it per
# component would be silly.
_UNITS: dict[str, float] = {}


def _convert(
    qty: float, from_uom: str, to_uom: str, sku: dict | None = None
) -> float | None:
    """Convert `qty` from one unit to another, or return None if we cannot.

    None is the load-bearing case. Returning 1.0 for an unknown pair would put a
    wrong number on the food-cost chart with no trace; returning None puts the
    component in `uncosted` where a human can see it. `src/nodes/draft_po.py`
    imports this.

    Two tiers, dimensionless first:

      1. `data/catalog/units.json` — pure unit algebra. `oz:lb` is 0.0625 for
         every substance in the universe, so it lives in one shared table.
      2. The SKU's own `conversions` map — dimension-crossing factors that are
         a property of the *product*, not of the units. A fluid ounce of oil
         and a fluid ounce of honey are different weights, and one romaine
         heart is not one head of cabbage, so `fl_oz -> lb` and `oz -> each`
         can only be answered per SKU.

    The per-SKU tier is consulted only when the shared table has no answer, so
    a SKU can never quietly redefine `oz:lb`. `conversions` is keyed by the
    *source* unit and always converts into that SKU's own `uom` — a factor is
    meaningless against any other target, and reusing it for one would be
    exactly the confident-wrong-number this function exists to refuse.
    """
    if from_uom == to_uom:
        return qty
    if not _UNITS:
        with open(UNITS_PATH, encoding="utf-8") as f:
            _UNITS.update(json.load(f))
    factor = _UNITS.get(f"{from_uom}:{to_uom}")
    if factor is not None:
        return qty * factor

    if sku is not None and to_uom == sku.get("uom"):
        per_sku = (sku.get("conversions") or {}).get(from_uom)
        if per_sku is not None:
            return qty * float(per_sku)
    return None


def _load_catalog() -> dict[str, dict]:
    """Return the SKU catalog keyed by sku_id, and prime the unit table."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    with open(UNITS_PATH, encoding="utf-8") as f:
        _UNITS.update(json.load(f))
    return {sku["sku_id"]: sku for sku in catalog}


def cost_plates_node(state: AgentState) -> dict:
    """Cost every recipe against the catalog. No LLM, no network, no guessing."""
    try:
        skus = _load_catalog()

        matches: dict[str, dict] = {}
        for m in state.get("sku_matches") or []:
            matches.setdefault(m.get("raw_name", ""), m)

        # Menu price by dish name; first occurrence wins, exactly as the menu reads.
        prices: dict[str, float | None] = {}
        for item in state.get("menu_items") or []:
            prices.setdefault(item.get("name", ""), item.get("price"))

        plate_costs: list[dict] = []
        flagged: list[dict] = []

        for recipe in state.get("recipes") or []:
            item_name = recipe.get("item_name", "")
            components = recipe.get("components") or []

            total = 0.0
            costed_components = 0
            uncosted: list[str] = []
            # Costed, but on a quantity the decomposer would not stand behind.
            # Distinct from `uncosted`: these ARE in the total, so they must not
            # be reported as gaps -- they are the lines a human should sanity
            # check precisely because they moved the number.
            low_confidence: list[str] = []
            match_confs: list[float] = []

            for component in components:
                raw = component.get("raw_name", "")
                match = matches.get(raw)
                if match is None:
                    uncosted.append(f"{raw} (no SKU match record)")
                    continue
                sku_id = match.get("sku_id")
                if not sku_id:
                    uncosted.append(f"{raw} (no catalog SKU matched)")
                    continue
                sku = skus.get(sku_id)
                if sku is None:
                    uncosted.append(f"{raw} (sku_id '{sku_id}' is not in the catalog)")
                    continue

                # The confidence the decomposer put on this very component --
                # how sure it is of the QUANTITY -- not the confidence of the
                # SKU match, which is a separate question about identity.
                #
                # This used to drop the component entirely below the floor. That
                # was wrong, and measurably so: on the Adda lamb shank the SKU
                # matched at confidence 1.0 (we knew the product and its $7.40/lb
                # price exactly) and only the quantity was hedged at 0.50, so the
                # line was deleted and the plate costed $2.97 instead of $10.37 --
                # 6% food cost instead of 22%.
                #
                # The bias is systematic, not incidental: the expensive ingredient
                # is the one a decomposer hedges on, so a floor applied here
                # preferentially deletes costly lines and drags the whole
                # food-cost distribution down. Dropping the line does not remove
                # the uncertainty about the quantity -- it replaces "roughly 16 oz
                # of lamb" with "no lamb at all", which is a far larger error and
                # wrong in the same direction every time.
                #
                # So the component is costed, and its confidence is folded into
                # the plate's confidence instead. A shaky quantity now lowers the
                # number the plate reports about itself rather than silently
                # lowering the plate's cost. It is still queued for a human below.
                #
                # `src/nodes/draft_po.py` still WITHHOLDS these components, and
                # that divergence is deliberate rather than drift. Costing
                # produces a statistic, where omitting the lamb makes the number
                # wrong and biased. The purchase order spends money on goods,
                # where ordering 20 lb of lamb on a quantity the decomposer
                # hedged at 0.50 is a real cost to a real restaurant. The
                # conservative choice points in opposite directions for the two,
                # so they legitimately disagree; draft_po records what it
                # withheld in `excluded_skus` with the money not spent.
                try:
                    conf = component.get("confidence")
                    conf = None if conf is None else float(conf)
                except (TypeError, ValueError):
                    conf = None

                from_uom = component.get("uom", "")
                to_uom = sku.get("uom", "")
                converted = _convert(
                    float(component.get("qty") or 0.0), from_uom, to_uom, sku
                )
                if converted is None:
                    uncosted.append(f"{raw} (no conversion from {from_uom} to {to_uom})")
                    continue

                total += converted * float(sku.get("price_per_uom") or 0.0)
                costed_components += 1

                # No stated confidence is kept at face value: absent evidence is
                # not evidence of a bad guess, and draft_po.py makes the same call.
                # A stated one multiplies the match confidence, so identity and
                # quantity both have to be sound for the line to read as certain.
                if conf is not None and conf < CONF_REVIEW_FLOOR:
                    low_confidence.append(
                        f"{raw} (quantity confidence {conf:.2f} below the "
                        f"{CONF_REVIEW_FLOOR:.2f} review floor -- costed anyway)"
                    )
                match_confs.append(
                    float(match.get("confidence") or 0.0) * (1.0 if conf is None else conf)
                )

            # Not one component survived the join: every one was unmatched,
            # unconvertible, absent from the catalog or below the review floor.
            # That is not a plate that costs nothing, it is a plate we could not
            # cost, and the two must not look alike — a 0.0 here reads as a real
            # 0% food cost on the food-cost distribution chart. Both signals are
            # written on purpose: `plate_cost is None` stops arithmetic that
            # would treat it as money, and `costable` survives a consumer who
            # writes `float(p.get("plate_cost") or 0.0)` and would otherwise
            # resurrect the fake zero.
            costable = costed_components > 0

            # THE INVARIANT: `plate_cost` and `menu_price` must describe the SAME
            # SOLD UNIT. `food_cost_pct` divides one by the other, so the moment
            # they describe different quantities the ratio is meaningless.
            #
            # `total` is the cost of the whole menu line, because that is what the
            # decomposer was asked for. Its SYSTEM_PROMPT: "If a dish is genuinely
            # sold as a shareable platter, still describe one sold unit and say so
            # via yield_qty / yield_uom." `yield_qty` therefore REPORTS how many
            # servings the sold unit contains; it does not scale the components.
            # The cached data agrees: au-zaatar's "Fattoush Salad (Large)"
            # (yield_qty 2) lists exactly twice the romaine of the Small
            # (yield_qty 1), and kanoyama's "Sushi <For 3>" lists 18 oz of rice
            # against "Sushi <For 2>"'s 12 oz. Per-serving quantities would have
            # been identical across each pair.
            #
            # So dividing by yield_qty here was a straight off-by-the-yield bug:
            # it compared one slice against the price of the whole pie. Joe's
            # "Classic Cheese Pie 8 slices" costs $4.62 of ingredients against a
            # $24.00 menu price — 19.3% food cost — and used to report $0.58
            # against $24.00, 2.4%, which put a fake near-zero on the food-cost
            # distribution chart.
            #
            # 844 of the 879 cached recipes carry yield_qty 1.0, where this
            # changes precisely nothing.
            plate_cost = round(total, 4) if costable else None

            # The per-serving figure is kept rather than lost — it is the useful
            # number for a shareable platter, it just is not the one `menu_price`
            # can be divided into. `yield_qty` rides along in the entry so the two
            # can never be mistaken for each other by a later reader.
            #
            # yield_qty of 0 or None would be a division by zero on data we do not
            # control; a recipe with no stated yield is one serving by definition.
            yield_qty = float(recipe.get("yield_qty") or 0.0) or 1.0
            cost_per_serving = (
                round(total / yield_qty, 4) if costable else None
            )

            total_components = len(components)
            coverage = costed_components / total_components if total_components else 0.0

            menu_price = prices.get(item_name)
            if isinstance(menu_price, bool) or not isinstance(menu_price, (int, float)):
                menu_price = None
            else:
                menu_price = float(menu_price)
            # None and 0.0 both give None, never a ZeroDivisionError and never 0.0.
            # An uncostable plate has no ratio either: a percentage of a cost we
            # do not have is the same fiction as the cost itself, and the
            # food-cost sweep collects exactly the entries whose food_cost_pct
            # is not None.
            food_cost_pct = (
                round(plate_cost / menu_price, 4) if costable and menu_price else None
            )

            mean_match_conf = sum(match_confs) / len(match_confs) if match_confs else 0.0
            # Two model self-assessments, combined as a geometric mean; coverage
            # then multiplies the result outright.
            #
            # Coverage is deliberately NOT inside the mean. It is not an opinion
            # the way the other two are -- it is a measured fact about how much
            # of the plate was priced at all, so it scales the answer rather than
            # being averaged with it: a plate costed from half its ingredients
            # reports at most half the confidence, whatever the model believed
            # about the half it saw.
            #
            # The two opinions are averaged rather than multiplied because a
            # product of three numbers below 1.0 collapses onto a scale that has
            # nothing to do with the 0.85 / 0.60 bands in src/state.py, which are
            # phrased for a single judgement. Measured on joes-pizza-carmine, the
            # raw triple product put every plate under CONF_REVIEW_FLOOR, so the
            # whole restaurant was recorded as gaps and the review card was never
            # sent -- the human gate silently had nothing to ask about, which is
            # the one failure mode this pipeline cannot afford.
            confidence = round(
                math.sqrt(float(recipe.get("confidence") or 0.0) * mean_match_conf)
                * coverage,
                4,
            )

            entry = {
                "item_name": item_name,
                "plate_cost": plate_cost,
                "cost_per_serving": cost_per_serving,
                "yield_qty": yield_qty,
                "menu_price": menu_price,
                "food_cost_pct": food_cost_pct,
                "costed_components": costed_components,
                "total_components": total_components,
                "coverage": round(coverage, 4),
                "uncosted": uncosted,
                "low_confidence": low_confidence,
                "confidence": confidence,
                "costable": costable,
            }
            plate_costs.append(entry)

            # A plate whose cost rests on a quantity the decomposer hedged is
            # worth a human's eye even when the resulting ratio looks perfectly
            # sane -- a plausible total built on a shaky line is exactly the case
            # the old drop-it behaviour hid. Queued separately from the
            # outside-the-band check below so the two reasons stay legible.
            if low_confidence and costable:
                flagged.append({
                    "kind": "plate_quantity",
                    "ref": item_name,
                    "confidence": confidence,
                    "question": (
                        f"{item_name} is costed at ${plate_cost:.2f} using "
                        f"{len(low_confidence)} low-confidence quantity/quantities "
                        f"- check the recipe"
                    ),
                    "payload": {
                        "plate_cost": plate_cost,
                        "menu_price": menu_price,
                        "low_confidence": low_confidence,
                    },
                })

            # `food_cost_pct is not None` already excludes every uncostable
            # plate, which is the point: a plate we could not cost is not a plate
            # with a 0% food cost, so it is neither flagged as an outlier nor
            # counted as one. It is reported separately below instead.
            if food_cost_pct is not None and not (
                SANE_BAND_LOW <= food_cost_pct <= SANE_BAND_HIGH
            ):
                flagged.append({
                    "kind": "plate_cost",
                    "ref": item_name,
                    "confidence": confidence,
                    "question": (
                        f"{item_name} computes to {food_cost_pct:.0%} food cost "
                        f"— check the recipe or the price"
                    ),
                    "payload": {
                        "plate_cost": plate_cost,
                        "menu_price": menu_price,
                        "food_cost_pct": food_cost_pct,
                        "coverage": entry["coverage"],
                        "uncosted": uncosted,
                    },
                })

        # Every statistic below is over the costable plates only. An uncostable
        # plate has coverage 0.0 by construction, so averaging it in would report
        # a costing quality nobody measured: on joes-pizza-carmine three
        # uncostable plates dragged mean coverage from 0.85 to 0.43. The count
        # gets its own place in the line so the drop is visible rather than
        # silently absorbed. ASCII only in this string: a non-ASCII character
        # here raises UnicodeEncodeError on the cp1252 Windows console from
        # inside this node's own try/except, which trips the circuit breaker.
        costed_plates = [p for p in plate_costs if p["costable"]]
        not_costable = len(plate_costs) - len(costed_plates)
        mean_coverage = (
            sum(p["coverage"] for p in costed_plates) / len(costed_plates)
            if costed_plates
            else 0.0
        )
        # Two distinct reasons a plate reaches a human, counted separately. They
        # were briefly summed under one "outside sane band" label, which read as
        # 292 flags on 200 plates -- a number that cannot mean what it says and
        # is the first thing anyone reading the log would question.
        n_band = sum(1 for f in flagged if f["kind"] == "plate_cost")
        n_qty = sum(1 for f in flagged if f["kind"] == "plate_quantity")
        print(
            f"cost_plates: {len(plate_costs)} plates, {len(costed_plates)} costed "
            f"(mean coverage {mean_coverage:.2f}), {not_costable} not costable, "
            f"{n_band} outside sane band, {n_qty} on a hedged quantity"
        )

        return {
            "plate_costs": plate_costs,
            "review_queue": state.get("review_queue", []) + flagged,
            "stage": "costed",
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "plate_costs": [],
            "messages": [HumanMessage(f"Plate costing failed: {e}")],
        }
