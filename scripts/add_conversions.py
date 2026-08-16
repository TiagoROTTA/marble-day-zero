"""One-shot: add dimension-crossing `conversions` to the SKUs that need them.

Run once, committed for provenance, not part of the pipeline.

`data/catalog/units.json` answers pure unit algebra — `oz:lb` is 0.0625 for
every substance there is. It cannot answer `fl_oz -> lb` (needs a density) or
`oz -> each` (needs a unit weight), because those are properties of the product
rather than of the units. Before this, `cost_plates` refused those components
and coverage fell: 15 plates lost their fry-oil line, 11 lost romaine.

Every factor below is an **estimate** from standard foodservice pack sizes and
published densities, and is marked as such in each SKU's `conversions_source`.
That matters: the whole point of the food-cost chart is that the numbers under
it are honest, so a fabricated-but-unlabelled density would be worse than the
gap it closes. These are good to roughly +/-10%, which moves a plate cost by
less than the rounding on the menu price it is divided by.

Derivations, so a reader can check rather than trust:

  Oils, fl_oz -> lb   : cooking oil ~0.92 g/mL; 1 fl_oz = 29.5735 mL = 27.2 g;
                        1 lb = 453.592 g  ->  0.060 lb per fl_oz
  Mayo,  oz -> gal    : ~0.94 g/mL; 1 gal = 3785.4 mL = 3558 g = 125.5 oz
                        ->  0.00797 gal per oz
  #10 can, oz -> each : crushed tomato 102 oz, paste 111 oz, chickpeas 108 oz
  Coconut milk        : 13.5 fl_oz can  ->  0.0741 each per fl_oz
  Produce, oz -> each : trimmed head/fruit weights - romaine 20.8 oz,
                        iceberg 22.4 oz, cauliflower 32 oz, pineapple 48 oz
  Scallion            : 3 oz per bunch  ->  0.3333 bunch per oz

Keys are the SOURCE unit; the target is always that SKU's own `uom`.
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG = Path("data/catalog/skus.json")
SOURCE_NOTE = "estimated 2026-08 from standard foodservice pack sizes and densities"

CONVERSIONS: dict[str, dict[str, float]] = {
    # fl_oz -> lb (density)
    "COND-OIL-FRYER": {"fl_oz": 0.060},
    "COND-OIL-COCONUT": {"fl_oz": 0.060},
    # oz -> gal (density, inverted)
    "COND-MAYO-HEAVY": {"oz": 0.00797},
    # oz -> each (#10 cans)
    "DRY-TOMATO-CRUSHEDCAN": {"oz": 0.00980},
    "DRY-TOMATO-PASTECAN": {"oz": 0.00901},
    "DRY-CHICKPEA-CANNED": {"oz": 0.00926},
    # fl_oz -> each (13.5 fl_oz can)
    "DRY-COCONUTMILK-CAN": {"fl_oz": 0.0741},
    # oz -> each (trimmed produce weights)
    "PROD-LETTUCE-ROMAINE": {"oz": 0.0481},
    "PROD-LETTUCE-ICEBERG": {"oz": 0.0446},
    "PROD-CAULIFLOWER-HEAD": {"oz": 0.03125},
    "PROD-PINEAPPLE-GOLDEN": {"oz": 0.0208},
    # oz -> bunch
    "PROD-ONION-SCALLION": {"oz": 0.3333},
}


def main() -> int:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_id = {s["sku_id"]: s for s in catalog}

    missing = [k for k in CONVERSIONS if k not in by_id]
    if missing:
        print(f"sku_id(s) not in catalog, aborting: {', '.join(missing)}")
        return 1

    for sku_id, conv in CONVERSIONS.items():
        sku = by_id[sku_id]
        # A conversion whose target is not this SKU's own uom can never fire
        # (`_convert` checks `to_uom == sku["uom"]`), so a typo here would be a
        # silent no-op rather than a wrong number. Catch it now instead.
        if sku["uom"] in conv:
            print(f"{sku_id}: source unit equals target uom {sku['uom']!r}, aborting")
            return 1
        sku["conversions"] = conv
        sku["conversions_source"] = SOURCE_NOTE
        print(f"  {sku_id:<26} {conv} -> {sku['uom']}")

    CATALOG.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{len(CONVERSIONS)} SKUs updated in {CATALOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
