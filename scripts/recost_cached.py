"""Recompute `plate_costs` on every cached sweep record. No LLM, no network.

`cost_plates_node` is pure arithmetic over the catalog, the units table and the
state's own `recipes` / `sku_matches`, all of which are already on disk. So a fix
to the costing arithmetic does NOT require re-paying for extraction: this runs the
cheap half of the pipeline again, in seconds, against `data/output/*-costed.json`.

Two things this is careful about, both of which would silently corrupt the cache:

  1. **`review_queue` is filtered, not wiped.** `cost_plates_node` APPENDS its
     flags to whatever queue it is handed, so recomputing naively would double
     the queue on every run. But the queue also holds entries this node did not
     produce — `sku_match` from the canonicalizer, `recipe_uom` from the
     decomposer — and dropping those would lose review items that cost real
     tokens to generate. So only the kinds this node emits are stripped, and it
     re-adds them. (`scripts/validate_foodcost.py --recost` clears the whole
     queue, which is why this exists separately.)
  2. **A failed run is skipped, not costed.** Six sweep records carry a non-empty
     `error`. Five never got past ingestion and have no recipes at all. The sixth,
     `bubbys-tribeca`, is the one that matters: it stopped at `menu_extracted`
     with 125 recipes but ZERO `sku_matches`, so costing it would write 125
     uncostable plates into a record that has none — inventing a costing run that
     never happened and inflating "plates costed" in `scripts/sweep_report.py`
     from 0 to 125. A recost must reproduce the arithmetic, never extend the
     corpus, so both conditions are checked.

`--dry-run` writes nothing and prints the before/after median food cost per
restaurant, which is the number any change to the costing arithmetic has to be
judged on.
"""
import argparse
import json
from pathlib import Path
from statistics import median

from src.nodes.cost_plates import cost_plates_node

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output"

# The review-queue kinds `cost_plates_node` produces, and therefore the only ones
# safe to strip before letting it produce them again. Anything else in the queue
# came from an upstream node and must survive untouched.
COST_PLATE_KINDS = {"plate_cost", "plate_quantity"}


def _pcts(plate_costs: list[dict]) -> list[float]:
    """Every `food_cost_pct` that exists, as floats.

    An uncostable plate has `food_cost_pct is None` and is excluded rather than
    counted as 0% — the same rule the rest of the pipeline applies, and the whole
    reason `costable` exists.
    """
    return [
        float(p["food_cost_pct"])
        for p in plate_costs or []
        if p.get("food_cost_pct") is not None
    ]


def _fmt(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct:.1%}"


def recost_file(
    path: Path, dry_run: bool
) -> tuple[str, list[float], list[float]] | None:
    """Recompute one cached record.

    Returns `(slug, pcts_before, pcts_after)`, or None when the record is a
    failed run and must be left exactly as it is. The per-plate lists come back
    rather than their medians so the caller can take a corpus-wide median over
    PLATES — a 12-plate menu and a 90-plate menu should not each count as one vote.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    state = dict(record.get("state") or {})
    if record.get("error") or not state.get("recipes"):
        return None

    before = _pcts(state.get("plate_costs") or [])

    # Strip only what this node will re-add; keep sku_match / recipe_uom / anything else.
    state["review_queue"] = [
        entry
        for entry in (state.get("review_queue") or [])
        if entry.get("kind") not in COST_PLATE_KINDS
    ]
    state.update(cost_plates_node(state))

    after = _pcts(state.get("plate_costs") or [])

    if not dry_run:
        record["state"] = state
        # Byte-for-byte the same formatting the sweep writes with
        # (`scripts/validate_foodcost.py::_write_cache`), so recosting shows up
        # in git as a data diff and not as a whole-file reformat.
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    return record.get("slug", path.stem), before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the before/after medians without writing anything",
    )
    args = parser.parse_args()

    paths = sorted(OUTPUT_DIR.glob("*-costed.json"))
    if not paths:
        print(f"no cached records in {OUTPUT_DIR}")
        return

    print(f"{'restaurant':<30} {'before':>8} {'after':>8}   plates")
    print("-" * 62)

    n_recosted = 0
    skipped = 0
    all_before: list[float] = []
    all_after: list[float] = []

    for path in paths:
        outcome = recost_file(path, args.dry_run)
        if outcome is None:
            skipped += 1
            print(f"{path.stem.replace('-costed', ''):<30} {'-':>8} {'-':>8}   failed run, skipped")
            continue
        slug, before, after = outcome
        n_recosted += 1
        all_before.extend(before)
        all_after.extend(after)

        med_before = median(before) if before else None
        med_after = median(after) if after else None
        moved = "" if med_before == med_after else "  <-- moved"
        print(
            f"{slug:<30} {_fmt(med_before):>8} {_fmt(med_after):>8}   {len(after)}{moved}"
        )

    print("-" * 62)
    print(
        f"corpus median over {len(all_after)} plates: "
        f"{_fmt(median(all_before) if all_before else None)} -> "
        f"{_fmt(median(all_after) if all_after else None)}"
    )
    print(
        f"{n_recosted} recosted, {skipped} skipped"
        + ("  (dry run, nothing written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
