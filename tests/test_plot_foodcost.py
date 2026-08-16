"""Guards on the food-cost chart's data selection.

The chart is how the pipeline's output gets judged, so what it silently drops
matters more than how it looks. Nothing here asserts on pixels: the selection
rules, the exclusion counts and the annotation string are the contract, and the
annotation is what the chart has to be read against.

Synthetic cache records are written to tmp_path — no real corpus run required.
"""
import json

import scripts.plot_foodcost as plot
from scripts.validate_foodcost import BAND_HIGH, BAND_LOW, COVERAGE_FLOOR


def _plate(pct, coverage=0.9, name="Plate"):
    return {
        "item_name": name,
        "plate_cost": 4.0,
        "menu_price": 16.0,
        "food_cost_pct": pct,
        "coverage": coverage,
        "costable": pct is not None,
    }


def _record(slug, plates, cuisine="vietnamese"):
    return {"slug": slug, "cuisine": cuisine, "error": "", "state": {"plate_costs": plates}}


def _write(tmp_path, records):
    for record in records:
        (tmp_path / f"{record['slug']}-costed.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return tmp_path


# --- the two exclusions -----------------------------------------------------


def test_low_coverage_plates_are_excluded_and_counted():
    # A plate costed from under half its ingredients is always artificially
    # cheap; including it silently would flatter the median.
    records = [
        _record("a", [
            _plate(0.30, coverage=0.9),
            _plate(0.31, coverage=COVERAGE_FLOOR),   # exactly at the floor: included
            _plate(0.05, coverage=0.2),              # under it: excluded
            _plate(0.06, coverage=0.49),
        ])
    ]

    sel = plot.select(records)

    assert sel["n_included"] == 2
    assert sel["n_excluded"] == 2
    assert sel["ratios"] == [0.30, 0.31]


def test_none_ratios_never_reach_the_chart_or_the_excluded_count():
    # cost_plates writes None, not 0.0, when nothing could be costed. Such a
    # plate never had a ratio, so it is neither plotted nor counted as an
    # exclusion — counting it would inflate the "excluded" number with plates
    # the coverage floor never judged.
    records = [_record("a", [_plate(0.30), _plate(None, coverage=0.0), _plate(None, coverage=0.9)])]

    sel = plot.select(records)

    assert sel["n_included"] == 1
    assert sel["n_excluded"] == 0
    assert sel["ratios"] == [0.30]


def test_a_missing_coverage_field_is_treated_as_zero_and_excluded():
    sel = plot.select([_record("a", [{"item_name": "X", "food_cost_pct": 0.3}])])

    assert sel["n_included"] == 0
    assert sel["n_excluded"] == 1


# --- the annotation ---------------------------------------------------------


def test_annotation_carries_the_computed_counts():
    records = [
        _record("a", [_plate(0.28), _plate(0.30), _plate(0.10, coverage=0.1)]),
        _record("b", [_plate(0.32), _plate(0.12, coverage=0.3)]),
    ]

    line = plot.annotation_text(plot.select(records))

    assert line == (
        "n = 3 plates - 2 restaurants - "
        "plates below 50% ingredient coverage excluded (n = 2)"
    )
    assert line.isascii(), "this line is printed to a Windows console too"


def test_restaurant_count_only_counts_restaurants_that_contributed_a_plate():
    # A restaurant whose every plate was excluded is not "one of the 20 the
    # chart is drawn from", and claiming it would overstate the corpus.
    records = [
        _record("a", [_plate(0.30)]),
        _record("b", [_plate(0.09, coverage=0.1)]),
        _record("c", []),
    ]

    sel = plot.select(records)

    assert sel["n_restaurants"] == 1
    assert "1 restaurants" in plot.annotation_text(sel)


# --- median and clipping ----------------------------------------------------


def test_median_is_over_included_plates_only():
    records = [_record("a", [
        _plate(0.30), _plate(0.32), _plate(0.34),
        _plate(0.01, coverage=0.1), _plate(0.02, coverage=0.1),
    ])]

    assert plot.select(records)["median"] == 0.32


def test_x_limit_clips_the_tail_and_reports_what_it_dropped():
    ratios = [0.30] * 99 + [4.0]

    hi, clipped = plot.x_limit(ratios)

    assert clipped == 1
    assert hi < 4.0
    assert hi >= BAND_HIGH, "the industry band must stay fully on screen"


def test_x_limit_never_clips_below_the_band():
    hi, clipped = plot.x_limit([0.28, 0.30, 0.31])

    assert clipped == 0
    assert hi > BAND_HIGH > BAND_LOW


# --- the cache-only contract ------------------------------------------------


def test_missing_cache_files_exit_non_zero_and_name_the_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(plot, "list_corpus", lambda tier: ["a", "b"])
    _write(tmp_path, [_record("a", [_plate(0.30)])])

    code = plot.main(["--cache-dir", str(tmp_path), "--out", str(tmp_path / "x.png")])
    err = capsys.readouterr().err

    assert code == 2
    assert "b" in err
    assert "scripts.validate_foodcost" in err
    assert not (tmp_path / "x.png").exists()


def test_all_plates_excluded_exits_non_zero_without_writing_a_png(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(plot, "list_corpus", lambda tier: ["a"])
    _write(tmp_path, [_record("a", [_plate(0.05, coverage=0.1), _plate(None)])])

    code = plot.main(["--cache-dir", str(tmp_path), "--out", str(tmp_path / "x.png")])

    assert code == 1
    assert "nothing to plot" in capsys.readouterr().err
    assert not (tmp_path / "x.png").exists()


def test_a_full_cache_renders_a_png_and_prints_the_annotation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(plot, "list_corpus", lambda tier: ["a", "b"])
    _write(tmp_path, [
        _record("a", [_plate(0.28), _plate(0.31), _plate(0.02, coverage=0.1)]),
        _record("b", [_plate(0.33), _plate(0.36)]),
    ])
    out = tmp_path / "chart.png"

    code = plot.main(["--cache-dir", str(tmp_path), "--out", str(out)])
    printed = capsys.readouterr().out

    assert code == 0
    assert out.is_file() and out.stat().st_size > 0
    assert "n = 4 plates - 2 restaurants" in printed
