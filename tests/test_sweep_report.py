"""Guards on the sweep aggregation.

Nothing here touches the network or the graph: the script reads cached JSON and
does arithmetic, so every expected number below is worked out by hand in a
comment rather than recorded from a run. Synthetic records live in tmp_path;
`data/output/` is never written to.
"""
import json

import pytest

import scripts.sweep_report as sweep
from scripts.validate_foodcost import COVERAGE_FLOOR


def _record(slug, cuisine="pizza", menu_format="html", **overrides):
    """A minimal cache record in the shape scripts/sweep_report.py reads."""
    state = {
        "menu_items": [],
        "recipes": [],
        "sku_matches": [],
        "plate_costs": [],
        "review_queue": [],
    }
    state.update(overrides.pop("state", {}))
    record = {
        "slug": slug,
        "tier": "B",
        "cuisine": cuisine,
        "menu_format": menu_format,
        "elapsed_s": 40.0,
        "usage": {"calls": 5, "input_tokens": 20_000, "output_tokens": 3_000,
                  "cache_read": 17_000, "cache_write": 1_200},
        "error": "",
        "state": state,
    }
    record.update(overrides)
    return record


def _write(directory, records):
    for record in records:
        (directory / f"{record['slug']}-costed.json").write_text(
            json.dumps(record), encoding="utf-8"
        )


# --- the canonicalization breakdown ----------------------------------------


def test_breakdown_groups_on_the_method_field():
    matches = [
        {"raw_name": "flour", "sku_id": "FLR-01", "method": "alias"},
        {"raw_name": "mozzarella", "sku_id": "MOZ-01", "method": "alias"},
        {"raw_name": "roma tomatoes", "sku_id": "TOM-01", "method": "normalized"},
        {"raw_name": "san marzano", "sku_id": "TOM-01", "method": "llm"},
    ]

    assert sweep.canon_breakdown(matches) == {
        "alias": 2, "normalized": 1, "llm": 1, "unmatched": 0
    }


def test_a_null_sku_id_is_unmatched_whatever_the_method_says():
    """`canonicalize` returns sku_id None with method "llm" when the catalog
    holds no genuine match. That is a catalog gap, not an llm match, and
    counting it as one would flatter exactly the number this report exposes."""
    matches = [
        {"raw_name": "star anise", "sku_id": None, "method": "llm"},
        {"raw_name": "fish sauce", "sku_id": None, "method": "llm"},
        {"raw_name": "rice paper", "sku_id": None, "method": "alias"},  # defensive
        {"raw_name": "flour", "sku_id": "FLR-01", "method": "llm"},
    ]

    assert sweep.canon_breakdown(matches) == {
        "alias": 0, "normalized": 0, "llm": 1, "unmatched": 3
    }


def test_breakdown_of_nothing_is_all_zeroes():
    assert sweep.canon_breakdown([]) == {
        "alias": 0, "normalized": 0, "llm": 0, "unmatched": 0
    }


def test_unmatched_rate_is_the_null_share():
    state = {"sku_matches": [
        {"sku_id": "A", "method": "alias"},
        {"sku_id": None, "method": "llm"},
        {"sku_id": None, "method": "llm"},
        {"sku_id": "B", "method": "llm"},
    ]}
    assert sweep.unmatched_rate(state) == 0.5


def test_unmatched_rate_without_matches_is_none_not_zero():
    """Zero would read as perfect coverage; None reads as no measurement."""
    assert sweep.unmatched_rate({"sku_matches": []}) is None


# --- the coverage floor on the median --------------------------------------


def test_median_food_cost_excludes_plates_below_the_coverage_floor():
    # 0.90 and 0.60 are in; 0.40 is below the 0.5 floor and must not drag the
    # median down. Included ratios are 0.30 and 0.32 -> median 0.31.
    state = {"plate_costs": [
        {"item_name": "A", "food_cost_pct": 0.30, "coverage": 0.90},
        {"item_name": "B", "food_cost_pct": 0.32, "coverage": 0.60},
        {"item_name": "C", "food_cost_pct": 0.08, "coverage": 0.40},
    ]}
    assert sweep.median_food_cost(state) == pytest.approx(0.31)


def test_a_plate_exactly_at_the_floor_is_included():
    state = {"plate_costs": [
        {"item_name": "A", "food_cost_pct": 0.20, "coverage": COVERAGE_FLOOR},
    ]}
    assert sweep.median_food_cost(state) == pytest.approx(0.20)


def test_median_food_cost_ignores_plates_with_no_ratio():
    state = {"plate_costs": [
        {"item_name": "A", "food_cost_pct": None, "coverage": 1.0},
        {"item_name": "B", "food_cost_pct": 0.25, "coverage": 1.0},
    ]}
    assert sweep.median_food_cost(state) == pytest.approx(0.25)


def test_median_food_cost_of_nothing_is_none():
    assert sweep.median_food_cost({"plate_costs": []}) is None


def test_mean_coverage_is_over_costable_plates_only():
    # (0.9 + 0.7) / 2 = 0.80, not / 3.
    state = {"plate_costs": [
        {"coverage": 0.9, "costable": True},
        {"coverage": 0.7, "costable": True},
        {"coverage": 0.0, "costable": False},
    ]}
    assert sweep.mean_coverage(state) == pytest.approx(0.80)


def test_a_plate_without_the_costable_flag_counts_as_costable():
    """Cache files written before the field existed must not read as uncostable."""
    assert sweep.mean_coverage({"plate_costs": [{"coverage": 0.9}]}) == pytest.approx(0.9)


# --- the cross-cuts --------------------------------------------------------


def _with_items(slug, cuisine, menu_format, confidences, coverages):
    return _record(
        slug, cuisine=cuisine, menu_format=menu_format,
        state={
            "menu_items": [{"name": f"i{i}", "confidence": c}
                           for i, c in enumerate(confidences)],
            "plate_costs": [{"coverage": c, "costable": True} for c in coverages],
        },
    )


def test_format_cross_cut_reports_n_and_the_two_means():
    records = [
        _with_items("a", "pizza", "html", [1.0, 0.8], [0.9, 0.9]),
        _with_items("b", "thai", "html", [0.6], [0.5]),
        _with_items("c", "sushi", "pdf", [0.4], [0.3]),
    ]

    groups = {g["menu_format"]: g for g in sweep.by_format(records)}

    assert groups["html"]["n"] == 2
    # Mean per restaurant, then across restaurants: (0.9 + 0.6) / 2 = 0.75.
    assert groups["html"]["extraction_confidence"] == pytest.approx(0.75)
    assert groups["html"]["coverage"] == pytest.approx(0.70)
    assert groups["pdf"]["n"] == 1
    assert groups["pdf"]["extraction_confidence"] == pytest.approx(0.40)


def test_format_cross_cut_averages_per_restaurant_not_per_item():
    """A 90-item menu must not decide the format's score on its own."""
    big = _with_items("big", "pizza", "html", [0.2] * 90, [0.2])
    small = _with_items("small", "thai", "html", [1.0], [1.0])

    group = sweep.by_format([big, small])[0]

    # Per restaurant: (0.2 + 1.0) / 2 = 0.6. Pooled per item it would be ~0.209.
    assert group["extraction_confidence"] == pytest.approx(0.6)


def test_cuisine_cross_cut_reports_unmatched_rate_and_group_size():
    def matches(n_matched, n_null):
        return [{"sku_id": "A", "method": "alias"}] * n_matched + \
               [{"sku_id": None, "method": "llm"}] * n_null

    records = [
        _record("v1", cuisine="vietnamese", state={"sku_matches": matches(1, 3)}),
        _record("v2", cuisine="vietnamese", state={"sku_matches": matches(3, 1)}),
        _record("p1", cuisine="pizza", state={"sku_matches": matches(9, 1)}),
    ]

    groups = {g["cuisine"]: g for g in sweep.by_cuisine(records)}

    assert groups["vietnamese"]["n"] == 2
    # (0.75 + 0.25) / 2 = 0.50
    assert groups["vietnamese"]["unmatched_rate"] == pytest.approx(0.50)
    assert groups["pizza"]["n"] == 1
    assert groups["pizza"]["unmatched_rate"] == pytest.approx(0.10)


def test_a_group_with_no_measurable_restaurant_reports_none_not_zero():
    records = [_record("empty", cuisine="fusion")]
    group = sweep.by_cuisine(records)[0]

    assert group["n"] == 1
    assert group["unmatched_rate"] is None


# --- the cost arithmetic ---------------------------------------------------


def test_cost_prices_cache_reads_at_the_cached_rate():
    # One restaurant: 20k input total, of which 17k cache read and 1.2k cache
    # write, leaving 1.8k fresh input; 3k output.
    #   fresh   1_800 * 5.00 / 1e6 = 0.009
    #   read   17_000 * 0.50 / 1e6 = 0.0085
    #   write   1_200 * 6.25 / 1e6 = 0.0075
    #   out     3_000 * 25.00 / 1e6 = 0.075
    #                              total 0.100
    cost = sweep.cost_rollup([_record("a")])

    assert cost["fresh_input_tokens"] == 1_800
    assert cost["total_cost_usd"] == pytest.approx(0.100)
    assert cost["cost_per_restaurant_usd"] == pytest.approx(0.100)
    assert cost["cache_read_share"] == pytest.approx(17_000 / 20_000)


def test_cost_scales_with_the_corpus_and_reports_per_restaurant():
    cost = sweep.cost_rollup([_record("a"), _record("b"), _record("c")])

    assert cost["input_tokens"] == 60_000
    assert cost["total_cost_usd"] == pytest.approx(0.300)
    assert cost["cost_per_restaurant_usd"] == pytest.approx(0.100)


def test_cost_never_bills_negative_fresh_input():
    """If a provider ever reports cache_read alone as the input total, the
    subtraction must clamp rather than produce a negative bill."""
    record = _record("a", usage={"calls": 1, "input_tokens": 1_000,
                                 "output_tokens": 0, "cache_read": 5_000,
                                 "cache_write": 0})
    cost = sweep.cost_rollup([record])

    assert cost["fresh_input_tokens"] == 0
    assert cost["total_cost_usd"] >= 0


def test_latency_rollup_uses_mean_and_median():
    records = [
        _record("a", elapsed_s=10.0),
        _record("b", elapsed_s=20.0),
        _record("c", elapsed_s=60.0),
    ]
    cost = sweep.cost_rollup(records)

    assert cost["mean_elapsed_s"] == pytest.approx(30.0)
    assert cost["median_elapsed_s"] == pytest.approx(20.0)


def test_cost_of_an_empty_corpus_does_not_divide_by_zero():
    cost = sweep.cost_rollup([])
    assert cost["total_cost_usd"] == 0.0
    assert cost["cost_per_restaurant_usd"] == 0.0


# --- the table has no empty cells ------------------------------------------


def test_every_cell_is_non_empty_even_for_a_bare_record():
    row = sweep.restaurant_row(_record("bare", cuisine=None, menu_format=None))

    for key, _ in sweep._COLUMNS:
        assert row[key], f"{key} is empty"
        assert row[key].strip() == row[key] or row[key].strip(), f"{key} is blank"
    assert row["cuisine"] == "--"
    assert row["menu_format"] == "--"
    assert row["item_conf"] == "--"
    assert row["coverage"] == "--"
    assert row["food_cost"] == "--"
    assert row["items"] == "0"


def test_rendered_table_never_emits_an_empty_cell():
    records = [
        _record("bare", cuisine=None, menu_format=None),
        _with_items("full", "thai", "pdf", [0.9], [0.8]),
        _record("broken", error="halted at a human gate before costing"),
    ]
    table = sweep.render_table([sweep.restaurant_row(r) for r in records])

    body = table.splitlines()[2:]
    assert len(body) == 3
    for line in body:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(sweep._COLUMNS)
        assert all(cells), f"empty cell in: {line}"


def test_the_full_report_renders_valid_markdown_without_crashing():
    records = [
        _with_items("a", "pizza", "html", [1.0], [0.9]),
        _with_items("b", "vietnamese", "pdf", [0.5], [0.4]),
        _record("c", cuisine=None, menu_format=None, error="boom"),
    ]

    text = sweep.render(records, tier="B")

    assert text.startswith("# Sweep report")
    assert "Tier swept: **B**" in text
    assert "3 restaurants" in text
    assert "1 failed" in text
    assert "## Cross-cut: menu format" in text
    assert "## Cross-cut: cuisine" in text
    assert "## Cost and latency" in text
    # Group-size warnings must be stated in prose, not left implicit.
    assert "n=1" in text
    assert "|  |" not in text, "an empty table cell leaked into the report"


def test_a_thin_format_group_is_called_out_in_prose():
    records = [_with_items("only-pdf", "sushi", "pdf", [0.5], [0.5])]
    prose = "\n".join(sweep._format_prose(sweep.by_format(records)))

    assert "n=1" in prose
    assert "anecdote" in prose


def test_the_format_ordering_is_computed_not_assumed():
    """When pdf beats html the report must say the expectation was wrong."""
    records = [
        _with_items("h1", "pizza", "html", [0.5], [0.2]),
        _with_items("h2", "pizza", "html", [0.5], [0.2]),
        _with_items("h3", "pizza", "html", [0.5], [0.2]),
        _with_items("p1", "sushi", "pdf", [0.9], [0.9]),
        _with_items("p2", "sushi", "pdf", [0.9], [0.9]),
        _with_items("p3", "sushi", "pdf", [0.9], [0.9]),
    ]
    prose = "\n".join(sweep._format_prose(sweep.by_format(records)))

    assert "contradicts" in prose
    assert "pdf > html" in prose


# --- the failure list ------------------------------------------------------


def test_worst_plates_are_the_lowest_coverage_ones_with_their_restaurant():
    records = [
        _record("a", state={"plate_costs": [
            {"item_name": "Pho", "coverage": 0.2, "uncosted": ["star anise"]},
            {"item_name": "Banh Mi", "coverage": 0.9, "uncosted": []},
        ]}),
        _record("b", state={"plate_costs": [
            {"item_name": "Ramen", "coverage": 0.1, "uncosted": ["kombu", "mirin"]},
        ]}),
    ]

    worst = sweep.worst_plates(records, 2)

    assert [p["item_name"] for p in worst] == ["Ramen", "Pho"]
    assert worst[0]["slug"] == "b"
    assert worst[0]["uncosted"] == ["kombu", "mirin"]


def test_uncosted_names_are_tallied_case_insensitively_across_the_corpus():
    records = [
        _record("a", state={"plate_costs": [
            {"item_name": "x", "coverage": 0.1, "uncosted": ["Star Anise", "kombu"]},
            {"item_name": "y", "coverage": 0.1, "uncosted": ["star anise"]},
        ]}),
        _record("b", state={"plate_costs": [
            {"item_name": "z", "coverage": 0.1, "uncosted": [" STAR ANISE "]},
        ]}),
    ]

    counts = dict(sweep.uncosted_counts(records))

    assert counts["star anise"] == 3
    assert counts["kombu"] == 1


def test_top_failures_output_is_ascii_only(capsys):
    """A U+2192 in this project once crashed the cp1252 Windows console."""
    records = [_record("a", state={"plate_costs": [
        {"item_name": "Pho", "coverage": 0.2, "uncosted": ["star anise"]},
    ]})]

    sweep.print_top_failures(records, 5)
    out = capsys.readouterr().out

    out.encode("ascii")  # raises if anything non-ASCII slipped in
    assert "star anise" in out


# --- the CLI ---------------------------------------------------------------


def test_no_cache_files_exits_non_zero_and_says_what_to_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sweep, "OUTPUT_DIR", tmp_path)

    code = sweep.main(["--out", str(tmp_path / "r.md")])
    err = capsys.readouterr().err

    assert code != 0
    assert "scripts.validate_foodcost" in err


def test_main_writes_the_report_from_the_cache(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sweep, "OUTPUT_DIR", tmp_path)
    _write(tmp_path, [
        _with_items("a", "pizza", "html", [0.9], [0.8]),
        _with_items("b", "vietnamese", "pdf", [0.6], [0.4]),
    ])
    target = tmp_path / "sweep-report.md"

    code = sweep.main(["--out", str(target)])
    capsys.readouterr()

    assert code == 0
    text = target.read_text(encoding="utf-8")
    assert "# Sweep report" in text
    assert "2 restaurants" in text
    assert "| a |" in text and "| b |" in text


def test_main_never_runs_the_pipeline(monkeypatch, tmp_path, capsys):
    """The script must be free to re-run. No graph, no LLM, ever."""
    import src.graph as graph_module

    def boom(*args, **kwargs):
        raise AssertionError("sweep_report must never compile the graph")

    monkeypatch.setattr(graph_module, "get_compiled_graph", boom)
    monkeypatch.setattr(sweep, "OUTPUT_DIR", tmp_path)
    _write(tmp_path, [_with_items("a", "pizza", "html", [0.9], [0.8])])

    assert sweep.main(["--out", str(tmp_path / "r.md")]) == 0
    capsys.readouterr()
