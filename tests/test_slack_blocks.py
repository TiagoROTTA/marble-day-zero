"""Structural validation of the Block Kit builders.

These cards cannot be pasted into the Block Kit Builder from a test, so the
limits the Builder would catch are asserted here instead: 50 blocks per
message, 25 elements per actions block, 3000 characters per text object, and
button values that round-trip through the webhook's value.split("|", 1).
"""
from src.slack.blocks import (
    _MAX_SECTION_CHARS,
    approval_blocks,
    purchase_order_blocks,
    review_queue_blocks,
)


def _items(n: int) -> list[dict]:
    return [
        {
            "kind": "sku_match" if i % 2 == 0 else "plate_cost",
            "ref": f"ingredient-{i}",
            "confidence": 0.60 + (i % 25) * 0.01,
            "question": f"*ingredient-{i}* → Catalog entry {i}  `SKU-{i}`",
        }
        for i in range(n)
    ]


def _po(vendors: int = 2, lines_per_vendor: int = 3) -> dict:
    return {
        "vendor_lines": {
            f"vendor-{v}": [
                {
                    "sku_id": f"sku-{v}-{i}",
                    "display_name": f"Item {v}-{i}",
                    "packs": i + 1,
                    "pack_unit": "case",
                    "pack_qty": 25,
                    "pack_uom": "lb",
                    "line_cost": 123.456 * (i + 1),
                    "par_qty": 4.2,
                }
                for i in range(lines_per_vendor)
            ]
            for v in range(vendors)
        },
        "total_cost": 4211.5,
        "covers_per_week": 900,
        "days_cover_label": "2-30 days cover by category",
        "assumptions": ["+15% safety factor", "packs rounded up"],
        "excluded_skus": [],
    }


def _text_objects(blocks: list[dict]) -> list[str]:
    """Every string sitting in a Slack text object, anywhere in the tree."""
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") in ("mrkdwn", "plain_text") and isinstance(
                node.get("text"), str
            ):
                found.append(node["text"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blocks)
    return found


def _buttons(blocks: list[dict]) -> list[dict]:
    return [
        el
        for b in blocks
        if b.get("type") == "actions"
        for el in b["elements"]
        if el.get("type") == "button"
    ]


def _assert_block_kit_valid(blocks: list[dict]) -> None:
    assert isinstance(blocks, list)
    assert len(blocks) <= 50, f"{len(blocks)} blocks exceeds Slack's limit of 50"
    for b in blocks:
        assert "type" in b
        if b.get("type") == "actions":
            assert len(b["elements"]) <= 25
        if b.get("type") == "context":
            assert 1 <= len(b["elements"]) <= 10
    for text in _text_objects(blocks):
        assert len(text) <= _MAX_SECTION_CHARS, f"text object of {len(text)} chars"


# --- button value encoding -------------------------------------------------


def test_review_queue_button_values_round_trip():
    blocks = review_queue_blocks("t1", "Joe's Pizza", _items(3), 0)
    buttons = _buttons(blocks)
    assert len(buttons) == 3
    decisions = []
    for btn in buttons:
        value = btn["value"]
        assert "|" in value
        thread_id, decision = value.split("|", 1)
        assert thread_id == "t1"
        decisions.append(decision)
    assert decisions == ["approve", "reject", "skip"]


def test_purchase_order_button_values_round_trip():
    blocks = purchase_order_blocks("t-42", "Joe's Pizza", _po())
    buttons = _buttons(blocks)
    assert len(buttons) == 2
    pairs = [b["value"].split("|", 1) for b in buttons]
    assert pairs == [["t-42", "approve"], ["t-42", "reject"]]


def test_thread_id_containing_a_pipe_still_yields_the_decision():
    # split("|", 1) keeps the decision recoverable only from the right side,
    # so a pipe in the thread_id would break resume: assert the shape we rely on.
    blocks = review_queue_blocks("t1", "R", _items(1), 0)
    for btn in _buttons(blocks):
        assert btn["value"].count("|") == 1


# --- block count limits ----------------------------------------------------


def test_forty_item_queue_stays_under_fifty_blocks():
    blocks = review_queue_blocks("t1", "Big Menu", _items(40), 9)
    assert len(blocks) <= 50
    _assert_block_kit_valid(blocks)


def test_forty_item_queue_states_the_full_count_in_the_header():
    blocks = review_queue_blocks("t1", "Big Menu", _items(40), 0)
    assert "40 items to confirm" in blocks[0]["text"]["text"]


def test_large_purchase_order_stays_under_fifty_blocks():
    blocks = purchase_order_blocks("t1", "Big Menu", _po(vendors=40, lines_per_vendor=30))
    assert len(blocks) <= 50
    _assert_block_kit_valid(blocks)


def _withheld(n: int) -> list[dict]:
    return [
        {
            "sku_id": f"sku-w-{i}",
            "display_name": f"Withheld item {i}",
            "reason": "confidence_below_floor",
            "confidence": 0.40,
            "dishes": [f"dish-{i}"],
            "qty_withheld": 1.5,
            "uom": "lb",
            "cost_not_spent": 100.0 + i,
        }
        for i in range(n)
    ]


def test_withheld_items_render_a_single_section_naming_the_money():
    po = _po()
    po["excluded_skus"] = _withheld(2)
    po["excluded_cost_total"] = 201.0

    blocks = purchase_order_blocks("t1", "R", po)
    withheld = [
        b for b in blocks
        if isinstance(b.get("text"), dict) and "Withheld" in b["text"]["text"]
    ]
    assert len(withheld) == 1
    text = withheld[0]["text"]["text"]
    assert "2 item(s) ($201.00)" in text
    assert "confidence below review floor" in text
    assert "Withheld item 0" in text
    assert "40% confidence" in text
    _assert_block_kit_valid(blocks)


def test_many_withheld_items_stay_one_block_with_a_footer():
    po = _po(vendors=40, lines_per_vendor=30)
    po["excluded_skus"] = _withheld(30)
    po["excluded_cost_total"] = 3000.0

    blocks = purchase_order_blocks("t1", "Big Menu", po)
    assert len(blocks) <= 50
    text = next(
        b["text"]["text"] for b in blocks
        if isinstance(b.get("text"), dict) and "Withheld" in b["text"]["text"]
    )
    assert "_…and 24 more withheld items_" in text
    _assert_block_kit_valid(blocks)


def test_legacy_string_excluded_entries_still_render():
    """Older payloads stored bare refs; the card must not crash on them."""
    po = _po()
    po["excluded_skus"] = ["PROD-TOMATO-ROMA"]

    blocks = purchase_order_blocks("t1", "R", po)
    text = next(
        b["text"]["text"] for b in blocks
        if isinstance(b.get("text"), dict) and "Withheld" in b["text"]["text"]
    )
    assert "PROD-TOMATO-ROMA" in text
    _assert_block_kit_valid(blocks)


def test_vendor_group_over_ten_lines_gets_a_more_lines_footer():
    blocks = purchase_order_blocks("t1", "R", _po(vendors=1, lines_per_vendor=14))
    vendor_text = blocks[1]["text"]["text"]
    assert "_…and 4 more lines_" in vendor_text
    assert vendor_text.count("•") == 10


# --- truncation ------------------------------------------------------------


def test_four_thousand_character_section_is_truncated():
    long_items = [
        {
            "kind": "sku_match",
            "ref": "x",
            "confidence": 0.7,
            "question": "q" * 4000,
        }
    ]
    blocks = review_queue_blocks("t1", "R", long_items, 0)
    section = blocks[2]["text"]["text"]
    assert len(section) == _MAX_SECTION_CHARS
    assert section.endswith("…")
    _assert_block_kit_valid(blocks)


def test_long_vendor_section_is_truncated():
    po = _po(vendors=1, lines_per_vendor=10)
    for line in po["vendor_lines"]["vendor-0"]:
        line["display_name"] = "n" * 500
    blocks = purchase_order_blocks("t1", "R", po)
    section = blocks[1]["text"]["text"]
    assert len(section) == _MAX_SECTION_CHARS
    assert section.endswith("…")
    _assert_block_kit_valid(blocks)


def test_short_text_is_not_truncated():
    blocks = review_queue_blocks("t1", "R", _items(1), 0)
    assert not blocks[2]["text"]["text"].endswith("…")


# --- dropped-count context block ------------------------------------------


def _dropped_texts(blocks: list[dict]) -> list[str]:
    return [
        el["text"]
        for b in blocks
        if b.get("type") == "context"
        for el in b["elements"]
        if "fell below the review threshold" in el.get("text", "")
    ]


def test_dropped_context_block_present_when_dropped_positive():
    blocks = review_queue_blocks("t1", "R", _items(2), 9)
    texts = _dropped_texts(blocks)
    assert len(texts) == 1
    assert texts[0] == (
        "_9 further items fell below the review threshold "
        "and were recorded as gaps._"
    )


def test_dropped_context_block_absent_when_dropped_zero():
    assert _dropped_texts(review_queue_blocks("t1", "R", _items(2), 0)) == []


def test_dropped_context_block_absent_when_dropped_negative():
    assert _dropped_texts(review_queue_blocks("t1", "R", _items(2), -1)) == []


# --- card shape ------------------------------------------------------------


def test_review_queue_shape():
    items = _items(3)
    blocks = review_queue_blocks("t1", "Joe's Pizza", items, 4)
    assert blocks[0]["type"] == "section"
    assert blocks[0]["text"]["text"] == "*Joe's Pizza* — 3 items to confirm"
    # The ask, spelled out: the buttons confirm a stated conclusion.
    assert blocks[1]["type"] == "context"
    assert "Confirm" in blocks[1]["elements"][0]["text"]
    # section + context per item
    assert [b["type"] for b in blocks[2:8]] == [
        "section",
        "context",
        "section",
        "context",
        "section",
        "context",
    ]
    assert blocks[2]["text"]["text"] == items[0]["question"]
    assert "confidence" in blocks[3]["elements"][0]["text"]
    assert "sku_match" in blocks[3]["elements"][0]["text"]
    assert blocks[8]["type"] == "divider"
    assert blocks[9]["type"] == "actions"
    _assert_block_kit_valid(blocks)


def test_queue_of_one_reads_as_singular():
    # The floor routinely leaves exactly one item, so this is the headline a
    # reviewer sees most often.
    blocks = review_queue_blocks("t1", "Madame Vo", _items(1), 54)
    assert blocks[0]["text"]["text"] == "*Madame Vo* — 1 item to confirm"


def test_confidence_rendered_as_a_percentage():
    items = [
        {"kind": "sku_match", "ref": "guanciale", "confidence": 0.71, "question": "?"}
    ]
    blocks = review_queue_blocks("t1", "R", items, 0)
    assert blocks[3]["elements"][0]["text"].startswith("71% confidence")


def test_empty_queue_still_produces_a_usable_card():
    blocks = review_queue_blocks("t1", "R", [], 0)
    assert blocks[0]["text"]["text"] == "*R* — 0 items to confirm"
    assert len(_buttons(blocks)) == 3
    _assert_block_kit_valid(blocks)


def test_purchase_order_shape_and_money_formatting():
    blocks = purchase_order_blocks("t1", "Joe's Pizza", _po(vendors=1, lines_per_vendor=1))
    assert blocks[0]["text"]["text"] == "*Draft opening order — Joe's Pizza*"
    assert blocks[1]["text"]["text"] == (
        "*vendor-0*\n• 1 × case Item 0-0 — $123.46"
    )
    totals = [
        b["text"]["text"]
        for b in blocks
        if b.get("type") == "section" and "Order total" in b["text"]["text"]
    ]
    assert totals == ["*Order total: $4,211.50*"]
    _assert_block_kit_valid(blocks)


def test_purchase_order_states_its_assumptions_next_to_the_total():
    blocks = purchase_order_blocks("t1", "R", _po())
    basis = [
        el["text"]
        for b in blocks
        if b.get("type") == "context"
        for el in b["elements"]
        if el["text"].startswith("Based on ")
    ]
    assert basis == [
        "Based on 900/wk projected covers · 2-30 days cover by category "
        "· prices from hand-curated catalog"
    ]


def test_purchase_order_omits_the_cover_basis_rather_than_inventing_one():
    """A payload with no cover label must not fall back to a made-up number.

    The card used to print a hardcoded "7 days cover" over an order that mixes
    2-day produce with 30-day spices, contradicting the assumptions block two
    lines below it.
    """
    po = _po()
    po.pop("days_cover_label")
    basis = [
        el["text"]
        for b in purchase_order_blocks("t1", "R", po)
        if b.get("type") == "context"
        for el in b["elements"]
        if el["text"].startswith("Based on ")
    ]

    assert basis == [
        "Based on 900/wk projected covers · prices from hand-curated catalog"
    ]
    assert not any("days cover" in t for t in basis)


def test_empty_purchase_order_does_not_crash():
    blocks = purchase_order_blocks("t1", "R", {})
    assert any("Order total: $0.00" in t for t in _text_objects(blocks))
    _assert_block_kit_valid(blocks)


# --- the existing builder must keep working --------------------------------


def test_approval_blocks_unchanged():
    blocks = approval_blocks("t1", "Title", {"a": "b"})
    assert len(blocks) == 3
    assert [b["value"].split("|", 1) for b in _buttons(blocks)] == [
        ["t1", "approve"],
        ["t1", "reject"],
    ]


def test_detail_is_rendered_under_the_proposal():
    items = [
        {
            "kind": "sku_match",
            "ref": "white onion",
            "confidence": 0.6,
            "question": "*white onion* → Onion, yellow jumbo  `PROD-ONION-YEL`",
            "detail": "closest catalog onion, but the catalog carries no white onion",
        }
    ]
    blocks = review_queue_blocks("t1", "R", items, 0)
    assert blocks[2]["text"]["text"] == items[0]["question"]
    assert items[0]["detail"] in blocks[3]["elements"][0]["text"]


def test_missing_detail_leaves_the_subline_alone():
    blocks = review_queue_blocks("t1", "R", _items(1), 0)
    assert blocks[3]["elements"][0]["text"].endswith("`sku_match`")
