import json
from pathlib import Path

import pytest

from src.nodes.ingest import ingest_node
from src.tools.snapshot import (
    REQUIRED_META_KEYS,
    list_corpus,
    load_menu_source,
    load_restaurant,
    strip_html,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "restaurants"

GOOD_META = {
    "slug": "test-diner",
    "name": "Test Diner",
    "url": "https://example.com/menu",
    "cuisine": "american",
    "price_tier": "$$",
    "seats": 40,
    "service_style": "full-service",
    "neighborhood": "Nowhere",
    "snapshot_date": "2026-08-08",
    "menu_format": "html",
    "popular_times_index": [0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 0.8],
    "review_count": 120,
    "tier": "A",
    # Extra key the real corpus carries; required-key validation must tolerate it.
    "estimated_fields": ["popular_times_index", "seats", "review_count"],
}

SAMPLE_HTML = """<!doctype html>
<html><head>
  <title>Test Diner</title>
  <style>.menu { color: #fff; }</style>
  <script>var price = "<h1>not a heading</h1>";</script>
</head>
<body>
  <!-- a comment that should vanish -->
  <h1>Dinner</h1>
  <div class="item"><span>Cheeseburger</span><span>$16.50</span></div>
  <div class="item"><span>Caesar&nbsp;Salad</span><span>$13.00</span></div>
  <p>Fish &amp; Chips &mdash; $19.00</p>
</body></html>
"""


def _write_restaurant(root: Path, meta: dict, source_name: str, source_text: str) -> Path:
    directory = root / meta["slug"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / source_name).write_text(source_text, encoding="utf-8")
    return directory


# --- load_restaurant -------------------------------------------------------


def test_load_restaurant_good_load(tmp_path):
    _write_restaurant(tmp_path, GOOD_META, "source.html", SAMPLE_HTML)

    meta = load_restaurant("test-diner", root=str(tmp_path))

    assert meta["name"] == "Test Diner"
    assert meta["menu_format"] == "html"
    assert len(meta["popular_times_index"]) == 7
    # The extra corpus key survives validation untouched.
    assert meta["estimated_fields"] == ["popular_times_index", "seats", "review_count"]
    snapshot = Path(meta["snapshot_path"])
    assert snapshot.is_absolute()
    assert snapshot.name == "source.html"
    assert snapshot.is_file()


def test_load_restaurant_missing_slug_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_restaurant("does-not-exist", root=str(tmp_path))
    # The attempted path must be in the message, or the operator cannot debug it.
    assert "does-not-exist" in str(excinfo.value)


def test_load_restaurant_missing_meta_json_raises_filenotfound(tmp_path):
    (tmp_path / "empty-diner").mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_restaurant("empty-diner", root=str(tmp_path))
    assert "meta.json" in str(excinfo.value)


def test_load_restaurant_short_popular_times_index_raises_valueerror(tmp_path):
    meta = dict(GOOD_META, popular_times_index=[0.5, 0.6, 0.7])
    _write_restaurant(tmp_path, meta, "source.html", SAMPLE_HTML)

    with pytest.raises(ValueError) as excinfo:
        load_restaurant("test-diner", root=str(tmp_path))
    assert "popular_times_index" in str(excinfo.value)


@pytest.mark.parametrize("missing", REQUIRED_META_KEYS)
def test_load_restaurant_missing_required_key_names_the_key(tmp_path, missing):
    meta = {k: v for k, v in GOOD_META.items() if k != missing}
    directory = tmp_path / "test-diner"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / "source.html").write_text(SAMPLE_HTML, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_restaurant("test-diner", root=str(tmp_path))
    assert missing in str(excinfo.value)


def test_load_restaurant_missing_source_artifact_raises_filenotfound(tmp_path):
    directory = tmp_path / "test-diner"
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps(GOOD_META), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_restaurant("test-diner", root=str(tmp_path))


# --- strip_html / load_menu_source ----------------------------------------


def test_strip_html_keeps_menu_content_and_drops_markup():
    text = strip_html(SAMPLE_HTML)

    assert "<" not in text and ">" not in text
    assert "Cheeseburger" in text
    assert "$16.50" in text
    assert "Caesar Salad" in text          # &nbsp; became a real space
    assert "Fish & Chips — $19.00" in text  # entities unescaped
    # script/style bodies are gone, including markup hidden inside a string
    assert "not a heading" not in text
    assert "color: #fff" not in text
    assert "a comment that should vanish" not in text
    # no runaway blank lines
    assert "\n\n\n" not in text


def test_load_menu_source_html_returns_text(tmp_path):
    _write_restaurant(tmp_path, GOOD_META, "source.html", SAMPLE_HTML)
    restaurant = load_restaurant("test-diner", root=str(tmp_path))

    kind, payload = load_menu_source(restaurant)

    assert kind == "text"
    assert "Cheeseburger" in payload


def test_load_menu_source_image_returns_unwrapped_base64(tmp_path):
    meta = dict(GOOD_META, menu_format="image")
    directory = tmp_path / meta["slug"]
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # 900 bytes so the encoding is long enough that a wrapping encoder would
    # insert newlines — the Anthropic API rejects wrapped base64.
    (directory / "source.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x01\x02\x03" * 300)

    restaurant = load_restaurant("test-diner", root=str(tmp_path))
    kind, payload = load_menu_source(restaurant)

    assert kind == "image_b64"
    assert "\n" not in payload
    assert len(payload) > 1000


def test_load_menu_source_rejects_unknown_format(tmp_path):
    meta = dict(GOOD_META, menu_format="html")
    _write_restaurant(tmp_path, meta, "source.html", SAMPLE_HTML)
    restaurant = load_restaurant("test-diner", root=str(tmp_path))
    restaurant["menu_format"] = "docx"

    with pytest.raises(ValueError):
        load_menu_source(restaurant)


# --- list_corpus ------------------------------------------------------------


def test_list_corpus_reads_real_index():
    all_slugs = list_corpus(root=str(CORPUS))
    tier_a = list_corpus("A", root=str(CORPUS))
    tier_b = list_corpus("B", root=str(CORPUS))

    assert all_slugs
    assert set(tier_a) | set(tier_b) == set(all_slugs)
    assert not set(tier_a) & set(tier_b)
    assert "joes-pizza-carmine" in tier_a


def test_list_corpus_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_corpus(root=str(tmp_path))


# --- real corpus round trip -------------------------------------------------


def test_real_tier_a_restaurant_loads_and_yields_a_payload():
    slug = list_corpus("A", root=str(CORPUS))[0]
    restaurant = load_restaurant(slug, root=str(CORPUS))
    kind, payload = load_menu_source(restaurant)

    assert restaurant["name"]
    assert kind in ("text", "image_b64")
    assert len(payload) > 500


# --- ingest_node ------------------------------------------------------------


def test_ingest_node_success_sets_restaurant_and_stage():
    slug = list_corpus("A", root=str(CORPUS))[0]
    out = ingest_node({"input": slug, "retry_count": 0})

    assert out["stage"] == "ingested"
    assert out["restaurant"]["slug"] == slug
    assert "retry_count" not in out


def test_ingest_node_missing_slug_increments_retry_count():
    out = ingest_node({"input": "no-such-restaurant", "retry_count": 2})

    assert out == {
        "retry_count": 3,
        "last_error": out["last_error"],
    }
    assert out["last_error"].startswith("FileNotFoundError: ")
    assert "restaurant" not in out
    assert "stage" not in out
