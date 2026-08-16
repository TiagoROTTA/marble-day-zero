"""Snapshot loader: reads the frozen restaurant corpus under `data/restaurants/`.

Plain functions, no LLM calls, no node imports (see CLAUDE.md: "Tools don't
import nodes"). Standard library plus `src.tools.document_parser` only.

Everything here fails loudly. A silent `{}` or a near-empty menu string would
resurface three nodes later as an unexplainable forecast, and there is no time
to debug that during a two-day build.
"""
import base64
import html as html_mod
import json
import re
from pathlib import Path

from src.tools.document_parser import parse_pdf_path

# Keys every meta.json must carry. `estimated_fields`, `tier`, `neighborhood`,
# `service_style` and `snapshot_date` are extra and deliberately tolerated.
REQUIRED_META_KEYS = (
    "slug",
    "name",
    "url",
    "cuisine",
    "price_tier",
    "seats",
    "menu_format",
    "popular_times_index",
    "review_count",
)

# Which artifact file belongs to which declared menu_format, in preference order.
_SNAPSHOT_CANDIDATES = {
    "html": ("source.html", "source.htm"),
    "pdf": ("source.pdf",),
    "image": ("source.jpg", "source.jpeg", "source.png"),
}

# Below this many characters a PDF is almost certainly a scanned image with no
# text layer, not a menu we can hand to the extractor.
MIN_PDF_TEXT_CHARS = 200


def load_restaurant(slug: str, root: str = "data/restaurants") -> dict:
    """Load and validate `<root>/<slug>/meta.json`.

    Raises FileNotFoundError naming the attempted path if the snapshot
    directory or its metadata is missing, ValueError naming the offending key
    if the metadata is incomplete or malformed.
    """
    directory = Path(root) / slug
    if not directory.is_dir():
        raise FileNotFoundError(f"No snapshot directory: {directory.resolve()}")

    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No meta.json: {meta_path.resolve()}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path.resolve()} is not a JSON object")

    for key in REQUIRED_META_KEYS:
        if key not in meta:
            raise ValueError(f"{meta_path.resolve()} is missing required key: {key}")

    index = meta["popular_times_index"]
    if not isinstance(index, list) or len(index) != 7:
        raise ValueError(
            f"{meta_path.resolve()}: popular_times_index must hold 7 values "
            f"(Mon..Sun), got {len(index) if isinstance(index, list) else type(index).__name__}"
        )

    meta["snapshot_path"] = str(_find_snapshot(directory, meta["menu_format"]))
    return meta


def _find_snapshot(directory: Path, menu_format: str) -> Path:
    """Absolute path of the menu artifact sitting next to meta.json."""
    for name in _SNAPSHOT_CANDIDATES.get(menu_format, ()):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()

    for candidate in sorted(directory.glob("source.*")):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"No source.* artifact in {directory.resolve()} for menu_format={menu_format!r}"
    )


def strip_html(raw: str) -> str:
    """Tags out, readable text in. No BeautifulSoup, no new dependency."""
    cleaned = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S | re.I)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.S)
    cleaned = re.sub(r"<[^>]+>", "\n", cleaned)
    cleaned = html_mod.unescape(cleaned)
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_menu_source(restaurant: dict) -> tuple[str, str]:
    """Return `(kind, payload)` where kind is "text" or "image_b64"."""
    menu_format = restaurant["menu_format"]
    path = Path(restaurant["snapshot_path"])

    if menu_format == "html":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return "text", strip_html(raw)

    if menu_format == "pdf":
        extracted = parse_pdf_path(str(path))
        if len(extracted.strip()) < MIN_PDF_TEXT_CHARS:
            raise ValueError(
                f"{path} yielded {len(extracted.strip())} chars of text: it is a "
                f"scanned image with no text layer. Re-snapshot "
                f"{restaurant.get('slug', '?')} as .jpg and set menu_format=\"image\"."
            )
        return "text", extracted

    if menu_format == "image":
        data = path.read_bytes()
        # standard_b64encode emits a single unwrapped line; the Anthropic API
        # rejects base64 containing newlines.
        return "image_b64", base64.standard_b64encode(data).decode()

    raise ValueError(f"Unknown menu_format: {menu_format!r}")


def list_corpus(tier: str | None = None, root: str = "data/restaurants") -> list[str]:
    """Slugs from `index.json`, optionally filtered to tier "A" or "B"."""
    index_path = Path(root) / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No corpus index: {index_path.resolve()}")

    entries = json.loads(index_path.read_text(encoding="utf-8"))
    return [e["slug"] for e in entries if tier is None or e.get("tier") == tier]
