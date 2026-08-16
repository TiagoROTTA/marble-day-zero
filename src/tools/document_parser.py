"""Lightweight document parsing: PDF (bytes or path) and URL (raw HTML/text).

For richer parsing (OCR, layout-aware extraction, table detection),
swap pypdf here for unstructured / pdfplumber / Claude vision.
"""
import io

import httpx
from pypdf import PdfReader


def parse_pdf_bytes(data: bytes) -> str:
    """Extract text from a PDF given its bytes."""
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_pdf_path(path: str) -> str:
    """Extract text from a PDF on disk."""
    with open(path, "rb") as f:
        return parse_pdf_bytes(f.read())


def parse_url_text(url: str, timeout: float = 20.0) -> str:
    """Fetch a URL and return its raw text body."""
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.text
