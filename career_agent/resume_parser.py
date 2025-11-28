"""Utilities for extracting and cleaning resume text."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pypdf import PdfReader

from .config import RESUME_MAX_CHARS


def extract_text_from_pdf(source: Union[str, Path]) -> str:
    """Read a PDF from disk and return concatenated text."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover
            text_parts.append(f"[Could not read page: {exc}]")
    return _clean_text("\n".join(text_parts))


def extract_text_from_upload(upload) -> str:
    """Handle Streamlit file_uploader objects."""
    if upload is None:
        return ""
    reader = PdfReader(upload)
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return _clean_text("\n".join(text_parts))


def _clean_text(raw: str) -> str:
    text = " ".join(raw.split())
    return text[:RESUME_MAX_CHARS]

