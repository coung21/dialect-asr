"""Shared Vietnamese transcript normalization utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_vietnamese_text(text: str) -> str:
    """Normalize ASR text while preserving Vietnamese diacritics and digits."""
    normalized = unicodedata.normalize("NFC", text).lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()
