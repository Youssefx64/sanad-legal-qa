"""Position-aware bilingual PDF extractor for the Egyptian Civil Code."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf


@dataclass
class PageLine:
    """A line of text with vertical coordinate on the page."""

    y0: float
    text: str
    is_heading: bool = False


@dataclass
class PageData:
    """Extracted bilingual lines and metadata for a single PDF page."""

    page_number: int
    ar_lines: list[PageLine] = field(default_factory=list)
    en_lines: list[PageLine] = field(default_factory=list)


PAGE_FURNITURE_PATTERNS = [
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),  # e.g. - 105 -
    re.compile(r"^\s*ITI\b", re.IGNORECASE),  # ITI headers/footers
]


def is_page_furniture(text: str) -> bool:
    """Check if text is running furniture (page numbers, footers)."""
    stripped = text.strip()
    if not stripped:
        return True
    return any(pat.match(stripped) for pat in PAGE_FURNITURE_PATTERNS)


def extract_pdf_pages(pdf_path: str | Path) -> list[PageData]:
    """Extract position-aware bilingual columns from the Egyptian Civil Code PDF.

    Splits each page into two columns:
    - Left column (x < midpoint): English text
    - Right column (x >= midpoint): Arabic text
    Handles span-level boundaries for lines crossing the center.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    doc = pymupdf.open(str(path))
    pages_data: list[PageData] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_width = page.rect.width
        split_x = page_width / 2.0

        d: dict[str, Any] = page.get_text("dict")
        en_lines: list[PageLine] = []
        ar_lines: list[PageLine] = []

        for block in d.get("blocks", []):
            if block.get("type") != 0:  # Skip image blocks
                continue
            for line in block.get("lines", []):
                lx0, ly0, lx1, _ = line["bbox"]
                line_spans = line.get("spans", [])
                line_text = " ".join(
                    s["text"].strip() for s in line_spans if s.get("text", "").strip()
                ).strip()

                if not line_text or is_page_furniture(line_text):
                    continue

                # Determine column assignment based on horizontal coordinates
                if lx1 < split_x:
                    en_lines.append(PageLine(y0=ly0, text=line_text))
                elif lx0 >= split_x - 15:
                    ar_lines.append(PageLine(y0=ly0, text=line_text))
                else:
                    # Line crosses center column; split spans into their respective columns
                    for s in line_spans:
                        stext = s.get("text", "").strip()
                        if not stext or is_page_furniture(stext):
                            continue
                        sx0, sy0, _, _ = s["bbox"]
                        if sx0 < split_x:
                            en_lines.append(PageLine(y0=sy0, text=stext))
                        else:
                            ar_lines.append(PageLine(y0=sy0, text=stext))

        # Sort lines strictly top-to-bottom
        en_lines.sort(key=lambda line: line.y0)
        ar_lines.sort(key=lambda line: line.y0)

        pages_data.append(
            PageData(
                page_number=page_num,
                ar_lines=ar_lines,
                en_lines=en_lines,
            )
        )

    doc.close()
    return pages_data
