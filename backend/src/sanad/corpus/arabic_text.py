"""Arabic text normalization and cleaning utilities for Sanad."""

import re

# Arabic-Indic to Western digits translation table
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
INDIC_TO_WESTERN = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)

# Diacritics regex: Fathatan, Dammatan, Kasratan, Fatha, Damma, Kasra, Shadda, Sukun, Superscript Alef
TASHKEEL_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")
TATWEEL_CHAR = "\u0640"

# Alef variations
ALEF_PATTERN = re.compile(r"[إأآٱ]")


def to_western_digits(text: str) -> str:
    """Convert Arabic-Indic digits (٠-٩) to Western ASCII digits (0-9)."""
    return text.translate(INDIC_TO_WESTERN) if text else ""


def normalize_arabic(text: str, norm_teh_marbuta: bool = False) -> str:
    """Normalize Arabic text for search, indexing, and embeddings.

    Applies the following transformations:
    - Removes tatweel (kashida)
    - Removes diacritics (harakat / tashkeel)
    - Unifies alef forms (أ, إ, آ, ٱ -> ا)
    - Unifies alef maksura (ى -> ي)
    - Optionally normalizes teh marbuta (ة -> ه)
    - Translates Arabic-Indic digits to Western digits
    - Collapses multiple whitespace characters to a single space
    """
    if not text:
        return ""

    # Remove tatweel
    normalized = text.replace(TATWEEL_CHAR, "")

    # Remove diacritics
    normalized = TASHKEEL_PATTERN.sub("", normalized)

    # Unify alef forms
    normalized = ALEF_PATTERN.sub("ا", normalized)

    # ى -> ي
    normalized = normalized.replace("ى", "ي")

    # Optional teh marbuta normalization
    if norm_teh_marbuta:
        normalized = normalized.replace("ة", "ه")

    # Normalize Arabic-Indic digits
    normalized = normalized.translate(INDIC_TO_WESTERN)

    # Collapse multiple whitespaces and strip
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def clean_arabic_raw(text: str) -> str:
    """Clean raw extracted Arabic text from PDF artifacts while preserving readability.

    - Removes private use Unicode glyphs (e.g. \\ue812)
    - Cleans stray detached tanween or accents at line starts/ends
    - Normalizes punctuation marks
    - Collapses abnormal spaces
    """
    if not text:
        return ""

    # Remove private-use unicode range (\uE000-\uF8FF)
    cleaned = re.sub(r"[\uE000-\uF8FF]", "", text)

    # Remove non-breaking and unusual spaces
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "")

    # Clean stray standalone tanween
    cleaned = re.sub(r"(?:^|\s)[\u064B-\u064D](?:\s|$)", " ", cleaned)

    # Normalize Arabic quotation marks
    cleaned = cleaned.replace("«", '"').replace("»", '"')
    cleaned = cleaned.replace("”", '"').replace("“", '"')
    cleaned = cleaned.replace("’", "'").replace("‘", "'")

    # Collapse excessive spaces within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    # Remove empty lines while preserving paragraph boundaries (max 1 empty line)
    result_lines: list[str] = []
    for line in lines:
        if line:
            result_lines.append(line)
        elif result_lines and result_lines[-1] != "":
            result_lines.append("")

    return "\n".join(result_lines).strip()


def extract_arabic_digits(text: str) -> int | None:
    """Extract and parse article number digits from an Arabic marker.

    Handles both Western and Arabic-Indic digits, and detects reversed digits.
    """
    if not text:
        return None

    digits_raw = re.findall(r"[\u0660-\u0669\d]+", text)
    if not digits_raw:
        return None

    # Concatenate found digit segments
    raw_str = "".join(digits_raw).translate(INDIC_TO_WESTERN)
    if not raw_str:
        return None

    try:
        val = int(raw_str)
        # Check if the reversed representation is in range 1..1149
        rev_val = int(raw_str[::-1])
        if 1 <= val <= 1149 and 1 <= rev_val <= 1149:
            # If both in range, return val (caller cross-checks with expected sequence)
            return val
        if 1 <= rev_val <= 1149:
            return rev_val
        if 1 <= val <= 1149:
            return val
        return val
    except ValueError:
        return None


def extract_article_references(
    text_ar: str, text_en: str = "", current_article: int | None = None
) -> list[int]:
    """Extract article numbers cross-referenced in the article body."""
    refs: set[int] = set()

    # Search in Arabic text
    # e.g., "المادة 147", "الماده 1", "مادة 5", "م/ 123", "المادتين 219 و 220", "المواد من 54 إلى 80"
    ar_norm = text_ar.translate(INDIC_TO_WESTERN)
    for m in re.finditer(
        r"(?:الماد[ةه]|ماد[ةه]|المادتين|المادتان|المواد|م\s*/?)\s*(?:رقم\s*)?(?:من\s*)?(\d+)(?:\s*(?:و|أو|او|إلى|الي|حتى|حتي|-)\s*(\d+))?",
        ar_norm,
    ):
        for grp in (m.group(1), m.group(2)):
            if grp:
                num = int(grp)
                if 1 <= num <= 1149 and num != current_article:
                    refs.add(num)

    # Search in English text
    # e.g., "Article 147", "Articles 219 and 220"
    for m in re.finditer(r"Articles?\s+(\d+)", text_en, re.IGNORECASE):
        num = int(m.group(1))
        if 1 <= num <= 1149 and num != current_article:
            refs.add(num)

    # Check for "Articles X and Y"
    for m in re.finditer(r"Articles?\s+(\d+)\s+and\s+(\d+)", text_en, re.IGNORECASE):
        n1, n2 = int(m.group(1)), int(m.group(2))
        if 1 <= n1 <= 1149 and n1 != current_article:
            refs.add(n1)
        if 1 <= n2 <= 1149 and n2 != current_article:
            refs.add(n2)

    return sorted(refs)
