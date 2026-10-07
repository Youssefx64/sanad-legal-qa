"""PII detection and redaction guardrails for user queries."""

import re

from sanad.corpus.arabic_text import to_western_digits

# Egyptian National ID: 14 digits, starts with 2 (born 1900-1999) or 3 (born 2000-2099)
EGYPTIAN_NATIONAL_ID_PATTERN = re.compile(r"\b[23]\d{13}\b")

# Egyptian Phone numbers: +20 1x xxxxxxxx or 01x xxxxxxxx
EGYPTIAN_PHONE_PATTERN = re.compile(r"(?:\+?20\s*|0)?1[0125][\d\s\-]{8}\b")

# General email address
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Egyptian IBAN: EG + 2 digits + 25 alphanumeric chars
EGYPTIAN_IBAN_PATTERN = re.compile(r"\bEG\d{2}[A-Za-z0-9]{25}\b", re.IGNORECASE)

# Credit Card numbers: 16 digits
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")


def redact_pii(text: str) -> tuple[str, list[dict[str, str]]]:
    """Scrub Personally Identifiable Information (PII) from user text before sending upstream.

    Returns:
        redacted_text: Text with sensitive entities replaced by [TYPE_REDACTED] markers.
        redactions: List of detected redaction metadata dictionaries.
    """
    if not text:
        return "", []

    redactions: list[dict[str, str]] = []
    # Normalize digits for consistent regex evaluation
    redacted = to_western_digits(text)

    # 1. Egyptian National ID
    for m in EGYPTIAN_NATIONAL_ID_PATTERN.finditer(redacted):
        redactions.append({"type": "national_id", "value": m.group(0)})
    redacted = EGYPTIAN_NATIONAL_ID_PATTERN.sub("[NATIONAL_ID_REDACTED]", redacted)

    # 2. Egyptian IBAN
    for m in EGYPTIAN_IBAN_PATTERN.finditer(redacted):
        redactions.append({"type": "iban", "value": m.group(0)})
    redacted = EGYPTIAN_IBAN_PATTERN.sub("[IBAN_REDACTED]", redacted)

    # 3. Email
    for m in EMAIL_PATTERN.finditer(redacted):
        redactions.append({"type": "email", "value": m.group(0)})
    redacted = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", redacted)

    # 4. Egyptian Mobile Phone
    for m in EGYPTIAN_PHONE_PATTERN.finditer(redacted):
        redactions.append({"type": "phone", "value": m.group(0)})
    redacted = EGYPTIAN_PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)

    # 5. Credit Cards
    for m in CREDIT_CARD_PATTERN.finditer(redacted):
        redactions.append({"type": "credit_card", "value": m.group(0)})
    redacted = CREDIT_CARD_PATTERN.sub("[CARD_REDACTED]", redacted)

    return redacted, redactions
