"""Conservative detection of contact personally-identifiable information.

Policy for this repo: third-party tweets are republished only after rows that
contain a *personal* phone number or an e-mail address have been dropped. The
detector below is deliberately tuned to catch real contact details and to
ignore the much more common look-alikes (years like ``2014 2019``, sums like
``72000``, complaint reference numbers inside prose).

The detector is keyword-anchored where possible:

* e-mail -> regex, no anchor needed;
* phone -> a 10-13 digit run that either carries a country/trunk prefix, or
  appears within a short window of a contact keyword ("call", "whatsapp",
  "mbl", "hp", "tlp", "telp", "no", "number"), or is written with a
  separator pattern typical of phone numbers (``+91 98765 43210``).

Anything that is flagged is dropped and counted -- never silently used.
"""

from __future__ import annotations

import re

import pandas as pd

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# A *contiguous* digit run of mobile/landline length. We deliberately do NOT
# accept space-separated digit groups here: enumerations like "117 118 122 133"
# (rankings, years) are far more common in tweets than spaced phone numbers,
# and genuinely spaced numbers are caught by the keyword/`+CC` rules below.
PHONE_BARE_RE = re.compile(r"(?<!\d)\d{10,13}(?!\d)")

# Keyword-anchored: contact word appears within ~12 chars of a digit run.
CONTACT_KEYWORDS = (
    "call", "whatsapp", "wa", "mbl", "mobile", "hp", "tlp", "telp", "telepon",
    "phone", "number", "no.", "kontak", "contact", "sms", "hubungi",
)
_CONTACT_KW_RE = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in CONTACT_KEYWORDS) + r")[^a-z0-9]{0,12}"
    r"(?:\+?\d[\d\s\-().]{7,17}\d)",
    re.IGNORECASE,
)

# Country-code style: +CC then 8-12 digits (optionally spaced).
INTL_PHONE_RE = re.compile(r"\+\d{1,3}[\s\-]?\d{6,12}")


def find_emails(text: str) -> list[str]:
    return EMAIL_RE.findall(text or "")


def find_phones(text: str) -> list[str]:
    """Return plausible personal phone numbers in ``text``.

    Three independent rules, union of matches:

    1. a contiguous 10-13 digit run (Indian/Indonesian mobile or landline);
    2. a digit run sitting next to a contact keyword ("call", "whatsapp",
       "mbl", "number", ...);
    3. a ``+CC`` international number.

    Separated digit groups are *not* matched on their own, because tweets are
    full of year/rank/sum enumerations that would swamp the true positives.
    """
    if not text:
        return []
    hits: list[str] = []
    hits.extend(PHONE_BARE_RE.findall(text))
    hits.extend(INTL_PHONE_RE.findall(text))
    hits.extend(m.group(0) for m in _CONTACT_KW_RE.finditer(text))
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def has_contact_pii(text: str) -> bool:
    """True if the row looks like it contains an e-mail or a phone number."""
    return bool(find_emails(text)) or bool(find_phones(text))


def flag_pii(series: pd.Series) -> pd.Series:
    """Boolean mask of rows containing contact PII."""
    return series.astype("string").fillna("").map(has_contact_pii)


def find_pii_hits(series: pd.Series) -> pd.DataFrame:
    """DataFrame of rows with PII and the matched strings (for the audit log)."""
    hits = []
    for idx, text in series.astype("string").fillna("").items():
        t = str(text)
        emails = find_emails(t)
        phones = find_phones(t)
        if emails or phones:
            hits.append(
                {
                    "row_index": idx,
                    "emails": ";".join(emails),
                    "phones": ";".join(phones),
                }
            )
    return pd.DataFrame(hits)
