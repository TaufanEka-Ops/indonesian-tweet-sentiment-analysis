"""Indonesian informal-word (slang / texting) normalisation map.

The map is deliberately small and conservative: it only contains tokens whose
standard-Indonesian expansion is unambiguous. It is applied *token by token*
after lower-casing and before stop-word removal, so the stop-word step can then
drop the standardised function words (e.g. ``yg`` -> ``yang`` -> stop-word).

Keeping the map small is a deliberate choice: an over-aggressive map silently
rewrites sentiment-bearing words and hurts the supervised model. Anything not
in this map is left untouched, and the caller can always pass an empty map.

Spellings come from the informal forms actually observed in the Novel Baswedan
tweet set (see ``scripts/02_preprocess_novel.py``) plus common Indonesian
Twitter abbreviations. Only *non-identity* mappings are listed: a token that
would map to itself does not belong here.
"""

from __future__ import annotations

# fmt: off
SLANG_MAP: dict[str, str] = {
    # pronouns / particles
    "aq": "aku", "ak": "aku", "gw": "aku", "gue": "aku", "gua": "aku",
    "sy": "saya", "q": "aku",
    "kmu": "kamu", "km": "kamu", "u": "kamu", "lo": "kamu", "lu": "kamu",
    "mrk": "mereka",
    # common connectors / prepositions
    "yg": "yang", "yng": "yang",
    "dgn": "dengan", "dg": "dengan",
    "utk": "untuk", "untk": "untuk", "buat": "untuk",
    "krn": "karena", "krna": "karena", "karna": "karena",
    "jg": "juga", "jga": "juga",
    "tp": "tapi", "tpi": "tapi", "tetapi": "tapi",
    "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "spt": "seperti", "sprti": "seperti", "kya": "seperti", "kyk": "seperti",
    "kayak": "seperti",
    "dr": "dari", "dri": "dari",
    "kpd": "kepada",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "uda": "sudah",
    "blm": "belum", "blom": "belum",
    "bgt": "sangat", "bngt": "sangat", "banget": "sangat",
    "gmn": "bagaimana", "gimana": "bagaimana", "bgmn": "bagaimana",
    "knp": "kenapa", "napa": "kenapa",
    "trs": "terus", "trus": "terus",
    "sm": "sama", "sma": "sama",
    "aja": "saja", "aj": "saja", "doang": "saja",
    "gak": "tidak", "ga": "tidak", "g": "tidak", "tdk": "tidak",
    "engga": "tidak", "enggak": "tidak", "ngga": "tidak", "nggak": "tidak",
    "kga": "tidak",
    "bs": "bisa", "bsa": "bisa",
    "hrs": "harus", "hrus": "harus",
    "skrg": "sekarang", "skr": "sekarang",
    "hr": "hari", "hrini": "hari ini",
    "thn": "tahun", "thun": "tahun",
    "org": "orang", "orng": "orang",
    "byk": "banyak", "bnyk": "banyak",
    "dkit": "sedikit", "sdikit": "sedikit",
    "jd": "jadi", "jdi": "jadi",
    "msh": "masih", "masi": "masih",
    "pke": "pakai", "pk": "pakai", "pake": "pakai",
    "smua": "semua",
    "mgkn": "mungkin",
    "dpt": "dapat", "dapet": "dapat",
    "tau": "tahu", "tao": "tahu",
    "bkn": "bukan", "bukannya": "bukan",
    "pd": "pada", "pda": "pada",
    "dlm": "dalam", "dlam": "dalam",
    "kluar": "keluar",
    "msuk": "masuk",
    "dkt": "dekat",
    "skit": "sakit",
    "bner": "benar", "bener": "benar", "bnr": "benar",
    "pemrintah": "pemerintah",
    "negri": "negeri",
    "bbrp": "beberapa", "bbrpa": "beberapa",
}
# fmt: on


def normalize(token: str, mapping: dict[str, str] | None = None) -> str:
    """Return the standard-Indonesian form of ``token``.

    Tokens absent from the map are returned unchanged. Matching is exact and
    case-sensitive by design -- callers lower-case first so that case is not
    silently conflated (see :func:`idtweets.preprocessing.lowercase`).
    """
    return (SLANG_MAP if mapping is None else mapping).get(token, token)
