"""Address key normalization for joining two bases (FIAS <-> OSM).

FIAS gives canonical names + hierarchy but no coordinates; the OSM base has
coordinates. We match on a normalized (city, street, house) key so lat/lon can
be carried from OSM onto the FIAS canon.

The key strips type words ("ул", "улица", "г", "город" ...), lowercases,
folds ё->е, and canonicalizes the house/корпус spelling so that
    "12 к 1" == "12к1" == "12/1" == "12 корпус 1"   (== norm_house("12","1")).
"""
import re

# type words dropped from a city / street name (full and abbreviated forms)
_TYPE_WORDS = {
    "ул", "улица", "пер", "переулок", "пр", "пр-кт", "проспект", "пр-д",
    "проезд", "б-р", "бул", "бульвар", "ш", "шоссе", "наб", "набережная",
    "пл", "площадь", "туп", "тупик", "аллея", "линия", "кв-л", "квартал",
    "мкр", "микрорайон", "тракт", "г", "город", "гор", "с", "село", "д",
    "деревня", "пос", "п", "поселок", "посёлок", "пгт", "рп", "ст", "станция",
    "ст-ца", "станица", "х", "хутор", "тер", "территория",
}

# tokens keep internal hyphens (пр-кт, Ростов-на-Дону); we strip them afterward
_WORD = re.compile(r"[0-9a-zа-яё-]+")
_HOUSE_SEP = re.compile(r"^(\d+)([а-яё]?)[кk/\\]+(\d+[а-яё]?)$")   # 12к1, 12/1
_HOUSE_CORE = re.compile(r"(\d+[а-яёa-z]?)")


def _fold(s):
    return (s or "").lower().replace("ё", "е").strip()


def _strip_types(name):
    """Drop type words, keep meaningful tokens, join with single spaces."""
    toks = [t.strip("-") for t in _WORD.findall(_fold(name))]
    toks = [t for t in toks if t and t not in _TYPE_WORDS]
    return " ".join(toks)


def norm_house(house, korp=None):
    """Canonical house token: '<core>' or '<core>к<korp>'.

    Explicit korp wins; otherwise an embedded корпус/строCTURE separator
    (к / k / / / \\) in the house string is detected.
    """
    h = _fold(house).replace(" ", "")
    if korp:
        m = _HOUSE_CORE.match(h)
        core = m.group(1) if m else h
        return f"{core}к{_fold(korp).replace(' ', '')}"
    m = _HOUSE_SEP.match(h)
    if m:
        return f"{m.group(1)}{m.group(2)}к{m.group(3)}"
    m = _HOUSE_CORE.match(h)
    return m.group(1) if m else h


def norm_key(city, street, house, korp=None):
    """Tuple key used for the FIAS<->OSM coordinate join."""
    return (_strip_types(city), _strip_types(street), norm_house(house, korp))
