"""Canonical address source.

Default: a deterministic *synthetic* generator of Russian-style canonical
addresses, so the whole pipeline runs end-to-end with no downloads. Swap in the
real GAR/FIAS parser (see parse_gar stub) for the production run (>=500k objects).
"""
import random

REGIONS = [
    "Республика Татарстан", "Московская область", "Свердловская область",
    "Краснодарский край", "Новосибирская область", "Ростовская область",
    "Республика Башкортостан", "Самарская область",
]
CITIES = [
    "Казань", "Москва", "Екатеринбург", "Краснодар", "Новосибирск",
    "Ростов-на-Дону", "Уфа", "Самара", "Набережные Челны", "Сочи",
    "Нижнекамск", "Тольятти",
]
STREETS = [
    "Ленина", "Гагарина", "Пушкина", "Баумана", "Мира", "Советская",
    "Победы", "Кремлёвская", "Центральная", "Молодёжная", "Лесная",
    "Садовая", "Школьная", "Первомайская", "Октябрьская", "Комсомольская",
    "Горького", "Чехова", "Строителей", "Космонавтов", "Дружбы",
    "Пролетарская", "Северная", "Южная", "Заречная", "Полевая",
    "Луговая", "Набережная", "Тукая", "Профсоюзная",
]


def canonical_string(c):
    """Flat string used for embedding."""
    parts = [c["region"], f"г {c['city']}", f"ул {c['street']}", f"д {c['house']}"]
    if c.get("korp"):
        parts.append(f"к {c['korp']}")
    return ", ".join(parts)


def generate_synthetic(n, seed=20260605):
    """Return n unique canonical address dicts (id, region, city, street, house, korp)."""
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n:
        region = rng.choice(REGIONS)
        city = rng.choice(CITIES)
        street = rng.choice(STREETS)
        house = rng.randint(1, 200)
        korp = rng.choice([None, None, None, "1", "2", "3", "А"])
        key = (region, city, street, house, korp)
        if key in seen:
            continue
        seen.add(key)
        c = dict(id=len(out), region=region, city=city, street=street,
                 house=str(house), korp=korp)
        c["text"] = canonical_string(c)
        out.append(c)
    return out


def parse_gar(xml_dir, region_codes=None, limit=None):
    """STUB for the real run. Parse GAR gar_xml.zip -> canonical dicts.

    GAR ships AS_ADDR_OBJ (settlements/streets) and AS_HOUSES (houses) as XML per
    region. Join by OBJECTID/parent to rebuild the hierarchy region->city->street
    ->house, then build canonical_string(). Keep OBJECTGUID as the FIAS id and
    region/level as metadata for filtering.

    Recommended: parse only the needed regions and only ADDR_OBJ levels + HOUSES
    to stay under memory. Return the same schema as generate_synthetic().
    """
    raise NotImplementedError(
        "Plug in GAR XML parsing here. Download gar_xml.zip from fias.nalog.ru, "
        "or use a preconverted dump (Kaggle/HF). Return dicts with keys: "
        "id, region, city, street, house, korp, text.")
