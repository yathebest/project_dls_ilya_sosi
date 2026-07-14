"""Canonical address source.

Default: a deterministic *synthetic* generator of Russian-style canonical
addresses, so the whole pipeline runs end-to-end with no downloads. Swap in the
real GAR/FIAS parser (see parse_gar stub) for the production run (>=500k objects).
"""
import json
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
    parts = []
    if c.get("region"):
        parts.append(c["region"])
    if c.get("city"):
        parts.append(f"г {c['city']}")
    parts.append(f"ул {c['street']}")
    parts.append(f"д {c['house']}")
    if c.get("korp"):
        parts.append(f"к {c['korp']}")
    return ", ".join(parts)


def save_canon(canon, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in canon:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_canon(path):
    """Load a canonical base from jsonl; ids are re-numbered to row index
    (search returns positions, so id must equal position)."""
    canon = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            c["id"] = len(canon)
            if not c.get("text"):
                c["text"] = canonical_string(c)
            canon.append(c)
    return canon


def get_canonicals(n, dataset=None, seed=20260605):
    """Synthetic base by default, or a real jsonl base if --dataset given."""
    if dataset:
        return load_canon(dataset)
    return generate_synthetic(n, seed)


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
    """Parse the official GAR / FIAS XML into canonical dicts.

    GAR ships AS_ADDR_OBJ (settlements/streets) and AS_HOUSES (houses) as XML per
    region, joined via AS_ADM_HIERARCHY (PATH). We rebuild region->city->street
    ->house, keep OBJECTGUID as the FIAS id, and build the canonical text.

    Implemented (streaming) in src/fias.py. For millions of houses use
    fias.iter_canon(...) + build_dataset_fias.py instead of this list form.
    """
    from src.fias import parse_gar as _parse_gar
    return _parse_gar(xml_dir, region_codes=region_codes, limit=limit)
