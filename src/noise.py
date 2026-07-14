"""Synthetic noise generator for Russian addresses.

Turns a clean canonical address into a "dirty" user-typed variant, labelled by
error category. Categories follow the AddrLLM error taxonomy (Table 5, arXiv
2411.13584), adapted to Russian:

    misspelling      - character-level typos
    abbreviation     - expand/contract/drop type words (ул<->улица, д<->дом ...)
    missing_region   - drop the region (address-entity-prediction case)
    irrelevant_words - inject junk (postal index, "РФ", "рядом с метро", ...)
    reorder          - shuffle components / append a second nested place
    transliteration  - render the street name in Latin letters

Each variant is tied to the known canonical FIAS id -> free (query, relevance)
labels for evaluation.
"""
import random
import string

# type-word abbreviations, both directions
TYPE_MAP = {
    "улица": "ул", "проспект": "пр-т", "переулок": "пер", "проезд": "пр-д",
    "бульвар": "б-р", "шоссе": "ш", "набережная": "наб", "площадь": "пл",
    "дом": "д", "корпус": "к", "строение": "стр", "город": "г",
    "область": "обл", "район": "р-н", "республика": "респ", "деревня": "д.",
}
ABBR_TO_FULL = {v: k for k, v in TYPE_MAP.items()}

JUNK = ["РФ", "Россия", "рядом с метро", "напротив школы", "2 подъезд",
        "кв 45", "домофон 12", "оф 3", "ЖК Солнечный"]

# minimal RU->Latin table for the transliteration category
TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

CATEGORIES = ["misspelling", "abbreviation", "missing_region",
              "irrelevant_words", "reorder", "transliteration"]


def _typo(token, rng):
    """Apply one random character operation to a token."""
    if len(token) < 3:
        return token
    i = rng.randrange(len(token))
    op = rng.choice(["swap", "delete", "duplicate", "replace"])
    if op == "swap" and i < len(token) - 1:
        return token[:i] + token[i + 1] + token[i] + token[i + 2:]
    if op == "delete":
        return token[:i] + token[i + 1:]
    if op == "duplicate":
        return token[:i] + token[i] + token[i:]
    # replace with a random cyrillic letter
    return token[:i] + rng.choice("абвгдеёжзиклмнопрстуфхцчшщыэюя") + token[i + 1:]


def transliterate(text):
    return "".join(TRANSLIT.get(ch, ch) for ch in (text or "").lower())


def make_dirty(comp, category, rng):
    """Build a dirty string for the given structured components and category.

    comp: dict(region, city, street, house, korp)
    returns: dirty string
    """
    region = comp.get("region")
    city = comp.get("city")
    street = comp.get("street")
    house = comp.get("house")
    korp = comp.get("korp")

    def canon(reg=True):
        # None-safe: real FIAS rows may lack a city (rural) or a street (house
        # hangs directly under a settlement), so only emit the parts we have.
        parts = []
        if reg and region:
            parts.append(region)
        if city:
            parts.append(f"г {city}")
        if street:
            parts.append(f"ул {street}")
        if house:
            parts.append(f"д {house}")
        if korp:
            parts.append(f"к {korp}")
        return ", ".join(parts)

    if category == "misspelling":
        toks = canon().split()
        # corrupt 1-2 word tokens
        idxs = [i for i, t in enumerate(toks) if len(t) >= 3]
        for i in rng.sample(idxs, k=min(2, len(idxs))):
            toks[i] = _typo(toks[i], rng)
        return " ".join(toks)

    if category == "abbreviation":
        s = canon()
        # randomly expand or contract each abbreviation, or drop the type word
        for full, ab in TYPE_MAP.items():
            if rng.random() < 0.5:
                s = s.replace(f"{ab} ", f"{full} ")   # expand
        if rng.random() < 0.4:
            s = s.replace("ул ", "").replace("г ", "")  # drop type words
        return s

    if category == "missing_region":
        if rng.random() < 0.5:
            return canon(reg=False)
        loc = " ".join(p for p in [street, house] if p) or house or ""
        return ", ".join(p for p in [f"г {city}" if city else "", loc] if p)

    if category == "irrelevant_words":
        s = canon()
        junk = rng.choice(JUNK)
        idx6 = "".join(rng.choice(string.digits) for _ in range(6))
        pos = rng.choice(["prefix", "suffix"])
        s = f"{idx6}, {s}, {junk}" if pos == "suffix" else f"{junk}, {s}"
        return s

    if category == "reorder":
        parts = canon().split(", ")
        rng.shuffle(parts)
        s = ", ".join(parts)
        if rng.random() < 0.4:      # nested: append a second random place
            s += ", " + rng.choice(["Москва", "Санкт-Петербург", "Новосибирск"])
        return s

    if category == "transliteration":
        target = street or city                       # transliterate whatever name exists
        if not target:
            return canon()
        head = f"{city}, " if (street and city) else ""
        return f"{head}ul {transliterate(target)} {house}"

    raise ValueError(category)


def make_eval_set(canonicals, per_category, seed=20260605):
    """Build (dirty_query, gold_id, category) triples.

    canonicals: list of dicts with keys id, region, city, street, house, korp
    """
    rng = random.Random(seed)
    eval_set = []
    for cat in CATEGORIES:
        sample = rng.sample(canonicals, k=min(per_category, len(canonicals)))
        for c in sample:
            eval_set.append((make_dirty(c, cat, rng), c["id"], cat))
    rng.shuffle(eval_set)
    return eval_set
