"""Stream a canonical address base from the official GAR / FIAS XML (fias.nalog.ru).

GAR ships one folder per region (2-digit code) with three files we need:

    AS_ADDR_OBJ_*.XML   - address objects (regions/cities/settlements/streets)
                          <OBJECT OBJECTID OBJECTGUID NAME TYPENAME LEVEL
                                  ISACTUAL ISACTIVE .../>
    AS_HOUSES_*.XML     - houses
                          <HOUSE OBJECTID OBJECTGUID HOUSENUM ADDNUM1 ADDNUM2
                                 ISACTUAL ISACTIVE .../>
    AS_ADM_HIERARCHY_*  - the tree; PATH is a dot-chain of OBJECTIDs from the
                          region down to the object
                          <ITEM OBJECTID PARENTOBJID PATH ISACTIVE .../>

We reconstruct region -> city -> street for every actual house and emit the
pipeline schema: dict(region, city, street, house, korp, lat=None, lon=None,
fias_guid=<house OBJECTGUID>, text). Coordinates are attached later by
join_coords.py (GAR has no lat/lon).

Streaming is mandatory: the XML is huge. We parse row-by-row and keep only the
addr-object dictionary + the house->PATH map for one region in memory at a time,
which is fine for the demo regions (Курская обл, Татарстан, Москва ...). For the
whole country prefer a few regions or a preparsed dump.

Address-object LEVEL codes (GAR "Уровни адресных объектов"):
    1 subject (region)      5 city              8 street (road network element)
    2 admin district        6 settlement        9 land plot
    3 municipal district    7 planning struct.  10 building
    4 urban/rural poselenie
"""
import os
import re
import glob
import zipfile

LEVEL_REGION = "1"
CITY_LEVELS = ("5", "6")          # city preferred, then any populated place
STREET_LEVELS = ("8", "7")        # street preferred, then planning structure
# address-object levels emitted as their own canonical records (region/city/
# settlement/planning/street) so a street-only query matches the STREET object
# (with its OBJECTGUID), not the houses on it.
OBJECT_LEVELS = ("1", "5", "6", "7", "8")

POSTAL_TYPEID = "5"               # AS_*_PARAMS TYPEID for the postal index

# HOUSETYPE -> canonical prefix (AS_HOUSE_TYPES SHORTNAME)
HOUSE_TYPE = {
    "1": "влд.", "2": "д.", "3": "двлд.", "4": "г-ж", "5": "зд.", "6": "шахта",
    "7": "стр.", "8": "соор.", "9": "литера", "10": "к.", "11": "подв.",
    "12": "кот.", "13": "п-б", "14": "ОНС",
}
# ADDTYPE1/2 -> separator for the extra house part (AS_ADDHOUSE_TYPES)
ADD_TYPE = {"1": "к.", "2": "стр.", "3": "соор.", "4": "литера"}

# region-style types read best as "<name> <type>" (Курская обл, Пермский край);
# everything else as "<type> <name>" (Респ Татарстан, г Казань, ул Баумана).
_TYPE_AFTER = {"обл", "край", "ао", "аобл", "р-н", "округ"}


def _iter_rows(path):
    """Yield each non-root element of a GAR XML, keeping memory flat.

    Prefers lxml (huge_tree, recover); falls back to the stdlib parser so the
    module imports even without lxml installed.
    """
    try:
        from lxml import etree
        context = etree.iterparse(path, events=("end",), recover=True,
                                  huge_tree=True)
        for _, elem in context:
            yield elem
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]
        del context
    except ImportError:
        import xml.etree.ElementTree as ET
        context = ET.iterparse(path, events=("start", "end"))
        _, root = next(context)                     # ('start', root)
        for event, elem in context:
            if event != "end" or elem is root:
                continue
            yield elem
            root.clear()                            # drop processed siblings


def _find(region_dir, prefix):
    """First XML file named '<prefix>_<date>_...XML' in region_dir.

    GAR region folders carry look-alike siblings (AS_HOUSES_PARAMS,
    AS_ADDR_OBJ_PARAMS, AS_ADDR_OBJ_DIVISION ...). A plain startswith would grab
    the 979 MB *_PARAMS file instead of the object file, so we require the token
    right after the prefix to be the date (digits), not another word.
    """
    pat = re.compile("^" + re.escape(prefix) + r"_\d", re.I)
    for f in sorted(glob.glob(os.path.join(region_dir, "*"))):
        base = os.path.basename(f)
        if base.upper().endswith(".XML") and pat.match(base):
            return f
    return None


def _fmt(name, typename):
    if not name:
        return None
    typename = (typename or "").strip()
    if not typename:
        return name
    if typename.rstrip(".").lower() in _TYPE_AFTER:   # GAR types carry a dot ("обл.")
        return f"{name} {typename}"
    return f"{typename} {name}"


def _region_dirs(root, region_codes=None):
    """GAR root holds one folder per region code; also accept a single region
    folder passed directly."""
    if _find(root, "AS_HOUSES"):                    # root IS a region folder
        return [root]
    dirs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if region_codes and name not in region_codes:
            continue
        if _find(full, "AS_HOUSES"):
            dirs.append(full)
    return dirs


def _load_addr_objects(path):
    """OBJECTID -> (name, typename, level, guid) for actual+active address objects."""
    id2obj = {}
    for elem in _iter_rows(path):
        oid = elem.get("OBJECTID")
        if oid is None or elem.get("NAME") is None:
            continue
        if elem.get("ISACTUAL") != "1" or elem.get("ISACTIVE") != "1":
            continue
        id2obj[oid] = (elem.get("NAME"), elem.get("TYPENAME"), elem.get("LEVEL"),
                       elem.get("OBJECTGUID"))
    return id2obj


def _load_paths(path):
    """OBJECTID -> PATH string (dot-chain of OBJECTIDs), active items only."""
    path_of = {}
    for elem in _iter_rows(path):
        oid = elem.get("OBJECTID")
        p = elem.get("PATH")
        if oid is None or not p:
            continue
        if elem.get("ISACTIVE") not in (None, "1"):
            continue
        path_of[oid] = p
    return path_of


def _resolve(path_str, id2obj):
    """Walk a house PATH and pick region / city / street.

    Returns (region_text, city_bare, street_bare, city_text, street_text):
    bare names feed the coordinate join, the *_text (with type words) feed the
    canonical string.
    """
    region = None
    city = street = city_full = street_full = None
    for oid in path_str.split("."):
        obj = id2obj.get(oid)
        if not obj:
            continue
        name, typename, level, _guid = obj
        if level == LEVEL_REGION and region is None:
            region = _fmt(name, typename)
        elif level in CITY_LEVELS:
            if city is None or level == "5":        # prefer a real city
                city = name                         # bare name for the coord join
                city_full = _fmt(name, typename)
        elif level in STREET_LEVELS:
            if street is None or level == "8":      # prefer a real street
                street = name
                street_full = _fmt(name, typename)
    return region, city, street, city_full, street_full


def _match_member(members, prefix):
    """Zip member '<prefix>_<date>_...XML' (skips *_PARAMS / *_DIVISION siblings)."""
    pat = re.compile("^" + re.escape(prefix) + r"_\d", re.I)
    for n in sorted(members):
        base = n.split("/")[-1]
        if base.upper().endswith(".XML") and pat.match(base):
            return n
    return None


def _regions_from_dir(root, region_codes):
    """Yield (label, open_ao, open_houses, open_hier, open_hparams)."""
    for region_dir in _region_dirs(root, region_codes):
        ao = _find(region_dir, "AS_ADDR_OBJ")
        houses = _find(region_dir, "AS_HOUSES")
        hier = (_find(region_dir, "AS_ADM_HIERARCHY")
                or _find(region_dir, "AS_MUN_HIERARCHY"))
        hparams = _find(region_dir, "AS_HOUSES_PARAMS")
        if ao and houses and hier:
            label = os.path.basename(os.path.normpath(region_dir))
            op = lambda p: (lambda q=p: open(q, "rb"))
            yield (label, op(ao), op(houses), op(hier),
                   op(hparams) if hparams else None)


def _regions_from_zip(zip_path, region_codes):
    """Same, reading members straight out of gar_xml.zip — no extraction."""
    zf = zipfile.ZipFile(zip_path)
    by_top = {}
    for n in zf.namelist():
        parts = n.split("/")
        if len(parts) < 2 or not parts[1]:
            continue
        by_top.setdefault(parts[0], []).append(n)
    op = lambda m: (lambda x=m: zf.open(x))
    for code in sorted(by_top):
        if region_codes and code not in region_codes:
            continue
        members = by_top[code]
        ao = _match_member(members, "AS_ADDR_OBJ")
        houses = _match_member(members, "AS_HOUSES")
        hier = (_match_member(members, "AS_ADM_HIERARCHY")
                or _match_member(members, "AS_MUN_HIERARCHY"))
        hparams = _match_member(members, "AS_HOUSES_PARAMS")
        if ao and houses and hier:
            yield (code, op(ao), op(houses), op(hier),
                   op(hparams) if hparams else None)


def _load_postal(src):
    """OBJECTID -> postal index from an AS_*_PARAMS file (current rows only)."""
    postal = {}
    for elem in _iter_rows(src):
        if elem.get("TYPEID") != POSTAL_TYPEID:
            continue
        if (elem.get("CHANGEIDEND") or "0") != "0":   # keep the current value
            continue
        oid, val = elem.get("OBJECTID"), elem.get("VALUE")
        if oid and val:
            postal[oid] = val
    return postal


def _house_bits(elem):
    """Render the house as canonical parts: 'д. 5', optionally '+ к. 1', '+ стр. 2'.

    Returns (parts, korp) where korp is the корпус number for the coordinate join
    (строение is kept in the text but not in the join key, matching OSM).
    """
    htype = HOUSE_TYPE.get(elem.get("HOUSETYPE") or "", "д.")
    parts = [f"{htype} {elem.get('HOUSENUM')}"]
    add1, at1 = elem.get("ADDNUM1"), elem.get("ADDTYPE1")
    add2, at2 = elem.get("ADDNUM2"), elem.get("ADDTYPE2")
    if add1:
        parts.append(f"{ADD_TYPE.get(at1 or '', 'к.')} {add1}")
    if add2:
        parts.append(f"{ADD_TYPE.get(at2 or '', 'стр.')} {add2}")
    korp = add1 if (add1 and (at1 in (None, "", "1"))) else None
    return parts, korp


def iter_canon(source, region_codes=None, limit=None, want_postal=False,
               include_houses=True, include_objects=False):
    """Yield canonical records from a GAR tree *or* gar_xml.zip (streaming).

    `source` may be the unzipped GAR root, a single region folder, or the
    gar_xml.zip itself — in the last case members are read directly from the
    archive (no disk extraction). With want_postal=True the postal index is
    joined from AS_HOUSES_PARAMS (a large file; opt-in).

    include_houses -> emit one record per house (default). include_objects ->
    also (or only) emit the address OBJECTS themselves (region/city/street) so a
    street-only query resolves to the STREET object, not the houses on it.
    """
    if isinstance(source, str) and source.lower().endswith(".zip"):
        regions = _regions_from_zip(source, region_codes)
    else:
        regions = _regions_from_dir(source, region_codes)

    yielded = 0
    for label, open_ao, open_houses, open_hier, open_hparams in regions:
        with open_ao() as f:
            id2obj = _load_addr_objects(f)
        with open_hier() as f:
            path_of = _load_paths(f)
        house_postal = {}
        if include_houses and want_postal and open_hparams:
            with open_hparams() as f:
                house_postal = _load_postal(f)

        if include_houses:
            with open_houses() as fh:
                for elem in _iter_rows(fh):
                    oid = elem.get("OBJECTID")
                    housenum = elem.get("HOUSENUM")
                    if oid is None or not housenum:
                        continue
                    if elem.get("ISACTUAL") != "1" or elem.get("ISACTIVE") != "1":
                        continue
                    p = path_of.get(oid)
                    if not p:
                        continue
                    region, city, street, city_full, street_full = _resolve(p, id2obj)
                    if not street and not city:
                        continue                    # nothing to anchor on
                    house_parts, korp = _house_bits(elem)
                    postal = house_postal.get(oid)

                    parts = []
                    if postal:
                        parts.append(postal)
                    if region:
                        parts.append(region)
                    if city_full:
                        parts.append(city_full)
                    if street_full:
                        parts.append(street_full)
                    parts.extend(house_parts)

                    yield dict(region=region, city=city, street=street,
                               house=housenum, korp=korp, lat=None, lon=None,
                               region_code=label, postal=postal,
                               fias_guid=elem.get("OBJECTGUID"),
                               text=", ".join(parts))
                    yielded += 1
                    if limit and yielded >= limit:
                        return

        if include_objects:
            for oid, (name, typename, level, guid) in id2obj.items():
                if level not in OBJECT_LEVELS:
                    continue
                p = path_of.get(oid)
                if not p:
                    continue
                region, city, street, city_full, street_full = _resolve(p, id2obj)
                oparts = [x for x in (region, city_full, street_full) if x]
                if not oparts:
                    continue
                yield dict(region=region, city=city, street=street,
                           house=None, korp=None, lat=None, lon=None,
                           region_code=label, postal=None, fias_guid=guid,
                           level=level, text=", ".join(oparts))
                yielded += 1
                if limit and yielded >= limit:
                    return

        id2obj.clear()
        path_of.clear()


def parse_gar(xml_dir, region_codes=None, limit=None, want_postal=False):
    """API-compatible with the src.data stub: return a list of canonical dicts.

    For millions of houses prefer iter_canon(...) + streaming to jsonl
    (build_dataset_fias.py); this materialises everything and is meant for small
    subsets / tests.
    """
    out = []
    for i, c in enumerate(iter_canon(xml_dir, region_codes, limit, want_postal)):
        c = dict(c, id=i)
        out.append(c)
    return out
