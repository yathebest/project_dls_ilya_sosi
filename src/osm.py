"""Load real Russian addresses (with coordinates) from OpenStreetMap.

Primary path: Overpass API — fetch addressed nodes/ways (addr:street +
addr:housenumber) for an administrative area or bbox. Coordinates come for
free (→ enables the geocoding-aware encoder head, AddrLLM Fig.4).

For the full >=500k run, either query several regions here, or read a
Geofabrik region .pbf (see load_pbf, needs GDAL/geopandas).

Output schema matches the pipeline: dict(region, city, street, house, korp,
lat, lon, osm_id) -> to_canonicals() adds id (row index) + text.
"""
import re
import time
import requests

from src.data import canonical_string

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "dlc-course-project/1.0 (address normalization)"}
HOUSE_RE = re.compile(r"^\s*(\d+[а-яёa-z]?)\s*[кКk/]\s*(\d+[а-яёa-z]?)\s*$", re.I)


def _overpass(query, timeout=180):
    last = None
    for ep in ENDPOINTS:
        try:
            r = requests.post(ep, data={"data": query}, headers=HEADERS,
                              timeout=timeout + 30)
            if r.status_code == 200:
                return r.json().get("elements", [])
            last = f"{ep} -> {r.status_code}"
        except Exception as ex:                       # noqa: BLE001
            last = f"{ep} -> {type(ex).__name__}: {ex}"
        time.sleep(2)
    raise RuntimeError(f"Overpass failed: {last}")


def _split_house(hn):
    m = HOUSE_RE.match(hn or "")
    if m:
        return m.group(1), m.group(2)
    return (hn or "").strip(), None


def _parse(elements, region_label, default_city=None):
    out = []
    for el in elements:
        t = el.get("tags", {})
        street, house = t.get("addr:street"), t.get("addr:housenumber")
        if not street or not house:
            continue
        c = el.get("center", {})
        lat = el.get("lat", c.get("lat"))
        lon = el.get("lon", c.get("lon"))
        if lat is None or lon is None:
            continue
        h, korp = _split_house(house)
        out.append(dict(
            region=t.get("addr:region") or region_label,
            city=t.get("addr:city") or default_city or "",
            street=street, house=h, korp=korp,
            lat=float(lat), lon=float(lon),
            osm_id=f"{el['type']}/{el['id']}"))
    return out


def fetch_area(area_name, region_label, timeout=180):
    """All addressed objects inside a named administrative area (city/region)."""
    q = (f'[out:json][timeout:{timeout}];'
         f'area["name"="{area_name}"]["boundary"="administrative"]->.a;'
         f'(node["addr:housenumber"]["addr:street"](area.a);'
         f' way["addr:housenumber"]["addr:street"](area.a););'
         f'out center tags;')
    return _parse(_overpass(q, timeout), region_label, default_city=area_name)


def fetch_bbox(bbox, region_label, city=None, timeout=180):
    """bbox = (south, west, north, east)."""
    s, w, n, e = bbox
    q = (f'[out:json][timeout:{timeout}];'
         f'(node["addr:housenumber"]["addr:street"]({s},{w},{n},{e});'
         f' way["addr:housenumber"]["addr:street"]({s},{w},{n},{e}););'
         f'out center tags;')
    return _parse(_overpass(q, timeout), region_label, default_city=city)


def to_canonicals(records):
    """Dedup + assign id (row index) + build canonical text."""
    seen, canon = set(), []
    for r in records:
        key = (r["region"], r["city"], r["street"], r["house"], r["korp"])
        if key in seen:
            continue
        seen.add(key)
        c = dict(id=len(canon), **r)
        c["text"] = canonical_string(c)
        canon.append(c)
    return canon


def load_pbf(path, region_label):
    """Full-region path (>=500k): read a Geofabrik .osm.pbf via GDAL/geopandas.

    import geopandas as gpd
    pts = gpd.read_file(path, layer="points")       # nodes with tags
    # filter rows with addr:street & addr:housenumber, take geometry.x/.y,
    # then reuse _parse-like normalization. multipolygons layer -> building
    # centroids for way/relation addresses.
    """
    raise NotImplementedError(
        "For >=500k use a Geofabrik region .pbf + geopandas.read_file, or run "
        "build_dataset.py over several fetch_area(...) regions.")
