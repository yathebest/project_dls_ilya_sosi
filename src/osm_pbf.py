"""Parse a full Geofabrik `.osm.pbf` into canonical addresses (whole-country run).

The Overpass path (src/osm.py) is fine per-city, but for **all of Russia** we
read the offline Geofabrik dump `russia-latest.osm.pbf` directly with pyosmium.
Streaming: every addressed node/way is written to a jsonl as it is read, so we
never hold tens of millions of records in RAM.

    python -m src.osm_pbf data/russia-latest.osm.pbf data/canon_ru.jsonl

Output rows match the pipeline schema (region, city, street, house, korp, lat,
lon, osm_id, text) so build_index.py / load_canon consume them unchanged.

Notes on OSM address quality: `addr:street` + `addr:housenumber` are almost
always present on addressed objects, but `addr:city` / `addr:region` frequently
are NOT (OSM leans on boundary relations instead). We keep the tags where they
exist and report coverage; region enrichment by point-in-polygon is optional
(see enrich_regions.py). Ways get their centroid from the node-location cache.
"""
import hashlib
import json
import re
import sys

import osmium

HOUSE_RE = re.compile(r"^\s*(\d+[а-яёa-z]?)\s*[кКk/]\s*(\d+[а-яёa-z]?)\s*$", re.I)


def _split_house(hn):
    m = HOUSE_RE.match(hn or "")
    if m:
        return m.group(1), m.group(2)
    return (hn or "").strip(), None


def _canonical_string(region, city, street, house, korp):
    parts = []
    if region:
        parts.append(region)
    if city:
        parts.append(f"г {city}")
    parts.append(f"ул {street}")
    parts.append(f"д {house}")
    if korp:
        parts.append(f"к {korp}")
    return ", ".join(parts)


class AddrHandler(osmium.SimpleHandler):
    """Emit one jsonl row per addressed node/way (street + housenumber + coords)."""

    def __init__(self, fout):
        super().__init__()
        self.f = fout
        self.n = 0
        self.with_city = 0
        self.with_region = 0
        self._seen = set()  # 8-byte hashes -> dedup near-identical objects

    def _emit(self, t, lat, lon):
        street = t.get("addr:street")
        house = t.get("addr:housenumber")
        if not street or not house or lat is None or lon is None:
            return
        h, korp = _split_house(house)
        city = t.get("addr:city") or t.get("addr:town") or t.get("addr:village") or ""
        region = t.get("addr:region") or t.get("addr:state") or ""
        key = f"{city}|{street}|{h}|{korp}|{round(lat, 4)}|{round(lon, 4)}"
        hk = hashlib.md5(key.encode("utf-8")).digest()[:8]
        if hk in self._seen:
            return
        self._seen.add(hk)
        row = {
            "region": region, "city": city, "street": street,
            "house": h, "korp": korp,
            "lat": round(float(lat), 7), "lon": round(float(lon), 7),
            "osm_id": None,
            "text": _canonical_string(region, city, street, h, korp),
        }
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1
        if city:
            self.with_city += 1
        if region:
            self.with_region += 1
        if self.n % 200000 == 0:
            sys.stderr.write(f"  ... {self.n:,} addresses\n")
            sys.stderr.flush()

    def node(self, n):
        t = n.tags
        if "addr:housenumber" in t and "addr:street" in t and n.location.valid():
            self._emit(t, n.location.lat, n.location.lon)

    def way(self, w):
        # Most building addresses are on (closed) building ways. We don't need a
        # true polygon — mean of the node coords is a good-enough pin. Single
        # pass, no area assembler (locations=True fills w.nodes[*].location).
        t = w.tags
        if "addr:housenumber" not in t or "addr:street" not in t:
            return
        slat = slon = 0.0
        cnt = 0
        for nd in w.nodes:
            if nd.location.valid():
                slat += nd.location.lat
                slon += nd.location.lon
                cnt += 1
        if cnt:
            self._emit(t, slat / cnt, slon / cnt)


def parse_pbf(pbf_path, out_path):
    with open(out_path, "w", encoding="utf-8") as fout:
        h = AddrHandler(fout)
        # locations=True resolves way/area node coords from a disk-backed cache
        # (safe for a country-sized file); areas need the area assembler.
        h.apply_file(pbf_path, locations=True,
                     idx="sparse_file_array," + out_path + ".nodecache")
    return h


def main():
    if len(sys.argv) < 3:
        print("usage: python -m src.osm_pbf <russia-latest.osm.pbf> <out.jsonl>")
        sys.exit(1)
    pbf, out = sys.argv[1], sys.argv[2]
    print(f"parsing {pbf} -> {out}")
    h = parse_pbf(pbf, out)
    print(f"done: {h.n:,} addresses  "
          f"(with city: {h.with_city:,}, with region: {h.with_region:,})")


if __name__ == "__main__":
    main()
