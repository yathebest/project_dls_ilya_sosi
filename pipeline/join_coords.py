"""Attach coordinates to the FIAS canon by joining to the OSM base.

GAR has the canonical hierarchy + official GUID but no lat/lon; the OSM base
(data/canon_ru.jsonl, built by src/osm_pbf.py) has coordinates. We match on a
normalized (city, street, house) key and copy lat/lon onto the FIAS rows.

    python join_coords.py --fias data/canon_fias.jsonl \
        --osm data/canon_ru.jsonl --out data/canon_fias_geo.jsonl

Rows without an OSM match keep lat/lon = null: they are still searchable, just
not pinned on the map (graceful degradation). Match rate is high in big cities,
low in villages -- expected.
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.matching import norm_key


def load_osm_coords(path):
    """norm_key -> (lat, lon) from the OSM base (streaming, keep only coords)."""
    coords = {}
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lat, lon = r.get("lat"), r.get("lon")
            if lat is None or lon is None:
                continue
            k = norm_key(r.get("city", ""), r.get("street", ""),
                         r.get("house", ""), r.get("korp"))
            if not k[1]:                       # need at least a street
                continue
            coords.setdefault(k, (lat, lon))   # first wins; dedup collisions
            n += 1
    print(f"OSM base: {n} coordinated rows -> {len(coords)} unique keys")
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fias", default="data/canon_fias.jsonl")
    ap.add_argument("--osm", default="data/canon_ru.jsonl")
    ap.add_argument("--out", default="data/canon_fias_geo.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.osm):
        sys.exit(f"OSM base not found: {args.osm}\n"
                 f"  build it first (src/osm_pbf.py) or copy data/canon_ru.jsonl "
                 f"from the main machine.")

    coords = load_osm_coords(args.osm)

    total = matched = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.fias, encoding="utf-8") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            # federal cities (Москва/СПб) keep the city name at the region level,
            # so fall back to region when city is empty.
            city = c.get("city") or c.get("region") or ""
            k = norm_key(city, c.get("street", ""),
                         c.get("house", ""), c.get("korp"))
            hit = coords.get(k)
            if hit:
                c["lat"], c["lon"] = hit
                matched += 1
            total += 1
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    pct = f"{100.0 * matched / total:.1f}%" if total else "0%"
    print(f"joined {total} FIAS rows; with coordinates: {matched} ({pct})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
