"""Patch lat/lon into a prebuilt index's meta.sqlite from an OSM base — WITHOUT
re-encoding. Coordinates are metadata (not part of the embedding), so we can
attach them to an existing IVF-PQ index for free.

    python patch_coords.py --fias data/canon_fias_demo.jsonl \
        --osm data/canon_osm_kazan.jsonl --db index_ru/meta.sqlite

rowid in meta.sqlite == the 0-based index of the (non-empty) row in the SAME
canon file that build_index_pq.py indexed, so we stream that file, match each row
to the OSM base by norm_key, and UPDATE lat/lon by rowid. Only matched rows change.
"""
import argparse
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from src.matching import norm_key


def load_osm(path):
    coords = {}
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
            if k[1]:
                coords.setdefault(k, (lat, lon))
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fias", required=True, help="the canon file that was indexed")
    ap.add_argument("--osm", required=True, help="OSM base with coordinates")
    ap.add_argument("--db", default="index_ru/meta.sqlite")
    args = ap.parse_args()

    coords = load_osm(args.osm)
    print(f"OSM keys with coords: {len(coords):,}")

    updates = []
    total = 0
    with open(args.fias, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # federal cities (Москва/СПб) carry the city name at the region level
            # with city=None, so fall back to region for the join key.
            city = r.get("city") or r.get("region") or ""
            k = norm_key(city, r.get("street") or "",
                         r.get("house") or "", r.get("korp"))
            hit = coords.get(k)
            if hit:
                updates.append((hit[0], hit[1], total))
            total += 1

    db = sqlite3.connect(args.db)
    db.executemany("UPDATE addr SET lat=?, lon=? WHERE rowid=?", updates)
    db.commit()
    got = db.execute("SELECT COUNT(*) FROM addr WHERE lat IS NOT NULL").fetchone()[0]
    db.close()
    pct = f"{100.0 * len(updates) / total:.1f}%" if total else "0%"
    print(f"rows scanned: {total:,}   matched now: {len(updates):,} ({pct})")
    print(f"total rows with coords in db: {got:,}")


if __name__ == "__main__":
    main()
