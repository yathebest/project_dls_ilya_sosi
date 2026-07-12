"""Build a real canonical address base from OpenStreetMap -> data/canon.jsonl.

    python build_dataset.py                       # default city list
    python build_dataset.py --cities "Казань:Республика Татарстан,Уфа:Республика Башкортостан"
    python build_dataset.py --target 500000       # keep fetching until >= target

Each city is an admin area; addresses come with coordinates. For >=500k, add
more/larger regions (or a Geofabrik .pbf via src.osm.load_pbf).
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.osm import fetch_bbox, fetch_area, to_canonicals
from src.data import save_canon

# (city, region, bbox=(south, west, north, east)) — bbox is reliable; area-by-name
# often fails on Overpass. Add more/larger boxes to reach >=500k.
CITY_BBOX = [
    ("Иннополис", "Республика Татарстан", (55.73, 48.72, 55.77, 48.80)),
    ("Казань", "Республика Татарстан", (55.60, 48.90, 55.90, 49.35)),
    ("Набережные Челны", "Республика Татарстан", (55.66, 52.22, 55.80, 52.48)),
    ("Зеленодольск", "Республика Татарстан", (55.82, 48.45, 55.90, 48.62)),
]


def parse_cities(s):
    out = []
    for item in s.split(","):
        name, _, region = item.partition(":")
        out.append((name.strip(), region.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default=None,
                    help='"City:Region,City:Region" (default = built-in list)')
    ap.add_argument("--out", default="data/canon.jsonl")
    ap.add_argument("--target", type=int, default=0,
                    help="stop once >= this many unique addresses (0 = all cities)")
    ap.add_argument("--sleep", type=float, default=3.0)
    args = ap.parse_args()

    # --cities uses area-by-name; default uses the reliable bbox list
    if args.cities:
        jobs = [(n, r, None) for n, r in parse_cities(args.cities)]
    else:
        jobs = CITY_BBOX
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    records = []
    for name, region, *bbox in jobs:
        try:
            if bbox and bbox[0]:
                recs = fetch_bbox(bbox[0], region, city=name)
            else:
                recs = fetch_area(name, region)
            records.extend(recs)
            print(f"  {name} ({region}): +{len(recs)}  total={len(records)}", flush=True)
        except Exception as ex:                       # noqa: BLE001
            print(f"  {name}: FAILED {type(ex).__name__}: {ex}", flush=True)
        if args.target and len(records) >= args.target:
            break
        time.sleep(args.sleep)

    canon = to_canonicals(records)
    save_canon(canon, args.out)

    # coverage summary
    with_coords = sum(1 for c in canon if c.get("lat"))
    regions = {}
    for c in canon:
        regions[c["region"]] = regions.get(c["region"], 0) + 1
    print(f"\nunique canonical addresses: {len(canon)}  (with coords: {with_coords})")
    print("by region:", regions)
    print(f"saved -> {args.out}")
    if canon:
        print("sample:", canon[0]["text"], "|", canon[0]["lat"], canon[0]["lon"])


if __name__ == "__main__":
    main()
