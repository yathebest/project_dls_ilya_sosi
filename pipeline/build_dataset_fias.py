"""Build a canonical address base from the official GAR / FIAS XML.

    # download + unzip gar_xml.zip (or a per-region archive) from fias.nalog.ru
    python build_dataset_fias.py --gar path/to/gar_xml            # all regions in the tree
    python build_dataset_fias.py --gar path/to/gar_xml --regions 46,16,77,78,66
    python build_dataset_fias.py --gar path/to/gar_xml --limit 200000

Unlike the OSM base, every GAR house carries the full hierarchy
(region -> city -> street -> house) and an official OBJECTGUID, so the output has
region+city on (almost) every row. Coordinates are added afterwards by
join_coords.py (GAR has no lat/lon).

Output: data/canon_fias.jsonl in the pipeline schema
    {region, city, street, house, korp, lat:null, lon:null, fias_guid, text}
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.fias import iter_canon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gar", required=True,
                    help="gar_xml.zip, an unzipped GAR root (folder-per-region), "
                         "or a single region folder")
    ap.add_argument("--regions", default=None,
                    help='comma-separated region codes to keep, e.g. "46,16,77" (default = all)')
    ap.add_argument("--out", default="data/canon_fias.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after N houses (0 = all)")
    ap.add_argument("--postal", action="store_true",
                    help="join the postal index from AS_HOUSES_PARAMS (large file, slower)")
    ap.add_argument("--objects-only", action="store_true",
                    help="emit only address OBJECTS (streets/cities/regions), not houses")
    args = ap.parse_args()

    region_codes = set(c.strip() for c in args.regions.split(",")) if args.regions else None
    limit = args.limit or None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    seen = set()
    total = with_region = with_city = with_postal = 0
    regions = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for c in iter_canon(args.gar, region_codes, limit, want_postal=args.postal,
                            include_houses=not args.objects_only,
                            include_objects=args.objects_only):
            key = (c["region"], c["city"], c["street"], c["house"], c["korp"])
            if key in seen:
                continue
            seen.add(key)
            c["id"] = total
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            total += 1
            if c["region"]:
                with_region += 1
                regions[c["region"]] = regions.get(c["region"], 0) + 1
            if c["city"]:
                with_city += 1
            if c.get("postal"):
                with_postal += 1
            if total % 100000 == 0:
                print(f"  ... {total} houses", flush=True)

    def pct(x):
        return f"{100.0 * x / total:.1f}%" if total else "0%"

    print(f"\nunique canonical houses: {total}")
    print(f"  with region: {with_region} ({pct(with_region)})   "
          f"with city: {with_city} ({pct(with_city)})"
          + (f"   with postal: {with_postal} ({pct(with_postal)})" if args.postal else ""))
    print("by region:", dict(sorted(regions.items(), key=lambda kv: -kv[1])))
    print(f"saved -> {args.out}")
    if total:
        with open(args.out, encoding="utf-8") as f:
            first = json.loads(f.readline())
        print("sample:", first["text"], "| guid:", first["fias_guid"])


if __name__ == "__main__":
    main()
