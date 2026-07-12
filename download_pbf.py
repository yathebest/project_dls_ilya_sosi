"""Resumable download of the Geofabrik Russia pbf (survives dropped connections).

Uses HTTP Range to continue from whatever is already on disk, and retries in a
loop so a broken connection just resumes instead of restarting the 4 GB.

    python download_pbf.py
"""
import os
import sys
import time

import requests

URL = "https://download.geofabrik.de/russia-latest.osm.pbf"
OUT = "data/russia-latest.osm.pbf"
HEADERS = {"User-Agent": "dlc-course-project/1.0 (address normalization)"}


def total_size():
    r = requests.head(URL, headers=HEADERS, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers.get("Content-Length", 0))


def main():
    os.makedirs("data", exist_ok=True)
    total = total_size()
    print(f"remote total: {total/1e9:.2f} GB", flush=True)
    attempt = 0
    while True:
        have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
        if total and have >= total:
            print(f"complete: {have/1e9:.2f} GB", flush=True)
            return
        attempt += 1
        hdr = dict(HEADERS)
        if have:
            hdr["Range"] = f"bytes={have}-"
        print(f"[attempt {attempt}] resume from {have/1e9:.2f} GB "
              f"({have/total*100:.1f}%)", flush=True)
        try:
            with requests.get(URL, headers=hdr, stream=True, timeout=120) as r:
                if r.status_code not in (200, 206):
                    print(f"  unexpected status {r.status_code}", flush=True)
                    time.sleep(5)
                    continue
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0  # server ignored Range -> restart clean
                with open(OUT, mode) as f:
                    for ch in r.iter_content(1 << 20):
                        if ch:
                            f.write(ch)
                            have += len(ch)
        except Exception as ex:                          # noqa: BLE001
            print(f"  dropped at {have/1e9:.2f} GB: {type(ex).__name__}: {ex}",
                  flush=True)
            time.sleep(5)
            continue


if __name__ == "__main__":
    main()
