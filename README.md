# Address Normalization via Vector Search (FIAS/GAR)

Course project — Deep Learning for Search 2026, Innopolis.

**Task.** Dirty free-text Russian address → canonical GAR/FIAS record (FIAS id +
structure). Framed as **retrieval / matching** (not LLM rewriting). The retriever
design and evaluation follow **AddrLLM** (JD Logistics, arXiv 2411.13584); we
scale down to the vector-search core that this course grades.

## Project layout

```
src/            core library — importable modules (no CLI). Run scripts from the repo root.
  fias.py         GAR/FIAS XML → canonical records + OBJECTGUID (streaming, reads gar_xml.zip directly)
  matching.py     address-key normalization for the FIAS↔OSM coordinate join
  data.py         canonical base: synthetic generator + parse_gar entry point
  noise.py        dirty-query generator (AddrLLM error taxonomy)
  vectorizer.py   char-n-gram / sentence-transformers encoders
  index.py        numpy Flat + FAISS Flat/HNSW/IVF-PQ
  lexical.py      word-level BM25    ·    hybrid.py   RRF fusion + rerank
  metrics.py      Recall@k, MRR, nDCG, per-category robustness, geo Acc@radius
  osm.py osm_pbf.py   OSM address loaders (Overpass / Geofabrik .pbf)
pipeline/       FIAS production pipeline (the deployed system)
  build_dataset_fias.py   GAR zip → data/canon_fias.jsonl (region/city/street/house + GUID + postal)
  train_encoder.py        fine-tune the e5 encoder → models/addr-e5-ft
  build_index_pq.py       stream-encode → FAISS IVF-PQ + SQLite metadata
  join_coords.py patch_coords.py   attach OSM coordinates to the base / a built index (no re-encode)
  append_objects.py       add street/city OBJECTS to an existing index
  eval_index.py           ablation / evaluation of the live index
  download_pbf.py         fetch a Geofabrik .osm.pbf
experiments/    earlier course iterations (synthetic / small OSM, comparative studies)
  run_baseline.py (Iter 0) · run_experiments.py (Iter 1: Flat/HNSW/IVF-PQ)
  run_shrink.py (quantization) · run_hybrid.py (Iter 2: dense+BM25+rerank)
  build_dataset.py (OSM Overpass) · build_index.py (HNSW) · make_plots.py (figures)
service/        FastAPI demo + Yandex-map UI (app.py, index.html)
tests/          offline unit tests (no downloads)
results/        eval metric outputs (json)   ·   figures/   plots   ·   docs/   presentation
data/ models/ index_ru/    build artifacts (gitignored)
```

## Quick start

```bash
# 1. offline baseline — numpy only, deterministic synthetic data, no downloads
pip install numpy
python experiments/run_baseline.py      # Iter 0: encode → index → dirty queries → metrics
python experiments/run_experiments.py   # Iter 1: Flat vs HNSW vs IVF-PQ (Pareto)

# 2. live FIAS demo (needs models/addr-e5-ft + index_ru/ — see the FIAS section below)
python -m uvicorn service.app:app       # http://127.0.0.1:8000
```

**Noise categories** (AddrLLM Table 5, adapted to RU): misspelling, abbreviation,
missing_region, irrelevant_words, reorder, transliteration.

## Web demo (search UI + Yandex map)

Type a dirty address → canonical candidates pinned on a Yandex map (coordinates
come from the OSM base). Uses the fine-tuned encoder + a prebuilt HNSW index.

```bash
python experiments/build_index.py --limit 0                 # embed the base once (GPU); index/ (gitignored)
set YANDEX_MAPS_API_KEY=<your Yandex JS API key> # same env style as ortouz (VITE_YANDEX_MAPS_API_KEY)
pip install fastapi "uvicorn[standard]"
uvicorn service.app:app                          # open http://127.0.0.1:8000
```

`service/app.py` serves `/search?q=` (fine-tuned model → HNSW → candidates with
lat/lon) and `/` (the map page; the key is injected from the env var, never
committed). Yandex Maps loading is ported from ortouz (`api-maps.yandex.ru/2.1`
→ `ymaps.ready` → `Map` + `Placemark` + `setBounds`). Without a key the search
still works, results show as a list. Without `index/` it falls back to a small
synthetic char-n-gram engine (no coordinates).

## Results (real OSM data)

Base: **623,116 addresses across 14 Russian regions** (OpenStreetMap, with
coordinates) — meets the ≥500k requirement with regional variety. Detailed
quality runs below use CPU-feasible 8k/80k subsets of this base.


- **Iter 0 baseline** (char-n-gram, Flat): R@1 0.82, MRR@10 0.83, **Acc@500m 0.83**,
  p50 31 ms. Robust on typos/abbrev/junk, but **transliteration 0.02** (char-n-gram
  can't bridge Cyrillic↔Latin).
- **Iter 1 index** (80k): Flat 0.82 / 1310 MB / 31 ms → HNSW 0.74 / **0.28 ms**
  (×110 faster) → IVF-PQ 0.54 / **10 MB** (×130 smaller); add exact re-rank to
  recover IVF-PQ recall (L9).
- **Iter 1 shrink** (method): int8 ≈free ×4 · PCA-256 ×16 · binary ×32.
- **Iter 3a neural encoder** (multilingual-e5-small off-the-shelf, 8k, HNSW):
  R@1 **0.86**, R@5 **0.93**, Acc@500m 0.92, **transliteration 0.04 → 0.67**, 384-dim.
- **Iter 3b fine-tuned encoder** (contrastive on dirty↔canonical, e5-small, GPU
  ~2 min, evaluated on **unseen** addresses): R@1 **0.95**, R@5 **0.99**, nDCG
  **0.97**, Acc@500m **0.98**, **transliteration 0.88** — beats off-the-shelf in
  every category. **Headline result** (`python pipeline/train_encoder.py`).
- **Hybrid caveat (real finding):** neural dense alone (R@1 **0.88**) *beats* naive
  neural+BM25 RRF (0.83) here — when one retriever dominates, fusing a weaker one
  adds noise (L8); and the *lexical* reranker hurts transliteration (0.62→0.10),
  so cross-lingual needs a **neural** cross-encoder, not lexical. Apply hybrid/rerank
  judiciously, not by default.

## Real data (OpenStreetMap)

`build_dataset.py` fetches real Russian addresses **with coordinates** from the
Overpass API and writes `data/canon.jsonl`; every run script takes `--dataset`:

```bash
python experiments/build_dataset.py                                  # default city list
python experiments/build_dataset.py --cities "Казань:Республика Татарстан,Уфа:Республика Башкортостан"
python experiments/run_baseline.py --dataset data/canon.jsonl        # Acc@300m/500m too (coords)
python experiments/run_hybrid.py   --dataset data/canon.jsonl
```

Coordinates come for free → they feed the geocoding-aware head (Iter 3). For the
full **≥500k**, add more/larger regions, or read a Geofabrik `.osm.pbf` via
`src.osm.load_pbf` (GDAL/geopandas). For the official FIAS id instead of OSM,
implement `data.parse_gar()` (gar_xml.zip from fias.nalog.ru).

## FIAS/GAR as the canonical source (official state registry)

The headline framing of the task — "dirty text → **canonical record**" — is strongest
when the canonical record is an **official** one. OSM is convenient and carries
coordinates, but it is crowd-sourced and *structurally incomplete*: on our 11.13M
national base only **41.7%** of rows have a city and **2.3%** a region (OSM leans on
boundary relations instead of `addr:*` tags). That makes many OSM rows ambiguous
country-wide (just street + house).

**GAR** (ГАР, ФНС — the successor to ФИАС, `fias.nalog.ru`) fixes exactly this: every
object carries the full hierarchy **region → district → city → street → house** and an
official **`OBJECTGUID`**. That GUID *is* the canonical target (as in AddrLLM), so a
match becomes "normalized into state-registry object `<guid>`" — a real plus for the
grade.

**The one catch:** GAR has **no coordinates** for houses, but the map needs lat/lon.
So we do a **hybrid**: canon + GUID come from FIAS, coordinates are joined from the
existing OSM base by a normalized `(city, street, house)` key
([`src/matching.py`](src/matching.py)). Rows without an OSM match stay searchable, just
without a map pin (graceful degradation) — high match-rate in cities, low in villages.

```bash
# 1. read the base straight from gar_xml.zip — NO unzip needed (full unzip = 424 GB;
#    the parser streams the 3 needed files per region out of the archive)
python pipeline/build_dataset_fias.py --gar D:/gar_xml.zip --regions 46,16,77,78,66 --postal
#    -> data/canon_fias.jsonl  (region+city on ~every row, + fias_guid + postal index)
python pipeline/join_coords.py --fias data/canon_fias.jsonl --osm data/canon_ru.jsonl \
       --out data/canon_fias_geo.jsonl        # attach lat/lon from OSM
# 2. index + serve — unchanged infrastructure
python pipeline/build_index_pq.py --dataset data/canon_fias_geo.jsonl --out index_ru
uvicorn service.app:app                        # /search returns fias_guid + postal
```

Record schema (jsonl): `{region, city, street, house, korp, region_code, postal,
fias_guid, lat, lon, text}`. The canonical `text` uses real GAR types —
`влд./д./двлд./стр./к.` for houses, `обл./г./ул./пгт/с.` for objects, e.g.
`305000, Курская обл., г. Курск, ул. Ленина, д. 5, к. 1`. `--postal` joins the
index from `AS_HOUSES_PARAMS` (a large per-region file, so it is opt-in); the index
is a strong disambiguator and is often present in dirty user queries.

`src/fias.py` streams the GAR XML with `iterparse` (lxml, stdlib fallback) so peak RAM
is ~one region, not the whole country: it reads `AS_ADDR_OBJ` (objects: NAME/TYPENAME/
LEVEL), reconstructs ancestry from `AS_ADM_HIERARCHY` (`PATH`), and emits one canonical
row per **actual** house in `AS_HOUSES`. Only `ISACTUAL=1 AND ISACTIVE=1` records are
kept. Offline sanity check (no download, no lxml required): `python tests/test_fias.py`.

**FIAS vs OSM (why both):**

| | OSM | FIAS/GAR |
|---|---|---|
| region on row | 2.3% | ~100% |
| city on row | 41.7% | ~100% |
| official id | — | `OBJECTGUID` |
| coordinates | ✅ | ❌ (joined from OSM) |

## Scaling up (neural)
2. **Neural encoder (Iter 2/3):**
   `python experiments/run_baseline.py --embedder st --model deepvk/USER-bge-m3 --index hnsw`
3. **Hybrid + rerank (Iter 2):** fuse dense + char-n-gram via RRF, then a
   cross-encoder on the top-k.
4. **Fine-tune (Iter 3, headline):** contrastive on (dirty↔canonical) pairs with
   hard negatives + auxiliary **geocoding head** (predict lat/lon → embedding
   distance ∝ geographic distance, AddrLLM Fig.4); distill to a tiny model.
   Needs `<address, lat/lon>` (OpenAddresses/OSM/Nominatim); if unavailable, drop
   the geocoding head and keep contrastive.
5. **Service:** Qdrant + FastAPI + mini UI.

Reproducibility: fixed seed `20260605`.
