# Address Normalization via Vector Search (FIAS/GAR)

Course project — Deep Learning for Search 2026, Innopolis.

**Task.** Dirty free-text Russian address → canonical GAR/FIAS record (FIAS id +
structure). Framed as **retrieval / matching** (not LLM rewriting). The retriever
design and evaluation follow **AddrLLM** (JD Logistics, arXiv 2411.13584); we
scale down to the vector-search core that this course grades.

## Runs offline right now

```bash
pip install numpy            # baseline needs only numpy
python run_baseline.py       # Iteration 0: encode → index → dirty queries → metrics
python run_experiments.py    # Iteration 1: Flat vs HNSW vs IVF-PQ (Pareto)
```

No downloads: uses a deterministic **synthetic** RU address base + a char-n-gram
encoder, so the whole pipeline works end-to-end. Swap in real GAR + a neural
encoder for the full run (below).

## What's here

| File | Role | Element (assignment) |
|---|---|---|
| `src/data.py` | canonical base (synthetic; `parse_gar` stub for real run) | data |
| `src/noise.py` | synthetic dirty-query generator, AddrLLM error taxonomy | validation set |
| `src/vectorizer.py` | char-n-gram (default) / sentence-transformers | 1. model |
| `src/index.py` | numpy Flat + FAISS Flat/HNSW/IVF-PQ | 2. DB + 3. index |
| `src/lexical.py` | word-level BM25 (sparse) | 4. search |
| `src/hybrid.py` | RRF fusion + precise / cross-encoder rerank | 4. search |
| `src/metrics.py` | Recall@k, MRR, nDCG, per-category robustness (by hand) | evaluation |
| `run_baseline.py` | **Iter 0** — end-to-end baseline | 4. search |
| `run_experiments.py` | **Iter 1** — Flat vs HNSW vs IVF-PQ | comparative study |
| `run_shrink.py` | **Iter 1** — quantization (int8/binary) + PCA | shrink/quantize |
| `run_hybrid.py` | **Iter 2** — dense+bm25 (RRF) + rerank | comparative study |
| `train_encoder.py` | **Iter 3** — contrastive + geocoding fine-tune, distill | 1. model |
| `make_plots.py` | figures for the presentation | comparative study |
| `service/app.py` + `index.html` | FastAPI demo + mini UI | deployment |

**Noise categories** (AddrLLM Table 5, adapted to RU): misspelling, abbreviation,
missing_region, irrelevant_words, reorder, transliteration.

Run everything:
```bash
python run_baseline.py && python run_experiments.py && python run_shrink.py
python run_hybrid.py  && python make_plots.py        # -> figures/*.png
uvicorn service.app:app                              # demo at http://127.0.0.1:8000
```

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
  every category. **Headline result** (`python train_encoder.py`).
- **Hybrid caveat (real finding):** neural dense alone (R@1 **0.88**) *beats* naive
  neural+BM25 RRF (0.83) here — when one retriever dominates, fusing a weaker one
  adds noise (L8); and the *lexical* reranker hurts transliteration (0.62→0.10),
  so cross-lingual needs a **neural** cross-encoder, not lexical. Apply hybrid/rerank
  judiciously, not by default.

## Real data (OpenStreetMap)

`build_dataset.py` fetches real Russian addresses **with coordinates** from the
Overpass API and writes `data/canon.jsonl`; every run script takes `--dataset`:

```bash
python build_dataset.py                                  # default city list
python build_dataset.py --cities "Казань:Республика Татарстан,Уфа:Республика Башкортостан"
python run_baseline.py --dataset data/canon.jsonl        # Acc@300m/500m too (coords)
python run_hybrid.py   --dataset data/canon.jsonl
```

Coordinates come for free → they feed the geocoding-aware head (Iter 3). For the
full **≥500k**, add more/larger regions, or read a Geofabrik `.osm.pbf` via
`src.osm.load_pbf` (GDAL/geopandas). For the official FIAS id instead of OSM,
implement `data.parse_gar()` (gar_xml.zip from fias.nalog.ru).

## Scaling up (neural)
2. **Neural encoder (Iter 2/3):**
   `python run_baseline.py --embedder st --model deepvk/USER-bge-m3 --index hnsw`
3. **Hybrid + rerank (Iter 2):** fuse dense + char-n-gram via RRF, then a
   cross-encoder on the top-k.
4. **Fine-tune (Iter 3, headline):** contrastive on (dirty↔canonical) pairs with
   hard negatives + auxiliary **geocoding head** (predict lat/lon → embedding
   distance ∝ geographic distance, AddrLLM Fig.4); distill to a tiny model.
   Needs `<address, lat/lon>` (OpenAddresses/OSM/Nominatim); if unavailable, drop
   the geocoding head and keep contrastive.
5. **Service:** Qdrant + FastAPI + mini UI.

Reproducibility: fixed seed `20260605`.
