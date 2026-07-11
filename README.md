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

## Current demo numbers (synthetic, N=5000, char-n-gram)

**Iter 0** (Flat): Recall@1 ≈ 0.80, MRR@10 ≈ 0.82, p50 ≈ 2 ms. Weak on
**transliteration (0.14)** and **missing_region (0.68)** → motivates Iter 2/3.

**Iter 1 index:** Flat 0.80 / 82 MB / 2 ms → HNSW 0.75 / 0.2 ms → IVF-PQ 0.63 /
**5 MB** / 0.16 ms (add exact re-rank to recover recall — L9).

**Iter 1 shrink:** int8 0.80 / **20 MB** (≈free ×4) · PCA-256 0.78 / **5 MB** (×16)
· PCA-64 0.71 / **1.3 MB** (×63) · binary 0.77 / **2.6 MB** (×32).

**Iter 2 hybrid:** dense 0.798 → bm25 0.872 → hybrid 0.847 → **hybrid+rerank
0.903** (R@10 = 1.000, MRR@10 = 0.940). Best in every noise category; transliteration
0.14 → 0.46, missing_region 0.68 → 0.98.

## Scaling up (production run)

1. **Real data:** implement `data.parse_gar()` — download `gar_xml.zip` from
   fias.nalog.ru (or a preconverted Kaggle/HF dump), keep ≥500k objects across
   2–3 contrasting regions, use OBJECTGUID as the FIAS id.
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
