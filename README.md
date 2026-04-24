# 🔍 The Explorer — Hybrid Search Engine

A full-stack semantic search engine built with **FastAPI**, **BM25**, **Neural Embeddings**, and **FAISS**.

## Stack

| Layer | Technology |
|-------|-----------|
| API Server | FastAPI + Uvicorn |
| Lexical Retrieval | BM25Okapi (rank_bm25) |
| Neural Embeddings | all-MiniLM-L6-v2 (Sentence-Transformers) |
| Vector Index | FAISS (IndexFlatIP) |
| Document Store | SQLite + FTS5 |
| Query Expansion | NLTK WordNet |
| Crawler | aiohttp + BeautifulSoup |
| Frontend | Vanilla HTML/CSS/JS |

## How It Works

1. **Query Expansion** — WordNet synonyms expand the query for better recall
2. **BM25 Scoring** — Lexical retrieval on tokenised corpus
3. **Neural Scoring** — Dense vector similarity via FAISS
4. **RRF Fusion** — Reciprocal Rank Fusion blends both rankings into a final list

## Setup

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu \
            beautifulsoup4 requests rank_bm25 numpy aiohttp nltk
```

## Run

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

Then open `frontend.html` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/search?q=...` | Hybrid search |
| POST | `/index` | Index a document |
| POST | `/index/wikipedia` | Index Wikipedia topics |
| POST | `/crawl` | Web crawl from seed URLs |
| GET | `/stats` | Engine statistics |
| GET | `/suggest?q=...` | Autocomplete suggestions |
| GET | `/health` | Health check |
| DELETE | `/index` | Clear index |
