# Z-AI Platform

A local-first **RAG (Retrieval-Augmented Generation) platform** for scientific documents: upload PDFs, ask questions in natural language (Greek or English), and get evidence-based answers with page-level citations — powered by a fully local retrieval stack and Gemini for generation.

Built as a diploma thesis project focusing on **cross-lingual retrieval quality** and **measurable, reproducible evaluation**.

## Demo

![Z-AI Platform — anti-hallucination gate + bilingual Q&A](docs/demo.gif)

> Upload a PDF, ask in Greek or English, and get grounded answers with page-level citations — plus a clear *"not found"* when the answer isn't in the documents.

## Features

- **Hybrid retrieval** — dense semantic search (BAAI/bge-m3, 1024-dim, cosine) fused with lexical BM25 via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 reorders fused candidates for precision
- **Anti-hallucination relevance gate** — if even the best chunk scores below a measured threshold (0.15 sigmoid), the system answers "not found" instead of feeding irrelevant context to the LLM
- **Cross-lingual QA** — Greek questions are translated for retrieval over English papers (translate-then-retrieve); answers come back in the user's language
- **Greek-aware BM25** — accent-stripping tokenizer so unaccented queries still match
- **Built-in evaluation framework** — retrieval metrics (MRR, nDCG, keyword coverage) plus LLM-as-judge answer scoring (accuracy, completeness, relevance, faithfulness)
- **Multi-user** — JWT auth, per-user document ownership, public/private sharing, login rate limiting
- **Observability** — request IDs, per-phase latency (retrieval vs generation), token usage logging

## Architecture

```
┌────────────┐  HTTP   ┌─────────────────────────────────┐
│  Streamlit │ ──────► │            FastAPI               │
│  frontend  │         │                                  │
│   :8501    │         │  /chat ──► RAG pipeline          │
└────────────┘         │  /upload ─► background ingest    │
                       └──────┬──────────┬───────────┬────┘
                              │          │           │
                       ┌──────▼───┐ ┌────▼─────┐ ┌───▼────────┐
                       │ ChromaDB │ │ Postgres │ │ Gemini 2.5 │
                       │ (chunks, │ │ (users,  │ │   Flash    │
                       │ vectors) │ │  chats)  │ │ (gen+judge)│
                       └──────────┘ └──────────┘ └────────────┘
```

### RAG pipeline (per question)

1. **Query optimization** — Greek questions translated to English search queries (cached, best-effort)
2. **Dense search** — bge-m3 embeddings, top-30 by cosine similarity
3. **Sparse search** — BM25 with Greek-aware tokenization, top-30
4. **Fusion** — Reciprocal Rank Fusion (k=60), keep top-15
5. **Rerank** — bge-reranker-v2-m3 cross-encoder scores each (query, chunk) pair
6. **Relevance gate** — best score < 0.15 → "no relevant documents" (no hallucination)
7. **Generation** — top-5 chunks with source/page metadata sent to Gemini 2.5 Flash, streamed back

## Evaluation results

Measured on **`golden_set_20`** — 20 bilingual (EN+EL) questions over two Berkeley cloud-computing papers, including 2 deliberate out-of-corpus questions that probe the anti-hallucination gate. Config: `bge-m3 + BM25 + RRF + bge-reranker-v2-m3` (rerank-15), `chunk=1500`, page-level (parent-document) expansion, relevance gate `0.15`.

| Metric | Score |
|---|---|
| Retrieval MRR (in-corpus) | **0.846** |
| Keyword coverage (in-corpus) | **97.2%** |
| Answer accuracy | **5.0 / 5** |
| Answer completeness | **5.0 / 5** |
| Answer relevance | **5.0 / 5** |
| Faithfulness (no hallucinations) | **5.0 / 5** |

![Evaluation dashboard](docs/dashboard.png)

The 2 out-of-corpus questions correctly return *"not found in the documents"* — the relevance gate fires (best reranker score ≈ 0.00 < 0.15), so the model does not hallucinate.

Chunk-size experiment (retrieval metrics):

| chunk size | chunks | MRR | nDCG | keyword coverage |
|---|---|---|---|---|
| 500 | 1082 | 0.740 | 0.750 | 85.7% |
| 1000 | 564 | 0.724 | 0.750 | 90.5% |
| **1500** | **386** | **0.803** | **0.821** | **95.2%** |

Reranker score separation (relevance gate calibration): relevant questions score **0.92–1.00**, irrelevant **0.000–0.004** → threshold `0.15` blocks hallucination-inducing context with a wide margin (lowered from 0.30 so translated Greek queries are not dropped).

Full methodology and per-question results: [`backend/evaluation/RESULTS.md`](backend/evaluation/RESULTS.md)

## Quickstart

Requirements: Docker Desktop (WSL2 backend on Windows), a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/kostas221/local-rag-assistant.git
cd local-rag-assistant
cp .env.example .env        # fill in your keys
docker compose up -d --build
```

First start downloads the embedding + reranker models (~4.5 GB, cached in a volume afterwards).

- UI: http://localhost:8501 — register, upload a PDF, wait for "ready", ask away
- API docs: http://localhost:8000/docs

## Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key (generation, translation, LLM-judge) |
| `POSTGRES_PASSWORD` | Postgres password (compose wires the DSN) |
| `SECRET_KEY` | JWT signing key — generate with `openssl rand -hex 32` |

Note: the Gemini **free tier allows ~20 requests/day** — enough for chatting, but full evaluation runs need a paid key or careful budgeting.

## Tests & evaluation

```bash
# unit tests (validators, security)
docker compose run --rm backend python -m pytest tests/ -v

# retrieval calibration (no API quota used)
docker compose exec backend python measure_reranker.py

# chunk-size experiment (retrieval metrics)
docker compose exec backend python chunk_experiment.py

# full answer-quality eval (uses Gemini quota: ~2-3 calls/question)
docker compose exec backend python faithfulness_eval.py
```

## Project structure

```
backend/
  ai_core.py            # RAG pipeline: ingest, hybrid search, rerank, gate, generation
  main.py               # FastAPI app: auth, documents, conversations, chat streaming
  models.py, schemas.py # SQLAlchemy models, Pydantic validators
  security.py           # bcrypt + JWT
  evaluation/           # golden sets + LLM-judge eval engine
  tests/                # pytest suite
frontend/
  app_ui.py             # Streamlit chat UI
docker-compose.yml      # postgres + backend + frontend, named volumes
```

## Tech stack

FastAPI · Streamlit · ChromaDB · PostgreSQL · sentence-transformers (bge-m3, bge-reranker-v2-m3) · rank-bm25 · Gemini 2.5 Flash · Docker Compose
