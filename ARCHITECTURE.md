# Architecture & Developer Guide

Technical reference for the **Z-AI Platform** codebase — architecture, development workflow, and the non-obvious constraints behind the key design decisions.

## What this is

**Z-AI Platform** — a local-first RAG platform for scientific PDFs (diploma thesis). Upload PDFs, ask questions in Greek or English, get evidence-based answers with page-level citations. Retrieval is fully local (bge-m3 embeddings + BM25 + bge-reranker); only generation/translation/judging call out to Gemini.

Code comments, log messages, and eval output are intentionally in **Greek** — keep that convention when editing existing files.

## Running & developing

Everything runs through Docker Compose (Postgres + backend + frontend). There is no local-venv workflow; the `venv/` at repo root is incidental and not used by the app.

```bash
docker compose up -d --build      # first build downloads ~4.5GB of models into a volume
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

The backend runs `uvicorn` **without `--reload`** (see [backend/Dockerfile](backend/Dockerfile)), even though `./backend` is bind-mounted. After editing backend code, restart to pick it up:

```bash
docker compose restart backend
```

### Tests

```bash
# whole suite
docker compose run --rm backend python -m pytest tests/ -v

# a single test
docker compose run --rm backend python -m pytest tests/test_rag_core.py::test_where_single_file -v
```

`conftest.py` makes `/app` the rootdir so tests can `import ai_core`, `import security`, etc. **Importing `ai_core` loads the embedding + reranker models at module level (~1 min the first time)** — any test or script that touches `ai_core` pays this cost once up front, then runs fast.

### Evaluation suite

All eval scripts import `ai_core`, so run them inside the backend container. Anything that calls the LLM-judge or generation consumes Gemini quota (**free tier ≈ 20 req/day**); the retrieval-only scripts do not.

```bash
docker compose exec backend python measure_reranker.py    # reranker score separation — NO quota
docker compose exec backend python chunk_experiment.py     # chunk-size sweep (retrieval metrics) — NO quota
docker compose exec backend python run_eval.py             # full retrieval + LLM-judge over tests.jsonl — USES quota
docker compose exec backend python faithfulness_eval.py    # answer-quality + faithfulness — USES quota
```

`run_eval.py` sleeps **30s between questions** to stay under the per-minute rate limit, and writes per-question results to `evaluation/results*.csv` (UTF-8-BOM so Greek renders in Excel). Golden sets live in `backend/evaluation/*.jsonl`.

## Architecture

### Two data stores, linked by `doc_id`
- **Postgres** (SQLAlchemy, [models.py](backend/models.py)) — users, document metadata + ingest `status` (`processing`/`ready`/`failed`), conversations, messages, feedback.
- **ChromaDB** ([backend/vector_db](backend/vector_db), `chromadb_data` volume) — the chunks + vectors. Each chunk carries `{file_name, page, user_id, is_public, doc_id}` metadata; **authorization is enforced in the vector query**, not just in SQL.

A document therefore exists in both: a Postgres `Document` row (status tracking) and N ChromaDB chunks. Deletion must hit both ([main.py](backend/main.py) `delete_document` → `ai_core.delete_file_from_db`).

### Request flow
`Streamlit (app_ui.py)` → HTTP → `FastAPI (main.py)` → `ai_core.py` (RAG) + Gemini. The frontend holds a JWT in `st.session_state` and sends it as a Bearer token; every protected endpoint depends on `get_current_user`.

Upload is async: [main.py](backend/main.py) `/upload` streams the file to disk (50 MB cap, sanitized filename), creates the `Document` row as `processing`, and hands ingest to a `BackgroundTasks` worker (`process_document`) that flips status to `ready`/`failed`. The frontend's library is an `@st.fragment(run_every="4s")` that polls `/documents`, so `processing → ready` appears without a manual refresh.

### RAG pipeline ([ai_core.py](backend/ai_core.py) `search_documents` → `ask_ai`)
1. **Translate-then-retrieve** — `optimize_query` detects Greek (`_has_greek`) and translates the query to English for retrieval only (docs are English); the answer stays in the user's language via the system prompt. Translations are cached.
2. **Dense** — bge-m3 (1024-dim, cosine), top-30, via Chroma's embedding function.
3. **Sparse** — BM25 with `el_tokenize` (lowercases + strips Greek accents so unaccented queries match), top-30.
4. **Fusion** — Reciprocal Rank Fusion (k=60), keep top-15.
5. **Rerank** — bge-reranker-v2-m3 cross-encoder scores each (query, chunk) pair.
6. **Relevance gate (anti-hallucination, line 1)** — if the best reranker score `< MIN_RERANK_SCORE` (currently **0.15**), return `[]` → the answer becomes "no relevant documents" instead of feeding weak context to the LLM.
7. **Generation** — top-5 chunks + source/page metadata + persona style + history → Gemini 2.5 Flash, **streamed** back as NDJSON.

CPU-bound steps (dense embed, BM25, rerank) run in `asyncio.to_thread` to avoid blocking the event loop.

**Two-line anti-hallucination defense:** the relevance gate (step 6) plus the system prompt's "if not in the SOURCE TEXT, say you can't find it" rule. The 0.15 threshold and this prompt together are what produce the ~4.95/5 faithfulness — don't change one without re-checking the other.

### Streaming + partial-save
`/chat` returns a `StreamingResponse` of NDJSON packets (`{"type":"sources"|"text"}`). The assistant message is saved in a `finally` block, so if the user disconnects mid-stream the partial answer is still persisted to history.

## Non-obvious constraints — read before changing these

- **ChromaDB 0.4.6 does not support `$in`.** `_build_where` builds an `$or` of `$eq` clauses instead. Don't "simplify" it to `$in`.
- **bge-m3 needs cosine and no prefixes.** The collection is created with `metadata={"hnsw:space": "cosine"}` (Chroma defaults to L2). Unlike the old e5 model, bge-m3 takes raw text — no `"query:"`/`"passage:"` prefixes.
- **Pinned versions are deliberate** ([requirements.txt](backend/requirements.txt)): Pydantic **V1** / FastAPI 0.99.1, for thesis reproducibility against the working container. Schemas use V1 `@validator` syntax — don't migrate to V2 patterns.
- **`chunk_size=1500, chunk_overlap=300`** in `ingest_pdf` is the measured winner of the chunk experiment (MRR 0.803, coverage 95%). Re-running the experiment is the way to change it, not a guess.
- **`MIN_RERANK_SCORE = 0.15`** was recalibrated down from 0.30 because *translated* Greek queries score lower than native English ones; 0.30 was silently dropping valid Greek questions. Re-measure with `measure_reranker.py` before touching it.
- **Models download to `HF_HOME=/home/appuser/.cache/huggingface`**, mapped to the `huggingface_cache` volume. If that path/volume mapping drifts, models re-download (~4.5GB) on every restart.
- **Required env vars** ([.env.example](.env.example)): `GEMINI_API_KEY`, `POSTGRES_PASSWORD`, `SECRET_KEY` (`openssl rand -hex 32`). `security.py` and `database.py` raise on startup if their keys are missing — by design.
- **`migrate_add_userid.py`** is a one-shot backfill for pre-multi-user chunks that lack `user_id`/`is_public` metadata — not part of normal startup.
