# Architecture & Developer Guide

Technical reference for the **Z-AI Platform** codebase — architecture, development workflow, and the non-obvious constraints behind the key design decisions.

## What this is

**Z-AI Platform** — a local-first RAG platform for scientific PDFs (diploma thesis). Upload PDFs, ask questions in Greek or English, get evidence-based answers with page-level citations. Retrieval is fully local (bge-m3 embeddings + BM25 + a cross-encoder reranker); Gemini is called only for generation, query translation, the corrective/conversational rewrites, and evaluation judging. That split is also why the thing is cheap: embedding and reranking cost **$0.00 per call**, so 1,000 questions come to **$5.69** and the only variable cost is generation.

Code comments, log messages, and eval output are intentionally in **Greek** — keep that convention when editing existing files.

## Running & developing

Everything runs through Docker Compose (Postgres + backend + frontend). There is no local-venv workflow; the `venv/` at repo root is incidental and not used by the app.

```bash
docker compose up -d --build      # first build downloads ~2.5GB of models into a volume
# UI:  http://localhost:8502      (container listens on 8501)
# API: http://localhost:8010/docs (container listens on 8000)
# DB:  localhost:5434             (container listens on 5432)
```

The backend runs `uvicorn` **without `--reload`** (see [backend/Dockerfile](backend/Dockerfile)), even though `./backend` is bind-mounted. After editing backend code, restart to pick it up:

```bash
docker compose restart backend
```

### Tests

```bash
# whole suite — 80 tests
docker compose exec backend python -m pytest tests/ -q

# the 51 model-free ones: no 2.4GB download, no Postgres, ~1s (this is the fast CI job)
docker compose exec backend python -m pytest tests/test_schemas.py tests/test_security.py \
    tests/test_gemini_rest.py tests/test_api_endpoints.py tests/test_metrics.py \
    tests/test_rate_limit.py -q

# a single test
docker compose exec backend python -m pytest tests/test_rag_core.py::test_where_single_file -v
```

`conftest.py` makes `/app` the rootdir so tests can `import ai_core`, `import security`, etc. **Importing `ai_core` loads the embedding + reranker models at module level (~1 min the first time)** — any test or script that touches `ai_core` pays this cost once up front, then runs fast.

### Evaluation suite

All eval scripts import `ai_core`, so run them inside the backend container. Anything that calls the LLM-judge or generation consumes Gemini quota (**free tier ≈ 20 req/day**); the retrieval-only scripts do not.

```bash
# --- free: no generation, no judge, fully deterministic ---
python run_eval.py evaluation/golden_set_50.jsonl --retrieval-only  # MRR / nDCG / coverage
python evaluation/random_coverage_baseline.py       # the chance floor every % is read against
python evaluation/ablation_ladder.py                # what each pipeline stage actually adds
python evaluation/bootstrap_ci.py --compare OLD.csv NEW.csv   # is the delta real at this n?
python evaluation/compare_pages_rerankers.py        # do the pages reaching Gemini change?
python evaluation/trace_query.py --target f.pdf:15  # at which stage was a page lost?
python evaluation/measure_gate_margin.py            # gate calibration, in vs out-of-corpus
python evaluation/measure_cost.py --ratio 4.53      # $ per 1,000 questions
python evaluation/measure_latency.py                # warm retrieval, per phase

# --- consume Gemini quota ---
python run_eval.py evaluation/golden_set_50.jsonl   # + LLM-judge
python evaluation/make_risk_subset.py --n 12        # judge only the questions that changed most
python evaluation/verify_corrective.py              # corrective agent on the real search path
```

Two things make the free tier survivable. First, `run_eval.py` sleeps **30s between questions** (`--delay`, full run only — `--retrieval-only` never waits). Second, **48 of 50 questions already score 5/5/5/5**, so a judge run can only prove *non-regression*; that means only the new configuration needs judging, at half the cost. `compare_pages_rerankers.py` answers "is a judge run even worth it?" for free.

Per-question results land in `evaluation/runs/*.csv` (UTF-8-BOM so Greek renders in Excel — read them back with `encoding="utf-8-sig"`, or the first column silently becomes `﻿id`). Golden sets live in `backend/evaluation/*.jsonl`.

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
2. **Dense** — bge-m3 (1024-dim, cosine), **exact brute-force** over the cached normalized matrix (`_get_dense_matrix` / `_dense_exact_ids`), top-30. Chroma's HNSW is bypassed on purpose — see constraints below.
3. **Sparse** — BM25 with `el_tokenize` (lowercases + strips Greek accents so unaccented queries match), top-30.
4. **Fusion** — Reciprocal Rank Fusion (k=60) with a deterministic tie-break on chunk id, keep top-15. Supports an optional third branch (`extra_ids`) and per-branch `weights`; both currently unused — see rejected experiments in the README.
5. **Rerank** — `RERANKER_MODEL` cross-encoder scores each (query, chunk) pair. Default `ms-marco-MiniLM-L-12-v2` (English) because step 1 already translated the query.
6. **Relevance gate (anti-hallucination, line 1)** — if the best reranker score `< MIN_RERANK_SCORE` (**−2.6**, raw logits), return `[]` → the answer becomes "no relevant documents" instead of feeding weak context to the LLM. **The threshold is tied to the model's score scale**, not a universal constant.
6b. **Corrective retry (CRAG)** — runs **only if step 6 fired**, so the happy path costs nothing. `_corrective_retry` asks Gemini to rewrite the query, then repeats the *full* pipeline (steps 2–5) and judges the result against `CORRECTIVE_MIN_SCORE` (**+0.4**) — deliberately **stricter** than the gate it is bypassing. Toggle with `ENABLE_CORRECTIVE`. The strictness is not cosmetic: at the gate's own threshold, two questions passed **without correct material**, i.e. the agent was defeating the defence by keyword-stuffing its own rewrite.
7. **Page-level expansion** — `_expand_to_pages` maps the top reranked chunks back to their source pages and returns up to `MAX_PAGES` **whole pages**, so tables and lists arrive intact rather than truncated mid-structure.
8. **Generation** — page texts + source/page metadata + persona style + history → Gemini 2.5 Flash over the **v1beta REST endpoint** ([gemini_rest.py](backend/gemini_rest.py)), **streamed** back as NDJSON. Parts flagged `"thought": true` are the model's internal reasoning and are filtered out explicitly; `THINKING_BUDGET` caps how many of those tokens it may spend before answering.

CPU-bound steps (dense embed, BM25, rerank) run in `asyncio.to_thread` to avoid blocking the event loop. `TORCH_THREADS` controls how many cores each of those steps may use — see the concurrency note below.

**Three-line anti-hallucination defense:** the relevance gate (step 6), the corrective retry's *stricter* second threshold (step 6b), and the system prompt's "if not in the SOURCE TEXT, say you can't find it" rule. Together they produce **5.00/5 faithfulness**, RAGAS **0.992** from an independent judge, and 5/5 correct refusals on out-of-corpus questions — don't change one without re-checking the others.

The ablation ladder shows why the gate matters more than its metrics suggest: dense-only scores MRR 0.731, +BM25/RRF 0.774, +reranker 0.793, **+gate 0.793**. The gate improves *no ranking metric at all* — and takes correct refusals from **0/5 to 5/5**. All the hybrid machinery buys +8.5% MRR; the gate buys 100% of the trustworthiness. It is only visible because the golden set contains questions the corpus **cannot** answer — most sets don't, which is why they cannot measure this defence at all.

### Streaming + partial-save
`/chat` returns a `StreamingResponse` of NDJSON packets (`{"type":"sources"|"text"}`). The assistant message is saved in a `finally` block, so if the user disconnects mid-stream the partial answer is still persisted to history.

## Non-obvious constraints — read before changing these

- **Η ακύρωση των caches είναι πλέον cross-process — αλλά μόνο στο ΙΔΙΟ filesystem.** Τα τρία caches (`_bm25_cache`, `_dense_cache`, `_sparse_cache`) είναι module-level dicts, με κλειδί `_corpus_signature() = (τοπικός μετρητής, mtime του chroma.sqlite3)`. Παλιότερα το κλειδί ήταν μόνο ο in-process μετρητής, οπότε με `--workers > 1` ένα ingest σε έναν worker **δεν** ακύρωνε τα caches των άλλων — σέρβιραν stale index αόριστα, **χωρίς κανένα σφάλμα**. Τρία πράγματα επαληθεύτηκαν αντί να θεωρηθούν δεδομένα ([check_multiprocess_safety.py](backend/evaluation/check_multiprocess_safety.py)): δεύτερη διεργασία γράφει και η πρώτη το βλέπει χωρίς restart· `add`/`delete` μετακινούν το mtime ενώ 30 συνεχόμενες **αναγνώσεις** όχι· το `os.stat` κοστίζει 2.4 µs. **Ο περιορισμός που ΜΕΝΕΙ:** το mtime είναι τοπικό αρχείο — πολλαπλά *μηχανήματα* θέλουν ακόμα Redis ή DB-backed ακύρωση. Ο ένας worker παραμένει default, αλλά τώρα από επιλογή χωρητικότητας (κορεσμός στους ~4 ταυτόχρονους), όχι από ορθότητα.
- **Rate limiting lives in Postgres, and the increment happens *inside* the database** ([rate_limit.py](backend/rate_limit.py)). It used to be in-memory `deque`s, which meant the limit silently multiplied by the worker count and reset on every restart. The replacement is a fixed window keyed on `(bucket, key, window_start)` with a single atomic `INSERT … ON CONFLICT DO UPDATE … RETURNING`. **Do not "simplify" this into read-modify-write in Python** — that reintroduces the race with no visible symptom, which is exactly what `test_rate_limit.py` pins down (two sessions, second must read 2). A dialect switch keeps it working on SQLite so the endpoint tests can hit real `/login` and `/chat` without Postgres. Two behaviour changes were accepted deliberately: blocked requests now count toward the limit (stricter), and the window is aligned rather than sliding (up to 2× in an instantaneous burst at the boundary). Redis was rejected on the same grounds as Langfuse — operational cost, not capability.
- **`/metrics` is Prometheus text exposition with zero dependencies and zero extra containers** ([metrics.py](backend/metrics.py), stdlib `threading` + `time` only). The per-request numbers already existed in logs and the UI; what was missing was the *aggregate*. It exposes `gate_block_rate`, `corrective_success_rate`, token counts for FinOps, and latency per phase — "how often does the gate actually fire in real use" had never been measurable before.
- **ChromaDB 0.4.6 does not support `$in`.** `_build_where` builds an `$or` of `$eq` clauses instead. Don't "simplify" it to `$in`.
- **Η ChromaDB χρησιμοποιείται ως document store, ΟΧΙ ως vector DB.** Μετά τη μετάβαση σε exact search, το `collection.query()` δεν καλείται πουθενά στον παραγωγικό κώδικα — μόνο `get`/`add`/`delete`. Αν κάποτε αντικατασταθεί, το ερώτημα είναι «ποιο document store» και όχι «ποιο vector DB» (ο Postgres που ήδη τρέχει θα έδινε ACID ingest και SQL authz).
- **bge-m3 needs cosine and no prefixes.** The collection is created with `metadata={"hnsw:space": "cosine"}` (Chroma defaults to L2). Unlike the old e5 model, bge-m3 takes raw text — no `"query:"`/`"passage:"` prefixes.
- **Pinned versions are deliberate** ([requirements.txt](backend/requirements.txt)): Pydantic **V1** / FastAPI 0.99.1, for thesis reproducibility against the working container. Schemas use V1 `@validator` syntax — don't migrate to V2 patterns.
- **`chunk_size=1500, chunk_overlap=300`** was inherited from a v1 experiment on a 2-paper corpus that also had duplicated chunks from an ingest/delete race — and has since been **re-validated on the current corpus** (`chunk_experiment.py --axis chunk`, which uses a temporary collection and scales retrieval depths inversely so it compares *chunking* rather than "how much text reached the end"). Two golden sets × two depth methods, with a control that reproduced the recorded numbers exactly: **750** loses 0.077 MRR *and* costs +25% latency; **1000** at scaled depths gains ≈1 question on the hard set for the same +25%; **1000 at fixed depths** — the only variant that could have *saved* time — gains nothing at all, 0.8 s vs 0.8 s, because reranker batch padding is dominated by the longest chunk in the batch either way.

  Going *upward* is closed by construction rather than by quality: `CrossEncoder` inherits `max_seq_length=512` and truncates **silently**, and `measure_chunk_tokens.py` puts the real budget at ≈2,230 characters (493 tokens × 4.53 chars/token). Today 15 of 418 chunks are clipped (3.6% of chunks, 0.5% of corpus text — negligible), but token counts vary 2× across chunks, so at `chunk_size=2000` roughly a quarter would be. That experiment would be measuring truncation while calling it chunking.
- **`MIN_RERANK_SCORE` is coupled to `RERANKER_MODEL`.** Score scales differ fundamentally between models: `bge-reranker-v2-m3` emits sigmoid in `[0,1]` (threshold 0.05), the `ms-marco-MiniLM` family emits raw logits in roughly `[-11,+11]`. Swapping the model without recalibrating will silently reject correct answers or admit irrelevant ones — and **this is not hypothetical**: the L-6 → L-12 upgrade shifted the lowest correct score from −1.80 to −2.08, so the old −2.0 would have started refusing a question it used to answer. The current **−2.6** is the midpoint of the entire `[−3.12, −2.08]` band that scores 61/61, chosen to maximise the *worst-case* margin on both sides rather than to favour one. Measure with `evaluation/measure_gate_margin.py` (full distribution + suggested threshold) or `compare_rerankers.py` for a model-to-model gap.

  `CORRECTIVE_MIN_SCORE` is the same coupling one level down, and is **the most fragile number in the system**: it is calibrated on n=6 with no validation set, and it got *weaker*, not safer, with the better reranker — because L-12 now solves the easy cases on its own, the calibration base shrank from 10 questions to 6 and the threshold now rests on **two**. Worse, the input is not reproducible: the rewrite is a Gemini call with no seed, and two runs of identical code recovered 2 questions and 1. **Do not compare two `verify_corrective.py` runs as if they were the same experiment.**
- **The reranker is English-only on purpose.** Step 1 translates Greek to English *before* retrieval and the corpus is English, so the cross-encoder always sees English↔English. A multilingual model here pays for capability it never uses: measured 15.048 ms → 693 ms (21.7×) with no ranking quality loss.
- **`TORCH_THREADS` defaults to every visible core, and measurement says leave it there.** PyTorch's own heuristic picked 4 and left 1.39× on the table under WSL2. The obvious worry is that it counts threads *per* `predict()` while requests run in `asyncio.to_thread`, so N concurrent questions should contend — but [concurrency_benchmark.py](backend/evaluation/concurrency_benchmark.py) (separate processes per setting, 4 waves per cell) shows the trade is not worth taking:

  | concurrent users | p50 @ 8 threads | p50 @ 4 threads | throughput 8t | throughput 4t |
  |---|---|---|---|---|
  | 1 | **0.45 s** | 0.54 s | **2.17 req/s** | 1.86 req/s |
  | 2 | 0.80 s | **0.72 s** | 2.50 | **2.69** |
  | 4 | 1.50 s | **1.39 s** | 2.57 | **2.76** |
  | 8 | 5.72 s | 5.63 s | 1.36 | 1.40 |

  Four threads win under load by 7–10% but lose 20% for the single user, and at 8 concurrent users the two are indistinguishable — both collapse. So `TORCH_THREADS=4` was written up as a recommendation, then **withdrawn**: it had been asserted before it was measured, and the measurement did not support it.

  ⚠️ **That table is L-6 era.** After the L-12 upgrade the whole curve was re-measured at 8 threads:

  | concurrent users | p50 | p95 | throughput |
  |---|---|---|---|
  | 1 | 0.61 s | 0.62 s | 1.65 req/s |
  | 2 | 1.27 s | 1.38 s | 1.55 |
  | 4 | 2.30 s | 2.67 s | **1.67** ← saturation |
  | 8 | 7.02 s | 7.25 s | 1.12 ← collapse |

  Throughput fell 2.57 → 1.67 req/s (**−35%**), which is *exactly* the predicted ceiling: rerank got 1.61× heavier, so 1/1.61 = 62% was expected and 65% was measured. **The agreement is the useful part** — it says there is no hidden contention, only the reranker's CPU. **The capacity number still matters more than the thread count: throughput saturates at 4 concurrent users and *drops* at 8**, and the saturation point did not move. Past saturation, more users produce fewer answers, not slower ones. That ceiling is a property of one CPU-bound worker.

  ⚠️ **Measuring this needs separate processes.** `torch.set_num_threads()` only affects threads created *after* the call, so changing it mid-benchmark leaves the existing `to_thread` workers untouched and both conditions silently measure the same value. Production is fine — `ai_core` sets it at import, before any worker exists (verified: the reranker really does run with `TORCH_THREADS` threads inside `to_thread`).
- **`THINKING_BUDGET` is not free performance — it was measured against the judge.** Disabling thinking entirely (0) is 2.73× faster to first token and costs **faithfulness 5.0 → 2.0** on a multi-hop question; every regression was multi-hop, and every one returned to 5.0 at 512. Keyword coverage showed no loss at all, which is exactly why a generation change must be validated with `make_judge_subset.py` (per-question diff) and not with coverage metrics.
- **The generation path deliberately does not use the SDK.** `google.generativeai` 0.8.6 — and the `google.ai.generativelanguage` protobuf under it — has **no `thinking_config` field**, verified by introspection; it predates thinking models. Since `google-genai` needs pydantic ≥2.12.5 and the stack is pinned to V1, REST over httpx is the only path to that setting. Note also that `CrossEncoder.forward` in sentence-transformers 5.x passes the feature dict as a **positional** argument to child modules, so replacing `reranker.model` with a new object (e.g. a quantized copy) breaks the call chain — mutate in place instead.
- **Where this design breaks, measured** ([scaling_benchmark.py](backend/evaluation/scaling_benchmark.py)). The corpus is 418 chunks and every architectural choice is right *at that size* — but "right at 418" is not an answer to "what would you change at 500k". Synthetic scaling, same code paths:

  | chunks | dense exact | HNSW | BM25 query | BM25 build | RRF | RAM |
  |---|---|---|---|---|---|---|
  | 418 | 0.04 ms | 0.06 ms | 0.2 ms | 0.1 s | 0.06 ms | — |
  | 10.000 | 1.6 ms | 0.53 ms | 10.4 ms | 1.5 s | 0.47 ms | +40 MB |
  | 50.000 | 9.2 ms | 0.85 ms | 61.1 ms | 7.8 s | 3.9 ms | +330 MB |
  | 200.000 | 46.2 ms | 0.82 ms | — | — | 22.0 ms | +350 MB |

  **The lexical branch breaks first, not the vector one.** At 50k, BM25 querying is 6.7× slower than brute-force dense, and its 8 s rebuild repeats on *every* ingest — `rank_bm25` is pure Python and holds every tokenized document in memory. The first component that would need replacing is BM25 (Tantivy/Lucene, or Postgres full-text), **not** the vector search.

  **Exact search stays correct far longer than folklore suggests.** At 50k, ANN would save 8 ms — 2% of the reranker's fixed 706 ms — in exchange for a 17 s index build per ingest and the loss of determinism. The reranker sees a constant 15 pairs at any corpus size, so the step that dominates latency today (97.3% of the 725 ms warm path) is the one thing that does not scale with the corpus.

  Caveat kept explicit: vectors are synthetic (random, normalized). For **time and memory** that is equivalent to real embeddings; for **ANN recall** it is the worst case, since random vectors in 1024 dimensions are near-equidistant. Treat the recall column as a lower bound, not a prediction.
- **Exact search replaces HNSW deliberately.** ChromaDB rebuilds its in-memory HNSW graph from the WAL on every process start and did not always yield the same top-30. At this corpus size a 418×1024 matmul costs ~0.5 ms, so ANN buys nothing. Exact search also reads vectors from the store rather than the graph, so a partially-built index cannot hide (this exact failure — 77 of 422 vectors missing — happened once, caused by numpy 2 removing `np.NaN`, which is why `numpy<2` is pinned).
- **PDF extraction is PyMuPDF + NFKC, not pypdf.** pypdf inserted spaces inside words (`A WS`, `distribu ted`), producing tokens that matched nothing. Measured: broken tokens 3.7% → 2.5%, MRR +0.026, extraction 4× faster.
- **Models download to `HF_HOME=/home/appuser/.cache/huggingface`**, mapped to the `huggingface_cache` volume. If that path/volume mapping drifts, models re-download (~2.5GB) on every restart.
- **Required env vars** ([.env.example](.env.example)): `GEMINI_API_KEY`, `POSTGRES_PASSWORD`, `SECRET_KEY` (`openssl rand -hex 32`). `security.py` and `database.py` raise on startup if their keys are missing — by design.
- **`migrate_add_userid.py`** is a one-shot backfill for pre-multi-user chunks that lack `user_id`/`is_public` metadata — not part of normal startup.
