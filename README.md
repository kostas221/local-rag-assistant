# Z-AI Platform

[![tests](https://github.com/kostas221/local-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/kostas221/local-rag-assistant/actions/workflows/ci.yml)

A local-first **RAG (Retrieval-Augmented Generation) platform** for scientific documents: upload PDFs, ask questions in natural language (Greek or English), and get evidence-based answers with page-level citations — powered by a fully local retrieval stack and Gemini for generation.

Built as a diploma thesis project focusing on **cross-lingual retrieval quality** and **measurable, reproducible evaluation**.

> 📊 **Headline results** (89 questions across 4 golden sets, deterministic): LLM-judge **5.00 / 4.98 / 5.00 / 5.00** on accuracy, completeness, relevance & faithfulness · RAGAS faithfulness **0.992** from an *independent* judge · **zero hallucinations** — all 5 out-of-corpus probes correctly refused · keyword coverage **98.5%** against a measured random-chance floor of **33.5%** · warm retrieval **725 ms**, time-to-first-token **2.78 s** — [details below](#evaluation-results)

## Demo

![Z-AI Platform — anti-hallucination gate + bilingual Q&A](docs/demo.gif)

> Upload a PDF, ask in Greek or English, and get grounded answers with page-level citations — plus a clear *"not found"* when the answer isn't in the documents.

## What each pipeline stage actually buys

Every component is justified by an ablation on the **same 45 in-corpus questions**, not by reputation:

| Stage | MRR | Out-of-corpus correctly refused |
|---|---|---|
| Dense only (bge-m3) | 0.731 | 0/5 |
| \+ BM25 via RRF | 0.774 | 0/5 |
| \+ cross-encoder rerank | 0.793 | 0/5 |
| \+ **relevance gate** | 0.793 | **5/5** |

**The gate improves no ranking metric — and is the single most important component.** Without it the system answers *every* question, including the five it has no material for. All the hybrid machinery buys **+8.5% MRR**; one calibrated threshold buys 100% of the trustworthiness.

This is only visible because the golden set contains questions with **no answer in the corpus**. Most golden sets don't, which is why they cannot measure the anti-hallucination defence at all.

```bash
docker compose exec backend python evaluation/ablation_ladder.py
```

## Features

- **Hybrid retrieval** — dense semantic search (BAAI/bge-m3, 1024-dim, cosine) fused with lexical BM25 via Reciprocal Rank Fusion (RRF)
- **Exact vector search** — brute-force cosine over the full index instead of ANN. Measured: identical top-30 to HNSW (30/30, same order) with no speed cost at this corpus size, and *deterministic* across processes
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-12-v2` reorders fused candidates. English-only by design: the pipeline translates before retrieval, so a multilingual reranker paid for capability it never used
- **Anti-hallucination relevance gate** — if even the best chunk scores below a *measured* threshold, the system answers "not found" instead of feeding irrelevant context to the LLM. Calibrated from the score gap between in-corpus and out-of-corpus questions, not guessed
- **Corrective retrieval agent (CRAG)** — when the gate fires, the query is rewritten and a full second retrieval pass runs against a *stricter* threshold. Zero cost on the happy path; recovers questions phrased without domain jargon
- **Cross-lingual QA** — Greek questions are translated for retrieval over English papers (translate-then-retrieve, permanently cached, domain-aware); answers come back in the user's language
- **Conversational rewriting** — follow-up questions are rewritten into self-contained queries using history. Measured at **90% coverage vs a 41.2% random floor**; leak tests confirm an off-topic follow-up is still refused
- **Greek-aware BM25** — accent-stripping tokenizer so unaccented queries still match
- **Robust PDF extraction** — PyMuPDF with Unicode NFKC normalization and de-hyphenation. Fixes broken intra-word spacing (`A WS` → `AWS`), ligatures, and line-break hyphens that silently break both lexical and semantic matching
- **Built-in evaluation framework** — retrieval metrics, LLM-as-judge scoring, RAGAS cross-validation, per-stage query tracing, random-chance baselines, a determinism checker, and a CI regression gate
- **Prometheus metrics** — `/metrics` in text exposition format, **zero dependencies, zero extra containers**: gate block rate, corrective success rate, token counts (FinOps), latency per phase
- **Feedback capture** — 👍/👎 on every answer with an optional comment, upserted per message in Postgres — ground truth for error analysis
- **Multi-user** — JWT auth, per-user document ownership, public/private sharing, and rate limiting **shared across processes** via Postgres

## Architecture

```
┌────────────┐  HTTP   ┌─────────────────────────────────┐
│  Streamlit │ ──────► │            FastAPI               │
│  frontend  │         │                                  │
│   :8502    │         │  /chat ──► RAG pipeline          │
└────────────┘         │  /upload ─► background ingest    │
                       │  /metrics ► Prometheus text      │
                       └──────┬──────────┬───────────┬────┘
                              │          │           │
                       ┌──────▼───┐ ┌────▼─────┐ ┌───▼────────┐
                       │ ChromaDB │ │ Postgres │ │ Gemini 2.5 │
                       │ (chunks, │ │ (users,  │ │   Flash    │
                       │ vectors) │ │  chats)  │ │ (gen+judge)│
                       └──────────┘ └──────────┘ └────────────┘
```

### RAG pipeline (per question)

1. **Query optimization** — Greek questions translated to English search queries with a *domain-aware* prompt (permanently cached to disk, so the same question always yields the same retrieval)
2. **Dense search** — bge-m3 embeddings, exact cosine over the whole index, top-30
3. **Sparse search** — BM25 with Greek-aware tokenization, top-30
4. **Fusion** — Reciprocal Rank Fusion (k=60) with deterministic tie-breaking, keep top-15
5. **Rerank** — `ms-marco-MiniLM-L-12-v2` cross-encoder scores each (query, chunk) pair as **raw logits**
6. **Relevance gate** — best score below −2.6 → "no relevant documents" (no hallucination)
7. **Corrective retry** — *only if step 6 fired*: rewrite the query, run steps 2–5 again, accept only above a stricter +0.4
8. **Page-level expansion** — top chunks map back to their source pages; whole pages (max 8) go to the LLM, so tables and lists arrive intact
9. **Generation** — Gemini 2.5 Flash over the v1beta REST endpoint (thinking budget capped), streamed with source/page metadata

Design decisions and trade-offs behind each step: [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Evaluation results

Four golden sets, **89 questions total**, over 7 open-access cloud/serverless papers (122 pages, 418 chunks). They are kept separate on purpose — each one measures something the others cannot.

| Set | n | Purpose |
|---|---|---|
| `golden_set_50` | 45 + 5 out-of-corpus | The stable baseline |
| `golden_multihop_new` | 11 | Cross-document reasoning |
| `golden_hard_paraphrase` | 16 | **Deliberately badly-worded** paraphrases — a stress test, not a baseline |
| `golden_conversations` | 12 | Multi-turn, incl. 2 topic-leak probes |

Every number is **reproducible**: the same question yields the same retrieval on every run (see [determinism](#determinism)).

### Answer quality (LLM-as-judge)

| Metric | Typical use (n=46) | All questions (n=50) |
|---|---|---|
| Accuracy | **5.00 / 5** | 4.94 / 5 |
| Completeness | **4.98 / 5** | 4.90 / 5 |
| Relevance | **5.00 / 5** | 4.96 / 5 |
| Faithfulness (no hallucinations) | **5.00 / 5** | 4.94 / 5 |

**48 of 50 questions already score 5/5/5/5.** This is stated plainly because it has a consequence: a judge run can no longer demonstrate *improvement* — it is purely a non-regression check. That fact is exploited to halve evaluation cost, since a perfect score on the new configuration cannot be hiding a drop.

### RAGAS cross-validation (independent judge, n=45)

| Metric | Score |
|---|---|
| Faithfulness | **0.9920** |
| Context recall | **1.0000** |
| Answer relevancy | 0.8552 |
| Context precision | 0.7826 |

Two independent judges agree on faithfulness, so **self-preference bias did not materialise**. `context_precision 0.783` is the measured price of page-level expansion: precision fell 0.800 → 0.783 while recall rose 0.944 → **1.000**. A deliberate trade, with no illusions about it.

### Retrieval

| Metric | In-corpus (n=45) | Random-chance floor |
|---|---|---|
| Keyword coverage | **98.5%** | 33.5% |
| MRR | **0.793** | 0.149 |
| nDCG | 0.807 | — |

Per category: `direct_fact` 0.869 · `enumeration` 0.751 · `reasoning` 0.831 · `multi_hop` 0.541.

### Every percentage is read against its chance floor

Coverage scores are meaningless without knowing what a *random* retriever would score. Computed analytically (hypergeometric, zero simulation) rather than estimated:

| Set | Chance floor | Observed | Margin |
|---|---|---|---|
| `golden_multihop_new` | 22.6% | 98.5% | **+75.9** |
| `golden_set_50` | 33.5% | 98.5% | **+65.0** |
| `golden_conversations` | 41.2% | 90.0% | **+48.8** |

This exercise also **found bugs in the evaluation itself**: 31 keywords in the main set are found on >60% of pages by chance (`serverless` appears on 65 of 122 pages — 99.8% random hit rate), and one verification tool had been silently crashing on the conversations set, so that set had never been checked at all.

```bash
docker compose exec backend python evaluation/random_coverage_baseline.py
```

### The relevance gate, measured

| Model | In-corpus min | In-corpus median | Out-of-corpus max | Gap |
|---|---|---|---|---|
| MiniLM-L-6 | −1.80 | 4.15 | −2.69 | 0.89 |
| **MiniLM-L-12** | −2.08 | 4.62 | −3.12 | **1.04** |

The threshold **−2.6** is the midpoint of the entire range `[−3.12, −2.08]` that scores 61/61, maximising the *worst-case* margin on both sides. The old −2.0 would have broken under L-12: it would have refused a correct answer scoring −2.08.

### Determinism

Retrieval is bit-for-bit reproducible across processes. This was not free — two sources of non-determinism were found and fixed:

- `list(set(...))` in the fusion step made iteration order depend on `PYTHONHASHSEED`, which is randomised per process. With **15 tied RRF scores out of 51 candidates**, ties broke differently on every run and changed which chunks survived the top-15 cut.
- ChromaDB's in-memory HNSW graph is rebuilt from the write-ahead log on every process start and did not always produce the same top-30.

Before the fix, the *same code* produced in-corpus MRR of both 0.764 and 0.755. Every measurement in this README was taken after it.

One measurement is **explicitly not reproducible** and is labelled as such: corrective-agent verification depends on a Gemini rewrite with no seed, so two runs of the same code recovered 2 and 1 questions respectively. That is documented rather than averaged away.

```bash
docker compose exec backend python evaluation/check_determinism.py
```

### Performance

| | Value |
|---|---|
| Warm retrieval | **725 ms** (rerank 706 = 97.3%, dense 0.5, BM25 1.4, expand 7.9) |
| End-to-end | 2.8–4.3 s · TTFT 2.78 s |
| Prompt size | ~9,800 tokens (438 pages of context) |
| Throughput | 1.65 req/s, saturating at **4 concurrent users** |

Concurrency, re-measured after the reranker upgrade:

| Users | p50 | p95 | Throughput |
|---|---|---|---|
| 1 | 0.61 s | 0.62 s | 1.65 req/s |
| 2 | 1.27 s | 1.38 s | 1.55 req/s |
| 4 | 2.30 s | 2.67 s | 1.67 req/s ← saturation |
| 8 | 7.02 s | 7.25 s | 1.12 req/s ← collapse |

The L-6 → L-12 upgrade cost **−35% throughput**, which is *exactly* the predicted ceiling (rerank got 1.61× heavier → 1/1.61 = 62%; measured 65%). The agreement matters more than the number: it proves there is **no hidden contention** — only the reranker's CPU cost. The saturation point did not move.

## Technical decisions

What was measured, kept, and — more often — **rejected**. Each row is a real experiment, not an opinion.

### Kept

| Change | Measured effect |
|---|---|
| **PyMuPDF + NFKC** instead of pypdf | Broken tokens 3.7% → 2.5%; 721 ligatures and 941 hyphenations eliminated. MRR +0.026 |
| **English reranker** (568M → 22M params) | The pipeline translates to English *before* retrieval, so the cross-encoder always sees English↔English. Latency 15,048 ms → **693 ms (21.7×)**, and answer accuracy went *up* (4.96 → 5.00) |
| **MiniLM-L-6 → MiniLM-L-12** (22M → 33M) + recalibrating both thresholds | Hard set **10/16 → 12/16** · gate gap 0.89 → **1.04** · correct chunk at rank 1 **36 → 40 / 56** · corrective-pass hallucinations **2 → 0 at every threshold**. Unchanged: coverage (identical in all 45), 61/61 gate, 5/5 refusals, judge. Cost: **+275 ms (+9% e2e)** |
| **Exact search** instead of HNSW | Identical top-30 (30/30, same order), no speed cost at 418 vectors, and deterministic. Also reads vectors from the store rather than the graph, so a partially-built index cannot hide |
| **Deterministic fusion** | `dict.fromkeys` + tie-break on chunk id. Made every subsequent measurement trustworthy |
| **CPU thread count** | PyTorch picked 4 threads via a physical-core heuristic that comes out conservative under WSL2, while the container had 8. Rerank 667 ms → **479 ms (1.39×)**; retrieval scores bit-identical, since only the parallelism of the same computation changed |
| **Reranker `batch_size` 32 → 4** | At the default, all 15 candidates land in one batch and every pair is padded to the longest one. Smaller batches group similar lengths: 479 ms → **434 ms (1.15×)**, Pearson ρ **1.0000**, **0** top-1 changes |
| **Capped thinking budget** (REST) | The 2.5-flash model spent ~900–1700 hidden "thinking" tokens before the first visible character — 94% of the wait on a blank screen, billed in full. Capping it at 512 cut TTFT 3.90 s → **2.78 s** with judge scores unchanged |
| **Domain-aware translation prompt** | Without domain context the translator produced everyday English, not field terminology: Greek *«μηχανήματα»* → `machinery` (industrial!) instead of `servers`. Two questions gained **+7.74** and **+4.14** logits. Control: 30 English questions, max \|Δ\| = **0.000** |
| **Corrective retrieval agent** | 4 questions went from silence to a correct answer with **0 hallucinations**, out-of-corpus still 5/5, main set 61/61 untouched. Latency cost only on questions that were *already* being refused |
| **Stricter corrective threshold** than the gate | Without it, 2 questions passed the gate **without correct material** — the agent was bypassing the defence via keyword stuffing in its own rewrite |
| **Rate limiting moved to Postgres** | Atomic `INSERT … ON CONFLICT DO UPDATE … RETURNING`, so the counter increments *inside the database*. Shared across processes, survives restart, **zero new containers** |
| **`/metrics` endpoint** | The per-request numbers already existed in logs and the UI; the *aggregates* did not. "How often does the gate fire in real use" had never been measurable |

### Rejected, with numbers

| Change | Result | Why it failed |
|---|---|---|
| **Query enrichment with domain terms** (3 variants) | Best variant fixed the single known hallucination (coverage 0% → 100%) and pushed 4 more questions to 100% — **and leaked one out-of-corpus question** through the gate | The finding is architectural, not a bug: enrichment helps retrieval **only when the reranker also sees the added words** — and that is exactly when the gate is misled. An asymmetric variant (enriched query for search, original for judging) protected the gate perfectly but netted **+1/−1 = zero**. You cannot have one without the other |
| **`bge-reranker-v2-m3` (568M)** | 12/16 on the hard set — but against the 33M model: **+1 real answer AND +1 hallucination** = net zero, for 17× the parameters and ~15 s/question | Also *more* willing to answer without material. The first rejection (2025) had been recorded as "a tie" on a golden set that later proved to have survivorship bias — re-tested, and the right reason found |
| **`bge-reranker-base` (278M)** | **Worse than the 22M model**: 3/16 hard *and* 6/10 normal | Not a "smaller v2-m3": it scores out-of-corpus chunks *high*, so any threshold that refuses them also kills the correct ones. **Discriminability does not scale with size** |
| **Cascade reranking** (cheap model filters, expensive model judges) | Rejected by arithmetic, without running: L-6×15 (434 ms) + L-12×6 (282 ms) = **716 ms > 706 ms** | Cascades need a first stage 10–100× cheaper; here the ratio is **1.63×** |
| **`chunk_size` 750 / 1000** | Re-measured on the current system with a reproducing control. 750: −0.077 MRR *and* +25% latency. 1000 at fixed depths: **MRR +0.029 while coverage −5.2pp** | The keyword-proxy MRR rises *mechanically* with smaller chunks (higher keyword density per chunk) while coverage shows 2–3 questions **lost material**. Fifth confirmation that MRR is not the judge |
| **`chunk_size` above 1500** | Closed by construction: the cross-encoder truncates **silently** at 512 tokens. Budget = 493 tokens × 4.53 chars/token ≈ **2,230 characters** | Today only 3.6% of chunks are clipped (0.5% of the corpus — negligible), but chunk-length variance is 2×, so 2000 would clip ~25%. We would have measured truncation and called it chunking |
| **`RERANK_CANDIDATES` 15 → 12 / 10** | Every summary metric came out **identical** — best logit, gate gap, keyword@1, 56/56. Apparently 250 ms for free | **False.** Comparing the *pages that actually reach the LLM* showed N=12 changes the prompt in **42/56** questions — a bigger change than the model swap itself. The reranker **reorders**: a chunk at RRF rank 14 can rise to rank 3 |
| PyTorch dynamic INT8 (fbgemm) | **1.37× faster**, Pearson ρ **0.9970** — and rejected anyway | It shifts the worst in-corpus score below the gate: a question answered correctly today would be silently refused. Separation between relevant and irrelevant narrows by **43%**, so recalibrating only moves the problem. **ρ=0.997 with a broken gate is the lesson**: score correlation is the wrong metric for a reranker |
| Thinking budget 0 (fully disabled) | 2.73× faster TTFT, then **faithfulness 5.0 → 2.0** on a multi-hop question | Keyword coverage showed *zero* loss — a saturated metric that missed it entirely. All regressions were multi-hop, and all returned to 5.0 at budget=512 |
| Query decomposition for multi-hop | Three merge strategies, n=15. Best design scored +2.3% — which is **35 → 36 keywords out of 45. One.** | Costs +1.5–2.3 s on *every* question plus a mandatory Gemini call, because routing must run even to decide "don't split". Multi-hop questions already score 5.00/5.00/5.00/5.00 — there was no quality problem to solve |
| Page score = sum of top-K chunk scores | MRR +0.0006; one question collapsed 1.000 → 0.389 | Structurally biased toward text-dense pages, which produce more chunks regardless of relevance |
| BGE-M3 native sparse as a 3rd RRF branch | 0.799 → 0.779 (equal weights), → 0.791 (weighted) | Finds more but ranks worse: each extra branch dilutes the dense signal, the strongest one |
| Context volume 8 → 6 pages | nDCG **identical**, coverage 97.0% → **94.8%** | ~25% fewer prompt tokens is real, but −2.2pp coverage is real lost content |
| Langfuse for tracing | 6 containers for ~50 traces; the project runs 3 | Rejected on operational cost, not capability — the same trade later applied to Redis for rate limiting |
| Postgres/pgvector migration | Zero product improvement | The scaling benchmark settled it: **BM25 breaks first**, not the vector store |
| Contextual compression | Solves no measured problem (faithfulness 0.992, recall 1.000) | Criterion if ever revisited: recall must stay 1.000 **and** faithfulness ≥ 0.99 |
| HyDE / ColBERT / Docling | not attempted, with reasons | Late interaction needs multi-vector storage the pinned store lacks; layout-aware extraction was abandoned after diagnosis showed the failing questions fail at the *reranker*, with the keywords present in the extracted text all along |

### The finding that mattered most

Retrieval MRR turned out to be **decoupled from answer quality** in this system:

```
reasoning questions   MRR 0.823  →  5.00 / 5.00 / 5.00 / 5.00
11 new multi-hop      MRR 0.493  →  5.00 / 5.00 / 5.00 / 5.00
one question          MRR 1.000  →  accuracy 4, completeness 4
reranker upgrade      MRR 0.500 → 1.000  →  accuracy 5 → 4, with
                                            provably identical material
```

With coverage above 97%, the correct material already reaches the model — MRR only measures whether it arrives *first*, and "lost in the middle" did not materialise at ~9,800 tokens with Gemini 2.5 Flash.

**And the converse: no intermediate metric predicts the final prompt.** The reranker upgrade produced *identical* coverage in 45/45 questions — and **35 of 56** questions received different pages. `RERANK_CANDIDATES=10` produced identical best-logit, gate gap and keyword@1 — and changed the pages in **52 of 56**. Coverage measures keywords, best-logit measures the top-1; **the prompt is pages**. Only one tool looks at what actually reaches the LLM, and it costs nothing to run:

```bash
docker compose exec backend python evaluation/compare_pages_rerankers.py
```

### The second finding: your golden set hides your failures, because you wrote it

The gate scored a perfect **61/61** — and it was false. One question had been written naturally, was refused, and **was deleted from the set**. Sixteen deliberately badly-worded paraphrases of *existing* questions then exposed **11/16 wrong refusals**. A production user does not get the option to delete the question that breaks your system.

Three lessons that generalise:

1. **The reranker is lexical-driven.** The same question without the papers' terminology drops by up to **12 logits** — with the correct chunk still at rank 1.
2. **Threshold tuning is a dead end when distributions overlap** (7.27 logits here). The fix belongs in the *query*, not the threshold.
3. **A "better" prompt can destroy your ability to filter.** In one variant the hallucinations were the two *lowest* scores — filterable. In the "improved" one, the **highest** score was a hallucination — no threshold catches that.

### Where a failure actually happens

A per-stage tracer answers "at which step was this page lost?" — translation, dense, BM25, RRF, rerank, gate, or prompt. It settled the one remaining known hallucination: the page **is** retrieved by BM25 at rank 1 when the question contains the rare term `vpxenc`, and never retrieved when the same question says "encoder". Feeding the term in via query enrichment fixes retrieval but breaks the gate, and supplying it to BM25 alone lets the reranker bury the page again — so it is a **cross-encoder comprehension limit**, not a retrieval bug.

```bash
docker compose exec backend python evaluation/trace_query.py --id q028 --target excamera-nsdi17.pdf:15
```

## Known limitations

- **The corrective threshold rests on n=2.** After the reranker upgrade the L-12 model solved the easy cases on its own, so the calibration base *shrank* from 10 questions to 6, and the recovered ones from 4 to 2. A sweep shows the entire range `[−2.6, +0.8]` gives 2 correct / 0 hallucinations, so the chosen value is the midpoint — defensible, but **not optimised, and honestly not optimisable at this sample size**
- **Corrective verification is not reproducible.** The rewrite is a Gemini call with no seed; two runs of identical code recovered 2 and 1 questions. Every other measurement in this project is deterministic — this one is not, and comparing two of its runs as if they were the same experiment is invalid
- **4 of 16 hard questions remain unanswered — and 3 of them *should*.** They use bare demonstratives (*"that older system"*, *"the other paper"*) with no antecedent, across 7 papers. Without conversation history they have no objective answer, so refusing is correct behaviour, not failure
- **One known hallucination** (h002): the gate passes it at +1.27 with zero keyword coverage, so the corrective agent never even runs. High confidence, wrong material — the defence catches "I don't know", not "I think I know"
- **Golden sets are self-authored**, ~89 questions against an industry norm of ~100 and a statistical-confidence requirement closer to 250. Confidence intervals are not yet reported
- **`google.generativeai` is deprecated** and `google-genai` cannot be adopted: it requires pydantic ≥2.12.5, incompatible with the pinned Pydantic V1 stack. Generation bypasses the SDK entirely via the v1beta REST endpoint — not a workaround but a requirement, since the 0.8.6 `GenerationConfig` has **no `thinking_config` field at all**
- **Reranker dominates retrieval latency** — 706 ms of the 725 ms warm path (97.3%). Quantization breaks the gate, ONNX gains nothing, a smaller model loses accuracy, and a cascade is arithmetically impossible. **This is the CPU ceiling, not an unfinished optimisation**
- **Throughput saturates at 4 concurrent users and collapses at 8.** One CPU-bound worker is the ceiling — past it, more users produce fewer answers rather than slower ones. Single worker is now a *capacity* decision, not a correctness one: the cache-invalidation bug that used to force it was fixed and verified
- **The corpus is 418 chunks, and the design is tuned for that** — but the breaking points are measured rather than guessed. BM25 fails first (6.7× slower than brute-force dense at 50k chunks, with an 8 s rebuild per ingest); exact vector search stays the right call well past 100k
- **`ai_core.py` is ~1,210 lines with 7 responsibilities.** A split is planned with a specific hazard identified: 8 files monkeypatch its module globals, so a facade with `import *` would break them **silently**
- **Free-tier Gemini quota** (~20 requests/day) is enough for chatting but not for full evaluation runs

## Quickstart

Requirements: Docker Desktop (WSL2 backend on Windows), a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/kostas221/local-rag-assistant.git
cd local-rag-assistant
cp .env.example .env        # fill in your keys
docker compose up -d --build
```

First start downloads the embedding + reranker models (~2.5 GB, cached in a volume afterwards).

- UI: http://localhost:8502 — register, upload a PDF, wait for "ready", ask away
- API docs: http://localhost:8010/docs
- Metrics: http://localhost:8010/metrics

## Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key (generation, translation, LLM-judge) |
| `POSTGRES_PASSWORD` | Postgres password (compose wires the DSN) |
| `SECRET_KEY` | JWT signing key — generate with `openssl rand -hex 32` |
| `RERANKER_MODEL` | Cross-encoder for reranking. **Changing this requires recalibrating `MIN_RERANK_SCORE`** — score scales differ between models |
| `MIN_RERANK_SCORE` | Relevance gate threshold, in **raw logits**. Measure with `evaluation/measure_gate_margin.py` |
| `ENABLE_CORRECTIVE`, `CORRECTIVE_MIN_SCORE` | Corrective agent on/off and its stricter acceptance threshold |
| `DENSE_CANDIDATES`, `RERANK_CANDIDATES`, `EXPAND_INPUT`, `MAX_PAGES` | Pipeline depths, tunable without rebuild |

## Tests & evaluation

```bash
# full suite: validators, security, fusion logic, relevance gate, corrective
# agent, metrics, REST generation, and HTTP endpoints incl. authorization
docker compose exec backend python -m pytest tests/ -q          # 72 tests

# the model-free ones — no 2.4GB download, no Postgres, ~2s (43 tests, fast CI job)

# determinism — all stage signatures must be identical across runs
docker compose exec backend python evaluation/check_determinism.py

# random-chance floor for every golden set (analytic, zero cost)
docker compose exec backend python evaluation/random_coverage_baseline.py

# what each pipeline stage contributes (zero cost)
docker compose exec backend python evaluation/ablation_ladder.py

# at which stage was a page lost? translate/dense/BM25/RRF/rerank/gate/prompt
docker compose exec backend python evaluation/trace_query.py --id q028

# does a change alter the pages that reach the LLM? — run BEFORE spending judge quota
docker compose exec backend python evaluation/compare_pages_rerankers.py

# gate calibration: best-logit distribution and the margin on both sides
docker compose exec backend python evaluation/measure_gate_margin.py

# retrieval only — no API quota, ~1 minute
docker compose exec backend python run_eval.py evaluation/golden_set_50.jsonl --retrieval-only

# where the time goes; where the design breaks; CPU contention
docker compose exec backend python evaluation/measure_latency.py
docker compose exec backend python evaluation/scaling_benchmark.py
docker compose exec backend python evaluation/concurrency_benchmark.py

# full answer-quality eval with LLM-judge (uses Gemini quota)
docker compose exec backend python run_eval.py evaluation/golden_set_50.jsonl
```

## Project structure

```
backend/
  ai_core.py            # RAG pipeline: ingest, hybrid search, rerank, gate, corrective, generation
  gemini_rest.py        # streaming generation over v1beta REST — thinking budget control
  rate_limit.py         # cross-process rate limiting, atomic upsert in Postgres
  metrics.py            # Prometheus text exposition, zero dependencies
  main.py               # FastAPI app: auth, documents, conversations, chat streaming
  models.py, schemas.py # SQLAlchemy models, Pydantic validators
  reingest_corpus.py    # controlled full re-ingest with per-step verification
  evaluation/
    golden_set_50.jsonl          # the stable baseline, incl. 5 out-of-corpus
    golden_multihop_new.jsonl    # 11 cross-document questions
    golden_hard_paraphrase.jsonl # 16 deliberately badly-worded — stress test
    golden_conversations.jsonl   # 12 multi-turn, incl. 2 leak probes
    ablation_ladder.py           # what each stage contributes
    random_coverage_baseline.py  # analytic chance floor per set
    trace_query.py               # per-stage tracer: where was the page lost?
    compare_pages_rerankers.py   # do the pages reaching the LLM change?
    measure_gate_margin.py       # gate calibration
    verify_corrective.py         # corrective agent on the real search path
    suggest_keywords.py          # proposes rare keywords, weighted by 1/df
    scaling_benchmark.py         # 418 → 200k chunks
    concurrency_benchmark.py     # latency vs throughput
    runs/                        # CSVs from rejected experiments, kept as evidence
  tests/                # 72 tests; 43 need neither models nor Postgres
frontend/
  app_ui.py             # Streamlit chat UI
docker-compose.yml      # postgres + backend + frontend, named volumes
```

## Tech stack

FastAPI · Streamlit · ChromaDB · PostgreSQL · sentence-transformers (bge-m3, ms-marco-MiniLM) · rank-bm25 · PyMuPDF · Gemini 2.5 Flash · Docker Compose
