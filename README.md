# Z-AI Platform

[![tests](https://github.com/kostas221/local-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/kostas221/local-rag-assistant/actions/workflows/ci.yml)

A **RAG system for scientific papers**: upload PDFs, ask in Greek or English, get answers with page-level citations — and a plain *"not found"* when the papers don't contain the answer. Retrieval runs locally on CPU (bge-m3 + BM25 + cross-encoder); Gemini 3.8 Flash writes the answer.

Built as a diploma thesis, then pushed further under one rule: **no change without a measurement on this project's own data** — including changes that the literature says should work. Every decision rule is written down *before* the run it judges.

> **Measured, not eyeballed** — 235 hand-checked questions over 7 papers. Current system (`gemini-3.8-flash`, 18–19 Nov 2026), graded by GPT-4.1 against reference answers, next to the previous model graded by the same judge:
>
> | Test set | 3.8 Flash (now) | 2.5 Flash (Sep 2026) |
> |---|---|---|
> | Standard questions answered correctly | **45 / 45** | 42 / 45 |
> | Off-topic questions — the answer declines | **5 / 5** | 5 / 5 |
> | Near-miss questions declined (related to the papers, but not answerable from them) | **42 / 42** | 40 / 42 |
> | Questions that need two papers — both halves right | **24 / 29** | 18 / 29 |
> | Tables — exact values, and value + unit + row/column | **69 / 69 · 60 / 60** | 69 / 69 · 60 / 60 |
>
> The two-paper gain is beyond noise (10 halves won, 2 lost, p = 0.039); the rest is a single draw — 3.8 accepts no `temperature`. One pre-registered bar was **not** met: on 3.8 the retrieval gate itself silences only 3 of the 5 off-topic questions; the other two are declined by the answer ([why, and three fixes that failed](#known-limitations)). The main components were measured against alternatives before they were kept — and 20+ ideas were [rejected with numbers](#rejected-with-numbers).

## Demo

![Z-AI Platform demo — a Greek question answered with citations, a question that spans two papers, and two questions it correctly declines](docs/demo.gif)

> A Greek question answered in Greek from English papers with citations, a question that spans two papers, a question the papers don't answer (it says so), and an off-topic question (refused).

## Two judges — and the one that was wrong

Answers were graded by **two LLM judges from two vendors**: Gemini 2.5 Flash (judge 1, used until October 2026) and GPT-4.1 (judge 2), which grades **against the reference answer only**. On this system's answers they agreed on 95% of two-paper halves (κ = 0.86), 100% of near-miss verdicts and 41 of 42 standard questions — so the scores are not one model grading its own family. In November 2026 judge 1 stopped being available to the project's API key (`gemini-2.5-flash` returns 404 and is redirected from 1 Dec 2026), so judge 2 — whose verdicts on the old answers were cached — is now the common judge for old and new answers.

Judge 2 exists because judge 1 turned out to be wrong in a specific way. While testing simpler retrieval set-ups, the answer judge used since v1 gave **5/5/5/5 to *"I cannot find the answer … in the provided source text"*** on questions whose answer *is* in the papers, whenever retrieval had missed the page. Its criteria are relative to the retrieved context, so a failed search followed by an honest refusal looks perfect. The rule was corrected — a refusal on an answerable question is not a perfect answer — and the reference-only judge was added to check it. A context-relative judge is fine for comparing configurations of *one* retriever; across retrievers it rewards the worse one.

```bash
python backend/evaluation/compare_judge2.py                # judge 2 on the host; every verdict is cached and committed
python backend/evaluation/compare_judge2.py --add g38b     # the 3.8 Flash answers, next to the stored ones
```

## Why retrieve at all? Against "just put all the PDFs in the prompt"

The 7 papers are ~132k tokens and fit in Gemini's context window, so the obvious question is why retrieve at all. The same `ask_ai` was run with all 122 pages instead of 8 (on 2.5 Flash, Sep 2026):

| | RAG (8 pages) | All 122 pages |
|---|---|---|
| Quality (keyword proxy, 3 sets) | equal | equal |
| Near-miss correctly declined | 40/42 | 41/42 |
| Tokens per question | ~10,000 | ~132,000 |
| Cost per question | 0.48¢ | 1.20¢ with 98% served from Gemini's cache (~8.6× without) |
| Time to first token (median / p90) | 3.2 s + ~0.8 s retrieval | 3.9 s / 10.7 s |

**At 7 papers, RAG does not win on quality.** It wins on what long context cannot offer: corpora beyond ~1M tokens, cost that doesn't depend on a cache hit, per-user isolation (one prompt per user means no shared cache), and a gate that stays silent before generating anything. Measured with a keyword proxy on both sides, since an LLM judge would have to read all 132k tokens per verdict.

## Questions that need two papers

The biggest gap found in v2: on 29 questions that name two papers, the page holding the evidence for each half arrived only **30 of 58** times — and when it didn't, the half was answered correctly 6 of 28 times. The answer was being lost in retrieval, not in generation.

The fix is deliberately small. When a question **names at least two papers** (an alias list per paper), the same query is re-run inside each named paper and up to 4 pages are added. No LLM call, and the gate is untouched.

| Two-paper questions (29), 2.5 Flash, judge 1 | Before | After |
|---|---|---|
| Correct halves | 33/58 | **44/58** (Δ +11, 95% CI [+5, +17]) |
| Both halves correct | 7 | **17** |
| Unsupported halves | 4 | 2 |
| Faithfulness | 4.52 | 4.75 |

On 3.8 Flash, with the same pages, GPT-4.1 counts **53/58** correct halves and **24/29** questions with both right.

Across all 235 scoreboard questions, pages changed **only** in the 41 routed ones; the gate and corrective agent decided identically in 235/235. The routing rule fires on **0 of 170** questions that don't name two papers.

The path there is documented, including the rejected route: an LLM router that decides whether to split a question splits 35 of 50 *standard* questions too — an extra call and ~3 extra pages on almost every question, for the gain a name match gets for free.

## What each pipeline stage actually buys

Every component is justified by an ablation on the **same 45 in-corpus questions**, not by reputation:

| Stage | MRR | Out-of-corpus correctly refused |
|---|---|---|
| Dense only (bge-m3) | 0.731 | 0/5 |
| \+ BM25 via RRF | 0.774 | 0/5 |
| \+ cross-encoder rerank | 0.793 | 0/5 |
| \+ **relevance gate** | 0.793 | **5/5** |

**The gate improves no ranking metric — and is the single most important component.** Without it the system answers *every* question, including the five it has no material for. All the hybrid machinery buys **+8.5% MRR**; one calibrated threshold buys the refusals.

This is only visible because the golden set contains questions with **no answer in the corpus**. Most golden sets don't, which is why they cannot measure the anti-hallucination defence at all.

```bash
docker compose exec backend python evaluation/ablation_ladder.py
```

## Features

- **Hybrid retrieval** — dense semantic search (BAAI/bge-m3, 1024-dim, cosine) fused with lexical BM25 via Reciprocal Rank Fusion (RRF)
- **Exact vector search** — brute-force cosine over the full index instead of ANN. Measured: identical top-30 to HNSW (30/30, same order) with no speed cost at this corpus size, and *deterministic* across processes
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-12-v2` reorders fused candidates. English-only by design: the pipeline translates before retrieval
- **Anti-hallucination relevance gate** — if even the best chunk scores below a *measured* threshold, the system answers "not found" instead of feeding irrelevant context to the LLM
- **Corrective retrieval agent (CRAG)** — when the gate fires, the query is rewritten and a full second pass runs with its own calibrated threshold. Zero cost on the happy path
- **Per-document search for two-paper questions** — questions that name two papers are searched inside each one; no LLM call ([details](#questions-that-need-two-papers))
- **Cross-lingual QA** — Greek questions are translated for retrieval (translate-then-retrieve, domain- and glossary-aware, cached); answers come back in the user's language
- **Inline citations** — `[S3]` markers resolved server-side to (file, page), so the model is never asked to copy a page number — the step that used to go wrong. Each source shows the passage that matched the question, not the start of the page, and the sources the answer actually cites are listed apart from the other pages it was given
- **Web interface (React)** — click a citation and the real PDF page opens beside the answer, with the sentence's passage highlighted when its place on the page is unique (481 of 488 development passages; every table passage), and never a guessed spot. Per-sentence source passages, an evidence view that shows *where* each passage was found (never "verified"), streaming with Stop, keyboard shortcuts, phone layout. Covered by Playwright end-to-end tests against a fake API and by axe accessibility checks (WCAG 2.1 AA)
- **Conversational rewriting** — follow-ups are rewritten into self-contained queries; leak tests confirm an off-topic follow-up is still refused
- **Robust PDF extraction** — PyMuPDF with Unicode NFKC normalization and de-hyphenation (`A WS` → `AWS`, ligatures, line-break hyphens)
- **Evaluation framework** — 8 golden sets in one frozen scoreboard, two LLM judges, paired bootstrap, chance floors, per-stage tracing, determinism checks
- **Prometheus metrics** — `/metrics` with zero dependencies: gate block rate, corrective success rate, tokens, latency per phase
- **Multi-user** — JWT auth, per-user document isolation enforced in the vector store, rate limiting shared across processes via Postgres

## Architecture

```
┌────────────┐  /api   ┌─────────────────────────────────┐
│  React UI  │ ──────► │            FastAPI               │
│  + nginx   │         │                                  │
│   :8502    │         │  /chat ──► RAG pipeline          │
└────────────┘         │  /upload ─► background ingest    │
                       │  /metrics ► Prometheus text      │
                       └──────┬──────────┬───────────┬────┘
                              │          │           │
                       ┌──────▼───┐ ┌────▼─────┐ ┌───▼────────┐
                       │ ChromaDB │ │ Postgres │ │ Gemini 3.8 │
                       │ (chunks, │ │ (users,  │ │   Flash    │
                       │ vectors) │ │  chats)  │ │(translate, │
                       │          │ │          │ │ rewrite,   │
                       │          │ │          │ │ generate)  │
                       └──────────┘ └──────────┘ └────────────┘
```

### RAG pipeline (per question)

1. **Query optimization** — Greek questions translated to English search queries with a domain- and glossary-aware prompt (cached, so the same question always yields the same retrieval)
2. **Dense search** — bge-m3, exact cosine over the whole index, top-30
3. **Sparse search** — BM25 with Greek-aware tokenization, top-30
4. **Fusion** — Reciprocal Rank Fusion (k=60), deterministic tie-breaking, top-15
5. **Rerank** — `ms-marco-MiniLM-L-12-v2` cross-encoder, **raw logits**
6. **Relevance gate** — best score below −2.6 → "not found"
7. **Corrective retry** — *only if step 6 fired*: rewrite, re-run 2–5, accept above −3.8 (the rewrite produces noun phrases, whose scores sit lower on this model's scale — calibrated, not guessed)
8. **Page-level expansion** — top chunks map back to whole pages (max 8), so tables and lists arrive intact
9. **Per-document search** — *only if the question names ≥ 2 papers*: steps 2–5 inside each named paper, up to 4 extra pages
10. **Generation** — Gemini 3.8 Flash over REST, streamed, with `[S#]` citations (3.8 ignores `temperature` and `thinkingBudget`; both were set for 2.5)

Design decisions and trade-offs behind each step: [`ARCHITECTURE.md`](ARCHITECTURE.md) · the experiment log: [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md)

## Evaluation

Eight golden sets, **235 questions**, over 7 open-access cloud/serverless papers (122 pages, 418 chunks), run as one frozen **scoreboard**: every Gemini call (translations, rewrites) is cached by its full prompt, so two runs differ only if the *system* changed, and every change is reported per question.

| Set | n | What it measures |
|---|---|---|
| `golden_set_50` | 45 + 5 off-topic | the stable baseline |
| `golden_multihop_new` / `_v2` | 11 / 29 | cross-document questions; v2 names two papers and marks the evidence page for each |
| `golden_hard_paraphrase` / `golden_hard_new` | 16 / 59 | deliberately badly-worded and hard questions — stress tests, not baselines |
| `golden_near_ooc` | 42 | **near-miss**: related material exists, the specific answer doesn't |
| `golden_conversations` | 12 | multi-turn, incl. 2 topic-leak probes |
| `golden_tables` | 16 | 69 exact values read from tables |

The PDFs are not redistributed here; all seven are open-access (arXiv, USENIX, UC Berkeley, Google Research). Two more sets probe transfer: `golden_test_domains` (25 questions on two non-CS papers — does the gate calibrated on cloud papers transfer?) and `golden_dangling` (14 questions with the paper's name removed).

```bash
docker compose exec backend python evaluation/scoreboard.py --label mytest --compare evaluation/runs/scoreboard/g38b.json
docker compose exec backend python evaluation/scoreboard_answers.py --set main --label mytest --no-judge   # answers
python backend/evaluation/compare_judge2.py --add mytest                                                   # judge 2 (host)
```

### Answer quality

Judge 2 (GPT-4.1, against reference answers) on both systems. Each model answers from its own retrieval: the same pipeline, but Greek questions are translated by the model itself, so the pages can differ.

| Set | 3.8 Flash (Nov 2026) | 2.5 Flash (Sep 2026) |
|---|---|---|
| Standard (45 + 5) | **45/45 correct** · off-topic **5/5 declined** | 42/45 (2 partial, 1 refused) · 5/5 · judge 1 means 4.90 / 4.80 / 4.92 / 4.96 |
| Two papers (29) | **53/58 halves**, **24/29 both** · 0 wrong | 45/58, 18/29 · 0 wrong |
| Near-miss (42) | **42/42** declined · 0 leaks | 40/42 · 1 soft leak, 1 leak (`n038`, `n024`) |
| Tables (16) | **16/16** · 69/69 values · **60/60** strict | 16/16 · 69/69 · 60/60 strict |

*Strict* table scoring (`evaluation/table_scoring.py`, Nov 2026) checks the value as a whole number with its sign, its unit, and its row/column label. The original check only looked for the expected text anywhere in the answer — it would have passed "9000 seconds and 10.5 GB" for an expected 900 s and 0.5 GB. All 16 stored answers were re-checked by hand against the PDFs first: the old 69/69 was right; the test could not have caught an error.

A single judge run is one draw: the judge itself is not deterministic (2 of 9 identical answers moved by one point between two runs). A label change on one question is not a finding; the per-question comparisons above use paired tests for that reason. Judge 2 does not see the retrieved pages, so it measures correctness, not grounding.

### Every percentage is read against its chance floor

Coverage scores are meaningless without knowing what a *random* retriever would score. Computed analytically (hypergeometric, zero simulation):

| Set | Chance floor | Observed | Margin |
|---|---|---|---|
| `golden_multihop_new` | 22.6% | 98.5% | **+75.9** |
| `golden_set_50` | 33.5% | 98.5% | **+65.0** |
| `golden_conversations` | 41.2% | 90.0% | **+48.8** |

This exercise also **found bugs in the evaluation itself**: 31 keywords in the main set are found on >60% of pages by chance (`serverless` appears on 65 of 122 pages), and one verification tool had been silently crashing on the conversations set, so that set had never been checked at all.

### The relevance gate, measured

| Model | In-corpus min | In-corpus median | Out-of-corpus max | Gap |
|---|---|---|---|---|
| MiniLM-L-6 | −1.80 | 4.15 | −2.69 | 0.89 |
| **MiniLM-L-12** | −2.08 | 4.62 | −3.12 | **1.04** |

The threshold **−2.6** is the midpoint of the range that separated all 61 calibration questions, maximising the *worst-case* margin on both sides. On 2.5 Flash two answerable questions were refused on the first pass (`q025`, `q059`), tracing to translation variance upstream of the gate — see [the third finding](#the-third-finding-anything-upstream-of-the-gate-changes-the-gate). On 3.8 Flash the gate passes all **61/61**.

### Determinism

Retrieval is bit-for-bit reproducible across processes. Two sources of non-determinism were found and fixed: `list(set(...))` in the fusion step (iteration order depended on `PYTHONHASHSEED`, with 15 tied RRF scores out of 51 candidates), and ChromaDB's HNSW graph, which was rebuilt from the WAL on every start and did not always return the same top-30. Before the fix, the *same code* produced MRR 0.764 and 0.755.

Generation is **not** reproducible on 3.8 Flash: the model ignores `temperature`, so the same prompt can return a different translation, rewrite or answer. The scoreboard therefore freezes every Gemini call by its full prompt (`runs/scoreboard_gemini_<model>.json`), and model-dependent decisions are measured over **three independent draws** (`probe_corrective_ab.py`) rather than one.

```bash
docker compose exec backend python evaluation/check_determinism.py
```

### Confidence intervals — and what they disqualify

A **paired bootstrap** (resampling questions, since both configurations saw the same ones) replaced the rule of thumb "don't believe a delta under 0.01":

| L-6 → L-12, same 45 questions | Δ | 95% CI | Verdict |
|---|---|---|---|
| MRR 0.770 → 0.793 | +0.023 | [−0.016, +0.064] | **not proven** |
| nDCG 0.786 → 0.807 | +0.021 | [−0.009, +0.052] | **not proven** |
| Coverage 98.52 → 98.52 | 0.000 | [0.000, 0.000] | **identical in 45/45** |

The noise floor at n=45 is **±0.04, not ±0.01**: most decisions in this project were made in a region where the statistics cannot adjudicate. The reranker upgrade is therefore kept as a **defensible decision, not a proven one** — four converging indications, none significant at its own sample size.

```bash
docker compose exec backend python evaluation/bootstrap_ci.py \
    --compare evaluation/runs/retrieval_l6.csv evaluation/runs/retrieval_l12.csv
```

### Performance

Measured on a Ryzen 7 5700X, CPU only (container: 8 cores), with 2.5 Flash (Oct 2026). Memory matters more than cores: the embedding model alone needs ~2.3 GB, and with Docker limited to 2.8 GB on a laptop a single retrieval took **~60 s** instead of under one second. Give Docker at least 4 GB.

| | Value |
|---|---|
| Warm retrieval | **846 ms** (rerank 784 = 93%, expand 53, authz 5, BM25 1.4, dense 0.5) |
| End-to-end | 2.8–4.3 s · TTFT 2.78 s |
| Throughput | 1.18 req/s for one user, 1.38 req/s at **4 concurrent users**, where it saturates |

Page expansion went from 8 to 53 ms when a per-user authorization check was added to it — about 1% of end to end, kept. Past saturation the throughput plateaus (~1.0 req/s up to 32 users) and latency grows linearly — Little's law, a queue that works, not a collapse. Per-document search adds two BM25+rerank passes **only** on routed questions; its latency has not been measured on this machine yet.

### Cost

| Per 1,000 questions | |
|---|---|
| Gemini input (9,693 tokens/q) | $2.91 |
| Gemini output (1,112 tokens/q) | $2.78 |
| Embeddings, reranking, BM25 | **$0.00** — local, CPU |
| **Total** | **$5.69** |

Measured on 2.5 Flash. A month at that volume costs about **€10**: €5 for a CPU-only VPS plus ~€5 of API. Output tokens cost 8.3× more than input tokens, so the context-size decisions were pulling the *cheap* lever; `THINKING_BUDGET=512` is 21% of the bill — setting it to 0 was measured and dropped one multi-hop answer's faithfulness from 5.0 to **2.0**.

3.8 Flash lists at $0.75 / $3.75 per 1M tokens until 31 Dec 2026 and $1.50 / $7.50 from 2027 (2.5: $0.30 / $2.50) — roughly **2.4×** the per-question cost now and **~5×** from 2027, estimated from the price list, not measured.

```bash
docker compose exec backend python evaluation/measure_cost.py --ratio 4.53
```

## Technical decisions

What was measured, kept, and — more often — **rejected**. Each row is a real experiment.

### Kept

| Change | Measured effect |
|---|---|
| **Per-document search** for questions naming two papers | Two-paper halves **33 → 44/58** (CI [+5, +17]), both halves **7 → 17/29**; 0/170 false triggers; gate identical 235/235; 0 LLM calls |
| **PyMuPDF + NFKC** instead of pypdf | Broken tokens 3.7% → 2.5%; 721 ligatures and 941 hyphenations eliminated. MRR +0.026 |
| **English reranker** (568M → 22M params) | The pipeline translates *before* retrieval, so the cross-encoder always sees English. Latency 15,048 ms → **693 ms (21.7×)**, accuracy 4.96 → 5.00 |
| **MiniLM-L-6 → L-12** + recalibrating both thresholds | Hard set **10/16 → 12/16** · gate gap 0.89 → **1.04** · correct chunk at rank 1 **36 → 40/56** · corrective hallucinations **2 → 0**. Cost **+9% e2e**. Defensible, not proven |
| **Exact search** instead of HNSW | Identical top-30, no speed cost at 418 vectors, deterministic |
| **Deterministic fusion** | `dict.fromkeys` + tie-break on chunk id. Made every later measurement trustworthy |
| **Reranker threads and `batch_size`** | 667 → **434 ms** with bit-identical scores (thread heuristic under WSL2; padding in one large batch) |
| **Capped thinking budget** (REST) | TTFT 3.90 s → **2.78 s**, judge scores unchanged |
| **Domain-aware translation** | Greek *«μηχανήματα»* had become `machinery` instead of `servers`. The first measurement (+7.74 / +4.14 logits on two questions) turned out to be **leakage**: the prompt's examples were those two questions' answers. The examples were removed; the clean size of the gain was never measured — kept as a design choice, not as a result |
| **Corrective agent** + its own threshold | Silence → correct answer on the questions it targets, **0 hallucinations**, off-topic still 5/5 (2.5 Flash). On 3.8 it rescues ~10 hard questions per run — and lets ~1–2 off-topic ones through to the answer, which declines them ([limitations](#known-limitations)) |
| **Strict table scoring** (value, sign, unit, row/column) | Catches substring matches (`900` in `9000`), swapped values and wrong units that the original check passed; hand-check of all 16 stored answers first: the old 69/69 stood |
| **Model guard** in the evaluation | From 1 Dec 2026 `gemini-2.5-flash` silently answers as `gemini-3.5-flash-lite`; a run naming it now stops instead of recording the wrong judge or model |
| **Server-side citation labels** (`[S3]` instead of `[file, p.12]`) | With labels written by the model, 10/97 page numbers were wrong — it copied the page number *printed* in the PDF. The labels themselves were right 97/97. What you don't ask the model to produce cannot come out wrong |
| **Rate limiting in Postgres**, **`/metrics`** | Shared across processes, zero new containers |

### Rejected, with numbers

| Change | Result | Why it failed |
|---|---|---|
| **`gte-reranker-modernbert-base`** (150M) | The project's first **proven** ranking gain: MRR 0.785 → **0.874**, CI [+0.025, +0.160] | Rerank **5.5× slower** on CPU (0.59 → 3.26 s), and its compressed score scale weakens the "not here" signal: an off-domain question (volcanology) passed the gate, two hard questions passed with **zero** relevant material. Better ranking, worse refusal |
| **Qwen3-Embedding-0.6B** | Dense top-30 clearly better (both papers present 12 → 17/29) | After fusion and rerank the gain evaporated (7 → 9); every CI crossed zero; pages changed in 221/260 questions — and **two new leaks** at margins of 0.08 and 0.13 logits. Same reranker + same threshold is *not* the same gate |
| **Contextual chunk prefixes** (Anthropic-style Contextual Retrieval / doc2query) | Direction probe, n=33: in-corpus scores **fell** (median −0.85 logits, 22 → 18 above the gate) while an out-of-corpus question rose **+2.72** and crossed it | The direction was wrong on both sides, so the full re-ingest was not run. A technique reported to cut retrieval failures by 49–67% elsewhere, measured here first |
| **Query enrichment with domain terms** (3 variants) | Fixed one hallucination — and **leaked one out-of-corpus question** | Enrichment helps retrieval only when the reranker also sees the added words — exactly when the gate is misled. Architectural, not a bug |
| **Corrective rewrite rule** "never add field words the question lacks" (3.8, 3 draws per prompt) | Off-topic leaks **not** removed (`q048` passed 3/3); rescued hard questions **10.3 → 6.7** per run | The model kept "cloud service providers" on GDPR and dropped "cloud" exactly where it was helping |
| **Higher corrective threshold** (offline, same draws) | 0 leaks only at −0.5, where rescues fall **10.3 → 7.0** | Off-topic (up to −0.52) and rescued (from −3.44) scores overlap |
| **Re-checking the rewrite's chunks against the original question** (CRAG-style, offline) | 0 leaks only when rescues fall **10.3 → 1.0** | Inverted: "GDPR compliance" matches Sarbanes-Oxley text (−3.1) better than vague answerable questions match their own evidence (−8 to −11) — the gap the rewrite exists to bridge |
| **LLM router / query decomposition** (Aug and Sep) | Aug: +1 keyword of 45. Sep: decomposition solved the two-paper problem, but the router split **35/50** standard questions too | An extra call and ~3 pages on almost every question. The name-based router keeps the gain with 0 false triggers and no LLM |
| **"Which paper do you mean?" agent** | 12/14 questions stripped of the paper's name were answered normally anyway | On the 2 that needed it, restricting the search to the right paper was *still* refused: restriction removes competitors, it does not raise the gate score |
| **Figure ingestion** (multimodal) | Audit of all 33 figures: **3** (7 generously) carry a fact absent from the text | Below the pre-registered threshold of 10. The authors state each figure's message and numbers in the text; vector diagrams already yield their labels |
| **`bge-reranker-v2-m3` (568M)** | +1 real answer **and** +1 hallucination = net zero, ~15 s/question | More willing to answer without material |
| **`bge-reranker-base` (278M)** | **Worse than the 22M model**: 3/16 hard | Scores out-of-corpus chunks high — discriminability does not scale with size |
| **`RERANK_CANDIDATES` 15 → 12 / 10** | Every summary metric identical | **False economy**: N=12 changed the pages reaching the LLM in **42/56** questions |
| **PyTorch dynamic INT8** | 1.37× faster, Pearson ρ **0.997** | Shifts a correct answer below the gate. **ρ = 0.997 with a broken gate** — correlation is the wrong metric for a reranker |
| **Thinking budget 0** | 2.73× faster TTFT | Faithfulness **5.0 → 2.0** on a multi-hop answer, invisible to keyword coverage |
| **`chunk_size` 750 / 1000** | Main set better, hard set worse — **the sign flips between sets** | The main set is at its coverage ceiling and cannot show the loss |
| **Cascade reranking** | Rejected by arithmetic: 434 + 282 ms > 706 ms | Needs a first stage 10–100× cheaper; here 1.63× |
| **Langfuse**, **pgvector**, **contextual compression**, **BGE-M3 sparse** | Operational cost with no product gain, or no measured problem to solve | Criteria to revisit each are recorded in the log |

### The finding that mattered most

With coverage above 97%, the correct material already reaches the model — **retrieval MRR is decoupled from answer quality** here, and **no intermediate metric predicts the final prompt**. The reranker upgrade left coverage identical in 45/45 questions while **35 of 56** received different pages; `RERANK_CANDIDATES=10` left best-logit, gate gap and keyword@1 identical while changing the pages in **52 of 56**. The prompt is pages, and only one tool looks at them:

```bash
docker compose exec backend python evaluation/compare_pages_rerankers.py
```

### The second finding: your golden set hides your failures, because you wrote it

The gate once scored a perfect **61/61** — and it was false. One question had been written naturally, was refused, and **was deleted from the set**. Sixteen deliberately badly-worded paraphrases then exposed **11/16 wrong refusals**. A production user does not get the option to delete the question that breaks your system. The near-miss set (42) and the hard set (59) exist because of this.

### The third finding: anything upstream of the gate changes the gate

The gate judges the best score *among the candidates it is shown*. A different embedder (Qwen3), a different reranker scale (gte), or a different translation of the same Greek question (`q025`) changes the candidates — and borderline cases flip, in both directions, at margins of a tenth of a logit. Every upstream change is therefore re-tested against the refusal sets, not just the accuracy sets.

### Where a failure actually happens

A per-stage tracer answers "at which step was this page lost?" — translation, dense, BM25, RRF, rerank, gate, or prompt. It is how the two-paper problem was found: 21 of 28 lost evidence pages were lost *before* the reranker ever saw them.

```bash
docker compose exec backend python evaluation/trace_query.py --id q028 --target excamera-nsdi17.pdf:15
```

## Known limitations

- **The retrieval gate lets some off-topic questions through on 3.8 Flash.** The pre-registered bar asked for 5/5 off-topic questions silenced *by retrieval*; 3.8 gives **3/5** (and 4/5 on the other-domain set). `q048` (GDPR) and `q050` (GPU tensor cores) are cut by the gate, then the corrective rewrite adds field words ("cloud service providers") and they pass. In all such cases the **answer** says the papers do not cover it (5/5 off-topic, 42/42 near-miss declined). Three fixes were measured — a stricter rewrite prompt, a higher threshold, a re-check against the original question — and none separates these from the vague-but-answerable questions the same rewrite rescues ([rejected](#rejected-with-numbers)). The defence is in the answer, and next in a claim-level grounding check
- **Near-miss leaks.** On 2.5 Flash two of 42 leaked (`n024`, a synthesis of two facts no paper links; `n038`, the 2024 time limit dropped). On 3.8 both are declined, but this is one draw; a claim-level check (e.g. MiniCheck, HHEM) is the planned safeguard
- **Other-domain off-topic: `o4` (Vesuvius).** On 2.5 the translation was "Vesuvius eruptions" (−3.46, silenced); on 3.8 it is "Vesuvius last eruption date" (−1.56), which passes the gate on a volcanology paper about Stromboli. No corpus word was added — the glossary rule held; the literal wording alone moved the score. The answer for this set was not graded
- **Text-layer PDFs only.** A scanned PDF is flagged in the UI as *"No extractable text (scanned PDF, needs OCR)"* instead of silently showing as ready; there is no OCR or layout model. Figures were audited instead of ingested
- **Not measured:** questions about the whole corpus at once ("what do all seven papers say about cold starts?"), multi-step agentic retrieval, and prompt injection hidden in uploaded PDFs (uploads are private to their owner, so an injected document can only mislead its own uploader)
- **Two-paper routing uses hand-written aliases** for the 7 papers; paraphrases ("the video-processing system") and pronouns are not caught. Automatic alias extraction at ingest is open
- **7 papers.** The design is tuned to 418 chunks. Breaking points are measured (BM25 fails first, 6.7× slower than dense at 50k chunks), but quality at 50+ papers is not
- **LLM judges.** Two vendors now agree, but both are LLMs; the golden sets are self-authored and hand-verified, not externally annotated
- **Translations, rewrites and answers are not reproducible on 3.8 Flash** (no `temperature`); the corrective threshold rests on a handful of questions — the midpoint of a range that behaves identically, not an optimum
- **Throughput saturates at 4 concurrent users** on one CPU-bound worker (measured with 2.5 Flash)
- **Model history.** Everything before November 2026 was measured on `gemini-2.5-flash`, which is no longer offered to API keys and is redirected to `gemini-3.5-flash-lite` from 1 Dec 2026. `gemini-3.8-flash` became the default after the full scoreboard (18–19 Nov 2026); it ignores `thinkingBudget` (it can think despite a budget of 0) and occasionally returns no text for a harmless prompt (`v4`, an answerable volcanology question: `blockReason: OTHER` in October, an empty translation in November)
- **`google.generativeai` is deprecated** — generation uses the Gemini REST endpoint directly

## Quickstart

Requirements: Docker Desktop (WSL2 backend on Windows) with **at least 4 GB of memory** for Docker (8 GB recommended — with 2.8 GB a retrieval took ~60 s), and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/kostas221/local-rag-assistant.git
cd local-rag-assistant
cp .env.example .env        # fill in your keys
docker compose up -d --build
```

First start downloads the embedding + reranker models (~2.5 GB, cached in a volume afterwards).

> Upgrading an install from before November 2026? Its vector store is Chroma 0.4; read [`docs/STABILIZATION_2026-11-01.md`](docs/STABILIZATION_2026-11-01.md) and start with `-f docker-compose.yml -f docker-compose.upgrade.yml`. A fresh clone needs neither.

- UI: http://localhost:8502 — register, upload a PDF, wait for "ready", ask away; click a citation to see the page it comes from
- API docs: http://localhost:8010/docs
- Metrics: http://localhost:8010/metrics

## Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key (generation, translation, judge 1) |
| `GEMINI_MODEL` | Model for translation, rewrites and answers. Default `gemini-3.8-flash` |
| `JUDGE_MODEL` | Model of judge 1 in the evaluation scripts. Separate from `GEMINI_MODEL`, so testing a new model does not also change the judge. Its old default, `gemini-2.5-flash`, is no longer available: from 1 Dec 2026 a run that names it stops (`evaluation/model_guard.py`); use `--no-judge` and judge 2 |
| `OPENAI_API_KEY` | only for the second judge (`compare_judge2.py`), now the common judge |
| `POSTGRES_PASSWORD` | Postgres password |
| `SECRET_KEY` | JWT signing key — `openssl rand -hex 32` |
| `RERANKER_MODEL` | Cross-encoder. **Changing it requires recalibrating `MIN_RERANK_SCORE`** |
| `MIN_RERANK_SCORE` | Relevance gate, in **raw logits** (`evaluation/measure_gate_margin.py`) |
| `ENABLE_CORRECTIVE`, `CORRECTIVE_MIN_SCORE` | Corrective agent and its threshold |
| `ENABLE_PERDOC`, `PERDOC_EXTRA` | Per-document search for two-paper questions, and how many pages it may add |
| `DENSE_CANDIDATES`, `RERANK_CANDIDATES`, `EXPAND_INPUT`, `MAX_PAGES` | Pipeline depths |

## Tests & evaluation

```bash
docker compose exec backend python -m pytest tests/ -q     # 149 tests; CI runs the model-free ones first

docker compose exec backend python evaluation/scoreboard.py            # all 8 sets, frozen, per-question diff
docker compose exec backend python evaluation/check_determinism.py
docker compose exec backend python evaluation/random_coverage_baseline.py
docker compose exec backend python evaluation/ablation_ladder.py
docker compose exec backend python evaluation/trace_query.py --id q028
docker compose exec backend python evaluation/compare_pages_rerankers.py
docker compose exec backend python evaluation/measure_gate_margin.py

cd web && npm ci && npm test && npx playwright test   # UI: unit tests, then end-to-end tests against a fake API (no backend, no cost)
```

## Project structure

```
backend/
  ai_core.py            # RAG pipeline: ingest, hybrid search, rerank, gate, corrective, generation
  doc_routing.py        # which papers a question names -> per-document search
  corpus_names.json     # aliases per paper (the measured list)
  gemini_rest.py        # streaming generation over REST — thinking budget control
  rate_limit.py         # cross-process rate limiting in Postgres
  metrics.py            # Prometheus text exposition, zero dependencies
  main.py               # FastAPI: auth, documents, conversations, chat streaming
  page_view.py          # PDF page image with the passage highlighted, only where its place is unique
  evaluation/
    golden_*.jsonl               # 8 scoreboard sets + domains + dangling
    scoreboard.py                # all sets, one frozen run, per-question diff
    scoreboard_answers.py        # answers + judges per set
    compare_judge2.py            # second judge (OpenAI), agreement (κ)
    probe_*.py                   # one experiment each — prediction and result in the docstring
    runs/                        # every result, including the rejected ones
  tests/                # 149 tests
web/                    # React UI: TypeScript, Vite, Tailwind, shadcn/ui
  src/features/         # chat, library, source page viewer, sign-in
  e2e/                  # Playwright tests against a fake API, with axe accessibility checks
  nginx.conf            # /api proxy with streaming, 50 MB uploads, security headers
scripts/                # backup, README charts
docker-compose.yml      # postgres + backend + web
```

## Tech stack

FastAPI · React (TypeScript, Vite, Tailwind, shadcn/ui) · nginx · ChromaDB · PostgreSQL · sentence-transformers (bge-m3, ms-marco-MiniLM) · rank-bm25 · PyMuPDF · Gemini 3.8 Flash (REST) · Docker Compose · GPT-4.1 as the evaluation judge
