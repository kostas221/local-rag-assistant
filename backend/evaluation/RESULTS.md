# Evaluation Results

Authoritative, reproducible evaluation of the retrieval + generation pipeline.
Run with `python run_eval.py evaluation/golden_set_20.jsonl` inside the backend container.

## Setup
- **Corpus:** two Berkeley papers — *Above the Clouds* (EECS-2009-28) + *A Berkeley
  View on Serverless Computing* (arXiv 1902.03383).
- **Golden set:** [`golden_set_20.jsonl`](golden_set_20.jsonl) — 20 bilingual (EN + EL)
  questions, incl. 2 deliberate out-of-corpus questions (quantum, bitcoin) to test the
  anti-hallucination gate.
- **Config:** bge-m3 (dense, 1024-d, cosine) + BM25 (Greek-aware) + RRF (k=60) +
  bge-reranker-v2-m3 (`RERANK_CANDIDATES=15`), `chunk_size=1500`, **parent-document
  (page-level) expansion**, relevance gate `MIN_RERANK_SCORE=0.15`. Generation: Gemini
  2.5 Flash, `temperature=0.3` (reproducible). Eval **pinned** to the 2 papers
  (`target_filenames`) so other uploaded docs don't pollute the measurement.
- **Judge:** Gemini 2.5 Flash, `temperature=0` (deterministic), scoring each answer 1–5.

## Retrieval (golden_set_20, pinned corpus)
| Metric | In-corpus (18 Q) | Overall (20 Q) |
|---|---|---|
| MRR | **0.846** | 0.762 |
| Keyword coverage | **97.2%** | 87.5% |
| nDCG | — | 0.743 |

> The 2 out-of-corpus questions (quantum, bitcoin) score MRR/coverage **0 by design** —
> they have no relevant document and test the anti-hallucination gate, not ranking. The
> **in-corpus** columns (18 Q) are the retrieval-ranking number; both sets are gated
> correctly. (rerank-10 baseline: in-corpus MRR 0.824 / cov 93.5%.)

### Chunk-size experiment (retrieval, same corpus)
| chunk size | chunks | MRR | nDCG | coverage |
|---|---|---|---|---|
| 500 | 1082 | 0.740 | 0.750 | 85.7% |
| 1000 | 564 | 0.724 | 0.750 | 90.5% |
| **1500** | **386** | **0.803** | **0.821** | **95.2%** |

→ `chunk_size=1500` is the measured winner (used in production).

## Answer quality (LLM-as-judge)
| Metric | Score |
|---|---|
| Accuracy | **5.0 / 5** |
| Completeness | **5.0 / 5** |
| Relevance | **5.0 / 5** |
| Faithfulness (no hallucinations) | **5.0 / 5** |

The 2 out-of-corpus questions correctly return *"not found in the documents"* — the
relevance gate fires (best reranker score ≈ 0.00 < 0.15), so the model does not hallucinate.

## Anti-hallucination gate calibration
Reranker score separation on the golden set:
- **Relevant** questions: **0.92 – 1.00**
- **Out-of-corpus** (quantum/bitcoin): **0.000 – 0.004**

→ wide margin; threshold `0.15` blocks irrelevant context while still passing
lower-scoring *translated* Greek queries.

## Performance (local, CPU — AMD Ryzen 7 5700X)
The cross-encoder reranker dominates latency on CPU. Measured per-query (1 user) on the
BM25-cached + page-expansion pipeline — the rerank-depth trade-off:
| `RERANK_CANDIDATES` | retrieval latency | in-corpus MRR / coverage |
|---|---|---|
| **15** (default) | ~13 s | **0.846 / 97.2%** |
| 10 | ~8 s | 0.824 / 93.5% |

→ 15 buys +3.7% coverage (and fixes the ExCamera/enumeration case) at ~+60% rerank time;
**env-configurable** (`RERANK_CANDIDATES=10` for CPU under load — answer quality is
identical at both). Caching the BM25 index (invalidate on ingest/delete) already gives a
large speedup; a **GPU** (or ONNX int8 quantization) removes the remaining latency, making
15 effectively free.

> Per-question outputs: [`results.csv`](results.csv) (UTF-8-BOM for Excel).
> Methodology note: LLM-judge scores carry small run-to-run variance (`temperature=0`
> reduces but doesn't eliminate it); the in-corpus retrieval numbers are deterministic.
