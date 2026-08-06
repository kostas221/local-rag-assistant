"""Πειράματα ανάκτησης: chunk-size ΜΕ κανονικοποιημένο βάθος, και όγκος context.

ΔΥΟ ΑΞΟΝΕΣ
----------
--axis chunk  (default)  Σαρώνει chunk_size. Απαιτεί re-ingest ανά μέγεθος (ΑΡΓΟ).
--axis pages             Σαρώνει max_pages, δηλαδή ΠΟΣΟ CONTEXT φτάνει στο LLM.
                         Δεν χρειάζεται ingest — τρέχει πάνω στην υπάρχουσα
                         συλλογή, μόνο για ανάγνωση. Πολύ φθηνότερο.

ΓΙΑΤΙ ΞΑΝΑΓΡΑΦΤΗΚΕ Ο ΑΞΟΝΑΣ CHUNK
----------------------------------
Η πρώτη εκδοχή σύγκρινε 500/1000/1500 κρατώντας ΣΤΑΘΕΡΑ τα βάθη (30 dense,
15 rerank, 12 προς page-expansion). Αυτά μετρούν CHUNKS, όχι κείμενο: σε
chunk_size=500 τα 30 chunks είναι ~15.000 χαρακτήρες, σε 1500 είναι ~45.000.
Άρα το παλιό πείραμα σύγκρινε «πόσο κείμενο έφτασε στο τέλος», όχι το chunking.

Εδώ τα βάθη κλιμακώνονται αντιστρόφως προς το chunk_size:

    chunk_size   dense/BM25   rerank   -> page-expansion
    500              90         45            36
    1000             45         22            18
    1500             30         15            12

Με --depth-mode fixed αναπαράγεται η παλιά (confounded) μέθοδος, για σύγκριση.

ΠΡΟΣΟΧΗ ΣΤΗΝ ΕΡΜΗΝΕΙΑ ΤΟΥ ΑΞΟΝΑ PAGES: περισσότερες σελίδες ανεβάζουν ΠΑΝΤΑ το
coverage, τετριμμένα — υπάρχει απλώς περισσότερο κείμενο να περιέχει το keyword.
Το retrieval-only δείχνει μόνο ΠΟΥ ΠΛΑΤΩΝΕΙ η καμπύλη. Το πραγματικό κόστος της
υπερ-ανάκτησης («lost in the middle», -25..-45% ανάκληση σε μακρύ context)
φαίνεται μόνο με LLM-judge.

ΑΣΦΑΛΕΙΑ: ο άξονας chunk δουλεύει σε ΠΡΟΣΩΡΙΝΗ συλλογή· η παραγωγική
'ai_research_docs' δεν αγγίζεται ποτέ. Ο άξονας pages είναι μόνο ανάγνωση.

Τρέξε:
    docker compose exec backend python chunk_experiment.py --axis pages
    docker compose exec backend python chunk_experiment.py --sizes 1500 --limit 5
    docker compose exec backend python chunk_experiment.py
    docker compose exec backend python chunk_experiment.py --depth-mode fixed
"""
import argparse
import asyncio
import glob
import json
import os
import re
import statistics
import time
import uuid

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS, TestQuestion, evaluate_retrieval

TMP_COLLECTION = "chunk_experiment_tmp"
PDF_DIR = "/app/uploaded_docs"
BASELINE_SIZE = 1500          # η τρέχουσα παραγωγή· ως προς αυτή κλιμακώνονται τα βάθη
BASE_DEPTHS = (30, 15, 12)    # (dense/BM25, rerank, είσοδος page-expansion)
CHARS_PER_PAGE = 4500         # μετρημένος μέσος όρος στο τρέχον corpus
# Το όνομα στον δίσκο είναι "{uuid32}_{πραγματικό όνομα}" (main.py:288).
DISK_PREFIX = re.compile(r"^[0-9a-f]{32}_")


def depths_for(size: int, mode: str) -> tuple:
    """Βάθη ανά chunk_size. 'scaled' = ίσος όγκος κειμένου σε κάθε συνθήκη."""
    if mode == "fixed":
        return BASE_DEPTHS
    f = BASELINE_SIZE / size
    return tuple(max(1, round(d * f)) for d in BASE_DEPTHS)


def ingest_at(collection, size: int) -> int:
    """Χτίζει το corpus στο δοσμένο chunk_size, ΜΟΝΟ για τα αρχεία του GOLDEN_CORPUS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=size // 5,
        separators=["\n\n", "\n", ".", " "])
    docs, metas, ids = [], [], []
    for doc_id, path in enumerate(sorted(glob.glob(f"{PDF_DIR}/*.pdf")), start=1):
        name = DISK_PREFIX.sub("", os.path.basename(path))
        if name not in GOLDEN_CORPUS:
            continue  # demo uploads κ.λπ. δεν μολύνουν τη μέτρηση
                # ΙΔΙΑ εξαγωγή με το ai_core.ingest_pdf (pymupdf + NFKC/συλλαβισμός).
        # Αλλιώς η σάρωση συγκρίνει chunk sizes πάνω σε ΑΛΛΟ κείμενο απ' ό,τι
        # έχει η παραγωγή, και ο "νικητής" δεν μεταφέρεται.
        with pymupdf.open(path) as pdf:
            for page_idx, page in enumerate(pdf):
                text = ai_core._normalize_pdf_text(page.get_text() or "")
                if not text.strip():
                    continue
                for chunk_idx, chunk in enumerate(splitter.split_text(text)):
                    docs.append(chunk)
                    # doc_id: το _expand_to_pages ομαδοποιεί σελίδες με (doc_id, page)
                    metas.append({"file_name": name, "page": page_idx + 1,
                                  "user_id": 1, "is_public": True, "doc_id": doc_id})
                    ids.append(f"{name}_p{page_idx}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")
    if not docs:
        raise RuntimeError(
            f"Κανένα PDF του GOLDEN_CORPUS δεν βρέθηκε στο {PDF_DIR}. "
            f"Ψάχνω: {GOLDEN_CORPUS}")
    for i in range(0, len(docs), 200):   # batches: το embed είναι το ακριβό μέρος
        collection.add(documents=docs[i:i + 200], metadatas=metas[i:i + 200],
                       ids=ids[i:i + 200])
    return len(docs)


def load_tests(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [TestQuestion(**json.loads(line)) for line in f if line.strip()]


async def eval_all(tests: list) -> dict:
    """Retrieval-only. Επιστρέφει συγκεντρωτικά + ανάλυση ανά κατηγορία."""
    per_cat, rows = {}, []
    for t in tests:
        r = await evaluate_retrieval(t)
        rows.append((t.category, r.mrr, r.ndcg, r.keyword_coverage))
        per_cat.setdefault(t.category or "-", []).append(r.mrr)
    inc = [r for r in rows if r[0] != "out_of_corpus"] or rows
    return {
        "mrr": statistics.mean(r[1] for r in inc),
        "ndcg": statistics.mean(r[2] for r in inc),
        "cov": statistics.mean(r[3] for r in inc),
        "per_cat": {c: statistics.mean(v) for c, v in sorted(per_cat.items())},
        "n_incorpus": len(inc),
    }


async def run_chunk_axis(args, tests: list) -> list:
    """Σάρωση chunk_size σε ΠΡΟΣΩΡΙΝΗ συλλογή (η παραγωγική δεν αγγίζεται)."""
    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"Άξονας : chunk_size {sizes}")
    print(f"Βάθος  : {args.depth_mode}" + (
        "  (ΠΡΟΣΟΧΗ: αναπαράγει την παλιά confounded μέθοδο)"
        if args.depth_mode == "fixed" else "") + "\n")

    saved = (ai_core.collection, ai_core.DENSE_CANDIDATES,
             ai_core.RERANK_CANDIDATES, ai_core.EXPAND_INPUT)
    tmp = ai_core.chroma_client.get_or_create_collection(
        name=TMP_COLLECTION,
        embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = tmp   # όλο το pipeline διαβάζει αυτό το global

    rows = []
    try:
        for size in sizes:
            existing = tmp.get()["ids"]
            if existing:
                tmp.delete(ids=existing)
            dense, rerank, expand = depths_for(size, args.depth_mode)
            ai_core.DENSE_CANDIDATES = dense
            ai_core.RERANK_CANDIDATES = rerank
            ai_core.EXPAND_INPUT = expand

            print(f"--- chunk_size={size} | βάθη {dense}/{rerank}/{expand} "
                  f"| ingest (ΑΡΓΟ σε CPU)…", flush=True)
            t0 = time.perf_counter()
            n = ingest_at(tmp, size)
            ai_core._bump_corpus_version()   # invalidate BM25 cache
            t_ingest = time.perf_counter() - t0

            print(f"    {n} chunks σε {t_ingest:.0f}s. Τρέχω retrieval eval…", flush=True)
            t0 = time.perf_counter()
            res = await eval_all(tests)
            res.update(label=size, chunks=n,
                       t_query=(time.perf_counter() - t0) / max(1, len(tests)))
            rows.append(res)
            print(f"    MRR={res['mrr']:.3f} nDCG={res['ndcg']:.3f} "
                  f"coverage={res['cov']:.1f}%  ({res['t_query']:.1f}s/ερώτηση)\n",
                  flush=True)
    finally:
        (ai_core.collection, ai_core.DENSE_CANDIDATES,
         ai_core.RERANK_CANDIDATES, ai_core.EXPAND_INPUT) = saved
        ai_core._bump_corpus_version()
        try:
            ai_core.chroma_client.delete_collection(TMP_COLLECTION)
        except Exception as e:
            print(f"(προσωρινή συλλογή δεν διαγράφηκε: {e})")
    return rows


async def run_pages_axis(args, tests: list) -> list:
    """Σάρωση max_pages στην ΥΠΑΡΧΟΥΣΑ συλλογή. Καμία εγγραφή, κανένα ingest."""
    values = [int(p) for p in args.pages.split(",")]
    print(f"Άξονας : max_pages {values}  (μόνο ανάγνωση, χωρίς ingest)\n")
    saved = ai_core.MAX_PAGES
    rows = []
    try:
        for pages in values:
            ai_core.MAX_PAGES = pages
            chars = pages * CHARS_PER_PAGE
            print(f"--- max_pages={pages} | ~{chars:,} χαρακτήρες "
                  f"≈ {chars // 4:,} tokens context…", flush=True)
            t0 = time.perf_counter()
            res = await eval_all(tests)
            res.update(label=pages, chunks=chars,
                       t_query=(time.perf_counter() - t0) / max(1, len(tests)))
            rows.append(res)
            print(f"    MRR={res['mrr']:.3f} nDCG={res['ndcg']:.3f} "
                  f"coverage={res['cov']:.1f}%\n", flush=True)
    finally:
        ai_core.MAX_PAGES = saved
    return rows


def report(rows: list, axis: str) -> None:
    head, second = ("chunk", "chunks") if axis == "chunk" else ("pages", "χαρακτ.")
    print("=" * 78)
    print(f"{head:>6}{second:>10}{'MRR':>8}{'nDCG':>8}{'coverage':>10}{'s/query':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:>6}{r['chunks']:>10,}{r['mrr']:>8.3f}{r['ndcg']:>8.3f}"
              f"{r['cov']:>9.1f}%{r['t_query']:>9.1f}")
    cats = sorted({c for r in rows for c in r["per_cat"]})
    print("-" * 78)
    print("MRR ανά κατηγορία:")
    print(f"{'':>8}" + "".join(f"{c[:13]:>15}" for c in cats))
    for r in rows:
        print(f"{r['label']:>8}" + "".join(
            f"{r['per_cat'].get(c, float('nan')):>15.3f}" for c in cats))
    best = max(rows, key=lambda r: r["mrr"])
    print("=" * 78)
    print(f"🏆 Καλύτερο MRR: {head}={best['label']} ({best['mrr']:.3f}) "
          f"— in-corpus n={best['n_incorpus']}")
    if axis == "pages":
        print("   ΣΗΜ: το coverage ανεβαίνει τετριμμένα με περισσότερες σελίδες.")
        print("   Κοίτα ΠΟΥ ΠΛΑΤΩΝΕΙ, όχι ποια τιμή είναι η μέγιστη.")


async def main(args) -> int:
    tests = load_tests(args.dataset)
    if args.limit:
        tests = tests[:args.limit]
    print(f"Dataset: {args.dataset} — {len(tests)} ερωτήσεις")

    rows = (await run_pages_axis(args, tests) if args.axis == "pages"
            else await run_chunk_axis(args, tests))
    if not rows:
        return 1
    report(rows, args.axis)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Πειράματα ανάκτησης (retrieval-only).")
    ap.add_argument("--axis", choices=["chunk", "pages"], default="chunk")
    ap.add_argument("--dataset", default="evaluation/golden_set_50.jsonl")
    ap.add_argument("--sizes", default="500,1000,1500", help="άξονας chunk")
    ap.add_argument("--pages", default="3,5,8,12", help="άξονας pages")
    ap.add_argument("--depth-mode", choices=["scaled", "fixed"], default="scaled",
                    help="scaled = ίσος όγκος κειμένου (σωστό)· fixed = η παλιά μέθοδος")
    ap.add_argument("--limit", type=int, default=None,
                    help="τρέξε μόνο τις πρώτες N ερωτήσεις (γρήγορο πέρασμα)")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
