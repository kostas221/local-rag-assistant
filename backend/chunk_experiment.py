"""Chunk-size experiment (retrieval metrics). Τρέξε:
docker compose exec backend python chunk_experiment.py"""
import asyncio, glob, json, uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import ai_core
from evaluation.eval_engine import evaluate_retrieval, TestQuestion

CHUNK_SIZES = [500, 1000, 1500]
PDF_DIR = "/app/uploaded_docs"
TESTS = "evaluation/tests_cloud.jsonl"

def wipe():
    ids = ai_core.collection.get()["ids"]
    if ids:
        ai_core.collection.delete(ids=ids)

def ingest_at(size):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=size // 5,
        separators=["\n\n", "\n", ".", " "])
    docs, metas, ids = [], [], []
    for path in glob.glob(f"{PDF_DIR}/*.pdf"):
        name = path.split("/")[-1]
        for pi, page in enumerate(PdfReader(path).pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            for ci, ch in enumerate(splitter.split_text(text)):
                docs.append(ch)
                metas.append({"file_name": name, "page": pi + 1, "user_id": 1, "is_public": True})
                ids.append(f"{name}_p{pi}_c{ci}_{uuid.uuid4().hex[:8]}")
    ai_core.collection.add(documents=docs, metadatas=metas, ids=ids)
    return len(docs)

def load_tests():
    with open(TESTS, encoding="utf-8") as f:
        return [TestQuestion(**json.loads(l)) for l in f if l.strip()]

async def main():
    tests = load_tests()
    print(f"Tests: {len(tests)} | sizes: {CHUNK_SIZES}\n")
    rows = []
    for size in CHUNK_SIZES:
        print(f"--- chunk_size={size}: wipe + re-ingest (bge-m3, ΑΡΓΟ ~2-5') ---", flush=True)
        wipe()
        n = ingest_at(size)
        print(f"    {n} chunks ingested. Τρέχω retrieval eval...", flush=True)
        mrr = ndcg = cov = 0.0
        for t in tests:
            r = await evaluate_retrieval(t)
            mrr += r.mrr; ndcg += r.ndcg; cov += r.keyword_coverage
        k = len(tests)
        rows.append((size, n, mrr/k, ndcg/k, cov/k))
        print(f"    MRR={mrr/k:.3f} nDCG={ndcg/k:.3f} coverage={cov/k:.1f}%\n", flush=True)
    print("=" * 56)
    print(f"{'chunk':>6}{'chunks':>8}{'MRR':>8}{'nDCG':>8}{'coverage':>10}")
    print("-" * 56)
    for size, n, mrr, ndcg, cov in rows:
        print(f"{size:>6}{n:>8}{mrr:>8.3f}{ndcg:>8.3f}{cov:>9.1f}%")
    best = max(rows, key=lambda r: r[2])
    print("=" * 56)
    print(f"🏆 Best MRR: chunk_size={best[0]} (MRR={best[2]:.3f})")

asyncio.run(main())